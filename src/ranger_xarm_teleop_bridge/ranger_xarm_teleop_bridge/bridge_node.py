# Copyright 2026 Shuaib Olanrewaju
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ROS 2 node that turns headset frames from the vr-teleop-kit relay into /teleop/input/* topics.

The node is a thin wrapper: ``frame_parse`` validates the wire format, ``axes`` does the one
axis conversion, ``joy_layout`` fixes the Joy index order. ``handle_text`` is the single entry
point for live WebSocket text and for replayed recordings, so both paths share one parser and
one conversion (INV-1, INV-6). It never raises (INV-7): malformed input is logged and counted.

Per headset frame it publishes, all with the identical ``header.stamp`` (ROS time, taken once):
  /teleop/input/left/pose, /teleop/input/right/pose  geometry_msgs/PoseStamped (present hands)
  /teleop/input/head/pose                            geometry_msgs/PoseStamped (viewer present)
  /teleop/input/joy                                  sensor_msgs/Joy
  /teleop/input/status                               ranger_xarm_teleop_interfaces/InputStatus

With ``feedback_enabled`` it also forwards operator feedback to the headset over the same relay
connection: it subscribes to /teleop/haptic and /teleop/operator/status (and nothing else) and
sends ``teleop_haptic`` and ``teleop_status`` messages (``relay_messages``). The ROS callbacks
only fill an ``Outbox``; a sender task beside the reader in the connection's event loop sends
from it, so reading and publishing headset frames never waits for a send (INV-1). Old items are
dropped, never sent late, and nothing queues while the link is down (INV-4). With feedback off,
nothing is subscribed or sent (INV-7).
"""

import asyncio
from dataclasses import dataclass, field
import random
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from geometry_msgs.msg import PoseStamped
from ranger_xarm_teleop_bridge.axes import xr_pose_to_ros
from ranger_xarm_teleop_bridge.frame_parse import (Ignored, Malformed, parse_text, ParsedFrame,
                                                   ResetEvent)
from ranger_xarm_teleop_bridge.joy_layout import build_joy_arrays
from ranger_xarm_teleop_bridge.jsonl_io import RecordWriter
from ranger_xarm_teleop_bridge.operator_status import SimSpeed
from ranger_xarm_teleop_bridge.outbox import Outbox
from ranger_xarm_teleop_bridge.relay_messages import (echo_of, encode, FrameInfo,
                                                      haptic_payload, panel_config,
                                                      status_payload)
# The topic names and QoS live in ``topics`` (no WebSocket import) and are re-exported here.
from ranger_xarm_teleop_bridge.topics import (HANDS, HAPTIC_QOS, HAPTIC_TOPIC,  # noqa: F401
                                              HEAD_POSE_TOPIC, JOY_QOS, JOY_TOPIC,
                                              OPERATOR_STATUS_QOS, OPERATOR_STATUS_TOPIC,
                                              POSE_QOS, POSE_TOPICS, STATUS_QOS, STATUS_TOPIC)
from ranger_xarm_teleop_interfaces.msg import HandStatus, Haptic, InputStatus, OperatorStatus
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Joy
import websockets

LOG_THROTTLE_S = 2.0
MIN_SEND_BUFFER = 1024
MAX_SEND_BUFFER = 32768


@dataclass
class FrameMessages:
    """The messages built from one headset frame; they share one stamp."""

    poses: Dict[str, PoseStamped] = field(default_factory=dict)
    # The headset pose; never a controller pose, so it is not in ``poses``.
    head: Optional[PoseStamped] = None
    joy: Optional[Joy] = None
    status: Optional[InputStatus] = None


def _ros_time(stamp) -> str:
    return f'{stamp.sec}.{stamp.nanosec:09d}'


def _reset_movement(event: ResetEvent) -> str:
    """Describe how far the headset says a reset moved the frame."""
    if event.position is None or event.orientation is None:
        return 'movement not reported by the headset'
    return f'moved {event.distance_m:.3f} m, rotated {event.angle_deg:.2f} deg'


class BridgeNode(Node):
    """Publishes the input side of the interface contract from relay text messages."""

    def __init__(self, node_name: str = 'teleop_input_bridge',
                 source_label: str = 'relay', **node_kwargs) -> None:
        super().__init__(node_name, **node_kwargs)
        self.declare_parameter('relay_url', 'ws://127.0.0.1:8443/ws')
        self.declare_parameter('reconnect_interval_s', 1.0)
        self.declare_parameter('frame_id', 'quest_world')
        self.declare_parameter('record_path', '')
        self.declare_parameter('source', source_label)
        self.declare_parameter('stats_period_s', 5.0)
        self.declare_parameter('feedback_enabled', False)
        self.declare_parameter('panel_anchor', 'left_controller')
        self.declare_parameter('panel_offset', [0.0, 0.10, 0.0])
        self.declare_parameter('panel_width_m', 0.16)
        self.declare_parameter('panel_visible_default', True)
        self.declare_parameter('status_stale_s', 1.0)
        self.declare_parameter('status_max_age_s', 0.4)
        self.declare_parameter('haptic_queue', 8)
        self.declare_parameter('send_buffer_limit_bytes', 16384)
        self.declare_parameter('sim_slow_below', 0.6)
        self.declare_parameter('sim_slow_clear', 0.7)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.source = str(self.get_parameter('source').value)

        self.pose_pubs = {hand: self.create_publisher(PoseStamped, POSE_TOPICS[hand], POSE_QOS)
                          for hand in HANDS}
        self.head_pub = self.create_publisher(PoseStamped, HEAD_POSE_TOPIC, POSE_QOS)
        self.joy_pub = self.create_publisher(Joy, JOY_TOPIC, JOY_QOS)
        self.status_pub = self.create_publisher(InputStatus, STATUS_TOPIC, STATUS_QOS)

        self.frames = 0
        self.malformed = 0
        self.ignored = 0
        self.ignored_types: Dict[str, int] = {}
        self.last_seq: Optional[int] = None
        self.last_session_id: Optional[int] = None
        self.last_reset_count: Optional[int] = None
        self.connected = False
        self._frames_at_last_stats = 0
        self._last_warn = 0.0

        record_path = str(self.get_parameter('record_path').value)
        self.recorder: Optional[RecordWriter] = None
        if record_path:
            self.recorder = RecordWriter(record_path)
            self.get_logger().info(f'recording relay text to {record_path}')

        period = float(self.get_parameter('stats_period_s').value)
        self._stats_timer = self.create_timer(period, self._log_stats) if period > 0 else None
        self._setup_feedback()

    # ---- operator feedback (only with feedback_enabled) ----

    def _setup_feedback(self) -> None:
        param = self.get_parameter
        self.feedback_enabled = bool(param('feedback_enabled').value)
        # Newest headset frame (session id, seq, t_client ms, monotonic arrival), for the echo.
        self.last_frame: Optional[FrameInfo] = None
        # Set by the relay connection: wakes its sender (thread-safe), or None when unconnected.
        self.feedback_wake: Optional[Callable[[], None]] = None
        self.feedback_sent = {'teleop_haptic': 0, 'teleop_status': 0}
        self.feedback_errors = 0
        self.feedback_stalls = 0
        if not self.feedback_enabled:
            return
        self.run_id = random.SystemRandom().randrange(1, 2 ** 31)
        self._haptic_id = 0
        self._status_seq = 0
        self.panel = panel_config(str(param('panel_anchor').value),
                                  list(param('panel_offset').value),
                                  float(param('panel_width_m').value),
                                  bool(param('panel_visible_default').value))
        self.status_stale_ms = 1000.0 * float(param('status_stale_s').value)
        # Measured from the statuses' arrivals (what the page can expect), 200 ms until known.
        self.status_period_ms = 200.0
        self._last_status_at: Optional[float] = None
        # The ROS clock against the wall clock, from the statuses' stamps: a slow simulator.
        self.sim_speed = SimSpeed(slow_below=float(param('sim_slow_below').value),
                                  clear_above=float(param('sim_slow_clear').value))
        self.send_buffer_limit = int(param('send_buffer_limit_bytes').value)
        # Below the connection's own write limit (websockets: 64 KiB), above which a send, and
        # the protocol's own replies to the relay's pings, would wait for the link to drain.
        if not MIN_SEND_BUFFER <= self.send_buffer_limit <= MAX_SEND_BUFFER:
            raise ValueError(f'send_buffer_limit_bytes must be {MIN_SEND_BUFFER}..'
                             f'{MAX_SEND_BUFFER}')
        self.outbox = Outbox(haptic_max=int(param('haptic_queue').value),
                             status_max_age_s=float(param('status_max_age_s').value))
        self.create_subscription(Haptic, HAPTIC_TOPIC, self._on_haptic, HAPTIC_QOS)
        self.create_subscription(OperatorStatus, OPERATOR_STATUS_TOPIC, self._on_status,
                                 OPERATOR_STATUS_QOS)
        self.get_logger().info(f'operator feedback on: forwarding {HAPTIC_TOPIC} and '
                               f'{OPERATOR_STATUS_TOPIC} to the relay (run {self.run_id})')

    def _wake(self) -> None:
        wake = self.feedback_wake
        if wake is not None:
            try:
                wake()
            except RuntimeError:         # the connection's loop has just closed
                pass

    def _on_haptic(self, msg: Haptic) -> None:
        try:
            self._haptic_id += 1
            payload = haptic_payload(msg, self.run_id, self._haptic_id)
            self.outbox.put_haptic(payload, time.time() - msg.wall_time, float(msg.max_age))
            self._wake()
        except Exception as exc:  # noqa: B902 - feedback must never take the bridge down
            self._feedback_error(f'haptic: {type(exc).__name__}: {exc}')

    def _on_status(self, msg: OperatorStatus) -> None:
        try:
            self._status_seq += 1
            now = time.monotonic()
            if self._last_status_at is not None:
                period = 1000.0 * (now - self._last_status_at)
                self.status_period_ms = min(2000.0, max(20.0, period))
            self._last_status_at = now
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.sim_speed.update(stamp, float(msg.wall_time))
            payload = status_payload(msg, self.run_id, self._status_seq, self.panel,
                                     self.status_stale_ms, self.status_period_ms,
                                     self.sim_speed.text())
            self.outbox.set_status(payload, time.time() - msg.wall_time)
            self._wake()
        except Exception as exc:  # noqa: B902 - feedback must never take the bridge down
            self._feedback_error(f'status: {type(exc).__name__}: {exc}')

    def _feedback_error(self, text: str) -> None:
        self.feedback_errors += 1
        self.get_logger().warning(f'operator feedback: {text}',
                                  throttle_duration_sec=LOG_THROTTLE_S)

    def feedback_texts(self) -> List[Tuple[str, str]]:
        """Return (type, JSON text) to send now (called by the sender, in its event loop)."""
        out = []
        for payload, age in self.outbox.take():
            out.append((payload['type'],
                        encode(payload, age, echo_of(self.last_frame, time.monotonic()))))
        return out

    # ---- the single entry point for live and replayed text ----

    def handle_text(self, text: str, t_recv: Optional[float] = None) -> None:
        """Record, parse, convert and publish one relay text message. Never raises (INV-7)."""
        try:
            if self.recorder is not None:
                self.recorder.write_text(text, t_recv)
            result = parse_text(text)
            if isinstance(result, Ignored):
                self.ignored += 1
                if result.type not in self.ignored_types:
                    self.get_logger().info(f'ignoring relay message type {result.type!r}')
                self.ignored_types[result.type] = self.ignored_types.get(result.type, 0) + 1
                return
            if isinstance(result, Malformed):
                self._count_malformed(result.reason)
                return
            stamp = self.get_clock().now().to_msg()
            messages = self.build_messages(result, stamp)
            self.publish_frame(messages)
            self.frames += 1
            if result.session_id != self.last_session_id:
                self.get_logger().info(f'XR session id {result.session_id} '
                                       f'(was {self.last_session_id}); quest_world redefined')
            else:
                if self.last_seq is not None and result.seq != self.last_seq + 1:
                    self.get_logger().warning(f'sequence jump {self.last_seq} -> {result.seq}',
                                              throttle_duration_sec=LOG_THROTTLE_S)
                if (self.last_reset_count is not None
                        and result.reset_count != self.last_reset_count):
                    self._log_reset(result, stamp)
            if result.reset_event is not None:
                self.get_logger().info(
                    f'reset_event at ROS time {_ros_time(stamp)}, seq {result.seq}: '
                    f'count {result.reset_event.count}, {_reset_movement(result.reset_event)}')
            self.last_session_id = result.session_id
            self.last_seq = result.seq
            self.last_reset_count = result.reset_count
            if self.feedback_enabled:
                self.last_frame = (result.session_id, result.seq, result.client_time_ms,
                                   time.monotonic())
        except Exception as exc:  # noqa: B902 - a frame must never take the node down
            self._count_malformed(f'unexpected {type(exc).__name__}: {exc}')

    def build_messages(self, frame: ParsedFrame, stamp) -> FrameMessages:
        """Build the messages for one parsed frame; every message carries ``stamp``."""
        out = FrameMessages()
        status = InputStatus()
        status.header.stamp = stamp
        status.header.frame_id = self.frame_id
        status.session_id = int(frame.session_id)
        status.seq = int(frame.seq)
        status.xr_time_ms = float(frame.xr_time_ms)
        status.client_time_ms = float(frame.client_time_ms)
        status.reset_count = int(frame.reset_count)
        status.malformed_count = int(self.malformed)
        status.ignored_count = int(self.ignored)
        status.source = self.source
        status.viewer_present = frame.viewer is not None
        status.viewer_emulated_position = bool(frame.viewer is not None
                                               and frame.viewer.emulated_position)

        hand_like = [hand for hand in HANDS
                     if getattr(frame, hand) is not None and frame.is_hand_tracked(hand)]
        if hand_like:
            self._warn_hand_like(hand_like, frame)
        for hand in HANDS:
            # Only a source that looks like a real controller counts (defence in depth
            # against a tracked hand presented as a controller-like source).
            sample = frame.controller(hand)
            hand_status = HandStatus()
            hand_status.hand_tracked = frame.is_hand_tracked(hand)
            if sample is None:
                # Absent or a tracked hand: no pose, nothing substituted (INV-2).
                hand_status.pose_present = False
                hand_status.emulated_position = False
            else:
                hand_status.pose_present = True
                hand_status.emulated_position = bool(sample.emulated_position)
                # The one axis conversion, once per present controller pose (INV-1).
                out.poses[hand] = self._pose_stamped(sample.position, sample.orientation, stamp)
            setattr(status, hand, hand_status)

        if frame.viewer is not None:
            # The headset pose: same conversion, same stamp, never a controller pose.
            out.head = self._pose_stamped(frame.viewer.position, frame.viewer.orientation, stamp)

        axes, buttons = build_joy_arrays(frame)
        joy = Joy()
        joy.header.stamp = stamp
        joy.header.frame_id = self.frame_id
        joy.axes = [float(a) for a in axes]
        joy.buttons = [int(b) for b in buttons]
        out.joy = joy
        out.status = status
        return out

    def _pose_stamped(self, xr_position, xr_orientation, stamp) -> PoseStamped:
        """Convert one WebXR pose with the one axis conversion (INV-1) into a PoseStamped."""
        position, orientation = xr_pose_to_ros(xr_position, xr_orientation)
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = orientation
        return pose

    def publish_frame(self, messages: FrameMessages) -> None:
        """Publish the messages of one frame; a no-op once the context is shut down."""
        if not self.context.ok():
            return
        for hand, pose in messages.poses.items():
            self.pose_pubs[hand].publish(pose)
        if messages.head is not None:
            self.head_pub.publish(messages.head)
        if messages.joy is not None:
            self.joy_pub.publish(messages.joy)
        if messages.status is not None:
            self.status_pub.publish(messages.status)

    # ---- bookkeeping ----

    def _log_reset(self, frame: ParsedFrame, stamp) -> None:
        """Log (never throttled) a change of the reference-space reset count."""
        event = frame.reset_event
        movement = (_reset_movement(event) if event is not None
                    else 'movement not reported by the headset')
        self.get_logger().info(
            f'reference-space reset at ROS time {_ros_time(stamp)}, seq {frame.seq}: '
            f'reset_count {self.last_reset_count} -> {frame.reset_count}, {movement}')

    def _warn_hand_like(self, hands: List[str], frame: ParsedFrame) -> None:
        what = []
        for hand in hands:
            source = getattr(frame, f'{hand}_source')
            what.append(f'{hand} ({source.button_count} buttons, {source.axis_count} axes, '
                        f'profiles {list(source.profiles)})')
        joined = ', '.join(what)
        self.get_logger().warning(
            f'source looks like a tracked hand, published as not present + hand_tracked: {joined}',
            throttle_duration_sec=LOG_THROTTLE_S)

    def _count_malformed(self, reason: str) -> None:
        self.malformed += 1
        now = time.monotonic()
        if now - self._last_warn >= LOG_THROTTLE_S:
            self._last_warn = now
            self.get_logger().warning(f'malformed frame #{self.malformed}: {reason}')

    def _log_stats(self) -> None:
        period = float(self.get_parameter('stats_period_s').value)
        new_frames = self.frames - self._frames_at_last_stats
        self._frames_at_last_stats = self.frames
        rate = new_frames / period if period > 0 else 0.0
        feedback = ''
        if self.feedback_enabled:
            box = self.outbox
            feedback = (f' feedback: sent haptic={self.feedback_sent["teleop_haptic"]} '
                        f'status={self.feedback_sent["teleop_status"]} dropped old '
                        f'haptic={box.dropped_old_haptic} status={box.dropped_old_status} '
                        f'overflow={box.dropped_overflow} stalls={self.feedback_stalls} '
                        f'errors={self.feedback_errors}')
        self.get_logger().info(
            f'{self.source}: connected={self.connected} {rate:.1f} frames/s '
            f'frames={self.frames} malformed={self.malformed} ignored={self.ignored} '
            f'last_seq={self.last_seq}' + feedback)

    def close(self) -> None:
        """Flush and close the recorder, if any."""
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None


class WebSocketSource:
    """Feeds relay text to a node from a background thread, reconnecting forever."""

    def __init__(self, node: BridgeNode) -> None:
        self.node = node
        self.url = str(node.get_parameter('relay_url').value)
        self.reconnect_interval = max(0.1, float(node.get_parameter('reconnect_interval_s').value))
        self._stop = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._thread = threading.Thread(target=self._run, name='relay-websocket', daemon=True)

    def start(self) -> None:
        """Start the background thread."""
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Cancel the connection and join the thread."""
        self._stop.set()
        if self._loop is not None and self._task is not None:
            try:
                self._loop.call_soon_threadsafe(self._task.cancel)
            except RuntimeError:
                pass
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # noqa: B902 - must not bring down the process
            self.node.get_logger().error(f'websocket thread ended: {exc!r}')

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        log = self.node.get_logger()
        attempt = 0
        while not self._stop.is_set():
            attempt += 1
            try:
                async with websockets.connect(self.url) as ws:
                    self.node.connected = True
                    attempt = 0
                    log.info(f'connected to relay {self.url}')
                    sender = self._start_sender(ws)
                    try:
                        async for message in ws:
                            if self._stop.is_set():
                                break
                            if isinstance(message, bytes):
                                message = message.decode('utf-8', errors='replace')
                            self.node.handle_text(message, time.time())
                    finally:
                        self._stop_sender(sender)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: B902 - connection errors: log, wait, retry
                if (attempt <= 1 or attempt % 30 == 0) and self.node.context.ok():
                    log.warning(f'relay {self.url} unavailable ({type(exc).__name__}: {exc}); '
                                f'retrying every {self.reconnect_interval:.1f} s')
            finally:
                if self.node.connected and not self._stop.is_set() and self.node.context.ok():
                    log.warning('relay connection closed')
                self.node.connected = False
            if self._stop.is_set():
                break
            try:
                await asyncio.sleep(self.reconnect_interval)
            except asyncio.CancelledError:
                break

    # ---- operator feedback: a sender beside the reader, on the same connection ----

    def _start_sender(self, ws) -> Optional[asyncio.Task]:
        if not self.node.feedback_enabled:
            return None
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        self.node.feedback_wake = lambda: loop.call_soon_threadsafe(wake.set)
        wake.set()                                  # send whatever is still young enough
        return asyncio.create_task(self._send_loop(ws, wake), name='feedback-sender')

    def _stop_sender(self, sender: Optional[asyncio.Task]) -> None:
        if sender is None:
            return
        self.node.feedback_wake = None
        sender.cancel()

    async def _send_loop(self, ws, wake: asyncio.Event) -> None:
        """Send queued feedback whenever woken; skip while the link does not drain."""
        node = self.node
        while True:
            await wake.wait()
            wake.clear()
            try:
                if _write_buffer_size(ws) > node.send_buffer_limit:
                    # The link is not draining: send nothing, let the items age out, retry.
                    node.feedback_stalls += 1
                    node.outbox.expire()
                    await asyncio.sleep(0.05)
                    wake.set()
                    continue
                for kind, text in node.feedback_texts():
                    await ws.send(text)
                    node.feedback_sent[kind] += 1
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosed:
                return
            except Exception as exc:  # noqa: B902 - the reader must not be affected
                node._feedback_error(f'send: {type(exc).__name__}: {exc}')


def _write_buffer_size(ws) -> int:
    """Bytes the connection has not yet handed to the network (0 if unknown)."""
    transport = getattr(ws, 'transport', None)
    try:
        return int(transport.get_write_buffer_size()) if transport is not None else 0
    except Exception:  # noqa: B902
        return 0


def main(args=None) -> None:
    """Run the live bridge: relay WebSocket in, /teleop/input/* out."""
    rclpy.init(args=args)
    try:
        node = BridgeNode()
    except FileExistsError as err:
        rclpy.logging.get_logger('teleop_input_bridge').error(f'record_path: {err}')
        rclpy.try_shutdown()
        raise SystemExit(1)
    source = WebSocketSource(node)
    source.start()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        source.stop()
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
