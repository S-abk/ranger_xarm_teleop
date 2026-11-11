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
  /teleop/input/joy                                  sensor_msgs/Joy
  /teleop/input/status                               ranger_xarm_teleop_interfaces/InputStatus
"""

import asyncio
from dataclasses import dataclass, field
import threading
import time
from typing import Dict, Optional

from geometry_msgs.msg import PoseStamped
from ranger_xarm_teleop_bridge.axes import xr_pose_to_ros
from ranger_xarm_teleop_bridge.frame_parse import Ignored, Malformed, parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.joy_layout import build_joy_arrays
from ranger_xarm_teleop_bridge.jsonl_io import RecordWriter
from ranger_xarm_teleop_interfaces.msg import HandStatus, InputStatus
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
import websockets

HANDS = ('left', 'right')
POSE_TOPICS = {hand: f'/teleop/input/{hand}/pose' for hand in HANDS}
JOY_TOPIC = '/teleop/input/joy'
STATUS_TOPIC = '/teleop/input/status'

# Interface contract: poses and joy best effort, depth 1; status reliable, depth 10.
POSE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                      depth=1, durability=DurabilityPolicy.VOLATILE)
JOY_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                     depth=1, durability=DurabilityPolicy.VOLATILE)
STATUS_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                        depth=10, durability=DurabilityPolicy.VOLATILE)

LOG_THROTTLE_S = 2.0


@dataclass
class FrameMessages:
    """The messages built from one headset frame; they share one stamp."""

    poses: Dict[str, PoseStamped] = field(default_factory=dict)
    joy: Optional[Joy] = None
    status: Optional[InputStatus] = None


class BridgeNode(Node):
    """Publishes the input side of the interface contract from relay text messages."""

    def __init__(self, node_name: str = 'teleop_input_bridge',
                 source_label: str = 'relay') -> None:
        super().__init__(node_name)
        self.declare_parameter('relay_url', 'ws://127.0.0.1:8443/ws')
        self.declare_parameter('reconnect_interval_s', 1.0)
        self.declare_parameter('frame_id', 'quest_world')
        self.declare_parameter('record_path', '')
        self.declare_parameter('source', source_label)
        self.declare_parameter('stats_period_s', 5.0)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.source = str(self.get_parameter('source').value)

        self.pose_pubs = {hand: self.create_publisher(PoseStamped, POSE_TOPICS[hand], POSE_QOS)
                          for hand in HANDS}
        self.joy_pub = self.create_publisher(Joy, JOY_TOPIC, JOY_QOS)
        self.status_pub = self.create_publisher(InputStatus, STATUS_TOPIC, STATUS_QOS)

        self.frames = 0
        self.malformed = 0
        self.ignored = 0
        self.ignored_types: Dict[str, int] = {}
        self.last_seq: Optional[int] = None
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
            if self.last_seq is not None and result.seq != self.last_seq + 1:
                self.get_logger().warning(f'sequence jump {self.last_seq} -> {result.seq}',
                                          throttle_duration_sec=LOG_THROTTLE_S)
            self.last_seq = result.seq
        except Exception as exc:  # noqa: B902 - a frame must never take the node down
            self._count_malformed(f'unexpected {type(exc).__name__}: {exc}')

    def build_messages(self, frame: ParsedFrame, stamp) -> FrameMessages:
        """Build the messages for one parsed frame; every message carries ``stamp``."""
        out = FrameMessages()
        status = InputStatus()
        status.header.stamp = stamp
        status.header.frame_id = self.frame_id
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

        for hand in HANDS:
            sample = getattr(frame, hand)
            hand_status = HandStatus()
            if sample is None:
                hand_status.pose_present = False
                hand_status.emulated_position = False
            else:
                hand_status.pose_present = True
                hand_status.emulated_position = bool(sample.emulated_position)
                # The one axis conversion, once per present controller pose (INV-1).
                position, orientation = xr_pose_to_ros(sample.position, sample.orientation)
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = self.frame_id
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
                (pose.pose.orientation.x, pose.pose.orientation.y,
                 pose.pose.orientation.z, pose.pose.orientation.w) = orientation
                out.poses[hand] = pose
            setattr(status, hand, hand_status)

        axes, buttons = build_joy_arrays(frame)
        joy = Joy()
        joy.header.stamp = stamp
        joy.header.frame_id = self.frame_id
        joy.axes = [float(a) for a in axes]
        joy.buttons = [int(b) for b in buttons]
        out.joy = joy
        out.status = status
        return out

    def publish_frame(self, messages: FrameMessages) -> None:
        """Publish the messages of one frame; a no-op once the context is shut down."""
        if not self.context.ok():
            return
        for hand, pose in messages.poses.items():
            self.pose_pubs[hand].publish(pose)
        if messages.joy is not None:
            self.joy_pub.publish(messages.joy)
        if messages.status is not None:
            self.status_pub.publish(messages.status)

    # ---- bookkeeping ----

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
        self.get_logger().info(
            f'{self.source}: connected={self.connected} {rate:.1f} frames/s '
            f'frames={self.frames} malformed={self.malformed} ignored={self.ignored} '
            f'last_seq={self.last_seq}')

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
                    async for message in ws:
                        if self._stop.is_set():
                            break
                        if isinstance(message, bytes):
                            message = message.decode('utf-8', errors='replace')
                        self.node.handle_text(message, time.time())
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: B902 - connection errors: log, wait, retry
                if attempt <= 1 or attempt % 30 == 0:
                    log.warning(f'relay {self.url} unavailable ({type(exc).__name__}: {exc}); '
                                f'retrying every {self.reconnect_interval:.1f} s')
            finally:
                if self.node.connected:
                    log.warning('relay connection closed')
                self.node.connected = False
            if self._stop.is_set():
                break
            try:
                await asyncio.sleep(self.reconnect_interval)
            except asyncio.CancelledError:
                break


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
