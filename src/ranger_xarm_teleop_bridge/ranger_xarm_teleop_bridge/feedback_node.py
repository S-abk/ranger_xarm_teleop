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
Operator feedback node: what the operator feels and sees, from the teleop and episode state.

Observes /teleop/state (every 50 Hz message, so a one-cycle fault is not lost),
/teleop/episode/state and /teleop/reset/state; decides nothing for teleop, the gate, the episode
node or the reset node. Publishes
  /teleop/haptic           ranger_xarm_teleop_interfaces/Haptic          one per new episode or
                                                                         reset event
  /teleop/operator/status  ranger_xarm_teleop_interfaces/OperatorStatus  5 Hz, on the steady clock
The bridge forwards both to the headset over its relay connection. The logic is in the pure
modules ``episode_feedback``, ``reset_feedback`` and ``operator_status``; this node only converts
messages.
"""

import math
import time
from typing import Dict

from ranger_xarm_teleop_bridge.episode_feedback import (DEFAULT_PATTERNS, EpisodeEvents,
                                                        EpisodeSample, EVENTS, Pattern)
from ranger_xarm_teleop_bridge.operator_status import (EpisodeView, ResetView, StatusTracker,
                                                       TeleopSample)
from ranger_xarm_teleop_bridge.reset_feedback import (DEFAULT_PATTERNS as RESET_DEFAULT_PATTERNS,
                                                      EVENTS as RESET_EVENTS, ResetEvents,
                                                      ResetSample)
from ranger_xarm_teleop_interfaces.msg import (EpisodeState, Haptic, OperatorStatus, ResetState,
                                               TeleopState)
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

TELEOP_STATE_TOPIC = '/teleop/state'
EPISODE_STATE_TOPIC = '/teleop/episode/state'
RESET_STATE_TOPIC = '/teleop/reset/state'
HAPTIC_TOPIC = '/teleop/haptic'
STATUS_TOPIC = '/teleop/operator/status'

# Every TeleopState is wanted (latching), so a reliable queue like the publisher's.
TELEOP_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                        depth=10, durability=DurabilityPolicy.VOLATILE)
# As published (reliable, transient local); depth 10 so back-to-back events do not collapse.
EPISODE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
# Reliable, depth 10, volatile, as the reset node publishes (10 Hz and on change).
RESET_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                       depth=10, durability=DurabilityPolicy.VOLATILE)
HAPTIC_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                        depth=10, durability=DurabilityPolicy.VOLATILE)
# Contract: 5 Hz, reliable, depth 1, volatile.
STATUS_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                        depth=1, durability=DurabilityPolicy.VOLATILE)


def _seconds(t) -> float:
    return t.sec + 1e-9 * t.nanosec


class FeedbackNode(Node):
    """Publishes vibration requests and the operator status."""

    def __init__(self, **node_kwargs) -> None:
        super().__init__('operator_feedback', **node_kwargs)
        p = self.declare_parameter
        self.hand = str(p('hand', 'left').value)
        if self.hand not in ('left', 'right'):
            raise ValueError("hand must be 'left' or 'right'")
        self.max_age = float(p('haptic_max_age_s', 0.75).value)
        rate = float(p('status_rate_hz', 5.0).value)
        # At least two statuses per second: the page shows NO DATA after 1 s without one.
        if not 2.0 <= rate <= 30.0:
            raise ValueError('status_rate_hz must be 2..30')
        patterns: Dict[str, Pattern] = {}
        for event in EVENTS:
            default = DEFAULT_PATTERNS[event]
            flat = p(f'haptic.{event}', [float(v) for v in default.flat()]).value
            priority = p(f'haptic_priority.{event}', default.priority).value
            patterns[event] = Pattern.from_flat(flat, priority)
        self.events = EpisodeEvents(patterns, self.max_age)
        self.reset_hand = str(p('reset_hand', 'right').value)
        if self.reset_hand not in ('left', 'right'):
            raise ValueError("reset_hand must be 'left' or 'right'")
        reset_patterns: Dict[str, Pattern] = {}
        for event in RESET_EVENTS:
            default = RESET_DEFAULT_PATTERNS[event]
            flat = p(f'reset_haptic.{event}', [float(v) for v in default.flat()]).value
            priority = p(f'reset_haptic_priority.{event}', default.priority).value
            reset_patterns[event] = Pattern.from_flat(flat, priority)
        self.reset_events = ResetEvents(reset_patterns, self.max_age)
        self.tracker = StatusTracker(
            teleop_stale_s=float(p('teleop_stale_s', 0.5).value),
            episode_stale_s=float(p('episode_stale_s', 1.0).value),
            fault_latch_s=float(p('fault_latch_s', 10.0).value),
            limit_hold_s=float(p('limit_hold_s', 1.0).value),
            reset_stale_s=float(p('reset_stale_s', 0.5).value),
            # From the limits file when it is passed to this node (NaN: shown in rad).
            gripper_open=float(p('gripper_open_position', math.nan).value),
            gripper_closed=float(p('gripper_closed_position', math.nan).value))
        self.haptic_pub = self.create_publisher(Haptic, HAPTIC_TOPIC, HAPTIC_QOS)
        self.status_pub = self.create_publisher(OperatorStatus, STATUS_TOPIC, STATUS_QOS)
        self.create_subscription(TeleopState, TELEOP_STATE_TOPIC, self.on_teleop, TELEOP_QOS)
        self.create_subscription(EpisodeState, EPISODE_STATE_TOPIC, self.on_episode,
                                 EPISODE_QOS)
        self.create_subscription(ResetState, RESET_STATE_TOPIC, self.on_reset, RESET_QOS)
        # The steady clock: the status keeps its rate when simulation time is slow or paused.
        self.create_timer(1.0 / rate, self.publish_status,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.sent = 0
        self.get_logger().info(
            f'operator feedback: {self.hand} hand, status {rate:g} Hz, haptic max age '
            f'{self.max_age:g} s; patterns ' + ', '.join(
                f'{e}={list(patterns[e].flat())}' for e in EVENTS)
            + f'; reset events to the {self.reset_hand} hand: ' + ', '.join(
                f'{e}={list(reset_patterns[e].flat())}' for e in RESET_EVENTS))

    def on_teleop(self, m: TeleopState) -> None:
        self.tracker.on_teleop(TeleopSample(
            m.motion_enabled, m.armed, m.clutch_engaged, m.calibrated, m.translation_scale,
            m.hold_reason, m.active_limit, m.gripper_enabled, m.gripper_active, m.gripper_hold,
            m.gripper_request, m.gripper_measured_valid, m.gripper_measured, m.mux_status),
            time.monotonic())

    def on_episode(self, m: EpisodeState) -> None:
        now, wall = time.monotonic(), time.time()
        sample = EpisodeSample(m.session, m.event_count, m.last_event, m.detail,
                               _seconds(m.header.stamp), _seconds(m.last_event_stamp),
                               m.wall_time)
        missed_before = self.events.missed_total
        too_old_before = self.events.too_old
        request = self.events.update(sample, wall)
        self.tracker.events_missed = self.events.missed_total
        self.tracker.on_episode(EpisodeView(
            m.state, m.episode_index, m.elapsed, m.kept_count, m.discarded_count,
            m.discard_armed, m.last_event, m.detail, sample.event_wall_time, m.event_count,
            m.disk_minutes_left), now)
        if self.events.missed_total > missed_before:
            self.get_logger().warning(
                f'{self.events.missed_total - missed_before} episode event(s) missed between '
                f'state messages (event_count {m.event_count}); not felt')
        if self.events.too_old > too_old_before:
            self.get_logger().warning(f'episode event {m.event_count} older than '
                                      f'{self.max_age:g} s on arrival: not sent')
        if request is not None:
            self.publish_haptic(request)

    def on_reset(self, m: ResetState) -> None:
        now, wall = time.monotonic(), time.time()
        sample = ResetSample(m.event_count, m.last_event, m.last_event_wall_time)
        missed_before = self.reset_events.missed_total
        too_old_before = self.reset_events.too_old
        request = self.reset_events.update(sample, wall)
        self.tracker.on_reset(ResetView(m.state, m.reason, m.posture, m.moveit_error_code,
                                        m.last_event, m.last_event_wall_time), now)
        if self.reset_events.missed_total > missed_before:
            self.get_logger().warning(
                f'{self.reset_events.missed_total - missed_before} reset event(s) missed '
                f'between state messages (event_count {m.event_count}); not felt')
        if self.reset_events.too_old > too_old_before:
            self.get_logger().warning(f'reset event {m.event_count} older than '
                                      f'{self.max_age:g} s on arrival (or its time unknown): '
                                      'not sent')
        if request is not None:
            self.publish_haptic(request, self.reset_hand, 'reset')

    def publish_haptic(self, r, hand=None, source='episode') -> None:
        hand = self.hand if hand is None else hand
        msg = Haptic()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.wall_time = float(r.wall_time)
        msg.hand = hand
        msg.source = source
        msg.event = r.event
        msg.priority = int(r.pattern.priority)
        msg.intensity = [float(i) for i, _, _ in r.pattern.pulses]
        msg.on_ms = [int(on) for _, on, _ in r.pattern.pulses]
        msg.off_ms = [int(off) for _, _, off in r.pattern.pulses]
        msg.max_age = self.max_age
        msg.source_count = int(r.source_count)
        msg.missed = int(r.missed)
        self.haptic_pub.publish(msg)
        self.sent += 1
        if source == 'episode':             # the episode's line as it was before the reset
            self.get_logger().info(f'haptic {r.event} (event {r.source_count}) to the {hand} '
                                   'controller')
        else:
            self.get_logger().info(f'haptic {r.event} ({source} event {r.source_count}) to the '
                                   f'{hand} controller')

    def publish_status(self) -> None:
        snap = self.tracker.snapshot(time.monotonic(), time.time())
        msg = OperatorStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.wall_time = time.time()
        for name, value in vars(snap).items():
            setattr(msg, name, value)
        self.status_pub.publish(msg)


def main(args=None) -> None:
    """Run the operator feedback node."""
    rclpy.init(args=args)
    node = None
    try:
        node = FeedbackNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
