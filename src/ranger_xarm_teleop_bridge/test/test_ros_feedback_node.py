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
"""The operator feedback node on a ROS graph: one Haptic per event, status 5 Hz, contract QoS."""

import threading
import time

import pytest
from ranger_xarm_teleop_bridge import feedback_node as fn
from ranger_xarm_teleop_bridge.feedback_node import FeedbackNode
from ranger_xarm_teleop_interfaces.msg import EpisodeState, Haptic, OperatorStatus, TeleopState
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy


class Rig:

    def __init__(self):
        self.node = FeedbackNode()
        self.probe = rclpy.create_node('feedback_node_probe')
        self.haptics, self.statuses = [], []
        self.probe.create_subscription(Haptic, fn.HAPTIC_TOPIC, self.haptics.append, fn.HAPTIC_QOS)
        self.probe.create_subscription(OperatorStatus, fn.STATUS_TOPIC,
                                       lambda m: self.statuses.append((time.monotonic(), m)),
                                       fn.STATUS_QOS)
        self.episode_pub = self.probe.create_publisher(EpisodeState, fn.EPISODE_STATE_TOPIC,
                                                       fn.EPISODE_QOS)
        self.teleop_pub = self.probe.create_publisher(TeleopState, fn.TELEOP_STATE_TOPIC,
                                                      fn.TELEOP_QOS)
        self.executors = []
        for n in (self.node, self.probe):
            ex = SingleThreadedExecutor()
            ex.add_node(n)
            threading.Thread(target=ex.spin, daemon=True).start()
            self.executors.append(ex)
        deadline = time.monotonic() + 5
        while (self.episode_pub.get_subscription_count() == 0
               or self.probe.count_publishers(fn.STATUS_TOPIC) == 0) \
                and time.monotonic() < deadline:
            time.sleep(0.05)

    def episode(self, count, event, state=EpisodeState.STATE_RECORDING, detail='',
                session='20261118-000000-abcd', index=1):
        m = EpisodeState()
        now = self.probe.get_clock().now().to_msg()
        m.header.stamp, m.last_event_stamp, m.wall_time = now, now, time.time()
        m.session, m.event_count, m.last_event, m.detail = session, count, event, detail
        m.state, m.episode_index = state, index
        self.episode_pub.publish(m)
        time.sleep(0.05)

    def teleop(self, hold=TeleopState.HOLD_CLUTCH_RELEASED, limit=0):
        m = TeleopState()
        m.armed, m.calibrated, m.motion_enabled = True, True, True
        m.hold_reason, m.active_limit, m.translation_scale = hold, limit, 0.5
        self.teleop_pub.publish(m)

    def close(self):
        for ex in self.executors:
            ex.shutdown()
        self.node.destroy_node()
        self.probe.destroy_node()


@pytest.fixture
def rig():
    rclpy.init()
    r = Rig()
    yield r
    r.close()
    rclpy.shutdown()


def test_each_event_gives_exactly_one_haptic_and_repeats_none(rig):
    E = EpisodeState
    rig.episode(3, E.EVENT_KEEP, E.STATE_IDLE)                  # latched old state: baseline
    for count, event in [(4, E.EVENT_START), (4, E.EVENT_START), (5, E.EVENT_MARK),
                         (6, E.EVENT_STOP), (6, E.EVENT_STOP), (7, E.EVENT_KEEP),
                         (8, E.EVENT_REFUSED), (10, E.EVENT_AUTO_STOP), (10, E.EVENT_AUTO_STOP)]:
        rig.episode(count, event, detail='teleop is not armed' if event == E.EVENT_REFUSED else '')
    time.sleep(0.3)
    assert [h.event for h in rig.haptics] == ['START', 'MARK', 'STOP', 'KEEP', 'REFUSED',
                                              'AUTO_STOP']
    assert [h.missed for h in rig.haptics] == [0, 0, 0, 0, 0, 1]
    refused = rig.haptics[4]
    assert refused.hand == 'left' and refused.source == 'episode' and refused.priority == 3
    assert list(refused.on_ms) == [350, 350] and list(refused.off_ms) == [200, 0]
    assert refused.max_age == pytest.approx(0.75)
    status = rig.statuses[-1][1]
    assert status.events_missed == 1 and status.last_refusal_text == 'arm first (B)'


def test_status_rate_content_and_staleness(rig):
    E = EpisodeState
    rig.episode(1, E.EVENT_START, index=4)
    start = time.monotonic()
    while time.monotonic() - start < 2.0:
        rig.teleop(hold=TeleopState.HOLD_INPUT_JUMP if time.monotonic() - start < 0.03
                   else TeleopState.HOLD_CLUTCH_RELEASED,
                   limit=TeleopState.LIMIT_KEEP_OUT if time.monotonic() - start < 0.5 else 0)
        rig.episode(1, E.EVENT_START, index=4)               # the episode node repeats at 5 Hz
        time.sleep(0.02)
    window = [m for t, m in rig.statuses if start + 0.2 <= t <= start + 2.0]
    assert 8 <= len(window) <= 11, len(window)               # 5 Hz over 1.8 s
    last = window[-1]
    assert last.teleop_valid and last.armed and last.translation_scale == 0.5
    assert last.hold_text == 'Ready: hold right grip to move'
    assert 'too fast' in last.last_fault_text                # one-cycle fault, latched
    assert last.limit == 0                                   # keep-out ended 1.5 s ago
    early = [m for t, m in rig.statuses if start + 0.2 <= t <= start + 1.0]
    assert any(m.limit_text == 'keep-out zone' for m in early)
    assert last.episode_valid and last.episode_index == 4 and last.episode_state == 2
    time.sleep(1.2)                                          # both sources fall silent
    s = rig.statuses[-1][1]
    assert not s.teleop_valid and not s.armed and s.hold_text == ''
    assert not s.episode_valid and s.episode_index == 0


def test_a_status_rate_below_two_per_second_is_refused():
    rclpy.init()
    try:
        with pytest.raises(ValueError):
            FeedbackNode(parameter_overrides=[Parameter('status_rate_hz', value=1.0)])
    finally:
        rclpy.shutdown()


def test_publishers_have_the_contract_qos(rig):
    status = rig.node.get_publishers_info_by_topic(fn.STATUS_TOPIC)
    assert [(i.qos_profile.reliability, i.qos_profile.durability) for i in status] == [
        (ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE)]
    assert fn.STATUS_QOS.depth == 1
    assert rig.node.get_publishers_info_by_topic(fn.HAPTIC_TOPIC)
