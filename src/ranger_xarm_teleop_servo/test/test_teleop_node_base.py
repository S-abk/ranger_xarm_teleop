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
The arm's teleop node with base driving on (G-19, INV-2), on the recorded clutch_line stroke.

The node reads the base's odometry and the base command, never publishes toward the base,
and refuses the clutch while the left grip is held, while odometry is missing or says the
base moves, and after the base turned past the set angle since the calibration. With base
driving off it subscribes to neither.
"""

from dataclasses import replace
import math

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import pytest
from ranger_xarm_teleop_interfaces.msg import TeleopState
from ranger_xarm_teleop_servo.teleop_node import StartupRefused, TeleopNode
from ranger_xarm_teleop_bridge.bridge_node import BridgeNode  # noqa: I100
import rclpy
from test_teleop_node import Harness, LIMITS, overrides  # noqa: I100

BASE = {'base_driving': True, 'base_turn_recalibrate_deg': 5.0}


class BaseHarness(Harness):
    """The harness with odometry delivered every cycle as the test sets it."""

    def __init__(self, bridge, **params):
        super().__init__(bridge, **params)
        self.odom = True
        self.speed = 0.0
        self.yaw = 0.3
        self.left_grip = False
        build = self.node.build_frame
        self.node.build_frame = lambda status, arrival: replace(
            build(status, arrival), left_grip=self.left_grip)

    def tick(self, now):
        self.now = now
        if self.odom:
            msg = Odometry()
            msg.twist.twist.linear.x = self.speed
            msg.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
            self.node._on_base_odom(msg)
        super().tick(now)


@pytest.fixture(scope='module')
def bridge():
    rclpy.init()
    node = BridgeNode(node_name='bridge_for_base_test', source_label='test')
    yield node
    node.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def base(bridge):
    made = []

    def make(plain=False, **kwargs):
        made.append(Harness(bridge, **kwargs) if plain
                    else BaseHarness(bridge, **{**BASE, **kwargs}))
        return made[-1]
    yield make
    for h in made:
        h.close()


def holds(h):
    return {s.hold_reason for s in h.states}


def subscribed(node):
    return {s.topic_name for s in node.subscriptions}


def published(node):
    return {p.topic_name for p in node.publishers}


def test_base_driving_off_reads_nothing_of_the_base(base):
    h = base(plain=True)
    assert not {'/odom', '/teleop/base/cmd_vel'} & subscribed(h.node)
    assert not h.node.base_driving


def test_base_driving_needs_the_calibration_turn_value(bridge):
    values = {**LIMITS, 'base_driving': True}
    with pytest.raises(StartupRefused, match='base_turn_recalibrate_deg is required'):
        TeleopNode(node_name='teleop_base_missing', parameter_overrides=overrides(values))


def test_the_node_reads_the_base_and_publishes_nothing_toward_it(base):
    """INV-4: subscriptions only; no publisher of a base command."""
    h = base()
    assert {'/odom', '/teleop/base/cmd_vel'} <= subscribed(h.node)
    assert not {'/cmd_vel', '/teleop/base/cmd_vel'} & published(h.node)


def test_with_the_base_at_rest_the_stroke_is_the_same(base):
    h = base()
    h.play('clutch_line')
    plain = base(plain=True)
    plain.play('clutch_line')
    assert len(h.targets) == len(plain.targets) > 100


def test_the_left_grip_refuses_the_clutch(base):
    h = base()
    h.left_grip = True
    h.play('clutch_line')
    assert not h.targets
    assert TeleopState.HOLD_BASE_DRIVE in holds(h)


def test_no_odometry_refuses_the_clutch(base):
    h = base()
    h.odom = False
    h.play('clutch_line')
    assert not h.targets
    assert TeleopState.HOLD_BASE_ODOM in holds(h)


def test_a_moving_base_holds_the_clutch_until_it_rests(base):
    h = base()
    h.speed = 0.05
    h.play('clutch_line')
    assert not h.targets
    assert TeleopState.HOLD_BASE_MOVING in holds(h)


def test_a_turn_past_the_limit_since_the_calibration_refuses_the_clutch(base):
    h = base()
    original = h.tick

    def turning(now):
        if now > 1.0:
            h.yaw = 0.3 + math.radians(5.5)
        original(now)
    h.tick = turning
    h.play('clutch_line')
    assert TeleopState.HOLD_BASE_TURNED in holds(h)


def test_a_turn_within_the_limit_keeps_the_calibration(base):
    h = base()
    original = h.tick

    def turning(now):
        if now > 1.0:
            h.yaw = 0.3 + math.radians(4.5)
        original(now)
    h.tick = turning
    h.play('clutch_line')
    assert TeleopState.HOLD_BASE_TURNED not in holds(h)
    assert len(h.targets) > 100


def test_a_turn_commanded_while_odometry_was_missing_refuses_the_clutch(base):
    h = base()
    original = h.tick
    sent = []

    def gap(now):
        h.odom = not (0.8 < now < 1.4)
        if 1.0 < now and not sent:
            msg = TwistStamped()
            msg.twist.angular.z = 0.2
            h.node._on_base_cmd(msg)
            sent.append(now)
        original(now)
    h.tick = gap
    h.play('clutch_line')
    assert TeleopState.HOLD_BASE_TURNED in holds(h)
