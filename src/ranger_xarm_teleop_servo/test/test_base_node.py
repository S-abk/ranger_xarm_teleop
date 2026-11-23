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
The base node (G-19), its callbacks driven directly with real messages on a fake clock.

What it publishes is captured. The decisions themselves are tested without ROS
(test/unit/test_base.py, test_base_rest.py); here: the messages in and out, both commands
together and equal, one zero at once and then silence for every trigger (INV-1), the arm's
clutch both ways (INV-2), no path to the arm (INV-4), and a second publisher (INV-5).
"""

import math
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import pytest
from ranger_xarm_teleop_interfaces.msg import BaseState, InputStatus, ResetState, TeleopState
from ranger_xarm_teleop_servo import base_node
from ranger_xarm_teleop_servo.base_node import BaseNode, StartupRefused
import rclpy
from rclpy.parameter import Parameter
import rclpy.time
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String

SIM_DESCRIPTION = ('<robot name="r"><ros2_control name="a" type="system"><hardware><plugin>'
                   'mock_components/GenericSystem</plugin></hardware></ros2_control></robot>')
REAL_DESCRIPTION = ('<robot name="r"><ros2_control name="a" type="system"><hardware><plugin>'
                    'some_vendor/RealArm</plugin></hardware></ros2_control></robot>')
# A base, an arm base above it and one joint; the tool 0.7 m ahead of the turning axis.
ARM_DESCRIPTION = SIM_DESCRIPTION.replace('</robot>', (
    '<link name="base_link"/><link name="arm_base"/><link name="l1"/><link name="tcp"/>'
    '<joint name="mount" type="fixed"><parent link="base_link"/><child link="arm_base"/>'
    '<origin xyz="0 0 0.39"/></joint>'
    '<joint name="j1" type="revolute"><parent link="arm_base"/><child link="l1"/>'
    '<axis xyz="0 0 1"/><limit lower="-3" upper="3" velocity="3"/></joint>'
    '<joint name="tip" type="fixed"><parent link="l1"/><child link="tcp"/>'
    '<origin xyz="0.7 0 0.2"/></joint></robot>'))
PARAMS = {'enabled': True, 'base_linear_cap': 0.2, 'base_turn_cap': 0.4,
          'base_linear_accel': 0.4, 'base_turn_accel': 0.8}
L_TRIGGER, L_GRIP, LX, LY, R_TRIGGER, R_GRIP, RX, RY = range(8)
B_LGRIP, B_RGRIP = 1, 6


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


class FakeTime:

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def time(self):
        return 1.7e9 + self.t


class Rig:
    """A BaseNode on a fake clock with captured publishers and a fake graph."""

    def __init__(self, monkeypatch, description=SIM_DESCRIPTION, **params):
        self.clock = FakeTime()
        monkeypatch.setattr(base_node, 'time', self.clock)
        values = {**PARAMS, **params}
        self.node = BaseNode(node_name='base_under_test', parameter_overrides=[
            Parameter(k, value=v) for k, v in values.items()])
        self.cmds, self.stamped, self.states = [], [], []
        self.node.cmd_pub.publish = self.cmds.append
        self.node.stamped_pub.publish = self.stamped.append
        self.node.state_pub.publish = self.states.append
        self.others = []
        self.node.get_publishers_info_by_topic = lambda topic: list(self.others)
        self.node.count_publishers = lambda topic: 1
        self.seq, self.session = 0, 3
        self.clutch, self.stopping, self.teleop = False, False, True
        self.reset = ResetState.STATE_IDLE
        self.odom, self.moving = True, False
        self.joints = None
        self.node._on_description(String(data=description))

    def stamp(self):
        return rclpy.time.Time(nanoseconds=int(1e9 + self.seq * 1e7)).to_msg()

    def frame(self, grip=False, right_grip=False, left=(0.0, 0.0), right=(0.0, 0.0),
              left_missing=False, pose=True):
        self.seq += 1
        stamp = self.stamp()
        if pose:
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.pose.orientation.w = 1.0
            self.node._on_pose(msg)
        head = PoseStamped()
        head.header.stamp = stamp
        head.pose.orientation.w = 1.0
        self.node._on_head(head)
        joy = Joy()
        joy.header.stamp = stamp
        axes = [0.0] * 8
        buttons = [0] * 10
        axes[L_TRIGGER] = math.nan if left_missing else 0.0
        if not left_missing:
            buttons[B_LGRIP] = int(grip)
            axes[LX], axes[LY] = left
        buttons[B_RGRIP] = int(right_grip)
        axes[RX], axes[RY] = right
        joy.axes, joy.buttons = axes, buttons
        self.node._on_joy(joy)
        status = InputStatus()
        status.header.stamp = stamp
        status.session_id = self.session
        status.seq = self.seq
        status.xr_time_ms = 5000.0 + self.seq * 11.0
        status.right.pose_present = pose
        status.viewer_present = True
        self.node._on_status(status)

    def world(self):
        if self.teleop:
            self.node._on_teleop(TeleopState(clutch_engaged=self.clutch, stopping=self.stopping))
        self.node._on_reset(ResetState(state=self.reset))
        if self.odom:
            msg = Odometry()
            msg.twist.twist.linear.x = 0.1 if self.moving else 0.0
            self.node._on_odom(msg)
        if self.joints is not None:
            self.node._on_joints(JointState(name=['j1'], position=[self.joints]))

    def step(self, frame=True, **kw):
        self.clock.t += 0.02
        if frame:
            self.frame(**kw)
        self.world()
        self.node._cycle()

    def hold(self, seconds, **kw):
        for _ in range(int(round(seconds / 0.02))):
            self.step(**kw)

    def drive(self, seconds=1.0, **kw):
        self.hold(0.4)                      # odometry at rest, the grip released
        self.hold(seconds, grip=True, left=(0.0, -1.0), **kw)
        assert self.node.base.driving, self.states[-1].reason_text
        assert self.cmds[-1].linear.x > 0.0

    def state(self):
        return self.states[-1]

    def close(self):
        self.node.destroy_node()


@pytest.fixture
def rig(ros, monkeypatch):
    made = []

    def make(**params):
        made.append(Rig(monkeypatch, **params))
        return made[-1]
    yield make
    for r in made:
        r.close()


def stops_with_one_zero(r, reason, **kw):
    before = len(r.cmds)
    r.step(**kw)
    assert len(r.cmds) == before + 1, 'zero at once'
    zero = r.cmds[-1]
    assert (zero.linear.x, zero.linear.y, zero.angular.z) == (0.0, 0.0, 0.0)
    assert r.stamped[-1].twist.linear.x == 0.0
    assert r.state().reason == reason, r.state().reason_text
    r.hold(0.5, **kw)
    assert len(r.cmds) == before + 1, 'then publishing stops'


# ---- start-up and configuration ----

@pytest.mark.parametrize('missing', ['base_linear_cap', 'base_turn_cap', 'base_linear_accel',
                                     'base_turn_accel'])
def test_the_node_refuses_to_start_without_a_required_value(ros, missing):
    values = {k: v for k, v in PARAMS.items() if k != missing}
    with pytest.raises(StartupRefused, match=f'{missing} is required'):
        BaseNode(node_name='base_refused', parameter_overrides=[
            Parameter(k, value=v) for k, v in values.items()])


def test_off_by_default_the_grip_does_nothing(rig):
    r = rig(enabled=False)
    r.hold(0.4)
    r.hold(1.0, grip=True, left=(0.0, -1.0))
    assert not r.cmds and not r.stamped
    assert r.state().state == BaseState.STATE_OFF and not r.state().enabled


def test_a_robot_that_is_not_a_simulator_is_never_driven(rig):
    r = rig(description=REAL_DESCRIPTION)
    r.hold(0.4)
    r.hold(1.0, grip=True, left=(0.0, -1.0))
    assert not r.cmds
    assert r.state().reason == BaseState.REASON_NOT_SIMULATION


def test_the_node_publishes_three_topics_and_asks_nothing(rig):
    """No path to the arm: no publisher toward it, no service client, no action (INV-4)."""
    r = rig()
    topics = {pub.topic_name for pub in r.node.publishers}
    assert topics - {'/rosout', '/parameter_events'} == {
        '/cmd_vel', '/teleop/base/cmd_vel', '/teleop/base/state'}
    assert not list(r.node.clients)


# ---- driving ----

def test_both_commands_go_out_together_with_equal_values_at_the_cycle_rate(rig):
    r = rig()
    r.drive(1.0)
    assert len(r.cmds) == len(r.stamped) == 50
    for twist, stamped in zip(r.cmds, r.stamped):
        assert (twist.linear.x, twist.linear.y, twist.angular.z) == (
            stamped.twist.linear.x, stamped.twist.linear.y, stamped.twist.angular.z)
        assert stamped.header.frame_id == 'base_link'
    assert r.cmds[-1].linear.x == pytest.approx(0.2 * 0.98 ** 0, abs=0.01)
    assert r.state().state == BaseState.STATE_DRIVING and r.state().driving


def test_a_turn_from_the_right_stick_and_the_left_stick_wins(rig):
    r = rig()
    r.hold(0.4)
    r.hold(1.0, grip=True, right=(-1.0, 0.0))
    assert r.cmds[-1].angular.z == pytest.approx(0.4) and r.cmds[-1].linear.x == 0.0
    r.hold(1.5, grip=True, left=(0.0, -1.0), right=(-1.0, 0.0))
    assert r.cmds[-1].angular.z == 0.0 and r.cmds[-1].linear.x > 0.0
    assert r.state().turn_ignored


def test_a_pull_back_on_the_right_stick_does_not_turn_the_base(rig):
    r = rig()
    r.hold(0.4)
    r.hold(1.0, grip=True, right=(0.6, 1.0))
    assert all(c.angular.z == 0.0 for c in r.cmds)


# ---- one zero at once, then nothing: one test per trigger (INV-1) ----

def test_release(rig):
    r = rig()
    r.drive()
    stops_with_one_zero(r, BaseState.REASON_RELEASED, grip=False)


def test_stale_input(rig):
    r = rig()
    r.drive()
    r.hold(0.1, frame=False)
    stops_with_one_zero(r, BaseState.REASON_INPUT_STALE, frame=False)


def test_missing_left_controller(rig):
    r = rig()
    r.drive()
    stops_with_one_zero(r, BaseState.REASON_CONTROLLER_MISSING, left_missing=True)


def test_session_change(rig):
    r = rig()
    r.drive()
    r.session += 1
    stops_with_one_zero(r, BaseState.REASON_SESSION_CHANGED, grip=True, left=(0.0, -1.0))


def test_sequence_backwards(rig):
    r = rig()
    r.drive()
    r.seq -= 3
    stops_with_one_zero(r, BaseState.REASON_SEQ_BACKWARDS, grip=True, left=(0.0, -1.0))


def test_the_arms_clutch_engaged_stops_the_base(rig):
    r = rig()
    r.drive()
    r.clutch = True
    stops_with_one_zero(r, BaseState.REASON_CLUTCH_ENGAGED, grip=True, left=(0.0, -1.0))


def test_the_arms_state_going_stale_stops_the_base(rig):
    r = rig()
    r.drive()
    r.teleop = False
    r.hold(0.46, grip=True, left=(0.0, -1.0))
    assert r.node.base.driving                      # fresh for 0.5 s
    before = len(r.cmds)
    r.hold(0.1, grip=True, left=(0.0, -1.0))
    assert not r.node.base.driving and r.state().reason == BaseState.REASON_TELEOP_UNKNOWN
    zero = r.cmds[-1]
    assert (zero.linear.x, zero.angular.z) == (0.0, 0.0)
    assert r.cmds[before:].count(zero) == 1 and r.cmds[-2].linear.x > 0.0


def test_odometry_going_stale_stops_the_base(rig):
    r = rig()
    r.drive()
    r.odom = False
    r.hold(0.2, grip=True, left=(0.0, -1.0))
    stops_with_one_zero(r, BaseState.REASON_ODOM_MISSING, grip=True, left=(0.0, -1.0))


def test_a_reset_starting_stops_the_base(rig):
    r = rig()
    r.drive()
    r.reset = ResetState.STATE_PLANNING
    stops_with_one_zero(r, BaseState.REASON_RESET_ACTIVE, grip=True, left=(0.0, -1.0))


def test_a_second_publisher_stops_the_base_and_is_named(rig):
    r = rig()
    r.drive()

    class Info:
        node_namespace, node_name = '/', 'intruder'
    r.others = [Info()]
    r.hold(0.5, grip=True, left=(0.0, -1.0))          # found within one graph look (2 Hz)
    assert not r.node.base.driving
    assert r.state().reason == BaseState.REASON_OTHER_PUBLISHER
    assert '/intruder' in r.state().reason_text
    assert sum(1 for c in r.cmds[-30:] if c.linear.x == 0.0) == 1


def test_a_second_publisher_refuses_the_start(rig):
    r = rig()

    class Info:
        node_namespace, node_name = '/', 'intruder'
    r.others = [Info()]
    r.hold(0.4)
    r.step(grip=True, left=(0.0, -1.0))
    assert not r.cmds and r.state().state == BaseState.STATE_REFUSED
    assert r.state().reason == BaseState.REASON_OTHER_PUBLISHER


def test_the_shutdown_sends_one_zero(rig):
    r = rig()
    r.drive()
    assert r.node.stop_now()
    zero = r.cmds[-1]
    assert (zero.linear.x, zero.linear.y, zero.angular.z) == (0.0, 0.0, 0.0)
    assert r.state().reason == BaseState.REASON_SHUTDOWN
    assert not r.node.stop_now()


def test_pose_only_conditions_do_not_stop_the_base(rig):
    r = rig()
    r.drive()
    r.hold(0.5, grip=True, left=(0.0, -1.0), pose=False)
    assert r.node.base.driving and r.cmds[-1].linear.x > 0.0


# ---- the arm's clutch the other way, and the first holder (INV-2) ----

@pytest.mark.parametrize('attr, value, reason', [
    ('clutch', True, BaseState.REASON_CLUTCH_ENGAGED),
    ('stopping', True, BaseState.REASON_CLUTCH_ENGAGED),
    ('teleop', False, BaseState.REASON_TELEOP_UNKNOWN),
    ('odom', False, BaseState.REASON_ODOM_MISSING),
    ('reset', ResetState.STATE_MOVING, BaseState.REASON_RESET_ACTIVE),
])
def test_a_drive_is_refused_with_its_reason(rig, attr, value, reason):
    r = rig()
    setattr(r, attr, value)
    r.hold(0.6)
    r.step(grip=True, left=(0.0, -1.0))
    assert not r.cmds
    assert r.state().state == BaseState.STATE_REFUSED and r.state().reason == reason
    setattr(r, attr, not value if isinstance(value, bool) else ResetState.STATE_IDLE)
    r.hold(0.6, grip=True, left=(0.0, -1.0))
    assert not r.cmds, 'a held grip never starts by itself'


def test_both_grips_in_one_frame_nothing_moves(rig):
    r = rig()
    r.hold(0.4)
    r.hold(0.5, grip=True, right_grip=True, left=(0.0, -1.0))
    assert not r.cmds
    assert r.state().reason == BaseState.REASON_RIGHT_GRIP_HELD


def test_the_right_grip_during_a_drive_is_the_arms_to_refuse(rig):
    r = rig()
    r.drive()
    r.hold(0.5, grip=True, right_grip=True, left=(0.0, -1.0))
    assert r.node.base.driving and r.cmds[-1].linear.x > 0.0


def test_driving_with_teleop_disarmed(rig):
    """A fresh state with the clutch released is enough; teleop need not be armed."""
    r = rig()
    r.drive()
    assert r.node.base.driving


# ---- rest and the tool's speed in the room ----

def test_at_rest_is_reported_from_odometry(rig):
    r = rig()
    r.moving = True
    r.drive()
    assert not r.state().at_rest
    r.step(grip=False)
    r.moving = False
    r.hold(0.4)
    assert r.state().at_rest and r.state().last_event == BaseState.EVENT_AT_REST


def test_the_tool_speed_cap_scales_a_turn(rig):
    r = rig(description=ARM_DESCRIPTION, base_tool_speed_cap=0.1, tool_frame='tcp')
    r.joints = 0.0
    r.hold(0.4)
    r.hold(1.5, grip=True, right=(-1.0, 0.0))
    assert r.cmds[-1].angular.z == pytest.approx(0.1 / 0.7, rel=1e-6)
    assert r.state().active_limit == BaseState.LIMIT_TOOL_SPEED
    assert r.state().tool_speed_scale == pytest.approx(0.1 / 0.28, rel=1e-6)


def test_the_tool_speed_cap_needs_the_joints(rig):
    r = rig(description=ARM_DESCRIPTION, base_tool_speed_cap=0.1, tool_frame='tcp')
    r.hold(0.4)
    r.step(grip=True, right=(-1.0, 0.0))
    assert not r.cmds and r.state().reason == BaseState.REASON_TOOL_POSE_UNKNOWN


def test_without_a_tool_speed_cap_no_joint_states_are_read(rig):
    r = rig()
    assert '/joint_states' not in {s.topic_name for s in r.node.subscriptions}
    assert math.isnan(r.state().tool_speed_cap) if r.states else True


# ---- the real graph and the real process ----

def test_the_graph_query_takes_little_of_the_cycle(ros):
    """
    Time the 2 Hz graph look on a real node: it takes little of the 20 ms cycle.

    The stack test times it again with the whole stack in the graph.
    """
    node = BaseNode(node_name='base_graph_timing', parameter_overrides=[
        Parameter(k, value=v) for k, v in PARAMS.items()])
    try:
        times = []
        for _ in range(500):
            start = time.perf_counter()
            node.look_at_graph(0.0)
            times.append(time.perf_counter() - start)
        times.sort()
        p99, worst = times[int(0.99 * len(times))], times[-1]
        print(f'graph look: p50 {times[250] * 1e3:.3f} ms, p99 {p99 * 1e3:.3f} ms, '
              f'max {worst * 1e3:.3f} ms over 500')
        assert p99 < 0.002, 'the graph look takes more than a tenth of the 20 ms cycle'
    finally:
        node.destroy_node()
