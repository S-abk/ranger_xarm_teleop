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
Teleop node in process: INV-2, INV-10, INV-12, pairing by stamp, tool-to-tip conversion.

The node is fed real bridge messages built from the fixtures (same parser and message
builder as the live bridge). The robot side is replaced by an ideal arm: the measured
tool pose is the last target, MoveIt Servo becomes active one cycle after it is asked
to, and the mux grants ownership as soon as arming is requested. Time is a fake
monotonic clock, so the test is deterministic.
"""

import math
import os

from geometry_msgs.msg import TransformStamped
import pytest
from ranger_xarm_teleop_bridge.bridge_node import BridgeNode
from ranger_xarm_teleop_bridge.frame_parse import parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.jsonl_io import read_records
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from ranger_xarm_teleop_servo import teleop_node
from ranger_xarm_teleop_servo.mapping import tool_to_tip
from ranger_xarm_teleop_servo.rotations import Pose, q_between, v_norm, v_sub
from ranger_xarm_teleop_servo.teleop_node import (pose_of, stamp_key, StartupRefused,
                                                  TeleopNode)
import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
LIMITS = {
    'max_offset_translation_m': 0.30,
    'max_offset_rotation_deg': 60.0,
    'view_cone_half_angle_deg': 45.0,
    'step_translation_limit_m': 0.030,
    'step_rotation_limit_deg': 10.0,
    'linear_speed_cap_m_s': 0.25,
    'angular_speed_cap_rad_s': 1.0,
}
TOOL_START = Pose((0.30, 0.0, 0.40), (0.0, 0.0, 0.0, 1.0))
TIP_IN_TOOL = Pose((0.0, 0.0, -0.10), (0.0, 0.0, 0.0, 1.0))


def overrides(values):
    return [Parameter(name, value=value) for name, value in values.items()]


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    bridge = BridgeNode(node_name='bridge_for_teleop_test', source_label='test')
    yield bridge
    bridge.destroy_node()
    rclpy.shutdown()


class Harness:
    """A TeleopNode with an ideal arm, a fake clock and captured publishers."""

    def __init__(self, bridge, tool_frame='xarm_link_tcp', **params):
        values = {**LIMITS, 'motion_enabled': True, 'tool_frame': tool_frame, **params}
        self.bridge = bridge
        self.node = TeleopNode(node_name='teleop_under_test',
                               parameter_overrides=overrides(values))
        node = self.node
        self.now = 0.0
        node.mono = lambda: self.now
        self.targets, self.commands, self.states = [], [], []
        node.target_pub.publish = self.targets.append
        node.servo_pub.publish = self.commands.append
        node.state_pub.publish = self.states.append
        self.static = []
        node.tf_static_pub.publish = self.static.append
        self.tool = TOOL_START
        node.measured_tool_pose = lambda: (self.tool, 0.0)
        node.manage_servo = self._servo
        node.servo_verified = True
        node.manage_ownership = self._ownership
        node.owner = ArmOwner.OWNER_MOVEIT
        node.owner_arrival = 0.0
        if tool_frame != 'xarm_link_tcp':
            t = TransformStamped()
            t.header.frame_id = tool_frame
            t.child_frame_id = 'xarm_link_tcp'
            (t.transform.translation.x, t.transform.translation.y,
             t.transform.translation.z) = TIP_IN_TOOL.p
            t.transform.rotation.w = 1.0
            node._on_tf_static(TFMessage(transforms=[t]))
        else:
            node._on_tf_static(TFMessage(transforms=[]))

    def _servo(self, now, want_active):
        self.node.servo_state = (teleop_node.SERVO_ACTIVE if want_active
                                 else teleop_node.SERVO_PAUSED)

    def _ownership(self, now, owner):
        self.node.owner = (ArmOwner.OWNER_TELEOP if self.node.state.want_armed
                           else ArmOwner.OWNER_MOVEIT)
        self.node.owner_arrival = now

    def close(self):
        self.node.close()
        self.node.destroy_node()

    def feed(self, parsed, arrival, drop=()):
        """Deliver the messages of one frame at ``arrival`` (pose, head, joy, status)."""
        self.now = arrival
        stamp = rclpy.time.Time(nanoseconds=int(1e9 + parsed.seq * 1e7)).to_msg()
        messages = self.bridge.build_messages(parsed, stamp)
        if 'right' in messages.poses and 'pose' not in drop:
            self.node._on_pose(messages.poses['right'])
        if messages.head is not None and 'head' not in drop:
            self.node._on_head(messages.head)
        if 'joy' not in drop:
            self.node._on_joy(messages.joy)
        self.node._on_status(messages.status)

    def tick(self, now):
        self.now = now
        before = len(self.targets)
        self.node._on_timer()
        if len(self.targets) > before:
            self.tool = pose_of(self.targets[-1].pose)

    def play(self, name, drop=lambda parsed: ()):
        records = read_records(os.path.join(FIXTURES, name + '.jsonl'))
        frames = []
        for record in records:
            parsed = parse_text(record.text())
            if isinstance(parsed, ParsedFrame):
                frames.append((record.t_recv - records[0].t_recv, parsed))
        index, k = 0, 0
        end = frames[-1][0] + 0.3
        while k * 0.02 <= end:
            now = k * 0.02
            while index < len(frames) and frames[index][0] <= now:
                self.feed(frames[index][1], frames[index][0], drop(frames[index][1]))
                index += 1
            self.tick(now)
            k += 1


@pytest.fixture
def harness(ros):
    made = []

    def make(**kwargs):
        made.append(Harness(ros, **kwargs))
        return made[-1]
    yield make
    for h in made:
        h.close()


# ---- INV-10 ----

@pytest.mark.parametrize('missing', sorted(LIMITS))
def test_node_refuses_to_start_without_a_pending_value(ros, missing):
    values = {k: v for k, v in LIMITS.items() if k != missing}
    with pytest.raises(StartupRefused, match=missing):
        TeleopNode(node_name='teleop_missing_value', parameter_overrides=overrides(values))


def test_motion_enabled_defaults_to_false_and_then_nothing_is_published(ros):
    h = Harness(ros)
    h.close()
    node = TeleopNode(node_name='teleop_default_motion', parameter_overrides=overrides(LIMITS))
    try:
        assert node.params.motion_enabled is False
    finally:
        node.close()
        node.destroy_node()
    quiet = Harness(ros, motion_enabled=False)
    try:
        quiet.play('clutch_line')
        assert quiet.targets == [] and quiet.commands == []
        assert all(s.hold_reason == TeleopState.HOLD_MOTION_DISABLED for s in quiet.states)
        assert not any(s.motion_enabled or s.armed or s.clutch_engaged for s in quiet.states)
    finally:
        quiet.close()


def urdf(plugin):
    return String(data='<robot name="r"><ros2_control name="arm" type="system"><hardware>'
                       f'<plugin>{plugin}</plugin></hardware></ros2_control></robot>')


def test_simulation_only_values_need_a_simulator_description(harness):
    h = harness(limits_are_simulation_only=True)
    assert not h.node.robot_description_checked
    h.play('clutch_line')                       # no description yet: nothing may move
    assert h.targets == []
    h.node._on_robot_description(urdf('mock_components/GenericSystem'))
    assert h.node.robot_description_checked
    other = harness(limits_are_simulation_only=True)
    with pytest.raises(StartupRefused, match='not a simulator'):
        other.node._on_robot_description(urdf('example_vendor/RealArmHardware'))


def test_simulation_only_values_without_any_description_stop_the_node(harness):
    h = harness(limits_are_simulation_only=True, robot_description_timeout_s=1.0)
    h.node.started = 0.0                        # on the fake clock
    h.tick(0.5)
    with pytest.raises(StartupRefused, match='no robot description'):
        h.tick(1.6)


# ---- INV-2, INV-12 ----

def test_target_state_and_servo_command_match_the_contract(harness):
    h = harness()
    h.play('clutch_line')
    assert len(h.targets) > 100 and len(h.targets) == len(h.commands)
    state_stamps = {stamp_key(s.header.stamp): s for s in h.states}
    for target, command in zip(h.targets, h.commands):
        assert target.header.frame_id == 'xarm_link_base'
        # The pose sent to MoveIt Servo is the same target, same stamp, same frame.
        assert command.header == target.header
        assert command.pose == target.pose
        state = state_stamps[stamp_key(target.header.stamp)]
        assert state.clutch_engaged and state.armed and state.motion_enabled
        assert state.hold_reason == TeleopState.HOLD_NONE
        assert state.owner == ArmOwner.OWNER_TELEOP and state.calibrated
        assert state.measured_tool_pose_valid
    # No target outside the clutch: every state without a target says why.
    engaged = [s for s in h.states if s.clutch_engaged]
    assert len(engaged) == len(h.targets)
    for state in h.states:
        if not state.clutch_engaged:
            assert state.hold_reason != TeleopState.HOLD_NONE
    first, last = pose_of(h.targets[0].pose), pose_of(h.targets[-1].pose)
    assert v_norm(v_sub(first.p, TOOL_START.p)) < 1e-9
    assert v_sub(last.p, TOOL_START.p) == pytest.approx((0.20, 0.0, 0.0), abs=1e-4)
    assert h.states[-1].calibration_heading == pytest.approx(math.radians(-20.0), abs=1e-6)
    assert h.states[-1].input_age > 0.1                    # measured: the stream has ended
    assert h.states[-1].hold_reason == TeleopState.HOLD_INPUT_STALE


def test_state_is_published_every_cycle_and_calibration_goes_to_tf_static(harness):
    h = harness()
    h.play('clutch_line')
    assert len(h.states) == pytest.approx(4.9 / 0.02, abs=3)
    (message,) = h.static
    (transform,) = message.transforms
    assert transform.header.frame_id == 'xarm_link_base'
    assert transform.child_frame_id == 'quest_world'
    assert transform.transform.translation.x == 0.0
    yaw = 2.0 * math.atan2(transform.transform.rotation.z, transform.transform.rotation.w)
    assert yaw == pytest.approx(math.radians(-20.0), abs=1e-6)


def test_tool_frame_offset_from_the_tip_rotates_about_the_tool_point(harness):
    """INV-12 as relaxed: the Servo command is the target converted by the fixed transform."""
    h = harness(tool_frame='custom_tool', filter_cutoff_hz=0.0)
    h.play('clutch_rotation')
    assert len(h.targets) > 100
    turned = 0.0
    for target, command in zip(h.targets, h.commands):
        tool, tip = pose_of(target.pose), pose_of(command.pose)
        expected = tool_to_tip(tool, TIP_IN_TOOL)
        assert v_norm(v_sub(tip.p, expected.p)) < 1e-12
        assert q_between(tip.q, expected.q) < 1e-7
        assert command.header == target.header
        # The tool point does not move; the tip circles it at the offset distance.
        assert v_norm(v_sub(tool.p, TOOL_START.p)) < 1e-9
        assert v_norm(v_sub(tip.p, tool.p)) == pytest.approx(0.10, abs=1e-12)
        turned = max(turned, q_between(tool.q, TOOL_START.q))
    assert turned == pytest.approx(math.radians(40.0), abs=1e-3)


def test_tool_frame_without_a_static_transform_to_the_tip_never_moves(ros):
    h = Harness(ros)
    h.close()
    stuck = Harness.__new__(Harness)
    Harness.__init__(stuck, ros, tool_frame='xarm_link_tcp')
    try:
        stuck.node.tool_to_tip = None
        stuck.node.tool_frame = 'frame_on_a_moving_joint'
        stuck.play('clutch_line')
        assert stuck.targets == []
        assert all(s.hold_reason in (TeleopState.HOLD_NOT_ARMED, TeleopState.HOLD_NOT_OWNER,
                                     TeleopState.HOLD_TOOL_POSE_STALE,
                                     TeleopState.HOLD_CLUTCH_RELEASED,
                                     TeleopState.HOLD_INPUT_STALE) for s in stuck.states)
        assert any(s.hold_reason == TeleopState.HOLD_TOOL_POSE_STALE for s in stuck.states)
    finally:
        stuck.close()


# ---- pairing by stamp ----

def test_pose_missing_for_a_status_that_announces_it_counts_as_absent(harness):
    h = harness()
    h.play('clutch_line', drop=lambda parsed: ('pose',) if 180 <= parsed.seq < 200 else ())
    lost = [s for s in h.states if s.hold_reason == TeleopState.HOLD_POSE_ABSENT]
    assert lost
    assert len(h.targets) < 60                  # released at seq 180, about 2.0 s in
    after = [s for s in h.states if stamp_key(s.header.stamp) >
             stamp_key(lost[0].header.stamp)]
    assert not any(s.clutch_engaged for s in after)       # latched: grip never released


def test_joy_missing_means_the_grip_is_unknown_and_the_clutch_releases(harness):
    h = harness()
    h.play('clutch_line', drop=lambda parsed: ('joy',) if 180 <= parsed.seq < 200 else ())
    assert any(s.hold_reason == TeleopState.HOLD_POSE_ABSENT for s in h.states)
    assert len(h.targets) < 60


def test_gate_trigger_through_the_node(harness):
    h = harness()
    h.play('gate_emulated')
    reasons = [s.hold_reason for s in h.states]
    assert TeleopState.HOLD_POSE_EMULATED in reasons
    engaged = [s.clutch_engaged for s in h.states]
    # Engaged, released by the trigger, engaged again only after the re-grip, and
    # released when the recording ends.
    changes = sum(1 for a, b in zip(engaged, engaged[1:]) if a != b)
    assert changes == 4


# ---- MoveIt Servo verification ----

class _Value:

    def __init__(self, double_value=0.0, bool_value=False):
        self.double_value, self.bool_value = double_value, bool_value


class _Result:

    def __init__(self, *values):
        self.values = list(values)


@pytest.mark.parametrize('values, text', [
    ((_Value(0.4), _Value(1.0), _Value(bool_value=True)), 'differ'),
    ((_Value(0.25), _Value(0.8), _Value(bool_value=True)), 'differ'),
    ((_Value(0.25), _Value(1.0), _Value(bool_value=False)), 'collision checking is off'),
    ((), 'did not report'),
])
def test_servo_with_other_speed_caps_or_no_collision_check_is_refused(harness, values, text):
    h = harness()
    h.node._verified(_Result(*values))
    assert not h.node.servo_verified and text in h.node.servo_refusal


def test_servo_with_matching_caps_and_collision_checking_is_accepted(harness):
    h = harness()
    h.node._verified(_Result(_Value(0.25), _Value(1.0), _Value(bool_value=True)))
    assert h.node.servo_verified and not h.node.servo_refusal


# ---- reviewer findings: tool pose stamp, Servo re-verification, Servo halt ----

class _Buffer:
    """A TF buffer that returns one transform with a chosen stamp."""

    def __init__(self, stamp):
        self.stamp = stamp

    def lookup_transform(self, target, source, when):
        t = TransformStamped()
        t.header.stamp = self.stamp
        t.transform.translation.x = 0.3
        t.transform.rotation.w = 1.0
        return t


def test_tool_pose_age_uses_joint_state_arrival_and_the_transform_stamp(ros):
    node = TeleopNode(node_name='teleop_tool_age',
                      parameter_overrides=overrides({**LIMITS, 'motion_enabled': True}))
    try:
        node.tf_buffer = _Buffer(node.get_clock().now().to_msg())
        node.joint_state_arrival = None
        assert node.measured_tool_pose() == (None, math.inf)        # no joint state yet
        node.mono = lambda: 100.0
        node.joint_state_arrival = 100.0                            # a joint state just arrived
        pose, age = node.measured_tool_pose()
        assert pose.p == (0.3, 0.0, 0.0) and age < 0.05
        # Old by its stamp, however fresh the joint state.
        node.tf_buffer = _Buffer((node.get_clock().now()
                                  - rclpy.duration.Duration(seconds=0.5)).to_msg())
        pose, age = node.measured_tool_pose()
        assert age > 0.45 and age > node.params.tool_pose_timeout
        # Fresh by its stamp, but the joint states stopped half a second ago (real time).
        node.tf_buffer = _Buffer(node.get_clock().now().to_msg())
        node.joint_state_arrival = 99.5
        assert node.measured_tool_pose()[1] == pytest.approx(0.5)
        # No time in the transform at all (static chain): not a measurement.
        node.joint_state_arrival = 100.0
        node.tf_buffer = _Buffer(rclpy.time.Time().to_msg())
        assert node.measured_tool_pose() == (None, math.inf)
        # Stamped in the future (clock reset or skew): its age cannot be judged.
        node.tf_buffer = _Buffer((node.get_clock().now()
                                  + rclpy.duration.Duration(seconds=5.0)).to_msg())
        assert node.measured_tool_pose() == (None, math.inf)
    finally:
        node.close()
        node.destroy_node()


class _Midway(Harness):
    """A harness that changes something on the node at a given time."""

    def __init__(self, bridge, at, change, **params):
        super().__init__(bridge, **params)
        self.at, self.change = at, change

    def tick(self, now):
        if now >= self.at and self.change is not None:
            self.change(self.node)
            self.change = None
        super().tick(now)


def test_servo_parameters_changed_while_running_release_the_clutch(ros):
    """G-7: the caps are read again while running; a change ends the engagement."""
    def raise_cap(node):
        node._verified(_Result(_Value(0.25), _Value(1.0), _Value(bool_value=True)))
        assert node.servo_verified
        node._verified(_Result(_Value(1.0), _Value(1.0), _Value(bool_value=True)))
    h = _Midway(ros, 2.5, raise_cap)
    try:
        h.play('clutch_line')
        assert not h.node.servo_verified and 'differ' in h.node.servo_refusal
        assert 40 < len(h.targets) < 80                 # engaged at 1.2 s, released at 2.5 s
        after = [s for s in h.states if s.hold_reason == TeleopState.HOLD_SERVO_NOT_READY]
        assert after and not any(s.clutch_engaged for s in h.states[-60:])
    finally:
        h.close()


@pytest.mark.parametrize('code', [2, 5, 6])   # halt for singularity, for collision, joint bound
def test_servo_halt_releases_the_clutch_and_is_reported(ros, code):
    def halt(node):
        node.servo_status = code
    h = _Midway(ros, 2.5, halt)
    try:
        h.play('clutch_line')
        assert 40 < len(h.targets) < 80
        halted = [s for s in h.states if s.hold_reason == TeleopState.HOLD_SERVO_HALTED]
        assert halted and all(s.servo_status == code and not s.clutch_engaged for s in halted)
    finally:
        h.close()


@pytest.mark.parametrize('code', [0, 1, 3, 4])    # no warning, and the three "decelerate" codes
def test_servo_slowing_does_not_release_the_clutch(ros, code):
    def slow(node):
        node.servo_status = code
    h = _Midway(ros, 2.5, slow)
    try:
        h.play('clutch_line')
        assert len(h.targets) > 140
        assert not any(s.hold_reason == TeleopState.HOLD_SERVO_HALTED for s in h.states)
    finally:
        h.close()


class _Future:

    def __init__(self, result):
        self._result = result

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self._result


class _Client:
    """A service client that answers at once with the next prepared result."""

    def __init__(self, make_result):
        self.make_result = make_result
        self.requests = []
        self.ready = True

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        return _Future(self.make_result(request))


class _Ok:

    success = True
    message = ''


def test_servo_management_reads_the_parameters_again_every_second(ros):
    """The real ``manage_servo``: pause, verify, start, and re-verify while running (G-7)."""
    node = TeleopNode(node_name='teleop_servo_management',
                      parameter_overrides=overrides({**LIMITS, 'motion_enabled': True}))
    try:
        clock = {'t': 50.0}
        node.mono = lambda: clock['t']
        caps = {'linear': 0.25}
        node.pause_client = _Client(lambda request: _Ok())
        node.type_client = _Client(lambda request: _Ok())
        node.servo_params_client = _Client(lambda request: _Result(
            _Value(caps['linear']), _Value(1.0), _Value(bool_value=True)))

        def cycle(want, dt=0.02):
            clock['t'] += dt
            node.manage_servo(clock['t'], want)

        cycle(False)                                    # unknown -> pause
        assert node.servo_state == teleop_node.SERVO_PAUSED
        assert [r.data for r in node.pause_client.requests] == [True]
        cycle(False)                                    # first verification
        assert node.servo_verified and len(node.servo_params_client.requests) == 1
        node.servo_status = 5                           # a halt reported before this start
        cycle(True)                                     # pose mode, then unpause
        assert node.servo_state == teleop_node.SERVO_ACTIVE
        assert node.pause_client.requests[-1].data is False
        assert node.servo_status == -1                  # ... is history once it runs again
        for _ in range(40):                             # 0.8 s: no new read yet
            cycle(True)
        assert len(node.servo_params_client.requests) == 1
        for _ in range(15):                             # past one second: read again
            cycle(True)
        assert len(node.servo_params_client.requests) == 2 and node.servo_verified
        # The cap is raised while Servo runs: the next read catches it.
        caps['linear'] = 1.0
        for _ in range(55):
            cycle(True)
        assert len(node.servo_params_client.requests) == 3
        assert not node.servo_verified and 'differ' in node.servo_refusal
        # Put back: verified again on the next read, with no restart needed.
        caps['linear'] = 0.25
        for _ in range(55):
            cycle(True)
        assert node.servo_verified and node.servo_refusal == ''
        # The parameter service disappears: not verified.
        node.servo_params_client.ready = False
        for _ in range(55):
            cycle(True)
        assert not node.servo_verified
        # The gate is told so: under the clutch that is a release.
        node.servo_state = teleop_node.SERVO_ACTIVE
        assert not (node.servo_state == teleop_node.SERVO_ACTIVE and node.servo_verified)
        # Not wanted any more: paused, whatever the verification state.
        cycle(False)
        assert node.pause_client.requests[-1].data is True
        assert node.servo_state == teleop_node.SERVO_PAUSED
    finally:
        node.close()
        node.destroy_node()


def test_servo_that_stops_answering_is_unknown_and_unverified(ros):
    node = TeleopNode(node_name='teleop_servo_silent',
                      parameter_overrides=overrides({**LIMITS, 'motion_enabled': True}))
    try:
        node.mono = lambda: 10.0
        node.servo_state = teleop_node.SERVO_ACTIVE
        node.servo_verified = True
        node.servo_call_since = 8.0                     # a call sent two seconds ago
        node.pause_client = _Client(lambda request: _Ok())
        node.pause_client.ready = False
        node.manage_servo(10.0, True)
        assert node.servo_state == teleop_node.SERVO_UNKNOWN and not node.servo_verified
    finally:
        node.close()
        node.destroy_node()
