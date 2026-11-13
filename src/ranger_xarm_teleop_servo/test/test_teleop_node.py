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
Teleop node in process: INV-2, INV-10, INV-12, pairing by stamp, joint-jog command.

The node is fed real bridge messages built from the fixtures (same parser and message
builder as the live bridge). The robot side is replaced by an ideal arm: MoveIt Servo
integrates each joint-jog command, its output (every window point but the newest, so the
state before the latest step) reaches the node one cycle later, and the measured joints are
that commanded state; MoveIt Servo becomes active as soon as it is
asked to, and the mux grants ownership as soon as arming is requested. Time is a fake
monotonic clock, so the test is deterministic.
"""

import math
import os
import sys

from geometry_msgs.msg import TransformStamped
from moveit_msgs.msg import ServoStatus
from moveit_msgs.srv import ServoCommandType
import numpy as np
import pytest
from ranger_xarm_teleop_bridge.bridge_node import BridgeNode
from ranger_xarm_teleop_bridge.frame_parse import parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.jsonl_io import read_records
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from ranger_xarm_teleop_servo import solver as dls
from ranger_xarm_teleop_servo import teleop_node
from ranger_xarm_teleop_servo.rotations import (compose, inverse, Pose, pose_from_matrix,
                                                q_between, v_norm, v_sub)
from ranger_xarm_teleop_servo.teleop_node import (pose_of, stamp_key, StartupRefused,
                                                  TeleopNode)
import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unit'))
from arm_urdf import arm_urdf, READY  # noqa: E402, I100

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
LIMITS = {
    'max_offset_translation_m': 0.30,
    'max_offset_rotation_deg': 60.0,
    'view_cone_half_angle_deg': 45.0,
    'step_translation_limit_m': 0.030,
    'step_rotation_limit_deg': 10.0,
    'linear_speed_cap_m_s': 0.25,
    'angular_speed_cap_rad_s': 1.0,
    'max_lead_translation_m': 0.05,
    'max_lead_rotation_deg': 12.0,
}
TIP_IN_TOOL = Pose((0.0, 0.0, -0.10), (0.0, 0.0, 0.0, 1.0))
TCP_START = pose_from_matrix(dls.tool_pose(
    dls.make_arm(dls.chain_from_urdf(arm_urdf())), READY))
TOOL_START = TCP_START
JOINTS = [f'xarm_joint{i}' for i in range(1, 7)]


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
    """A TeleopNode with an ideal joint-level arm, a fake clock and captured publishers."""

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
        # The arm: Servo's commanded state, followed exactly by the measured joints.
        self.q = np.array(READY, dtype=float)
        self.output_on = True           # Servo publishes its output
        self.arm_follows = True         # the measured joints follow Servo's output
        self.measured_q = self.q.copy()
        self.pending_output = None
        self.tools = []                 # measured tool pose after every cycle
        node.urdf = arm_urdf()
        node.measured = self._measured
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

    @property
    def tool(self):
        """Return the measured tool pose (forward kinematics, tool offset included)."""
        _, pose, _ = self._measured()
        return pose

    def _measured(self):
        node = self.node
        if node.arm is None or node.tool_to_tip is None:
            return None, None, math.inf
        return (self.measured_q.copy(),
                pose_from_matrix(dls.tool_pose(node.arm, self.measured_q)), 0.0)

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
        if self.pending_output is not None and self.output_on:
            # Servo's output for the previous command arrives one cycle later.
            self.node.servo_output = (now, dict(zip(JOINTS, self.pending_output)))
        self.pending_output = None
        before = len(self.commands)
        self.node._on_timer()
        for command in self.commands[before:]:
            assert list(command.joint_names) == JOINTS
            # Servo publishes every window point but the newest: the state before this step.
            self.pending_output = self.q.copy()
            self.q = self.q + np.array(command.velocities) * 0.02
        if self.arm_follows:
            self.measured_q = self.q.copy()
        if self.node.arm is not None and self.node.tool_to_tip is not None:
            self.tools.append(self.tool)

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

def test_target_state_and_joint_command_match_the_contract(harness):
    """Replaces the pose-mode test: one joint-jog command per target, same stamp."""
    h = harness()
    h.play('clutch_line')
    assert len(h.targets) > 100
    commands = {stamp_key(c.header.stamp): c for c in h.commands}
    # One command per target, plus one zero command when the clutch releases.
    assert len(h.commands) == len(h.targets) + 1
    assert not any(h.commands[-1].velocities)
    state_stamps = {stamp_key(s.header.stamp): s for s in h.states}
    for target in h.targets:
        assert target.header.frame_id == 'xarm_link_base'
        command = commands[stamp_key(target.header.stamp)]
        assert command.header == target.header
        assert list(command.joint_names) == JOINTS and len(command.velocities) == 6
        assert all(abs(v) <= 3.14 * 0.95 + 1e-9 for v in command.velocities)
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
    # The arm followed: the measured tool ends at the last target.
    assert v_norm(v_sub(h.tool.p, last.p)) < 2e-3
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
    """
    INV-12 as relaxed: a pure rotation of the target turns the arm about the tool point.

    Replaces the pose-mode test (the tip pose sent to Servo): the solver works at the tool
    point, so the measured tool point stays put while the tip circles it.
    """
    h = harness(tool_frame='custom_tool', filter_cutoff_hz=0.0)
    start = compose(TCP_START, inverse(TIP_IN_TOOL))
    assert v_norm(v_sub(h.tool.p, start.p)) < 1e-9
    h.play('clutch_rotation')
    assert len(h.targets) > 100
    turned, drift = 0.0, 0.0
    for target in h.targets:
        tool = pose_of(target.pose)
        assert v_norm(v_sub(tool.p, start.p)) < 1e-9
        turned = max(turned, q_between(tool.q, start.q))
    assert turned == pytest.approx(math.radians(40.0), abs=1e-3)
    drift = max(v_norm(v_sub(t.p, start.p)) for t in h.tools)
    assert drift < 1e-3
    measured_turn = max(q_between(t.q, start.q) for t in h.tools)
    assert measured_turn == pytest.approx(math.radians(40.0), abs=0.01)
    tips = [v_norm(v_sub(compose(t, TIP_IN_TOOL).p, start.p)) for t in h.tools]
    assert max(tips) > 0.05                 # the tip did move, around the tool point


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

    def __init__(self, double_value=0.0, bool_value=False, string_value='',
                 double_array_value=()):
        self.double_value, self.bool_value = double_value, bool_value
        self.string_value, self.double_array_value = string_value, list(double_array_value)


class _Result:

    def __init__(self, *values):
        self.values = list(values)


def servo_parameters(collisions=True, units='speed_units', margins=(0.1,), timeout=0.06):
    """Return what MoveIt Servo reports, in the order the node asks for it."""
    return _Result(_Value(bool_value=collisions), _Value(string_value=units),
                   _Value(double_array_value=margins), _Value(double_value=timeout))


@pytest.mark.parametrize('values, text', [
    ({'collisions': False}, 'collision checking is off'),
    ({'units': 'unitless'}, 'not speed_units'),
    ({'margins': (0.15,)}, 'narrower'),
    ({'margins': (0.1, 0.1, 0.2, 0.1, 0.1, 0.1)}, 'narrower'),
    ({'margins': ()}, 'narrower'),
    ({'timeout': 0.5}, 'input timeout'),
    ({'timeout': 0.0}, 'input timeout'),
])
def test_servo_with_wrong_parameters_is_refused(harness, values, text):
    """D5: replaces the speed-cap comparison of pose mode (the caps are the node's now)."""
    h = harness()
    h.node._verified(servo_parameters(**values))
    assert not h.node.servo_verified and text in h.node.servo_refusal


def test_servo_that_reports_nothing_is_refused(harness):
    h = harness()
    h.node._verified(_Result())
    assert not h.node.servo_verified and 'did not report' in h.node.servo_refusal


def test_servo_with_the_right_parameters_is_accepted(harness):
    h = harness()
    h.node._verified(servo_parameters())
    assert h.node.servo_verified and not h.node.servo_refusal


# ---- reviewer findings: tool pose stamp, Servo re-verification, Servo halt ----

def test_measured_tool_pose_is_forward_kinematics_of_the_joint_state_and_its_age(ros):
    """A1: the measured pose comes from /joint_states, aged by arrival and by stamp."""
    node = TeleopNode(node_name='teleop_tool_age',
                      parameter_overrides=overrides({**LIMITS, 'motion_enabled': True}))
    try:
        node._on_tf_static(TFMessage(transforms=[]))
        assert node.measured() == (None, None, math.inf)            # no description yet
        node._on_robot_description(String(data=arm_urdf()))
        assert node.arm is not None and node.arm.joint_names == JOINTS
        assert node.measured() == (None, None, math.inf)            # no joint state yet
        node.mono = lambda: 100.0
        positions = dict(zip(JOINTS, READY))

        def state(stamp, arrival=100.0, values=positions):
            node.joint_state = (arrival, stamp.nanoseconds, dict(values))

        now = node.get_clock().now()
        state(now)
        q, pose, age = node.measured()
        assert list(q) == READY and age < 0.05
        assert v_norm(v_sub(pose.p, TCP_START.p)) < 1e-12
        # Old by its stamp, however recently it arrived.
        state(now - rclpy.duration.Duration(seconds=0.5))
        assert node.measured()[2] > 0.45
        # Fresh by its stamp, but the joint states stopped half a second ago (real time).
        state(node.get_clock().now(), arrival=99.5)
        assert node.measured()[2] == pytest.approx(0.5, abs=0.01)
        # No time at all, or stamped in the future: not a measurement.
        state(rclpy.time.Time())
        assert node.measured() == (None, None, math.inf)
        state(node.get_clock().now() + rclpy.duration.Duration(seconds=5.0))
        assert node.measured() == (None, None, math.inf)
        # A joint missing from the joint state: not a measurement.
        state(node.get_clock().now(), values={k: v for k, v in positions.items()
                                              if k != 'xarm_joint3'})
        assert node.measured() == (None, None, math.inf)
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
    """G-7: Servo's parameters are read again while running; a change ends the engagement."""
    def collisions_off(node):
        node._verified(servo_parameters())
        assert node.servo_verified
        node._verified(servo_parameters(collisions=False))
    h = _Midway(ros, 2.5, collisions_off)
    try:
        h.play('clutch_line')
        assert not h.node.servo_verified and 'collision' in h.node.servo_refusal
        assert 40 < len(h.targets) < 80                 # engaged at 1.2 s, released at 2.5 s
        after = [s for s in h.states if s.hold_reason == TeleopState.HOLD_SERVO_NOT_READY]
        assert after and not any(s.clutch_engaged for s in h.states[-60:])
    finally:
        h.close()


def status(node, code):
    """Deliver a MoveIt Servo status message now."""
    node._on_servo_status(ServoStatus(code=code))


@pytest.mark.parametrize('code', [5, 6])   # halt for collision, joint bound
def test_servo_halt_releases_the_clutch_and_is_reported(ros, code):
    def halt(node):
        status(node, code)
    h = _Midway(ros, 2.5, halt)
    try:
        h.play('clutch_line')
        assert 40 < len(h.targets) < 80
        halted = [s for s in h.states if s.hold_reason == TeleopState.HOLD_SERVO_HALTED]
        assert halted and all(s.servo_status == code and not s.clutch_engaged for s in halted)
    finally:
        h.close()


@pytest.mark.parametrize('code', [1, 2, 3])  # singularity codes: never caused by joint jog
def test_servo_singularity_status_means_wrong_mode_and_releases(ros, code):
    """D5: Servo's singularity handling never runs on joint jog; seeing it, refuse."""
    def singular(node):
        status(node, code)
    h = _Midway(ros, 2.5, singular)
    try:
        h.play('clutch_line')
        assert 40 < len(h.targets) < 80
        assert h.node.servo_mode_wrong
        assert any(s.hold_reason == TeleopState.HOLD_SERVO_NOT_READY for s in h.states)
    finally:
        h.close()


@pytest.mark.parametrize('code', [0, 4])    # no warning, decelerate for collision
def test_servo_slowing_does_not_release_the_clutch(ros, code):
    def slow(node):
        status(node, code)
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
        caps = {'collisions': True}
        node.pause_client = _Client(lambda request: _Ok())
        node.type_client = _Client(lambda request: _Ok())
        node.servo_params_client = _Client(lambda request: servo_parameters(
            collisions=caps['collisions']))

        def cycle(want, dt=0.02):
            clock['t'] += dt
            node.manage_servo(clock['t'], want)

        cycle(False)                                    # unknown -> pause
        assert node.servo_state == teleop_node.SERVO_PAUSED
        assert [r.data for r in node.pause_client.requests] == [True]
        cycle(False)                                    # first verification
        assert node.servo_verified and len(node.servo_params_client.requests) == 1
        node.servo_status = 5                           # a halt reported before this start
        cycle(True)                                     # joint-jog mode, then unpause
        assert node.servo_state == teleop_node.SERVO_ACTIVE
        (mode,) = node.type_client.requests
        assert mode.command_type == ServoCommandType.Request.JOINT_JOG
        assert list(node.servo_params_client.requests[0].names) == list(
            teleop_node.SERVO_CHECKED)
        assert node.pause_client.requests[-1].data is False
        assert node.servo_status == -1                  # ... is history once it runs again
        for _ in range(40):                             # 0.8 s: no new read yet
            cycle(True)
        assert len(node.servo_params_client.requests) == 1
        for _ in range(15):                             # past one second: read again
            cycle(True)
        assert len(node.servo_params_client.requests) == 2 and node.servo_verified
        # Collision checking is switched off while Servo runs: the next read catches it.
        caps['collisions'] = False
        for _ in range(55):
            cycle(True)
        assert len(node.servo_params_client.requests) == 3
        assert not node.servo_verified and 'collision' in node.servo_refusal
        # Put back: verified again on the next read, with no restart needed.
        caps['collisions'] = True
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


# ---- joint-jog driving path (dls) ----

def _engaged_span(h):
    return [s for s in h.states if s.clutch_engaged]


def test_release_sends_a_zero_command_in_the_same_cycle(harness):
    """INV-8: the cycle that releases the clutch sends velocity zero, same stamp as its state."""
    h = harness()
    h.play('clutch_reengage')
    commands = {stamp_key(c.header.stamp): c for c in h.commands}
    releases = [b for a, b in zip(h.states, h.states[1:]) if a.clutch_engaged
                and not b.clutch_engaged]
    assert len(releases) >= 2
    for state in releases:
        command = commands[stamp_key(state.header.stamp)]
        assert list(command.velocities) == [0.0] * 6
    # Nothing is sent while the clutch stays released.
    held = [s for s in h.states if not s.clutch_engaged and s not in releases]
    assert not any(stamp_key(s.header.stamp) in commands for s in held)


@pytest.mark.parametrize('fixture, hold', [('gate_emulated', TeleopState.HOLD_POSE_EMULATED),
                                           ('gate_out_of_view', TeleopState.HOLD_OUT_OF_VIEW),
                                           ('clutch_stall', TeleopState.HOLD_INPUT_STALE)])
def test_every_g4_trigger_and_stale_input_stop_the_joint_command_at_once(harness, fixture, hold):
    """INV-8 for G-3 and G-4 triggers."""
    h = harness()
    h.play(fixture)
    commands = {stamp_key(c.header.stamp): c for c in h.commands}
    first = next(s for s in h.states if s.hold_reason == hold)
    assert list(commands[stamp_key(first.header.stamp)].velocities) == [0.0] * 6


def test_state_reports_the_solver(harness):
    h = harness()
    h.play('clutch_line')
    engaged = _engaged_span(h)
    assert engaged
    for s in engaged:
        assert s.conditioning > 0.5 and not s.damped and s.damping == 0.0
        assert s.solver_error_translation < 0.01
        assert s.lead_translation <= 0.05 + 1e-9
        assert s.commanded_linear_speed <= 0.25 + 1e-9
        assert s.commanded_angular_speed <= 1.0 + 1e-9
    assert max(s.commanded_linear_speed for s in engaged) > 0.05
    released = [s for s in h.states if not s.clutch_engaged]
    assert all(s.conditioning == 0.0 and s.commanded_linear_speed == 0.0 for s in released)


def test_servo_output_going_quiet_releases_within_the_timeout(ros):
    """D2/D5: the reference is Servo's output; when it stops, the clutch releases."""
    def quiet(node):
        h.output_on = False
    h = _Midway(ros, 2.5, quiet)
    try:
        h.play('clutch_line')
        # States are published every cycle from t = 0: the index is the time.
        first = next(i for i, s in enumerate(h.states)
                     if i * 0.02 > 2.5 and s.hold_reason == TeleopState.HOLD_SERVO_NOT_READY)
        assert h.states[first - 1].clutch_engaged
        assert not any(s.clutch_engaged for s in h.states[first:])
        # 2.5 s plus at most the timeout (and one cycle to see it).
        assert 2.5 < first * 0.02 <= 2.5 + teleop_node.SERVO_OUTPUT_TIMEOUT_S + 0.04
    finally:
        h.close()


def test_servo_that_never_follows_the_first_command_releases_after_the_grace(ros):
    """A Servo that ignores the commands (wrong mode, halted for collision) publishes nothing."""
    h = Harness(ros)
    try:
        h.output_on = False
        h.play('clutch_line')
        assert 0 < len(h.targets) <= 7          # the grace: 0.1 s at 50 Hz, then released
        assert any(s.hold_reason == TeleopState.HOLD_SERVO_NOT_READY for s in h.states)
    finally:
        h.close()


def test_a_halt_status_left_from_before_the_engagement_does_not_release(ros):
    """D5: the first status after unpausing can be the old one; it counts once Servo follows."""
    class Stale(Harness):

        def tick(self, now):
            if not self.node.state.engaged:
                status(self.node, 5)                    # an old collision halt, still shown
            super().tick(now)
    h = Stale(ros)
    try:
        h.play('clutch_line')
        assert len(h.targets) > 140
        assert not any(s.hold_reason == TeleopState.HOLD_SERVO_HALTED for s in h.states)
        # The same code, arriving after Servo followed this engagement, does release.
        later = _Midway(ros, 2.5, lambda node: status(node, 5))
        try:
            later.play('clutch_line')
            assert any(s.hold_reason == TeleopState.HOLD_SERVO_HALTED for s in later.states)
        finally:
            later.close()
    finally:
        h.close()


def test_arm_that_stops_moving_holds_the_target_within_the_lead(ros):
    """G-14 through the node: the arm stands still, the target stays within the lead."""
    def stuck(node):
        h.arm_follows = False
    h = _Midway(ros, 1.6, stuck)
    try:
        h.play('clutch_line')
        engaged = _engaged_span(h)
        assert engaged and all(s.lead_translation <= 0.05 + 1e-9 for s in engaged)
        assert any(s.active_limit == TeleopState.LIMIT_LEAD for s in engaged)
        stuck_at = h.tool
        for target in h.targets[-20:]:
            assert v_norm(v_sub(pose_of(target.pose).p, stuck_at.p)) <= 0.05 + 1e-9
    finally:
        h.close()


def test_measured_arm_far_from_the_commanded_state_stops_the_solver(ros):
    """Commanded and measured tool pose apart by more than two lead limits: HOLD_SOLVER."""
    def pushed(node):
        h.arm_follows = False
        h.measured_q = h.measured_q + np.array([0.4, 0, 0, 0, 0, 0])
    h = _Midway(ros, 1.6, pushed)
    try:
        h.play('clutch_line')
        first = next(i for i, s in enumerate(h.states)
                     if s.hold_reason == TeleopState.HOLD_SOLVER)
        assert first * 0.02 <= 1.6 + 0.04
        assert not any(s.clutch_engaged for s in h.states[first:])
    finally:
        h.close()
