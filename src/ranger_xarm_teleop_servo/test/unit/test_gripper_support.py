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
"""Gripper task: the gate's per-frame fault check, the relay's logic, the model's finger pairs."""

from dataclasses import replace
import math

import pytest
from ranger_xarm_teleop_servo import gripper_relay as relay
from ranger_xarm_teleop_servo import keepout as ko
from ranger_xarm_teleop_servo.gate import Env, Frame, frame_fault, GateState, Hold, tick
from ranger_xarm_teleop_servo.gripper import joint_limits
from ranger_xarm_teleop_servo.rotations import Pose

from helpers import HAND, HEAD, params  # noqa: I100
from keepout_robot import resolve, robot_urdf  # noqa: I100

PARAMS = params()

CONTROLLER = HAND


def frame(**kw):
    values = {'arrival': 0.0, 'xr_time': 0.0, 'session_id': 1, 'seq': 10, 'reset_count': 0,
              'controller': CONTROLLER, 'controller_present': True,
              'controller_emulated': False, 'head': HEAD, 'head_present': True,
              'head_emulated': False, 'grip': False, 'calibrate': False, 'arm': False,
              'scale_toggle': False, 'trigger': 0.0}
    values.update(kw)
    return Frame(**values)


@pytest.mark.parametrize('change, expected', [
    ({}, Hold.NONE),
    ({'session_id': 2}, Hold.SESSION_CHANGED),
    ({'seq': 5}, Hold.SEQ_BACKWARDS),
    ({'seq': 10}, Hold.SEQ_BACKWARDS),
    ({'xr_time': 0.0}, Hold.SEQ_BACKWARDS),
    ({'session_id': 2, 'reset_count': 1}, Hold.SESSION_CHANGED),
    ({'reset_count': 1, 'seq': 5}, Hold.SEQ_BACKWARDS),
])
def test_frame_fault_sees_session_and_sequence_faults_whatever_the_clutch(change, expected):
    prev = frame(arrival=0.0, xr_time=0.0)
    now = replace(frame(arrival=0.011, xr_time=0.011, seq=11), **change)
    assert frame_fault(prev, now) == expected


@pytest.mark.parametrize('change', [
    {'reset_count': 1},
    {'controller_present': False},
    {'controller': None},
    {'controller_emulated': True},
    {'head_present': False},
    {'head': None},
    {'head_emulated': True},
    {'controller': Pose((HAND.p[0] + 0.3, HAND.p[1], HAND.p[2]), HAND.q)},    # raw jump
    {'controller': Pose((HAND.p[0], HAND.p[1] - 1.0, HAND.p[2] + 2.0), HAND.q)},   # out of view
])
def test_frame_fault_ignores_pose_only_conditions(change):
    prev = frame(arrival=0.0, xr_time=0.0)
    now = replace(frame(arrival=0.011, xr_time=0.011, seq=11), **change)
    assert frame_fault(prev, now) == Hold.NONE


def test_frame_fault_of_the_first_frame_is_none():
    assert frame_fault(None, frame()) == Hold.NONE
    assert frame_fault(None, frame(controller=None, controller_present=False)) == Hold.NONE


def test_gate_output_reports_the_condition_with_the_grip_set_aside():
    state = GateState(frame=frame(), buttons=(False, False, False, False))
    env = Env(now=0.0, owner_is_teleop=True, tool_pose=CONTROLLER, tool_pose_age=0.0,
              servo_active=True)
    _, out = tick(state, env, PARAMS)
    assert out.target is None
    assert out.condition == Hold.NOT_ARMED
    armed = replace(state, want_armed=True)
    _, out = tick(armed, replace(env, now=0.5), PARAMS)
    assert out.condition == Hold.INPUT_STALE


def test_relay_command_point():
    assert relay.command_point('j', 'j', 0.4, 0.02) == (0.4, (0, 20_000_000))
    assert relay.command_point('j', 'k', 0.4, 0.02) is None
    assert relay.command_point('j', 'j', math.nan, 0.02) is None


def test_relay_held_reference():
    assert relay.held_reference('j', ['a', 'j'], [0.1, 0.7]) == 0.7
    assert relay.held_reference('j', ['a'], [0.1]) is None
    assert relay.held_reference('j', ['j'], []) is None
    assert relay.held_reference('j', ['j'], [math.nan]) is None


def test_relay_modes():
    relay.check_mode('trajectory')
    with pytest.raises(ValueError, match='not built'):
        relay.check_mode('action')
    with pytest.raises(ValueError, match='unknown'):
        relay.check_mode('x')


def test_joint_limits_from_the_description():
    assert joint_limits(robot_urdf(), 'finger_left_joint') == (0.0, 0.04)
    with pytest.raises(ValueError, match='no joint'):
        joint_limits(robot_urdf(), 'nope')
    with pytest.raises(ValueError, match='not a revolute'):
        joint_limits(robot_urdf(), 'l6_to_gripper')
    with pytest.raises(ValueError, match='not valid XML'):
        joint_limits('<robot', 'x')
    # A ros2_control block naming the same joint (without a type) is not the joint.
    urdf = robot_urdf().replace(
        '</robot>', '<ros2_control name="g" type="system"><joint name="finger_left_joint">'
        '<command_interface name="position"/></joint></ros2_control></robot>')
    assert urdf.index('<ros2_control') < urdf.rindex('finger_left_joint')
    assert joint_limits(urdf, 'finger_left_joint') == (0.0, 0.04)
    urdf = robot_urdf().replace('<robot name="test_robot">', '<robot name="test_robot">'
                                '<ros2_control name="g" type="system"><joint '
                                'name="finger_left_joint"/></ros2_control>')
    assert joint_limits(urdf, 'finger_left_joint') == (0.0, 0.04)


def test_model_pairs_moved_by_the_gripper_joint_are_the_fingers():
    model = ko.build(robot_urdf(), resolve, 'xarm_link_base', 'xarm_link_tcp', 0.025)
    indices = model.pairs_moved_by('finger_left_joint')
    assert indices
    assert {model.pairs[i][0] for i in indices} == {'finger_left', 'finger_right'}
    assert model.pairs_moved_by('no_joint') == []


def test_joy_row_records_a_missing_trigger_as_null():
    import json
    from types import SimpleNamespace as NS
    from ranger_xarm_teleop_servo import session
    row = session.joy_row(NS(header=NS(stamp=NS(sec=1, nanosec=0)), buttons=[0, 1],
                             axes=[0.5, math.nan, math.inf]))
    assert row['axes'] == [0.5, None, None]
    json.dumps(row, allow_nan=False)


def test_session_rows_record_the_gripper_command_and_the_measured_opening_apart():
    from types import SimpleNamespace as NS
    from ranger_xarm_teleop_servo import session
    header = NS(stamp=NS(sec=5, nanosec=7), frame_id='base')
    assert session.gripper_row(NS(header=header, joint_name='j', position=0.4, request=2,
                                  reference_source=1)) == {
        'k': 'gripper', 'stamp': 5_000_000_007, 'joint': 'j', 'position': 0.4, 'request': 2,
        'source': 1}
    pose = NS(position=NS(x=0.0, y=0.0, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))
    state = NS(header=header, motion_enabled=True, armed=True, clutch_engaged=False, owner=2,
               mux_status=0, hold_reason=5, active_limit=0, servo_status=0, calibrated=True,
               input_age=0.01, input_seq=1, measured_tool_pose_valid=False,
               measured_tool_pose=pose, conditioning=0.0, damped=False, damping=0.0,
               solver_error_translation=0.0, solver_error_rotation=0.0, lead_translation=0.0,
               lead_rotation=0.0, commanded_linear_speed=0.0, commanded_angular_speed=0.0,
               translation_scale=1.0, gripper_enabled=True, gripper_active=False,
               gripper_hold=4, gripper_command=0.3, gripper_request=2,
               gripper_reference_source=2, gripper_measured_valid=True, gripper_measured=0.25,
               gripper_measured_stamp=NS(sec=5, nanosec=500_000_000))
    row = session.state_row(state)
    assert row['gripper_command'] == 0.3 and row['gripper_measured'] == 0.25
    assert row['gripper_measured_stamp'] == 5.5 and row['gripper_hold'] == 4
    assert not row['gripper_active'] and row['gripper_source'] == 2
    state.gripper_command = math.nan
    state.gripper_measured_valid = False
    row = session.state_row(state)
    assert row['gripper_command'] is None and row['gripper_measured'] is None
