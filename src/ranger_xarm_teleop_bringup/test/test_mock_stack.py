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
The teleop stack in mock execution: command mux (INV-7), end-to-end replay, collision stop.

One mock stack (``teleop_mock.launch.py``: ros2_control with mock components, move_group,
MoveIt Servo, the mux, the teleop node) is started for the module on its own ROS domain;
the tests run against it in file order. No simulator and no hardware is involved.
"""

import math
import os
import time

from moveit_msgs.msg import ServoStatus
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState

from stack_helpers import (child_environment, Probe, Process, READY, replay,
                           replay_with_clean_input)

MOVEIT, TELEOP = 'xarm6_traj_controller', 'teleop_arm_controller'
ELSEWHERE = [0.3, -0.4, -0.9, 0.2, 1.1, -0.3]


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_stack')
    env = child_environment(offset=1)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py'],
                     env, str(logs / 'launch.log'))
    probe = Probe(env)
    try:
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: probe.controllers().get(TELEOP) == 'inactive', 60.0,
                       'the teleop controller to be loaded inactive')
        probe.move_to(READY)
        yield probe, env, logs, launch
    finally:
        probe.close()
        launch.stop()
        print(launch.output()[-6000:])


def run_replay(probe, env, logs, fixture, seconds):
    """Replay a fixture from a disarmed stack; return the states seen during it."""
    probe.disarm()
    start = time.monotonic()
    process = replay(env, fixture, str(logs / f'replay_{fixture}.log'))
    try:
        assert process.wait(seconds + 30.0) == 0, process.output()
    finally:
        process.stop()
    time.sleep(1.0)
    return probe.states_since(start)


def test_stack_starts_with_moveit_as_owner_and_teleop_idle(stack):
    probe, _, _, _ = stack
    controllers = probe.controllers()
    assert controllers[MOVEIT] == 'active' and controllers[TELEOP] == 'inactive'
    state = probe.state()
    assert state.motion_enabled and not state.armed and not state.clutch_engaged
    assert state.owner == ArmOwner.OWNER_MOVEIT
    assert state.hold_reason == TeleopState.HOLD_NOT_ARMED


def test_servo_output_has_no_effect_while_moveit_owns_the_arm(stack):
    probe, _, _, _ = stack
    before = probe.joint_positions()
    start = time.monotonic()
    for _ in range(25):                                  # 0.5 s of commands, as Servo sends
        probe.publish_trajectory([q + 0.3 for q in before], seconds=0.1)
        time.sleep(0.02)
    time.sleep(0.5)
    assert probe.max_joint_change(start, before) < 1e-6
    assert probe.controllers()[TELEOP] == 'inactive'
    assert probe.owner_value() == ArmOwner.OWNER_MOVEIT


def test_switch_is_refused_while_a_trajectory_executes_and_ownership_stays(stack):
    probe, _, _, _ = stack
    handle = probe.send_goal(ELSEWHERE, seconds=4.0)
    assert handle.accepted
    time.sleep(1.0)
    asked = time.monotonic()
    response = probe.set_owner(ArmOwner.OWNER_TELEOP)
    assert time.monotonic() - asked < 2.5        # a goal in progress is refused, not waited out
    assert not response.success
    assert response.status in (ArmOwner.STATUS_REFUSED_GOAL_ACTIVE,
                               ArmOwner.STATUS_REFUSED_MOVING)
    assert response.owner == ArmOwner.OWNER_MOVEIT
    controllers = probe.controllers()
    assert controllers[MOVEIT] == 'active' and controllers[TELEOP] == 'inactive'
    result = handle.get_result_async()
    probe.wait_for(result.done, 20.0, 'the goal result')
    assert result.result().status == 4                   # the trajectory was not disturbed
    reached = probe.joint_positions()
    assert max(abs(a - b) for a, b in zip(reached, ELSEWHERE)) < 0.02


def test_arming_during_a_trajectory_is_refused_and_teleop_state_reports_it(stack):
    probe, env, logs, _ = stack
    probe.disarm()
    handle = probe.send_goal(READY, seconds=6.0)
    assert handle.accepted
    start = time.monotonic()
    process = replay(env, 'clutch_line', str(logs / 'replay_refused.log'), start_delay=1.0)
    try:
        assert process.wait(40.0) == 0
    finally:
        process.stop()
    states = probe.states_since(start)
    refused = [s for s in states if s.mux_status in (ArmOwner.STATUS_REFUSED_GOAL_ACTIVE,
                                                     ArmOwner.STATUS_REFUSED_MOVING)]
    assert refused, 'TeleopState never reported the refused ownership request'
    assert not any(s.armed or s.clutch_engaged for s in states)
    assert all(s.owner == ArmOwner.OWNER_MOVEIT for s in states)
    result = handle.get_result_async()
    probe.wait_for(result.done, 20.0, 'the goal result')
    assert result.result().status == 4
    time.sleep(0.6)


def test_switch_to_teleop_starts_from_the_measured_state_without_a_jump(stack):
    probe, _, _, _ = stack
    probe.move_to(ELSEWHERE)
    before = probe.joint_positions()
    start = time.monotonic()
    response = probe.set_owner(ArmOwner.OWNER_TELEOP)
    # The disarmed teleop node hands the arm back at once; ask the mux directly instead.
    assert response.status == ArmOwner.STATUS_OK, response.detail
    time.sleep(1.0)
    assert probe.max_joint_change(start, before) < 1e-4
    assert len(probe.joints_since(start)) > 50


def test_failed_switch_leaves_the_owner_where_it_was(stack):
    """A mux pointed at a controller that does not exist: the strict switch fails."""
    probe, env, logs, _ = stack
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 10.0, 'MoveIt owner')
    bad = Process(
        ['ros2', 'run', 'ranger_xarm_teleop_servo', 'arm_mux_node', '--ros-args',
         '-r', '__node:=bad_mux', '-r', '/teleop/mux/set_owner:=/bad_mux/set_owner',
         '-r', '/teleop/mux/owner:=/bad_mux/owner', '-p', 'teleop_controller:=missing'],
        env, str(logs / 'bad_mux.log'))
    try:
        from ranger_xarm_teleop_interfaces.srv import SetArmOwner
        client = probe.node.create_client(SetArmOwner, '/bad_mux/set_owner')
        assert client.wait_for_service(timeout_sec=20.0)
        time.sleep(1.5)             # it needs its own 0.3 s of joint states first
        before = probe.joint_positions()
        start = time.monotonic()
        response = probe.call(client, SetArmOwner.Request(owner=ArmOwner.OWNER_TELEOP), 30.0)
        assert not response.success
        assert response.status == ArmOwner.STATUS_SWITCH_FAILED
        assert response.owner == ArmOwner.OWNER_MOVEIT
        assert 'switch' in response.detail
        controllers = probe.controllers()
        assert controllers[MOVEIT] == 'active' and controllers[TELEOP] == 'inactive'
        assert probe.max_joint_change(start, before) < 1e-6
    finally:
        bad.stop()


def test_line_replay_moves_the_tip_along_the_commanded_path(stack):
    probe, env, logs, _ = stack
    probe.disarm()
    tool = {}

    def ready():
        probe.move_to(READY)
        tool['start'] = probe.tool_position()
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_line.log'), 'clutch_line',
                                       expected_engaged_s=2.8, before=ready)
    start_tool = tool['start']
    states = probe.states_since(start)
    engaged = [s for s in states if s.clutch_engaged]
    assert len(engaged) > 120, 'the clutch did not engage'
    assert all(s.owner == ArmOwner.OWNER_TELEOP and s.armed for s in engaged)
    end_tool = probe.tool_position()
    moved = [b - a for a, b in zip(start_tool, end_tool)]
    assert moved[0] == pytest.approx(0.20, abs=0.002)
    assert abs(moved[1]) < 0.002 and abs(moved[2]) < 0.002
    # While engaged the tip stays on the commanded straight line.
    for s in engaged:
        p = s.measured_tool_pose.position
        assert abs(p.y - start_tool[1]) < 0.002 and abs(p.z - start_tool[2]) < 0.002
        assert start_tool[0] - 0.002 <= p.x <= start_tool[0] + 0.202


def test_moveit_goal_is_rejected_while_teleop_owns_the_arm(stack):
    probe, _, _, _ = stack
    # Still armed from the replay: teleop owns the arm.
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_TELEOP, 5.0, 'teleop owner')
    controllers = probe.controllers()
    assert controllers[MOVEIT] == 'inactive' and controllers[TELEOP] == 'active'
    before = probe.joint_positions()
    start = time.monotonic()
    handle = probe.send_goal(ELSEWHERE, seconds=2.0)
    assert handle is not None and not handle.accepted
    time.sleep(1.0)
    assert probe.max_joint_change(start, before) < 1e-6
    assert probe.owner_value() == ArmOwner.OWNER_TELEOP


def test_handing_back_to_moveit_does_not_jump_and_moveit_works_again(stack):
    probe, _, _, _ = stack
    before = probe.joint_positions()
    start = time.monotonic()
    probe.disarm()
    time.sleep(1.0)
    assert probe.max_joint_change(start, before) < 1e-4
    controllers = probe.controllers()
    assert controllers[MOVEIT] == 'active' and controllers[TELEOP] == 'inactive'
    probe.move_to(READY)


def test_target_into_self_collision_ends_in_a_stop_reported_in_teleop_state(stack):
    probe, env, logs, _ = stack
    probe.disarm()
    probe.move_to(READY)
    start_tool = probe.tool_position()
    states = run_replay(probe, env, logs, 'clutch_into_body', 10.0)
    engaged = [s for s in states if s.clutch_engaged]
    assert len(engaged) > 60
    # MoveIt Servo slowed the arm near the collision (reported as the active limit) ...
    slowed = [s for s in engaged if s.servo_status == ServoStatus.DECELERATE_FOR_COLLISION]
    assert slowed and all(s.active_limit == TeleopState.LIMIT_SERVO for s in slowed)
    # ... then halted, and the gate released the clutch so the target could not run on.
    halted = [s for s in states if s.hold_reason == TeleopState.HOLD_SERVO_HALTED]
    assert halted, 'the halt for collision was never reported'
    assert all(s.servo_status == ServoStatus.HALT_FOR_COLLISION and not s.clutch_engaged
               for s in halted)
    first_halt = states.index(halted[0])
    assert not any(s.clutch_engaged for s in states[first_halt:])    # grip held: latched
    # The hand went 0.28 m toward the arm base; the tip stopped well short and stayed.
    end_tool = probe.tool_position()
    travelled = start_tool[0] - end_tool[0]
    assert 0.05 < travelled < 0.23
    last = [s.measured_tool_pose.position.x for s in states[-40:]]
    assert max(last) - min(last) < 0.001
    assert math.isclose(end_tool[0], last[-1], abs_tol=0.002)
    probe.disarm()


def test_log_has_no_gate_or_mux_errors(stack):
    _, _, _, launch = stack
    output = launch.output()
    assert 'simulation-only limits accepted' in output
    assert 'MoveIt Servo verified' in output
    assert 'Traceback' not in output
    assert os.path.getsize(launch.log_path) > 0
