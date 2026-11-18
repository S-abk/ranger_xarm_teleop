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
Link keep-out (G-15) and the retreat on the mock stack (fake hardware, real MoveIt Servo).

The robot description here has a box-shaped structure on the base behind the arm: what
every collision halt of the live sessions was against.
"""

import json
import math
import os
import sys
import time

from moveit_msgs.msg import RobotState, ServoStatus
from moveit_msgs.srv import GetStateValidity
import numpy as np
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from stack_helpers import (ARM_JOINTS, assert_stack_nodes, child_environment, MOCK_STACK_NODES,
                           Probe, Process, READY, replay_with_clean_input, SERVO_FIXTURES)

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, '..', '..', '..', 'tools')
sys.path.insert(0, os.path.join(HERE, '..', '..', 'ranger_xarm_teleop_servo'))
from ranger_xarm_teleop_servo import keepout as ko  # noqa: E402, I100
from ranger_xarm_teleop_servo import solver as dls  # noqa: E402

MARGIN = 0.025              # the simulation-only file's value
SERVO_HALT_CODES = (ServoStatus.DECELERATE_FOR_COLLISION, ServoStatus.HALT_FOR_COLLISION)


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_keepout')
    env = child_environment(offset=7)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                      'rviz:=false'],
                     env, str(logs / 'launch.log'))
    probe = Probe(env)
    try:
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the command mux with MoveIt as owner')
        probe.wait_for(lambda: probe.description is not None, 30.0, 'the robot description')
        from ament_index_python.packages import get_package_share_directory
        model = ko.build(probe.description, ko.make_resolver(get_package_share_directory),
                         'xarm_link_base', 'xarm_link_tcp', MARGIN)
        arm = dls.arm_from_urdf(probe.description, 'xarm_link_base', 'xarm_link_tcp')
        yield probe, env, logs, launch, model, arm
    finally:
        probe.close()
        launch.stop()


def _extra(model):
    return {name: 0.0 for name in model.extra_joints}


def _tool(arm, q):
    return dls.tool_pose(arm, q)[:3, 3]


def test_retreat_without_a_recorded_path_moves_nothing(stack):
    """INV-8: right after start-up the teleop node has no path."""
    probe, env, logs, _, _, _ = stack
    probe.move_to(READY)
    before = probe.joint_positions()
    process = Process(['python3', os.path.join(TOOLS, 'retreat_arm.py')], env,
                      str(logs / 'retreat_no_path.log'))
    try:
        code = process.wait(120.0)
    finally:
        process.stop()
    output = process.output()
    assert code == 5 and 'no recorded path' in output and 'Nothing moved' in output, output
    assert probe.joint_positions() == pytest.approx(before, abs=1e-4)


def test_pulled_into_the_structure_the_arm_keeps_the_margin_slides_and_leaves(stack):
    """G-15: the slide fixture: back into the structure, sideways along it, away."""
    probe, env, logs, _, model, arm = stack

    def ready():
        probe.move_to(READY)
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_slide.log'),
                                       'clutch_keepout_slide', expected_engaged_s=7.0,
                                       before=ready)
    time.sleep(1.0)
    timed = probe.timed_states_since(start)
    engaged = [s for _, s in timed if s.clutch_engaged]
    assert len(engaged) > 300
    # MoveIt Servo's collision slow-down is never reached; the keep-out did the work.
    assert not any(s.servo_status in SERVO_HALT_CODES for _, s in timed)
    assert any(s.keepout_bound for s in engaged)
    assert any(s.active_limit == TeleopState.LIMIT_LINK_KEEPOUT for s in engaged)
    # INV-1 at MoveIt Servo's commanded state (what the node reports) ...
    assert min(s.keepout_clearance_commanded for s in engaged) >= MARGIN - 1e-3
    # ... and on the measured joints, by the model.
    joints = probe.joints_since(start)
    extra = _extra(model)
    measured = [model.clearance(q, extra)[0] for q in joints[::3]]
    assert min(measured) >= MARGIN - 2e-3, min(measured)
    # Sideways along it: the tool moved to the left (not stopped) ...
    tools = np.array([_tool(arm, q) for q in joints])
    assert tools[:, 1].max() - tools[0, 1] > 0.05
    # ... and away at the end: forward again, by most of the hand's 0.25 m.
    low = int(np.argmin(tools[:, 0]))
    assert tools[-1, 0] - tools[low, 0] > 0.18
    probe.disarm()


def test_turning_back_from_the_structure_is_followed_at_once(stack):
    """G-15 with G-14: the target does not stay inside the blocked direction."""
    probe, env, logs, _, model, arm = stack

    def ready():
        probe.move_to(READY)
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_reverse.log'),
                                       'clutch_keepout_reverse', expected_engaged_s=5.0,
                                       before=ready)
    time.sleep(1.0)
    timed = probe.timed_states_since(start)
    assert not any(s.servo_status in SERVO_HALT_CODES for _, s in timed)
    targets = [(t, p) for t, p in probe.targets if t >= start]
    t_target = np.array([t for t, _ in targets])
    x_target = np.array([p[0] for _, p in targets])
    v_target = np.gradient(x_target, t_target)
    # The hand turns forward at 5.0 s of the recording: the first time the target moves
    # forward at 20 mm/s after it moved back.
    back = int(np.argmin(x_target))
    turned = next(i for i in range(back, len(v_target)) if v_target[i] > 0.02)
    timed_tools = [(when, s.measured_tool_pose.position.x) for when, s in timed
                   if s.measured_tool_pose_valid]
    t_tool = np.array([t for t, _ in timed_tools])
    x_tool = np.array([x for _, x in timed_tools])
    window = 5
    v_tool = (x_tool[window:] - x_tool[:-window]) / (t_tool[window:] - t_tool[:-window])
    after = np.nonzero((t_tool[:-window] >= t_target[turned]) & (v_tool > 0.01))[0]
    assert len(after), 'the arm never followed the hand back'
    delay = t_tool[after[0]] - t_target[turned]
    print(f'reverse: the measured tool follows the turned target after {delay:.3f} s')
    # The lag of the mock stack is about 0.11 s; at most a few cycles more.
    assert delay < 0.25, delay
    probe.disarm()


def _contact_paths():
    with open(os.path.join(SERVO_FIXTURES, 'session3_contact_paths.jsonl')) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _drive_into_contact(probe, path):
    """
    Stand in for teleop and MoveIt Servo: drive the arm along a recorded path into contact.

    Teleop is armed, so the teleop node records the measured path and teleop's controller
    is active; the clutch is released and MoveIt Servo paused, so nothing else commands it.
    The recorded path is sent to that controller as Servo's output would be. (Driving Servo
    itself there ends in its distance-only slow-down short of the contact, the trap found
    in session 2, instead of the halt that ended the live engagements.)
    """
    positions = np.array(path['positions'])
    probe.publish_path(positions, path['times'])
    time.sleep(path['times'][-1] + 2.0)
    q = np.array(probe.joint_positions())
    off = float(np.max(np.abs(q - positions[-1])))
    print(f"{path['name']}: {off:.4f} rad from the recorded contact posture")
    return off < 0.01


def _arm_teleop(probe, env, logs, index):
    """Arm teleop from a recording that presses calibrate and arm only (no grip)."""
    for attempt in range(2):
        arming = Process(['ros2', 'run', 'ranger_xarm_teleop_bridge', 'replay_node',
                          '--ros-args', '-p',
                          f'file:={os.path.join(SERVO_FIXTURES, "arm_only.jsonl")}',
                          '-p', 'wait_for_subscriber_s:=15.0', '-p', 'start_delay_s:=3.0'],
                         env, str(logs / f'arm_{index}.{attempt}.log'))
        try:
            assert arming.wait(60.0) == 0, arming.output()
        finally:
            arming.stop()
        try:
            probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_TELEOP, 10.0,
                           'teleop owning')
            return
        except AssertionError:
            # The button presses of a replay sometimes do not reach the node while the
            # new publisher is being discovered (see replay_with_clean_input).
            print(f'arming {index}: attempt {attempt} did not arm; repeated')
    raise AssertionError('teleop was not armed')


def _recover(env, logs, index):
    """Run recover_arm.py (MoveIt plans to the ready posture); again if not handed back."""
    for attempt in range(2):
        recover = Process(['python3', os.path.join(TOOLS, 'recover_arm.py')], env,
                          str(logs / f'recover_{index}.{attempt}.log'))
        try:
            code = recover.wait(180.0)
        finally:
            recover.stop()
        if code == 0:
            return
        # Exit 4: the owner report was not fresh (nothing moved); seen at the baseline too.
        assert code == 4, recover.output()
        print(f'recover {index}: {recover.output()}; repeated')
    raise AssertionError('recover_arm.py did not recover')


def _valid(probe, q, extra):
    client = probe.node.create_client(GetStateValidity, '/check_state_validity')
    assert client.wait_for_service(timeout_sec=10.0)
    state = RobotState()
    state.joint_state.name = ARM_JOINTS + list(extra)
    state.joint_state.position = [float(v) for v in q] + list(extra.values())
    future = client.call_async(GetStateValidity.Request(robot_state=state, group_name='xarm6'))
    probe.wait_for(future.done, 10.0, 'the state validity')
    return future.result().valid


@pytest.mark.parametrize('index', [0, 1, 2])
def test_retreat_frees_the_arm_from_each_recorded_contact(stack, index):
    """
    INV-7, INV-8: from each of the three session 3 halts, as it happens live.

    The arm is put at the start of the recorded path on MoveIt's controller, teleop is
    armed, the arm is driven along the recorded measured path into the contact
    (``_drive_into_contact``), then ``retreat_arm.py`` alone: the arm ends clear by
    the margin in the model, valid for move_group, plannable again, and teleop drives again.
    """
    probe, env, logs, _, model, arm = stack
    path = _contact_paths()[index]
    extra = _extra(model)
    probe.disarm()
    probe.move_to(path['positions'][0], seconds=4.0)
    _arm_teleop(probe, env, logs, index)
    assert _drive_into_contact(probe, path), 'the arm did not reach the recorded contact'
    halted_at = probe.joint_positions()
    assert model.clearance(halted_at, extra)[0] < MARGIN
    assert not _valid(probe, halted_at, extra)                # move_group: in collision
    process = Process(['python3', os.path.join(TOOLS, 'retreat_arm.py')], env,
                      str(logs / f'retreat_{index}.log'))
    try:
        code = process.wait(180.0)
    finally:
        process.stop()
    output = process.output()
    print(output)
    assert code == 0 and 'retreated' in output, output
    freed = probe.joint_positions()
    assert model.clearance(freed, extra)[0] >= MARGIN - 1e-3
    assert _valid(probe, freed, extra)
    # MoveIt plans from there again.
    _recover(env, logs, index)
    assert probe.joint_positions() == pytest.approx(READY, abs=5e-3)
    # And teleop drives again.
    before = _tool(arm, probe.joint_positions())
    start, _ = replay_with_clean_input(probe, env, str(logs / f'replay_after_{index}.log'),
                                       'clutch_line', expected_engaged_s=2.8)
    states = probe.states_since(start)
    assert not any(s.hold_reason == TeleopState.HOLD_SERVO_NOT_READY and not s.clutch_engaged
                   and s.servo_status == ServoStatus.HALT_FOR_COLLISION for s in states)
    moved = _tool(arm, probe.joint_positions()) - before
    assert moved[0] == pytest.approx(0.20, abs=0.01)
    probe.disarm()


def _retreat(env, logs, name, *options):
    process = Process(['python3', os.path.join(TOOLS, 'retreat_arm.py'), *options], env,
                      str(logs / f'{name}.log'))
    try:
        code = process.wait(180.0)
    finally:
        process.stop()
    output = process.output()
    print(output)
    return code, output


def test_retreat_asked_for_more_clearance_refuses_or_walks_further_back(stack):
    """
    Addition 4 and a longer walk-back, from the first session 3 contact.

    ``--clearance 0.08``: no point of the recorded path within the bounds is that clear by
    the model (link 2 and the arm base read 31-40 mm in this whole region: inflation of the
    covers), so nothing moves and the message gives the best clearance found and the next
    step. ``--clearance 0.03`` is reached within the bounds, further back than the margin.
    """
    probe, env, logs, _, model, arm = stack
    path = _contact_paths()[0]
    extra = _extra(model)
    probe.disarm()
    probe.move_to(path['positions'][0], seconds=4.0)
    _arm_teleop(probe, env, logs, 'far')
    assert _drive_into_contact(probe, path), 'the arm did not reach the recorded contact'
    halted_at = probe.joint_positions()
    code, output = _retreat(env, logs, 'retreat_far_080', '--clearance', '0.08')
    assert code == 5, output
    assert 'REFUSED' in output and 'is clear by 80.0 mm' in output, output
    assert 'the best clearance is' in output and 'Nothing moved' in output, output
    assert 'Next:' in output, output
    assert probe.joint_positions() == pytest.approx(halted_at, abs=1e-4)
    code, output = _retreat(env, logs, 'retreat_far_030', '--clearance', '0.03')
    assert code == 0 and 'retreated' in output, output
    freed = probe.joint_positions()
    assert model.clearance(freed, extra)[0] >= 0.03 - 1e-3
    assert _valid(probe, freed, extra)
    _recover(env, logs, 'far')
    assert probe.joint_positions() == pytest.approx(READY, abs=5e-3)


def test_retreat_refuses_when_the_arm_was_moved_since(stack):
    """INV-8: the path's end must be where the arm is."""
    probe, env, logs, _, _, _ = stack
    probe.disarm()
    probe.move_to([0.3, -0.5, -0.8, 0.0, 1.3, 0.0])
    before = probe.joint_positions()
    process = Process(['python3', os.path.join(TOOLS, 'retreat_arm.py')], env,
                      str(logs / 'retreat_moved.log'))
    try:
        code = process.wait(120.0)
    finally:
        process.stop()
    output = process.output()
    assert code == 5 and 'has moved since' in output, output
    assert probe.joint_positions() == pytest.approx(before, abs=1e-4)
    assert math.isfinite(before[0])


def test_no_other_stack_or_test_shares_this_domain(stack):
    """Isolation: the domain holds this stack's nodes and the probe, nothing else."""
    assert_stack_nodes(stack[0], stack[3], MOCK_STACK_NODES)
