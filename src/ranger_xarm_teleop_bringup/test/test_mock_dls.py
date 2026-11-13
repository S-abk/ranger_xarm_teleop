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
The joint-jog driving path against MoveIt Servo in mock execution.

Covers: the measured tool pose (forward kinematics of /joint_states) agrees with the
transform tree for the same joint state, on the chain of MoveIt's group; the replay that
halted MoveIt Servo's singularity stop passes through without a halt; after a collision
halt, ``recover_arm.py`` brings the arm back and teleop drives it again. One mock stack
for the module, on its own ROS domain.
"""

import math
import os
import time
import xml.etree.ElementTree as ElementTree

from ament_index_python.packages import get_package_share_directory
from moveit_msgs.msg import ServoStatus
import numpy as np
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from ranger_xarm_teleop_servo import solver as dls
from ranger_xarm_teleop_servo.rotations import (pose_from_matrix, q_between, v_norm,
                                                v_sub)
import rclpy

from stack_helpers import (child_environment, Probe, Process, READY,
                           replay_with_clean_input)

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, '..', '..', '..', 'tools')
# Wrist 0.05 rad from aligned: MoveIt Servo's singularity measure is 34.6 there, beyond
# its hard stop; with pose commands every engagement halted at once (servo_diag).
BEYOND_HARD_STOP = [0.0, -0.6, -0.7, 0.0, 0.05, 0.0]
ELSEWHERE = [0.4, -0.3, -1.0, 0.5, 0.9, -0.4]
LEAD = 0.05             # the simulation-only lead limit (m)
# How an engagement of these replays may end: the grip opens, or the recording ends.
EXPECTED_RELEASES = (TeleopState.HOLD_CLUTCH_RELEASED, TeleopState.HOLD_INPUT_STALE)


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_dls')
    env = child_environment(offset=6)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py'],
                     env, str(logs / 'launch.log'))
    probe = Probe(env)
    try:
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: probe.controllers().get('teleop_arm_controller') == 'inactive',
                       60.0, 'the teleop controller to be loaded inactive')
        probe.wait_for(lambda: probe.description is not None, 30.0, 'the robot description')
        probe.move_to(READY)
        yield probe, env, logs, launch
    finally:
        probe.close()
        launch.stop()
        print(launch.output()[-6000:])


def releases(states):
    """Return the hold reason of every state that ended an engagement."""
    return [b.hold_reason for a, b in zip(states, states[1:])
            if a.clutch_engaged and not b.clutch_engaged]


def group_chain():
    """Return the base and tip link of MoveIt's arm group (the SRDF)."""
    share = get_package_share_directory('ranger_xarm_moveit_config')
    root = ElementTree.parse(os.path.join(share, 'config', 'ranger_xarm.srdf')).getroot()
    chain = root.find("group[@name='xarm6']/chain")
    return chain.get('base_link'), chain.get('tip_link')


def test_measured_tool_pose_agrees_with_the_transform_tree(stack):
    """A1: forward kinematics on /joint_states = TF for the same joint state, MoveIt's chain."""
    probe, _, _, launch = stack
    base, tip = group_chain()
    assert (base, tip) == ('xarm_link_base', 'xarm_link_tcp')
    arm = dls.arm_from_urdf(probe.description, base, tip)
    assert arm.joint_names == [f'xarm_joint{i}' for i in range(1, 7)]
    # The teleop node uses the same chain, in the same joint order.
    assert f'solver chain {base} -> {tip}: {arm.joint_names}' in launch.output()
    probe.disarm()

    def compare(samples):
        worst = (0.0, 0.0)
        compared = 0
        for stamp, q in samples:
            try:
                t = probe.tf_buffer.lookup_transform(base, tip, rclpy.time.Time.from_msg(stamp))
            except Exception:  # noqa: B902 - not every stamp is in the buffer; others are
                continue
            tr, ro = t.transform.translation, t.transform.rotation
            fk = pose_from_matrix(dls.tool_pose(arm, q))
            worst = (max(worst[0], v_norm(v_sub(fk.p, (tr.x, tr.y, tr.z)))),
                     max(worst[1], q_between(fk.q, (ro.x, ro.y, ro.z, ro.w))))
            compared += 1
        return worst, compared

    # At rest in three postures: identical, and the node reports that same pose.
    for posture in (ELSEWHERE, BEYOND_HARD_STOP, READY):
        probe.move_to(posture, seconds=3.0)
        time.sleep(0.5)
        (offset, turn), compared = compare(probe.stamped_joints_since(time.monotonic() - 0.3))
        assert compared > 5 and offset < 1e-6 and turn < 1e-6, (posture, offset, turn)
        state = probe.state()
        p, o = state.measured_tool_pose.position, state.measured_tool_pose.orientation
        fk = pose_from_matrix(dls.tool_pose(arm, probe.joint_positions()))
        assert v_norm(v_sub(fk.p, (p.x, p.y, p.z))) < 1e-6
        assert q_between(fk.q, (o.x, o.y, o.z, o.w)) < 1e-6
    # In motion, looked up at each joint state's stamp: the transform tree interpolates
    # between its own samples, so allow what 1 ms of motion at the commanded speed moves.
    start = time.monotonic()
    probe.move_to(ELSEWHERE, seconds=4.0)
    probe.move_to(READY, seconds=4.0)
    (offset, turn), compared = compare(probe.stamped_joints_since(start)[::10])
    print(f'forward kinematics vs transform tree in motion: {offset * 1e6:.1f} um, '
          f'{turn * 1e6:.1f} urad over {compared} joint states')
    assert compared > 20 and offset < 1e-3 and turn < 1e-3


def test_singularity_replay_passes_through_without_a_halt(stack):
    """INV-5: the replay that halted at once with pose commands moves through the wrist."""
    probe, env, logs, _ = stack

    def at_the_singularity():
        probe.move_to(BEYOND_HARD_STOP)
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_singularity.log'),
                                       'clutch_singularity', expected_engaged_s=2.0,
                                       before=at_the_singularity)
    states = probe.states_since(start)
    engaged = [s for s in states if s.clutch_engaged]
    assert len(engaged) > 150, 'the clutch did not stay engaged'
    ended = releases(states)
    assert ended and all(r in EXPECTED_RELEASES for r in ended), ended
    assert not any(s.servo_status in (ServoStatus.HALT_FOR_SINGULARITY,
                                      ServoStatus.HALT_FOR_COLLISION, ServoStatus.JOINT_BOUND)
                   for s in states)
    # It moved, near and through the poorly conditioned region, damped and reported.
    assert probe.max_joint_change(start, BEYOND_HARD_STOP) > 0.2
    assert min(s.conditioning for s in engaged) < 0.25
    assert any(s.damped for s in engaged)
    assert all(s.lead_translation <= LEAD + 1e-9 for s in engaged)
    assert all(s.commanded_linear_speed <= 0.25 + 1e-9 for s in engaged)
    assert all(s.commanded_angular_speed <= 1.0 + 1e-9 for s in engaged)
    assert probe.joint_speeds_since(start) <= 2.0 + 0.05
    # Position held while the wrist absorbed the loss (orientation error reported).
    assert max(s.solver_error_translation for s in engaged) < 0.01
    probe.disarm()


def test_after_a_collision_halt_recovery_is_refused_and_teleop_works_again(stack):
    """
    INV-9, and ``recover_arm.py`` after a collision halt (servo_diag left it untested).

    Found here: after a collision halt the arm is in contact, and MoveIt refuses to plan
    from a start state in collision, so ``recover_arm.py`` reports it and moves nothing.
    The test then moves the arm out directly on MoveIt's controller (no planning, a test
    tool only) and checks that teleop drives again: a halt status left from before must
    not release the new engagement.
    """
    probe, env, logs, _ = stack

    def ready():
        probe.move_to(READY)
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_into_body.log'),
                                       'clutch_into_body', expected_engaged_s=1.0,
                                       before=ready)
    states = probe.states_since(start)
    halted = [s for s in states if s.hold_reason == TeleopState.HOLD_SERVO_HALTED]
    assert halted and all(s.servo_status == ServoStatus.HALT_FOR_COLLISION for s in halted)
    assert probe.owner_value() == ArmOwner.OWNER_TELEOP     # still armed after the halt
    halted_at = probe.joint_positions()
    process = Process(['python3', os.path.join(TOOLS, 'recover_arm.py')], env,
                      str(logs / 'recover_collision.log'))
    try:
        code = process.wait(120.0)
    finally:
        process.stop()
    output = process.output()
    assert 'disarm: disarmed' in output and 'MoveIt owns the arm' in output
    assert code != 0 and 'start state in collision' in output, output
    assert probe.joint_positions() == pytest.approx(halted_at, abs=1e-4)
    probe.move_to(READY)                    # straight to the controller: test only
    assert probe.joint_positions() == pytest.approx(READY, abs=5e-3)
    # Teleop again, from the recovered posture: the arm follows the line.
    before = probe.tool_position()
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_after_recovery.log'),
                                       'clutch_line', expected_engaged_s=2.8)
    states = probe.states_since(start)
    ended = releases(states)
    assert ended and all(r in EXPECTED_RELEASES for r in ended), ended
    moved = [b - a for a, b in zip(before, probe.tool_position())]
    assert moved[0] == pytest.approx(0.20, abs=0.003)
    probe.disarm()


def test_line_replay_reports_speeds_within_the_caps(stack):
    """INV-4 measured: tool speed from the measured pose stays within the caps."""
    probe, env, logs, _ = stack

    def ready():
        probe.move_to(READY)
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_line_speed.log'),
                                       'clutch_line', expected_engaged_s=2.8, before=ready)
    timed = [(when, s) for when, s in probe.timed_states_since(start) if s.clutch_engaged]
    assert len(timed) > 120
    points = np.array([[s.measured_tool_pose.position.x, s.measured_tool_pose.position.y,
                        s.measured_tool_pose.position.z] for _, s in timed])
    stamps = np.array([s.header.stamp.sec + s.header.stamp.nanosec * 1e-9 for _, s in timed])
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.maximum(np.diff(stamps), 1e-3)
    # Five-cycle windows: a single cycle's speed carries the joint-state sampling jitter.
    window = [v_norm(points[i + 5] - points[i]) / (stamps[i + 5] - stamps[i])
              for i in range(len(points) - 5)]
    assert max(window) <= 0.25 * 1.05, max(window)
    assert math.isfinite(float(speeds.max()))
    probe.disarm()


def test_log_has_no_errors(stack):
    _, _, _, launch = stack
    output = launch.output()
    assert 'MoveIt Servo verified' in output
    assert 'Traceback' not in output
    assert 'stopped following' not in output
    assert 'not in joint-jog mode' not in output
