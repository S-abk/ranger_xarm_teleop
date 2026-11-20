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
Gripper teleop in headless Gazebo: a replay closes and opens the simulated gripper.

Opt-in: set ``TELEOP_TEST_GAZEBO=1``. The stack runs on its own ROS domain and gz partition.
In Gazebo the gripper's opening is measured by the physics engine (the followers are driven
from the driven joint by gz_ros2_control), so the measured opening trails the command.
"""

import os
import time

import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, GripperCommand, TeleopState

from stack_helpers import (arm_by_replay, child_environment, enabled_rigid_pairs,
                           kill_gz_partition, Process, READY, replay, start_input)
from test_mock_gripper import assert_same_tool_speed, GripperProbe, speed_run

pytestmark = pytest.mark.skipif(os.environ.get('TELEOP_TEST_GAZEBO') != '1',
                                reason='set TELEOP_TEST_GAZEBO=1 to run the Gazebo tests')

MOVEIT, TELEOP = 'xarm_xarm6_traj_controller', 'teleop_arm_controller'
MOVEIT_G, TELEOP_G = 'xarm_xarm_gripper_traj_controller', 'teleop_gripper_controller'
CLOSED = 0.849


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('gazebo_gripper')
    env = child_environment(offset=10)
    env['GZ_PARTITION'] = f'teleop_gripper_test_{os.getpid()}'
    env['GZ_IP'] = '127.0.0.1'
    sim = Process(['ros2', 'launch', 'ranger_xarm_gazebo', 'sim.launch.py', 'headless:=true'],
                  env, str(logs / 'sim.log'))
    probe = GripperProbe(env, moveit_controller=MOVEIT, moveit_gripper=MOVEIT_G)
    teleop = None
    try:
        probe.wait_for(lambda: probe.joint_positions() is not None, 180.0, 'joint states')
        probe.wait_for(lambda: probe.controllers().get(MOVEIT) == 'active'
                       and probe.controllers().get(MOVEIT_G) == 'active', 120.0,
                       'the arm and gripper controllers')
        teleop = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup',
                          'teleop_sim.launch.py'], env, str(logs / 'teleop.log'))
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 120.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.controllers().get(TELEOP_G) == 'inactive', 60.0,
                       'the teleop gripper controller to be loaded inactive')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        time.sleep(5.0)             # MoveIt Servo start-up and verification
        probe.move_to(READY, seconds=4.0)
        probe.move_gripper(0.0, 1.5)
        time.sleep(3.0)             # discovery of the probe's clients settles
        start_input(probe, env, logs)
        yield probe, env, logs
    finally:
        probe.close()
        if teleop is not None:
            teleop.stop()
            print(teleop.output()[-4000:])
        sim.stop()
        kill_gz_partition(env['GZ_PARTITION'])


def test_replay_closes_and_opens_the_simulated_gripper(stack):
    probe, env, logs = stack
    probe.disarm()
    start = time.monotonic()
    process = replay(env, 'gripper_clutch_strokes', str(logs / 'replay_strokes.log'))
    try:
        assert process.wait(90.0) == 0, process.output()
    finally:
        process.stop()
    time.sleep(2.0)
    positions = [g.position for g in probe.grips_since(start)]
    assert positions and max(positions) == pytest.approx(CLOSED)
    assert positions[-1] == pytest.approx(0.0)
    measured = probe.gripper_since(start)
    assert measured, 'no gripper joint state during the replay'
    assert max(measured) > CLOSED - 0.05, max(measured)
    assert measured[-1] < 0.05, measured[-1]
    states = probe.states_since(start)
    assert any(s.gripper_active and s.gripper_measured_valid for s in states)
    probe.disarm()


def test_arming_starts_from_the_opening_moveit_left(stack):
    probe, env, logs = stack
    probe.disarm()
    probe.move_gripper(0.37, 1.5)
    # Once: the input player was acknowledged in the fixture, so the calibrate and arm presses
    # arrive; see arm_by_replay.
    start, _ = arm_by_replay(probe, env, str(logs / 'replay_held.log'))
    grips = probe.grips_since(start)
    assert grips, 'no gripper command while armed'
    assert grips[0].position == pytest.approx(0.37, abs=1e-6)
    assert grips[0].reference_source == GripperCommand.SOURCE_HELD
    assert all(g.position == grips[0].position for g in grips)
    measured = probe.gripper_since(start)
    assert measured, 'no gripper joint state during the replay'
    assert max(abs(p - 0.37) for p in measured) < 0.01
    assert any(s.gripper_hold == TeleopState.GRIPPER_HOLD_AWAIT_TRIGGER
               for s in probe.states_since(start))
    probe.disarm()


def test_tool_speed_is_the_same_with_the_gripper_open_and_closed(stack):
    """gripper_fix: the same hand motion, gripper open and closed (see test_mock_gripper)."""
    probe, env, logs = stack
    pairs = enabled_rigid_pairs(probe)
    if pairs:
        pytest.skip(f'the SRDF leaves {len(pairs)} self-collision pairs enabled that no arm '
                    f'motion can change (MoveIt Servo slows the arm while the gripper is '
                    f'closed): {pairs}')
    assert_same_tool_speed(speed_run(probe, env, logs, 'open'),
                           speed_run(probe, env, logs, 'closed'))
