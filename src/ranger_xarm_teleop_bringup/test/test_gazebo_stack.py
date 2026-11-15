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
The teleop stack in headless Gazebo: clocks, the 100 ms input timeout in real time, replay.

Opt-in: set ``TELEOP_TEST_GAZEBO=1`` (the simulator takes a minute to start and needs
the robot underlay's Gazebo packages). Without it every test here is skipped.

The stack runs on its own ROS domain and gz partition. MoveIt Servo, the mux, the teleop
node and the controllers use simulation time; the input replay uses the system clock.
The timeout test runs with the simulation slowed to 0.3 times real time (100 ms of
simulation time then lasts 333 ms) and requires the clutch to release 100 ms
of real time after the input stops. (A paused simulation cannot be used for this: with
no joint states the measured tool pose goes stale and the gate releases for that
reason first.)
"""

import os
import subprocess
import time

from moveit_msgs.msg import ServoStatus
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from rcl_interfaces.srv import GetParameters

from stack_helpers import (child_environment, kill_gz_partition, Probe, Process, READY,
                           replay_with_clean_input)

pytestmark = pytest.mark.skipif(os.environ.get('TELEOP_TEST_GAZEBO') != '1',
                                reason='set TELEOP_TEST_GAZEBO=1 to run the Gazebo tests')

MOVEIT, TELEOP = 'xarm_xarm6_traj_controller', 'teleop_arm_controller'
WORLD = 'empty_ground'


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('gazebo_stack')
    env = child_environment(offset=3)
    env['GZ_PARTITION'] = f'teleop_test_{os.getpid()}'
    env['GZ_IP'] = '127.0.0.1'
    sim = Process(['ros2', 'launch', 'ranger_xarm_gazebo', 'sim.launch.py', 'headless:=true'],
                  env, str(logs / 'sim.log'))
    probe = Probe(env, moveit_controller=MOVEIT)
    teleop = None
    try:
        probe.wait_for(lambda: probe.joint_positions() is not None, 180.0, 'joint states')
        probe.wait_for(lambda: probe.controllers().get(MOVEIT) == 'active', 120.0,
                       'the arm controller')
        teleop = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup',
                          'teleop_sim.launch.py'], env, str(logs / 'teleop.log'))
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 120.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.controllers().get(TELEOP) == 'inactive', 60.0,
                       'the teleop controller to be loaded inactive')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        time.sleep(5.0)             # MoveIt Servo start-up and verification
        probe.move_to(READY, seconds=4.0)
        yield probe, env, logs
    finally:
        probe.close()
        if teleop is not None:
            teleop.stop()
            print(teleop.output()[-4000:])
        sim.stop()
        kill_gz_partition(env['GZ_PARTITION'])     # the gz server can outlive its launch


def set_real_time_factor(env, factor: float) -> None:
    subprocess.run(
        ['gz', 'service', '-s', f'/world/{WORLD}/set_physics', '--reqtype', 'gz.msgs.Physics',
         '--reptype', 'gz.msgs.Boolean', '--timeout', '3000',
         '--req', f'real_time_factor: {factor}'],
        env=env, check=True, capture_output=True, timeout=20)


def stamp_seconds(state) -> float:
    return state.header.stamp.sec + state.header.stamp.nanosec * 1e-9


def use_sim_time(probe, node_name):
    client = probe.node.create_client(GetParameters, f'/{node_name}/get_parameters')
    result = probe.call(client, GetParameters.Request(names=['use_sim_time']), 20.0)
    return result.values[0].bool_value


def test_time_source_of_every_node(stack):
    probe, _, _ = stack
    for node_name in ('servo_node', 'teleop_servo', 'arm_mux', 'move_group', TELEOP, MOVEIT):
        assert use_sim_time(probe, node_name) is True, node_name
    # Six new clients were just announced to every node. While endpoints are being
    # matched a publish call can block for 0.1-0.2 s, which under a held clutch is a
    # (correct) release on stale input; let the discovery settle before the replays.
    time.sleep(3.0)


def test_line_replay_moves_the_tip_along_the_commanded_path(stack):
    probe, env, logs = stack
    probe.disarm()
    tool = {}

    def ready():
        probe.move_to(READY, seconds=4.0)
        tool['start'] = probe.tool_position()
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_line.log'), 'clutch_line',
                                       expected_engaged_s=2.8, before=ready)
    start_tool = tool['start']
    engaged = [s for s in probe.states_since(start) if s.clutch_engaged]
    assert len(engaged) > 120
    end_tool = probe.tool_position()
    moved = [b - a for a, b in zip(start_tool, end_tool)]
    assert moved[0] == pytest.approx(0.20, abs=0.004)
    assert abs(moved[1]) < 0.004 and abs(moved[2]) < 0.004
    for s in engaged:
        p = s.measured_tool_pose.position
        assert abs(p.y - start_tool[1]) < 0.004 and abs(p.z - start_tool[2]) < 0.004
    probe.disarm()


def test_input_timeout_is_100_ms_of_real_time_not_of_simulation_time(stack):
    probe, env, logs = stack
    probe.disarm()
    probe.move_to(READY, seconds=4.0)
    # Slow the simulation before the run (changing it while the clutch is held makes
    # the joint states pause, which the gate rightly answers with TOOL_POSE_STALE).
    set_real_time_factor(env, 0.3)              # simulation time now runs at 0.3 x real time
    try:
        time.sleep(2.0)
        slowed_at = time.monotonic()
        # The recording has its own 0.5 s hole 1.3 s after the grip closes; a run in
        # which the stream broke before that is repeated.
        start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_stall.log'),
                                           'clutch_stall', expected_engaged_s=1.1)
    finally:
        set_real_time_factor(env, 1.0)

    states = probe.timed_states_since(start)
    assert any(s.clutch_engaged for _, s in states), 'the clutch did not engage'
    first_engaged = min(when for when, s in states if s.clutch_engaged)
    # The first engagement: the grip stays held through the hole in the stream.
    release_time, release = next((when, s) for when, s in states
                                 if when > first_engaged and not s.clutch_engaged)
    last_engaged = max(when for when, s in states if s.clutch_engaged and when < release_time)
    assert last_engaged > slowed_at + 0.3, 'the clutch was not held while the simulation ran slow'
    last_input = max(t for t in probe.inputs_since(start) if t <= release_time)

    # Simulation time really ran slow around the release ...
    window = [(when, s) for when, s in states if first_engaged <= when <= release_time]
    wall = window[-1][0] - window[0][0]
    simulated = stamp_seconds(window[-1][1]) - stamp_seconds(window[0][1])
    assert wall > 0.2 and 0.15 < simulated / wall < 0.5, (simulated, wall)
    # ... and the clutch still released on the timeout 100 ms of real time after the
    # last input (plus at most two 20 ms cycles and scheduling). On simulation time it
    # would have taken 333 ms.
    assert release.hold_reason == TeleopState.HOLD_INPUT_STALE
    assert release.input_age > 0.1
    assert 0.1 < release_time - last_input < 0.17
    assert last_engaged - last_input <= 0.1 + 0.02
    probe.disarm()


def test_singularity_replay_passes_through_without_a_halt_and_one_command_recovers(stack):
    """
    INV-5 in Gazebo: the replay that halted MoveIt Servo at once now moves through.

    Replaces the pose-mode test (halt, re-engaging cannot move). Recovery to the ready
    posture from where the replay ends is still one command.
    """
    probe, env, logs = stack
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tools',
                        'recover_arm.py')
    if not os.path.exists(tool):
        pytest.skip('tools/ is not next to the source tree')
    # Wrist 0.05 rad from aligned: MoveIt Servo's measure is 34.6, beyond its hard stop.
    posture = [0.0, -0.6, -0.7, 0.0, 0.05, 0.0]

    def at_the_singularity():
        probe.move_to(posture, seconds=4.0)
        time.sleep(1.0)
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_singularity.log'),
                                       'clutch_singularity', expected_engaged_s=2.0,
                                       before=at_the_singularity)
    time.sleep(1.0)
    states = probe.states_since(start)
    engaged = [s for s in states if s.clutch_engaged]
    assert len(engaged) > 150
    ended = [b.hold_reason for a, b in zip(states, states[1:])
             if a.clutch_engaged and not b.clutch_engaged]
    assert ended and all(r in (TeleopState.HOLD_CLUTCH_RELEASED, TeleopState.HOLD_INPUT_STALE)
                         for r in ended), ended
    assert not any(s.servo_status in (ServoStatus.HALT_FOR_SINGULARITY,
                                      ServoStatus.HALT_FOR_COLLISION, ServoStatus.JOINT_BOUND)
                   for s in states)
    assert probe.max_joint_change(start, posture) > 0.2
    assert any(s.damped for s in engaged) and min(s.conditioning for s in engaged) < 0.25
    assert all(s.lead_translation <= 0.05 + 1e-9 for s in engaged)
    assert probe.owner_value() == ArmOwner.OWNER_TELEOP
    process = Process(['python3', tool], env, str(logs / 'recover.log'))
    try:
        assert process.wait(180.0) == 0, process.output()
    finally:
        process.stop()
    assert 'recovered: the arm is at' in process.output()
    time.sleep(1.0)
    assert probe.joint_positions() == pytest.approx(READY, abs=0.02)
    assert probe.owner_value() == ArmOwner.OWNER_MOVEIT and not probe.state().armed


def test_shoulder_region_is_crossed_without_a_halt(stack):
    """INV-5 (shoulder) in Gazebo: pulled back over joint 1's axis, damped, never halted."""
    probe, env, logs = stack

    def ready():
        probe.move_to(READY, seconds=4.0)
        time.sleep(1.0)
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_shoulder.log'),
                                       'clutch_shoulder', expected_engaged_s=1.0, before=ready)
    time.sleep(1.0)
    states = probe.states_since(start)
    engaged = [s for s in states if s.clutch_engaged]
    assert len(engaged) > 150
    ended = [b.hold_reason for a, b in zip(states, states[1:])
             if a.clutch_engaged and not b.clutch_engaged]
    assert ended and all(r in (TeleopState.HOLD_CLUTCH_RELEASED, TeleopState.HOLD_INPUT_STALE)
                         for r in ended), ended
    assert not any(s.servo_status in (ServoStatus.HALT_FOR_SINGULARITY,
                                      ServoStatus.HALT_FOR_COLLISION, ServoStatus.JOINT_BOUND)
                   for s in engaged)
    assert min(s.conditioning for s in engaged) < 0.2 and any(s.damped for s in engaged)
    assert all(s.lead_translation <= 0.05 + 1e-9 for s in engaged)
    assert probe.joint_speeds_since(start) <= 2.0 + 0.1
    probe.disarm()
