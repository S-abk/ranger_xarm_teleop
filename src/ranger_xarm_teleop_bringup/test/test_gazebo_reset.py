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
The reset (G-16) in headless Gazebo: the arm's controllers have physics and simulation time.

Opt-in: set ``TELEOP_TEST_GAZEBO=1``. The same steps as the mock module where the backend
matters: the arm reaches the ready posture from other postures, a release and stale input
stop it (measured), looking away does not, and the refusals; the reset node killed mid-motion
is stopped by the watchdog (measured), last.
"""

import os
import time

import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, EpisodeState, ResetState
from reset_helpers import (at, EpisodeStandIn, gripper, kill, off_ready, phase, play,
                           ready_after, refused, ResetWatcher, stop_measure, stopped_midway)
from stack_helpers import (arm_by_replay, child_environment, kill_gz_partition, Probe, Process,
                           READY, start_input)

pytestmark = pytest.mark.skipif(os.environ.get('TELEOP_TEST_GAZEBO') != '1',
                                reason='set TELEOP_TEST_GAZEBO=1 to run the Gazebo tests')

S = ResetState
MOVEIT, TELEOP = 'xarm_xarm6_traj_controller', 'teleop_arm_controller'
STARTS = [off_ready([0.9, 0.3, 0.2, 0.3, -0.3, 0.4]),
          off_ready([-0.8, -0.2, 0.3, -0.4, 0.2, -0.5])]
FAR = off_ready([1.2, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('gazebo_reset')
    env = child_environment(offset=18)
    env['GZ_PARTITION'] = f'teleop_test_{os.getpid()}_reset'
    env['GZ_IP'] = '127.0.0.1'
    sim = Process(['ros2', 'launch', 'ranger_xarm_gazebo', 'sim.launch.py', 'headless:=true'],
                  env, str(logs / 'sim.log'))
    probe = Probe(env, moveit_controller=MOVEIT)
    teleop, episode, results = None, None, []
    try:
        watcher = ResetWatcher(probe)
        episode = EpisodeStandIn(probe)
        probe.wait_for(lambda: probe.joint_positions() is not None, 180.0, 'joint states')
        probe.wait_for(lambda: probe.controllers().get(MOVEIT) == 'active', 120.0,
                       'the arm controller')
        teleop = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup',
                          'teleop_sim.launch.py', 'reset:=true'], env, str(logs / 'teleop.log'))
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 120.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: watcher.last() is not None and watcher.last().watchdog_alive
                       and watcher.last().enabled, 60.0, 'the reset node and its watchdog')
        time.sleep(5.0)             # MoveIt Servo start-up and verification
        probe.move_to(READY, seconds=4.0)
        start_input(probe, env, logs)
        yield {'probe': probe, 'env': env, 'logs': logs, 'launch': teleop, 'watch': watcher,
               'episode': episode, 'far': FAR, 'results': results}
    finally:
        if episode is not None:
            episode.close()
        probe.close()
        if teleop is not None:
            teleop.stop()
        sim.stop()
        kill_gz_partition(env['GZ_PARTITION'])
        for line in results:
            print(line)


@pytest.mark.parametrize('index', range(len(STARTS)))
def test_held_the_arm_reaches_the_ready_posture_in_gazebo(stack, index):
    probe, watch = stack['probe'], stack['watch']
    ready_after(stack, STARTS[index])
    grip = gripper(probe)
    start, _, _ = play(stack, f'reach_{index}', [phase(0.3), phase(10.0, 1.0), phase(0.3)])
    when, arrived = watch.wait(start, S.STATE_ARRIVED, 10.0, after=stack['count'])
    assert arrived.reason == S.REASON_REACHED
    probe.wait_for(lambda: at(probe, READY, 0.02), 3.0, 'the joints at the ready posture')
    assert gripper(probe) == grip
    stack['results'].append(f'gazebo reach {index}: arrived {when - start:.2f} s after the '
                            'recording began')


def test_release_midway_stops_the_arm_in_gazebo(stack):
    stack['results'].append('gazebo:')
    travel, rest, delay = stopped_midway(
        stack, 'release', [phase(0.3), phase(2.6, 1.0), phase(3.0, 0.0)], S.REASON_RELEASED)
    assert delay < 0.2 and travel < 0.15 and rest < 1.5, (delay, travel, rest)


def test_stale_input_stops_the_arm_in_gazebo(stack):
    travel, rest, delay = stopped_midway(
        stack, 'stale', [phase(0.3), phase(2.6, 1.0), phase(0.0)], S.REASON_INPUT_STALE)
    assert delay < 0.3 and travel < 0.15 and rest < 1.5, (delay, travel, rest)


def test_looking_away_does_not_stop_the_arm_in_gazebo(stack):
    probe, watch = stack['probe'], stack['watch']
    ready_after(stack, FAR)
    start, _, _ = play(stack, 'look_away', [phase(0.3), phase(1.2, 1.0),
                                            phase(1.5, 1.0, look='away'), phase(8.0, 1.0),
                                            phase(0.3)])
    watch.wait(start, S.STATE_ARRIVED, 10.0, S.REASON_REACHED, after=stack['count'])
    assert not any(s.state == S.STATE_STOPPING for _, s in watch.since(start, stack['count']))
    assert at(probe, READY, 0.02)


def test_already_at_the_posture_nothing_is_sent_in_gazebo(stack):
    ready_after(stack, READY)
    start, _, _ = play(stack, 'already', [phase(0.3), phase(1.0, 1.0), phase(0.3)])
    stack['watch'].wait(start, S.STATE_ARRIVED, 5.0, S.REASON_ALREADY_THERE,
                        after=stack['count'])


def test_a_press_while_armed_is_refused_in_gazebo(stack):
    probe, env, logs = stack['probe'], stack['env'], stack['logs']
    ready_after(stack, FAR)
    arm_by_replay(probe, env, str(logs / 'arm'), disarm=False)
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_TELEOP, 10.0, 'teleop owning')
    refused(stack, 'armed', S.REASON_ARMED)
    probe.disarm()


def test_a_press_while_an_episode_waits_for_review_is_refused_in_gazebo(stack):
    episode = stack['episode']
    episode.state = EpisodeState.STATE_REVIEW      # published at once; the press follows
    try:
        refused(stack, 'review', S.REASON_REVIEW)
    finally:
        episode.state = EpisodeState.STATE_IDLE


def test_the_reset_node_killed_midway_is_stopped_by_the_watchdog_in_gazebo(stack):
    probe, watch = stack['probe'], stack['watch']
    ready_after(stack, FAR)
    start, _, process = play(stack, 'kill_reset', [phase(0.3), phase(10.0, 1.0)], wait=False)
    try:
        watch.wait(start, S.STATE_MOVING, 10.0, after=stack['count'])
        probe.wait_for(lambda: probe.max_joint_change(start, FAR) > 0.1, 10.0, 'motion')
        killed = kill(stack, 'reset_node')
        travel, rest, speed, tool = stop_measure(probe, killed, settle=3.0)
        probe.wait_for(lambda: any(m.stops >= 1 and not m.stopping and m.last_report
                                   for _, m in watch.alive_since(killed)), 20.0,
                       'the watchdog to report its stop')
    finally:
        process.stop()
    stack['results'].append(
        f'gazebo, reset node killed: joint speed then {speed:.3f} rad/s; joints travelled '
        f'{travel:.4f} rad and the tool {tool * 1000.0:.1f} mm after the kill; rest '
        f'{rest:.2f} s after it')
    assert speed > 0.05 and not at(probe, READY, 0.05)
    assert rest < 2.0 and travel < 0.3, (rest, travel)
