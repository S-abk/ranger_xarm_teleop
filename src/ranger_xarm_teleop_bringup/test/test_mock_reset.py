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
The reset (G-16) on the mock stack.

Hold the right thumbstick back, disarmed, and the arm returns to the ready posture along a
MoveIt plan; let go and it stops.

Started with ``reset:=true``: the reset node and its watchdog start with the stack. An
episode node is stood in for by the probe (``EpisodeStandIn``). The stops are measured from
the joint states: how far the joints travelled after the stop request, and how long until they
rested. The last tests kill the reset node and then the watchdog, so they run last, in order.
"""

import os
import signal
import time

import numpy as np
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, EpisodeState, ResetState
from reset_helpers import (at, contact_paths, describe, EpisodeStandIn, off_ready, phase,
                           ResetWatcher, stop_measure, write_recording)
from stack_helpers import (arm_by_replay, assert_stack_nodes, child_environment,
                           MOCK_STACK_NODES, node_pids, Probe, Process, READY, replay_file,
                           start_input)

S = ResetState
STARTS = [off_ready([0.9, 0.3, 0.2, 0.3, -0.3, 0.4]),
          off_ready([-0.8, -0.2, 0.3, -0.4, 0.2, -0.5]),
          off_ready([0.5, -0.3, -0.3, 0.0, 0.3, 0.0])]
FAR = off_ready([1.2, 0.0, 0.0, 0.0, 0.0, 0.0])     # one joint far: a motion of a few seconds
RESULTS = []                                        # stop measurements, printed at the end


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_reset')
    env = child_environment(offset=17)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                      'rviz:=false', 'reset:=true'],
                     env, str(logs / 'launch.log'))
    probe = Probe(env)
    episode = None
    try:
        watcher = ResetWatcher(probe)
        episode = EpisodeStandIn(probe)
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the command mux with MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: watcher.last() is not None and watcher.last().watchdog_alive
                       and watcher.last().enabled, 60.0, 'the reset node and its watchdog')
        start_input(probe, env, logs)
        yield {'probe': probe, 'env': env, 'logs': logs, 'launch': launch, 'watch': watcher,
               'episode': episode}
    finally:
        if episode is not None:
            episode.close()
        probe.close()
        launch.stop()
        for line in RESULTS:
            print(line)


def play(stack, name, phases, wait=True):
    """Write and play a recording; return (start, phase begins, replay)."""
    path = str(stack['logs'] / f'{name}.jsonl')
    begins = write_recording(path, phases)
    stack['count'] = stack['watch'].count()      # events before this recording
    start = time.monotonic()
    process = replay_file(stack['env'], path, str(stack['logs'] / f'{name}.log'),
                          start_delay=0.0)
    if wait:
        assert process.wait(sum(p[0] for p in phases) + 20.0) == 0, process.output()
    return start, begins, process


def ready_after(stack, posture):
    probe = stack['probe']
    probe.move_to(posture, seconds=3.0)
    assert at(probe, posture, 0.005)


def gripper(probe):
    state = probe.state()
    return state.gripper_measured_valid, round(state.gripper_measured, 3)


def test_the_stack_runs_the_reset_node_and_its_watchdog(stack):
    probe, launch = stack['probe'], stack['launch']
    expected = {**MOCK_STACK_NODES, '/reset_node': 1, '/reset_watchdog': 1}
    assert_stack_nodes(probe, launch, expected)


@pytest.mark.parametrize('index', range(len(STARTS)))
def test_held_the_arm_reaches_the_ready_posture_from_another_posture(stack, index):
    probe, watch = stack['probe'], stack['watch']
    ready_after(stack, STARTS[index])
    grip = gripper(probe)
    start, _, _ = play(stack, f'reach_{index}', [phase(0.3), phase(8.0, 1.0), phase(0.3)])
    when, arrived = watch.wait(start, S.STATE_ARRIVED, 5.0, after=stack['count'])
    assert arrived.reason == S.REASON_REACHED, describe(arrived)
    time.sleep(0.5)
    assert at(probe, READY, 0.01), probe.joint_positions()
    assert gripper(probe) == grip                        # the gripper is never moved
    assert len(watch.goals_since(start)) == 1
    assert not any(s.state == S.STATE_STOPPING for _, s in watch.since(start, stack['count']))
    events = [s.last_event for _, s in watch.since(start, stack['count'])]
    assert S.EVENT_START in events and events[-1] == S.EVENT_ARRIVED
    RESULTS.append(f'reach {index}: arrived {when - start:.2f} s after the recording began')


def stopped_midway(stack, name, phases, reason):
    """From FAR, hold, then end the hold the way ``phases`` say; return the stop's numbers."""
    probe, watch = stack['probe'], stack['watch']
    ready_after(stack, FAR)
    start, begins, _ = play(stack, name, phases)
    when, stopping = watch.wait(start, S.STATE_STOPPING, 5.0, after=stack['count'])
    assert stopping.reason == reason, describe(stopping)
    travel, rest, speed, tool = stop_measure(probe, when)
    _, stopped = watch.wait(start, S.STATE_STOPPED, 10.0, after=stack['count'])
    assert stopped.reason == reason, describe(stopped)
    assert not at(probe, READY, 0.05), 'it ran on to the posture'
    assert speed > 0.05, f'the arm was not moving when the hold ended ({speed:.3f} rad/s)'
    release = start + begins[2]
    RESULTS.append(
        f'{name}: hold ended at {begins[2]:.2f} s; stop request {when - release:+.3f} s after '
        f'it; joint speed then {speed:.3f} rad/s; joints travelled {travel:.4f} rad and the tool '
        f'{tool * 1000.0:.1f} mm after the request; rest {rest:.2f} s after it (node: '
        f'{stopped.stop_distance:.4f} rad, {stopped.stop_time:.2f} s)')
    return travel, rest, when - release


def test_release_midway_stops_the_arm_within_a_measured_distance(stack):
    travel, rest, delay = stopped_midway(
        stack, 'release', [phase(0.3), phase(2.6, 1.0), phase(3.0, 0.0)], S.REASON_RELEASED)
    assert delay < 0.2 and travel < 0.1 and rest < 1.0, (delay, travel, rest)


def test_stale_input_stops_the_arm(stack):
    """The input stops (as when the relay disconnects): stale after 100 ms."""
    travel, rest, delay = stopped_midway(
        stack, 'stale', [phase(0.3), phase(2.6, 1.0), phase(0.0)], S.REASON_INPUT_STALE)
    assert delay < 0.3 and travel < 0.1 and rest < 1.0, (delay, travel, rest)


def test_a_lost_controller_stops_the_arm(stack):
    travel, rest, delay = stopped_midway(
        stack, 'controller_lost', [phase(0.3), phase(2.6, 1.0), phase(3.0, controller=False)],
        S.REASON_CONTROLLER_MISSING)
    assert delay < 0.2 and travel < 0.1, (delay, travel)


def test_a_new_session_stops_the_arm_and_a_held_stick_does_not_resume(stack):
    probe, watch = stack['probe'], stack['watch']
    held_on = phase(4.0, 1.0, session='new')
    stopped_midway(stack, 'session', [phase(0.3), phase(2.6, 1.0), held_on],
                   S.REASON_SESSION_CHANGED)
    last = watch.last()
    assert last.state == S.STATE_STOPPED and not last.goal_out    # held on: nothing restarted
    assert not at(probe, READY, 0.05)


def test_looking_away_and_other_pose_only_conditions_do_not_stop_the_arm(stack):
    """G-16: the operator may watch the arm, not the controller."""
    probe, watch = stack['probe'], stack['watch']
    ready_after(stack, FAR)
    start, _, _ = play(stack, 'pose_only', [
        phase(0.3), phase(1.2, 1.0), phase(0.6, 1.0, look='away'),
        phase(0.6, 1.0, emulated=True), phase(0.6, 1.0, head=False),
        phase(5.0, 1.0, reset_count=1), phase(0.3)])
    _, arrived = watch.wait(start, S.STATE_ARRIVED, 5.0, after=stack['count'])
    assert arrived.reason == S.REASON_REACHED
    assert not any(s.state == S.STATE_STOPPING for _, s in watch.since(start, stack['count']))
    assert at(probe, READY, 0.01)


def test_a_release_during_planning_cancels_the_goal(stack):
    probe, watch = stack['probe'], stack['watch']
    ready_after(stack, FAR)
    before = probe.joint_positions()
    start, _, _ = play(stack, 'planning', [phase(0.3), phase(0.56, 1.0), phase(3.0)])
    after = stack['count']
    watch.wait(start, S.STATE_STOPPING, 5.0, S.REASON_RELEASED, after=after)
    watch.wait(start, S.STATE_STOPPED, 10.0, S.REASON_RELEASED, after=after)
    moved = probe.max_joint_change(start, before)
    seen = sorted({s.state for _, s in watch.since(start, stack['count'])})
    RESULTS.append(f'release during planning: the joints moved {moved:.4f} rad; states seen '
                   f'{seen} (mock planning can end before the release is seen)')
    assert moved < 0.02, moved


def test_already_at_the_posture_nothing_is_sent(stack):
    watch = stack['watch']
    ready_after(stack, READY)
    start, _, _ = play(stack, 'already', [phase(0.3), phase(1.0, 1.0), phase(0.3)])
    watch.wait(start, S.STATE_ARRIVED, 5.0, S.REASON_ALREADY_THERE, after=stack['count'])
    assert not watch.goals_since(start)


def refused(stack, name, reason, goal=False):
    """Press and hold; expect a refusal for ``reason`` (``goal``: MoveIt refuses the goal)."""
    watch = stack['watch']
    start, _, _ = play(stack, name, [phase(0.3), phase(1.5, 1.0), phase(0.3)])
    _, state = watch.wait(start, S.STATE_REFUSED, 5.0, after=stack['count'])
    assert state.reason == reason, describe(state)
    assert len(watch.goals_since(start)) == (1 if goal else 0)
    return start


def test_a_press_while_armed_is_refused(stack):
    probe, env, logs = stack['probe'], stack['env'], stack['logs']
    ready_after(stack, FAR)
    arm_by_replay(probe, env, str(logs / 'arm'), disarm=False)
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_TELEOP, 10.0, 'teleop owning')
    before = probe.joint_positions()
    start = refused(stack, 'armed', S.REASON_ARMED)
    assert probe.max_joint_change(start, before) < 1e-3
    probe.disarm()


def test_a_press_while_an_episode_waits_for_review_is_refused(stack):
    episode = stack['episode']
    episode.state = EpisodeState.STATE_REVIEW
    try:
        time.sleep(0.5)
        refused(stack, 'review', S.REASON_REVIEW)
    finally:
        episode.state = EpisodeState.STATE_IDLE
        time.sleep(0.5)


def test_a_posture_moveit_considers_in_collision_moves_nothing_and_says_so(stack):
    """INV-5: from a recorded contact posture the plan fails; nothing moves; the panel says why."""
    probe, env, logs = stack['probe'], stack['env'], stack['logs']
    path = contact_paths()[0]
    probe.move_to(path['positions'][0], seconds=4.0)
    arm_by_replay(probe, env, str(logs / 'arm_contact'), start_delay=3.0, disarm=False)
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_TELEOP, 10.0, 'teleop owning')
    positions = np.array(path['positions'])
    probe.publish_path(positions, path['times'])
    time.sleep(path['times'][-1] + 2.0)
    assert float(np.max(np.abs(np.array(probe.joint_positions()) - positions[-1]))) < 0.01
    probe.disarm()
    before = probe.joint_positions()
    start = refused(stack, 'collision', S.REASON_START_IN_COLLISION, goal=True)
    _, state = stack['watch'].first(start, S.STATE_REFUSED, after=stack['count'])
    assert 'retreat_arm.py' in state.reason_text and 'did not move' in state.reason_text
    assert probe.max_joint_change(start, before) < 1e-3
    ready_after(stack, READY)


def kill(stack, executable):
    pids = node_pids(executable, pgid=os.getpgid(stack['launch'].proc.pid))
    assert len(pids) == 1, (executable, pids)
    os.kill(pids[0], signal.SIGKILL)
    return time.monotonic()


def test_the_reset_node_killed_midway_is_stopped_by_the_watchdog(stack):
    probe, watch = stack['probe'], stack['watch']
    ready_after(stack, FAR)
    start, _, process = play(stack, 'kill_reset', [phase(0.3), phase(8.0, 1.0)], wait=False)
    try:
        watch.wait(start, S.STATE_MOVING, 5.0, after=stack['count'])
        probe.wait_for(lambda: probe.max_joint_change(start, FAR) > 0.1, 5.0, 'motion')
        killed = kill(stack, 'reset_node')
        travel, rest, speed, tool = stop_measure(probe, killed, settle=2.5)
        probe.wait_for(lambda: any(m.stops >= 1 and not m.stopping and m.last_report
                                   for _, m in watch.alive_since(killed)), 15.0,
                       'the watchdog to report its stop')
    finally:
        process.stop()
    report = next(m.last_report for _, m in reversed(watch.alive_since(killed)) if m.last_report)
    RESULTS.append(f'reset node killed: joint speed then {speed:.3f} rad/s; joints travelled '
                   f'{travel:.4f} rad and the tool {tool * 1000.0:.1f} mm after the kill; rest '
                   f'{rest:.2f} s after it; watchdog: {report}')
    assert speed > 0.05 and not at(probe, READY, 0.05)
    assert rest < 1.5 and travel < 0.3, (rest, travel)
    assert "for this goal's own id" in report


def restart_reset_node(stack):
    from ament_index_python.packages import get_package_share_directory
    share = get_package_share_directory('ranger_xarm_teleop_bringup')
    node = Process(['ros2', 'run', 'ranger_xarm_teleop_servo', 'reset_node', '--ros-args',
                    '--params-file', os.path.join(share, 'config', 'reset.yaml'),
                    '-p', 'reset_speed_scaling:=0.2', '-p', 'enabled:=true'],
                   stack['env'], str(stack['logs'] / 'reset_node_again.log'))
    stack['reset_again'] = node
    since = time.monotonic()
    stack['probe'].wait_for(lambda: any(s.watchdog_alive and s.enabled
                                        for _, s in stack['watch'].since(since)), 30.0,
                            'the restarted reset node')
    time.sleep(6.0)                 # past its discovery time for the teleop and episode nodes


def test_the_watchdog_killed_midway_stops_the_motion(stack):
    probe, watch = stack['probe'], stack['watch']
    restart_reset_node(stack)
    ready_after(stack, FAR)
    start, _, process = play(stack, 'kill_watchdog', [phase(0.3), phase(8.0, 1.0)], wait=False)
    try:
        watch.wait(start, S.STATE_MOVING, 5.0, after=stack['count'])
        probe.wait_for(lambda: probe.max_joint_change(start, FAR) > 0.1, 5.0, 'motion')
        killed = kill(stack, 'reset_watchdog_node')
        when, _ = watch.wait(killed, S.STATE_STOPPING, 2.0, S.REASON_WATCHDOG_LOST,
                             after=stack['count'])
        travel, rest, speed, tool = stop_measure(probe, killed)
        watch.wait(killed, S.STATE_STOPPED, 10.0, S.REASON_WATCHDOG_LOST, after=stack['count'])
    finally:
        process.stop()
    RESULTS.append(f'watchdog killed: stop request {when - killed:.2f} s after the kill; joints '
                   f'travelled {travel:.4f} rad, tool {tool * 1000.0:.1f} mm; rest {rest:.2f} s '
                   'after the kill')
    assert when - killed < 0.8 and not at(probe, READY, 0.05)


def test_without_the_watchdog_a_press_is_refused(stack):
    ready_after(stack, FAR)
    refused(stack, 'no_watchdog', S.REASON_WATCHDOG_ABSENT)
    node = stack.pop('reset_again', None)
    if node is not None:
        node.stop()
