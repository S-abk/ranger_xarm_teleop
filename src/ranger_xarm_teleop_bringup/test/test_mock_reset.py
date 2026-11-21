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
from reset_helpers import (at, contact_paths, describe, EpisodeStandIn, gripper, kill,
                           node_pid, off_ready, phase, play, ready_after, refused, ResetWatcher,
                           stop_measure, stopped_midway)
from stack_helpers import (arm_by_replay, assert_stack_nodes, child_environment,
                           MOCK_STACK_NODES, Probe, Process, READY, start_input)

S = ResetState
STARTS = [off_ready([0.9, 0.3, 0.2, 0.3, -0.3, 0.4]),
          off_ready([-0.8, -0.2, 0.3, -0.4, 0.2, -0.5]),
          off_ready([0.5, -0.3, -0.3, 0.0, 0.3, 0.0])]
FAR = off_ready([1.2, 0.0, 0.0, 0.0, 0.0, 0.0])     # one joint far: a few seconds of motion


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_reset')
    env = child_environment(offset=17)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                      'rviz:=false', 'reset:=true'],
                     env, str(logs / 'launch.log'))
    probe = Probe(env)
    episode = None
    results = []
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
               'episode': episode, 'far': FAR, 'results': results}
    finally:
        if episode is not None:
            episode.close()
        probe.close()
        launch.stop()
        for line in results:
            print(line)


def note(stack, text):
    stack['results'].append(text)


def test_the_stack_runs_the_reset_node_and_its_watchdog(stack):
    probe, launch = stack['probe'], stack['launch']
    # Every controller loaded first (this test runs first in its module, and under load the
    # spawners can still be at work when the stack answers).
    probe.wait_for(lambda: {'teleop_arm_controller', 'teleop_gripper_controller'}
                   <= set(probe.controllers()), 60.0, 'the teleop controllers to be loaded')
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
    probe.wait_for(lambda: at(probe, READY, 0.01), 2.0, 'the joints at the ready posture')
    assert gripper(probe) == grip                        # the gripper is never moved
    assert len(watch.goals_since(start)) == 1
    assert not any(s.state == S.STATE_STOPPING for _, s in watch.since(start, stack['count']))
    events = [s.last_event for _, s in watch.since(start, stack['count'])]
    assert S.EVENT_START in events and events[-1] == S.EVENT_ARRIVED
    # The operator feedback node: one buzz at the start, one long at the arrival, right hand.
    probe.wait_for(lambda: watch.felt(start) == ['START', 'ARRIVED'], 3.0,
                   f'START and ARRIVED felt (felt {watch.felt(start)})')
    assert watch.hands(start) == {'right'}
    probe.wait_for(lambda: any(t.startswith('Reset: at ready') for t in watch.shown(start)),
                   3.0, f'the panel line at the arrival (shown {set(watch.shown(start))})')
    note(stack, f'reach {index}: arrived {when - start:.2f} s after the recording began')


def test_release_midway_stops_the_arm_within_a_measured_distance(stack):
    travel, rest, delay = stopped_midway(
        stack, 'release', [phase(0.3), phase(2.6, 1.0), phase(3.0, 0.0)], S.REASON_RELEASED)
    assert delay < 0.2 and travel < 0.1 and rest < 1.0, (delay, travel, rest)
    watch, start = stack['watch'], stack['start']
    assert watch.felt(start) == ['START']               # a release is not felt
    assert any(t.startswith('Reset stopped: released') for t in watch.shown(start))


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


def test_a_release_right_after_the_dwell_cancels_the_goal_and_nothing_moves(stack):
    """
    Released 60 ms after the dwell: the goal is out and is cancelled before the arm moves.

    Mock planning is quick, so the release can reach the node while move_group is still
    planning or just as it starts executing; what is asserted is that the joints did not move.
    """
    probe, watch = stack['probe'], stack['watch']
    ready_after(stack, FAR)
    before = probe.joint_positions()
    start, _, _ = play(stack, 'planning', [phase(0.3), phase(0.56, 1.0), phase(3.0)])
    after = stack['count']
    watch.wait(start, S.STATE_STOPPING, 5.0, S.REASON_RELEASED, after=after)
    watch.wait(start, S.STATE_STOPPED, 10.0, S.REASON_RELEASED, after=after)
    moved = probe.max_joint_change(start, before)
    seen = sorted({s.state for _, s in watch.since(start, stack['count'])})
    note(stack, f'release right after the dwell: the joints moved {moved:.4f} rad; states seen '
                f'{seen} (3 planning, 4 moving, 5 stopping, 6 stopped)')
    assert len(watch.goals_since(start)) == 1
    assert moved < 0.02, moved


def test_already_at_the_posture_nothing_is_sent(stack):
    watch = stack['watch']
    ready_after(stack, READY)
    start, _, _ = play(stack, 'already', [phase(0.3), phase(1.0, 1.0), phase(0.3)])
    watch.wait(start, S.STATE_ARRIVED, 5.0, S.REASON_ALREADY_THERE, after=stack['count'])
    assert not watch.goals_since(start)


def test_a_press_while_armed_is_refused(stack):
    probe, env, logs = stack['probe'], stack['env'], stack['logs']
    ready_after(stack, FAR)
    arm_by_replay(probe, env, str(logs / 'arm'), disarm=False)
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_TELEOP, 10.0, 'teleop owning')
    before = probe.joint_positions()
    start = refused(stack, 'armed', S.REASON_ARMED)
    assert probe.max_joint_change(start, before) < 1e-3
    watch = stack['watch']
    probe.wait_for(lambda: watch.felt(start) == ['REFUSED'], 3.0,
                   f'one REFUSED felt (felt {watch.felt(start)})')
    probe.wait_for(lambda: any(t.startswith('Reset refused: disarm first (B)')
                               for t in watch.shown(start)), 3.0,
                   f'the refusal on the panel (shown {set(watch.shown(start))})')
    probe.disarm()


def test_a_press_while_an_episode_waits_for_review_is_refused(stack):
    episode = stack['episode']
    episode.state = EpisodeState.STATE_REVIEW      # published at once; the press follows
    try:
        refused(stack, 'review', S.REASON_REVIEW)
    finally:
        episode.state = EpisodeState.STATE_IDLE


def test_a_press_while_an_episode_records_is_refused(stack):
    """Stood in: a real episode stops by itself when teleop is disarmed (then: review)."""
    episode = stack['episode']
    episode.state = EpisodeState.STATE_RECORDING
    try:
        refused(stack, 'recording', S.REASON_RECORDING)
    finally:
        episode.state = EpisodeState.STATE_IDLE


def test_a_posture_moveit_considers_in_collision_moves_nothing_and_says_so(stack):
    """INV-5: from a recorded contact posture the plan fails; nothing moves; the panel says why."""
    probe, env, logs = stack['probe'], stack['env'], stack['logs']
    path = contact_paths()[0]
    probe.move_to(path['positions'][0], seconds=4.0)
    arm_by_replay(probe, env, str(logs / 'arm_contact'), start_delay=3.0, disarm=False)
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_TELEOP, 10.0, 'teleop owning')
    positions = np.array(path['positions'])
    probe.publish_path(positions, path['times'])
    probe.wait_for(lambda: float(np.max(np.abs(np.array(probe.joint_positions())
                                               - positions[-1]))) < 0.01,
                   path['times'][-1] + 5.0, 'the recorded contact posture')
    probe.disarm()
    before = probe.joint_positions()
    start = refused(stack, 'collision', S.REASON_START_IN_COLLISION, goal=True)
    _, state = stack['watch'].first(start, S.STATE_REFUSED, after=stack['count'])
    assert 'retreat_arm.py' in state.reason_text and 'did not move' in state.reason_text
    assert probe.max_joint_change(start, before) < 1e-3
    ready_after(stack, READY)


def test_a_stall_of_the_reset_node_is_stopped_by_the_watchdog_as_heartbeat_late(stack):
    """
    The reset node frozen for 0.4 s mid-motion (as a join stall can freeze a publisher).

    The watchdog cancels at 0.2 s; the node, running again with the stick still held, says
    "heartbeat late" and starts nothing.
    """
    probe, watch = stack['probe'], stack['watch']
    ready_after(stack, stack['far'])
    start, _, process = play(stack, 'stall', [phase(0.3), phase(8.0, 1.0)], wait=False)
    try:
        watch.wait(start, S.STATE_MOVING, 5.0, after=stack['count'])
        probe.wait_for(lambda: probe.max_joint_change(start, FAR) > 0.1, 5.0, 'motion')
        pid = node_pid(stack, 'reset_node')
        os.kill(pid, signal.SIGSTOP)
        stalled = time.monotonic()
        try:
            time.sleep(0.4)                 # the stall itself: longer than the 0.2 s limit
        finally:
            os.kill(pid, signal.SIGCONT)
        travel, rest, speed, tool = stop_measure(probe, stalled)
        _, stopped = watch.wait(start, S.STATE_STOPPED, 10.0, S.REASON_HEARTBEAT_LATE,
                                after=stack['count'])
        probe.wait_for(lambda: any(t.startswith('Reset stopped: heartbeat late')
                                   for t in watch.shown(stalled)), 3.0,
                       f'the panel line (shown {set(watch.shown(stalled))})')
        process.wait(10.0)                  # held to the end of the recording: no restart
    finally:
        process.stop()
    note(stack, f'reset node stalled 0.4 s: joint speed then {speed:.3f} rad/s; joints '
                f'travelled {travel:.4f} rad and the tool {tool * 1000.0:.1f} mm after the stall '
                f'began; rest {rest:.2f} s after it')
    assert speed > 0.05 and not at(probe, READY, 0.05)
    assert len(watch.goals_since(start)) == 1 and not watch.last().goal_out
    assert rest < 1.0, rest


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
    note(stack, f'reset node killed: joint speed then {speed:.3f} rad/s; joints travelled '
                f'{travel:.4f} rad and the tool {tool * 1000.0:.1f} mm after the kill; rest '
                f'{rest:.2f} s after it; watchdog: {report}')
    assert speed > 0.05 and not at(probe, READY, 0.05)
    assert rest < 1.5 and travel < 0.3, (rest, travel)
    assert "for this goal's own id" in report


def sim_speed_scaling():
    """Return reset_speed_scaling from the simulation-only limits file, as the launch does."""
    import yaml
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory('ranger_xarm_teleop_bringup'), 'config',
                        'SIMULATION_ONLY_limits.yaml')
    with open(path, encoding='utf-8') as handle:
        return float(yaml.safe_load(handle)['/**']['ros__parameters']['reset_speed_scaling'])


def restart_reset_node(stack):
    from ament_index_python.packages import get_package_share_directory
    share = get_package_share_directory('ranger_xarm_teleop_bringup')
    node = Process(['ros2', 'run', 'ranger_xarm_teleop_servo', 'reset_node', '--ros-args',
                    '--params-file', os.path.join(share, 'config', 'reset.yaml'),
                    '-p', f'reset_speed_scaling:={sim_speed_scaling()}', '-p', 'enabled:=true'],
                   stack['env'], str(stack['logs'] / 'reset_node_again.log'))
    stack['reset_again'] = node
    since = time.monotonic()
    # Alive and enabled: it hears the watchdog's echo of its own heartbeat. The teleop and
    # episode states it hears at once (they are published), so no discovery wait is left.
    stack['probe'].wait_for(lambda: any(s.watchdog_alive and s.enabled
                                        for _, s in stack['watch'].since(since)), 30.0,
                            'the restarted reset node')


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
    note(stack, f'watchdog killed: stop request {when - killed:.2f} s after the kill; joints '
                f'travelled {travel:.4f} rad, tool {tool * 1000.0:.1f} mm; rest {rest:.2f} s '
                'after the kill')
    assert when - killed < 0.8 and not at(probe, READY, 0.05)


def test_without_the_watchdog_a_press_is_refused(stack):
    ready_after(stack, FAR)
    refused(stack, 'no_watchdog', S.REASON_WATCHDOG_ABSENT)
    node = stack.pop('reset_again', None)
    if node is not None:
        node.stop()
