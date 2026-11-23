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
Base driving (G-19) on the mock stack, with a stand-in base (``fake_base.py``).

Hold the left grip and the left stick drives the base; let go and one zero goes out, then
nothing. Started with ``base:=true reset:=true``: the base node, the reset node and its
watchdog start with the stack. The mock stack has no base, so a stand-in turns /cmd_vel into
/odom (acceleration-limited, with its own command timeout). The last test interrupts the base
node, so it runs last.
"""

import os
import re
import signal
import sys
import time

from base_helpers import after_stop, B, BaseWatcher, calibrate_and_arm, describe, phase, play
from geometry_msgs.msg import Twist
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, ResetState, TeleopState
from reset_helpers import EpisodeStandIn, node_pid, ResetWatcher
from stack_helpers import (assert_stack_nodes, child_environment, MOCK_STACK_NODES, Probe,
                           Process, READY, start_input)

FORWARD = (0.0, -1.0)           # left stick pushed up: robot forward
HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_base')
    env = child_environment(offset=27)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                      'rviz:=false', 'base:=true', 'reset:=true'], env,
                     str(logs / 'launch.log'))
    base = Process([sys.executable, os.path.join(HERE, 'fake_base.py')], env,
                   str(logs / 'fake_base.log'))
    probe = Probe(env)
    episode = None
    results = []
    try:
        watch = BaseWatcher(probe)
        resets = ResetWatcher(probe)
        episode = EpisodeStandIn(probe)
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the command mux with MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: watch.last() is not None and watch.last().enabled
                       and watch.last().at_rest, 60.0, 'the base node, enabled, base at rest')
        probe.wait_for(lambda: resets.last() is not None and resets.last().watchdog_alive,
                       60.0, 'the reset node and its watchdog')
        start_input(probe, env, logs)
        yield {'probe': probe, 'env': env, 'logs': logs, 'launch': launch, 'watch': watch,
               'resets': resets, 'results': results}
    finally:
        if episode is not None:
            episode.close()
        probe.close()
        base.stop()
        launch.stop()
        for line in results:
            print(line)


def note(stack, text):
    stack['results'].append(text)


def drive(stack, name, how_end, seconds=1.5, how=None, wait=True):
    """Idle, drive forward ``seconds``, then the end phase; return (start, begins, replay)."""
    phases = [phase(0.4), phase(seconds, grip=True, left=FORWARD, **(how or {})), how_end]
    return play(stack, name, phases, wait=wait)


def events(stack):
    """Return the base's event count now (the states that follow belong to what follows)."""
    last = stack['watch'].last()
    return last.event_count if last is not None else 0


def at_ready(stack):
    """Put the arm at the ready posture (arming checks the posture) and disarmed."""
    probe = stack['probe']
    if probe.state() is not None and probe.state().armed:
        probe.disarm()
    probe.move_to(READY, seconds=3.0)


def one_zero_then_silence(watch, start):
    """Assert the drive moved, then exactly one zero and nothing; return the stop's time."""
    cmds = watch.commands(start)
    assert any(vx > 0.1 for _, vx, _, _ in cmds), 'the base was never commanded forward'
    tail = after_stop(cmds)
    assert len(tail) == 1, f'after the stop: {tail[:5]} ...'
    assert tail[0][1:] == (0.0, 0.0, 0.0)
    return tail[0][0]


def test_the_stack_runs_the_base_node(stack):
    probe, launch = stack['probe'], stack['launch']
    probe.wait_for(lambda: {'teleop_arm_controller', 'teleop_gripper_controller'}
                   <= set(probe.controllers()), 60.0, 'the teleop controllers to be loaded')
    expected = {**MOCK_STACK_NODES, '/base_node': 1, '/reset_node': 1, '/reset_watchdog': 1,
                '/fake_base': 1}
    assert_stack_nodes(probe, launch, expected)


def test_only_the_base_node_publishes_the_base_command(stack):
    """G-9 and INV-4: one publisher of each base command, the base node."""
    node = stack['probe'].node
    for topic in ('/cmd_vel', '/teleop/base/cmd_vel'):
        names = {i.node_name for i in node.get_publishers_info_by_topic(topic)}
        assert names == {'base_node'}, (topic, names)


def test_held_the_base_drives_and_the_release_sends_one_zero(stack):
    watch, probe = stack['watch'], stack['probe']
    start, _, _ = drive(stack, 'drive_release', phase(1.0))
    stop_at = one_zero_then_silence(watch, start)
    cmds = watch.commands(start)
    driving = [c for c in cmds if c[0] < stop_at]
    rate = (len(driving) - 1) / (driving[-1][0] - driving[0][0])
    assert 40.0 < rate < 60.0, rate
    assert max(vx for _, vx, _, _ in driving) == pytest.approx(0.2, abs=1e-9)   # the cap
    assert all(vy == 0.0 and wz == 0.0 for _, _, vy, wz in driving)            # one mode
    stamped = watch.since('stamped', start)
    assert len(stamped) == len(cmds)
    assert all(m.header.frame_id == 'base_link' for _, m in stamped)
    _, stopped = watch.wait(start, lambda m: m.last_event == B.EVENT_RELEASED, 3.0, 'release')
    assert stopped.reason == B.REASON_RELEASED
    rest = watch.rest_at(stop_at)
    watch.wait(start, lambda m: m.last_event == B.EVENT_AT_REST, 3.0, 'at rest')
    probe.wait_for(lambda: watch.felt(start) == ['START', 'AT_REST'], 3.0,
                   f'START and AT_REST felt (felt {watch.felt(start)})')
    assert watch.hands(start) == {('START', 'left'), ('AT_REST', 'right')}
    assert any(t.startswith('Base: driving') for t in watch.shown(start))
    note(stack, f'mock release: stand-in base at rest {rest - stop_at:.2f} s after the zero')


def test_the_right_stick_turns_in_place_and_the_left_stick_wins(stack):
    watch = stack['watch']
    start, _, _ = play(stack, 'turn', [phase(0.4), phase(1.2, grip=True, right=(-1.0, 0.0)),
                                       phase(1.2, grip=True, left=FORWARD, right=(-1.0, 0.0)),
                                       phase(1.0)])
    cmds = watch.commands(start)
    assert max(wz for *_, wz in cmds) == pytest.approx(0.4)
    assert all(not ((vx or vy) and wz) for _, vx, vy, wz in cmds), 'one mode per command'
    assert any(vx > 0.1 for _, vx, _, _ in cmds)
    assert any(m.turn_ignored for _, m in watch.since('states', start))


@pytest.mark.parametrize('name, end, reason', [
    ('stale', phase(0.0), B.REASON_INPUT_STALE),
    ('left_missing', phase(1.0, left_missing=True), B.REASON_CONTROLLER_MISSING),
    ('session', phase(1.0, grip=True, left=FORWARD, session='new'), B.REASON_SESSION_CHANGED),
    ('seq_back', phase(1.0, grip=True, left=FORWARD, back=40), B.REASON_SEQ_BACKWARDS),
])
def test_each_input_fault_stops_the_base_with_one_zero(stack, name, end, reason):
    watch, probe = stack['watch'], stack['probe']
    count = events(stack)
    start, _, _ = drive(stack, f'fault_{name}', end)
    _, stopped = watch.wait(start, lambda m: m.state == B.STATE_STOPPED
                            and m.event_count > count, 3.0, 'a stop')
    assert stopped.reason == reason, describe(stopped)
    time.sleep(0.6)
    stop_at = one_zero_then_silence(watch, start)
    watch.rest_at(stop_at)
    probe.wait_for(lambda: 'STOPPED' in watch.felt(start), 3.0, 'the stop felt')


def test_another_publisher_refuses_the_drive_and_is_named(stack):
    watch, node = stack['watch'], stack['probe'].node
    intruder = node.create_publisher(Twist, '/cmd_vel', 10)
    try:
        time.sleep(1.5)                                     # discovery, then a graph look
        count = events(stack)
        start, _, _ = drive(stack, 'intruder', phase(0.5))
        _, refused = watch.wait(start, lambda m: m.state == B.STATE_REFUSED
                                and m.event_count > count, 3.0, 'a refusal')
        assert refused.reason == B.REASON_OTHER_PUBLISHER, describe(refused)
        assert node.get_name() in refused.reason_text
        assert not watch.commands(start)
    finally:
        node.destroy_publisher(intruder)
    time.sleep(1.5)


def test_a_pull_back_while_the_left_grip_is_held_is_refused_and_never_turns(stack):
    watch, resets = stack['watch'], stack['resets']
    start, _, _ = play(stack, 'pull_while_held', [
        phase(0.4), phase(1.5, grip=True, right=(0.2, 1.0)), phase(0.5)])
    _, refused = resets.wait(start, ResetState.STATE_REFUSED, 3.0)
    assert refused.reason == ResetState.REASON_BASE_DRIVE
    assert all(wz == 0.0 for *_, wz in watch.commands(start))


def test_a_turn_with_some_pull_turns_the_base_and_starts_no_reset(stack):
    watch, resets = stack['watch'], stack['resets']
    start, _, _ = play(stack, 'turn_diagonal', [
        phase(0.4), phase(1.5, grip=True, right=(-0.9, 0.85)), phase(0.8)])
    assert any(wz > 0.1 for *_, wz in watch.commands(start))
    busy = (ResetState.STATE_WAITING, ResetState.STATE_PLANNING, ResetState.STATE_MOVING)
    assert not [s for _, s in resets.since(start) if s.state in busy]


def holds(probe, t):
    return [s.hold_reason for s in probe.states_since(t)]


def test_the_arm_and_the_base_exclude_each_other(stack):
    """INV-2 both ways on the stack: first holder wins; the clutch waits for the base's rest."""
    watch, probe = stack['watch'], stack['probe']
    at_ready(stack)
    count = events(stack)
    start, begins, _ = play(stack, 'exclusion', calibrate_and_arm() + [
        phase(1.5, rgrip=True, move=(0.0, 0.0, -0.05)),                 # 5: an arm stroke
        phase(1.0, rgrip=True, grip=True, left=FORWARD),                # 6: left grip joins
        phase(0.5),                                                     # 7
        phase(1.5, grip=True, left=FORWARD),                            # 8: a drive
        phase(1.0, grip=True, left=FORWARD, rgrip=True),                # 9: right grip joins
        phase(0.5, rgrip=True),                                         # 10: left released
        phase(0.3),                                                     # 11
        phase(1.5, rgrip=True, move=(0.0, 0.0, 0.05)),                  # 12: right grip again
        phase(0.5)])
    states = probe.timed_states_since(start)
    assert any(s.clutch_engaged for _, s in states), 'the arm was never engaged'
    # While the arm was engaged, the left grip was refused and nothing went to the base.
    refused = [m for _, m in watch.since('states', start)
               if m.state == B.STATE_REFUSED and m.event_count > count]
    assert refused and refused[0].reason in (B.REASON_RIGHT_GRIP_HELD,
                                             B.REASON_CLUTCH_ENGAGED), describe(refused[0])
    # The drive: commanded, and the clutch never engaged while it was.
    cmds = watch.commands(start)
    assert any(vx > 0.1 for _, vx, _, _ in cmds)
    first, last = cmds[0][0], cmds[-1][0]
    assert not [s for w, s in states if first <= w <= last and s.clutch_engaged]
    assert TeleopState.HOLD_BASE_DRIVE in holds(probe, start)
    # Engaged again after the base came to rest (phase 12), never during its motion.
    assert any(s.clutch_engaged for w, s in states if w > last)
    probe.disarm()


def test_a_turn_of_the_base_past_the_limit_needs_a_new_calibration(stack):
    watch, probe = stack['watch'], stack['probe']
    at_ready(stack)
    start, _, _ = play(stack, 'turned', calibrate_and_arm() + [
        phase(1.5, grip=True, right=(-1.0, 0.0)),                       # ~20 deg
        phase(1.0),
        phase(1.0, rgrip=True, move=(0.0, 0.0, -0.05)),
        phase(0.3),
        phase(0.2, a=True), phase(0.3),
        phase(1.0, rgrip=True, move=(0.0, 0.0, 0.05)), phase(0.3)])
    assert any(wz > 0.1 for *_, wz in watch.commands(start))
    seen = holds(probe, start)
    assert TeleopState.HOLD_BASE_TURNED in seen
    states = probe.timed_states_since(start)
    turned_at = min(w for w, s in states if s.hold_reason == TeleopState.HOLD_BASE_TURNED)
    assert any(s.clutch_engaged for w, s in states if w > turned_at), 'A did not restore it'
    probe.disarm()


def test_the_graph_look_and_the_cycle_with_the_whole_stack(stack):
    """Approval point 4: the base node's own timing line, with this stack in the graph."""
    path = stack['logs'] / 'launch.log'
    pattern = re.compile(r'base timing: cycle n (\d+) p50 ([\d.]+) p99 ([\d.]+) max ([\d.]+) '
                         r'ms; period n \d+ p50 ([\d.]+) p99 ([\d.]+) max ([\d.]+) ms; graph '
                         r'query n (\d+) p50 ([\d.]+) p99 ([\d.]+) max ([\d.]+) ms')

    def lines():
        with open(path, encoding='utf-8', errors='replace') as handle:
            return pattern.findall(handle.read())
    stack['probe'].wait_for(lines, 40.0, 'a timing line of the base node')
    for found in lines():
        values = [float(v) for v in found]
        cycle_max, period_p99, graph_p99, graph_max = values[3], values[5], values[9], values[10]
        note(stack, f'base timing: cycle p50 {values[1]} p99 {values[2]} max {cycle_max} ms; '
                    f'period p50 {values[4]} p99 {period_p99} max {values[6]} ms; graph query '
                    f'n {int(values[7])} p50 {values[8]} p99 {graph_p99} max {graph_max} ms')
        assert graph_p99 < 2.0, 'the graph look takes a tenth of the cycle'


def test_sigint_of_the_base_node_sends_the_zero_before_it_leaves(stack):
    """Runs last: the base node does not come back."""
    watch = stack['watch']
    pid = node_pid(stack, 'base_node')
    start, begins, process = drive(stack, 'sigint', phase(2.0, grip=True, left=FORWARD),
                                   seconds=1.5, wait=False)
    stack['probe'].wait_for(lambda: any(vx > 0.15 for _, vx, _, _ in watch.commands(start)),
                            5.0, 'the drive')
    sent = time.monotonic()
    os.kill(pid, signal.SIGINT)
    assert process.wait(30.0) == 0
    tail = after_stop(watch.commands(start))
    assert len(tail) == 1 and tail[0][1:] == (0.0, 0.0, 0.0), tail[:5]
    note(stack, f'SIGINT: the zero arrived {tail[0][0] - sent:.3f} s after the signal')
    watch.rest_at(tail[0][0])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError('the base node did not leave after SIGINT')
