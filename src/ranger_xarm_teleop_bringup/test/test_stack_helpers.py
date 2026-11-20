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
"""The stack tests' launch-log reading and expected-node check on recorded lines (no ROS graph)."""

import json
import os
import shutil
from types import SimpleNamespace

import pytest
from ranger_xarm_teleop_interfaces.msg import TeleopState
import stack_helpers
from stack_helpers import (arm_by_replay, assert_stack_nodes, exited_node_patterns, ignoring,
                           INPUT_NODE, keep_logs, launched_processes, lead_in_acknowledged,
                           lead_in_file, LEAD_IN_SESSION, MOCK_RECORD_NODES, MOCK_STACK_NODES,
                           press_frames, press_never_arrived, prime, replay,
                           replay_with_clean_input, SERVO_FIXTURES)

# Lines as a mock stack's launch prints them (abridged from a real log; ``rviz:=false``).
STARTED = """[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [robot_state_publisher-1]: process started with pid [3184664]
[INFO] [ros2_control_node-2]: process started with pid [3184665]
[INFO] [spawner-3]: process started with pid [3184666]
[INFO] [move_group-5]: process started with pid [3184672]
[INFO] [servo_node-9]: process started with pid [3184680]
[INFO] [arm_mux_node-10]: process started with pid [3184682]
[INFO] [teleop_node-11]: process started with pid [3184683]
[INFO] [gripper_relay_node-13]: process started with pid [3184689]
"""
SPAWNER_DONE = '[INFO] [spawner-3]: process has finished cleanly [pid 3184666]\n'
RELAY_DIED = ('[ERROR] [gripper_relay_node-13]: process has died [pid 3184689, exit code -2, '
              'cmd gripper_relay_node].\n')
# RViz started after all (the argument not reaching the MoveIt launch), then aborting off screen.
RVIZ_STARTED = ('[INFO] [rviz2-6]: process started with pid [3184673]\n'
                '[rviz2-6] [ERROR] [1795010055.962286643] [rviz2]: rviz::RenderSystem: error '
                'creating render window\n')
RVIZ_DIED = ('[ERROR] [rviz2-6]: process has died [pid 3184673, exit code -6, cmd '
             '/opt/ros/jazzy/lib/rviz2/rviz2 -d planning.rviz].\n')

# The nodes of a running mock stack (a live listing; the numbered names change per run).
LIVE = ['/arm_mux', '/arm_mux_client', '/arm_mux_observer', '/controller_manager',
        '/gripper_relay', '/joint_state_broadcaster', '/move_group', '/move_group/moveit',
        '/move_group_private_110062872965072', '/moveit_174894846', '/moveit_844807136',
        '/moveit_simple_controller_manager', '/operator_feedback', '/robot_state_publisher',
        '/servo_node',
        '/servo_node/moveit', '/servo_node_private_100409403087936', '/teleop_arm_controller',
        '/teleop_gripper_controller', '/teleop_servo', '/teleop_servo_watcher',
        '/teleop_test_probe', '/transform_listener_impl_641a02aa8e70',
        '/transform_listener_impl_72cdf4009a40', '/xarm6_traj_controller',
        '/xarm_gripper_traj_controller', '/xarm_mock_components_genericsystem',
        '/xarm_xarmgrippersystem']


class _Launch:
    """A launch whose output is fixed text."""

    def __init__(self, text):
        self.text = text

    def output(self):
        return self.text


class _Node:
    """A probe's node that sees a fixed list of nodes."""

    def __init__(self, names):
        self.names = names

    def get_name(self):
        return 'teleop_test_probe'

    def get_node_names_and_namespaces(self):
        return [(n.rsplit('/', 1)[1], n.rsplit('/', 1)[0] or '/') for n in self.names]


class _Probe:
    """A probe with that node."""

    def __init__(self, names):
        self.node = _Node(names)


def check(names, log, expected=MOCK_STACK_NODES):
    assert_stack_nodes(_Probe(names), _Launch(log), expected, settle=0.0)


def test_started_and_exited_processes_are_read_from_the_launch_log():
    assert launched_processes(_Launch(STARTED)) == (
        {'robot_state_publisher', 'ros2_control_node', 'spawner', 'move_group', 'servo_node',
         'arm_mux_node', 'teleop_node', 'gripper_relay_node'}, set())
    started, exited = launched_processes(_Launch(STARTED + SPAWNER_DONE + RVIZ_STARTED
                                                 + RVIZ_DIED))
    assert exited == {'rviz2', 'spawner'} and exited < started


def test_only_nodes_of_finished_spawners_are_left_out_and_any_other_exit_fails():
    patterns = exited_node_patterns(_Launch(STARTED + SPAWNER_DONE))
    assert ignoring(['/spawner_xarm6_traj_controller', '/spawner', '/move_group'],
                    patterns) == ['/spawner', '/move_group']
    assert exited_node_patterns(_Launch(STARTED)) == ()
    with pytest.raises(AssertionError, match='gripper_relay_node'):
        exited_node_patterns(_Launch(STARTED + SPAWNER_DONE + RELAY_DIED))
    with pytest.raises(AssertionError, match='rviz2'):
        exited_node_patterns(_Launch(STARTED + SPAWNER_DONE + RVIZ_STARTED + RVIZ_DIED))


def test_the_live_stack_matches_its_expected_nodes_with_or_without_a_spawner():
    check(LIVE, STARTED + SPAWNER_DONE)
    check(LIVE + ['/spawner_xarm_gripper_traj_controller'], STARTED)      # still loading
    check(LIVE + ['/spawner_xarm_gripper_traj_controller'], STARTED + SPAWNER_DONE)  # lingering
    check(LIVE + ['/episode_node', '/teleop_episode_recorder'], STARTED, MOCK_RECORD_NODES)


@pytest.mark.parametrize('names, log, message', [
    pytest.param(LIVE + ['/unrelated_tool_node'], '',
                 "nodes not of this stack: ['/unrelated_tool_node']", id='unrelated_node'),
    pytest.param(LIVE + ['/teleop_test_probe'], '', 'teleop_test_probe', id='second_probe'),
    pytest.param(LIVE + ['/arm_mux'], '', 'arm_mux', id='second_stack'),
    pytest.param([n for n in LIVE if n != '/teleop_servo'], '', 'teleop_servo',
                 id='missing_node'),
    pytest.param(LIVE + ['/moveit_123'], '', 'moveit_', id='numbered_node_too_many'),
    pytest.param(LIVE + ['/episode_node'], '', 'episode_node', id='record_node'),
    pytest.param(LIVE + ['/rviz'], RVIZ_STARTED, "nodes not of this stack: ['/rviz']",
                 id='rviz_running'),
    pytest.param(LIVE + ['/rviz'], RVIZ_STARTED + RVIZ_DIED,
                 "launched processes have exited: ['rviz2']", id='rviz_exited'),
])
def test_anything_but_this_stack_and_its_probe_fails(names, log, message):
    with pytest.raises(AssertionError, match=message.replace('[', r'\[').replace(']', r'\]')):
        check(names, STARTED + SPAWNER_DONE + log)


# ---- the stack's input player: one more expected node, and only one ----

class _PlayerProbe(_Probe):
    """A probe of a stack whose fixture started an input player."""

    domain = 'a domain of this test'


def test_the_input_player_is_expected_once_when_the_fixture_started_one(monkeypatch):
    monkeypatch.setitem(stack_helpers._PLAYERS, _PlayerProbe.domain, object())
    launch = _Launch(STARTED + SPAWNER_DONE)
    assert_stack_nodes(_PlayerProbe(LIVE + [INPUT_NODE]), launch, MOCK_STACK_NODES, settle=0.0)
    with pytest.raises(AssertionError, match='teleop_test_input'):      # it has gone
        assert_stack_nodes(_PlayerProbe(LIVE), launch, MOCK_STACK_NODES, settle=0.0)
    with pytest.raises(AssertionError, match='teleop_test_input'):      # a second one
        assert_stack_nodes(_PlayerProbe(LIVE + [INPUT_NODE] * 2), launch, MOCK_STACK_NODES,
                           settle=0.0)
    with pytest.raises(AssertionError, match='teleop_test_input'):      # none was started
        check(LIVE + [INPUT_NODE], STARTED + SPAWNER_DONE)


# ---- the lead-in: a calibrate press under a session of its own, nothing else ----

def test_the_lead_in_is_the_calibrate_press_of_arm_only_under_its_own_session(tmp_path):
    path, count = lead_in_file(tmp_path)
    with open(path, encoding='utf-8') as handle:
        frames = [json.loads(line)['msg'] for line in handle]
    assert count == len(frames) == 63                           # seq 0-62: the arm press is 63-80
    assert [f['seq'] for f in frames] == list(range(63))
    assert {f['session_id'] for f in frames} == {LEAD_IN_SESSION}
    with open(os.path.join(SERVO_FIXTURES, 'arm_only.jsonl'), encoding='utf-8') as handle:
        assert json.loads(handle.readline())['msg']['session_id'] != LEAD_IN_SESSION
    pressed = [[i for i, b in enumerate(f['controllers']['right']['buttons']) if b['p']]
               for f in frames]
    assert [f['seq'] for f, p in zip(frames, pressed) if p == [4]] == list(range(27, 45))
    assert all(p in ([], [4]) for p in pressed)         # calibrate (A) only: no arm, no grip
    assert pressed[0] == [] and pressed[-1] == []       # released at both ends: it can loop
    assert all(max(f['controllers']['right']['axes']) == 0.0 for f in frames)   # trigger open


def _state(**fields):
    base = {'armed': False, 'clutch_engaged': False, 'input_seq': 0, 'input_age': 0.01,
            'hold_reason': TeleopState.HOLD_NOT_ARMED, 'calibrated': False,
            'calibration_session_id': 0}
    return SimpleNamespace(**{**base, **fields})


def test_only_a_calibration_under_the_lead_in_session_acknowledges_it():
    assert not lead_in_acknowledged([])
    assert not lead_in_acknowledged([_state(input_seq=30)])     # frames arrive, not calibrated
    # A calibration left from a fixture's session, valid or not, is not the lead-in's.
    assert not lead_in_acknowledged([_state(calibrated=True, calibration_session_id=195939070)])
    assert not lead_in_acknowledged([_state(calibrated=False,
                                            calibration_session_id=LEAD_IN_SESSION)])
    assert lead_in_acknowledged([_state(), _state(calibrated=True,
                                                  calibration_session_id=LEAD_IN_SESSION)])


def test_a_replay_is_refused_without_an_acknowledged_input_player(monkeypatch):
    env = {'ROS_DOMAIN_ID': 'a domain of this test'}
    with pytest.raises(AssertionError, match='no acknowledged input player'):
        replay(env, 'arm_only', None)
    monkeypatch.setitem(stack_helpers._PLAYERS, env['ROS_DOMAIN_ID'],
                        SimpleNamespace(acknowledged=False))     # started, lead-in not yet seen
    with pytest.raises(AssertionError, match='no acknowledged input player'):
        replay(env, 'arm_only', None)


# ---- a recording has ended when the teleop node sees its input stop ----

def test_a_finished_recording_has_ended_once_the_node_reports_its_input_stopped():
    state = _state(input_age=0.03, input_seq=269)
    events = [{'event': 'done', 'id': 1, 'records': 270, 'passes': 1, 'stopped': False,
               'failed': False, 'loop': False, 't': stack_helpers.time.time()}]
    arrived = SimpleNamespace(age=0.01)                     # the newest state is new
    player = SimpleNamespace(events_of=lambda ident: events, running=lambda: True,
                             probe=SimpleNamespace(state=lambda: state,
                                                   state_age=lambda: arrived.age))
    ended = stack_helpers.Replay(player, 1, 'arm_only.jsonl', None)
    assert ended.returncode == 0                            # every record was sent ...
    assert ended.wait(0.05) is None                         # ... but the node still has input
    state.input_age = 0.12                                  # beyond the gate's 0.1 s
    assert ended.wait(0.05) == 0
    state.input_age = 0.03                                  # (the next recording's input)
    assert ended.wait(0.05) == 0                            # ended stays ended
    # Input that never stops after a recording's end is somebody else's: say so, in time.
    events[0]['t'] -= 3.0
    stuck = stack_helpers.Replay(player, 1, 'arm_only.jsonl', None)
    with pytest.raises(AssertionError, match='still reports fresh input'):
        stuck.wait(0.05)
    # ... unless that fresh input is only the last state the probe ever got: then the teleop
    # node or the probe stopped, and the message says that instead.
    arrived.age = 2.4
    with pytest.raises(AssertionError, match='no teleop state has reached the probe for 2.4 s'):
        stack_helpers.Replay(player, 1, 'arm_only.jsonl', None).wait(0.05)
    arrived.age = 0.01
    # A stopped recording, a failed one and a looped one do not wait for anything.
    for change, code in (({'stopped': True}, 2), ({'failed': True}, 1), ({'loop': True}, 0)):
        events[0] = {**events[0], 'stopped': False, 'failed': False, 'loop': False, **change}
        assert stack_helpers.Replay(player, 1, 'x.jsonl', None).wait(0.05) == code
    # Without a probe (the player on its own) the last record is the end.
    player.probe = None
    events[0] = {**events[0], 'stopped': False, 'failed': False, 'loop': False}
    assert stack_helpers.Replay(player, 1, 'x.jsonl', None).wait(0.05) == 0


def test_stopping_a_recording_waits_for_the_player_only_and_checks_nothing():
    # A recording that ends by itself just as it is stopped: every record sent, and the node
    # still reporting fresh input from something else. stop() runs in finally blocks, so it
    # must not fail there in place of the test.
    state = _state(input_age=0.03, input_seq=269)
    events, sent = [], []

    def send(**command):
        sent.append(command)
        events.append({'event': 'done', 'id': 1, 'records': 270, 'passes': 1, 'stopped': False,
                       'failed': False, 'loop': False, 't': stack_helpers.time.time() - 3.0})

    player = SimpleNamespace(events_of=lambda ident: events, running=lambda: True, send=send,
                             output=lambda: '', probe=SimpleNamespace(
                                 state=lambda: state, state_age=lambda: 0.01))
    playing = stack_helpers.Replay(player, 1, 'arm_only.jsonl', None)
    playing.stop()
    assert sent == [{'cmd': 'stop', 'id': 1}] and playing.returncode == 0
    with pytest.raises(AssertionError, match='still reports fresh input'):
        playing.wait(0.05)                                  # the check is wait's, as before


# ---- the helpers that replay: what is repeated, and what never is ----

class _Played:
    """A recording that has been played to its end."""

    def wait(self, timeout):
        return 0

    def stop(self):
        pass

    def output(self):
        return ''


class _ScriptedProbe:
    """
    A probe whose stack answers the n-th replay with the n-th list of states.

    ``late`` says what was late in the quarter second before a run's release: 'source' (the
    input source sent nothing for 0.15 s), 'joints' (no joint state reached the probe for
    0.12 s), 'probe' (nothing at all reached the probe for 0.12 s: the probe paused), or None.
    """

    def __init__(self, runs, late=None):
        self.runs, self.replays, self.disarms, self.late = runs, 0, 0, late

    def disarm(self):
        self.disarms += 1

    def states_since(self, start):
        return self.runs[self.replays - 1]

    def _all_states(self, start):
        return [(start + 0.02 * i, s) for i, s in enumerate(self.runs[self.replays - 1], 1)]

    def timed_states_since(self, since):
        # Scripted from the replay's start, which is the first time asked for.
        self.start = getattr(self, 'start', since) if self.replays == getattr(
            self, '_run', None) else since
        self._run = self.replays
        states = self._all_states(self.start)
        release = next((t for t, s in states if not s.clutch_engaged and t > states[0][0]), None)
        self.release = release
        if self.late == 'probe' and release is not None:
            states = [(t, s) for t, s in states if not release - 0.25 < t < release - 0.13]
        return [(t, s) for t, s in states if t >= since]

    def input_stamps_since(self, since):
        # 90 Hz up to the release, stamped by the source 1000 s off the probe's clock.
        times = [since + k / 90.0 for k in range(int((self.release + 0.3 - since) * 90))]
        if self.late == 'source':
            times = [t for t in times if not self.release - 0.25 < t < self.release - 0.10]
        return [(t, t + 1000.0) for t in times]

    def timed_joints_since(self, since):
        times = [since + k / 150.0 for k in range(int((self.release - since) * 150))]
        if self.late in ('joints', 'probe'):
            times = [t for t in times if not self.release - 0.25 < t < self.release - 0.13]
        return [(t, None) for t in times]


@pytest.fixture
def scripted(monkeypatch):
    """Return a function making a scripted probe; replays are counted, nothing is slept."""
    made = []

    def make(*runs, late=None):
        made.append(_ScriptedProbe(list(runs), late))
        return made[-1]

    def played(env, fixture, log_path, start_delay=3.0):
        made[-1].replays += 1
        return _Played()

    monkeypatch.setattr(stack_helpers, 'replay', played)
    monkeypatch.setattr(stack_helpers.time, 'sleep', lambda seconds: None)
    return make


ENGAGED = [_state(armed=True, clutch_engaged=True, hold_reason=TeleopState.HOLD_NONE)] * 100
STALE = _state(armed=True, hold_reason=TeleopState.HOLD_INPUT_STALE, input_age=0.15)
POSE_STALE = _state(armed=True, hold_reason=TeleopState.HOLD_TOOL_POSE_STALE)
RELEASED = _state(armed=True, hold_reason=TeleopState.HOLD_CLUTCH_RELEASED)
HALTED = _state(armed=True, hold_reason=TeleopState.HOLD_SERVO_HALTED)


POSE_ABSENT = _state(armed=True, hold_reason=TeleopState.HOLD_POSE_ABSENT, input_age=0.01)


@pytest.mark.parametrize('release, late', [(STALE, 'source'), (POSE_STALE, 'joints')],
                         ids=['the_source_paused', 'the_joint_states_paused'])
def test_only_data_late_outside_the_teleop_node_is_repeated_and_it_is_printed(
        scripted, capsys, release, late):
    # 2 s engaged, then released on stale data where 2.8 s were expected, and the probe can
    # show it: the source sent nothing for 0.15 s (input age 0.15 s), or no joint state
    # arrived for 0.12 s while teleop states kept coming.
    probe = scripted(ENGAGED + [release], ENGAGED + ENGAGED + [RELEASED], late=late)
    start, discarded = replay_with_clean_input(probe, {}, 'log', 'clutch_line', 2.8)
    assert probe.replays == 2 and len(discarded) == 1
    assert f'hold_reason {release.hold_reason}' in discarded[0]
    assert 'late on its way' in discarded[0]
    assert 'discarded replay of clutch_line: attempt 1' in capsys.readouterr().out


@pytest.mark.parametrize('release, late', [
    pytest.param(STALE, None, id='input_stale_but_sent_steadily'),
    pytest.param(STALE, 'probe', id='input_stale_and_only_the_probe_paused'),
    pytest.param(STALE, 'joints', id='input_stale_and_something_else_was_late'),
    pytest.param(POSE_STALE, None, id='tool_pose_stale_but_joint_states_steady'),
    pytest.param(POSE_STALE, 'probe', id='tool_pose_stale_and_the_probe_paused'),
    pytest.param(POSE_ABSENT, 'source', id='pose_absent_is_never_shown'),
])
def test_a_stale_release_not_shown_late_on_its_way_is_the_stacks_own_and_not_repeated(
        scripted, capsys, release, late):
    probe = scripted(ENGAGED + [release], ENGAGED + ENGAGED + [RELEASED], late=late)
    start, discarded = replay_with_clean_input(probe, {}, 'log', 'clutch_line', 2.8)
    assert probe.replays == 1 and discarded == []
    printed = capsys.readouterr().out
    assert 'not repeated' in printed and "the stack's own" in printed


@pytest.mark.parametrize('run', [
    pytest.param(ENGAGED + [HALTED], id='released_early_for_another_reason'),
    pytest.param(ENGAGED + [RELEASED], id='grip_released_early'),
    pytest.param([_state(armed=True, hold_reason=TeleopState.HOLD_NOT_CALIBRATED)] * 50,
                 id='armed_never_engaged'),
    pytest.param([_state(input_seq=5)] * 50, id='nothing_armed'),
    pytest.param(ENGAGED + ENGAGED + [STALE], id='late_data_after_the_expected_time'),
])
def test_any_other_outcome_is_returned_at_once_for_the_test_to_judge(scripted, capsys, run):
    probe = scripted(run, ENGAGED + ENGAGED + [RELEASED], late='source')
    start, discarded = replay_with_clean_input(probe, {}, 'log', 'clutch_line', 2.8)
    assert probe.replays == 1 and discarded == []
    assert capsys.readouterr().out == ''


def test_late_data_is_repeated_at_most_twice_then_returned_with_a_line(scripted, capsys):
    probe = scripted(*[ENGAGED + [STALE]] * 3, late='source')
    start, discarded = replay_with_clean_input(probe, {}, 'log', 'clutch_line', 2.8)
    assert probe.replays == 3 and len(discarded) == 2
    assert 'not repeated again after 3 attempts' in capsys.readouterr().out


def test_a_pause_of_the_source_shorter_than_the_gate_saw_does_not_count():
    # The gate saw no input for 150 ms; the source's own stamps show 60 ms at most.
    stamps = [(100.0 + k / 90.0, 5000.0 + k / 90.0) for k in range(90) if not 40 <= k < 45]
    probe = SimpleNamespace(input_stamps_since=lambda since: stamps)
    late, text = stack_helpers.late_on_the_way(probe, STALE, 101.0)
    assert not late and 'sent two frames 67 ms apart' in text, text
    # A pause as long as the gate saw (less one cycle) does.
    stamps = [(100.0 + k / 90.0, 5000.0 + k / 90.0) for k in range(90) if not 40 <= k < 52]
    late, text = stack_helpers.late_on_the_way(probe, STALE, 101.0)
    assert late and '144 ms apart' in text, text


def test_arming_is_tried_once_whatever_comes_of_it(scripted):
    seqs = [683, 683, 0, 2, 40, 80, 120, 269]
    armed = [_state(input_seq=q, armed=q >= 80) for q in seqs]
    refused = [_state(input_seq=q) for q in seqs]               # the presses arrived; not armed
    for run in (armed, refused):
        probe = scripted(run, armed)
        start, discarded = arm_by_replay(probe, {}, 'log')
        assert probe.replays == 1 and probe.disarms == 1 and discarded == []
    probe = scripted(armed)
    arm_by_replay(probe, {}, 'log', disarm=False)
    assert probe.replays == 1 and probe.disarms == 0


@pytest.mark.parametrize('seqs', [[683, 683, 683], [683, 144, 200, 269], []])
def test_presses_that_never_arrived_fail_as_the_inputs_failure_not_the_stacks(scripted, seqs):
    probe = scripted([_state(input_seq=q) for q in seqs], [_state(armed=True)])
    with pytest.raises(AssertionError, match='the presses never arrived'):
        arm_by_replay(probe, {}, 'log')
    assert probe.replays == 1                                   # and nothing was tried again


# ---- a new service client: a harmless request until the server answers it ----

class _Client:
    """A client whose server answers from the n-th request on (never: n = 0)."""

    srv_name = '/some/service'

    def __init__(self, answers_from, available=True):
        self.answers_from, self.available = answers_from, available
        self.requests, self.dropped = [], []

    def wait_for_service(self, timeout_sec):
        return self.available

    def call_async(self, request):
        self.requests.append(request)
        answered = bool(self.answers_from) and len(self.requests) >= self.answers_from
        return SimpleNamespace(done=lambda: answered)

    def remove_pending_request(self, future):
        self.dropped.append(future)


def test_priming_asks_the_harmless_request_until_it_is_answered_and_says_so(capsys):
    client = _Client(answers_from=1)
    assert prime(client, 'harmless', timeout=1.0, each=0.05) == 1
    assert client.requests == ['harmless'] and capsys.readouterr().out == ''
    client = _Client(answers_from=3)                # the first two answers are lost
    assert prime(client, 'harmless', timeout=5.0, each=0.05) == 3
    assert client.requests == ['harmless'] * 3 and len(client.dropped) == 2
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 2 and all('/some/service: no answer to harmless request' in line
                                     for line in printed)


def test_priming_fails_in_bounded_time_when_no_answer_ever_comes():
    client = _Client(answers_from=0)
    with pytest.raises(AssertionError, match='/some/service answered none of'):
        prime(client, 'harmless', timeout=0.3, each=0.05)
    assert 2 <= len(client.requests) <= 7
    with pytest.raises(AssertionError, match='no service /some/service'):
        prime(_Client(answers_from=1, available=False), 'harmless', timeout=0.1)


# ---- when a press counts as never arrived (the check behind that failure) ----

def test_the_presses_of_arm_only_are_seq_27_to_80():
    # A (calibrate) 27-44, B (arm) 63-80.
    assert press_frames(os.path.join(SERVO_FIXTURES, 'arm_only.jsonl')) == (27, 80)


def test_a_press_is_lost_only_when_the_first_frame_is_well_after_the_last_press():
    # The previous stream's newest frame (683) first, then the replay's own frames. The last
    # pressed frame is 80; with the margin (10) only a first frame after 90 means "never arrived".
    assert press_never_arrived([], 80)                          # nothing at all
    assert press_never_arrived([683, 683, 683], 80)             # no frame of the replay
    assert press_never_arrived([683, 144, 145], 80)             # the Gazebo failure: seq 144
    assert press_never_arrived([683, 91, 92], 80)               # just beyond the margin
    assert not press_never_arrived([683, 90, 91], 80)           # within it: may have arrived
    assert not press_never_arrived([683, 60, 61], 80)           # A lost, B can arm (same session)
    assert not press_never_arrived([683, 0, 1, 2], 80)          # from the start
    assert not press_never_arrived([683, 30, 31], 80)           # mid-press: the level registers
    assert not press_never_arrived([269, 28, 29], 80)           # 27 may have come between cycles
    assert not press_never_arrived([683, 20, 150], 80)          # a gap later: still arrived
    assert not press_never_arrived([5, 6, 7], 80)               # no previous stream


# ---- the environment of a stack ----

def test_each_child_environment_has_its_own_home_for_the_spawners_lock(monkeypatch):
    monkeypatch.setattr(stack_helpers, '_LOGS', [])
    monkeypatch.setattr(stack_helpers, 'TEMP_LOG_DIRS', [])
    monkeypatch.delenv('PYTEST_XDIST_WORKER', raising=False)
    monkeypatch.setenv('ROS_HOME', '/somewhere/shared')
    try:
        first = stack_helpers.child_environment(offset=1)
        second = stack_helpers.child_environment(offset=1)
        homes = [first['ROS_HOME'], second['ROS_HOME']]
        assert homes[0] != homes[1] and '/somewhere/shared' not in homes
        assert all(os.path.isdir(home) and not os.listdir(home) for home in homes)
        # Removed with the session's other temporary directories, and not the log directory.
        assert set(homes) <= set(stack_helpers.TEMP_LOG_DIRS)
        assert first['ROS_LOG_DIR'] != first['ROS_HOME']
    finally:
        for path in stack_helpers.TEMP_LOG_DIRS:
            shutil.rmtree(path, ignore_errors=True)


# ---- logs kept when a test fails ----

def test_keep_logs_copies_what_the_module_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(stack_helpers, '_LOGS', [])
    launch = tmp_path / 'launch.log'
    launch.write_text('launch output\n')
    ros_logs = tmp_path / 'ros_log'
    (ros_logs / 'run1').mkdir(parents=True)
    (ros_logs / 'run1' / 'teleop_node.log').write_text('node log\n')
    other = tmp_path / 'other.log'
    other.write_text('another module\n')
    monkeypatch.setenv('PYTEST_CURRENT_TEST', 'test/test_mock_x.py::test_a (setup)')
    stack_helpers.register_log(str(launch))
    stack_helpers.register_log(str(ros_logs))
    monkeypatch.setenv('PYTEST_CURRENT_TEST', 'test/test_mock_y.py::test_b (setup)')
    stack_helpers.register_log(str(other))
    target = keep_logs('test/test_mock_x.py', 'test_a', str(tmp_path / 'kept'))
    kept = sorted(os.path.relpath(os.path.join(d, f), target)
                  for d, _, files in os.walk(target) for f in files)
    assert kept == ['00_launch.log', os.path.join('01_ros_log', 'run1', 'teleop_node.log')]
    assert os.path.basename(target).endswith('test_mock_x.py__test_a')


def _mock_reports(periods, speed, start=0.0):
    """Return joint states as mock execution reports them: each update, the command before."""
    stamps, commands = [start], [0.0]
    for period in periods:
        stamps.append(stamps[-1] + period)
        commands.append(commands[-1] + speed * period)      # commanded over this period
    # The state stamped at update k carries the command of update k - 1.
    return [(stamps[k], [commands[k - 1], 0.0]) for k in range(1, len(stamps))]


def _own_stamp_speed(samples):
    return max(abs(q1[0] - q0[0]) / (s1 - s0) for (s0, q0), (s1, q1) in zip(samples, samples[1:]))


def test_mock_execution_speed_is_each_step_over_the_period_it_was_commanded_in():
    # A steady 1.9 rad/s with the uneven periods measured on the mock stack (ms), one doubled.
    periods = [p * 1e-3 for p in (6.67, 7.04, 6.37, 6.25, 7.08, 13.7, 1.95, 6.67, 6.66, 6.9)]
    samples = _mock_reports(periods, 1.9)
    assert stack_helpers.commanded_period_speed(samples) == pytest.approx(1.9, abs=1e-9)
    # The same states by each step's own stamp difference: the ratio of neighbouring periods.
    assert _own_stamp_speed(samples) > 2.05


def test_mock_execution_speed_still_shows_a_real_overshoot():
    # Ten updates at 1.9 rad/s, then twenty at 2.3 rad/s, at an even period.
    stamps, commands = [0.0], [0.0]
    for k in range(30):
        stamps.append(stamps[-1] + 6.67e-3)
        commands.append(commands[-1] + (1.9 if k < 10 else 2.3) * 6.67e-3)
    samples = [(stamps[k], [commands[k - 1], 0.0]) for k in range(1, len(stamps))]
    assert stack_helpers.commanded_period_speed(samples) == pytest.approx(2.3, abs=1e-9)


def test_mock_execution_speed_reads_too_high_never_too_low_when_a_state_is_missing():
    samples = _mock_reports([6.67e-3] * 20, 1.9)
    del samples[8]                              # one joint state never reached the probe
    assert stack_helpers.commanded_period_speed(samples) > 2.0 * 1.9 - 1e-6


def test_mock_execution_speed_needs_three_states_and_skips_a_repeated_stamp():
    assert stack_helpers.commanded_period_speed([]) == 0.0
    assert stack_helpers.commanded_period_speed([(0.0, [0.0]), (0.01, [0.5])]) == 0.0
    repeated = [(0.0, [0.0]), (0.0, [0.0]), (0.01, [0.5]), (0.02, [0.5 + 0.019])]
    assert stack_helpers.commanded_period_speed(repeated) == pytest.approx(1.9)


# ---- a release when a process joins: the stall and nothing else ----

BY_INPUT = TeleopState.HOLD_INPUT_STALE
BY_POSE = TeleopState.HOLD_TOOL_POSE_STALE
BY_SERVO = TeleopState.HOLD_SERVO_NOT_READY
INPUT, TELEOP = stack_helpers.INPUT_PAUSE, stack_helpers.TELEOP_PAUSE
JOINT, SERVO = stack_helpers.JOINT_PAUSE, stack_helpers.SERVO_PAUSE
QUIET = {INPUT: (0.012, []), TELEOP: (0.021, []), JOINT: (0.008, []), SERVO: (0.020, [])}
RELEASE = 102.4             # with the process started at 100.0


def paused(stream, length, began=None, taken=None):
    """Return QUIET with one pause of ``stream``: begun 0.2 s before the release by default."""
    began = RELEASE - 0.2 if began is None else began
    taken = length if taken is None else taken
    return {**QUIET, stream: (length, [(length, began, began + taken, taken)])}


def test_a_stale_release_after_the_join_with_a_pause_of_stall_length_is_the_join_stall():
    # The forms seen: the input source paused, the teleop node, the robot side.
    for reason, stream, gap in (
            (BY_INPUT, INPUT, 0.322), (BY_INPUT, TELEOP, 0.341), (BY_SERVO, TELEOP, 0.334),
            (BY_POSE, JOINT, 0.333), (BY_POSE, TELEOP, 0.34), (BY_SERVO, SERVO, 0.66),
            (BY_SERVO, JOINT, 0.34),                # MoveIt Servo stops without joint states
            (BY_INPUT, INPUT, 1.0)):                # three sleeps in a row
        stalled, text = stack_helpers.join_stall(100.0, RELEASE, reason, paused(stream, gap))
        assert stalled, text
        assert '2.40 s after the process was started' in text and stream in text
        assert f'{gap * 1000.0:.0f} ms' in text


@pytest.mark.parametrize('reason', [
    TeleopState.HOLD_POSE_ABSENT, TeleopState.HOLD_HEAD_ABSENT, TeleopState.HOLD_SERVO_HALTED,
    TeleopState.HOLD_CLUTCH_RELEASED, TeleopState.HOLD_NOT_OWNER, TeleopState.HOLD_SOLVER,
    TeleopState.HOLD_INPUT_JUMP, TeleopState.HOLD_SEQ_BACKWARDS, TeleopState.HOLD_NOT_ARMED])
def test_a_release_for_any_other_reason_is_not_the_join_stall(reason):
    stalled, text = stack_helpers.join_stall(100.0, RELEASE, reason, paused(INPUT, 0.33))
    assert not stalled and f'hold_reason {reason} is not what a stalled node causes' in text


def test_a_stale_release_before_the_join_or_long_after_it_is_not_the_join_stall():
    stalled, text = stack_helpers.join_stall(100.0, 99.6, BY_INPUT,
                                             paused(INPUT, 0.33, began=99.4))
    assert not stalled and '0.40 s before the process was started' in text
    late = 100.0 + stack_helpers.JOIN_WINDOW_S + 0.5
    stalled, text = stack_helpers.join_stall(100.0, late, BY_INPUT,
                                             paused(INPUT, 0.33, began=late - 0.2))
    assert not stalled and 'after the process was started, not within' in text


def test_a_stale_release_without_a_pause_of_stall_length_is_not_the_join_stall():
    # Nothing paused; a pause too short for a stall and one too long are no pauses at all.
    short = stack_helpers.sender_pauses([(0.0, 0.0), (0.24, 0.24), (0.25, 0.25)])
    long = stack_helpers.sender_pauses([(0.0, 0.0), (1.5, 1.5), (1.51, 1.51)])
    assert short == (pytest.approx(0.24), []) and long == (pytest.approx(1.5), [])
    for pauses in (QUIET, {**QUIET, INPUT: short}, {**QUIET, INPUT: long}):
        stalled, text = stack_helpers.join_stall(100.0, RELEASE, BY_INPUT, pauses)
        assert not stalled and 'no pause of stall length' in text
        assert f'{INPUT} for' in text       # the longest of each stream is listed


def test_a_pause_of_a_stream_that_does_not_cause_the_reason_is_not_the_join_stall():
    # A stale input is caused by the input source or the teleop node, not by the robot side;
    # a stale tool pose not by the input source or MoveIt Servo; and so on.
    for reason, stream in ((BY_INPUT, JOINT), (BY_INPUT, SERVO), (BY_POSE, INPUT),
                           (BY_POSE, SERVO), (BY_SERVO, INPUT)):
        stalled, text = stack_helpers.join_stall(100.0, RELEASE, reason, paused(stream, 0.33))
        assert not stalled, (reason, stream)
        assert f'{stream} for 330 ms, which does not cause hold_reason {reason}' in text


def test_a_pause_that_began_after_the_release_or_ended_long_before_it_is_not_its_cause():
    after = paused(INPUT, 0.33, began=RELEASE + 0.3)
    stalled, text = stack_helpers.join_stall(100.0, RELEASE, BY_INPUT, after)
    assert not stalled and 'but only from 0.30 s after the release' in text
    before = paused(INPUT, 0.33, began=RELEASE - 1.4)       # ended 1.07 s before
    stalled, text = stack_helpers.join_stall(100.0, RELEASE, BY_INPUT, before)
    assert not stalled and 'which had ended 1.07 s before the release' in text
    # During the pause, and up to half a second after its end, it is.
    for began in (RELEASE - 0.33, RELEASE - 0.05, RELEASE - 0.33 - 0.45):
        assert stack_helpers.join_stall(100.0, RELEASE, BY_INPUT,
                                        paused(INPUT, 0.33, began=began))[0], began


def test_a_gap_in_the_stamps_that_the_probe_made_itself_is_not_the_join_stall():
    # The probe paused for 0.6 s and kept the newest 0.25 s of the stream: the stamps show a
    # gap of 0.35 s, taken 0.6 s apart. Or it kept the oldest: taken together.
    for taken in (0.6, 0.01):
        stalled, text = stack_helpers.join_stall(100.0, RELEASE, BY_SERVO,
                                                 paused(SERVO, 0.35, taken=taken))
        assert not stalled, taken
        assert f'the probe took the two messages {taken * 1000.0:.0f} ms apart' in text
        assert "not shown to be the sender's" in text
    # Within the slack the pause is the sender's.
    assert stack_helpers.join_stall(100.0, RELEASE, BY_SERVO,
                                    paused(SERVO, 0.35, taken=0.38))[0]


def test_the_longest_shown_pause_is_named_when_more_than_one_fits():
    both = {**QUIET, INPUT: paused(INPUT, 0.33)[INPUT], TELEOP: paused(TELEOP, 0.66)[TELEOP]}
    stalled, text = stack_helpers.join_stall(100.0, RELEASE, BY_INPUT, both)
    assert stalled and f'{TELEOP} for 660 ms' in text


def test_pauses_are_between_two_messages_in_a_row_by_their_senders_stamps():
    from types import SimpleNamespace as NS

    def stamp(seconds):
        return NS(sec=int(seconds), nanosec=int(round((seconds - int(seconds)) * 1e9)))

    class _Probe:
        lock = __import__('threading').Lock()
        # The probe paused from 9.04 to 9.40 and lost nothing: every arrival shows it, no
        # stamp does, except the input's, whose source paused itself (50.01 to 50.33).
        stamped_joints = [(a, stamp(s), None) for a, s in (
            (9.00, 70.000), (9.04, 70.007), (9.40, 70.014), (9.40, 70.021), (9.41, 70.028))]

        def input_stamps_since(self, t):
            return [(a, s) for a, s in ((9.00, 50.00), (9.01, 50.01), (9.33, 50.33),
                                        (9.34, 50.34), (9.35, 50.35)) if a >= t]

        def timed_states_since(self, t):
            return [(a, NS(header=NS(stamp=stamp(s)))) for a, s in (
                (9.00, 60.00), (9.04, 60.02), (9.40, 60.04), (9.40, 60.06)) if a >= t]

    pauses = stack_helpers.pauses_around(
        _Probe(), 9.3, servo_output=[(7.0, 80.0), (9.1, 82.10), (9.12, 82.12), (12.0, 85.0)])
    longest, found = pauses[INPUT]
    assert longest == pytest.approx(0.32) and len(found) == 1
    assert found[0] == pytest.approx((0.32, 9.01, 9.33, 0.32))
    # The probe's own pause is nobody's pause.
    assert pauses[TELEOP] == (pytest.approx(0.02), [])
    assert pauses[JOINT] == (pytest.approx(0.007), [])
    # Only what arrived around the release counts; a stream that ended is no pause.
    assert pauses[SERVO] == (pytest.approx(0.02), [])
    stalled, text = stack_helpers.join_stall(7.5, 9.3, BY_INPUT, pauses)
    assert stalled and f'{INPUT} for 320 ms' in text
