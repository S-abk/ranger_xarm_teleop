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
    """A probe whose stack answers the n-th replay with the n-th list of states."""

    def __init__(self, runs, steady=False):
        self.runs, self.replays, self.disarms, self.steady = runs, 0, 0, steady

    def disarm(self):
        self.disarms += 1

    def states_since(self, start):
        return self.runs[self.replays - 1]

    def timed_states_since(self, start):
        return [(start + 0.02 * i, s) for i, s in enumerate(self.runs[self.replays - 1], 1)]

    def _arrivals(self, begin):
        # The half second from ``begin`` at 90 Hz; unless steady, 0.12 s of it are missing.
        times = [begin + k / 90.0 for k in range(46)]
        return [t for t in times if self.steady or not 0.30 < t - begin < 0.42]

    def inputs_since(self, begin):
        return self._arrivals(begin)

    def timed_joints_since(self, begin):
        return [(t, None) for t in self._arrivals(begin)]


@pytest.fixture
def scripted(monkeypatch):
    """Return a function making a scripted probe; replays are counted, nothing is slept."""
    made = []

    def make(*runs, steady=False):
        made.append(_ScriptedProbe(list(runs), steady))
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


@pytest.mark.parametrize('late', [STALE, POSE_STALE], ids=['input', 'joint_states'])
def test_only_data_late_on_its_way_is_repeated_and_it_is_printed(scripted, capsys, late):
    # 2 s engaged, then released on stale data where 2.8 s were expected, and the probe saw
    # the same stream 0.12 s late too: late on its way.
    probe = scripted(ENGAGED + [late], ENGAGED + ENGAGED + [RELEASED])
    start, discarded = replay_with_clean_input(probe, {}, 'log', 'clutch_line', 2.8)
    assert probe.replays == 2 and len(discarded) == 1
    assert f'hold_reason {late.hold_reason}' in discarded[0] and 'late on its way' in discarded[0]
    assert 'discarded replay of clutch_line: attempt 1' in capsys.readouterr().out


@pytest.mark.parametrize('late', [STALE, POSE_STALE], ids=['input', 'joint_states'])
def test_a_stale_release_with_a_steady_stream_is_the_stacks_own_and_not_repeated(
        scripted, capsys, late):
    # The same release, but the stream reached the probe steadily: the teleop node was late.
    probe = scripted(ENGAGED + [late], ENGAGED + ENGAGED + [RELEASED], steady=True)
    start, discarded = replay_with_clean_input(probe, {}, 'log', 'clutch_line', 2.8)
    assert probe.replays == 1 and discarded == []
    printed = capsys.readouterr().out
    assert 'not repeated' in printed and 'the teleop node was late itself' in printed


@pytest.mark.parametrize('run', [
    pytest.param(ENGAGED + [HALTED], id='released_early_for_another_reason'),
    pytest.param(ENGAGED + [RELEASED], id='grip_released_early'),
    pytest.param([_state(armed=True, hold_reason=TeleopState.HOLD_NOT_CALIBRATED)] * 50,
                 id='armed_never_engaged'),
    pytest.param([_state(input_seq=5)] * 50, id='nothing_armed'),
    pytest.param(ENGAGED + ENGAGED + [STALE], id='late_data_after_the_expected_time'),
])
def test_any_other_outcome_is_returned_at_once_for_the_test_to_judge(scripted, capsys, run):
    probe = scripted(run, ENGAGED + ENGAGED + [RELEASED])
    start, discarded = replay_with_clean_input(probe, {}, 'log', 'clutch_line', 2.8)
    assert probe.replays == 1 and discarded == []
    assert capsys.readouterr().out == ''


def test_late_data_is_repeated_at_most_twice_then_returned(scripted):
    probe = scripted(*[ENGAGED + [STALE]] * 3)
    start, discarded = replay_with_clean_input(probe, {}, 'log', 'clutch_line', 2.8)
    assert probe.replays == 3 and len(discarded) == 2


def test_a_stream_that_stopped_counts_as_a_gap():
    arrivals = [99.0 + k / 90.0 for k in range(136)]            # 99.0 to 100.5 s, then nothing
    probe = SimpleNamespace(inputs_since=lambda begin: [t for t in arrivals if t >= begin],
                            timed_joints_since=lambda begin: [(t, None) for t in arrivals
                                                              if t >= begin and t < 100.3])
    late = stack_helpers.late_on_the_way
    assert late(probe, TeleopState.HOLD_INPUT_STALE, 100.5) < 0.02          # steady to the end
    assert 0.19 < late(probe, TeleopState.HOLD_INPUT_STALE, 100.7) < 0.21   # stopped 0.2 s ago
    assert late(probe, TeleopState.HOLD_POSE_ABSENT, 100.7) > 0.19          # the same stream
    # A stale tool pose is judged by the joint states, which stopped at 100.3 s here.
    assert 0.19 < late(probe, TeleopState.HOLD_TOOL_POSE_STALE, 100.5) < 0.22
    assert late(probe, TeleopState.HOLD_TOOL_POSE_STALE, 100.25) < 0.02


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
