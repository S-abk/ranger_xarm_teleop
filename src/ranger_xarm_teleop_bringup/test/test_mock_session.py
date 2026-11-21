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
The session launcher (tools/teleop_session) on a mock stack.

One launcher session for most of the module, on its own ROS domain, with the input played
into the launcher's own relay by tools/fake_quest.py (the headset's path: relay, bridge). The
test's observer node joins before the launcher starts. Checked: the stack comes up disarmed
with MoveIt as owner (INV-1); its nodes and their parameters are those of the manual sequence
(INV-2); the launcher's probe has left before "stack ready" and nothing joins or leaves after
it (INV-6); a stop during an episode marks it interrupted and leaves nothing (INV-5); no label
reaches a log (INV-7); a second start is refused; a stop with the supervisor killed; the
launcher's refusals (INV-3).
"""

import glob
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time

import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, EpisodeState
from ranger_xarm_teleop_record import metadata as md
from ranger_xarm_teleop_session import procs
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.parameter import parameter_value_to_python
from stack_helpers import (child_environment, graph_nodes, MOCK_RECORD_NODES, prime, Process,
                           READY, register_log)
from test_mock_record import fixture_file, RecordProbe

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKOUT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TOOL = os.path.join(CHECKOUT, 'tools', 'teleop_session')
FAKE_QUEST = os.path.join(CHECKOUT, 'tools', 'fake_quest.py')
PORT = 18451                         # the relay's port for this module (not the person's 8443)
OFFSET = 19
PROBE_NODE = '/teleop_session_probe'
LAUNCHER_NODES = {**MOCK_RECORD_NODES, '/reset_node': 1, '/reset_watchdog': 1,
                  '/teleop_input_bridge': 1}
# The nodes whose parameters must be the manual launch's (every node of this repository).
COMPARED = ['/teleop_servo', '/arm_mux', '/servo_node', '/gripper_relay', '/operator_feedback',
            '/reset_node', '/reset_watchdog', '/episode_node', '/teleop_input_bridge']
PER_SESSION = {'/episode_node': {'session_dir'}}     # differs between any two sessions


class Launcher:
    """tools/teleop_session start, as a child process with its own state directory."""

    def __init__(self, env, logs, name, *args):
        self.env, self.logs = env, logs
        self.process = Process([sys.executable, TOOL, 'start', *args], env,
                               str(logs / f'{name}.out'))

    def state(self):
        return procs.read_state(procs.state_path(self.env)) or {}

    def wait_phase(self, phases, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.state().get('phase') in phases:
                return self.state()
            code = self.process.wait(0.0)
            if code is not None:
                raise AssertionError(f'the launcher ended (exit {code}):\n'
                                     + self.process.output()[-4000:])
            time.sleep(0.2)
        raise AssertionError(f'no phase {phases} within {timeout} s:\n'
                             + self.process.output()[-4000:])

    def output(self):
        return self.process.output()

    def stack_log(self):
        with open(os.path.join(self.state()['log_dir'], 'stack.log'), encoding='utf-8',
                  errors='replace') as handle:
            return handle.read()


def tool(env, *args, timeout=600.0):
    done = subprocess.run([sys.executable, TOOL, *args], env=env, capture_output=True,
                          text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return done.returncode, done.stdout + done.stderr


def environment(tmp, offset=OFFSET):
    env = child_environment(offset=offset)
    env[procs.STATE_DIR_VAR] = str(tmp / 'state')
    env['TELEOP_SESSION_NO_ADB'] = '1'
    return env


class StackLog:
    """The launcher's stack.log, for the node-set helpers (they read a launch's output)."""

    def __init__(self, launcher):
        self.launcher = launcher

    def output(self):
        return self.launcher.stack_log()


def read_parameters(probe, node):
    """Return every parameter of ``node`` as plain values."""
    lister = probe.node.create_client(ListParameters, f'{node}/list_parameters')
    getter = probe.node.create_client(GetParameters, f'{node}/get_parameters')
    prime(lister, ListParameters.Request(depth=0))          # new clients; reads only
    names = sorted(probe.call(lister, ListParameters.Request(depth=0), 20.0).result.names)
    prime(getter, GetParameters.Request(names=names[:1]))
    out = {}
    for name in names:      # one by one: a parameter declared without a value fails a batch
        values = probe.call(getter, GetParameters.Request(names=[name]), 20.0).values
        out[name] = parameter_value_to_python(values[0]) if values else '<not set>'
    return out


def stack_parameters(probe):
    return {node: {k: v for k, v in read_parameters(probe, node).items()
                   if k not in PER_SESSION.get(node, set())} for node in COMPARED}


def episode_recording(probe, since):
    return any(m.state == EpisodeState.STATE_RECORDING for m in probe.episode_since(since))


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('mock_session')
    data = tmp / 'data'
    labels = (f'task-{secrets.token_hex(4)}', f'operator-{secrets.token_hex(4)}')
    md.write_labels(str(data), *labels)
    env = environment(tmp)
    probe = RecordProbe(env)                    # joins before the launcher (INV-6)
    launcher = Launcher(env, tmp, 'launcher', '--profile', 'mock', '--no-rviz', '--yes',
                        '--keep-labels', '--data-dir', str(data), '--port', str(PORT))
    try:
        launcher.wait_phase(('stack_ready', 'session_ready'), 240.0)
        at_ready = graph_nodes(probe, settle=2.0)
        register_log(launcher.state()['log_dir'])
        yield {'probe': probe, 'launcher': launcher, 'env': env, 'tmp': tmp, 'data': data,
               'labels': labels, 'at_ready': at_ready, 'mark': launcher.state()['mark']}
    finally:
        if launcher.process.wait(0.0) is None:
            print(tool(env, 'stop')[1])
        probe.close()
        launcher.process.stop()
        print(launcher.output()[-6000:])


def test_the_stack_comes_up_disarmed_with_moveit_as_owner(session):
    probe, out = session['probe'], session['launcher'].output()
    probe.wait_for(lambda: probe.state() is not None, 30.0, 'the teleop state')
    assert not probe.state().armed
    assert probe.owner_value() == ArmOwner.OWNER_MOVEIT
    assert '[     ok] Teleop disarmed, MoveIt owns the arm' in out
    assert '[     ok] Recorder' in out and '[     ok] Reset node and watchdog' in out


def test_the_probe_had_left_before_stack_ready_and_the_nodes_are_the_manual_stacks(session):
    at_ready = session['at_ready']
    assert PROBE_NODE not in at_ready, at_ready
    expected = dict(LAUNCHER_NODES)
    seen = list(at_ready)
    seen.remove('/' + session['probe'].node.get_name())
    for pattern, count in expected.items():
        hits = [n for n in seen if re.fullmatch(pattern, n)]
        assert len(hits) == count, (pattern, hits)
        seen = [n for n in seen if n not in hits]
    assert seen == [], f'nodes not of the manual stack: {seen}'


def test_a_second_start_is_refused_while_the_session_runs(session):
    code, out = tool(session['env'], 'start', '--profile', 'mock', '--yes', timeout=60.0)
    name = session['launcher'].state()['session']
    assert code == 2, out
    assert f'refused: session {name}' in out and 'tools/teleop_session stop' in out
    assert session['launcher'].state()['session'] == name


def test_parameters_are_the_manual_launchs(session, tmp_path):
    """INV-2: every node of this repository has the manual launch's parameters."""
    probe = session['probe']
    launched = stack_parameters(probe)
    session['parameters'] = launched
    assert all(launched.values())


def test_stop_during_an_episode_marks_it_interrupted_and_leaves_nothing(session):
    probe, env, launcher = session['probe'], session['env'], session['launcher']
    probe.move_to(READY)
    fixture = fixture_file(str(session['tmp']), 'episodes_ten')
    start = time.monotonic()
    quest = Process([sys.executable, FAKE_QUEST, fixture, '--url', f'ws://127.0.0.1:{PORT}/ws'],
                    env, str(session['tmp'] / 'fake_quest.log'))
    try:
        probe.wait_for(lambda: episode_recording(probe, start), 60.0, 'an episode recording')
        launcher.wait_phase(('session_ready',), 30.0)
        before_stop = graph_nodes(probe, settle=0.5)
        assert before_stop == session['at_ready'], 'a node joined or left after stack ready'
        state = launcher.state()
        code, out = tool(env, 'stop')
    finally:
        quest.stop()
    assert code == 0, out
    assert 'STOP REPORT' in out and 'PASS' in out and '1 interrupted' in out, out
    assert launcher.process.wait(60.0) == 0, launcher.output()[-3000:]
    sidecars = sorted(glob.glob(os.path.join(state['session_dir'], 'episodes', '*.json')))
    with open(sidecars[-1], encoding='utf-8') as handle:
        assert json.load(handle)['episode']['status'] == 'interrupted'
    assert procs.marked(state['mark']) == []
    assert procs.open_files_under(state['session_dir']) == []
    assert not os.path.exists(procs.state_path(env))
    order = [line.split('stopping ')[1].split(':')[0] for line in launcher.output().splitlines()
             if ' stopping ' in line and ':' in line.split('stopping ')[1]]
    assert order == ['bridge', 'stack', 'relay'], order


def test_no_label_reaches_any_log(session):
    """INV-7: the labels are in the data directory only."""
    roots = [session['env']['ROS_LOG_DIR'], str(session['tmp'])]
    for root in roots:
        for path in glob.glob(os.path.join(root, '**', '*'), recursive=True):
            if not os.path.isfile(path) or path.startswith(str(session['data'])):
                continue
            with open(path, 'rb') as handle:
                text = handle.read()
            for label in session['labels']:
                assert label.encode() not in text, f'{label!r} in {path}'


def test_the_manual_launch_has_the_same_parameters(session, tmp_path):
    """INV-2, second half: the manual sequence, started by hand, read the same way."""
    assert 'parameters' in session, 'the launcher session was not read'
    env = session['env']
    data = tmp_path / 'data'
    md.write_labels(str(data), 'manual-task', 'manual-operator')
    relay = Process([os.path.join(CHECKOUT, 'tools', 'relay.sh'), '--port', str(PORT)], env,
                    str(tmp_path / 'relay.log'))
    stack = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                     'rviz:=false', 'record:=true', 'reset:=true', f'record_data_dir:={data}',
                     'record_session:=manual'], env, str(tmp_path / 'stack.log'))
    bridge = Process(['ros2', 'launch', 'ranger_xarm_teleop_bridge', 'bridge.launch.py',
                      f'relay_url:=ws://127.0.0.1:{PORT}/ws'], env, str(tmp_path / 'bridge.log'))
    probe = session['probe']
    started = time.monotonic()
    try:
        # States published by the manual stack itself (the probe's cached owner is the
        # stopped session's). A node answers parameter requests only once it spins, which is
        # after its constructor has declared every parameter.
        probe.wait_for(lambda: probe.states_since(started) and probe.episode_since(started),
                       120.0, 'the manual stack')
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 60.0,
                       'MoveIt as owner')
        names = [n.strip('/') for n in COMPARED]
        probe.wait_for(lambda: all(n in [x for x, _ in probe.node.get_node_names_and_namespaces()]
                                   for n in names), 60.0, 'every compared node')
        manual = stack_parameters(probe)
    finally:
        for p in (bridge, stack, relay):
            p.stop()
    for node in COMPARED:
        assert manual[node] == session['parameters'][node], node


def test_stop_after_the_supervisor_was_killed(session, tmp_path):
    env = session['env']
    launcher = Launcher(env, tmp_path, 'killed', '--profile', 'mock', '--no-rviz', '--yes',
                        '--no-record', '--port', str(PORT))
    try:
        state = launcher.wait_phase(('stack_ready', 'session_ready'), 240.0)
        os.kill(state['supervisor_pid'], signal.SIGKILL)
        assert launcher.process.wait(10.0) == -signal.SIGKILL
        assert procs.marked(state['mark']), 'the children should outlive the supervisor'
        code, out = tool(env, 'stop')
        assert code == 0, out
        assert f'the supervisor (pid {state["supervisor_pid"]}) of session' in out
        assert 'was already gone' in out and 'PASS' in out, out
        stops = [line.split(':')[0] for line in out.splitlines() if line.startswith('stopping')]
        assert stops == ['stopping bridge', 'stopping stack', 'stopping relay'], out
        assert procs.marked(state['mark']) == []
        assert not os.path.exists(procs.state_path(env))
    finally:
        launcher.process.stop()


def test_a_profile_that_is_not_a_simulation_is_refused(session):
    """
    INV-3: the launcher starts no stack for a profile it cannot show to be a simulation.

    From the command line this is the unknown-profile path (the launcher knows only simulation
    profiles); the "simulation: true" check itself is a unit test (test_plan.py).
    """
    code, out = tool(session['env'], 'start', '--profile', 'hardware', '--yes', timeout=60.0)
    assert code == 2, out
    assert 'refused' in out and 'hardware bring-up is started by hand' in out
    assert 'started relay' not in out and 'started stack' not in out
    assert not os.path.exists(procs.state_path(session['env']))


def overlay_without(tmp, missing):
    """Return an AMENT_PREFIX_PATH entry whose bringup share lacks one required limit value."""
    from ament_index_python.packages import get_package_share_directory
    real = get_package_share_directory('ranger_xarm_teleop_bringup')
    prefix = tmp / 'overlay'
    share = prefix / 'share' / 'ranger_xarm_teleop_bringup'
    (prefix / 'share' / 'ament_index' / 'resource_index' / 'packages').mkdir(parents=True)
    (prefix / 'share' / 'ament_index' / 'resource_index' / 'packages' /
     'ranger_xarm_teleop_bringup').write_text('')
    share.mkdir(parents=True)
    for entry in os.listdir(real):
        if entry not in ('config', 'launch'):
            os.symlink(os.path.join(real, entry), share / entry)
    # Copied, not linked: the launch file search does not follow a linked directory.
    shutil.copytree(os.path.join(real, 'launch'), share / 'launch')
    shutil.copytree(os.path.join(real, 'config'), share / 'config')
    limits = share / 'config' / 'SIMULATION_ONLY_limits.yaml'
    lines = [line for line in limits.read_text().splitlines()
             if not line.strip().startswith(missing + ':')]
    assert len(lines) < len(limits.read_text().splitlines()), f'{missing} not in the file'
    limits.write_text('\n'.join(lines) + '\n')
    return str(prefix)


def test_a_missing_required_value_still_stops_the_stack_and_the_launcher_says_so(
        session, tmp_path):
    """INV-3: the stack's refusal is not bypassed; the launcher reports it and stops."""
    env = dict(session['env'])
    missing = 'linear_speed_cap_m_s'
    env['AMENT_PREFIX_PATH'] = overlay_without(tmp_path, missing) + os.pathsep + \
        env['AMENT_PREFIX_PATH']
    launcher = Launcher(env, tmp_path, 'refused', '--profile', 'mock', '--no-rviz', '--yes',
                        '--no-record', '--port', str(PORT))
    try:
        code = launcher.process.wait(240.0)
        out = launcher.output()
        assert code == 1, out[-4000:]
        assert 'stack failed' in out and missing in out, out[-4000:]
        assert 'STACK READY' not in out
        assert 'PASS: no process of the session left' in out, out[-3000:]
        assert not os.path.exists(procs.state_path(env))
    finally:
        launcher.process.stop()
