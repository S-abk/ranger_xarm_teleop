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
The stop, on real processes (no ROS).

Only processes with the session's mark are signalled, in stop order; escalations and
leftovers are reported; a dead supervisor's processes are stopped; a second session is
refused while one runs.
"""

import json
import os
import secrets
import signal
import subprocess
import sys
import time

import pytest
from ranger_xarm_teleop_session import cli
from ranger_xarm_teleop_session import procs
from ranger_xarm_teleop_session import supervisor as sv

SLEEPER = 'import time\ntime.sleep(60)\n'
DEAF = 'import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(60)\n'
# A marked process that starts an unmarked one in its own process group, then waits.
PARENT_OF_UNMARKED = (
    'import os, subprocess, sys, time\n'
    'env = {k: v for k, v in os.environ.items() if k != "TELEOP_SESSION_MARK"}\n'
    'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env)\n'
    'print(child.pid, flush=True)\n'
    'time.sleep(60)\n')


@pytest.fixture
def mark():
    return 'test-' + secrets.token_hex(6)


@pytest.fixture
def started():
    """Start processes; kill whatever is left at the end (marked or not)."""
    out = []

    def start(code, mark=None, new_session=True, stdout=None):
        env = dict(os.environ)
        env.pop(procs.MARK_VAR, None)
        if mark:
            env[procs.MARK_VAR] = mark
        p = subprocess.Popen([sys.executable, '-c', code], env=env,
                             start_new_session=new_session, stdout=stdout)
        out.append(p)
        return p

    yield start
    for p in out:
        for pid in [p.pid] + getattr(p, 'extra', []):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        p.wait()


def wait_marked(mark, n, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        found = procs.marked(mark)
        if len(found) >= n:
            return found
        time.sleep(0.05)
    raise AssertionError(f'{n} marked processes did not appear')


def entry(name, p):
    return {'name': name, 'pid': p.pid, 'rank': sv.pl.STOP_ORDER.index(name)}


def test_marked_processes_are_found_by_their_environment(mark, started):
    a = started(SLEEPER, mark)
    started(SLEEPER)                                   # unmarked
    found = wait_marked(mark, 1)
    assert [p.pid for p in found] == [a.pid]
    assert procs.mark_of(a.pid) == mark
    assert procs.marked('test-' + secrets.token_hex(6)) == []


def test_steps_stop_in_order_with_sigint(mark, started):
    relay, bridge, stack = (started(SLEEPER, mark) for _ in range(3))
    wait_marked(mark, 3)
    said = []
    children = [entry('relay', relay), entry('stack', stack), entry('bridge', bridge)]
    problems = sv.stop_processes(mark, children, said.append, lambda: [
        p.poll() for p in (relay, bridge, stack)])
    assert problems == []
    assert [s.split(':')[0] for s in said] == ['stopping bridge', 'stopping stack',
                                               'stopping relay']
    assert [p.wait(5) for p in (relay, bridge, stack)] != [None] * 3
    assert procs.marked(mark) == []


def test_a_step_that_ignores_sigint_is_escalated_and_reported(mark, started):
    deaf = started(DEAF, mark)
    wait_marked(mark, 1)
    time.sleep(0.3)                                    # the handler is installed
    problems = sv.stop_processes(mark, [entry('stack', deaf)], lambda s: None,
                                 lambda: deaf.poll(), waits={'stack': 0.5})
    assert len(problems) == 1 and 'SIGTERM' in problems[0] and 'after SIGINT' in problems[0]
    assert deaf.wait(5) == -signal.SIGTERM


# A launch that does not take the SIGINT up: it ignores it, and ends once its child has ended.
DEAF_LAUNCH = (
    'import signal, subprocess, sys, time\n'
    'signal.signal(signal.SIGINT, signal.SIG_IGN)\n'
    'child = subprocess.Popen([sys.executable, "-c", "import signal, time; '
    'signal.signal(signal.SIGINT, signal.default_int_handler); time.sleep(60)"])\n'
    'print(f"[INFO] [sleeper-1]: process started with pid [{child.pid}]", flush=True)\n'
    'child.wait()\n')


def test_a_launch_that_does_not_take_up_the_sigint_gets_it_for_each_process(mark, started,
                                                                            tmp_path):
    log = tmp_path / 'stack.log'
    out = open(log, 'w+')
    launch = started(DEAF_LAUNCH, mark, stdout=out)
    wait_marked(mark, 2)
    notes = []
    child = dict(entry('stack', launch), log=str(log))
    problems = sv.stop_processes(mark, [child], lambda s: None, lambda: launch.poll(),
                                 notes=notes)
    assert problems == []
    assert len(notes) == 1 and 'did not take up the SIGINT' in notes[0]
    assert launch.wait(5) == 0
    report = sv.stop_report(None, problems, False, notes)
    assert 'note: stack: its launch did not take up' in report and 'PASS' in report


# A launch still starting: it loses a SIGINT until its first process is started (seen once in a
# stack test: the bridge's launch, 0.5 s old, kept going and started its node).
STARTING_LAUNCH = (
    'import signal, time\n'
    'signal.signal(signal.SIGINT, signal.SIG_IGN)\n'
    'time.sleep(1.5)\n'
    'signal.signal(signal.SIGINT, signal.default_int_handler)\n'
    'print("[INFO] [node-1]: process started with pid [1]", flush=True)\n'
    'time.sleep(60)\n')


def test_a_launch_is_signalled_only_once_it_has_started_its_first_process(
        mark, started, tmp_path):
    log = tmp_path / 'bridge.log'
    launch = started(STARTING_LAUNCH, mark, stdout=open(log, 'w'))
    wait_marked(mark, 1)
    notes = []
    problems = sv.stop_processes(mark, [dict(entry('bridge', launch), log=str(log))],
                                 lambda s: None, lambda: launch.poll(), notes=notes)
    assert problems == [] and notes == []
    assert launch.wait(5) is not None


# A launch that ends on SIGINT and leaves its child running (gz's front end and its server).
LEAVES_CHILD = (
    'import subprocess, sys, time\n'
    'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n'
    'print(f"[INFO] [gazebo-1]: process started with pid [{child.pid}]", flush=True)\n'
    'time.sleep(60)\n')


def test_a_process_that_outlives_its_launch_gets_the_sigint_a_terminal_would_give(
        mark, started, tmp_path):
    log = tmp_path / 'simulator.log'
    launch = started(LEAVES_CHILD, mark, stdout=open(log, 'w'))
    found = wait_marked(mark, 2)
    launch.extra = [p.pid for p in found if p.pid != launch.pid]
    notes = []
    problems = sv.stop_processes(mark, [dict(entry('simulator', launch), log=str(log))],
                                 lambda s: None, lambda: launch.poll(), notes=notes)
    assert problems == []
    assert len(notes) == 1 and 'outlived its launch' in notes[0], notes
    assert procs.marked(mark) == []


def test_a_launch_that_says_it_took_the_sigint_is_left_to_do_its_work(mark, started, tmp_path):
    log = tmp_path / 'stack.log'
    log.write_text('[WARNING] [launch]: user interrupted with ctrl-c (SIGINT)\n')
    assert not sv._acknowledged(str(log), log.stat().st_size)
    assert sv._acknowledged(str(log), 0)


def test_a_process_without_the_mark_is_never_signalled(mark, started, tmp_path):
    out = open(tmp_path / 'pid', 'w+')
    parent = started(PARENT_OF_UNMARKED, mark, stdout=out)
    end = time.monotonic() + 10
    while time.monotonic() < end and not open(tmp_path / 'pid').read().strip():
        time.sleep(0.05)
    unmarked = int(open(tmp_path / 'pid').read())
    parent.extra = [unmarked]
    assert procs.mark_of(unmarked) is None
    problems = sv.stop_processes(mark, [entry('stack', parent)], lambda s: None,
                                 lambda: parent.poll())
    assert problems == []
    assert parent.wait(5) is not None
    assert procs.alive(unmarked), 'the unmarked process in the same group was signalled'


def test_a_marked_process_outside_every_step_is_named_and_ended(mark, started):
    step = started(SLEEPER, mark)
    stray = started(SLEEPER, mark)                     # own session, in no step's group
    wait_marked(mark, 2)
    problems = sv.stop_processes(mark, [entry('stack', step)], lambda s: None,
                                 lambda: [step.poll(), stray.poll()])
    assert len(problems) == 1 and 'left behind' in problems[0] and str(stray.pid) in problems[0]
    assert stray.wait(5) is not None
    assert procs.marked(mark) == []


def test_stop_with_the_supervisor_already_gone(mark, started, tmp_path):
    gone = subprocess.Popen([sys.executable, '-c', 'pass'])
    gone.wait()
    bridge, stack = started(SLEEPER, mark), started(SLEEPER, mark)
    wait_marked(mark, 2)
    state_file = str(tmp_path / 'state.json')
    procs.write_state(state_file, {'session': 'S1', 'mark': mark, 'supervisor_pid': gone.pid,
                                   'children': [entry('stack', stack), entry('bridge', bridge)],
                                   'session_dir': None, 'log_dir': str(tmp_path)})
    said = []
    assert cli.stop_command(state_file, out=said.append) == sv.EXIT_OK
    assert 'was already gone' in said[0] and 'in order' in said[0]
    assert said[1].startswith('stopping bridge') and said[2].startswith('stopping stack')
    assert 'PASS' in said[-1] and 'already gone' in said[-1]
    assert procs.marked(mark) == []
    assert not os.path.exists(state_file)


def test_a_stale_state_file_is_reported_cleared_and_not_obeyed(mark, tmp_path):
    state_file = str(tmp_path / 'state.json')
    procs.write_state(state_file, {'session': 'OLD', 'mark': mark, 'supervisor_pid': 1})
    said = []
    assert cli.refuse_second_session(state_file, out=said.append) is None
    assert 'stale' in said[0] and 'OLD' in said[0] and 'not obeyed' in said[0]
    assert not os.path.exists(state_file)
    procs.write_state(state_file, {'session': 'OLD', 'mark': mark, 'supervisor_pid': 1})
    said = []
    assert cli.stop_command(state_file, out=said.append) == sv.EXIT_OK
    assert 'stale' in said[0] and not os.path.exists(state_file)


def test_a_second_session_is_refused_while_one_runs(mark, started, tmp_path):
    started(SLEEPER, mark)
    wait_marked(mark, 1)
    state_file = str(tmp_path / 'state.json')
    procs.write_state(state_file, {'session': 'RUNNING', 'mark': mark, 'profile': 'mock',
                                   'role': 'stack', 'phase': 'session_ready'})
    said = []
    assert cli.refuse_second_session(state_file, out=said.append) == sv.EXIT_REFUSED
    assert 'RUNNING' in said[0] and 'tools/teleop_session stop' in said[0]
    assert os.path.exists(state_file)


def test_status_and_state_file(tmp_path):
    env = {procs.STATE_DIR_VAR: str(tmp_path)}
    assert procs.state_path(env) == str(tmp_path / 'state.json')
    assert procs.state_dir({'XDG_RUNTIME_DIR': '/run/user/7'}) == '/run/user/7/teleop_session'
    path = procs.state_path(env)
    assert procs.read_state(path) is None
    procs.write_state(path, {'mark': 'm'})
    assert procs.read_state(path) == {'mark': 'm'}
    assert not procs.clear_state(path, 'other') and os.path.exists(path)
    assert procs.clear_state(path, 'm') and not os.path.exists(path)
    with open(path, 'w') as handle:
        handle.write('{not json')
    assert procs.read_state(path) == {'unreadable': True}


def test_after_stop_finds_open_files_and_episodes_still_recording(tmp_path):
    session = tmp_path / 'session'
    (session / 'episodes').mkdir(parents=True)
    with open(session / 'episodes' / 'ep_0001.json', 'w') as handle:
        json.dump({'episode': {'status': 'interrupted'}}, handle)
    assert sv.after_stop(str(session)) == []
    assert sv.episode_summary(str(session)) == '1 interrupted'
    with open(session / 'episodes' / 'ep_0002.json', 'w') as handle:
        json.dump({'episode': {'status': 'recording'}}, handle)
    held = open(session / 'bag.mcap', 'w')
    try:
        problems = sv.after_stop(str(session))
    finally:
        held.close()
    assert any('ep_0002.json' in p and 'still marked recording' in p for p in problems)
    assert any('bag.mcap' in p and 'open file' in p for p in problems)
    report = sv.stop_report(str(session), problems, supervisor_gone=False)
    assert 'PROBLEMS' in report and 'PASS' not in report


def test_descendants_by_parent_pid():
    ps = [procs.Proc(10, 10, 1, 'a', ''), procs.Proc(11, 10, 10, 'b', ''),
          procs.Proc(12, 12, 11, 'c', ''), procs.Proc(13, 13, 1, 'd', '')]
    assert sorted(p.pid for p in procs.descendants_of(10, ps)) == [10, 11, 12]
