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
The session supervisor: start the steps in order, run the pre-flight, stop in order. No ROS.

Each step is a child process in its own process group, with its output in a log file of the
session's log directory (never in the checkout). The supervisor prints only its own lines; the
logs stay on disk. It moves nothing, arms nothing, calls no service and publishes nothing
(INV-1): every process it starts is one of the manual sequence's.

Start (stack role): relay, simulator, simulator check, teleop stack, bridge, probe. When the
probe process has ended, "STACK READY": from then on nothing joins or leaves the ROS graph
(INV-6). The headset items then come from the bridge's and the relay's logs, until
"SESSION READY".

Stop (Ctrl-C, SIGTERM, SIGHUP, or ``teleop_session stop``): ``stop_processes`` in the order of
``plan.STOP_ORDER``, each with SIGINT (as Ctrl-C in its terminal would), escalating only past a
timeout, which is reported; then every process still carrying the session's mark is named,
ended and reported. Only processes carrying the mark are ever signalled.
"""

import glob
import json
import os
import signal
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

from . import logwatch
from . import plan as pl
from . import preflight as pf
from . import procs
from .facts import feedback_js

# How long each step gets after SIGINT before SIGTERM (the stack's launch escalates by itself:
# the recorder has 10 s + 10 s; MoveIt and the controllers take a few seconds).
STOP_WAIT_S = {'probe': 10.0, 'sim_check': 10.0, 'bridge': 20.0, 'stack': 60.0,
               'simulator': 40.0, 'relay': 15.0}
TERM_WAIT_S = 10.0
KILL_WAIT_S = 5.0
RELAY_UP_S = 30.0
SIM_WAIT_S = 180.0          # the simulator up and publishing joint states
PROBE_TIMEOUT_S = {'mock': 120.0, 'gazebo': 180.0}
EXIT_OK, EXIT_REFUSED, EXIT_FAILED = 0, 2, 1


class StopRequested(Exception):
    """A stop was asked for (signal); the supervisor stops everything it started."""


class Console:
    """The supervisor's own lines: on the console and in launcher.log (never a label)."""

    def __init__(self, out=None, log_path: Optional[str] = None):
        """Print to ``out`` (stdout) and append to ``log_path``."""
        self.out = out or sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8') if log_path else None

    def say(self, text: str = '') -> None:
        """Print a block of text, each line with the time."""
        stamp = time.strftime('%H:%M:%S')
        lines = text.split('\n') if text else ['']
        block = '\n'.join(f'{stamp} {line}' if line else '' for line in lines)
        print(block, file=self.out, flush=True)
        if self.log:
            self.log.write(block + '\n')
            self.log.flush()

    def attach_log(self, path: str) -> None:
        """Start writing launcher.log (once the log directory exists)."""
        self.log = open(path, 'a', encoding='utf-8')


class Tail:
    """New complete lines of a growing file."""

    def __init__(self, path: str):
        """Read ``path`` from its start."""
        self.path, self.pos, self.rest = path, 0, ''

    def lines(self) -> List[str]:
        """Return the lines written since the last call."""
        try:
            with open(self.path, encoding='utf-8', errors='replace') as handle:
                handle.seek(self.pos)
                data = handle.read()
                self.pos = handle.tell()
        except OSError:
            return []
        data = self.rest + data
        parts = data.split('\n')
        self.rest = parts.pop()
        return parts


class Child:
    """One step of the session, running."""

    def __init__(self, step: pl.Step, env: Dict[str, str], log_dir: str):
        """Start ``step`` in its own process group, its output in ``<log_dir>/<name>.log``."""
        self.step = step
        self.log_path = os.path.join(log_dir, f'{step.name}.log')
        self.log = open(self.log_path, 'a', encoding='utf-8')
        self.log.write('$ ' + ' '.join(step.argv) + '\n')
        self.log.flush()
        self.proc = subprocess.Popen(list(step.argv), env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        self.tail = Tail(self.log_path)

    @property
    def pid(self) -> int:
        """Return the process id (also its process group)."""
        return self.proc.pid

    def ended(self) -> Optional[int]:
        """Return the exit code once it has ended, else None (reaps it)."""
        return self.proc.poll()

    def last_lines(self, n: int = 12) -> str:
        """Return the end of its log."""
        try:
            with open(self.log_path, encoding='utf-8', errors='replace') as handle:
                return ''.join(handle.readlines()[-n:]).rstrip()
        except OSError:
            return ''

    def state(self) -> dict:
        """Return its entry in the state file."""
        return {'name': self.step.name, 'pid': self.pid, 'rank': self.step.stop_rank,
                'log': self.log_path}


# ------------------------------------------------------------------ stopping
def _signal_marked(pids, sig: int, mark: str) -> List[int]:
    """Send ``sig`` to each of ``pids`` that still carries ``mark``; return those signalled."""
    sent = []
    for pid in pids:
        if procs.mark_of(pid) != mark:
            continue
        try:
            os.kill(pid, sig)
            sent.append(pid)
        except ProcessLookupError:
            pass
    return sent


def _remaining(pids, mark: str) -> List[int]:
    return [p for p in pids if procs.mark_of(p) == mark and procs.alive(p)]


def _wait_gone(pids, mark: str, timeout: float, reap: Callable[[], None]) -> List[int]:
    end = time.monotonic() + timeout
    while True:
        reap()
        left = _remaining(pids, mark)
        if not left or time.monotonic() >= end:
            return left
        time.sleep(0.1)


def stop_processes(mark: str, children: List[dict], say: Callable[[str], None],
                   reap: Callable[[], None] = lambda: None,
                   waits: Optional[Dict[str, float]] = None) -> List[str]:
    """
    Stop the session's processes in stop order; return the problems (escalations, leftovers).

    ``children``: state-file entries ({'name', 'pid', 'rank'}). A child's processes are those
    carrying ``mark`` in its process group or below it. Each gets SIGINT; what is left after its
    wait gets SIGTERM, then SIGKILL, and that is a problem. Then every process still carrying
    ``mark`` (but the caller) is named, ended the same way, and is a problem too.
    """
    waits = dict(STOP_WAIT_S, **(waits or {}))
    problems = []
    for child in sorted(children, key=lambda c: c['rank']):
        live = procs.marked(mark)
        group = {p.pid for p in live if p.pgid == child['pid'] or p.pid == child['pid']}
        group |= {p.pid for p in procs.descendants_of(child['pid'], live)}
        if not group:
            continue
        name = child['name']
        # One SIGINT to the step's own process (ros2 launch, the relay): a launch passes it
        # on to its nodes once and escalates by itself. Signalling the whole group as well
        # would give every node a second SIGINT while it cleans up (the episode node marks an
        # episode interrupted in that cleanup). Only when the leader is gone do the rest of
        # its processes get the SIGINT directly.
        if child['pid'] in group:
            leader = [child['pid']]
            say(f'stopping {name}: SIGINT to pid {child["pid"]} ({len(group)} processes)')
        else:
            leader = sorted(group)
            say(f'stopping {name}: its first process has ended; SIGINT to the {len(group)} '
                'processes left')
        _signal_marked(leader, signal.SIGINT, mark)
        left = _wait_gone(group, mark, waits.get(name, 15.0), reap)
        # New processes of the same group (started while it shut down) are its too.
        for sig, wait, word in ((signal.SIGTERM, TERM_WAIT_S, 'SIGTERM'),
                                (signal.SIGKILL, KILL_WAIT_S, 'SIGKILL')):
            if not left:
                break
            problems.append(f'{name}: {len(left)} processes still running '
                            f'{waits.get(name, 15.0):.0f} s after SIGINT; sent {word} to '
                            + ', '.join(_describe(p) for p in left))
            say(problems[-1])
            _signal_marked(left, sig, mark)
            left = _wait_gone(left, mark, wait, reap)
        if left:
            problems.append(f'{name}: could not end ' + ', '.join(_describe(p) for p in left))
            say(problems[-1])
    leftovers = procs.marked(mark)
    if leftovers:
        problems.append('left behind after every step had stopped: '
                        + ', '.join(f'{p.pid} {p.name}' for p in leftovers) + '; ended now')
        say(problems[-1])
        pids = [p.pid for p in leftovers]
        _signal_marked(pids, signal.SIGTERM, mark)
        left = _wait_gone(pids, mark, TERM_WAIT_S, reap)
        if left:
            _signal_marked(left, signal.SIGKILL, mark)
            left = _wait_gone(left, mark, KILL_WAIT_S, reap)
        if left:
            problems.append('could not end ' + ', '.join(_describe(p) for p in left))
            say(problems[-1])
    return problems


def _describe(pid: int) -> str:
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as handle:
            cmd = handle.read().replace(b'\0', b' ').decode('utf-8', 'replace').strip()
    except OSError:
        cmd = '?'
    return f'{pid} ({cmd[:70]})'


def after_stop(session_dir: Optional[str]) -> List[str]:
    """Return what a stop must not leave: open files of the session, an episode still recording."""
    if not session_dir or not os.path.isdir(session_dir):
        return []
    problems = [f'open file in the session: {f}' for f in procs.open_files_under(session_dir)]
    sidecars = {}
    for path in sorted(glob.glob(os.path.join(session_dir, 'episodes', '*.json'))):
        try:
            with open(path, encoding='utf-8') as handle:
                sidecars[path] = json.load(handle)
        except (OSError, ValueError) as err:
            problems.append(f'sidecar {path} unreadable: {err}')
    problems += [f'episode still marked recording: {p}'
                 for p in procs.episodes_still_recording(sidecars)]
    return problems


def episode_summary(session_dir: Optional[str]) -> str:
    """Return the episodes of the session by status (for the stop report)."""
    counts: Dict[str, int] = {}
    for path in glob.glob(os.path.join(session_dir or '', 'episodes', '*.json')):
        try:
            with open(path, encoding='utf-8') as handle:
                status = (json.load(handle).get('episode') or {}).get('status', '?')
        except (OSError, ValueError):
            status = 'unreadable'
        counts[status] = counts.get(status, 0) + 1
    return ', '.join(f'{n} {s}' for s, n in sorted(counts.items())) or 'no episode'


# ------------------------------------------------------------------ the session
class Supervisor:
    """One session: start, check, watch, stop."""

    def __init__(self, opts: pl.Options, console: Console, *, checkout: str, mark: str,
                 state_file: str, session: str, log_dir: str, policy: str = 'ask',
                 keep_labels: bool = False, ask: Callable[[str], str] = input):
        """Prepare a session (nothing starts here)."""
        self.opts, self.console = opts, console
        self.checkout, self.mark, self.state_file = checkout, mark, state_file
        self.session, self.log_dir = session, log_dir
        self.policy, self.keep_labels, self.ask = policy, keep_labels, ask
        self.children: List[Child] = []
        self.phase = 'starting'
        self.items: Dict[str, pf.Item] = {}
        self.session_dir: Optional[str] = None
        self.stopping = False
        self.bridge = logwatch.BridgeState()
        self.relay = logwatch.RelayState()
        self.pairs = logwatch.PairsState()
        self.relay_here = opts.role == 'relay' or not opts.relay_url
        self.env = dict(os.environ)
        self.env[procs.MARK_VAR] = mark
        self.env.pop('TELEOP_SESSION_SELF', None)
        self.env['PYTHONUNBUFFERED'] = '1'
        self.record_profile: Optional[dict] = None

    # ---- bookkeeping
    def say(self, text: str = '') -> None:
        """Print one of the supervisor's lines."""
        self.console.say(text)

    def write_state(self) -> None:
        """Write the state file (who to stop, in which order)."""
        procs.write_state(self.state_file, {
            'session': self.session, 'mark': self.mark, 'supervisor_pid': os.getpid(),
            'profile': self.opts.profile, 'role': self.opts.role, 'phase': self.phase,
            'started_wall': self.started_wall, 'log_dir': self.log_dir,
            'session_dir': self.session_dir,
            'children': [c.state() for c in self.children]})

    def set_phase(self, phase: str) -> None:
        """Record the phase in the state file (tests and ``status`` read it)."""
        self.phase = phase
        self.write_state()

    def start_step(self, step: pl.Step) -> Child:
        """Start one step and record it before anything else happens."""
        self.check_stop()
        child = Child(step, self.env, self.log_dir)
        self.children.append(child)
        self.write_state()
        self.say(f'started {step.name} (pid {child.pid}): {" ".join(step.argv)}')
        return child

    def check_stop(self) -> None:
        """Raise StopRequested when a stop was asked for."""
        if self.stopping:
            raise StopRequested()

    def reap(self) -> None:
        """Reap ended children (no zombies)."""
        for c in self.children:
            c.ended()

    def pump(self) -> None:
        """Feed new log lines to the parsers."""
        for c in self.children:
            for line in c.tail.lines():
                if c.step.name == 'bridge':
                    self.bridge.feed(line)
                elif c.step.name == 'relay':
                    self.relay.feed(line)
                elif c.step.name == 'stack':
                    self.pairs.feed(line)

    def child(self, name: str) -> Optional[Child]:
        """Return the running step ``name``."""
        return next((c for c in self.children if c.step.name == name), None)

    def failed_child(self) -> Optional[Child]:
        """Return a long-running step that has ended."""
        for c in self.children:
            if not c.step.runs_to_end and c.ended() is not None:
                return c
        return None

    def sleep(self, seconds: float) -> None:
        """Wait, watching for a stop and for a long-running step that ends."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check_stop()
            self.reap()
            self.pump()
            dead = self.failed_child()
            if dead is not None:
                raise ChildEnded(dead)
            time.sleep(min(0.1, max(0.0, end - time.monotonic())))

    def show(self, items: List[pf.Item]) -> None:
        """Print items and remember them."""
        for i in items:
            self.items[i.key] = i
        self.say(pf.render(items))

    def gate(self, where: str, items: List[pf.Item]) -> None:
        """Stop or go on when items are missing, by the policy (ask, yes, strict)."""
        gaps = pf.missing(items)
        if not gaps:
            return
        names = ', '.join(i.title for i in gaps)
        if self.policy == 'yes':
            self.say(f'{where}: missing: {names}; going on (--yes)')
            return
        if self.policy == 'strict':
            self.say(f'{where}: missing: {names}; stopping (--strict)')
            raise Refused(f'missing at {where}: {names}')
        answer = self.ask(f'{where}: missing: {names}. Continue anyway? [y/N] ')
        if answer.strip().lower() not in ('y', 'yes'):
            raise Refused(f'stopped at {where} by the operator')

    # ---- the run
    def run(self, local_items: List[pf.Item], record_profile: Optional[dict],
            sim: Optional[dict]) -> int:
        """Start the session and watch it until a stop; return the exit code."""
        self.started_wall = time.time()
        self.record_profile = record_profile
        code = EXIT_OK
        try:
            self.set_phase('starting')
            if self.opts.role == 'relay':
                self.run_relay_role(local_items)
            else:
                self.run_stack_role(local_items, sim)
        except StopRequested:
            self.say('stop asked for')
        except Refused as why:
            self.say(str(why))
            code = EXIT_REFUSED
        except ChildEnded as dead:
            c = dead.child
            self.say(f'{c.step.name} ended (exit {c.ended()}) while the session ran; its log '
                     f'{c.log_path} ends with:\n{c.last_lines()}')
            code = EXIT_FAILED
        finally:
            problems = self.stop()
        if problems:
            code = code or EXIT_FAILED
        return code

    def wait_relay(self) -> None:
        """Wait until the relay listens."""
        end = time.monotonic() + RELAY_UP_S
        while not self.relay.up:
            if time.monotonic() > end:
                raise Refused(f'the relay did not start within {RELAY_UP_S:.0f} s; see '
                              f'{self.child("relay").log_path}')
            self.sleep(0.2)
        self.say('relay up')

    def run_relay_role(self, local_items: List[pf.Item]) -> None:
        """Relay only (operator PC): the relay, adb, the headset items from its log."""
        self.show(local_items)
        self.start_step(pl.build_plan(self.opts, checkout=self.checkout)[0])
        self.wait_relay()
        host = self.opts.listen
        self.say(f'the bridge on the robot computer connects with:  --relay '
                 f'ws://{host if not pl.is_loopback(host) else "<this PC>"}:{self.opts.port}/ws')
        self.set_phase('relay_ready')
        self.watch_headset()

    def run_stack_role(self, local_items: List[pf.Item], sim: Optional[dict]) -> None:
        """Stack role: everything on this machine, the relay here or elsewhere."""
        self.show(local_items)
        self.gate('before the start', local_items)
        steps = self.plan_steps(sim)
        by = {s.name: s for s in steps}
        if 'relay' in by:
            self.start_step(by['relay'])
            self.wait_relay()
        if 'simulator' in by:
            self.start_step(by['simulator'])
            item = self.run_sim_check(by['sim_check'])
            self.show([item])
            self.gate('after the simulator check', [item])
        self.start_step(by['stack'])
        if self.opts.record:
            self.session_dir = os.path.join(self.data_dir(), self.session)
            self.write_state()
        self.start_step(by['bridge'])
        self.set_phase('probing')
        probe = self.start_step(by['probe'])
        while probe.ended() is None:
            self.sleep(0.2)
        self.write_state()
        report = self.read_json('probe.json')
        self.sleep(0.5)          # the rigid-pairs line, when the stack is slower than the probe
        items = self.stack_items(report)
        self.show(items)
        if report is None:
            self.say(f'the probe gave no report; its log ends with:\n{probe.last_lines()}')
        self.set_phase('stack_ready')
        self.say('STACK READY: the probe has left; nothing joins the ROS graph from here.\n'
                 'Put on the headset now.')
        self.gate('at stack ready', items)
        self.watch_headset()

    def plan_steps(self, sim: Optional[dict]) -> List[pl.Step]:
        """Return the steps of this session."""
        bringup = ''
        if pl.PROFILES[self.opts.profile].simulator is not None:
            bringup = self.bringup_share()
        steps = pl.build_plan(self.opts, checkout=self.checkout, bringup_share=bringup,
                              session=self.session, log_dir=self.log_dir,
                              gpu=(sim or {}).get('gpu'))
        probe_timeout = PROBE_TIMEOUT_S.get(self.opts.profile, 120.0)
        return [pl.Step('probe', pl.probe_argv(self.opts, self.log_dir, probe_timeout), True)
                if s.name == 'probe' else s for s in steps]

    def bringup_share(self) -> str:
        """Return the bringup package's share directory (ROS sourced)."""
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('ranger_xarm_teleop_bringup')

    def data_dir(self) -> str:
        """Return the data directory in force."""
        d = self.opts.data_dir or (self.record_profile or {}).get('data_dir') or '~/teleop_data'
        return os.path.abspath(os.path.expanduser(d))

    def read_json(self, name: str) -> Optional[dict]:
        """Return a JSON file of the log directory, or None."""
        try:
            with open(os.path.join(self.log_dir, name), encoding='utf-8') as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def run_sim_check(self, step: pl.Step) -> pf.Item:
        """Run the simulator check once the simulator publishes joint states."""
        end = time.monotonic() + SIM_WAIT_S
        self.say('waiting for the simulator, then measuring it for '
                 f'{self.opts.sim_check_seconds:g} s (record_commission --simulator)')
        while True:
            child = self.start_step(step)
            while child.ended() is None:
                self.sleep(0.2)
            code = child.ended()
            self.children.remove(child)
            self.write_state()
            if code != 2 or time.monotonic() > end:   # 2: nothing publishes /joint_states yet
                break
            self.sleep(3.0)
        return pf.sim_check_item(self.read_json('sim_check.json'), code, child.last_lines(3))

    def stack_items(self, report: Optional[dict]) -> List[pf.Item]:
        """Return the stack items of the pre-flight."""
        profile = self.record_profile or {}
        return [pf.streams_item(report), pf.owner_item(report),
                pf.pairs_item(self.pairs.result, self.pairs.count, self.pairs.why),
                pf.recorder_item(report, self.opts.record, pf.headset_streams(profile)),
                pf.feedback_item(report, self.opts.feedback),
                pf.reset_item(report, self.opts.reset)]

    def headset_items(self) -> List[pf.Item]:
        """Return the headset items, from the logs only."""
        relay = {'peers': self.relay.headset_peers()} if self.relay_here else None
        bridge = None
        if self.opts.role == 'stack':
            bridge = {'lines': self.bridge.lines, 'connected': self.bridge.connected,
                      'rate': self.bridge.rate,
                      'url': self.opts.relay_url or pl.local_relay_url(self.opts.port)}
        expected = None
        try:
            with open(feedback_js(self.checkout), encoding='utf-8') as handle:
                expected = logwatch.feedback_version(handle.read())
        except OSError:
            pass
        return [pf.headset_item(bridge, relay, self.opts.role),
                pf.version_item(expected, self.relay_here, relay)]

    def watch_headset(self) -> None:
        """Watch the headset items until a stop; say when the session is ready."""
        shown: Dict[str, tuple] = {}
        ready = False
        while True:
            self.sleep(0.5)
            items = self.headset_items()
            # Printed when the status changes (not each new frame rate).
            changed = [i for i in items if shown.get(i.key) != i.status]
            if changed:
                self.show(changed)
                shown.update({i.key: i.status for i in changed})
            for i in items:
                self.items[i.key] = i
            if not ready and not pf.missing(items):
                keys = {i.key for i in items}
                earlier = [i for i in self.items.values()
                           if i.key not in keys and i.status in (pf.MISSING, pf.UNKNOWN)]
                checks = pf.unknown(items) + earlier
                ready = True
                self.set_phase('session_ready')
                self.say('SESSION READY'
                         + ('; not ok earlier or to check yourself: '
                            + '; '.join(f'{i.title} ({pf.SHOWN[i.status]})' for i in checks)
                            if checks else '')
                         + '\nStop with Ctrl-C here, or: tools/teleop_session stop')
            elif ready and pf.missing(items):
                ready = False
                self.set_phase('headset_lost')
                self.say('the headset items are missing again (above)')

    # ---- stopping
    def stop(self) -> List[str]:
        """Stop everything this session started, in order; print the stop report."""
        self.stopping = True
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: self.say('stopping already; please wait'))
        try:
            self.phase = 'stopping'
            self.write_state()
        except OSError:
            pass
        self.say('stopping the session')
        problems = stop_processes(self.mark, [c.state() for c in self.children], self.say,
                                  self.reap)
        for c in self.children:
            c.log.close()
        problems += after_stop(self.session_dir)
        report = stop_report(self.session_dir, problems, supervisor_gone=False)
        self.say(report)
        try:
            with open(os.path.join(self.log_dir, 'stop_report.txt'), 'w',
                      encoding='utf-8') as handle:
                handle.write(report + '\n')
        except OSError:
            pass
        procs.clear_state(self.state_file, self.mark)
        return problems


def stop_report(session_dir: Optional[str], problems: List[str], supervisor_gone: bool) -> str:
    """Return the stop report."""
    lines = ['STOP REPORT']
    if supervisor_gone:
        lines.append('the supervisor was already gone; its processes were stopped from the '
                     'state file and the session mark')
    if session_dir:
        lines.append(f'episodes: {episode_summary(session_dir)}')
    lines.append('PASS: no process of the session left, no episode file open, no episode '
                 'still recording' if not problems else
                 'PROBLEMS:\n- ' + '\n- '.join(problems))
    return '\n'.join(lines)


class Refused(Exception):
    """The session cannot go on; the message says why."""


class ChildEnded(Exception):
    """A long-running step ended on its own."""

    def __init__(self, child: Child):
        """Carry the step that ended."""
        super().__init__(child.step.name)
        self.child = child
