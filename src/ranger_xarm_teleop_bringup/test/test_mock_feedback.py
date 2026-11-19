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
Operator feedback end to end in mock execution: stack, feedback node, bridge, relay, headset.

The real vendored relay (from the checkout's ``.venv``, made by tools/relay_setup.sh) carries
the episode fixtures from a stand-in headset to the bridge and the bridge's feedback back. At
the headset end: exactly one ``teleop_haptic`` per episode event with its pattern, nothing older
than its maximum age, ``teleop_status`` at 5 Hz with the stack's state; with the feedback node
stopped or killed teleop goes on and the panel (feedback.js, run in node) shows NO DATA.
The feedback node slowed or stopped never delays the teleop node: its /teleop/state stream and
the joint-state age it judges stay as with the feedback node running, and the clutch holds.
Without ``.venv`` or node the module is skipped and reported NOT RUN.
"""

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time

import numpy as np
import pytest
from ranger_xarm_teleop_bridge.episode_feedback import DEFAULT_PATTERNS, EVENT_NAMES
from ranger_xarm_teleop_interfaces.msg import ArmOwner
from ranger_xarm_teleop_record import metadata as md
from stack_helpers import child_environment, cpu_seconds, node_pids, Process, READY
from test_mock_record import fixture_file, RecordProbe, wait_for_settled_graph
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKOUT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RELAY_PYTHON = os.path.join(CHECKOUT, '.venv', 'bin', 'python')
PANEL_REPLAY = os.path.join(CHECKOUT, 'src', 'ranger_xarm_teleop_bridge', 'test', 'js',
                            'panel_replay.js')
NODE = shutil.which('node') or shutil.which('nodejs')
SESSION = '20261118-000000-feed'
MAX_AGE_MS = 750.0

if not os.access(RELAY_PYTHON, os.X_OK):
    pytest.skip('NOT RUN: the relay environment .venv is missing (run tools/relay_setup.sh)',
                allow_module_level=True)
if NODE is None:
    pytest.skip('NOT RUN: node is not installed (apt package: nodejs)', allow_module_level=True)


def now_ms():
    return time.monotonic() * 1000.0


class Headset:
    """Stands in for the page: sends fixture frames, keeps every message with its arrival."""

    def __init__(self, url):
        self.url = url
        self.received = []          # (now_ms, message)
        self.loop = asyncio.new_event_loop()
        self.ws = None
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._connect(), self.loop).result(20)

    async def _connect(self):
        for _ in range(100):
            try:
                self.ws = await websockets.connect(self.url)
                break
            except OSError:
                await asyncio.sleep(0.2)
        assert self.ws is not None, 'relay not reachable'
        asyncio.ensure_future(self._read())

    async def _read(self):
        async for text in self.ws:
            self.received.append((now_ms(), json.loads(text)))

    async def _play(self, path):
        with open(path, encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        t0, start = records[0]['t_recv'], time.monotonic()
        for r in records:
            await asyncio.sleep(max(0.0, start + r['t_recv'] - t0 - time.monotonic()))
            frame = dict(r['msg'])
            frame['t_client'] = now_ms()        # the page clock the echo refers to
            await self.ws.send(json.dumps(frame))

    def play(self, path):
        asyncio.run_coroutine_threadsafe(self._play(path), self.loop).result(120)

    def of_type(self, kind, since_ms=0.0, until_ms=float('inf')):
        return [(t, m) for t, m in list(self.received)
                if m.get('type') == kind and since_ms <= t <= until_ms]

    def close(self):
        asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop).result(5)
        self.loop.call_soon_threadsafe(self.loop.stop)


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_feedback')
    data = tmp_path_factory.mktemp('feedback_data')
    md.write_labels(str(data), 'test-task', 'test-operator')
    env = child_environment(offset=13)
    port = free_port()
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                      'rviz:=false', 'record:=true', f'record_data_dir:={data}',
                      f'record_session:={SESSION}'], env, str(logs / 'launch.log'))
    relay_env = dict(env, PYTHONPATH=os.path.join(CHECKOUT, 'third_party', 'vr-teleop-kit', 'src'))
    relay = Process([RELAY_PYTHON, '-m', 'vr_teleop_kit.relay.server', '--port', str(port)],
                    relay_env, str(logs / 'relay.log'))
    url = f'ws://127.0.0.1:{port}/ws'
    bridge = Process(['ros2', 'launch', 'ranger_xarm_teleop_bridge', 'bridge.launch.py',
                      f'relay_url:={url}'], env, str(logs / 'bridge.log'))
    probe = RecordProbe(env)
    headset = None
    try:
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: probe.episode_state() is not None, 60.0, 'the episode state')
        probe.move_to(READY)
        wait_for_settled_graph(probe, launch)
        headset = Headset(url)
        probe.wait_for(lambda: headset.of_type('teleop_status'), 30.0, 'status at the headset')
        yield {'probe': probe, 'env': env, 'logs': logs, 'launch': launch, 'bridge': bridge,
               'headset': headset, 'url': url, 'results': {}}
    finally:
        if headset is not None:
            headset.close()
        probe.close()
        for p in (bridge, relay, launch):
            p.stop()
        print(launch.output()[-4000:])
        print(bridge.output()[-3000:])


def replay(s, name, during=None):
    """
    Play one episode fixture; return (start monotonic s, start ms, end ms).

    ``during`` runs in a thread started as the fixture starts playing (its times count from
    the fixture's 0 s).
    """
    probe = s['probe']
    probe.disarm()
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 30.0, 'MoveIt owner')
    path = fixture_file(str(s['logs']), name)
    start, start_ms = time.monotonic(), now_ms()
    worker = threading.Thread(target=during) if during else None
    if worker:
        worker.start()
    s['headset'].play(path)
    if worker:
        worker.join()
    time.sleep(1.5)
    return start, start_ms, now_ms()


def events_since(probe, start):
    """Return the episode events in order: (event_count, name, host wall time of the event)."""
    out, last = [], None
    for m in probe.episode_since(start):
        if m.event_count != last:
            if last is not None:
                assert m.event_count == last + 1, 'the probe lost an event'
                stamp = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
                at = m.last_event_stamp.sec + 1e-9 * m.last_event_stamp.nanosec
                out.append((m.event_count, EVENT_NAMES[m.last_event],
                            m.wall_time - max(0.0, stamp - at)))
            last = m.event_count
    return out


def total_age_ms(t, m, session_id):
    echo = m['echo']
    assert echo is not None and echo['session_id'] == session_id
    return m['age_ms'] + max(0.0, t - (echo['t_client'] + echo['held_ms']))


def panel(s, session_id, queries):
    """Return the panel feedback.js shows at each query time, from what reached the headset."""
    rows = [{'t_ms': t, 'msg': m} for t, m in s['headset'].received
            if m.get('type') in ('teleop_status', 'teleop_haptic')]
    rows += [{'query_ms': q} for q in queries]
    rows.sort(key=lambda r: r.get('t_ms', r.get('query_ms')))
    out = subprocess.run([NODE, PANEL_REPLAY, '--session', str(session_id)],
                         input='\n'.join(json.dumps(r) for r in rows), capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return [json.loads(line) for line in out.stdout.splitlines() if '"stale"' in line]


def fixture_session(s, name):
    with open(fixture_file(str(s['logs']), name), encoding='utf-8') as handle:
        return json.loads(handle.readline())['msg']['session_id']


def check_haptics(s, start, start_ms, end_ms, session_id):
    events = events_since(s['probe'], start)
    got = s['headset'].of_type('teleop_haptic', start_ms, end_ms)
    assert [(m['source_count'], m['event']) for _, m in got] == \
        [(c, e) for c, e, _ in events]                                   # one each, in order
    for t, m in got:
        p = DEFAULT_PATTERNS[m['event']]
        assert m['pulses'] == [list(x) for x in p.pulses] and m['priority'] == p.priority
        assert m['hand'] == 'left' and m['missed'] == 0
        assert total_age_ms(t, m, session_id) <= m['max_age_ms']
    return events


def test_disarm_fixture_gives_one_vibration_per_event(stack):
    sid = fixture_session(stack, 'episode_disarm')
    pgid = stack['launch'].proc.pid
    pids = {n: node_pids(n, pgid=pgid) for n in ('feedback_node', 'episode_node')}
    assert all(len(v) == 1 for v in pids.values()), pids
    stack['pids'] = {n: v[0] for n, v in pids.items()}
    start, start_ms, end_ms = replay(stack, 'episode_disarm')
    events = check_haptics(stack, start, start_ms, end_ms, sid)
    assert {'START', 'AUTO_STOP', 'KEEP', 'REFUSED'} <= {e for _, e, _ in events}, events
    stack['results']['input_age_on'] = input_ages(stack['probe'], start)
    stack['status_from_ms'] = start_ms


def test_marks_fixture_vibrations_status_rate_and_content_and_cpu(stack):
    sid = fixture_session(stack, 'episode_marks_engaged')
    cpu0 = {n: cpu_seconds(p) for n, p in stack['pids'].items()}
    t0 = time.monotonic()
    start, start_ms, end_ms = replay(stack, 'episode_marks_engaged')
    cpu = {n: 100.0 * (cpu_seconds(p) - cpu0[n]) / (time.monotonic() - t0)
           for n, p in stack['pids'].items()}
    stack['results']['cpu_percent_of_a_core'] = cpu
    events = check_haptics(stack, start, start_ms, end_ms, sid)
    assert {'START', 'MARK', 'STOP', 'KEEP', 'DISCARD'} <= {e for _, e, _ in events}, events
    # Status rate at the headset over the two fixtures (>= 20 s).
    span = (stack['status_from_ms'], end_ms)
    assert span[1] - span[0] >= 20000.0, span
    n = len(stack['headset'].of_type('teleop_status', *span))
    rate = 1000.0 * n / (span[1] - span[0])
    stack['results']['status_rate_hz'] = rate
    assert 4.5 <= rate <= 5.5, rate
    # Content: the newest status equals the stack's state.
    time.sleep(0.5)
    t, last = stack['headset'].of_type('teleop_status')[-1]
    ep, teleop = stack['probe'].episode_state(), stack['probe'].state()
    names = {0: 'NOT_READY', 1: 'IDLE', 2: 'RECORDING', 3: 'REVIEW'}
    assert last['episode']['state'] == names[ep.state]
    assert (last['episode']['index'], last['episode']['kept'], last['episode']['discarded']) == \
        (ep.episode_index, ep.kept_count, ep.discarded_count)
    assert last['teleop']['armed'] == teleop.armed
    assert last['teleop']['scale'] == pytest.approx(teleop.translation_scale)
    assert total_age_ms(t, last, sid) <= last['stale_ms']


def teleop_timing(timed_states, since, until):
    """Arrival intervals of /teleop/state at the probe and the joint-state age (s) per cycle."""
    states = [(w, s) for w, s in timed_states if since <= w < until]
    intervals = np.diff([w for w, _ in states])
    ages = np.array([(s.header.stamp.sec - s.measured_tool_pose_stamp.sec)
                     + (s.header.stamp.nanosec - s.measured_tool_pose_stamp.nanosec) * 1e-9
                     for _, s in states if s.measured_tool_pose_valid])
    return intervals, ages


def test_feedback_node_slowed_or_stopped_never_delays_the_teleop_node(stack):
    """
    Check that a slowed or stopped feedback node never delays the teleop node.

    The feedback node reads every /teleop/state. Whatever it does, slow or stopped, the teleop
    node's cycle, its /teleop/state stream and the joint-state age it judges are what they are
    with the feedback node running, and the clutch is not released. Engaged 1.2-13.0 s of the
    fixture: running 1.5-4.5 s (the baseline), slowed (stopped 90 ms of every 100 ms) 4.5-7.5 s,
    stopped 7.5-12.5 s.
    """
    pid, probe = stack['pids']['feedback_node'], stack['probe']
    marks = {}
    late_before = stack['launch'].output().count('late cycle')
    abort, finished = threading.Event(), threading.Event()

    def slow_then_stop():
        try:
            if abort.wait(4.5):
                return
            marks['slow'] = time.monotonic()
            while time.monotonic() < marks['slow'] + 3.0 and not abort.is_set():
                os.kill(pid, signal.SIGSTOP)
                time.sleep(0.09)
                os.kill(pid, signal.SIGCONT)
                time.sleep(0.01)
            if abort.is_set():
                return
            marks['stop'] = time.monotonic()
            os.kill(pid, signal.SIGSTOP)
            abort.wait(5.0)
            marks['cont'] = time.monotonic()
        finally:
            os.kill(pid, signal.SIGCONT)
            finished.set()

    try:
        start, _, _ = replay(stack, 'episode_marks_engaged', slow_then_stop)
    finally:
        abort.set()                     # a failed replay must not leave the node stopped
        finished.wait(10.0)
        os.kill(pid, signal.SIGCONT)
    timed = probe.timed_states_since(start)
    phases = {'running': (start + 1.5, marks['slow']), 'slowed': (marks['slow'], marks['stop']),
              'stopped': (marks['stop'], marks['cont'])}
    report = {}
    for name, (a, b) in phases.items():
        intervals, ages = teleop_timing(timed, a, b)
        # The teleop node publishes every cycle (50 Hz): a delayed node publishes fewer.
        assert len(intervals) + 1 >= 0.9 * 50.0 * (b - a), \
            f'{name}: {len(intervals) + 1} /teleop/state in {b - a:.1f} s (50 Hz expected)'
        assert len(ages) > 100, (name, len(ages))
        report[name] = {
            'state_interval_ms': {'median': 1e3 * float(np.median(intervals)),
                                  'p99': 1e3 * float(np.percentile(intervals, 99)),
                                  'max': 1e3 * float(intervals.max())},
            'joint_state_age_ms': {'median': 1e3 * float(np.median(ages)),
                                   'p99': 1e3 * float(np.percentile(ages, 99)),
                                   'max': 1e3 * float(ages.max())}}
    report['late_cycle_warnings'] = stack['launch'].output().count('late cycle') - late_before
    stack['results']['teleop_timing_feedback_slowed_stopped'] = report
    print('teleop timing, feedback node running / slowed / stopped:', json.dumps(report))
    base = report['running']
    for name in ('slowed', 'stopped'):
        got = report[name]
        bound = max(1.5 * base['state_interval_ms']['p99'],
                    base['state_interval_ms']['p99'] + 5.0)
        assert got['state_interval_ms']['p99'] <= bound, (name, report)
        assert got['state_interval_ms']['median'] <= base['state_interval_ms']['median'] + 2.0, \
            (name, report)
        # No single stall either: the longest gap stays within the teleop node's own late-cycle
        # limit (3 periods, 60 ms), or within 20 ms of the baseline's longest when that is larger.
        longest = max(60.0, base['state_interval_ms']['max'] + 20.0)
        assert got['state_interval_ms']['max'] <= longest, (name, report)
        assert got['joint_state_age_ms']['p99'] <= base['joint_state_age_ms']['p99'] + 10.0, \
            (name, report)
    # Engaged throughout: no release of any kind from 1.5 s to the end of the stopped window.
    engaged = [s.clutch_engaged for w, s in timed if start + 1.5 <= w < marks['cont']]
    assert engaged and all(engaged), [(round(w - start, 2), s.hold_reason) for w, s in timed
                                      if start + 1.5 <= w < marks['cont'] and
                                      not s.clutch_engaged][:5]


def test_feedback_node_stopped_teleop_goes_on_panel_no_data_no_late_vibration(stack):
    sid = fixture_session(stack, 'episode_marks_engaged')
    pid, probe, headset = stack['pids']['feedback_node'], stack['probe'], stack['headset']
    done = {}

    def stop_and_continue():
        time.sleep(5.5)                                   # after the first keep (5.0 s)
        done['stop'] = (now_ms(), time.monotonic())
        os.kill(pid, signal.SIGSTOP)
        time.sleep(4.8)                                   # start, stop, discard happen meanwhile
        done['cont'] = (now_ms(), time.monotonic())
        os.kill(pid, signal.SIGCONT)

    start, start_ms, end_ms = replay(stack, 'episode_marks_engaged', stop_and_continue)
    (stop_ms, stop_s), (cont_ms, cont_s) = done['stop'], done['cont']
    # Teleop went on while the node was stopped.
    targets = probe.targets_since(stop_s)
    armed = [1 for when, st in probe.timed_states_since(stop_s) if when <= cont_s and st.armed]
    assert len(armed) > 100
    assert len([1 for when, _ in targets if when <= cont_s]) > 100
    # No status reached the headset while stopped (after the bridge's slot aged out).
    assert not headset.of_type('teleop_status', stop_ms + 600.0, cont_ms)
    # Nothing older than its maximum age reached the headset; events from the stop: not felt.
    for t, m in headset.of_type('teleop_haptic', start_ms, end_ms):
        assert total_age_ms(t, m, sid) <= m['max_age_ms'], m
    events = events_since(probe, start)
    felt = {m['source_count'] for _, m in headset.of_type('teleop_haptic', start_ms, end_ms)}
    wall_cont = time.time() - (time.monotonic() - cont_s)
    wall_stop = wall_cont - (cont_s - stop_s)
    # Events from the stop until max age before the continue are older than max age by then.
    late = {c for c, _, w in events if wall_stop < w < wall_cont - MAX_AGE_MS / 1000.0}
    outside = {c for c, _, w in events if w < wall_stop - 0.1 or w > wall_cont + 0.1}
    assert late, events                                  # the window did contain events
    assert not felt & late, (felt, late)
    assert outside <= felt, (outside, felt, events)
    # The panel: NO DATA within 1.0 s (+ one period) of the last status, live again after.
    q = [stop_ms - 200.0, stop_ms + 1400.0, cont_ms - 100.0, cont_ms + 1000.0]
    shown = panel(stack, sid, q)
    assert [r['stale'] for r in shown] == [False, True, True, False], shown


def test_feedback_node_killed_teleop_goes_on_and_the_panel_shows_no_data(stack):
    sid = fixture_session(stack, 'episode_marks_engaged')
    pid, probe, headset = stack['pids']['feedback_node'], stack['probe'], stack['headset']
    done = {}

    def kill():
        time.sleep(3.0)                                   # engaged (grip from 1.2 s)
        done['kill'] = (now_ms(), time.monotonic())
        os.kill(pid, signal.SIGKILL)

    start, start_ms, end_ms = replay(stack, 'episode_marks_engaged', kill)
    kill_ms, kill_s = done['kill']
    assert len(probe.targets_since(kill_s)) > 300         # ~10 s engaged after the kill at 50 Hz
    assert any(s.armed for _, s in probe.timed_states_since(kill_s + 5.0))
    assert not headset.of_type('teleop_status', kill_ms + 600.0)
    assert not headset.of_type('teleop_haptic', kill_ms + 600.0)
    shown = panel(stack, sid, [kill_ms - 200.0, kill_ms + 1400.0, end_ms])
    assert [r['stale'] for r in shown] == [False, True, True], shown
    assert all('NO DATA' in ' '.join(r['lines']) for r in shown[1:])


def test_input_age_with_feedback_off(stack):
    """Reported, not asserted: input age with the bridge's feedback off (same fixture as on)."""
    stack['bridge'].stop()
    bridge = Process(['ros2', 'launch', 'ranger_xarm_teleop_bridge', 'bridge.launch.py',
                      f'relay_url:={stack["url"]}', 'feedback:=false'], stack['env'],
                     str(stack['logs'] / 'bridge_off.log'))
    try:
        time.sleep(3.0)
        n = len(stack['headset'].received)
        start, _, _ = replay(stack, 'episode_disarm')
        assert len(stack['headset'].received) == n          # nothing sent with feedback off
        stack['results']['input_age_off'] = input_ages(stack['probe'], start)
    finally:
        bridge.stop()
    results = stack['results']
    print('feedback mock results:', json.dumps(results, indent=1))
    out = os.environ.get('FEEDBACK_MOCK_OUT')
    if out:
        with open(out, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=1)


def input_ages(probe, start):
    ages = np.array([s.input_age for _, s in probe.timed_states_since(start)
                     if s.armed and np.isfinite(s.input_age)])
    if not len(ages):
        return None
    return {'n': len(ages), 'median_ms': 1e3 * float(np.median(ages)),
            'p99_ms': 1e3 * float(np.percentile(ages, 99))}
