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
Demonstration recording in headless Gazebo with both cameras on (opt-in).

``TELEOP_TEST_GAZEBO=1`` runs it. The session goes to ``TELEOP_TEST_RECORD_DIR`` when set
(outside any checkout), else to the test's temporary directory. Image streams are large:
the ten episodes record about 4 s each of two 848x480 colour streams.
"""

import json
import os
import time

import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner
from ranger_xarm_teleop_record import metadata as md
from ranger_xarm_teleop_record.bag_io import read_episode
from ranger_xarm_teleop_record.checker import check_episode
from ranger_xarm_teleop_record.export import build_export, write_export
from ranger_xarm_teleop_record.profile import load_checks
from stack_helpers import (child_environment, cpu_seconds, kill_gz_partition, node_pids,
                           Process, READY)
from test_mock_record import (engaged_spans, EVENT_NAMES, fixture_file, RecordProbe,
                              replay_file, wait_for_settled_graph)

pytestmark = pytest.mark.skipif(os.environ.get('TELEOP_TEST_GAZEBO') != '1',
                                reason='set TELEOP_TEST_GAZEBO=1 to run the Gazebo tests')

MOVEIT, MOVEIT_G = 'xarm_xarm6_traj_controller', 'xarm_xarm_gripper_traj_controller'
TELEOP_G = 'teleop_gripper_controller'


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('gazebo_record')
    data = os.environ.get('TELEOP_TEST_RECORD_DIR') or str(tmp_path_factory.mktemp('data'))
    os.makedirs(data, exist_ok=True)
    session = 'gazebo_' + time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    md.write_labels(data, 'test-task', 'test-operator')
    env = child_environment(offset=12)
    env['GZ_PARTITION'] = f'teleop_record_test_{os.getpid()}'
    env['GZ_IP'] = '127.0.0.1'
    sim = Process(['ros2', 'launch', 'ranger_xarm_gazebo', 'sim.launch.py', 'headless:=true',
                   'sensors:=true'], env, str(logs / 'sim.log'))
    probe = RecordProbe(env, moveit_controller=MOVEIT, moveit_gripper=MOVEIT_G)
    teleop = None
    try:
        probe.wait_for(lambda: probe.joint_positions() is not None, 180.0, 'joint states')
        probe.wait_for(lambda: probe.controllers().get(MOVEIT) == 'active'
                       and probe.controllers().get(MOVEIT_G) == 'active', 120.0,
                       'the arm and gripper controllers')
        teleop = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_sim.launch.py',
                          'record:=true', f'record_data_dir:={data}',
                          f'record_session:={session}'], env, str(logs / 'teleop.log'))
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 120.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.controllers().get(TELEOP_G) == 'inactive', 60.0,
                       'the teleop gripper controller to be loaded inactive')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: probe.episode_state() is not None, 60.0, 'the episode state')
        time.sleep(5.0)             # MoveIt Servo start-up and verification
        probe.move_to(READY, seconds=4.0)
        probe.move_gripper(0.0, 1.5)
        wait_for_settled_graph(probe)
        yield probe, env, logs, os.path.join(data, session)
    finally:
        probe.close()
        if teleop is not None:
            teleop.stop()
            print(teleop.output()[-4000:])
        sim.stop()
        kill_gz_partition(env['GZ_PARTITION'])


def replay_new(stack, name, seconds):
    probe, env, logs, session = stack
    probe.disarm()
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 60.0, 'MoveIt owner')
    path = fixture_file(str(logs), name)
    before = {os.path.basename(p) for p in md.iter_episodes(session)}
    start = time.monotonic()
    process = replay_file(env, path, str(logs / f'replay_{name}.log'))
    try:
        assert process.wait(seconds) == 0, process.output()
    finally:
        process.stop()
    time.sleep(3.0)
    new = [p for p in md.iter_episodes(session) if os.path.basename(p) not in before]
    return start, new


def test_presses_while_engaged_with_cameras_release_nothing(stack):
    """INV-1 in Gazebo with both cameras and the recorder running."""
    probe, env, logs, session = stack
    probe.disarm()
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 60.0, 'MoveIt owner')
    path = fixture_file(str(logs), 'episode_marks_engaged')
    # Reported, not asserted: CPU of the operator feedback node beside the episode node's
    # (both read the 50 Hz teleop state), this test's own processes only.
    mine = f'GZ_PARTITION={env["GZ_PARTITION"]}'
    pids = {n: node_pids(n, environ=mine) for n in ('feedback_node', 'episode_node')}
    pids = {n: v[0] for n, v in pids.items() if len(v) == 1}
    cpu0, t0 = {n: cpu_seconds(p) for n, p in pids.items()}, time.monotonic()
    process = replay_file(env, path, str(logs / 'replay_marks.log'))
    graphs = []
    try:
        probe.wait_for(lambda: probe.state() is not None and probe.state().clutch_engaged,
                       60.0, 'the clutch to engage')
        engaged_at = time.monotonic()
        while process.wait(0.3) is None:
            graphs.append((time.monotonic(), probe.graph()))
        assert process.proc.returncode == 0, process.output()
    finally:
        process.stop()
    cpu = {n: round(100.0 * (cpu_seconds(p) - cpu0[n]) / (time.monotonic() - t0), 1)
           for n, p in pids.items()}
    print(f'Gazebo CPU, % of one core during the episode fixture: {cpu}')
    with open(os.path.join(str(logs), 'cpu.json'), 'w', encoding='utf-8') as handle:
        json.dump(cpu, handle)
    time.sleep(3.0)
    with probe.lock:
        states = [(w, s) for w, s in probe.states if w >= engaged_at - 0.5]
    spans = engaged_spans(states)
    assert len(spans) == 1, [(round(b - a, 2), r) for a, b, r in spans]
    begin, end, reason = spans[0]
    assert end - begin > 10.5 and reason == 5, (end - begin, reason)
    names = []
    last = None
    for m in probe.episode_since(engaged_at):
        if m.last_event and m.event_count != last:
            names.append(EVENT_NAMES[m.last_event])
            last = m.event_count
    assert names == ['START', 'MARK', 'STOP', 'KEEP', 'START', 'STOP', 'DISCARD', 'START',
                     'STOP', 'KEEP'], names
    inside = [g for w, g in graphs if begin + 0.5 <= w <= end - 0.5]
    assert len(inside) > 20 and all(g == inside[0] for g in inside)
    walls = [w for w, s in states if begin <= w <= end]
    gaps = [b - a for a, b in zip(walls, walls[1:])]
    print(f'Gazebo engaged: state arrival gaps max {max(gaps) * 1e3:.1f} ms, '
          f'{sum(g > 0.05 for g in gaps)} of {len(gaps)} over 50 ms')


def test_ten_scripted_episodes_pass_the_checker(stack):
    probe, env, logs, session = stack
    start, paths = replay_new(stack, 'episodes_ten', 240.0)
    sidecars = [md.read_json(p) for p in paths]
    assert [s['episode']['status'] for s in sidecars] == ['kept'] * 10 + ['discarded']
    checks = load_checks()
    reports = []
    for path, sidecar in zip(paths, sidecars):
        assert md.missing(sidecar) == [], md.missing(sidecar)
        report = check_episode(read_episode(session, path), checks)
        md.write_json_atomic(os.path.join(session, 'checks', os.path.basename(path)), report)
        reports.append(report)
    summary = [(os.path.basename(p), r['verdict'], r['reasons'][:3],
                sorted({f['flag'] for f in r['flags']})) for p, r in zip(paths, reports)]
    with open(os.path.join(str(logs), 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))
    assert all(r['verdict'] == 'PASS' for r in reports[:10]), summary
    ep = read_episode(session, paths[0])
    arrays, meta = build_export(ep, checks, verdict=reports[0]['verdict'])
    written = write_export(os.path.join(session, 'exports', '0001'), arrays, meta, ep)
    assert any(w.endswith('.png') for w in written)
