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
Demonstration recording in mock execution (no cameras): episodes from the left controller.

One mock stack with ``record:=true`` for the module, on its own ROS domain, recording into
the test's temporary directory. The episode fixtures are generated here from
``make_fixtures.py`` (synthetic, too long to keep as files).
"""

import json
import os
import sys
import time

import numpy as np
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, EpisodeState
from ranger_xarm_teleop_record import metadata as md
from ranger_xarm_teleop_record.bag_io import read_episode
from ranger_xarm_teleop_record.checker import check_episode
from ranger_xarm_teleop_record.export import build_export, write_export
from ranger_xarm_teleop_record.profile import load_checks
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from stack_helpers import (assert_stack_nodes, child_environment, exited_node_patterns,
                           full_name, ignoring, launched_processes, MOCK_RECORD_NODES, Process,
                           READY, SERVO_FIXTURES)
from test_mock_gripper import GripperProbe

SESSION = 'test_session'
STATE_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL)


class RecordProbe(GripperProbe):
    """The stack probe (with the gripper's) plus the episode state."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.episode_states = []        # (monotonic, EpisodeState)
        self.node.create_subscription(EpisodeState, '/teleop/episode/state',
                                      self._on_episode, STATE_QOS)

    def _on_episode(self, msg):
        with self.lock:
            self.episode_states.append((time.monotonic(), msg))

    def episode_since(self, t):
        with self.lock:
            return [m for when, m in self.episode_states if when >= t]

    def episode_state(self):
        with self.lock:
            return self.episode_states[-1][1] if self.episode_states else None

    def graph(self, ignore=()):
        """
        Return the nodes and every topic's publisher and subscriber counts.

        Nodes matching a pattern of ``ignore`` are left out, and so are their publishers and
        subscribers (the nodes of launched processes that have exited, ``EXITING_NODES``), and
        a topic none of the rest uses.
        """
        nodes = tuple(sorted(ignoring(
            [full_name(n, ns) for n, ns in self.node.get_node_names_and_namespaces()], ignore)))
        names = [t for t, _ in self.node.get_topic_names_and_types()]
        if not ignore:
            return nodes, tuple(sorted(
                (t, self.node.count_publishers(t), self.node.count_subscribers(t))
                for t in names))

        def count(endpoints):
            return len(ignoring([full_name(e.node_name, e.node_namespace) for e in endpoints],
                                ignore))
        topics = [(t, count(self.node.get_publishers_info_by_topic(t)),
                   count(self.node.get_subscriptions_info_by_topic(t))) for t in names]
        return nodes, tuple(sorted(row for row in topics if row[1] or row[2]))


def fixture_file(directory, name):
    """Write one generated episode fixture; return its path."""
    sys.path.insert(0, SERVO_FIXTURES)
    try:
        import make_fixtures
    finally:
        sys.path.remove(SERVO_FIXTURES)
    path = os.path.join(directory, f'{name}.jsonl')
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        for text in getattr(make_fixtures, name)():
            handle.write(text + '\n')
    return path


def replay_file(env, path, log_path, start_delay=3.0):
    return Process(
        ['ros2', 'run', 'ranger_xarm_teleop_bridge', 'replay_node', '--ros-args',
         '-p', f'file:={path}', '-p', 'wait_for_subscriber_s:=15.0',
         '-p', f'start_delay_s:={start_delay}'], env, log_path)


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_record')
    data = tmp_path_factory.mktemp('record_data')
    md.write_labels(str(data), 'test-task', 'test-operator')
    env = child_environment(offset=11)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                      'rviz:=false', 'record:=true', f'record_data_dir:={data}',
                      f'record_session:={SESSION}'], env, str(logs / 'launch.log'))
    probe = RecordProbe(env)
    try:
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: probe.episode_state() is not None, 60.0, 'the episode state')
        probe.move_to(READY)
        wait_for_settled_graph(probe, launch)
        yield probe, env, logs, os.path.join(str(data), SESSION), launch
    finally:
        probe.close()
        launch.stop()
        print(launch.output()[-6000:])


def wait_for_settled_graph(probe, launch=None, quiet=5.0, timeout=90.0):
    """
    Wait until no node joins or leaves for ``quiet`` s (spawners start late).

    With ``launch``, nodes whose launched process has exited are left out: such a node can
    stay in the graph until its lease ends, and leave later.
    """
    end = time.monotonic() + timeout
    last, since = None, time.monotonic()
    while time.monotonic() < end:
        nodes = probe.graph(exited_node_patterns(launch) if launch else ())[0]
        if nodes != last:
            last, since = nodes, time.monotonic()
        elif time.monotonic() - since >= quiet:
            return
        time.sleep(0.25)
    raise AssertionError('the ROS graph did not settle')


def run_fixture(stack, name, seconds):
    probe, env, logs, session, _ = stack
    probe.disarm()
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 30.0, 'MoveIt owner')
    path = fixture_file(str(logs), name)
    before = {os.path.basename(p) for p in md.iter_episodes(session)}
    start = time.monotonic()
    process = replay_file(env, path, str(logs / f'replay_{name}.log'))
    try:
        assert process.wait(seconds) == 0, process.output()
    finally:
        process.stop()
    time.sleep(1.5)
    new = [p for p in md.iter_episodes(session) if os.path.basename(p) not in before]
    return start, [md.read_json(p) for p in new], new


def engaged_spans(states):
    spans, begin = [], None
    for when, s in states:
        if s.clutch_engaged and begin is None:
            begin = (when, s)
        if not s.clutch_engaged and begin is not None:
            spans.append((begin[0], when, s.hold_reason))
            begin = None
    return spans


def test_episode_presses_while_engaged_release_nothing_and_change_no_participant(stack):
    """INV-1: start, stop, keep, discard and mark during one engagement."""
    probe, env, logs, session, launch = stack
    probe.disarm()
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 30.0, 'MoveIt owner')
    # Nodes of launched processes that exited before the episodes (spawners) may still leave
    # the graph; a process that exits during them is a change, even one whose node lingers in
    # the graph (checked from the launch log below).
    exited = exited_node_patterns(launch)
    exited_before = launched_processes(launch)[1]
    path = fixture_file(str(logs), 'episode_marks_engaged')
    process = replay_file(env, path, str(logs / 'replay_marks.log'))
    graphs = []
    try:
        probe.wait_for(lambda: probe.state() is not None and probe.state().clutch_engaged,
                       40.0, 'the clutch to engage')
        engaged_at = time.monotonic()
        while process.wait(0.3) is None:
            graphs.append((time.monotonic(), probe.graph(exited)))
        assert process.proc.returncode == 0, process.output()
    finally:
        process.stop()
    exited_during = sorted(launched_processes(launch)[1] - exited_before)
    assert not exited_during, f'launched processes exited during the episodes: {exited_during}'
    time.sleep(1.5)
    with probe.lock:
        states = [(w, s) for w, s in probe.states if w >= engaged_at - 0.5]
    spans = engaged_spans(states)
    # one engagement of about 11.8 s, ended by the grip (G-2), not by anything else
    assert len(spans) == 1, [(round(b - a, 2), r) for a, b, r in spans]
    begin, end, reason = spans[0]
    assert end - begin > 10.5, end - begin
    assert reason == 5, f'released with hold reason {reason}'          # HOLD_CLUTCH_RELEASED
    events = [m for m in probe.episode_since(engaged_at) if m.last_event != 0]
    kinds = []
    for m in events:
        if not kinds or kinds[-1][0] != m.event_count:
            kinds.append((m.event_count, m.last_event, m.header.stamp))
    names = [EVENT_NAMES[k] for _, k, _ in kinds]
    assert names == ['START', 'MARK', 'STOP', 'KEEP', 'START', 'STOP', 'DISCARD', 'START',
                     'STOP', 'KEEP'], names
    # every graph snapshot while the episodes ran is the same: nothing joined or left
    inside = [g for w, g in graphs if begin + 0.5 <= w <= end - 0.5]
    assert len(inside) > 20
    changed = [(round(w - begin, 2), g) for w, g in graphs
               if begin + 0.5 <= w <= end - 0.5 and g != inside[0]]
    if changed:
        nodes0, topics0 = inside[0]
        nodes1, topics1 = changed[0][1]
        diff = (sorted(set(nodes0) ^ set(nodes1)), sorted(set(topics0) ^ set(topics1)))
        raise AssertionError(f'the ROS graph changed at +{changed[0][0]} s: {diff}')
    # cycle timing during the episode presses against the engaged time before the first one
    stamps = [(w, s.header.stamp.sec + s.header.stamp.nanosec * 1e-9) for w, s in states
              if begin <= w <= end]
    first_press = begin + 0.6
    gaps_before = [b[1] - a[1] for a, b in zip(stamps, stamps[1:]) if b[0] < first_press]
    gaps_during = [b[1] - a[1] for a, b in zip(stamps, stamps[1:]) if b[0] >= first_press]
    print(f'cycle intervals before: max {max(gaps_before) * 1e3:.1f} ms; during episodes: '
          f'max {max(gaps_during) * 1e3:.1f} ms, '
          f'{sum(g > 0.03 for g in gaps_during)} of {len(gaps_during)} over 30 ms')
    assert max(gaps_during) < 0.1                     # never near the input timeout (G-3)


EVENT_NAMES = {v: k[len('EVENT_'):] for k, v in vars(EpisodeState).items()
               if k.startswith('EVENT_') and isinstance(v, int)}


def test_ten_scripted_episodes_pass_the_checker_and_export(stack):
    probe, env, logs, session, _ = stack
    start, sidecars, paths = run_fixture(stack, 'episodes_ten', 150.0)
    statuses = [s['episode']['status'] for s in sidecars]
    assert statuses == ['kept'] * 10 + ['discarded'], statuses
    checks = load_checks()
    reports = []
    for path, sidecar in zip(paths, sidecars):
        assert md.missing(sidecar) == [], md.missing(sidecar)
        assert len(sidecar['episode']['marks']) == 1
        assert sidecar['base']['mode'].startswith('parked')
        # Mock execution publishes joint effort as NaN: noted, not failed.
        fields = sidecar['fields']['joint_states']
        assert fields['absent'] == ['effort'] and fields['how'] == {'effort': 'NaN'}, fields
        report = check_episode(read_episode(session, path), checks)
        reports.append(report)
        assert report['verdict'] == 'PASS', (path, report['reasons'])
        assert report['fields']['joint_states']['absent'] == ['effort']
    with open(os.path.join(str(logs), 'reports.json'), 'w', encoding='utf-8') as handle:
        json.dump(reports, handle, default=str)
    assert [r['usable'] for r in reports] == [True] * 10 + [False]
    ep = read_episode(session, paths[0])
    arrays, meta = build_export(ep, checks, verdict=reports[0]['verdict'])
    out = os.path.join(str(logs), 'export')
    written = write_export(out, arrays, meta, ep)
    assert 'data.npz' in written and 'export.json' in written
    assert arrays['clutch_engaged'].any() and not arrays['clutch_engaged'].all()
    assert not arrays['action_valid'][~arrays['clutch_engaged']].any()
    for name, value in arrays.items():
        if name.endswith('_age'):
            assert not np.any(value[np.isfinite(value)] < 0.0), name
    # Recorded now: the tool pose carries the stamp of its own joint state.
    assert meta['tool_pose_stamp'].startswith('TeleopState.measured_tool_pose_stamp')
    valid = arrays['tool_pose_measured_valid']
    assert valid.any() and np.isfinite(arrays['tool_pose_measured_age'][valid]).all()
    stamps = ep.streams['state'].fields['measured_tool_pose_stamp']
    assert np.all(stamps[ep.streams['state'].fields['measured_tool_pose_valid']] > 0.0)
    text = open(os.path.join(out, 'export.json'), encoding='utf-8').read()
    assert os.path.expanduser('~') not in text and session not in text


def test_disarm_stops_the_episode_and_a_start_while_disarmed_is_refused(stack):
    probe, env, logs, session, _ = stack
    start, sidecars, _ = run_fixture(stack, 'episode_disarm', 60.0)
    assert len(sidecars) == 1, 'one press with a lost frame inside must start one episode'
    e = sidecars[0]['episode']
    assert e['status'] == 'kept'
    assert e['stop_reason'] == 'disarmed'
    assert 'DISARMED' in {f['flag'] for f in e['flags']}
    refused = [m for m in probe.episode_since(start) if m.last_event == EpisodeState.EVENT_REFUSED]
    assert any('not armed' in m.detail for m in refused), {m.detail for m in refused}


def test_no_other_stack_or_test_shares_this_domain(stack):
    """Isolation: the domain holds this stack's nodes and the probe, nothing else."""
    assert_stack_nodes(stack[0], stack[4], MOCK_RECORD_NODES)
