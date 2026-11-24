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
Base driving (G-19) in headless Gazebo with the base drivable (opt-in).

``TELEOP_TEST_GAZEBO=1`` runs it. The simulator runs with ``drive_base:=true`` (the base on
its wheels, driven from /cmd_vel) and the cameras on; the stack with ``base:=true`` and the
recorder. A recording calibrates, arms, starts an episode, drives, stops, strokes the arm and
drives again; every G-19 trigger is shown to bring the base to rest by ground truth. Stops are
measured: time and distance from the zero to rest, the arm's joints and the tool (relative to
the base) after the abrupt stop, the tool's speed in the room during a turn and a sideways
move. The last test kills the base node.
"""

import json
import math
import os
import signal
import threading
import time

from base_helpers import after_stop, B, BaseWatcher, calibrate_and_arm, describe, phase, play
import numpy as np
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner
from ranger_xarm_teleop_record import metadata as md
from ranger_xarm_teleop_record.bag_io import read_episode
from ranger_xarm_teleop_record.checker import check_episode
from ranger_xarm_teleop_record.profile import load_checks
import rclpy.time
from stack_helpers import (child_environment, kill_gz_partition, node_pids, Process, READY,
                           start_input)
from test_mock_record import RecordProbe, wait_for_settled_graph

pytestmark = pytest.mark.skipif(os.environ.get('TELEOP_TEST_GAZEBO') != '1',
                                reason='set TELEOP_TEST_GAZEBO=1 to run the Gazebo tests')

MOVEIT, MOVEIT_G = 'xarm_xarm6_traj_controller', 'xarm_xarm_gripper_traj_controller'
FORWARD = (0.0, -1.0)
SIDEWAYS = (1.0, 0.0)            # the robot's right
STRETCH = [0.0, 0.85, -1.4, 0.0, 0.0, 0.0]      # 0.70 m horizontal reach (base task, step 1)
REST_V, REST_W = 0.005, 0.01     # ground truth at rest: below these for REST_FOR
REST_FOR = 0.3


class ToolSampler:
    """The tool in the base frame (TF base_link -> xarm_link_tcp) at 50 Hz."""

    def __init__(self, probe):
        self.probe, self.samples, self.stop = probe, [], threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop.is_set():
            try:
                tf = self.probe.tf_buffer.lookup_transform('base_link', 'xarm_link_tcp',
                                                           rclpy.time.Time())
                p = tf.transform.translation
                self.samples.append((time.monotonic(), p.x, p.y, p.z))
            except Exception:           # noqa: B902 - not yet known: try again
                pass
            time.sleep(0.02)

    def since(self, t):
        return [s for s in list(self.samples) if s[0] >= t]


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('gazebo_base')
    data = os.environ.get('TELEOP_TEST_RECORD_DIR') or str(tmp_path_factory.mktemp('data'))
    os.makedirs(data, exist_ok=True)
    session = 'gazebo_base_' + time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    md.write_labels(data, 'test-task', 'test-operator')
    env = child_environment(offset=28)
    env['GZ_PARTITION'] = f'teleop_base_test_{os.getpid()}'
    env['GZ_IP'] = '127.0.0.1'
    sim = Process(['ros2', 'launch', 'ranger_xarm_gazebo', 'sim.launch.py', 'headless:=true',
                   'sensors:=true', 'drive_base:=true'], env, str(logs / 'sim.log'))
    probe = RecordProbe(env, moveit_controller=MOVEIT, moveit_gripper=MOVEIT_G)
    teleop, tool = None, None
    results = []
    try:
        watch = BaseWatcher(probe)
        probe.wait_for(lambda: probe.joint_positions() is not None, 180.0, 'joint states')
        probe.wait_for(lambda: probe.controllers().get(MOVEIT) == 'active', 120.0,
                       'the arm controller')
        probe.wait_for(lambda: watch.since('odom', 0.0) and watch.since('gt', 0.0), 120.0,
                       'odometry and ground truth of the drivable base')
        teleop = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_sim.launch.py',
                          'base:=true', 'record:=true', f'record_data_dir:={data}',
                          f'record_session:={session}'], env, str(logs / 'teleop.log'))
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 120.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: probe.episode_state() is not None, 60.0, 'the episode state')
        probe.wait_for(lambda: watch.last() is not None and watch.last().enabled
                       and watch.last().at_rest, 60.0, 'the base node, enabled, base at rest')
        time.sleep(5.0)             # MoveIt Servo start-up and verification
        probe.move_to(READY, seconds=4.0)
        start_input(probe, env, logs)
        wait_for_settled_graph(probe)
        tool = ToolSampler(probe)
        yield {'probe': probe, 'env': env, 'logs': logs, 'watch': watch, 'tool': tool,
               'session': os.path.join(data, session), 'results': results, 'teleop': teleop}
    finally:
        if tool is not None:
            tool.stop.set()
        probe.close()
        if teleop is not None:
            teleop.stop()
        sim.stop()
        kill_gz_partition(env['GZ_PARTITION'])
        with open(os.path.join(str(logs), 'results.txt'), 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(results) + '\n')
        for line in results:
            print(line)


def note(stack, text):
    stack['results'].append(text)


def gt_rows(watch, t0, t1=math.inf):
    rows = []
    for w, m in watch.since('gt', t0):
        if w > t1:
            break
        p, q, tw = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        rows.append((w, p.x, p.y, yaw, math.hypot(tw.linear.x, tw.linear.y), abs(tw.angular.z)))
    return rows


def ground_rest(watch, stop_at, timeout=4.0):
    """Wait for the base at rest by ground truth; return (seconds, mm, deg) from ``stop_at``."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rows = gt_rows(watch, stop_at)
        still_since = None
        for r in rows:
            if r[4] < REST_V and r[5] < REST_W:
                still_since = r[0] if still_since is None else still_since
                if r[0] - still_since >= REST_FOR:
                    first = rows[0]
                    at = [x for x in rows if x[0] <= still_since][-1]
                    return (still_since - stop_at,
                            1e3 * math.hypot(at[1] - first[1], at[2] - first[2]),
                            math.degrees(abs((at[3] - first[3] + math.pi) % (2 * math.pi)
                                             - math.pi)))
            else:
                still_since = None
        time.sleep(0.1)
    raise AssertionError(f'the base did not come to rest by ground truth within {timeout} s')


def arm_after(stack, stop_at, seconds=2.0):
    """Return (largest joint deviation mrad, largest tool deviation in the base frame mm)."""
    probe, tool = stack['probe'], stack['tool']
    time.sleep(max(0.0, stop_at + seconds - time.monotonic()))
    with probe.lock:
        joints = [(w, np.array(q)) for w, q in probe.joints if w >= stop_at - 0.1]
    q0 = joints[0][1]
    dq = max(float(np.max(np.abs(q - q0))) for _, q in joints)
    tl = tool.since(stop_at - 0.1)
    p0 = np.array(tl[0][1:])
    dp = max(float(np.linalg.norm(np.array(s[1:]) - p0)) for s in tl)
    return dq * 1e3, dp * 1e3


def tool_world_speed(stack, t0, t1):
    """Return the tool's peak speed in the room (ground truth composed with TF) and its reach."""
    watch, tool = stack['watch'], stack['tool']
    gt = gt_rows(watch, t0 - 0.2, t1 + 0.2)
    samples = [s for s in tool.since(t0) if s[0] <= t1]
    times = np.array([g[0] for g in gt])
    pts = []
    for w, x, y, _ in samples:
        bx = np.interp(w, times, [g[1] for g in gt])
        by = np.interp(w, times, [g[2] for g in gt])
        yaw = np.interp(w, times, np.unwrap([g[3] for g in gt]))
        pts.append((w, bx + math.cos(yaw) * x - math.sin(yaw) * y,
                    by + math.sin(yaw) * x + math.cos(yaw) * y))
    speeds = [math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0])
              for a, b in zip(pts[::5], pts[5::5]) if b[0] > a[0]]
    reach = float(np.median([math.hypot(s[1], s[2]) for s in samples]))
    return max(speeds), reach


def drive_and_stop(stack, name, how, end=None, seconds=2.5):
    """Drive with ``how`` for ``seconds``, then ``end``; return (start, stop arrival)."""
    watch = stack['watch']
    end = end or phase(3.0)
    start, _, _ = play(stack, name, [phase(0.4), phase(seconds, grip=True, **how), end])
    stack['probe'].wait_for(lambda: after_stop(watch.commands(start)), 2.0, 'the zero')
    time.sleep(0.5)                             # and then nothing
    tail = after_stop(watch.commands(start))
    assert len(tail) == 1 and tail[0][1:] == (0.0, 0.0, 0.0), tail[:5]
    return start, tail[0][0]


def test_only_the_base_node_publishes_the_base_command(stack):
    node = stack['probe'].node
    for topic in ('/cmd_vel', '/teleop/base/cmd_vel'):
        assert {i.node_name for i in node.get_publishers_info_by_topic(topic)} == {'base_node'}


def test_drive_stop_stroke_and_drive_again_in_one_recorded_episode(stack):
    """The task's done criterion, recorded as an episode that the checker passes."""
    probe, watch = stack['probe'], stack['watch']
    probe.disarm()
    before = {os.path.basename(p) for p in md.iter_episodes(stack['session'])}
    start, begins, _ = play(stack, 'done', calibrate_and_arm() + [
        phase(0.2, x=True), phase(1.0),                                  # 5-6: episode start
        phase(2.0, grip=True, left=FORWARD), phase(1.5),                 # 7-8: drive, stop
        phase(2.0, rgrip=True, move=(0.0, 0.0, -0.05)), phase(0.5),      # 9-10: arm stroke
        phase(2.0, grip=True, left=SIDEWAYS), phase(1.5),                # 11-12: drive again
        phase(2.0, rgrip=True, move=(0.0, 0.0, 0.05)), phase(0.5),       # 13-14: stroke
        phase(0.2, x=True), phase(1.0), phase(0.2, y=True), phase(2.0)])  # stop, keep
    states = probe.timed_states_since(start)
    cmds = watch.commands(start)
    drives = []
    for k in (7, 11):
        t0 = start + begins[k]
        drives.append([c for c in cmds if t0 <= c[0] <= start + begins[k + 2]])
    for d in drives:
        assert any(abs(vx) > 0.1 or abs(vy) > 0.1 for _, vx, vy, _ in d)
    engaged = [w for w, s in states if s.clutch_engaged]
    assert any(start + begins[9] <= w <= start + begins[11] for w in engaged)
    assert any(start + begins[13] <= w <= start + begins[15] for w in engaged)
    moving = [r[0] for r in gt_rows(watch, start) if r[4] > 0.01 or r[5] > 0.02]
    commanded = [c[0] for c in cmds if c[1] or c[2] or c[3]]
    clash = [w for w in engaged if any(abs(w - m) < 0.02 for m in moving + commanded)]
    assert not clash, f'clutch engaged while the base moved or was commanded at {clash[:3]}'
    moved = gt_rows(watch, start)
    travel = math.hypot(moved[-1][1] - moved[0][1], moved[-1][2] - moved[0][2])
    assert travel > 0.3, travel
    time.sleep(3.0)
    paths = [p for p in md.iter_episodes(stack['session']) if os.path.basename(p) not in before]
    assert len(paths) == 1, paths
    report = check_episode(read_episode(stack['session'], paths[0]), load_checks())
    flags = sorted({f['flag'] for f in report['flags']})
    with open(os.path.join(str(stack['logs']), 'done_report.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(report, handle, indent=1, default=str)
    assert report['base']['mode'] == 'moved', report['base']
    assert report['base']['clutch_overlap']['checked'], report['base']['clutch_overlap']
    assert 'BASE_DURING_CLUTCH' not in flags, flags
    note(stack, f'done criterion: episode {os.path.basename(paths[0])} verdict '
                f'{report["verdict"]} flags {flags}; base travel {travel:.3f} m; summary '
                f'{report["base"].get("summary")}')
    probe.disarm()


@pytest.mark.parametrize('posture', ['ready', 'stretched'])
def test_stops_from_each_cap_and_the_tools_speed_in_the_room(stack, posture):
    """Release at each cap: stop time and distance, the arm after the abrupt stop."""
    probe, watch = stack['probe'], stack['watch']
    probe.disarm()
    probe.move_to(READY if posture == 'ready' else STRETCH, seconds=4.0)
    time.sleep(1.0)
    for name, how in (('forward', {'left': FORWARD}), ('sideways', {'left': SIDEWAYS}),
                      ('turn', {'right': (-1.0, 0.0)})):
        start, stop_at = drive_and_stop(stack, f'cap_{posture}_{name}', how)
        rest_s, mm, deg = ground_rest(watch, stop_at)
        dq, dp = arm_after(stack, stop_at)
        speed, reach = tool_world_speed(stack, start + 1.0, stop_at)
        note(stack, f'stop from the {name} cap at {posture} (release): rest {rest_s:.2f} s, '
                    f'{mm:.1f} mm, {deg:.2f} deg after the zero; arm: joints {dq:.2f} mrad, '
                    f'tool in the base frame {dp:.2f} mm; tool speed in the room while '
                    f'driving {speed:.3f} m/s (reach {reach:.2f} m)')
        assert rest_s < 1.0
        time.sleep(1.0)
    probe.move_to(READY, seconds=4.0)


@pytest.mark.parametrize('name, end, reason', [
    ('stale', phase(0.0), B.REASON_INPUT_STALE),
    ('left_missing', phase(3.0, left_missing=True), B.REASON_CONTROLLER_MISSING),
    ('session', phase(3.0, grip=True, left=FORWARD, session='new'), B.REASON_SESSION_CHANGED),
    ('seq_back', phase(3.0, grip=True, left=FORWARD, back=40), B.REASON_SEQ_BACKWARDS),
])
def test_each_g19_trigger_brings_the_base_to_rest(stack, name, end, reason):
    watch = stack['watch']
    count = watch.last().event_count
    start, stop_at = drive_and_stop(stack, f'trigger_{name}', {'left': FORWARD}, end=end)
    _, stopped = watch.wait(start, lambda m: m.state == B.STATE_STOPPED
                            and m.event_count > count, 3.0, 'a stop')
    assert stopped.reason == reason, describe(stopped)
    rest_s, mm, _ = ground_rest(watch, stop_at)
    note(stack, f'stop by {name}: rest {rest_s:.2f} s and {mm:.1f} mm after the zero')


def test_the_right_grip_during_a_drive_is_refused_and_the_drive_goes_on(stack):
    probe, watch = stack['probe'], stack['watch']
    probe.disarm()
    start, begins, _ = play(stack, 'clutch_attempt', calibrate_and_arm() + [
        phase(1.5, grip=True, left=FORWARD),
        phase(1.5, grip=True, left=FORWARD, rgrip=True, move=(0.0, 0.0, -0.05)),
        phase(1.5)])
    t0, t1 = start + begins[6], start + begins[7]
    assert not [s for w, s in probe.timed_states_since(t0) if w <= t1 and s.clutch_engaged]
    assert any(vx > 0.15 for w, vx, _, _ in watch.commands(t0) if w <= t1)
    probe.disarm()


def test_driving_does_not_disturb_the_arm_or_the_teleop_cycle(stack):
    """Armed, clutch released: the arm's joints and the teleop state's period, parked, driving."""
    probe = stack['probe']
    probe.disarm()
    start, begins, _ = play(stack, 'cycle', calibrate_and_arm() + [
        phase(4.0), phase(4.0, grip=True, left=SIDEWAYS), phase(2.0)])
    out = []
    for label, k in (('parked', 5), ('driving', 6)):
        t0, t1 = start + begins[k] + 0.5, start + begins[k] + 4.0
        walls = [w for w, _ in probe.timed_states_since(t0) if w <= t1]
        gaps = np.diff(walls) * 1e3
        with probe.lock:
            joints = [np.array(q) for w, q in probe.joints if t0 <= w <= t1]
        dq = float(np.max(np.abs(np.array(joints) - joints[0]))) * 1e3
        out.append((label, float(np.percentile(gaps, 50)), float(np.percentile(gaps, 99)),
                    float(gaps.max()), dq))
    for label, p50, p99, worst, dq in out:
        note(stack, f'{label}: teleop state period p50 {p50:.1f} p99 {p99:.1f} max '
                    f'{worst:.1f} ms; arm joints moved {dq:.3f} mrad')
    assert out[1][4] < 5.0, 'driving moved the held arm'
    probe.disarm()


def test_the_base_node_killed_mid_drive_stops_by_the_bases_own_timeout(stack):
    """Runs last: SIGKILL sends nothing; the simulated base's command timeout stops it."""
    watch = stack['watch']
    pids = node_pids('base_node', environ=f'GZ_PARTITION={stack["env"]["GZ_PARTITION"]}')
    assert len(pids) == 1, pids
    phases = [phase(0.4), phase(4.0, grip=True, left=FORWARD)]
    start, _, process = play(stack, 'sigkill', phases, wait=False)
    stack['probe'].wait_for(lambda: any(vx > 0.19 for _, vx, _, _ in watch.commands(start)),
                            5.0, 'the drive at the cap')
    os.kill(pids[0], signal.SIGKILL)
    killed = time.monotonic()
    last = watch.commands(start)[-1][0]
    rest_s, mm, _ = ground_rest(watch, last, timeout=6.0)
    process.wait(20.0)
    note(stack, f'base node SIGKILL at the forward cap: no zero; the last command '
                f'{last - killed:+.3f} s from the kill; rest {rest_s:.2f} s and {mm:.1f} mm after '
                'the last command')
