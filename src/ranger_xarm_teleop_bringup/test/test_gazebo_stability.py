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
The stability monitor in headless Gazebo against the wheels' loads (opt-in, INV-3, INV-2).

``TELEOP_TEST_GAZEBO=1`` runs it. The simulator runs with the base on its wheels and its
sensors (the IMU is bridged only then); the stack with ``base:=true stability:=true`` and the
recorder. The test drives the base directly on /cmd_vel, beyond the teleop caps, moves the arm
through MoveIt's controller and spawns a ramp and a bar through the world's create service.
The reference is the wheels' loads from the suspension (``stability_helpers``); the monitor's
figure is judged against it by the frozen rule. The last test kills the monitor.
"""

import hashlib
import json
import math
import os
import signal
import threading
import time

from ament_index_python.packages import get_package_share_directory
from base_helpers import BaseWatcher, calibrate_and_arm, phase, play
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, StabilityState
from ranger_xarm_teleop_record import metadata as md
from ranger_xarm_teleop_record.bag_io import read_episode
from ranger_xarm_teleop_record.checker import check_episode
from ranger_xarm_teleop_record.profile import load_checks
from reset_helpers import node_pid
import stability_helpers as sh
from stack_helpers import (child_environment, graph_nodes, kill_gz_partition, Process, READY,
                           start_input)
from test_mock_record import RecordProbe, wait_for_settled_graph

pytestmark = pytest.mark.skipif(os.environ.get('TELEOP_TEST_GAZEBO') != '1',
                                reason='set TELEOP_TEST_GAZEBO=1 to run the Gazebo tests')

OFFSET = 30
MOVEIT, MOVEIT_G = 'xarm_xarm6_traj_controller', 'xarm_xarm_gripper_traj_controller'
ARM = [f'xarm_joint{i}' for i in range(1, 7)]
FORWARD = (0.0, -1.0)
POSTURES = {
    'ready': list(READY),
    'stretch_fwd': [0.0, 0.85, -1.40, 0.0, 0.0, 0.0],
    'stretch_left': [math.pi / 2, 0.85, -1.40, 0.0, 0.0, 0.0],
    'stretch_right': [-math.pi / 2, 0.85, -1.40, 0.0, 0.0, 0.0],
    'tucked': [0.0, -1.0, -0.3, 0.0, 1.3, 0.0],
}
SHARE = get_package_share_directory('ranger_xarm_teleop_bringup')
SIM_MASS_FILE = os.path.join(SHARE, 'config', 'stability_mass_SIMULATION_ONLY.yaml')
WORLD = 'empty_ground'


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class Recorder:
    """The suspension, the monitor's state, ground truth and /cmd_vel, on simulation time."""

    def __init__(self, probe):
        self.lock = threading.Lock()
        self.susp, self.states, self.gt, self.cmds = [], [], [], []
        self.sim = 0.0
        n = probe.node
        n.create_subscription(DynamicJointState, '/dynamic_joint_states', self._djs, 100)
        n.create_subscription(StabilityState, '/teleop/stability/state', self._state, 100)
        n.create_subscription(Odometry, '/ground_truth/odom', self._gt, 50)
        n.create_subscription(Twist, '/cmd_vel', self._cmd, 50)

    def _djs(self, m):
        pos, vel = {}, {}
        for name, iv in zip(m.joint_names, m.interface_values):
            if name.endswith('_suspension_joint'):
                values = dict(zip(iv.interface_names, iv.values))
                pos[name], vel[name] = values.get('position'), values.get('velocity', 0.0)
        with self.lock:
            self.sim = max(self.sim, stamp(m.header))
            if pos:
                self.susp.append((stamp(m.header), pos, vel))

    def _state(self, m):
        with self.lock:
            self.states.append((stamp(m.header), time.monotonic(), m))

    def _gt(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        with self.lock:
            self.gt.append((stamp(m.header), p.x, p.y, q.x, q.y, q.z, q.w))

    def _cmd(self, m):
        with self.lock:
            self.cmds.append((self.sim, m.linear.x, m.linear.y, m.angular.z))

    def now(self):
        with self.lock:
            return self.sim

    def sleep(self, seconds):
        """Sleep ``seconds`` of simulation time."""
        t0, w0 = self.now(), time.monotonic()
        while self.now() - t0 < seconds:
            time.sleep(0.01)
            assert time.monotonic() - w0 < 60.0 * seconds + 30.0, 'simulation time stalled'

    def position(self):
        with self.lock:
            return self.gt[-1][1:3] if self.gt else None

    def down_between(self, t0, t1):
        """Return the mean direction of gravity in the body (ground truth) over [t0, t1]."""
        with self.lock:
            rows = [g for g in self.gt if t0 <= g[0] <= t1]
        downs = []
        for _, _, _, x, y, z, w in rows:
            # The world's -z in body axes: minus the third row of the rotation matrix.
            downs.append(-np.array([2 * (x * z - y * w), 2 * (y * z + x * w),
                                    1 - 2 * (x * x + y * y)]))
        d = np.mean(downs, axis=0)
        return d / np.linalg.norm(d)

    def last_state(self):
        with self.lock:
            return self.states[-1][2] if self.states else None


def box_sdf(name, size, pose):
    sx, sy, sz = size
    x, y, z, r = pose
    return (f'<?xml version="1.0"?><sdf version="1.9"><model name="{name}"><static>true</static>'
            f'<pose>{x} {y} {z} {r} 0 0</pose><link name="link"><collision name="c"><geometry>'
            f'<box><size>{sx} {sy} {sz}</size></box></geometry><surface><friction><ode><mu>1.5'
            '</mu><mu2>1.5</mu2></ode><torsional><coefficient>0.0</coefficient></torsional>'
            '</friction></surface></collision><visual name="v"><geometry><box><size>'
            f'{sx} {sy} {sz}</size></box></geometry></visual></link></model></sdf>')


def side_ramp(name, deg, y0, x_c, length=3.0, width=3.0, thick=1.2):
    """Return a static box whose top face rises along +y from (x_c, y0, -0.005)."""
    th = math.radians(deg)
    u = (0.0, math.cos(th), math.sin(th))
    n = (0.0, -math.sin(th), math.cos(th))
    c = [(x_c, y0, -0.005)[i] + length / 2 * u[i] - thick / 2 * n[i] for i in range(3)]
    return box_sdf(name, (width, length, thick), (c[0], c[1], c[2], th))


def spawn(name, sdf):
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.transport13 import Node as GzNode
    req = EntityFactory()
    req.sdf = sdf
    ok, rep = GzNode().request(f'/world/{WORLD}/create', req, EntityFactory, Boolean, 5000)
    assert ok and rep.data, f'spawning {name} failed'


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('gazebo_stability')
    data = os.environ.get('TELEOP_TEST_RECORD_DIR') or str(tmp_path_factory.mktemp('data'))
    os.makedirs(data, exist_ok=True)
    session = 'gazebo_stability_' + time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    md.write_labels(data, 'test-task', 'test-operator')
    env = child_environment(offset=OFFSET)
    env['GZ_PARTITION'] = f'teleop_stability_test_{os.getpid()}'
    env['GZ_IP'] = '127.0.0.1'
    saved = {k: os.environ.get(k) for k in ('GZ_PARTITION', 'GZ_IP')}
    os.environ['GZ_PARTITION'], os.environ['GZ_IP'] = env['GZ_PARTITION'], env['GZ_IP']
    sim = Process(['ros2', 'launch', 'ranger_xarm_gazebo', 'sim.launch.py', 'headless:=true',
                   'sensors:=true', 'drive_base:=true'], env, str(logs / 'sim.log'))
    probe = RecordProbe(env, moveit_controller=MOVEIT, moveit_gripper=MOVEIT_G)
    teleop = None
    results = []
    try:
        rec = Recorder(probe)
        watch = BaseWatcher(probe)
        probe.wait_for(lambda: probe.joint_positions() is not None, 180.0, 'joint states')
        probe.wait_for(lambda: probe.controllers().get(MOVEIT) == 'active', 120.0,
                       'the arm controller')
        probe.wait_for(lambda: len(rec.susp) > 10 and rec.position() is not None, 120.0,
                       'the suspension and ground truth')
        teleop = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_sim.launch.py',
                          'base:=true', 'stability:=true', 'record:=true',
                          f'record_data_dir:={data}', f'record_session:={session}'], env,
                         str(logs / 'teleop.log'))
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 120.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.episode_state() is not None, 60.0, 'the episode state')
        probe.wait_for(lambda: watch.last() is not None and watch.last().at_rest, 60.0,
                       'the base node, base at rest')
        probe.wait_for(lambda: rec.last_state() is not None and rec.last_state().available,
                       120.0, 'a stability figure')
        time.sleep(5.0)
        probe.move_to(READY, seconds=4.0)
        start_input(probe, env, logs)
        wait_for_settled_graph(probe)
        # The input player's join can stall every node for a few hundred ms; the monitor then
        # restarts its filters and settles again (about 9 s). The tests start from a figure.
        probe.wait_for(lambda: rec.last_state() is not None and rec.last_state().available,
                       120.0, 'a stability figure after the graph settled')
        yield {'probe': probe, 'env': env, 'logs': logs, 'rec': rec, 'watch': watch,
               'session': os.path.join(data, session), 'results': results, 'teleop': teleop,
               'windows': []}
    finally:
        probe.close()
        if teleop is not None:
            teleop.stop()
        sim.stop()
        kill_gz_partition(env['GZ_PARTITION'])
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for line in results:
            print(line)


def note(stack, text):
    stack['results'].append(text)


def posture(stack, name, rest=2.0):
    """Move the arm, settle, and keep a rest window; return (start, end) in simulation time."""
    stack['probe'].move_to(POSTURES[name], seconds=4.0)
    rec = stack['rec']
    rec.sleep(2.5)
    t0 = rec.now()
    rec.sleep(rest)
    return name, t0, rec.now()


def drive(stack, vx, vy, wz, hold, label, zero=True, ramp=None, judged=True, step=False):
    """Command the base directly (stepped, or ramped at ``ramp``), hold, then stop."""
    speed = max(math.hypot(vx, vy), abs(wz) * sh.TURN_WHEEL_RADIUS)
    rec, pub = stack['rec'], stack['pub']
    t0 = rec.now()
    peak = max(abs(vx), abs(vy), abs(wz)) / ramp if ramp else 0.0
    while True:
        t = rec.now() - t0
        k = min(1.0, t / peak) if peak > 0 else 1.0
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = vx * k, vy * k, wz * k
        pub.publish(m)
        if t > peak + hold:
            break
        time.sleep(0.02)
    if zero:
        pub.publish(Twist())
    tz = rec.now()
    rec.sleep(3.0 if zero else 4.0)
    stack['windows'].append({'name': label, 'start': t0, 'end': rec.now(), 'zero': tz,
                             'speed': speed, 'judged': judged, 'step': step})


def crab_to(stack, y_target, speed=0.15, accel=0.15, label=''):
    rec, pub = stack['rec'], stack['pub']
    t0 = rec.now()
    while True:
        dy = y_target - rec.position()[1]
        if abs(dy) < 0.03 or rec.now() - t0 > 60.0:
            break
        m = Twist()
        m.linear.y = math.copysign(min(speed, accel * (rec.now() - t0) + 0.02, max(0.05, abs(dy))),
                                   dy)
        pub.publish(m)
        time.sleep(0.02)
    pub.publish(Twist())
    rec.sleep(2.5)
    stack['windows'].append({'name': label, 'start': t0, 'end': rec.now(), 'zero': None,
                             'judged': True})


def test_the_monitor_gives_a_measured_figure(stack):
    s = stack['rec'].last_state()
    assert s.available and s.tilt_source == StabilityState.TILT_MEASURED, s.reason_text
    assert len(s.model_sha256) == 64 and s.stop_decel_m_s2 == 1.0
    assert 0.15 < s.margin_m < 0.2 and s.margin_pessimistic_m < s.margin_m
    note(stack, f'at the ready posture: figure {s.margin_m:.4f} m, pessimistic '
                f'{s.margin_pessimistic_m:.4f} m, energy reserve {s.energy_reserve_j:.1f} J')


def test_the_figure_agrees_with_the_wheels_loads(stack):
    """INV-3, the frozen rule, over rests, starts, crabbing, turns, each kind of stop, a ramp."""
    probe, rec = stack['probe'], stack['rec']
    urdf = probe.description
    # Rests in five postures: the reference's calibration, then judged like any window.
    rests = [posture(stack, name) for name in ('ready', 'stretch_fwd', 'stretch_left',
                                               'stretch_right', 'tucked')]
    with rec.lock:
        susp = list(rec.susp)
    t_s = np.array([s[0] for s in susp])
    calibration = []
    for name, t0, t1 in rests:
        sel = [s for s in susp if t0 <= s[0] <= t1]
        mean = {j: float(np.mean([s[1][j] for s in sel])) for j in sel[0][1]}
        q = dict(zip(ARM, POSTURES[name]))
        cop, weight = sh.model_cop_at_rest(urdf, q, rec.down_between(t0, t1))
        calibration.append((mean, cop, weight))
    reference = sh.LoadReference(urdf, {})
    worst = reference.calibrate(calibration)
    note(stack, f'reference: loads = {reference.a:.3f} k (x - x_ref) + c xdot + '
                f'{reference.b:.1f} N; worst centre-of-pressure error at rest '
                f'{worst * 1e3:.1f} mm')
    assert worst < 0.005, 'the load reference does not close at rest'
    for name, t0, t1 in rests:
        stack['windows'].append({'name': f'rest {name}', 'start': t0, 'end': t1, 'zero': None,
                                 'judged': True})

    # A stop by releasing the grip through the stack (the base node drives at its caps).
    posture(stack, 'ready', rest=0.5)
    t0 = rec.now()
    play(stack, 'release_drive', [phase(0.4), phase(1.5, grip=True, left=FORWARD), phase(3.0)])
    with rec.lock:
        cmds = [c for c in rec.cmds if c[0] >= t0]
    # The zero is the first all-zero command after the last moving one, in arrival order (the
    # two can share a simulation stamp).
    last_moving = max((i for i, c in enumerate(cmds) if abs(c[1]) > 0.01), default=None)
    moving = last_moving is not None
    zero = next((c[0] for c in cmds[(last_moving or 0) + 1:] if c[1:] == (0.0,) * 3), None) \
        if moving else None
    assert moving and zero is not None, 'the base node did not drive and stop'
    stack['windows'].append({'name': 'release through the stack 0.2', 'start': t0,
                             'end': rec.now(), 'zero': zero, 'speed': 0.2, 'judged': True})

    # From here the test commands the base itself (the base node now refuses: a second
    # publisher of the base command, G-9).
    stack['pub'] = probe.node.create_publisher(Twist, '/cmd_vel', 10)
    for v in (0.2, 0.4):
        drive(stack, v, 0, 0, 1.5, f'fwd {v} ready')
        drive(stack, -v, 0, 0, 1.5, f'back {v} ready')
        drive(stack, 0, v, 0, 1.5, f'left {v} ready')
        drive(stack, 0, -v, 0, 1.5, f'right {v} ready')
    drive(stack, 0.2, 0, 0, 1.5, 'fwd 0.2 ready silent (timeout)', zero=False)
    drive(stack, -0.2, 0, 0, 1.5, 'back 0.2 ready')
    for w in (0.4, 1.0):
        drive(stack, 0, 0, w, 2.0, f'turn ccw {w} ready')
        drive(stack, 0, 0, -w, 2.0, f'turn cw {w} ready')
    posture(stack, 'stretch_left', rest=0.5)
    drive(stack, 0, 0.2, 0, 1.5, 'left 0.2 stretch_left')
    drive(stack, 0, -0.2, 0, 1.5, 'right 0.2 stretch_left')
    drive(stack, 0, 0.8, 0, 1.5, 'left 0.8 stretch_left (a pair lifts)')
    drive(stack, 0, -0.8, 0, 1.5, 'right 0.8 stretch_left')
    # A 2 cm bar driven over: a step under a wheel, where only the pessimistic figure is judged.
    posture(stack, 'ready', rest=0.5)
    x, y = rec.position()
    spawn('bar2cm', box_sdf('bar2cm', (0.05, 2.0, 0.02), (x + 1.0, y, 0.01, 0.0)))
    rec.sleep(1.0)
    drive(stack, 0.2, 0, 0, 7.0, 'bar 0.2 ready', step=True)
    drive(stack, -0.5, 0, 0, 3.5, 'bar back 0.5 ready', step=True)
    drive(stack, 0.5, 0, 0, 3.0, 'bar 0.5 ready', step=True)
    # A 10 degree side ramp, the arm downhill. Down it at 0.4 m/s the simulated robot (estimated
    # masses) tips over in its stop: that is a tip case, not a judged manoeuvre; 0.3 m/s is the
    # highest of 0.3, 0.2 and 0.1 whose wheel loads stay at or above -0.02 m.
    x, y = rec.position()
    spawn('ramp10', side_ramp('ramp10', 10.0, y + 0.8, x))
    rec.sleep(1.0)
    crab_to(stack, y + 0.8 + 0.25 + 1.4, label='crab up the 10 deg ramp')
    posture(stack, 'stretch_right', rest=2.0)
    drive(stack, 0, -0.2, 0, 1.0, 'down 0.2 ramp10 arm downhill')
    drive(stack, 0, 0.2, 0, 1.0, 'up 0.2 ramp10 arm downhill', ramp=0.2)
    drive(stack, 0, -0.3, 0, 0.8, 'down 0.3 ramp10 arm downhill')
    drive(stack, 0, 0.4, 0, 0.8, 'up 0.4 ramp10 arm downhill', ramp=0.2)
    posture(stack, 'ready', rest=0.5)
    crab_to(stack, y, label='crab back to the floor')

    # The reference over the whole run, and the monitor's states, on simulation time.
    with rec.lock:
        susp, states = list(rec.susp), list(rec.states)
    t_s = np.array([s[0] for s in susp])
    ref = np.array([reference.figure(s[1], s[2]) for s in susp])
    avail = [s for s in states if s[2].available]
    t_m = np.array([s[0] for s in avail])
    col = {k: np.array([getattr(s[2], k) for s in avail]) for k in (
        'margin_m', 'margin_pessimistic_m', 'stop_margin_m', 'speed_m_s', 'turn_rate_rad_s')}
    np.savez_compressed(str(stack['logs'] / 'agreement_series.npz'), t_ref=t_s, ref=ref, t_mon=t_m,
                        windows=json.dumps(stack['windows']), **col)
    rows = sh.judge(stack['windows'], t_s, ref, t_m, col['margin_m'],
                    col['margin_pessimistic_m'], col['stop_margin_m'], col['speed_m_s'],
                    col['turn_rate_rad_s'])
    shift, rms = sh.lag(t_s, ref, t_m, col['margin_m'])
    table = ['manoeuvre | loads min | figure - loads | pessimistic - loads | '
             'most safer anywhere | stop (v, dev)']
    for r in rows:
        stop = f'{r["stop"][0]:.2f} {r["stop"][1]:+.3f}' if r['stop'] else ''
        mark = ' (step: figure reported, pessimistic judged)' if r['step'] else ''
        table.append(f'{r["name"]}{mark} | {r["loads_min"]:.3f} | {r["dev"]:+.3f} | '
                     f'{r["pess_dev"]:+.3f} | {r["safer_anywhere"]:+.3f} | {stop} '
                     f'{"; ".join(r["problems"])}')
    table.append(f'lag of the figure against the loads: {shift * 1e3:+.0f} ms (rms {rms:.4f} m)')
    unavailable = sum(1 for s in states if not s[2].available)
    table.append(f'monitor states: {len(states)}, unavailable {unavailable}')
    with open(stack['logs'] / 'agreement.txt', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(table) + '\n')
    note(stack, '\n'.join(table))
    failed = [(r['name'], r['problems']) for r in rows if r['problems']]
    assert len(stack['windows']) == 33 and len(rows) == 33     # every manoeuvre judged
    assert sum(r['step'] for r in rows) == 3                  # the bar: pessimistic judged only
    assert not failed, failed


def test_a_recorded_episode_carries_the_monitor_and_its_model(stack):
    """INV-5: the stream in the episode, the mass file's fingerprint in its sidecar."""
    before = {os.path.basename(p) for p in md.iter_episodes(stack['session'])}
    stack['probe'].move_to(READY, seconds=4.0)               # arming checks the posture
    play(stack, 'episode', calibrate_and_arm() + [
        phase(0.2, x=True), phase(4.0), phase(0.2, x=True), phase(1.0), phase(0.2, y=True),
        phase(2.0)])
    stack['probe'].wait_for(lambda: len([p for p in md.iter_episodes(stack['session'])
                                         if os.path.basename(p) not in before]) == 1, 60.0,
                            'the episode to be written')
    path = [p for p in md.iter_episodes(stack['session']) if os.path.basename(p) not in before][0]
    sidecar = md.read_json(path)
    with open(SIM_MASS_FILE, 'rb') as handle:
        sha = hashlib.sha256(handle.read()).hexdigest()
    assert sidecar['stability']['mass_file_sha256'] == sha
    episode = read_episode(stack['session'], path)
    report = check_episode(episode, load_checks())
    section = report['stability']
    assert section['recorded'] and math.isfinite(section['margin_m']['value'])
    fingerprints = section['model_sha256']
    assert len(fingerprints) == 1 and len(fingerprints[0]) == 64
    note(stack, f'episode {os.path.basename(path)}: verdict {report["verdict"]}; stability '
                f'{json.dumps(section, default=str)[:300]}')


def test_killed_the_monitor_changes_nothing_else_and_never_lengthened_the_cycle(stack):
    """
    INV-2 in Gazebo: the teleop node's period with the monitor and without it.

    Each measure lasts 20 s with the base driving back and forth (the monitor busy). Then, as
    information (not judged): a new monitor joining the running graph mid-session. By design
    the monitor starts with the stack, before arming, and is never started mid-session: any
    process joining the graph can stall the nodes on it (measured here: hundreds of ms).
    """
    probe, teleop, rec = stack['probe'], stack['teleop'], stack['rec']
    pub = stack.get('pub') or probe.node.create_publisher(Twist, '/cmd_vel', 10)

    def periods(seconds, during=None):
        t0 = time.monotonic()
        k = 0
        while time.monotonic() - t0 < seconds:
            m = Twist()
            m.linear.x = 0.2 if (k // 100) % 2 == 0 else -0.2       # 2 s each way
            pub.publish(m)
            if during is not None and k == 50:
                during()
            k += 1
            time.sleep(0.02)
        pub.publish(Twist())
        with probe.lock:
            arrivals = [t for t, _ in probe.states if t >= t0]
        d = np.diff(arrivals) * 1e3
        return np.percentile(d, 50), np.percentile(d, 99), d.max(), len(arrivals) / seconds

    def text(name, p):
        return f'{name} {p[0]:.1f}/{p[1]:.1f}/{p[2]:.1f} ms, {p[3]:.1f}/s'

    with_monitor = periods(20.0)
    os.kill(node_pid({'launch': teleop}, 'stability_node'), signal.SIGKILL)
    probe.wait_for(lambda: '/stability_node' not in graph_nodes(probe, settle=0.5), 40.0,
                   'discovery to drop the killed monitor')
    without = periods(20.0)
    params = os.path.join(SHARE, 'config', 'stability_SIMULATION_ONLY.yaml')
    started = []

    joined = []

    def start():
        joined.append(time.monotonic())
        started.append(Process(['ros2', 'run', 'ranger_xarm_teleop_servo', 'stability_node',
                                '--ros-args', '--params-file', params, '-p',
                                f'mass_file:={SIM_MASS_FILE}', '-p', 'use_sim_time:=true'],
                               stack['env'], str(stack['logs'] / 'stability_restart.log')))
    try:
        joining = periods(15.0, during=start)
        # The new monitor really joined: it is on the graph and published after its start.
        assert '/stability_node' in graph_nodes(probe, settle=0.5)
        with rec.lock:
            fresh = [s for s in rec.states if s[1] > joined[0]]
        assert fresh, 'the restarted monitor published nothing'
    finally:
        for p in started:
            p.stop()
    line = ('teleop state period p50/p99/max, rate, the base driving: ' +
            '; '.join([text('with the monitor', with_monitor), text('without', without),
                       text('across a monitor joining mid-session (information, not by design)',
                            joining)]))
    note(stack, line)
    with open(stack['logs'] / 'inv2.txt', 'w', encoding='utf-8') as handle:
        handle.write(line + '\n')
    assert abs(with_monitor[0] - without[0]) < 1.0
    assert with_monitor[1] <= without[1] + 2.0 and with_monitor[2] <= without[2] + 5.0
    assert probe.state() is not None and time.monotonic() - probe.states[-1][0] < 0.5
