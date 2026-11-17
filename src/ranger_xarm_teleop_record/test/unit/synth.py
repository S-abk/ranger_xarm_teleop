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
"""Synthetic episodes for the checker, consistency and export tests (no ROS)."""

import copy
import math

import numpy as np
from ranger_xarm_teleop_record import profile as profile_module
from ranger_xarm_teleop_record.streams import Episode, Stream

T0 = 100.0               # first simulation stamp
WALL0 = 1.7e9            # host wall time at T0
STATE_HZ = 50.0
JOINT_HZ = 142.857
CAMERA_HZ = 30.0
EPISODE_HZ = 5.0
LAG_ARM = 0.12           # the measured pose follows the target by this
LAG_GRIPPER = 0.10
LATENCY = 0.002          # receive time minus stamp, robot-side streams
ENGAGED = [(1.0, 4.0), (5.0, 8.0), (9.0, 11.5)]       # seconds after T0
GRIPPER_ACTIVE = [(1.5, 3.5), (5.5, 7.5)]
JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper_joint']
CAMERAS = ('camera_gantry', 'camera_wrist')
BASE_FRAME = 'xarm_link_base'
TOOL_FRAME = 'xarm_link_tcp'
IMAGE_SHAPE = (6, 8, 3)


# --------------------------------------------------------------------------------------------
# Quaternions (xyzw), numpy
# --------------------------------------------------------------------------------------------

def qmul(a, b):
    """Return the Hamilton product of two xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def qaxis(axis, angle):
    """Return the quaternion of a rotation of ``angle`` about the unit ``axis``."""
    s = math.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)])


def rotz_matrix(angle):
    """Return the 3x3 rotation about z."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotate_poses_about_z(poses, angle):
    """Return poses (n, 7) expressed after a rotation of the frame by ``angle`` about z."""
    out = np.array(poses, dtype=float)
    out[:, :3] = out[:, :3] @ rotz_matrix(angle).T
    qz = qaxis((0.0, 0.0, 1.0), angle)
    for i in range(len(out)):
        out[i, 3:] = qmul(qz, out[i, 3:])
    return out


# --------------------------------------------------------------------------------------------
# The synthetic robot
# --------------------------------------------------------------------------------------------

def target_pose(t):
    """Return the commanded tool pose (7,) at recording time ``t``."""
    u = t - T0
    p = [0.40 + 0.10 * math.sin(0.7 * u), 0.10 * math.cos(0.5 * u),
         0.30 + 0.05 * math.sin(0.9 * u)]
    q = qmul(qaxis((0.0, 0.0, 1.0), 0.3 * math.sin(0.5 * u)),
             qaxis((1.0, 0.0, 0.0), 0.2 * math.sin(0.4 * u)))
    return np.concatenate([p, q])


def gripper_fn(t):
    """Return the commanded gripper opening: closes 0 to 0.8 rad, later opens again."""
    u = t - T0
    if u < 1.5:
        return 0.0
    if u < 2.5:
        return 0.8 * (u - 1.5)
    if u < 5.5:
        return 0.8
    if u < 6.5:
        return 0.8 * (1.0 - (u - 5.5))
    return 0.0


def _in(u, spans):
    """Return True when ``u`` lies in one of ``spans``."""
    return any(a <= u <= b for a, b in spans)


def wall_of(t, rtf):
    """Return the host wall time at recording time ``t``."""
    return WALL0 + (t - T0) / rtf


def _state_fields(n):
    """Return state fields with their resting values."""
    ints = ['owner', 'mux_status', 'hold_reason', 'active_limit', 'servo_status',
            'gripper_hold', 'gripper_request', 'gripper_reference_source',
            'calibration_session_id', 'calibration_reset_count', 'input_seq']
    bools = ['armed', 'motion_enabled', 'clutch_engaged', 'calibrated',
             'measured_tool_pose_valid', 'damped', 'keepout_bound', 'gripper_enabled',
             'gripper_active', 'gripper_measured_valid']
    floats = ['calibration_heading', 'input_age', 'damping', 'lead_translation',
              'lead_rotation', 'translation_scale', 'gripper_command', 'gripper_measured',
              'gripper_measured_stamp']
    fields = {k: np.zeros(n, dtype=np.int64) for k in ints}
    fields.update({k: np.zeros(n, dtype=bool) for k in bools})
    fields.update({k: np.zeros(n) for k in floats})
    fields['measured_tool_pose'] = np.zeros((n, 7))
    return fields


def make_sidecar(profile, counts, n_pairs, stop_reason='operator_stop', status='kept'):
    """Return a complete episode sidecar (``metadata.missing`` is empty)."""
    sidecar = {
        'format': 'ranger_xarm_teleop.episode/1',
        'episode': {
            'index': 1, 'status': status, 'file': 'bag/0001/0001_0.mcap',
            'labels': {'task': 'test-task', 'operator': 'test-operator'},
            'start': {'ros': T0, 'wall': WALL0}, 'stop': {'ros': T0 + 12.0, 'wall': WALL0 + 12.0},
            'stop_reason': stop_reason, 'flags': []},
        'session': {'name': 'test-session'},
        'software': {'teleop_repo': {'commit': 'abc123', 'dirty': False},
                     'robot_workspace': {'commit': 'def456', 'dirty': False}},
        'robot': {'description_sha256': '00' * 32, 'description_file': 'robot_description.urdf',
                  'semantic_sha256': '11' * 32},
        'limits': {'file': 'limits.yaml', 'sha256': '22' * 32,
                   'parameters': {'teleop': {'max_speed': 0.25}}},
        'calibration': {'calibrated': True, 'heading': 0.1, 'session_id': 1, 'reset_count': 0,
                        'changed_during_episode': False},
        'frames': {'base': BASE_FRAME, 'tool': TOOL_FRAME, 'input': 'quest_world'},
        'scale': {'at_start': 0.5, 'values_used': [0.5]},
        'time': {'use_sim_time': True, 'recording_clock': 'ros', 'streams': {'state': 'robot'},
                 'headset': {'offset_s': 0.0}, 'clock_pairs': {'stream': 'episode_state',
                                                               'n': n_pairs}},
        'base': {'mode': 'parked'},
        'counts': counts,
        'profile': copy.deepcopy(profile),
        'cameras': {},
    }
    for name in CAMERAS:
        spec = profile['streams'][name]
        sidecar['cameras'][spec['camera']] = {
            'intrinsics': {'width': IMAGE_SHAPE[1], 'height': IMAGE_SHAPE[0],
                           'k': [1.0, 0.0, 4.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0],
                           'distortion_model': 'plumb_bob', 'frame_id': f'{name}_optical'},
            'extrinsics': {'mount_link': 'mount', 'in_mount': [0.0] * 6}}
    return sidecar


def image_pixels(name, row):
    """Return the pixels of one synthetic frame (deterministic from stream and row)."""
    seed = (sum(ord(c) for c in name) * 1000 + int(row)) % (2 ** 31)
    return np.random.default_rng(seed).integers(0, 256, IMAGE_SHAPE, dtype=np.uint8)


def build(rtf=1.0, duration=12.0, profile_name='gazebo', limited_share=0.0, seed=1,
          release_reason=None, base=None, tool_frame_id=False):
    """
    Return a clean synthetic Episode of profile ``profile_name`` that the checker passes.

    ``limited_share`` > 0 makes about that share of engaged cycles damped (limit
    LIMIT_KEEP_OUT) with a large error in the measured pose that follows. ``release_reason``
    maps an engaged span index to the hold reason of its first released cycle. ``base`` is
    None (no base topics), 'parked' or 'moved'.
    """
    rng = np.random.default_rng(seed)
    prof = profile_module.load_profile(profile_name)
    n = int(round(duration * STATE_HZ)) + 1
    ts = T0 + np.arange(n) / STATE_HZ
    u = ts - T0
    engaged = np.array([_in(x, ENGAGED) for x in u])
    active = np.array([_in(x, GRIPPER_ACTIVE) for x in u])
    # A span ends one cycle after its last engaged stamp: the first released cycle.
    state = _state_fields(n)
    state['armed'][:] = True
    state['motion_enabled'][:] = True
    state['calibrated'][:] = True
    state['gripper_enabled'][:] = True
    state['clutch_engaged'] = engaged
    state['gripper_active'] = active
    state['hold_reason'] = np.where(engaged, 0, 5).astype(np.int64)
    for index, reason in (release_reason or {}).items():
        end = ENGAGED[index][1]
        row = int(np.searchsorted(u, end + 1e-9, side='right'))
        state['hold_reason'][row] = reason
    state['input_age'] = 0.02 + 0.005 * rng.random(n)
    state['input_seq'] = np.arange(n, dtype=np.int64)
    state['translation_scale'][:] = 0.5
    state['calibration_session_id'][:] = 1
    state['measured_tool_pose_valid'][:] = True
    state['gripper_measured_valid'][:] = True

    # Limited cycles: damped with a limit; the measured pose that follows is far off.
    limited = np.zeros(n, dtype=bool)
    if limited_share > 0.0:
        rows = np.flatnonzero(engaged)
        chunk = np.arange(len(rows)) // 10
        pick = (chunk * 7) % 10 < int(round(limited_share * 10))
        limited[rows[pick]] = True
        state['damped'] = limited.copy()
        state['active_limit'] = np.where(limited, 3, 0).astype(np.int64)

    measured = np.array([target_pose(x - LAG_ARM) for x in ts])
    measured[:, :3] += rng.normal(0.0, 0.0005, (n, 3))
    shift = int(round(LAG_ARM * STATE_HZ))
    for row in np.flatnonzero(limited):
        if row + shift < n:
            measured[row + shift, :3] += 0.05
    state['measured_tool_pose'] = measured

    # Joint states, newest stamp not later than each state cycle.
    nj = int(math.floor(duration * JOINT_HZ + 1e-9)) + 1
    tj = T0 + np.arange(nj) / JOINT_HZ
    position = np.zeros((nj, len(JOINTS)))
    for k in range(6):
        position[:, k] = 0.3 * np.sin(0.5 * (tj - T0) * (k + 1) / 3.0)
    position[:, 6] = [gripper_fn(x - LAG_GRIPPER) for x in tj]
    jrow = np.searchsorted(tj, ts + 1e-9, side='right') - 1
    state['gripper_measured'] = position[jrow, 6]
    state['gripper_measured_stamp'] = tj[jrow]
    state['gripper_command'] = np.array([gripper_fn(x) for x in ts])
    state['gripper_request'] = np.where(active & (u < 5.0), 2, np.where(active, 1, 0))
    state['gripper_reference_source'] = np.where(u < 1.5, 3, np.where(active, 2, 2))
    state['gripper_reference_source'] = state['gripper_reference_source'].astype(np.int64)
    state['gripper_request'] = state['gripper_request'].astype(np.int64)

    streams = {}

    def add(name, stamp, fields=None, frame=None, joint_names=None, labels=None,
            recv=None, has_header=True):
        """Create one stream of the profile."""
        spec = prof['streams'][name]
        stamp = np.asarray(stamp, dtype=float)
        streams[name] = Stream(
            name=name, topic=spec['topic'], kind=spec['kind'], stamp=stamp,
            recv=stamp + LATENCY if recv is None else recv, fields=fields or {},
            frame_id=[] if frame is None else [frame] * len(stamp),
            joint_names=list(joint_names or []), labels=labels or {}, has_header=has_header)

    add('state', ts, state)
    rows = np.flatnonzero(engaged)
    add('target', ts[rows], {'pose': np.array([target_pose(x) for x in ts[rows]])},
        frame=TOOL_FRAME if tool_frame_id else BASE_FRAME)
    velocities = np.array([[0.01 * (k + 1) * math.sin(x - T0) for k in range(6)]
                           for x in ts[rows]])
    add('joint_command', ts[rows], {'velocities': velocities}, joint_names=JOINTS[:6])
    add('joint_states', tj, {'position': position, 'velocity': np.gradient(position, tj, axis=0),
                             'effort': np.zeros_like(position)}, joint_names=JOINTS)
    arows = np.flatnonzero(active)
    add('gripper_command', ts[arows],
        {'position': state['gripper_command'][arows],
         'request': state['gripper_request'][arows],
         'reference_source': state['gripper_reference_source'][arows]})

    nc = int(math.floor(duration * CAMERA_HZ + 1e-9)) + 1
    tc = T0 + np.arange(nc) / CAMERA_HZ
    images = {}
    for name in CAMERAS:
        spec = prof['streams'][name]
        add(name, tc, {'width': np.full(nc, IMAGE_SHAPE[1]), 'height': np.full(nc, IMAGE_SHAPE[0]),
                       'step': np.full(nc, IMAGE_SHAPE[1] * 3)},
            labels={'encoding': ['rgb8'] * nc})
        add(spec['info'], tc, {'width': np.full(nc, IMAGE_SHAPE[1]),
                               'height': np.full(nc, IMAGE_SHAPE[0]),
                               'k': np.tile(np.eye(3).reshape(1, 9), (nc, 1))})
        images[name] = (lambda row, _n=name: image_pixels(_n, row))

    # Wall-clock streams: stamp is the host wall time, receive time the recording clock.
    ns = int(math.floor(duration * 60.0 + 1e-9)) + 1
    t_status = T0 + np.arange(ns) / 60.0
    add('status', wall_of(t_status, rtf),
        {'session_id': np.ones(ns, dtype=np.int64), 'seq': np.arange(1, ns + 1),
         'reset_count': np.zeros(ns, dtype=np.int64), 'xr_time_ms': 1000.0 * np.arange(ns) / 60.0,
         'client_time_ms': 1000.0 * np.arange(ns) / 60.0, 'right_present': np.ones(ns, dtype=bool),
         'left_present': np.ones(ns, dtype=bool)}, recv=t_status + LATENCY)
    nq = int(math.floor(duration * 30.0 + 1e-9)) + 1
    t_joy = T0 + np.arange(nq) / 30.0
    add('joy', wall_of(t_joy, rtf), {'axes': np.zeros((nq, 4)),
                                     'buttons': np.zeros((nq, 6), dtype=np.int64)},
        recv=t_joy + LATENCY)
    ne = int(math.floor(duration * EPISODE_HZ + 1e-9)) + 1
    t_ep = T0 + np.arange(ne) / EPISODE_HZ
    add('episode_state', t_ep, {'state': np.ones(ne, dtype=np.int64),
                                'wall_time': wall_of(t_ep, rtf)})

    if base is not None:
        nb = 50
        t_b = T0 + np.arange(nb) * duration / nb
        pose = np.zeros((nb, 7))
        pose[:, 6] = 1.0
        if base == 'moved':
            pose[:, 0] = np.linspace(0.0, 0.1, nb)
        add('odom', t_b, {'pose': pose, 'twist': np.zeros((nb, 6))})
        twist = np.zeros((nb, 6))
        if base == 'moved':
            twist[10:20, 0] = 0.1
        add('cmd_vel', t_b, {'twist': twist}, has_header=False, recv=t_b)

    t_end = ts[-1]
    counts = {}
    for name, spec in prof['streams'].items():
        if not spec.get('count') or name not in streams:
            continue
        st = streams[name]
        if spec['clock'] == 'host_wall':
            window = [wall_of(T0, rtf) - 1e-3, wall_of(t_end, rtf) + 1e-3]
        else:
            window = [T0 - 1e-3, t_end + 1e-3]
        counted = int(np.sum((st.stamp >= window[0]) & (st.stamp <= window[1])))
        counts[name] = {'window': window, 'count': counted}
    sidecar = make_sidecar(prof, counts, ne)
    return Episode(streams=streams, sidecar=sidecar, profile=prof, images=images)


# --------------------------------------------------------------------------------------------
# Mutations (each returns a modified deep copy)
# --------------------------------------------------------------------------------------------

def select_rows(stream, keep):
    """Return a copy of ``stream`` with only the rows where ``keep`` is True."""
    keep = np.asarray(keep, dtype=bool)
    return Stream(
        name=stream.name, topic=stream.topic, kind=stream.kind, stamp=stream.stamp[keep],
        recv=stream.recv[keep], fields={k: v[keep] for k, v in stream.fields.items()},
        frame_id=[f for f, k in zip(stream.frame_id, keep) if k] if stream.frame_id else [],
        joint_names=list(stream.joint_names),
        labels={k: [x for x, kk in zip(v, keep) if kk] for k, v in stream.labels.items()},
        has_header=stream.has_header)


def drop_rows(ep, name, rows):
    """Return a copy of ``ep`` without the given row indices of stream ``name``."""
    out = copy.deepcopy(ep)
    st = out.streams[name]
    keep = np.ones(len(st), dtype=bool)
    keep[np.asarray(rows, dtype=int)] = False
    out.streams[name] = select_rows(st, keep)
    return out


def with_field(ep, name, key, value):
    """Return a copy of ``ep`` with a stream field replaced."""
    out = copy.deepcopy(ep)
    out.streams[name].fields[key] = np.asarray(value)
    return out


def shift_stamps(ep, name, delta, recv_too=False):
    """Return a copy with the stamps of ``name`` moved by ``delta`` (receive times kept)."""
    out = copy.deepcopy(ep)
    out.streams[name].stamp = out.streams[name].stamp + delta
    if recv_too:
        out.streams[name].recv = out.streams[name].recv + delta
    return out


def shift_recv(ep, name, delta):
    """Return a copy with the receive times of ``name`` moved by ``delta``."""
    out = copy.deepcopy(ep)
    out.streams[name].recv = out.streams[name].recv + delta
    return out


def snapshot(ep):
    """Return a deep, comparable copy of every array and the sidecar of ``ep``."""
    return ({n: (s.stamp.copy(), s.recv.copy(), {k: v.copy() for k, v in s.fields.items()},
                 list(s.frame_id)) for n, s in ep.streams.items()},
            copy.deepcopy(ep.sidecar), copy.deepcopy(ep.profile))


def same(a, b):
    """Return True when two snapshots are equal."""
    sa, ca, pa = a
    sb, cb, pb = b
    if sa.keys() != sb.keys() or ca != cb or pa != pb:
        return False
    for name in sa:
        same_times = (np.array_equal(sa[name][0], sb[name][0])
                      and np.array_equal(sa[name][1], sb[name][1]))
        if not (same_times and sa[name][3] == sb[name][3]
                and sa[name][2].keys() == sb[name][2].keys()):
            return False
        for key in sa[name][2]:
            if not np.array_equal(sa[name][2][key], sb[name][2][key], equal_nan=True):
                return False
    return True
