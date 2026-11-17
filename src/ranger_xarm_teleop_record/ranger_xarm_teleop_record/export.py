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
Fixed-rate export of one episode (no ROS).

The export puts every field on a common tick grid, ``t_k = t0 + k / rate`` over the first to
last stamp of the state stream (the action timeline). The only rule that touches the data is
a causal zero-order hold: each field takes the newest sample whose stamp is not later than
the tick, and the age of that sample is exported with it. Nothing is interpolated, no later
sample is used, and a gap in a source stream shows as growing age and an invalid tick.

Output of ``write_export``: ``data.npz`` (arrays, first dimension T), ``export.json`` (what the
arrays mean, the hold rule, the episode identity, the checker verdict) and ``images/<stream>/
<row>.png`` for each referenced image (``.npy`` float32 for 32FC1). Nothing written names an
absolute path, a host or a user. Streams on the host wall clock are inputs and not exported.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ranger_xarm_teleop_record import metadata
from ranger_xarm_teleop_record.checker import clock_pairs, jsonable, rec_to_wall
from ranger_xarm_teleop_record.profile import period, recorded
from ranger_xarm_teleop_record.streams import Episode, Stream

FORMAT = 'ranger_xarm_teleop.export/1'
RULE = ('causal zero-order hold: each field is the newest sample not later than the tick, '
        'with its age; nothing is interpolated. A cycle with the clutch released means hold: '
        'no new target, the arm finishes the points it already has; action_valid is False '
        'there. gripper_action_valid is False in cycles where teleop held no gripper command')
TIME_EPS = 1e-9          # a stamp this far past a tick still counts as not later (float noise)
STATE_FIELDS = ('armed', 'clutch_engaged', 'hold_reason', 'active_limit', 'damped',
                'translation_scale', 'input_age', 'measured_tool_pose',
                'measured_tool_pose_valid', 'gripper_active', 'gripper_command',
                'gripper_request', 'gripper_reference_source', 'gripper_measured',
                'gripper_measured_valid', 'gripper_measured_stamp')
STAMPED_KINDS = ('teleop_state', 'joint_states', 'image')   # streams whose age gates a tick


# --------------------------------------------------------------------------------------------
# Causal hold
# --------------------------------------------------------------------------------------------

def hold_index(stamp: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Return, for each tick, the row of the newest sample stamped not later than it, or -1.

    Rows are taken in stamp order (a stable sort of the recorded order); nothing later than
    the tick is ever chosen.
    """
    stamp = np.asarray(stamp, dtype=float)
    rows = np.flatnonzero(np.isfinite(stamp))
    order = rows[np.argsort(stamp[rows], kind='stable')]
    pos = np.searchsorted(stamp[order], t + TIME_EPS, side='right') - 1
    return np.where(pos >= 0, order[np.maximum(pos, 0)], -1) if len(order) else \
        np.full(len(t), -1)


def _age(stamp: np.ndarray, idx: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return ``t - stamp[idx]``, NaN where nothing is held yet."""
    out = np.full(len(t), np.nan)
    ok = idx >= 0
    out[ok] = t[ok] - np.asarray(stamp, dtype=float)[idx[ok]]
    return out


def _gather(values: np.ndarray, idx: np.ndarray, fill=np.nan) -> np.ndarray:
    """Return ``values[idx]`` row-wise with ``fill`` where ``idx`` is -1 (floats)."""
    values = np.asarray(values, dtype=float)
    out = np.full((len(idx),) + values.shape[1:], fill, dtype=float)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


# --------------------------------------------------------------------------------------------
# Field documentation
# --------------------------------------------------------------------------------------------

# name -> (unit, frame key or text, description)
FIELD_INFO: Dict[str, Tuple[str, str, str]] = {
    't': ('s', 'recording clock', 'tick time'),
    't_wall': ('s', 'host wall clock', 'tick time mapped to the host wall clock'),
    'clutch_engaged': ('', '', 'teleop was commanding the arm in the held state cycle'),
    'action_valid': ('', '', 'clutch engaged and a target stamped in the same cycle'),
    'action_pose': ('m, xyzw', 'base', 'commanded tool target (newest target)'),
    'action_pose_age': ('s', '', 'tick time minus the target stamp'),
    'gripper_action_valid': ('', '', 'teleop commanded the gripper in the held cycle'),
    'gripper_command': ('rad', '', 'gripper command in force (a measured stand-in when '
                        'gripper_action_valid is False and the source is MEASURED)'),
    'gripper_request': ('', '', 'GripperCommand.REQUEST_* of the command in force'),
    'gripper_source': ('', '', 'GripperCommand.SOURCE_* of the command in force'),
    'joint_command': ('rad/s', '', 'servo joint velocity command, joint_command_names order'),
    'joint_command_valid': ('', '', 'clutch engaged and a joint command of the same cycle'),
    'joint_command_age': ('s', '', 'tick time minus the joint command stamp'),
    'joint_names': ('', '', 'joint order of joint_position, joint_velocity, joint_effort'),
    'joint_command_names': ('', '', 'joint order of joint_command'),
    'joint_position': ('rad or m', '', 'measured joint positions'),
    'joint_velocity': ('rad/s or m/s', '', 'measured joint velocities'),
    'joint_effort': ('N m or N', '', 'measured joint efforts'),
    'joint_state_age': ('s', '', 'tick time minus the joint state stamp'),
    'tool_pose_measured': ('m, xyzw', 'base', 'forward kinematics of the measured joints; '
                           'NaN where not valid'),
    'tool_pose_measured_valid': ('', '', 'the measured tool pose was valid'),
    'gripper_measured': ('rad', '', 'measured gripper opening; NaN where not valid'),
    'gripper_measured_valid': ('', '', 'the measured opening was valid'),
    'gripper_measured_age': ('s', '', 'tick time minus the stamp of the measured opening'),
    'damped': ('', '', 'the solver was damped in the held cycle'),
    'active_limit': ('', '', 'TeleopState.LIMIT_* of the held cycle'),
    'hold_reason': ('', '', 'TeleopState.HOLD_* of the held cycle'),
    'armed': ('', '', 'teleop armed'),
    'translation_scale': ('', '', 'operator translation scale'),
    'input_age': ('s', '', 'age of the newest operator input frame in the held cycle'),
    'state_age': ('s', '', 'tick time minus the state stamp'),
    'tick_valid': ('', '', 'every required observation is fresh (state, joint states, '
                   'required images); action fields do not matter here'),
    'base_odom_pose': ('m, xyzw', 'odometry frame', 'base odometry pose; NaN before the first'),
    'base_odom_age': ('s', '', 'tick time minus the odometry stamp'),
}


def _relative(path) -> Optional[str]:
    """Return ``path`` as recorded when relative; only its file name when it is absolute."""
    if not isinstance(path, str) or not path:
        return None
    if os.path.isabs(path) or path.startswith('~') or (len(path) > 1 and path[1] == ':'):
        return os.path.basename(path.replace('\\', '/'))
    return path.replace('\\', '/')


def _encoding(ep: Episode, name: str) -> str:
    """Return the pixel encoding of an image stream."""
    st = ep.get(name)
    if st is not None and st.labels.get('encoding'):
        return str(st.labels['encoding'][0])
    return str(ep.profile['streams'][name].get('image', {}).get('encoding', 'rgb8'))


# --------------------------------------------------------------------------------------------
# build_export
# --------------------------------------------------------------------------------------------

def _rows(ep: Episode, name: str) -> Optional[Stream]:
    """Return the stream when it has rows."""
    st = ep.get(name)
    return st if st is not None and len(st) > 0 else None


def build_export(ep: Episode, checks: dict, rate_hz: Optional[float] = None,
                 verdict=None) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Return ``(arrays, meta)``: the episode on a fixed tick grid and its description.

    ``verdict`` is the checker's report (or just its verdict string) and is copied into the
    metadata. Raises ValueError when the state stream is missing or lacks a needed field.
    """
    rate = float(rate_hz if rate_hz else checks['export']['rate_hz'])
    max_age = float(checks['export']['max_age_periods'])
    state = _rows(ep, 'state')
    if state is None:
        raise ValueError('no state stream: no action timeline to export')
    absent = [k for k in STATE_FIELDS if k not in state.fields]
    if absent:
        raise ValueError(f'state stream lacks fields {absent}')
    specs = ep.profile['streams']
    state_period = period(specs['state'])
    t0 = float(np.min(state.stamp))
    t1 = float(np.max(state.stamp))
    n = int(math.floor((t1 - t0) * rate + 1e-9)) + 1
    t = t0 + np.arange(n) / rate
    arrays: Dict[str, np.ndarray] = {'t': t}
    pairs = clock_pairs(ep)
    if pairs is not None:
        arrays['t_wall'] = rec_to_wall(pairs, t)

    si = hold_index(state.stamp, t)
    arrays['state_age'] = _age(state.stamp, si, t)

    def held(key):
        """Return a state field at the held state row."""
        return np.asarray(state.fields[key])[si]

    engaged = held('clutch_engaged').astype(bool)
    arrays['clutch_engaged'] = engaged
    cycle_stamp = state.stamp[si]

    # Action: the target of the held cycle.
    target = _rows(ep, 'target')
    if target is not None:
        ti = hold_index(target.stamp, t)
        arrays['action_pose'] = _gather(target.fields['pose'], ti)
        arrays['action_pose_age'] = _age(target.stamp, ti, t)
        same = np.zeros(n, dtype=bool)
        ok = ti >= 0
        same[ok] = np.abs(target.stamp[ti[ok]] - cycle_stamp[ok]) <= 0.5 * state_period
        arrays['action_valid'] = engaged & same
    else:
        arrays['action_pose'] = np.full((n, 7), np.nan)
        arrays['action_pose_age'] = np.full(n, np.nan)
        arrays['action_valid'] = np.zeros(n, dtype=bool)

    arrays['gripper_action_valid'] = held('gripper_active').astype(bool)
    arrays['gripper_command'] = held('gripper_command').astype(float)
    arrays['gripper_request'] = held('gripper_request').astype(np.int64)
    arrays['gripper_source'] = held('gripper_reference_source').astype(np.int64)

    constants: Dict[str, List[str]] = {}
    command = _rows(ep, 'joint_command')
    if command is not None:
        ci = hold_index(command.stamp, t)
        arrays['joint_command'] = _gather(command.fields['velocities'], ci)
        arrays['joint_command_age'] = _age(command.stamp, ci, t)
        same = np.zeros(n, dtype=bool)
        ok = ci >= 0
        same[ok] = np.abs(command.stamp[ci[ok]] - cycle_stamp[ok]) <= 0.5 * state_period
        arrays['joint_command_valid'] = engaged & same
        constants['joint_command_names'] = list(command.joint_names)
        arrays['joint_command_names'] = np.array(command.joint_names, dtype=str)
    joints = _rows(ep, 'joint_states')
    ji = None
    if joints is not None:
        ji = hold_index(joints.stamp, t)
        for key in ('position', 'velocity', 'effort'):
            arrays[f'joint_{key}'] = _gather(joints.fields[key], ji)
        arrays['joint_state_age'] = _age(joints.stamp, ji, t)
        constants['joint_names'] = list(joints.joint_names)
        arrays['joint_names'] = np.array(joints.joint_names, dtype=str)

    pose_ok = held('measured_tool_pose_valid').astype(bool)
    pose = held('measured_tool_pose').astype(float)
    pose[~pose_ok] = np.nan
    arrays['tool_pose_measured'] = pose
    arrays['tool_pose_measured_valid'] = pose_ok
    g_ok = held('gripper_measured_valid').astype(bool)
    g_value = held('gripper_measured').astype(float)
    g_value[~g_ok] = np.nan
    g_stamp = held('gripper_measured_stamp').astype(float)
    arrays['gripper_measured'] = g_value
    arrays['gripper_measured_valid'] = g_ok
    arrays['gripper_measured_age'] = np.where(g_stamp > 0.0, t - g_stamp, np.nan)

    arrays['damped'] = held('damped').astype(bool)
    arrays['active_limit'] = held('active_limit').astype(np.int64)
    arrays['hold_reason'] = held('hold_reason').astype(np.int64)
    arrays['armed'] = held('armed').astype(bool)
    arrays['translation_scale'] = held('translation_scale').astype(float)
    arrays['input_age'] = held('input_age').astype(float)

    base_name = None
    for name, spec in recorded(ep.profile).items():
        if spec.get('base') and spec.get('kind') == 'odometry' and _rows(ep, name) is not None:
            base_name = name
            break
    if base_name is not None:
        odom = ep.streams[base_name]
        bi = hold_index(odom.stamp, t)
        arrays['base_odom_pose'] = _gather(odom.fields['pose'], bi)
        arrays['base_odom_age'] = _age(odom.stamp, bi, t)

    images: Dict[str, dict] = {}
    for name, spec in recorded(ep.profile).items():
        if spec.get('kind') != 'image' or _rows(ep, name) is None:
            continue
        st = ep.streams[name]
        ii = hold_index(st.stamp, t)
        arrays[f'image_{name}_index'] = ii.astype(np.int64)
        arrays[f'image_{name}_age'] = _age(st.stamp, ii, t)
        enc = _encoding(ep, name)
        ext = 'npy' if enc == '32FC1' else 'png'
        images[name] = {'encoding': enc, 'file_extension': ext,
                        'index_array': f'image_{name}_index',
                        'age_array': f'image_{name}_age',
                        'path': f'images/{name}/<row:06d>.{ext}'}
        if ext == 'npy':
            images[name]['note'] = 'float32 array stored with numpy.save, not a PNG'

    # tick_valid: every required observation fresh.
    valid = np.ones(n, dtype=bool)
    for name, spec in recorded(ep.profile).items():
        if not spec.get('required') or spec.get('kind') not in STAMPED_KINDS \
                or spec.get('rate_hz') is None:
            continue
        st = _rows(ep, name)
        if st is None:
            valid[:] = False
            continue
        idx = si if name == 'state' else hold_index(st.stamp, t)
        age = _age(st.stamp, idx, t)
        with np.errstate(invalid='ignore'):
            valid &= np.isfinite(age) & (age <= max_age * period(spec) + TIME_EPS)
    arrays['tick_valid'] = valid

    meta = _meta(ep, arrays, rate, t0, t1, constants, images, base_name, verdict, pairs)
    return arrays, meta


def _meta(ep: Episode, arrays: Dict[str, np.ndarray], rate: float, t0: float, t1: float,
          constants: Dict[str, List[str]], images: Dict[str, dict], base_name: Optional[str],
          verdict, pairs) -> dict:
    """Assemble export.json."""
    sidecar = ep.sidecar
    frames = {'base': metadata.lookup(sidecar, 'frames.base') or ep.profile['frames']['base'],
              'tool': metadata.lookup(sidecar, 'frames.tool') or ep.profile['frames']['tool']}
    fields = []
    for name, array in arrays.items():
        if name.startswith('image_') and name.endswith('_index'):
            info = ('', '', f'row of the {name[6:-6]} stream held at the tick, -1 if none yet')
        elif name.startswith('image_') and name.endswith('_age'):
            info = ('s', '', 'tick time minus the image stamp')
        else:
            info = FIELD_INFO.get(name, ('', '', ''))
        frame = frames['base'] if info[1] == 'base' else info[1]
        fields.append({'name': name, 'shape': list(array.shape), 'dtype': str(array.dtype),
                       'unit': info[0], 'frame': frame, 'description': info[2],
                       'per_tick': name not in constants and not name.endswith('_names')})
    checker = None
    if isinstance(verdict, dict):
        checker = {k: verdict.get(k) for k in ('verdict', 'reasons', 'flags', 'usable')}
    elif verdict is not None:
        checker = {'verdict': str(verdict)}
    meta = {
        'format': FORMAT,
        'rate_hz': rate,
        'n_ticks': int(len(arrays['t'])),
        'rule': RULE,
        'time': {'clock': 'recording clock (t)', 't_start': t0, 't_stop': t1,
                 'wall_pairs': None if pairs is None else int(len(pairs[0]))},
        'frames': {'base': frames['base'], 'tool': frames['tool'],
                   'poses': f'x y z qx qy qz qw in {frames["base"]} (quaternion xyzw)'},
        'fields': fields,
        'constants': constants,
        'episode': {
            'index': metadata.lookup(sidecar, 'episode.index'),
            'status': metadata.lookup(sidecar, 'episode.status'),
            'labels': dict(metadata.lookup(sidecar, 'episode.labels') or {}),
            'session': metadata.lookup(sidecar, 'session.name') or sidecar.get('session_name'),
        },
        'source': {'bag_file': _relative(metadata.lookup(sidecar, 'episode.file'))},
        'checker': checker,
        'base_odometry_stream': base_name,
        'images': images,
        'image_naming': {
            'pattern': 'images/<stream>/<row:06d>.<ext>',
            'row': '<row> is the row of the stream in the recorded episode; image_<stream>_index '
                   'holds it per tick (-1: no image yet)',
        },
    }
    return jsonable(meta)


# --------------------------------------------------------------------------------------------
# write_export
# --------------------------------------------------------------------------------------------

def _to_image(pixels: np.ndarray, encoding: str) -> Image.Image:
    """Return a PIL image for the pixels of a frame in ``encoding`` (not 32FC1)."""
    pixels = np.asarray(pixels)
    if encoding == 'rgb8':
        return Image.fromarray(np.ascontiguousarray(pixels.astype(np.uint8)), 'RGB')
    if encoding == 'bgr8':
        return Image.fromarray(np.ascontiguousarray(pixels[..., ::-1].astype(np.uint8)), 'RGB')
    if encoding == 'mono8':
        return Image.fromarray(np.ascontiguousarray(pixels.astype(np.uint8)), 'L')
    if encoding in ('16UC1', 'mono16'):
        return Image.fromarray(np.ascontiguousarray(pixels.astype('<u2')))
    raise ValueError(f'unsupported image encoding {encoding!r}')


def write_export(out_dir: str, arrays: Dict[str, np.ndarray], meta: dict, ep: Episode,
                 image_streams: Optional[List[str]] = None) -> List[str]:
    """Write the export into ``out_dir``; return the files written, relative to it."""
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    np.savez_compressed(os.path.join(out_dir, 'data.npz'), **arrays)
    written.append('data.npz')
    if image_streams is None:
        image_streams = [k[6:-6] for k in arrays if k.startswith('image_')
                         and k.endswith('_index')]
    for name in image_streams:
        info = (meta.get('images') or {}).get(name, {})
        encoding = info.get('encoding') or _encoding(ep, name)
        rows = np.unique(arrays[f'image_{name}_index'])
        folder = os.path.join(out_dir, 'images', name)
        os.makedirs(folder, exist_ok=True)
        for row in rows[rows >= 0]:
            pixels = ep.images[name](int(row))
            if encoding == '32FC1':
                rel = f'images/{name}/{int(row):06d}.npy'
                np.save(os.path.join(out_dir, rel), np.asarray(pixels, dtype=np.float32))
            else:
                rel = f'images/{name}/{int(row):06d}.png'
                _to_image(pixels, encoding).save(os.path.join(out_dir, rel))
            written.append(rel)
    document = dict(meta)
    document['files'] = written + ['export.json']
    metadata.write_json_atomic(os.path.join(out_dir, 'export.json'), jsonable(document))
    written.append('export.json')
    return written
