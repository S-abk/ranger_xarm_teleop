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

The export puts every field on a common tick grid. By default the grid is fixed,
``t_k = t0 + k / rate`` over the first to last stamp of the state stream (the action
timeline). With ``tick_stream`` the ticks are the stamps of one camera's frames instead, so
every frame of that camera is used exactly once (a fixed grid beats against a camera whose
frames are not exactly 1/rate apart and skips or repeats frames). The only rule that touches
the data is a causal zero-order hold: each field takes the newest sample whose stamp is not
later than the tick, and the age of that sample is exported with it. Nothing is interpolated,
no later sample is used, no age is negative, and a gap in a source stream shows as growing
age and an invalid tick.

The measured fields of TeleopState (tool pose, gripper opening) come from a joint state whose
stamp can be later than the state's own cycle stamp (the teleop node reads the newest joint
state when it publishes). They are held on the newest state row whose cycle stamp and
measured-gripper stamp are both not later than the tick, which may be an earlier row than the
one the other state fields come from. The tool pose has no stamp of its own; it is bounded by
the gripper sample's stamp where the row has one (``tool_pose_measured_bounded``), and may be
up to one joint-state period newer than its cycle where it has not.

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
from ranger_xarm_teleop_record.streams import Episode, OPTIONAL_FIELDS, Stream

FORMAT = 'ranger_xarm_teleop.export/1'
RULE = ('causal zero-order hold: each field is the newest sample not later than the tick, '
        'with its age; nothing is interpolated. A cycle with the clutch released means hold: '
        'no new target, the arm finishes the points it already has; action_valid is False '
        'there. gripper_action_valid is False in cycles where teleop held no gripper command')
SOURCE_MEASURED = 3      # GripperCommand.SOURCE_MEASURED / TeleopState.gripper_reference_source
TIME_EPS = 1e-9          # a stamp this far past a tick still counts as not later (float noise)
STATE_FIELDS = ('armed', 'clutch_engaged', 'hold_reason', 'active_limit', 'damped',
                'translation_scale', 'input_age', 'measured_tool_pose',
                'measured_tool_pose_valid', 'gripper_active', 'gripper_command',
                'gripper_request', 'gripper_reference_source', 'gripper_measured',
                'gripper_measured_valid', 'gripper_measured_stamp')
STAMPED_KINDS = ('teleop_state', 'joint_states', 'image')   # streams whose age gates a tick
TICK_KINDS = ('image',)                                      # streams that may give the ticks


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
    """
    Return ``t - stamp[idx]``, NaN where nothing is held yet.

    ``hold_index`` admits a stamp up to TIME_EPS after the tick (float rounding of the tick
    grid); such an age is clamped to 0. A sample stamped later than that would be a causality
    error in this module: it raises instead of being clamped or exported.
    """
    out = np.full(len(t), np.nan)
    ok = idx >= 0
    age = t[ok] - np.asarray(stamp, dtype=float)[idx[ok]]
    if np.any(age < -TIME_EPS):
        raise ValueError(f'a held sample is stamped {-float(age.min()) * 1e3:.3f} ms after '
                         'its tick')
    out[ok] = np.maximum(age, 0.0)
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
    't': ('s', 'recording clock', 'tick time (see ticks in export.json)'),
    't_wall': ('s', 'host wall clock', 'tick time mapped to the host wall clock'),
    'clutch_engaged': ('', '', 'teleop was commanding the arm in the held state cycle '
                       '(False before the first state cycle)'),
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
                           'NaN where not valid. Held on the measured row (see '
                           'measured_row_age); TeleopState carries no stamp of its own joint '
                           "state, which is not newer than the gripper sample's"),
    'tool_pose_measured_valid': ('', '', 'the measured tool pose was valid'),
    'tool_pose_measured_bounded': ('', '', 'the held row has a gripper sample stamp, which '
                                   "bounds the tool pose's joint state (not later than the "
                                   'tick); where False the tool pose may come from a joint '
                                   'state up to one joint-state period after its cycle stamp'),
    'gripper_measured': ('rad', '', 'measured gripper opening; NaN where not valid'),
    'gripper_measured_valid': ('', '', 'the measured opening was valid'),
    'gripper_measured_age': ('s', '', 'tick time minus the stamp of the measured opening'),
    'measured_row_age': ('s', '', 'tick time minus the later of the cycle stamp and the '
                         'gripper sample stamp of the row the measured fields come from'),
    'damped': ('', '', 'the solver was damped in the held cycle'),
    'active_limit': ('', '', 'TeleopState.LIMIT_* of the held cycle'),
    'hold_reason': ('', '', 'TeleopState.HOLD_* of the held cycle'),
    'armed': ('', '', 'teleop armed'),
    'translation_scale': ('', '', 'operator translation scale'),
    'input_age': ('s', '', 'age of the newest operator input frame in the held cycle, on the '
                  "teleop node's monotonic clock; NaN where it had no input frame"),
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


def _ticks(ep: Episode, state: Stream, rate_hz: Optional[float], checks: dict,
           tick_stream: Optional[str]) -> Tuple[np.ndarray, dict, Optional[np.ndarray]]:
    """
    Return the tick times, their description for export.json and the tick stream's rows.

    The rows are those of ``tick_stream`` at each tick (each row exactly once); None for a
    fixed grid.
    """
    t0 = float(np.min(state.stamp))
    t1 = float(np.max(state.stamp))
    if not tick_stream:
        rate = float(rate_hz if rate_hz else checks['export']['rate_hz'])
        n = int(math.floor((t1 - t0) * rate + 1e-9)) + 1
        t = t0 + np.arange(n) / rate
        return t, {'mode': 'fixed', 'rate_hz': rate,
                   'rule': 't_k = t_start + k / rate_hz, from the first to the last state '
                           'stamp'}, None
    spec = ep.profile['streams'].get(tick_stream)
    if spec is None or spec.get('kind') not in TICK_KINDS:
        raise ValueError(f'{tick_stream}: ticks can only follow a recorded image stream')
    st = _rows(ep, tick_stream)
    if st is None:
        raise ValueError(f'{tick_stream}: no frames recorded')
    if rate_hz:
        raise ValueError('a tick stream gives the ticks; a rate cannot be given with it')
    finite = np.flatnonzero(np.isfinite(st.stamp))
    if not len(finite):
        raise ValueError(f'{tick_stream}: no frame has a finite stamp')
    rows = finite[np.argsort(st.stamp[finite], kind='stable')]
    t = np.asarray(st.stamp[rows], dtype=float)
    dt = np.diff(t)
    return t, {
        'mode': 'stream', 'stream': tick_stream,
        'rate_hz': spec.get('rate_hz'),
        'rule': f'one tick per frame of {tick_stream}, at its stamp, in stamp order; rate_hz '
                "is that stream's nominal rate, the spacing varies with the frames",
        'dt_min_s': float(dt.min()) if len(dt) else None,
        'dt_max_s': float(dt.max()) if len(dt) else None,
        'ticks_before_first_state': int(np.sum(t < t0)),
        'ticks_after_last_state': int(np.sum(t > t1)),
    }, rows


def build_export(ep: Episode, checks: dict, rate_hz: Optional[float] = None,
                 verdict=None, tick_stream: Optional[str] = None
                 ) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Return ``(arrays, meta)``: the episode on a tick grid and its description.

    The grid is fixed at ``rate_hz`` (default: checks.yaml export.rate_hz), or, with
    ``tick_stream`` (an image stream), one tick per frame of that stream at its stamp.
    ``verdict`` is the checker's report (or just its verdict string) and is copied into the
    metadata. Raises ValueError when the state stream is missing or lacks a needed field.
    """
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
    t, ticks, tick_rows = _ticks(ep, state, rate_hz, checks, tick_stream)
    rate = ticks['rate_hz']
    n = len(t)
    arrays: Dict[str, np.ndarray] = {'t': t}
    pairs = clock_pairs(ep)
    if pairs is not None:
        arrays['t_wall'] = rec_to_wall(pairs, t)

    si = hold_index(state.stamp, t)
    arrays['state_age'] = _age(state.stamp, si, t)
    have = si >= 0                       # False only for ticks before the first state cycle

    def held(key, fill, rows=si):
        """Return a state field at the held state row (``fill`` before the first row)."""
        values = np.asarray(state.fields[key])
        out = values[np.maximum(rows, 0)].copy()
        out[rows < 0] = fill
        return out

    engaged = held('clutch_engaged', False).astype(bool)
    arrays['clutch_engaged'] = engaged
    cycle_stamp = np.where(have, state.stamp[np.maximum(si, 0)], np.nan)

    # Action: the target of the held cycle.
    target = _rows(ep, 'target')
    if target is not None:
        ti = hold_index(target.stamp, t)
        arrays['action_pose'] = _gather(target.fields['pose'], ti)
        arrays['action_pose_age'] = _age(target.stamp, ti, t)
        same = np.zeros(n, dtype=bool)
        ok = (ti >= 0) & have
        same[ok] = np.abs(target.stamp[ti[ok]] - cycle_stamp[ok]) <= 0.5 * state_period
        arrays['action_valid'] = engaged & same
    else:
        arrays['action_pose'] = np.full((n, 7), np.nan)
        arrays['action_pose_age'] = np.full(n, np.nan)
        arrays['action_valid'] = np.zeros(n, dtype=bool)

    # Contract (Recording): gripper cycles marked MEASURED are never actions, also when teleop
    # commanded from the measured opening (no held reference arrived): masked conservatively.
    source = held('gripper_reference_source', -1).astype(np.int64)
    arrays['gripper_action_valid'] = (held('gripper_active', False).astype(bool)
                                      & (source != SOURCE_MEASURED))
    arrays['gripper_command'] = held('gripper_command', np.nan).astype(float)
    arrays['gripper_request'] = held('gripper_request', -1).astype(np.int64)
    arrays['gripper_source'] = source

    constants: Dict[str, List[str]] = {}
    command = _rows(ep, 'joint_command')
    if command is not None:
        ci = hold_index(command.stamp, t)
        arrays['joint_command'] = _gather(command.fields['velocities'], ci)
        arrays['joint_command_age'] = _age(command.stamp, ci, t)
        same = np.zeros(n, dtype=bool)
        ok = (ci >= 0) & have
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

    # Measured fields: the newest row whose cycle stamp and gripper sample stamp are both not
    # later than the tick (the sample can be newer than the cycle; see the module docstring).
    all_g_stamp = np.asarray(state.fields['gripper_measured_stamp'], dtype=float)
    all_g_ok = (np.asarray(state.fields['gripper_measured_valid']).astype(bool)
                & (all_g_stamp > 0.0))
    measured_key = np.where(all_g_ok, np.maximum(state.stamp, all_g_stamp), state.stamp)
    mi = hold_index(measured_key, t)
    arrays['measured_row_age'] = _age(measured_key, mi, t)
    pose_ok = held('measured_tool_pose_valid', False, mi).astype(bool)
    pose = held('measured_tool_pose', np.nan, mi).astype(float)
    pose[~pose_ok] = np.nan
    arrays['tool_pose_measured'] = pose
    arrays['tool_pose_measured_valid'] = pose_ok
    # The tool pose has no stamp of its own: it is bounded by the gripper sample's stamp only
    # where the held row has one (see FIELD_INFO).
    arrays['tool_pose_measured_bounded'] = (mi >= 0) & all_g_ok[np.maximum(mi, 0)]
    g_ok = held('gripper_measured_valid', False, mi).astype(bool) & (mi >= 0)
    g_ok &= all_g_ok[np.maximum(mi, 0)]
    g_value = held('gripper_measured', np.nan, mi).astype(float)
    g_value[~g_ok] = np.nan
    arrays['gripper_measured'] = g_value
    arrays['gripper_measured_valid'] = g_ok
    g_age = _age(np.where(all_g_ok, all_g_stamp, np.nan), mi, t)   # stamps of valid samples
    g_age[~g_ok] = np.nan
    arrays['gripper_measured_age'] = g_age

    arrays['damped'] = held('damped', False).astype(bool)
    arrays['active_limit'] = held('active_limit', -1).astype(np.int64)
    arrays['hold_reason'] = held('hold_reason', -1).astype(np.int64)
    arrays['armed'] = held('armed', False).astype(bool)
    arrays['translation_scale'] = held('translation_scale', np.nan).astype(float)
    input_age = held('input_age', np.nan).astype(float)
    input_age[input_age == -1.0] = np.nan            # TeleopState's sentinel: no input frame
    arrays['input_age'] = input_age

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
        ii = tick_rows if name == tick_stream else hold_index(st.stamp, t)
        arrays[f'image_{name}_index'] = ii.astype(np.int64)
        arrays[f'image_{name}_age'] = _age(st.stamp, ii, t)
        enc = _encoding(ep, name)
        ext = 'npy' if enc == '32FC1' else 'png'
        used = np.unique(ii[ii >= 0])
        images[name] = {'encoding': enc, 'file_extension': ext,
                        'index_array': f'image_{name}_index',
                        'age_array': f'image_{name}_age',
                        'path': f'images/{name}/<row:06d>.{ext}',
                        'frames_recorded': int(len(st)),
                        'frames_used': int(len(used)),
                        'frames_unused': int(len(st) - len(used)),
                        'ticks_repeating_frame': int(np.sum((ii[1:] == ii[:-1])
                                                            & (ii[1:] >= 0)))}
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
        if name == 'state':
            idx = si
        elif name == tick_stream:
            idx = tick_rows
        else:
            idx = hold_index(st.stamp, t)
        age = _age(st.stamp, idx, t)
        with np.errstate(invalid='ignore'):
            valid &= np.isfinite(age) & (age <= max_age * period(spec) + TIME_EPS)
    arrays['tick_valid'] = valid

    absent_fields = {}
    for name, spec in recorded(ep.profile).items():
        st = _rows(ep, name)
        if st is not None and spec.get('kind') in OPTIONAL_FIELDS:
            absent_fields[name] = st.absent_fields()
    meta = _meta(ep, arrays, rate, t0, t1, constants, images, base_name, verdict, pairs)
    meta['ticks'] = jsonable(ticks)
    meta['fields_absent'] = absent_fields
    meta['state_window'] = {'t_first': t0, 't_last': t1}
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
        checker = {k: verdict.get(k) for k in ('verdict', 'reasons', 'flags', 'usable',
                                               'tolerances', 'checked_by')}
    elif verdict is not None:
        checker = {'verdict': str(verdict)}
    meta = {
        'format': FORMAT,
        'rate_hz': rate,
        'n_ticks': int(len(arrays['t'])),
        'rule': RULE,
        'time': {'clock': 'recording clock (t)', 't_start': float(arrays['t'][0]),
                 't_stop': float(arrays['t'][-1]),
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
