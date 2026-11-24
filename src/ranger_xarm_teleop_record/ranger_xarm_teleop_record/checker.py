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
Episode checker (no ROS): judges one recorded episode against the deployment profile.

``check_episode`` reads an ``Episode`` (stream tables, sidecar, profile) and the tolerances of
``config/checks.yaml`` and returns a JSON-serialisable report. It never changes the episode and
never repairs, resamples or filters raw data: a stream that is late, thin, on another clock or
missing is reported, and fails the episode.

Report keys: ``verdict`` (PASS, FAIL, REFUSED), ``reasons``, ``flags`` (INV-4 notices; each
``{flag, reason, time}``), ``usable``, ``streams`` (one entry per recorded profile stream),
``link``, ``rtf``, ``base``, ``stability`` (information only, never a reason or a flag: the
stability monitor's recorded state, see ``_stability_report``), ``limits``, ``consistency``,
``missing_metadata``, ``window``,
``fields`` (optional fields absent per stream, see streams.py) and ``tolerances`` (which
profile's overrides were applied over checks.yaml, which values are provisional, and every
value in force) and ``checked_by`` (the commit, dirty state and a source digest of the
checking code), so a verdict can be reproduced.

Tolerances: when ``checks`` has not been through ``profile.effective_checks`` the overrides of
the episode's own recorded profile apply. The command-line tools apply those of the package's
current profile of the same name instead, so a refitted or provisional tolerance also judges
episodes recorded before it was set (the report says which source was used).

Times are seconds on the recording clock unless a key says wall. A ``host_wall`` stream is
mapped onto the recording clock with the clock pairs of the ``episode_state`` stream
(``stamp`` is recording time, ``wall_time`` the host wall time at that stamp), piecewise
linear and linearly extrapolated at the ends.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ranger_xarm_teleop_record import consistency
from ranger_xarm_teleop_record import metadata
from ranger_xarm_teleop_record.profile import effective_checks, period, recorded
from ranger_xarm_teleop_record.provenance import checking_code
from ranger_xarm_teleop_record.streams import Episode, OPTIONAL_FIELDS, Stream

# Replicated from ranger_xarm_teleop_interfaces/msg/TeleopState.msg (HOLD_* and LIMIT_*).
HOLD_NAMES = {
    0: 'HOLD_NONE', 1: 'HOLD_MOTION_DISABLED', 2: 'HOLD_NOT_ARMED', 3: 'HOLD_NOT_OWNER',
    4: 'HOLD_NOT_CALIBRATED', 5: 'HOLD_CLUTCH_RELEASED', 6: 'HOLD_INPUT_STALE',
    7: 'HOLD_POSE_ABSENT', 8: 'HOLD_POSE_EMULATED', 9: 'HOLD_OUT_OF_VIEW',
    10: 'HOLD_HEAD_ABSENT', 11: 'HOLD_HEAD_EMULATED', 12: 'HOLD_RESET_CHANGED',
    13: 'HOLD_SESSION_CHANGED', 14: 'HOLD_SEQ_BACKWARDS', 15: 'HOLD_INPUT_JUMP',
    16: 'HOLD_STEP_REJECTED', 17: 'HOLD_TOOL_POSE_STALE', 18: 'HOLD_SERVO_NOT_READY',
    19: 'HOLD_SERVO_HALTED', 20: 'HOLD_SOLVER',
    21: 'HOLD_STOPPING'}
HOLD_NONE = 0
HOLD_MOTION_DISABLED = 1
HOLD_NOT_ARMED = 2
HOLD_CLUTCH_RELEASED = 5
# A clutch release with any other hold reason is a fault, not the operator's choice.
FAULT_HOLDS = frozenset(k for k in HOLD_NAMES
                        if k not in (HOLD_NONE, HOLD_MOTION_DISABLED, HOLD_NOT_ARMED,
                                     HOLD_CLUTCH_RELEASED))
LIMIT_NAMES = {
    0: 'LIMIT_NONE', 1: 'LIMIT_OFFSET_TRANSLATION', 2: 'LIMIT_OFFSET_ROTATION',
    3: 'LIMIT_KEEP_OUT', 4: 'LIMIT_SERVO', 5: 'LIMIT_LEAD', 6: 'LIMIT_TOOL_SPEED',
    7: 'LIMIT_JOINT_VELOCITY', 8: 'LIMIT_JOINT_POSITION', 9: 'LIMIT_LINK_KEEPOUT',
    10: 'LIMIT_JOINT_ACCELERATION'}
LIMIT_NONE = 0

# Replicated from ranger_xarm_teleop_interfaces/msg/StabilityState.msg (REASON_*, TILT_*, EDGE_*).
STABILITY_REASON_NAMES = {
    0: 'REASON_OK', 1: 'REASON_NO_DESCRIPTION', 2: 'REASON_DESCRIPTION_UNUSABLE',
    3: 'REASON_NO_MASS_FILE', 4: 'REASON_MASS_FILE_REFUSED', 5: 'REASON_NO_JOINT_STATES',
    6: 'REASON_JOINT_STATES_STALE', 7: 'REASON_NO_ODOMETRY', 8: 'REASON_ODOMETRY_STALE',
    9: 'REASON_NO_TILT', 10: 'REASON_MOVING_WITHOUT_TILT', 11: 'REASON_NOT_FINITE',
    12: 'REASON_NOT_SIMULATION'}
TILT_NONE = 0
TILT_MEASURED = 1
TILT_LEVEL_ASSUMED = 2
STABILITY_TILT_NAMES = {TILT_NONE: 'TILT_NONE', TILT_MEASURED: 'TILT_MEASURED',
                        TILT_LEVEL_ASSUMED: 'TILT_LEVEL_ASSUMED'}
# The edge of the contact hull by the direction of its outward normal in the base frame.
EDGE_NONE = 0
EDGE_FRONT = 1
EDGE_FRONT_LEFT = 2
EDGE_LEFT = 3
EDGE_REAR_LEFT = 4
EDGE_REAR = 5
EDGE_REAR_RIGHT = 6
EDGE_RIGHT = 7
EDGE_FRONT_RIGHT = 8
EDGE_NAMES = {EDGE_NONE: 'none', EDGE_FRONT: 'front', EDGE_FRONT_LEFT: 'front-left',
              EDGE_LEFT: 'left', EDGE_REAR_LEFT: 'rear-left', EDGE_REAR: 'rear',
              EDGE_REAR_RIGHT: 'rear-right', EDGE_RIGHT: 'right',
              EDGE_FRONT_RIGHT: 'front-right'}

CMD_VEL_EPS = 1e-6
# BASE_DURING_CLUTCH values (checks.yaml flags), used when an older tolerance file lacks them.
BASE_CLUTCH_DEFAULTS = {'base_clutch_cmd_window_s': 0.05, 'base_clutch_linear_mps': 0.01,
                        'base_clutch_turn_rps': 0.02}
DEFAULT_FAIL_FLAGS = ('DATA_LOST', 'RECORDER_ERROR', 'TELEOP_STATE_LOST')
# What a gated stream's absence means when its gate never opened in the episode.
GATE_ACTION = {'engaged': 'teleop arm action', 'gripper_active': 'teleop gripper action'}


# --------------------------------------------------------------------------------------------
# Clock helpers (shared with export)
# --------------------------------------------------------------------------------------------

def extrap_interp(x, xp, fp) -> np.ndarray:
    """
    Piecewise-linear interpolation of ``fp(xp)`` at ``x``, extrapolated linearly at the ends.

    ``xp`` must be strictly increasing with at least two points (``np.interp`` alone would
    hold the end values constant).
    """
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    if len(xp) < 2 or len(xp) != len(fp):
        raise ValueError('extrap_interp needs at least two matching points')
    y = np.interp(x, xp, fp)
    lo = x < xp[0]
    if np.any(lo):
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    hi = x > xp[-1]
    if np.any(hi):
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def clock_pairs(ep: Episode) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return ``(recording, wall)`` clock pairs from the episode_state stream, or None.

    Pairs are ordered by recording time; a pair that does not advance both clocks is dropped.
    At least two pairs are needed to map one clock onto the other.
    """
    st = ep.get('episode_state')
    if st is None or len(st) < 2 or 'wall_time' not in st.fields:
        return None
    rec = np.asarray(st.stamp, dtype=float)
    wall = np.asarray(st.fields['wall_time'], dtype=float)
    ok = np.isfinite(rec) & np.isfinite(wall) & (wall > 0.0)
    rec, wall = rec[ok], wall[ok]
    order = np.argsort(rec, kind='stable')
    rec, wall = rec[order], wall[order]
    keep = [0] if len(rec) else []
    for i in range(1, len(rec)):
        if rec[i] > rec[keep[-1]] and wall[i] > wall[keep[-1]]:
            keep.append(i)
    if len(keep) < 2:
        return None
    return rec[keep], wall[keep]


def wall_to_rec(pairs, wall) -> np.ndarray:
    """Map host wall times onto the recording clock."""
    return extrap_interp(wall, pairs[1], pairs[0])


def rec_to_wall(pairs, rec) -> np.ndarray:
    """Map recording times onto the host wall clock."""
    return extrap_interp(rec, pairs[0], pairs[1])


def rec_stamps(stream: Stream, spec: dict, pairs) -> Optional[np.ndarray]:
    """Return a stream's stamps on the recording clock, or None when they cannot be mapped."""
    if spec.get('clock') == 'host_wall':
        if pairs is None:
            return None
        return wall_to_rec(pairs, stream.stamp)
    return np.asarray(stream.stamp, dtype=float)


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------

def jsonable(obj):
    """Return ``obj`` with numpy types as Python ones and non-finite floats as None."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def _flag(flags: List[dict], flag: str, reason: str, time: Optional[float] = None) -> None:
    """Append one flag."""
    flags.append({'flag': flag, 'reason': reason,
                  'time': None if time is None else float(time)})


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return the ``(first, last)`` row indices of each run of True in ``mask``."""
    out = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _spans(stamp: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    """Return the time spans ``(first stamp, last stamp)`` of each run where ``mask`` holds."""
    return [(float(np.min(stamp[a:b + 1])), float(np.max(stamp[a:b + 1])))
            for a, b in _runs(np.asarray(mask, dtype=bool))]


def _max_gap(stamps: np.ndarray) -> float:
    """Return the largest forward step between consecutive stamps (0 for fewer than two)."""
    if len(stamps) < 2:
        return 0.0
    return float(max(0.0, np.max(np.diff(stamps))))


def _median(values: np.ndarray) -> Optional[float]:
    """Return the median, or None for an empty array."""
    return float(np.median(values)) if len(values) else None


def _state_field(state: Stream, key: str, problems: List[str]) -> Optional[np.ndarray]:
    """Return a state field, or None after recording that it is missing."""
    value = state.fields.get(key)
    if value is None:
        problems.append(f'state: field {key} missing')
    return value


# --------------------------------------------------------------------------------------------
# Per-stream checks
# --------------------------------------------------------------------------------------------

class _Context:
    """What the per-stream checks share."""

    def __init__(self, ep: Episode, checks: dict):
        """Collect the window, the clock pairs, the gate spans and the tolerances."""
        self.ep = ep
        self.tol = checks.get('streams', {})
        self.state = ep.streams['state']
        self.window = (float(np.min(self.state.stamp)), float(np.max(self.state.stamp)))
        self.pairs = clock_pairs(ep)
        spec = ep.profile['streams']['state']
        self.state_period = period(spec)
        self.state_latency = float(np.median(self.state.recv - self.state.stamp))
        self.gates: Dict[str, List[Tuple[float, float]]] = {}
        for gate, key in (('engaged', 'clutch_engaged'), ('gripper_active', 'gripper_active')):
            values = self.state.fields.get(key)
            self.gates[gate] = [] if values is None else _spans(
                self.state.stamp, np.asarray(values, dtype=bool))


def _gap_limit(tol: dict, per: Optional[float]) -> float:
    """Return the largest gap a stream may have."""
    return max(float(tol['gap_max_s']),
               float(tol['gap_max_periods']) * (per if per is not None else 0.0))


def _gated_stats(stamps: np.ndarray, spans, ctx: _Context) -> dict:
    """Rate and gap numbers of one stream over the gate spans."""
    pad = 0.5 * ctx.state_period
    min_span = float(ctx.tol['min_gate_span_s'])
    steps = 0
    duration = 0.0
    gap = 0.0
    n_in = 0
    for a, b in spans:
        sel = stamps[(stamps >= a - pad) & (stamps <= b + pad)]
        n_in += len(sel)
        if len(sel) == 0:
            gap = max(gap, b - a)
            continue
        gap = max(gap, _max_gap(sel), float(sel[0]) - a, b - float(sel[-1]))
        if b - a >= min_span and len(sel) >= 2:
            steps += len(sel) - 1
            duration += float(np.max(sel) - np.min(sel))
    rate = steps / duration if duration > 0.0 else None
    return {'n_window': n_in, 'rate_hz': rate, 'max_gap_s': gap, 'n_spans': len(spans)}


def _window_stats(stamps: np.ndarray, ctx: _Context) -> dict:
    """Rate and gap numbers of one stream over the episode window."""
    w0, w1 = ctx.window
    sel = stamps[(stamps >= w0 - 1e-9) & (stamps <= w1 + 1e-9)]
    span = float(np.max(sel) - np.min(sel)) if len(sel) >= 2 else 0.0
    rate = (len(sel) - 1) / span if span > 0.0 else None
    # The stretches before the first sample and after the last count as gaps too: a stream
    # that stalls at the start or the end of an episode is missing data there.
    if len(sel):
        gap = max(_max_gap(sel), float(np.min(sel)) - w0, w1 - float(np.max(sel)))
    else:
        gap = w1 - w0
    return {'n_window': len(sel), 'rate_hz': rate, 'max_gap_s': gap, 'n_spans': None}


def _check_stream(name: str, spec: dict, ctx: _Context, reasons: List[str],
                  domain: Dict[str, List[str]]) -> dict:
    """Judge one recorded stream; append reasons; return its report entry."""
    ep, tol = ctx.ep, ctx.tol
    st = ep.get(name)
    n = 0 if st is None else len(st)
    nominal = spec.get('rate_hz')
    per = period(spec)
    gate = spec.get('gate')
    out = {'present': n > 0, 'required': bool(spec.get('required')), 'n': n, 'gate': gate,
           'clock': spec.get('clock'), 'nominal_hz': nominal, 'ok': True, 'problems': []}
    problems: List[str] = out['problems']
    if n == 0:
        if spec.get('required') and gate and not ctx.gates.get(gate):
            # Nothing to record: the gate never opened (an episode without an engaged cycle).
            # check_episode reports this once for the episode, not once per gated stream.
            problems.append(f'absent: no {gate} cycle in the episode')
            out['ok'] = False
            out['gate_never_open'] = True
            return out
        if spec.get('required'):
            problems.append('required stream missing')
        if gate and ctx.gates.get(gate):
            problems.append(f'gated stream absent while its {gate} spans exist')
        out['ok'] = not problems
        reasons.extend(f'{name}: {p}' for p in problems)
        return out

    mapped = rec_stamps(st, spec, ctx.pairs)
    # Clock domain and offset: over every row, so a stream on another clock cannot hide by
    # having no row inside the window.
    has_header = st.has_header and spec.get('clock') in ('robot', 'host_wall')
    latency = None
    if has_header:
        if mapped is None:
            out['domain_ok'] = False
            problems.append('wall-clock stamps cannot be mapped: no clock pairs')
            domain.setdefault('no_pairs', []).append(name)
        else:
            latency = float(np.median(st.recv - mapped))
            out['median_latency_s'] = latency
            out['domain_ok'] = abs(latency) <= float(tol['domain_max_s'])
            if not out['domain_ok']:
                problems.append(f'median(recv - stamp) = {latency:.3f} s: another clock')
                key = 'robot' if spec.get('clock') == 'robot' else 'host_wall'
                domain.setdefault(key, []).append(name)
    if latency is not None and name != 'state':
        offset = latency - ctx.state_latency
        out['offset_s'] = offset
        if nominal is not None:
            # Rendered or captured images may arrive up to a frame after their stamp: their
            # factor is fitted separately (checks.yaml); every other stream keeps its own.
            factor = (tol.get('offset_max_periods_image', tol['offset_max_periods'])
                      if spec.get('kind') == 'image' else tol['offset_max_periods'])
            limit = float(factor) * max(per, ctx.state_period)
            out['offset_limit_s'] = limit
            out['offset_ok'] = abs(offset) <= limit
            if not out['offset_ok']:
                problems.append(f'offset to the action stream {offset * 1e3:.1f} ms exceeds '
                                f'{limit * 1e3:.1f} ms')

    # Monotonic stamps (reported always, not repaired).
    out['non_monotonic'] = int(np.sum(np.diff(st.stamp) < 0.0)) if n > 1 else 0
    judged = nominal is not None or gate is not None
    if out['non_monotonic'] and judged:
        problems.append(f'{out["non_monotonic"]} stamps earlier than the one before')

    # Rate and gaps.
    if mapped is not None and (nominal is not None or gate is not None):
        if gate:
            spans = ctx.gates.get(gate, [])
            stats = _gated_stats(mapped, spans, ctx)
            if not spans:
                stats['rate_hz'] = None
        else:
            stats = _window_stats(mapped, ctx)
        out.update(stats)
        limit = _gap_limit(tol, per)
        out['gap_limit_s'] = limit
        out['gap_ok'] = stats['max_gap_s'] <= limit
        if not out['gap_ok']:
            problems.append(f'largest gap {stats["max_gap_s"]:.3f} s exceeds {limit:.3f} s')
        if nominal is not None and stats['rate_hz'] is not None:
            floor = float(tol['rate_min_fraction']) * float(nominal)
            out['rate_ok'] = stats['rate_hz'] >= floor
            if not out['rate_ok']:
                problems.append(f'rate {stats["rate_hz"]:.2f} Hz below {floor:.2f} Hz')
            if ctx.pairs is not None and n > 1:
                first, last = float(mapped[0]), float(mapped[-1])
                wall = rec_to_wall(ctx.pairs, np.array([first, last]))
                if wall[1] > wall[0] and last > first:
                    out['rate_wall_hz'] = stats['rate_hz'] * (last - first) / (wall[1] - wall[0])
    out['ok'] = not problems
    reasons.extend(f'{name}: {p}' for p in problems)
    return out


def _field_report(ep: Episode, reasons: List[str]) -> dict:
    """Report the optional fields absent per stream; fail on a profile's required_fields."""
    out = {}
    for name, spec in recorded(ep.profile).items():
        st = ep.get(name)
        if spec.get('kind') not in OPTIONAL_FIELDS or st is None or not len(st):
            continue
        absent = st.absent_fields()
        required = list(spec.get('required_fields') or [])
        out[name] = {'absent': absent, 'required_fields': required}
        for key in required:
            if key in absent:
                reasons.append(f'FIELD_MISSING: {name}: required field {key} has no finite '
                               'value in the episode')
    return out


def _check_counts(ep: Episode, checks: dict, streams_report: dict, flags: List[dict],
                  reasons: List[str]) -> None:
    """Compare the episode node's publication counts with the rows recorded (INV-2, INV-4)."""
    counts = ep.sidecar.get('counts') or {}
    specs = ep.profile['streams']
    for name, entry in counts.items():
        st = ep.get(name)
        recorded_n = 0
        window = entry.get('window') if isinstance(entry, dict) else None
        counted = entry.get('count') if isinstance(entry, dict) else None
        if window is None or counted is None:
            reasons.append(f'{name}: malformed count entry')
            continue
        a, b = float(window[0]), float(window[1])
        if st is not None:
            recorded_n = int(np.sum((st.stamp >= a) & (st.stamp <= b)))
        info = {'window': [a, b], 'counted': int(counted), 'recorded': recorded_n}
        entries = [(name, recorded_n)]
        for image_name, image_spec in specs.items():
            if image_spec.get('kind') == 'image' and image_spec.get('info') == name \
                    and image_spec.get('record', True):
                image = ep.get(image_name)
                n_image = 0 if image is None else int(
                    np.sum((image.stamp >= a) & (image.stamp <= b)))
                info[f'recorded_{image_name}'] = n_image
                entries.append((image_name, n_image))
        for stream_name, n_rec in entries:
            if n_rec < counted:
                text = (f'{stream_name}: {n_rec} rows recorded, {counted} counted by the '
                        f'episode node ({counted - n_rec} lost)')
                _flag(flags, 'DATA_LOST', text, None)
                reasons.append(f'DATA_LOST: {text}')
        if name in streams_report:
            streams_report[name]['count'] = info
        else:
            streams_report[name] = {'count': info}


# --------------------------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------------------------

def _state_flags(ep: Episode, checks: dict, flags: List[dict], problems: List[str]) -> dict:
    """FAULT_RELEASE, LIMIT_BOUND and DISARMED from the state stream; return the limit report."""
    st = ep.streams['state']
    stamp = st.stamp
    engaged = _state_field(st, 'clutch_engaged', problems)
    hold = _state_field(st, 'hold_reason', problems)
    limit = _state_field(st, 'active_limit', problems)
    armed = _state_field(st, 'armed', problems)
    report: dict = {}
    if engaged is not None and hold is not None:
        engaged = np.asarray(engaged, dtype=bool)
        for i in range(1, len(engaged)):
            if engaged[i - 1] and not engaged[i] and int(hold[i]) in FAULT_HOLDS:
                name = HOLD_NAMES.get(int(hold[i]), f'HOLD_{int(hold[i])}')
                _flag(flags, 'FAULT_RELEASE', name, stamp[i])
    if engaged is not None and limit is not None:
        engaged = np.asarray(engaged, dtype=bool)
        cycles = int(np.sum(engaged))
        shares = {}
        if cycles:
            values = np.asarray(limit)[engaged].astype(int)
            for code in sorted(set(values.tolist())):
                shares[LIMIT_NAMES.get(code, f'LIMIT_{code}')] = float(
                    np.sum(values == code)) / cycles
            bound = float(np.sum(values != LIMIT_NONE)) / cycles
        else:
            bound = 0.0
        report = {'engaged_cycles': cycles, 'bound_share': bound, 'per_limit': shares}
        maximum = float(checks['flags']['limit_bound_max_share'])
        if bound > maximum:
            _flag(flags, 'LIMIT_BOUND',
                  f'{bound:.0%} of engaged cycles had a limit active (maximum {maximum:.0%})')
    stop_reason = metadata.lookup(ep.sidecar, 'episode.stop_reason')
    disarmed_at = None
    if armed is not None:
        armed = np.asarray(armed, dtype=bool)
        for i in range(1, len(armed)):
            if armed[i - 1] and not armed[i]:
                disarmed_at = float(stamp[i])
                break
    if disarmed_at is not None:
        _flag(flags, 'DISARMED', 'armed went false during the episode', disarmed_at)
    elif stop_reason == 'disarmed':
        _flag(flags, 'DISARMED', 'the episode stopped because teleop was disarmed',
              metadata.lookup(ep.sidecar, 'episode.stop.ros'))
    return report


def _rtf_report(ep: Episode, flags: List[dict]) -> dict:
    """Real-time factor from the clock pairs; flag LOW_RTF below the profile's threshold."""
    pairs = clock_pairs(ep)
    threshold = ep.profile.get('low_rtf')
    if pairs is None:
        return {'mean': None, 'min_1s': None, 'low_rtf': threshold, 'n_pairs': 0}
    rec, wall = pairs
    mean = float((rec[-1] - rec[0]) / (wall[-1] - wall[0]))
    lowest = None
    j = 0
    for i in range(len(wall)):
        j = max(j, i + 1)
        while j < len(wall) and wall[j] - wall[i] < 1.0:
            j += 1
        if j >= len(wall):
            break
        value = float((rec[j] - rec[i]) / (wall[j] - wall[i]))
        lowest = value if lowest is None else min(lowest, value)
    if threshold is not None and mean < float(threshold):
        _flag(flags, 'LOW_RTF',
              f'real-time factor {mean:.2f} below {float(threshold):.2f}', float(rec[0]))
    return {'mean': mean, 'min_1s': lowest, 'low_rtf': threshold, 'n_pairs': int(len(rec))}


def _yaw(q: np.ndarray) -> np.ndarray:
    """Return the yaw of quaternions ``(n, 4)`` in xyzw order."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _hold_spans(stamp: np.ndarray, mask: np.ndarray, cap: Optional[float] = None
                ) -> List[Tuple[float, float]]:
    """
    Return the ``(start, end)`` times of each run of True in ``mask``.

    A row holds until the next row's stamp (at most ``cap`` s when given); the last row holds
    for no time.
    """
    out = []
    for first, last in _runs(mask):
        end = float(stamp[last + 1]) if last + 1 < len(stamp) else float(stamp[last])
        if cap is not None:
            end = min(end, float(stamp[last]) + cap)
        out.append((float(stamp[first]), end))
    return out


def _overlaps(intervals: List[Tuple[float, float]], spans: List[Tuple[float, float]]
              ) -> Tuple[int, Optional[float]]:
    """Return how many ``intervals`` touch a span, and the first time any of them does."""
    count, first = 0, None
    for a, b in intervals:
        hit = [max(a, s0) for s0, s1 in spans if a <= s1 and b >= s0]
        if hit:
            count += 1
            first = min(hit) if first is None else min(first, min(hit))
    return count, first


def _nonzero_command(st: Stream) -> np.ndarray:
    """Return a mask of the rows of a twist stream that command any motion."""
    twist = np.asarray(st.fields.get('twist', np.zeros((len(st), 6))), dtype=float)
    return np.any(np.abs(twist) > CMD_VEL_EPS, axis=1) if twist.size else np.zeros(
        len(st), dtype=bool)


def _base_during_clutch(ep: Episode, tol: dict, flags: List[dict], by_command: List[Stream],
                        report: dict) -> None:
    """
    Flag BASE_DURING_CLUTCH: the base commanded or moving while the arm's clutch was engaged.

    Only for an episode with the base node's state or its command recorded (base driving on);
    otherwise the report says it was not checked.

    The clutch is the ``state`` stream's ``clutch_engaged`` (a cycle holds until the next).
    The base is commanded for ``base_clutch_cmd_window_s`` after each nonzero command of the
    base command streams (``base_cmd``, ``/cmd_vel``) and moving while the ``odom`` stream's
    twist is above ``base_clutch_linear_mps`` or ``base_clutch_turn_rps``.
    """
    # Checked for episodes recorded with base driving on: the base node's state or its own
    # command (/teleop/base/cmd_vel) recorded, either of them, so that a lost state stream
    # does not skip the check. The exclusion it checks is the base node's and the gate's (G-19).
    if not any(ep.get(name) is not None and len(ep.get(name)) > 0
               for name in ('base_state', 'base_cmd')):
        report['clutch_overlap'] = {'checked': False,
                                    'why': 'no base state or base command recorded: base '
                                           'driving was off'}
        return
    state = ep.get('state')
    if state is None or len(state) == 0 or 'clutch_engaged' not in state.fields:
        report['clutch_overlap'] = {'checked': False, 'why': 'no teleop state'}
        return
    engaged = np.asarray(state.fields['clutch_engaged'], dtype=bool)
    spans = _hold_spans(np.asarray(state.stamp, dtype=float), engaged)
    tol = {**BASE_CLUTCH_DEFAULTS, **tol}
    window = float(tol['base_clutch_cmd_window_s'])
    reasons = []
    times = []
    overlap: dict = {'checked': True, 'engaged_spans': len(spans), 'by_command': None,
                     'by_odometry': None}
    names = [st.name for st in by_command]
    commands = [(float(t), float(t) + window) for st in by_command
                for t in st.stamp[_nonzero_command(st)]]
    n, first = _overlaps(commands, spans)
    if n:
        overlap['by_command'] = {'streams': names, 'commands': n, 'first_time': first}
        reasons.append(f'the base was commanded ({"/".join(names)}: {n} nonzero commands) '
                       f'from {first:.2f} s')
        times.append(first)
    odom = ep.get('odom')
    if odom is not None and len(odom) and 'twist' in odom.fields:
        twist = np.asarray(odom.fields['twist'], dtype=float)
        with np.errstate(invalid='ignore'):
            moving = (np.hypot(twist[:, 0], twist[:, 1]) > float(tol['base_clutch_linear_mps'])) \
                | (np.abs(twist[:, 5]) > float(tol['base_clutch_turn_rps']))
        n, first = _overlaps(_hold_spans(np.asarray(odom.stamp, dtype=float), moving), spans)
        if n:
            overlap['by_odometry'] = {'stream': 'odom', 'spans': n, 'first_time': first}
            reasons.append(f'the base was moving by odometry (above '
                           f'{float(tol["base_clutch_linear_mps"]):g} m/s or '
                           f'{float(tol["base_clutch_turn_rps"]):g} rad/s) from {first:.2f} s')
            times.append(first)
    report['clutch_overlap'] = overlap
    if reasons:
        _flag(flags, 'BASE_DURING_CLUTCH',
              'the arm clutch was engaged while ' + ' and '.join(reasons)
              + f'; first overlap at {min(times):.2f} s', min(times))


def _base_report(ep: Episode, checks: dict, flags: List[dict]) -> dict:
    """
    Report base motion from the streams the profile marks ``base: true``.

    Per odometry stream: ``travel_m``, ``yaw_change_rad`` (and the largest excursions). Per
    base command stream: ``nonzero_rows``. ``summary``: travel and yaw change by every odometry
    stream (``/odom``, ``/ground_truth/odom``), ``drive_time_s`` (seconds with ``base_state``
    driving, else seconds of nonzero ``base_cmd``; a row holds until the next, a command row
    at most ``streams.gap_max_s``), ``cmd_cycles`` and ``cmd_nonzero_cycles`` (``base_cmd``
    rows). ``clutch_overlap``: see ``_base_during_clutch``.
    """
    tol = checks['flags']
    base_streams = [(n, s) for n, s in recorded(ep.profile).items() if s.get('base')]
    present = [(n, s, ep.get(n)) for n, s in base_streams if ep.get(n) is not None
               and len(ep.get(n)) > 0]
    state_st = ep.get('base_state')
    if state_st is not None and len(state_st) == 0:
        state_st = None
    if not present and state_st is None:
        return {'mode': 'parked: no base topics recorded', 'streams': {}}
    report: dict = {'streams': {}}
    moved: List[str] = []
    travel: Dict[str, float] = {}
    yaw_change: Dict[str, float] = {}
    by_command: List[Stream] = []
    for name, spec, st in present:
        entry: dict = {'kind': spec.get('kind'), 'n': len(st)}
        if spec.get('kind') == 'odometry' and 'pose' in st.fields:
            pose = np.asarray(st.fields['pose'], dtype=float)
            p = pose[:, :3]
            yaw = _yaw(pose[:, 3:7])
            excursion = np.linalg.norm(p - p[0], axis=1)
            dyaw = np.abs(_wrap(yaw - yaw[0]))
            entry.update({
                'travel_m': float(np.linalg.norm(p[-1] - p[0])),
                'max_excursion_m': float(np.max(excursion)),
                'yaw_change_rad': float(dyaw[-1]),
                'max_yaw_excursion_rad': float(np.max(dyaw))})
            travel[name] = entry['travel_m']
            yaw_change[name] = entry['yaw_change_rad']
            if entry['max_excursion_m'] > float(tol['base_moved_m']) \
                    or entry['max_yaw_excursion_rad'] > float(tol['base_moved_rad']):
                moved.append(f'{name}: {entry["max_excursion_m"] * 1e3:.1f} mm, '
                             f'{math.degrees(entry["max_yaw_excursion_rad"]):.2f} deg')
        elif spec.get('kind') in ('twist', 'twist_stamped') and 'twist' in st.fields:
            twist = np.abs(np.asarray(st.fields['twist'], dtype=float))
            nonzero = np.any(twist > CMD_VEL_EPS, axis=1)
            entry['nonzero_rows'] = int(np.sum(nonzero))
            entry['max_abs'] = float(np.max(twist)) if twist.size else 0.0
            by_command.append(st)
            if np.any(nonzero):
                moved.append(f'{name}: {int(np.sum(nonzero))} nonzero commands')
        report['streams'][name] = entry
    summary: dict = {'travel_m': travel, 'yaw_change_rad': yaw_change}
    cap = float(checks.get('streams', {}).get('gap_max_s', 0.1))
    cmd = ep.get('base_cmd')
    if cmd is not None and len(cmd) and 'twist' in cmd.fields:
        nonzero = _nonzero_command(cmd)
        summary['cmd_cycles'] = int(len(cmd))
        summary['cmd_nonzero_cycles'] = int(np.sum(nonzero))
        summary['drive_time_cmd_s'] = float(sum(
            b - a for a, b in _hold_spans(np.asarray(cmd.stamp, dtype=float), nonzero, cap)))
    if state_st is not None and 'driving' in state_st.fields:
        driving = np.asarray(state_st.fields['driving'], dtype=bool)
        summary['drive_time_state_s'] = float(sum(
            b - a for a, b in _hold_spans(np.asarray(state_st.stamp, dtype=float), driving)))
        report['streams']['base_state'] = {'kind': 'base_state', 'n': len(state_st),
                                           'driving_rows': int(np.sum(driving))}
    for key in ('drive_time_state_s', 'drive_time_cmd_s'):
        if key in summary:
            summary['drive_time_s'] = summary[key]
            break
    report['summary'] = summary
    _base_during_clutch(ep, tol, flags, by_command, report)
    if moved:
        report['mode'] = 'moved'
        report['moved'] = moved
        _flag(flags, 'BASE_MOVED', '; '.join(moved))
    else:
        report['mode'] = 'parked: base topics present, no motion'
    return report


def edge_name(edge) -> str:
    """Return the name of a stability edge number (``unknown <n>`` for one not in the table)."""
    return EDGE_NAMES.get(int(edge), f'unknown {int(edge)}')


def _smallest(st: Stream, key: str, edge_key: str, t0: float,
              also: Tuple[str, ...] = ()) -> Optional[dict]:
    """Return the smallest finite value of a stability field, its edge and time since ``t0``."""
    values = np.asarray(st.fields.get(key, []), dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    row = int(np.argmin(np.where(finite, values, np.inf)))
    out = {'value': float(values[row]), 'time_s': float(st.stamp[row]) - t0}
    out['edge'] = edge_name(st.fields[edge_key][row])
    for name in also:                       # fields of the same message, as it was then
        out[name] = float(st.fields[name][row])
    return out


def _stability_report(ep: Episode, t0: float) -> dict:
    """
    Report the stability monitor's recorded state (information only: no reason, no flag).

    Not recorded: ``{'recorded': False}``. Recorded: ``n`` messages; ``unavailable`` (``n``,
    ``share`` of the messages and ``by_reason`` counts per reason name); the smallest finite
    ``margin_m``, ``margin_pessimistic_m``, ``stop_margin_m`` (with the ``stop_decel_m_s2`` it
    assumed) and ``energy_reserve_j``, each with ``value``, ``edge`` (name, see EDGE_NAMES)
    and ``time_s`` from ``t0`` (the episode's start); ``tilt_sources`` (counts per source name),
    ``level_floor_assumed`` (some message had no tilt source: its figure is for a level floor)
    and ``model_sha256`` (the distinct mass models, sorted). NaN figures are skipped. A
    message of a kind or field the stream lacks gives no entry for it.
    """
    st = ep.get('stability_state')
    if st is None or len(st) == 0:
        return {'recorded': False}
    n = len(st)
    out: dict = {'recorded': True, 'n': n}
    if 'available' in st.fields:
        unavailable = ~np.asarray(st.fields['available'], dtype=bool)
        by_reason: Dict[str, int] = {}
        for code in np.asarray(st.fields['reason'], dtype=int)[unavailable]:
            name = STABILITY_REASON_NAMES.get(int(code), f'REASON_UNKNOWN_{int(code)}')
            by_reason[name] = by_reason.get(name, 0) + 1
        count = int(np.sum(unavailable))
        out['unavailable'] = {'n': count, 'share': count / n, 'by_reason': by_reason}
    for key, edge_key in (('margin_m', 'edge'), ('margin_pessimistic_m', 'edge_pessimistic'),
                          ('stop_margin_m', 'stop_edge'), ('energy_reserve_j', 'energy_edge')):
        out[key] = _smallest(st, key, edge_key, t0,
                             ('stop_decel_m_s2',) if key == 'stop_margin_m' else ())
    if 'tilt_source' in st.fields:
        sources: Dict[str, int] = {}
        for code in np.asarray(st.fields['tilt_source'], dtype=int):
            name = STABILITY_TILT_NAMES.get(int(code), f'TILT_UNKNOWN_{int(code)}')
            sources[name] = sources.get(name, 0) + 1
        out['tilt_sources'] = sources
        out['level_floor_assumed'] = 'TILT_LEVEL_ASSUMED' in sources
    out['model_sha256'] = sorted({s for s in st.labels.get('model_sha256', []) if s})
    return out


def stability_lines(section: dict) -> List[str]:
    """Return the report lines of a ``_stability_report`` section (for the command line)."""
    if not section.get('recorded'):
        return ['stability: not recorded']
    lines = [f'stability: {section["n"]} messages']
    gone = section.get('unavailable')
    if gone:
        reasons = ', '.join(f'{k} {v}' for k, v in sorted(gone['by_reason'].items()))
        lines.append(f'  unavailable: {gone["n"]} ({gone["share"] * 100:.1f}%)'
                     + (f': {reasons}' if reasons else ''))
    for key, unit in (('margin_m', 'm'), ('margin_pessimistic_m', 'm'), ('stop_margin_m', 'm'),
                      ('energy_reserve_j', 'J')):
        found = section.get(key)
        if found is None:
            lines.append(f'  smallest {key}: no finite value')
            continue
        decel = found.get('stop_decel_m_s2')       # None when it was not finite (jsonable)
        extra = f', assuming a stop at {decel:g} m/s^2' if decel is not None else ''
        lines.append(f'  smallest {key}: {found["value"]:.4f} {unit} at the {found["edge"]} '
                     f'edge, {found["time_s"]:.2f} s from the start{extra}')
    sources = section.get('tilt_sources') or {}
    if sources:
        lines.append('  tilt sources: ' + ', '.join(f'{k} {v}' for k, v in sorted(sources.items()))
                     + ('; level floor assumed' if section.get('level_floor_assumed') else ''))
    hashes = section.get('model_sha256') or []
    lines.append(f'  mass models: {len(hashes)}'
                 + (': ' + ', '.join(h[:12] for h in hashes) if hashes else ''))
    return lines


def _link_report(ep: Episode, checks: dict, flags: List[dict]) -> dict:
    """Operator link quality over the window; flag LINK_DEGRADED."""
    tol = checks['flags']
    st = ep.streams['state']
    link: dict = {}
    age = st.fields.get('input_age')
    armed = st.fields.get('armed')
    values = np.zeros(0)
    if age is not None and armed is not None:
        age = np.asarray(age, dtype=float)
        values = age[np.asarray(armed, dtype=bool) & np.isfinite(age)]
    link['input_age_median_s'] = _median(values)
    link['input_age_p99_s'] = float(np.percentile(values, 99)) if len(values) else None
    link['input_age_max_s'] = float(np.max(values)) if len(values) else None
    status = None
    for name, spec in recorded(ep.profile).items():
        if spec.get('kind') == 'input_status' and ep.get(name) is not None \
                and len(ep.get(name)) > 0:
            status = ep.get(name)
            link['status_stream'] = name
            break
    gaps = backwards = reconnects = changes = 0
    if status is not None:
        seq = np.asarray(status.fields.get('seq', np.zeros(len(status))), dtype=np.int64)
        session = np.asarray(status.fields.get('session_id', np.zeros(len(status))),
                             dtype=np.int64)
        same = session[1:] == session[:-1]
        dseq = np.diff(seq)
        gaps = int(np.sum(same & (dseq > 1)))
        backwards = int(np.sum(same & (dseq <= 0)))
        changes = int(np.sum(~same))
        stamp_gaps = int(np.sum(np.diff(status.stamp) > float(tol['reconnect_gap_s'])))
        reconnects = stamp_gaps + changes
        link['status_stamp_gaps'] = stamp_gaps
        xr = np.asarray(status.fields.get('xr_time_ms', np.zeros(len(status))), dtype=float)
        dxr = np.diff(xr)[same]
        if len(dxr):
            period_ms = float(np.median(dxr))
            deviation = dxr - period_ms
            link['xr_period_ms'] = period_ms
            link['xr_jitter_std_ms'] = float(np.std(deviation))
            link['xr_jitter_p99_ms'] = float(np.percentile(np.abs(deviation), 99))
    link['n_status'] = 0 if status is None else len(status)
    link['seq_gaps'] = gaps
    link['seq_not_forward'] = backwards
    link['session_changes'] = changes
    link['reconnects'] = reconnects
    why = []
    p99 = link['input_age_p99_s']
    if p99 is not None and p99 > float(tol['link_input_age_p99_s']):
        why.append(f'input age p99 {p99 * 1e3:.0f} ms')
    if reconnects:
        why.append(f'{reconnects} reconnects')
    if gaps:
        why.append(f'{gaps} sequence gaps')
    if backwards:
        why.append(f'{backwards} sequence numbers not increasing')
    if why:
        _flag(flags, 'LINK_DEGRADED', ', '.join(why))
    return link


# --------------------------------------------------------------------------------------------
# The checker
# --------------------------------------------------------------------------------------------

def check_episode(ep: Episode, checks: dict) -> dict:
    """Judge one episode; return the report (see the module docstring). Never mutates ``ep``."""
    if '_tolerances' not in checks:
        checks = effective_checks(checks, ep.profile, 'recorded profile (episode sidecar)')
    reasons: List[str] = []
    flags: List[dict] = []
    absent = metadata.missing(ep.sidecar)
    for key in absent:
        reasons.append(f'MISSING_METADATA: {key}')
    report: dict = {'verdict': 'FAIL', 'reasons': reasons, 'flags': flags, 'usable': False,
                    'streams': {}, 'link': {}, 'rtf': {}, 'base': {}, 'limits': {},
                    'consistency': {}, 'missing_metadata': list(absent), 'window': None,
                    'fields': {},
                    'tolerances': dict(checks['_tolerances'],
                                       values={k: v for k, v in checks.items()
                                               if k != '_tolerances'}),
                    'checked_by': checking_code()}
    state = ep.get('state')
    if state is None or len(state) < 2:
        reasons.append('state: fewer than two rows, no action timeline')
        if absent:
            report['verdict'] = 'REFUSED'
        return jsonable(report)
    ctx = _Context(ep, checks)
    report['window'] = {'start': ctx.window[0], 'stop': ctx.window[1],
                        'duration_s': ctx.window[1] - ctx.window[0]}
    # The recording must lie inside the episode the sidecar describes (start acknowledged to
    # stop request): data from before START or after STOP means the recorder was not paused.
    start = metadata.lookup(ep.sidecar, 'episode.start.acknowledged.ros')
    if start is None:
        start = metadata.lookup(ep.sidecar, 'episode.start.ros')
    stop = metadata.lookup(ep.sidecar, 'episode.stop.ros')
    slack = float(ctx.tol.get('window_slack_s', 0.5))
    if start is not None and stop is not None:
        report['window'].update({'sidecar_start': start, 'sidecar_stop': stop})
        if ctx.window[0] < start - slack or ctx.window[1] > stop + slack:
            reasons.append(f'WINDOW: recorded {ctx.window[0]:.3f}-{ctx.window[1]:.3f} s lies '
                           f'outside the episode {start:.3f}-{stop:.3f} s (+-{slack:g} s): the '
                           'recorder was not paused around it')
    report['gate_spans'] = {k: [list(s) for s in v] for k, v in ctx.gates.items()}

    stream_reasons: List[str] = []
    domain: Dict[str, List[str]] = {}
    for name, spec in recorded(ep.profile).items():
        report['streams'][name] = _check_stream(name, spec, ctx, stream_reasons, domain)
    if domain.get('robot'):
        reasons.append('ONE_HOST: robot-side streams on more than one clock: '
                       + ', '.join(domain['robot']))
    if domain.get('host_wall'):
        reasons.append('CLOCK_PAIRS: wall-clock streams do not map onto the recording clock: '
                       + ', '.join(domain['host_wall']))
    if domain.get('no_pairs'):
        reasons.append('NO_CLOCK_PAIRS: no episode_state clock pairs to map: '
                       + ', '.join(domain['no_pairs']))
    no_gate = {}
    for name, entry in report['streams'].items():
        if entry.get('gate_never_open'):
            no_gate.setdefault(entry['gate'], []).append(name)
    for gate, names in no_gate.items():
        what = GATE_ACTION.get(gate, f'{gate} cycle')
        reasons.append(f'NO_ACTION: no {what} in the episode ({", ".join(names)} absent; the '
                       f'episode lasted {ctx.window[1] - ctx.window[0]:.2f} s)')
    reasons.extend(stream_reasons)
    report['fields'] = _field_report(ep, reasons)

    _check_counts(ep, checks, report['streams'], flags, reasons)

    state_problems: List[str] = []
    report['limits'] = _state_flags(ep, checks, flags, state_problems)
    reasons.extend(state_problems)
    report['rtf'] = _rtf_report(ep, flags)
    report['base'] = _base_report(ep, checks, flags)
    report['stability'] = _stability_report(ep, ctx.window[0])
    report['link'] = _link_report(ep, checks, flags)

    raised = {f['flag'] for f in flags}         # the checker's own judgement of the same flag wins
    for item in metadata.lookup(ep.sidecar, 'episode.flags') or []:
        if isinstance(item, dict) and str(item.get('flag')) not in raised:
            _flag(flags, str(item.get('flag')), str(item.get('reason', '')), item.get('time'))

    report['consistency'] = consistency.check(ep, checks)
    if report['consistency']['verdict'] == 'FAIL':
        reasons.append('CONSISTENCY: ' + '; '.join(report['consistency'].get('reasons', [])))

    # Flags that mean the data cannot be trusted fail the episode (checks.yaml fail_flags;
    # a tolerance file without the key fails closed, on the default list).
    fail_flags = checks.get('flags', {}).get('fail_flags', DEFAULT_FAIL_FLAGS)
    for flag in sorted({f['flag'] for f in flags} & set(fail_flags)):
        if not any(r.startswith(flag + ':') for r in reasons):
            reasons.append(f'{flag}: ' + '; '.join(
                f['reason'] for f in flags if f['flag'] == flag)[:300])
    other = [r for r in reasons if not r.startswith('MISSING_METADATA')]
    if absent:
        report['verdict'] = 'REFUSED'
    elif other:
        report['verdict'] = 'FAIL'
    else:
        report['verdict'] = 'PASS'
    report['usable'] = (report['verdict'] == 'PASS'
                        and metadata.lookup(ep.sidecar, 'episode.status') == 'kept')
    return jsonable(report)
