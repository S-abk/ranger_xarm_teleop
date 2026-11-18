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
Command-versus-measured consistency of one episode (no ROS).

This replaces a replay test. The commanded target (``target`` stream, published only while
teleop commands) is compared with the measured tool pose carried in the state stream, at the
lag that fits best. It catches a frame error (heading, wrong frame id), a stamp offset and
command and measured fields swapped. Samples where the solver damped or a limit shaped the
command are "limited": their residuals are reported but never judged. The gripper command is
compared with the measured opening in the same way, excluding grasp intervals.

Lag fit: for each lag on the grid the measured sample nearest to ``t + lag`` is taken (no
interpolation) when it lies within half a state period; the lag is the one with the smallest
translation RMS over the clean samples. Because the measured series is sampled at the state
rate, several neighbouring lags give identical residuals; the fitted lag is the middle of that
plateau.
"""

from typing import Optional, Tuple

import numpy as np

from ranger_xarm_teleop_record import metadata
from ranger_xarm_teleop_record.profile import period, PROFILE_TOLERANCES
from ranger_xarm_teleop_record.streams import Episode

# Replicated from ranger_xarm_teleop_interfaces/msg/TeleopState.msg and GripperCommand.msg.
LIMIT_NONE = 0
REQUEST_OPEN = 1
REQUEST_CLOSE = 2

MIN_CLEAN_ARM = 25         # fewer clean samples than this: not judged
MIN_CLEAN_GRIPPER = 10
GRASP_STATIONARY_RAD = 0.005   # measured opening change over the window below this: stationary
GRASP_WINDOW_S = 0.2
TIE_TOL = 1e-9             # residual RMS values closer than this are the same plateau
PATH_MAX_STEP_PERIODS = 2.0    # path length sums steps between cycles at most this far apart


# --------------------------------------------------------------------------------------------
# Numpy helpers
# --------------------------------------------------------------------------------------------

def _unit(q: np.ndarray) -> np.ndarray:
    """Return quaternions ``(n, 4)`` scaled to unit norm."""
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n > 0.0, n, 1.0)


def quat_angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the angle in rad of the rotation taking ``a`` to ``b`` (rows of xyzw)."""
    a, b = _unit(a), _unit(b)
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    # b * conj(a)
    x = -bw * ax + bx * aw - by * az + bz * ay
    y = -bw * ay + by * aw - bz * ax + bx * az
    z = -bw * az + bz * aw - bx * ay + by * ax
    w = bw * aw + bx * ax + by * ay + bz * az
    return 2.0 * np.arctan2(np.sqrt(x * x + y * y + z * z), np.abs(w))


def nearest(times: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the index of the sample of sorted ``times`` nearest each ``x``, and the offset."""
    i = np.clip(np.searchsorted(times, x), 1, len(times) - 1) if len(times) > 1 \
        else np.zeros(len(x), dtype=int)
    if len(times) > 1:
        left = times[i - 1]
        right = times[i]
        i = np.where(np.abs(x - left) <= np.abs(right - x), i - 1, i)
    return i, times[i] - x


def _grid(tol: dict) -> np.ndarray:
    """Return the lag search grid."""
    start, stop, step = (float(v) for v in tol['lag_grid_s'])
    return start + step * np.arange(int(round((stop - start) / step)) + 1)


def _midpoint_lag(tol: dict) -> float:
    """Return the middle of the allowed lag range, rounded to the grid."""
    start, _, step = (float(v) for v in tol['lag_grid_s'])
    mid = 0.5 * (float(tol['lag_min_s']) + float(tol['lag_max_s']))
    return start + step * round((mid - start) / step)


def _plateau(grid: np.ndarray, curve: np.ndarray) -> Optional[Tuple[float, bool]]:
    """
    Return the middle of the minimum plateau of ``curve`` and whether it touches the grid ends.

    None when the curve has no finite value.
    """
    finite = np.isfinite(curve)
    if not np.any(finite):
        return None
    best = int(np.nanargmin(np.where(finite, curve, np.inf)))
    low = high = best
    while low > 0 and finite[low - 1] and curve[low - 1] <= curve[best] + TIE_TOL:
        low -= 1
    while high < len(grid) - 1 and finite[high + 1] and curve[high + 1] <= curve[best] + TIE_TOL:
        high += 1
    return float(np.mean(grid[low:high + 1])), (low == 0 or high == len(grid) - 1)


def _stats(values: np.ndarray) -> dict:
    """RMS, 99th percentile and maximum of ``values`` (None when empty)."""
    if len(values) == 0:
        return {'n': 0, 'rms': None, 'p99': None, 'max': None}
    return {'n': int(len(values)), 'rms': float(np.sqrt(np.mean(values ** 2))),
            'p99': float(np.percentile(values, 99)), 'max': float(np.max(values))}


def _half_state_period(ep: Episode) -> float:
    """Return half the nominal state period."""
    return 0.5 * period(ep.profile['streams']['state'])


# --------------------------------------------------------------------------------------------
# Arm
# --------------------------------------------------------------------------------------------

def _arm(ep: Episode, tol: dict) -> dict:
    """Compare the target stream with the measured tool pose."""
    out: dict = {'verdict': 'NOT_JUDGED', 'reasons': []}
    state = ep.streams['state']
    target = ep.get('target')
    if target is None or len(target) == 0:
        out['reason'] = 'no target rows'
        return out
    for key in ('measured_tool_pose', 'measured_tool_pose_valid', 'damped', 'active_limit'):
        if key not in state.fields:
            out['reason'] = f'state field {key} missing'
            return out
    half = _half_state_period(ep)
    fails = out['reasons']

    expected = metadata.lookup(ep.sidecar, 'frames.base') or ep.profile['frames']['base']
    frames = set(target.frame_id) if target.frame_id else set()
    out['frame_expected'] = expected
    out['frame_ids'] = sorted(frames)
    out['frame_ok'] = frames == {expected}
    if not out['frame_ok']:
        fails.append(f'target frame {sorted(frames) or "not recorded"} is not the base frame '
                     f'{expected}')

    order = np.argsort(state.stamp, kind='stable')
    s_stamp = state.stamp[order]
    damped = np.asarray(state.fields['damped'], dtype=bool)[order]
    limit = np.asarray(state.fields['active_limit'])[order]
    valid = np.asarray(state.fields['measured_tool_pose_valid'], dtype=bool)[order]
    pose = np.asarray(state.fields['measured_tool_pose'], dtype=float)[order]
    m_stamp = s_stamp[valid]
    m_pose = pose[valid]

    t_stamp = np.asarray(target.stamp, dtype=float)
    t_pose = np.asarray(target.fields['pose'], dtype=float)
    if len(m_stamp) == 0:
        out['reason'] = 'no valid measured tool pose'
        out['verdict'] = 'FAIL' if fails else 'NOT_JUDGED'
        return out
    # The state row of the same cycle decides whether a sample is limited.
    row, off = nearest(s_stamp, t_stamp)
    has_row = np.abs(off) <= half
    limited = has_row & (damped[row] | (limit[row] != LIMIT_NONE))
    clean = has_row & ~limited
    out['n_target'] = int(len(t_stamp))
    out['n_no_state_row'] = int(np.sum(~has_row))

    steps = np.linalg.norm(np.diff(t_pose[:, :3], axis=0), axis=1)
    near = np.diff(t_stamp) <= PATH_MAX_STEP_PERIODS * 2.0 * half
    out['path_m'] = float(np.sum(steps[near])) if len(steps) else 0.0
    grid = _grid(tol)
    identifiable = out['path_m'] >= float(tol['min_path_m'])
    out['lag_identifiable'] = bool(identifiable)

    def at(lag):
        """Residuals of every target sample against the measured sample nearest t + lag."""
        j, dt = nearest(m_stamp, t_stamp + lag)
        ok = np.abs(dt) <= half
        return ok, np.linalg.norm(m_pose[j, :3] - t_pose[:, :3], axis=1), j

    curve = np.full(len(grid), np.nan)
    for k, lag in enumerate(grid):
        ok, dp, _ = at(lag)
        use = ok & clean
        if np.any(use):
            curve[k] = float(np.sqrt(np.mean(dp[use] ** 2)))
    out['min_rms_curve_m'] = float(np.nanmin(curve)) if np.any(np.isfinite(curve)) else None

    lag = _midpoint_lag(tol)
    if identifiable:
        fit = _plateau(grid, curve)
        if fit is None:
            out['reason'] = 'no clean sample matches a measured pose'
            out['verdict'] = 'FAIL' if fails else 'NOT_JUDGED'
            return out
        lag, at_edge = fit
        out['lag_at_grid_edge'] = bool(at_edge)
    out['lag_s'] = float(lag)

    ok, dp, j = at(lag)
    use_clean = ok & clean
    use_limited = ok & limited
    drot = np.degrees(quat_angle_between(t_pose[:, 3:7], m_pose[j, 3:7]))
    out['n_clean'] = int(np.sum(use_clean))
    out['n_limited'] = int(np.sum(use_limited))
    stat = _stats(dp[use_clean])
    out.update({'rms_m': stat['rms'], 'p99_m': stat['p99'], 'max_m': stat['max'],
                'rot_p99_deg': float(np.percentile(drot[use_clean], 99))
                if out['n_clean'] else None,
                'rot_max_deg': float(np.max(drot[use_clean])) if out['n_clean'] else None})
    lim = _stats(dp[use_limited])
    out['limited'] = {'n': lim['n'], 'rms_m': lim['rms'], 'p99_m': lim['p99'],
                      'max_m': lim['max'],
                      'rot_p99_deg': float(np.percentile(drot[use_limited], 99))
                      if lim['n'] else None}
    ok0, dp0, _ = at(0.0)
    out['rms_lag0_m'] = _stats(dp0[ok0 & clean])['rms']

    if out['n_clean'] < MIN_CLEAN_ARM:
        out['reason'] = f'only {out["n_clean"]} clean samples (need {MIN_CLEAN_ARM})'
        out['verdict'] = 'FAIL' if fails else 'NOT_JUDGED'
        return out
    if identifiable:
        low, high = float(tol['lag_min_s']), float(tol['lag_max_s'])
        if not low <= lag <= high:
            fails.append(f'lag {lag * 1e3:.0f} ms outside [{low * 1e3:.0f}, {high * 1e3:.0f}] ms')
        elif out['lag_at_grid_edge']:
            fails.append('lag fit reached the edge of the search grid')
    if out['rms_m'] > float(tol['arm_rms_m']):
        fails.append(f'translation RMS {out["rms_m"] * 1e3:.1f} mm exceeds '
                     f'{float(tol["arm_rms_m"]) * 1e3:.1f} mm')
    if out['p99_m'] > float(tol['arm_p99_m']):
        fails.append(f'translation p99 {out["p99_m"] * 1e3:.1f} mm exceeds '
                     f'{float(tol["arm_p99_m"]) * 1e3:.1f} mm')
    if out['rot_p99_deg'] > float(tol['arm_rot_p99_deg']):
        fails.append(f'rotation p99 {out["rot_p99_deg"]:.2f} deg exceeds '
                     f'{float(tol["arm_rot_p99_deg"]):.2f} deg')
    out['verdict'] = 'FAIL' if fails else 'PASS'
    return out


# --------------------------------------------------------------------------------------------
# Gripper
# --------------------------------------------------------------------------------------------

def closed_direction(position: np.ndarray, request: np.ndarray) -> Optional[int]:
    """
    Return +1 when larger positions mean closed, -1 when smaller do, None when unknown.

    Inferred from the request field (CLOSE = 2, OPEN = 1) of the command rows: the mean
    position of the CLOSE rows against the OPEN rows, or against all rows when only one of the
    two requests occurs.
    """
    closing = position[request == REQUEST_CLOSE]
    opening = position[request == REQUEST_OPEN]
    if len(closing) and len(opening):
        diff = float(np.mean(closing) - np.mean(opening))
    elif len(closing):
        diff = float(np.mean(closing) - np.mean(position))
    elif len(opening):
        diff = float(np.mean(position) - np.mean(opening))
    else:
        return None
    if abs(diff) < 1e-9:
        return None
    return 1 if diff > 0 else -1


def _gripper(ep: Episode, tol: dict) -> dict:
    """Compare the gripper command with the measured opening."""
    tol = dict(tol, lag_min_s=tol.get('gripper_lag_min_s', tol['lag_min_s']),
               lag_max_s=tol.get('gripper_lag_max_s', tol['lag_max_s']))
    out: dict = {'verdict': 'NOT_JUDGED', 'reasons': [], 'applicable': False}
    command = ep.get('gripper_command')
    if command is None or len(command) == 0:
        out['reason'] = 'no gripper command rows'
        return out
    out['applicable'] = True
    state = ep.streams['state']
    for key in ('gripper_measured', 'gripper_measured_valid', 'gripper_measured_stamp'):
        if key not in state.fields:
            out['reason'] = f'state field {key} missing'
            return out
    half = _half_state_period(ep)
    fails = out['reasons']
    valid = np.asarray(state.fields['gripper_measured_valid'], dtype=bool)
    m_time = np.asarray(state.fields['gripper_measured_stamp'], dtype=float)
    m_time = np.where(m_time > 0.0, m_time, state.stamp)
    m_value = np.asarray(state.fields['gripper_measured'], dtype=float)
    keep = valid & np.isfinite(m_value)
    order = np.argsort(m_time[keep], kind='stable')
    m_time = m_time[keep][order]
    m_value = m_value[keep][order]
    if len(m_time) < 2:
        out['reason'] = 'no valid measured gripper opening'
        return out
    c_time = np.asarray(command.stamp, dtype=float)
    c_pos = np.asarray(command.fields['position'], dtype=float)
    c_req = np.asarray(command.fields.get('request', np.zeros(len(command))))
    sign = closed_direction(c_pos, c_req)
    out['closed_direction'] = sign
    out['travel_rad'] = float(np.sum(np.abs(np.diff(c_pos))))
    identifiable = out['travel_rad'] >= float(tol['gripper_min_travel_rad'])
    out['lag_identifiable'] = bool(identifiable)

    # Stationary measured samples: the opening moved less than the threshold over the window.
    hi, hi_off = nearest(m_time, m_time + 0.5 * GRASP_WINDOW_S)
    lo, lo_off = nearest(m_time, m_time - 0.5 * GRASP_WINDOW_S)
    covered = (np.abs(hi_off) <= half) & (np.abs(lo_off) <= half)
    stationary = covered & (np.abs(m_value[hi] - m_value[lo]) < GRASP_STATIONARY_RAD)
    abs_rad = float(tol['gripper_abs_rad'])

    def at(lag):
        """Return (matched, residual, grasp) for every command row at ``lag``."""
        j, dt = nearest(m_time, c_time + lag)
        ok = np.abs(dt) <= half
        residual = np.abs(m_value[j] - c_pos)
        if sign is None:
            grasp = np.zeros(len(c_pos), dtype=bool)
        else:
            grasp = ok & stationary[j] & (sign * (c_pos - m_value[j]) > abs_rad)
        return ok, residual, grasp

    grid = _grid(tol)
    curve = np.full(len(grid), np.nan)
    for k, lag in enumerate(grid):
        ok, res, grasp = at(lag)
        use = ok & ~grasp
        if np.any(use):
            curve[k] = float(np.sqrt(np.mean(res[use] ** 2)))
    lag = _midpoint_lag(tol)
    if identifiable:
        fit = _plateau(grid, curve)
        if fit is None:
            out['reason'] = 'no command row matches a measured opening'
            return out
        lag = fit[0]
    out['lag_s'] = float(lag)
    ok, res, grasp = at(lag)
    use = ok & ~grasp
    stat = _stats(res[use])
    out.update({'n_clean': stat['n'], 'n_grasp': int(np.sum(ok & grasp)),
                'rms_rad': stat['rms'], 'p99_rad': stat['p99'], 'max_rad': stat['max']})
    if stat['n'] < MIN_CLEAN_GRIPPER:
        out['reason'] = f'only {stat["n"]} clean samples (need {MIN_CLEAN_GRIPPER})'
        return out
    if identifiable:
        low, high = float(tol['lag_min_s']), float(tol['lag_max_s'])
        if not low <= lag <= high:
            fails.append(f'lag {lag * 1e3:.0f} ms outside [{low * 1e3:.0f}, {high * 1e3:.0f}] ms')
    if stat['p99'] > abs_rad:
        fails.append(f'|residual| p99 {stat["p99"]:.3f} rad exceeds {abs_rad:.3f} rad')
    out['verdict'] = 'FAIL' if fails else 'PASS'
    return out


# The tolerances of checks.yaml's consistency section that judge an episode (the lag range
# and the identifiability thresholds are physical plausibility, not fitted per deployment).
TOLERANCE_KEYS = PROFILE_TOLERANCES['consistency']


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------

def check(ep: Episode, checks: dict) -> dict:
    """
    Return the consistency report: ``verdict`` (PASS, FAIL, NOT_JUDGED), ``arm``, ``gripper``.

    ``reasons`` lists why the verdict is FAIL. A part that cannot be judged (too few clean
    samples, no gripper) is NOT_JUDGED and never fails the episode.
    """
    tol = checks['consistency']
    arm = _arm(ep, tol)
    gripper = _gripper(ep, tol)
    parts = [('arm', arm)]
    if gripper['applicable']:
        parts.append(('gripper', gripper))
    reasons = [f'{name}: {r}' for name, part in parts for r in part['reasons']]
    verdicts = [part['verdict'] for _, part in parts]
    if 'FAIL' in verdicts:
        verdict = 'FAIL'
    elif all(v == 'PASS' for v in verdicts):
        verdict = 'PASS'
    else:
        verdict = 'NOT_JUDGED'
    provisional = (checks.get('_tolerances') or {}).get('provisional') or {}
    limits = {key: tol.get(key) for key in TOLERANCE_KEYS}
    return {'verdict': verdict, 'reasons': reasons, 'arm': arm, 'gripper': gripper,
            'limits': limits,
            'provisional': sorted(k.partition('.')[2] for k in provisional
                                  if k.startswith('consistency.')
                                  and k.partition('.')[2] in TOLERANCE_KEYS)}
