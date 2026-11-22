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
Measures over recorded samples: periods, stale repeats, acceleration, oscillation, lead.

Pure functions on plain lists; no ROS imports. Times are seconds (floats).
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

ARM_JOINTS = tuple(f'xarm_joint{i}' for i in range(1, 7))
GRIPPER_JOINT = 'xarm_drive_joint'


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Return percentile ``q`` (0..100, linear interpolation), None when empty."""
    if not len(values):
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def median(values: Sequence[float]) -> Optional[float]:
    """Return the median or None when empty."""
    return percentile(values, 50.0)


def classify_joint_state(names, positions, velocities=(), efforts=()):
    """
    Sort one JointState message into the arm, the gripper proxy or neither.

    Returns ``('arm', pos6, vel6_or_None, effort_ok)``, ``('gripper', None, None, None)`` or
    ``('other', None, None, None)``. Only a message carrying all six arm joints is ``arm``;
    ``pos6`` is in joint order 1..6 whatever the order in the message. ``vel6`` is None unless
    the message has a velocity array of the length of ``names`` with finite arm entries.
    ``effort_ok`` is True when the effort array has the length of ``names`` and finite arm
    entries.
    """
    names = list(names)
    index = {}
    for i, name in enumerate(names):
        index.setdefault(name, i)
    if all(j in index for j in ARM_JOINTS):
        if len(positions) != len(names):
            return ('other', None, None, None)
        order = [index[j] for j in ARM_JOINTS]
        pos = tuple(float(positions[i]) for i in order)
        vel = None
        if len(velocities) == len(names) and all(math.isfinite(velocities[i]) for i in order):
            vel = tuple(float(velocities[i]) for i in order)
        eff_ok = (len(efforts) == len(names)
                  and all(math.isfinite(efforts[i]) for i in order))
        return ('arm', pos, vel, eff_ok)
    if GRIPPER_JOINT in index:
        return ('gripper', None, None, None)
    return ('other', None, None, None)


def period_distribution(times: Sequence[float], nominal: Optional[float]) -> dict:
    """
    Return the period statistics of an ascending list of times.

    Keys: count, rate_hz, min, p50, p90, p99, max (periods, s), over_2x (periods longer than
    twice ``nominal``), largest_gap (s). All None (count 0 or 1) when there are fewer than two
    times; ``over_2x`` is None when ``nominal`` is None.
    """
    out = {'count': len(times), 'rate_hz': None, 'min': None, 'p50': None, 'p90': None,
           'p99': None, 'max': None, 'over_2x': None, 'largest_gap': None}
    if len(times) < 2:
        return out
    periods = np.diff(np.asarray(times, dtype=float))
    span = float(times[-1] - times[0])
    out['rate_hz'] = (len(times) - 1) / span if span > 0 else None
    out['min'] = float(periods.min())
    out['p50'] = percentile(periods, 50.0)
    out['p90'] = percentile(periods, 90.0)
    out['p99'] = percentile(periods, 99.0)
    out['max'] = float(periods.max())
    out['largest_gap'] = out['max']
    if nominal:
        out['over_2x'] = int((periods > 2.0 * nominal).sum())
    return out


def stamp_offset(arrival: Sequence[float], stamps: Sequence[float]) -> Optional[float]:
    """Return the median of (local arrival time - header stamp), seconds."""
    return median([a - s for a, s in zip(arrival, stamps)])


# ---- stale repeats while moving -------------------------------------------------------

def stale_analysis(t: Sequence[float], pos: Sequence[Sequence[float]],
                   moving_speed: float, min_span: float) -> dict:
    """
    Find messages that repeat the previous positions exactly while the arm moves.

    A message is "moving" when the speed between the latest message whose positions differ
    from their predecessor's (at or before it) and the next such message exceeds
    ``moving_speed`` (rad/s) on any joint. Only moving spans of at least ``min_span`` (s)
    count. A "stale repeat" is a counted message whose positions equal the previous message's.

    Returns: judged (a span was seen), moving_messages, moving_seconds, stale_count,
    longest_stale_messages, longest_stale_seconds, value_change_fraction (changed messages
    over moving messages), value_change_rate_hz, message_rate_hz (both while moving).
    """
    n = len(pos)
    out = {'judged': False, 'moving_messages': 0, 'moving_seconds': 0.0, 'stale_count': 0,
           'longest_stale_messages': 0, 'longest_stale_seconds': 0.0,
           'value_change_fraction': None, 'value_change_rate_hz': None,
           'message_rate_hz': None}
    if n < 3:
        return out
    pos = [tuple(p) for p in pos]
    changed = [True] + [pos[i] != pos[i - 1] for i in range(1, n)]
    prev_d = [0] * n
    for i in range(n):
        prev_d[i] = i if changed[i] else prev_d[i - 1]
    next_d = [None] * n
    nxt = None
    for i in range(n - 1, -1, -1):
        next_d[i] = nxt
        if changed[i]:
            nxt = i
    moving = [False] * n
    for i in range(n):
        a, b = prev_d[i], next_d[i]
        if b is None:
            continue
        dt = t[b] - t[a]
        if dt <= 0:
            continue
        speed = max(abs(x - y) for x, y in zip(pos[b], pos[a])) / dt
        moving[i] = speed > moving_speed
    counted = [False] * n
    spans = 0
    seconds = 0.0
    i = 0
    while i < n:
        if not moving[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and moving[j + 1]:
            j += 1
        if t[j] - t[i] >= min_span:
            spans += 1
            seconds += t[j] - t[i]
            for k in range(i, j + 1):
                counted[k] = True
        i = j + 1
    if not spans:
        return out
    stale = [counted[i] and i > 0 and not changed[i] for i in range(n)]
    moving_messages = sum(counted)
    changed_moving = sum(1 for i in range(n) if counted[i] and changed[i])
    longest, longest_s = 0, 0.0
    i = 0
    while i < n:
        if not stale[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and stale[j + 1]:
            j += 1
        if j - i + 1 > longest:
            longest, longest_s = j - i + 1, t[j] - t[i - 1]
        i = j + 1
    out.update({
        'judged': True, 'moving_messages': moving_messages, 'moving_seconds': seconds,
        'stale_count': sum(stale), 'longest_stale_messages': longest,
        'longest_stale_seconds': longest_s,
        'value_change_fraction': changed_moving / moving_messages,
        'value_change_rate_hz': changed_moving / seconds if seconds > 0 else None,
        'message_rate_hz': moving_messages / seconds if seconds > 0 else None})
    return out


# ---- velocity, acceleration, oscillation ----------------------------------------------

def joint_velocities(t: Sequence[float], pos: Sequence[Sequence[float]],
                     reported: Optional[Sequence[Optional[Sequence[float]]]] = None):
    """
    Return (n x 6 array of velocities, source).

    ``source`` is 'reported' when every message carries a finite velocity array, otherwise
    'position' (central differences of positions between neighbouring messages by stamp;
    NaN at the ends and where the stamps do not advance).
    """
    n = len(pos)
    if n and reported is not None and all(r is not None for r in reported):
        return np.asarray(reported, dtype=float), 'reported'
    v = np.full((n, 6), np.nan)
    if n >= 3:
        p = np.asarray(pos, dtype=float)
        tt = np.asarray(t, dtype=float)
        dt = tt[2:] - tt[:-2]
        ok = dt > 0
        dv = (p[2:] - p[:-2])
        with np.errstate(divide='ignore', invalid='ignore'):
            vel = dv / dt[:, None]
        vel[~ok] = np.nan
        v[1:-1] = vel
    return v, 'position'


def peak_acceleration(t: Sequence[float], vel: np.ndarray) -> List[Optional[float]]:
    """
    Return the peak |acceleration| per joint (rad/s^2).

    Central difference of velocity over a 5-sample window, stamp-differenced; None for a joint
    with no usable window.
    """
    n = len(vel)
    peaks: List[Optional[float]] = [None] * 6
    if n < 5:
        return peaks
    tt = np.asarray(t, dtype=float)
    dt = tt[4:] - tt[:-4]
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = (vel[4:] - vel[:-4]) / dt[:, None]
    acc[dt <= 0] = np.nan
    for j in range(6):
        col = np.abs(acc[:, j])
        col = col[np.isfinite(col)]
        if len(col):
            peaks[j] = float(col.max())
    return peaks


def sign_changes_per_window(t: Sequence[float], v: Sequence[float], min_speed: float,
                            window: float) -> int:
    """Return the most velocity sign changes (|v| above ``min_speed``) in any ``window`` s."""
    events = []
    last = 0
    for ti, vi in zip(t, v):
        if not math.isfinite(vi) or abs(vi) <= min_speed:
            continue
        sign = 1 if vi > 0 else -1
        if last and sign != last:
            events.append(ti)
        last = sign
    best = 0
    lo = 0
    for hi in range(len(events)):
        while events[hi] - events[lo] >= window:
            lo += 1
        best = max(best, hi - lo + 1)
    return best


# ---- teleop lead and lag ---------------------------------------------------------------

def nearest_match(times: Sequence[float], target_times: Sequence[float], tol: float):
    """
    Return, per entry of ``times``, the index of the nearest entry of ``target_times``.

    None where none lies within ``tol`` seconds. ``target_times`` must be ascending.
    """
    out = []
    tt = np.asarray(target_times, dtype=float)
    for x in times:
        if not len(tt):
            out.append(None)
            continue
        k = int(np.searchsorted(tt, x))
        best = None
        for c in (k - 1, k):
            if 0 <= c < len(tt) and abs(tt[c] - x) <= tol and (
                    best is None or abs(tt[c] - x) < abs(tt[best] - x)):
                best = c
        out.append(best)
    return out


def lead_stats(state_t: Sequence[float], measured: Sequence[Tuple[float, float, float]],
               target_t: Sequence[float], target: Sequence[Tuple[float, float, float]],
               tol: float) -> dict:
    """
    Return the translation lead (m) between target and measured tool pose, stamp-matched.

    ``measured`` holds the valid measured positions at ``state_t``; stamps match within
    ``tol`` seconds. Keys: count (matched
    pairs), p50, p99, max (None when no pair).
    """
    idx = nearest_match(state_t, target_t, tol)
    leads = []
    for m, k in zip(measured, idx):
        if k is None:
            continue
        leads.append(math.dist(m, target[k]))
    return {'count': len(leads), 'p50': percentile(leads, 50.0),
            'p99': percentile(leads, 99.0), 'max': max(leads) if leads else None}


def lag_estimate(state_t: Sequence[float], measured: Sequence[Tuple[float, float, float]],
                 target_t: Sequence[float], target: Sequence[Tuple[float, float, float]],
                 max_lag: float = 1.0, step: float = 0.02,
                 min_samples: int = 25) -> Optional[float]:
    """
    Estimate how long the measured pose lags the target (s; positive: measured is later).

    Cross-correlation of the demeaned x, y and z over lags in [-max_lag, max_lag]; the target
    is compared with the measured pose interpolated at (target time + lag). None when there
    are too few samples or no motion. Report-only: it is a rough estimate.
    """
    if len(target_t) < min_samples or len(state_t) < 2:
        return None
    st = np.asarray(state_t, dtype=float)
    tt = np.asarray(target_t, dtype=float)
    m = np.asarray(measured, dtype=float)
    g = np.asarray(target, dtype=float)
    best, best_score = None, -math.inf
    for lag in np.arange(-max_lag, max_lag + step / 2, step):
        q = tt + lag
        inside = (q >= st[0]) & (q <= st[-1])
        if inside.sum() < min_samples:
            continue
        score = 0.0
        for axis in range(3):
            a = g[inside, axis]
            b = np.interp(q[inside], st, m[:, axis])
            a = a - a.mean()
            b = b - b.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom > 0:
                score += float(a @ b) / denom
        if score > best_score:
            best, best_score = float(lag), score
    return best if best_score > 0 else None
