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
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ranger_xarm_teleop_servo.tracking import resample, RESAMPLE_S

ARM_JOINTS = tuple(f'xarm_joint{i}' for i in range(1, 7))
GRIPPER_JOINT = 'xarm_drive_joint'

# "Still" for the reported-velocity noise (still_windows): no nonzero joint-jog command for
# STILL_SPAN_S before and during, and the joint position moving less than STILL_MOVE_RAD over
# each STILL_SPAN_S span; a window counts from STILL_MIN_S; a gap between two samples longer
# than STILL_MAX_GAP_S ends it (a lost stretch is not still time).
STILL_SPAN_S = 0.5
STILL_MOVE_RAD = 0.002
STILL_MIN_S = 1.0
STILL_MAX_GAP_S = 0.1


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
        arr = np.asarray(reported, dtype=float)
        moved = n >= 2 and float(np.max(np.ptp(np.asarray(pos, dtype=float), axis=0))) > 1e-4
        # A backend that reports zero velocity while the joints move (mock components do)
        # reports nothing: the positions are differenced instead.
        if not (moved and np.all(np.abs(arr) < 1e-9)):
            return arr, 'reported'
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
            'p99': percentile(leads, 99.0), 'max': max(leads) if leads else None,
            'values': leads}


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


# ---- the joint states on a command-period grid ------------------------------------------

def resampled_peak_acceleration(t: Sequence[float], pos: Sequence[Sequence[float]],
                                step: float = RESAMPLE_S) -> List[Optional[float]]:
    """
    Return the peak |acceleration| per joint (rad/s^2) of positions resampled to ``step`` s.

    Linear interpolation onto a uniform grid, then the second difference (the servo package's
    ``resample``); None for every joint when fewer than three grid samples remain. The window
    is one command period, so a ramp at the limit reads the limit.
    """
    if len(pos) < 3:
        return [None] * 6
    tt = np.asarray(t, dtype=float)
    times, positions = resample(tt - tt[0], np.asarray(pos, dtype=float), step)
    if len(times) < 3:
        return [None] * 6
    velocity = np.diff(positions, axis=0) / np.diff(times)[:, None]
    middle = (times[1:] + times[:-1]) / 2.0
    acc = np.abs(np.diff(velocity, axis=0) / np.diff(middle)[:, None])
    return [float(acc[:, j].max()) if np.isfinite(acc[:, j]).any() else None
            for j in range(6)]


# ---- MoveIt Servo's output: the position path of the trajectory messages -----------------

# Points whose times differ by less than this are one point (Servo re-sends the same states in
# later windows).
SERVO_MERGE_S = 1e-3
# A gap longer than this ends a run: the controller holds its last point, so a run starts and
# ends at rest.
SERVO_RUN_GAP_S = 0.1
# A segment is moving above this joint speed on any joint (rad/s).
SERVO_REST_SPEED = 1e-3


def command_points(messages) -> dict:
    """
    Return the arm joints' position path of JointTrajectory messages, merged and sorted.

    ``messages`` holds ``(stamp_ns, names, points)``: the header stamp in integer nanoseconds,
    the joint names, and ``[(time_from_start_s, positions), ...]``. A point's time is the
    stamp plus its ``time_from_start``; positions are taken by joint name (the arm joints, in
    order) and a message without all six, or with positions of another length, is skipped.
    Points less than ``SERVO_MERGE_S`` apart are merged (the first is kept).

    Returns ``{'origin_ns', 't', 'p', 'skipped'}``: ``t`` (n,) in seconds from ``origin_ns``
    (the earliest stamp), ``p`` (n, 6), and the number of messages skipped.
    """
    usable = []
    skipped = 0
    for stamp_ns, names, points in messages:
        index = {}
        for i, name in enumerate(names):
            index.setdefault(name, i)
        if not all(j in index for j in ARM_JOINTS) or not points:
            skipped += 1
            continue
        order = [index[j] for j in ARM_JOINTS]
        rows = []
        for tfs, positions in points:
            if len(positions) != len(names):
                rows = None
                break
            rows.append((float(tfs), [float(positions[i]) for i in order]))
        if rows is None:
            skipped += 1
            continue
        usable.append((int(stamp_ns), rows))
    if not usable:
        return {'origin_ns': None, 't': np.zeros(0), 'p': np.zeros((0, 6)), 'skipped': skipped}
    origin = min(u[0] for u in usable)
    times, positions = [], []
    for stamp_ns, rows in usable:
        for tfs, pos in rows:
            times.append((stamp_ns - origin) * 1e-9 + tfs)
            positions.append(pos)
    order = np.argsort(np.asarray(times), kind='stable')
    times = np.asarray(times)[order]
    positions = np.asarray(positions)[order]
    keep = [0]
    for i in range(1, len(times)):
        if times[i] - times[keep[-1]] >= SERVO_MERGE_S:
            keep.append(i)
    return {'origin_ns': origin, 't': times[keep], 'p': positions[keep], 'skipped': skipped}


def split_runs(t: Sequence[float], gap: float = SERVO_RUN_GAP_S) -> List[Tuple[int, int]]:
    """Return ``(first, stop)`` index ranges of ascending times split at gaps over ``gap``."""
    if not len(t):
        return []
    cuts = [i + 1 for i in range(len(t) - 1) if t[i + 1] - t[i] > gap]
    bounds = [0] + cuts + [len(t)]
    return list(zip(bounds[:-1], bounds[1:]))


def capture_edges(first_arrival: Optional[float], last_arrival: Optional[float],
                  began: Optional[float], ended: Optional[float],
                  gap: float = SERVO_RUN_GAP_S) -> Tuple[bool, bool]:
    """
    Return ``(open_start, open_end)``: whether the capture cut the stream at either end.

    The stream starts open when its first message arrived within ``gap`` of the capture
    beginning, and ends open when its last arrived within ``gap`` of the capture end: there
    the arm may have been moving, so "at rest" cannot be assumed. Arrivals and the capture
    times are on one clock.
    """
    if None in (first_arrival, last_arrival, began, ended):
        return False, False
    return first_arrival - began < gap, ended - last_arrival < gap


def command_analysis(t: Sequence[float], p: Sequence[Sequence[float]],
                     open_start: bool = False, open_end: bool = False,
                     gap: float = SERVO_RUN_GAP_S) -> dict:
    """
    Return the segment speeds and the accelerations of a position path (Servo's output).

    Segment velocity is dp / dt. The acceleration at a point is the difference of the
    neighbouring segment velocities over the mean of the two segment lengths, per joint. Runs
    (``split_runs``) start and end at rest: a virtual zero-velocity segment, as long as the
    real one beside it, is added at each end, except at the start of the first run when
    ``open_start`` and at the end of the last run when ``open_end`` (the capture cut the
    stream there; nothing is assumed, and the point at that end gets no acceleration).

    Returns ``{'runs', 'speed', 'acc', 'spacing', 'ends_moving'}``: ``speed`` (n,) the
    largest joint speed of the segment starting at each point (NaN for the last point of a
    run), ``acc`` (n, 6) in rad/s^2 (NaN where undefined), ``spacing`` the gaps between
    consecutive points of a run, ``ends_moving`` whether the final segment of an open-ended
    stream is moving.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float).reshape(len(t), 6)
    n = len(t)
    runs = split_runs(t, gap)
    speed = np.full(n, np.nan)
    acc = np.full((n, 6), np.nan)
    spacing: List[float] = []
    ends_moving = False
    for k, (a, b) in enumerate(runs):
        if b - a < 2:
            continue
        dt = np.diff(t[a:b])
        spacing.extend(float(x) for x in dt)
        v = np.diff(p[a:b], axis=0) / dt[:, None]
        speed[a:b - 1] = np.abs(v).max(axis=1)
        rows, lengths, first = [], [], a
        lead = not (k == 0 and open_start)
        tail = not (k == len(runs) - 1 and open_end)
        if lead:
            rows.append(np.zeros(6))
            lengths.append(dt[0])
        else:
            first = a + 1
        rows.extend(v)
        lengths.extend(dt)
        if tail:
            rows.append(np.zeros(6))
            lengths.append(dt[-1])
        else:
            ends_moving = bool(speed[b - 2] > SERVO_REST_SPEED)
        vel, length = np.asarray(rows), np.asarray(lengths)
        change = (vel[1:] - vel[:-1]) / ((length[1:] + length[:-1]) / 2.0)[:, None]
        acc[first:first + len(change)] = change
    return {'runs': runs, 'speed': speed, 'acc': acc, 'spacing': spacing,
            'ends_moving': ends_moving}


# MoveIt Servo integrates each command over its fixed publish period: a step of its output,
# divided by the period, is the joint-jog command it used, to floating-point rounding. A step
# within this of a command (rad/s, every joint) was made from it.
JOG_MATCH_RAD_S = 1e-6
# How far past the last matched command a step is looked for. Servo uses one command per cycle;
# a skip jumps one (rarely more).
JOG_SEARCH = 5
# How far past the last command matched in an earlier run the first step of a run is looked
# for (about a second of commands): one wrong match cannot carry the rest of the capture away.
JOG_START_SEARCH = 50


def jog_commands(messages) -> List[np.ndarray]:
    """
    Return the arm joints' joint-jog commands, in order.

    ``messages`` holds ``(names, velocities)`` of the teleop node's JointJog commands (rad/s);
    one without all six arm joints is left out. Each is returned as velocities (6,).
    """
    out = []
    for names, velocities in messages:
        index = {}
        for i, name in enumerate(names):
            index.setdefault(name, i)
        if not all(j in index for j in ARM_JOINTS) or len(velocities) != len(names):
            continue
        out.append(np.array([float(velocities[index[j]]) for j in ARM_JOINTS]))
    return out


def servo_skips(p: Sequence[Sequence[float]], runs: Sequence[Tuple[int, int]],
                commands: Sequence[np.ndarray], period: float,
                step_limit: float = math.inf) -> dict:
    """
    Return which commands MoveIt Servo skipped or used twice, from its output and the commands.

    Each step of Servo's output (point i to i + 1 of a run, over ``period``) is matched by value
    to the teleop node's commands, in order: the first command after the last one matched,
    up to ``JOG_SEARCH`` ahead, that it equals; or the last one again (a reuse). A match more
    than one command ahead is a skip, which joins the steps of the commands it jumped into
    one: the point at the start of that step (its acceleration) is explained by it, but only
    when every command step it jumped is itself within ``step_limit`` (rad/s per command: the
    acceleration limit times the period). A skip inside a run of equal commands is invisible
    and leaves the matching one command behind, so a later change of the command itself
    can look like a skip; that check keeps a too-large command step from being explained. The
    first step of a run is looked for up to ``JOG_START_SEARCH`` commands past the last match;
    a step at rest there matches a command at rest (the node may send them while the hand is
    still), and one that matches nothing is passed over. Any other step that matches nothing
    (Servo scaled it, for a collision or a limit) is counted unmatched and taken as made from
    the next command.

    Returns ``{'steps', 'matched', 'reused', 'skips', 'skipped_commands', 'unmatched',
    'skip_points'}``; ``skip_points`` are point indices.
    """
    p = np.asarray(p, dtype=float).reshape(-1, 6)
    cmds = np.asarray(commands, dtype=float).reshape(-1, 6)
    out = {'steps': 0, 'matched': 0, 'reused': 0, 'skips': 0, 'skipped_commands': 0,
           'unmatched': 0, 'skip_points': []}

    def equal(k, v):
        return bool(np.max(np.abs(cmds[k] - v)) <= JOG_MATCH_RAD_S)

    cursor = -1
    for a, b in runs:
        last = None                      # the command matched last in this run
        for i in range(a, b - 1):
            v = (p[i + 1] - p[i]) / period
            out['steps'] += 1
            if last is None:
                found = next((k for k in range(cursor + 1,
                                               min(cursor + 1 + JOG_START_SEARCH, len(cmds)))
                              if equal(k, v)), None)
                if found is None and not np.any(np.abs(v) > JOG_MATCH_RAD_S):
                    continue             # at rest with no command to match: passed over
            else:
                found = next((k for k in range(last + 1, min(last + 1 + JOG_SEARCH, len(cmds)))
                              if equal(k, v)), None)
            if found is None:
                if last is not None and equal(last, v):
                    out['reused'] += 1
                    out['matched'] += 1
                else:
                    # Servo changed the step (scaled it): taken as made from the next command,
                    # so the following step is not read as a skip.
                    out['unmatched'] += 1
                    if last is not None:
                        last = cursor = min(last + 1, len(cmds) - 1)
                continue
            if last is not None and found > last + 1:
                out['skips'] += 1
                out['skipped_commands'] += found - last - 1
                jumped = np.abs(np.diff(cmds[last:found + 1], axis=0)).max()
                if jumped <= step_limit * (1.0 + 1e-6):
                    out['skip_points'].append(i)
            out['matched'] += 1
            last = cursor = found
    return out


def clutch_events(samples: Sequence[Tuple[int, bool]]) -> Tuple[List[int], List[int]]:
    """
    Return ``(releases, engages)``: stamps (ns) where the clutch went False or True.

    ``samples`` are ``(stamp_ns, clutch_engaged)`` in arrival order. A release is a message with
    the clutch off after one with it on, so it is seen at the first cycle after the release.
    """
    releases, engages = [], []
    for (_, before), (stamp, after) in zip(samples[:-1], samples[1:]):
        if before and not after:
            releases.append(stamp)
        elif after and not before:
            engages.append(stamp)
    return releases, engages


def stop_durations(t: Sequence[float], analysis: dict,
                   releases: Sequence[float], engages: Sequence[float],
                   open_end: bool = False, latency: float = 0.0) -> List[dict]:
    """
    Return, per release with motion, how long Servo's output took to stop.

    ``releases`` and ``engages`` are times (s, the clock of ``t``). MoveIt Servo stamps each
    point ``latency`` (its ``max_expected_latency``) after the cycle that made it, so a release
    at ``r`` falls at ``r + latency`` on the points' timeline; every release and engage is
    shifted by it. A release's window ends at the next engage. The stop lasts from the
    (shifted) release to the end of the last segment whose speed (any joint) is above
    ``SERVO_REST_SPEED`` (all runs are scanned: a stall of Servo's output inside a stop counts
    into it); 0 when nothing moved after the release. A release has motion when the
    segment into that point or any later segment in the window moves; the others are left out.
    ``unfinished`` is True when the stream ended (``open_end``) while the last segment still
    moved: the duration is then the time seen so far.

    Each entry: ``{'release_s', 'first_point_s', 'duration_s', 'unfinished'}``.
    """
    t = np.asarray(t, dtype=float)
    speed = analysis['speed']
    runs = analysis['runs']
    out = []
    for release in releases:
        end = min([e for e in engages if e > release], default=math.inf) + latency
        release = release + latency
        first = int(np.searchsorted(t, release, side='left'))
        if first >= len(t) or t[first] >= end:
            continue
        a = next(r[0] for r in runs if r[0] <= first < r[1])
        into = first > a and speed[first - 1] > SERVO_REST_SPEED
        last = None
        for j in range(first, len(t) - 1):
            if t[j] >= end:
                break
            if speed[j] > SERVO_REST_SPEED:      # NaN (the last point of a run) is not
                last = j
        if not into and last is None:
            continue
        duration = max(float(t[last + 1] - release), 0.0) if last is not None else 0.0
        # The stream ended while the arm still moved: the capture cut it at a moving segment,
        # either the last one in the window or the one into the final point.
        if last is not None:
            unfinished = open_end and last == len(t) - 2 and t[last + 1] < end
        else:
            unfinished = open_end and first == len(t) - 1
        unfinished = bool(unfinished)
        out.append({'release_s': float(release), 'first_point_s': float(t[first]),
                    'duration_s': duration, 'unfinished': unfinished})
    return out


# ---- reversals of the command, still-arm velocity noise, teleop log summaries --------------

def jog_series(messages, arrivals) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (t (n,), v (n, 6)): the arm joints' joint-jog velocities with their arrival times.

    ``messages`` holds ``(names, velocities)`` as for ``jog_commands`` and ``arrivals`` the
    arrival time (s) of each, in the same order; a message without all six arm joints is left
    out. Empty when the two do not have the same length (no times were kept).
    """
    if len(messages) != len(arrivals):
        return np.zeros(0), np.zeros((0, 6))
    times, rows = [], []
    for (names, velocities), arrived in zip(messages, arrivals):
        index = {}
        for i, name in enumerate(names):
            index.setdefault(name, i)
        if not all(j in index for j in ARM_JOINTS) or len(velocities) != len(names):
            continue
        times.append(float(arrived))
        rows.append([float(velocities[index[j]]) for j in ARM_JOINTS])
    return np.asarray(times, dtype=float), np.asarray(rows, dtype=float).reshape(-1, 6)


def command_velocity_series(messages, velocities) -> dict:
    """
    Return the arm joints' velocity of MoveIt Servo's output over time, and where it came from.

    ``messages`` holds ``(arrival, stamp_ns, names, points)`` as ``data['command']`` and
    ``velocities`` the velocities of each message's points, ``[(v, ...), ...]`` per message
    in the same order (empty tuples where a point has none). When every message with all
    six arm joints carries the first point's velocities (one per name), those are used, at the
    header stamp plus the point's ``time_from_start`` ('velocities'). Otherwise the velocity
    is the central difference of the merged position path (``command_points``, 'positions';
    NaN at the ends). Returns ``{'source', 't', 'v', 'messages'}``; ``source`` is None when
    there is no usable message.
    """
    out = {'source': None, 't': np.zeros(0), 'v': np.zeros((0, 6)), 'messages': len(messages)}
    usable, with_velocity = [], True
    paired = len(velocities) == len(messages)
    for k, (_arrival, stamp_ns, names, points) in enumerate(messages):
        index = {}
        for i, name in enumerate(names):
            index.setdefault(name, i)
        if not all(j in index for j in ARM_JOINTS) or not points:
            continue
        vel = velocities[k][0] if paired and len(velocities[k]) else ()
        if len(vel) != len(names):
            with_velocity = False
            usable.append((stamp_ns, names, points, None))
            continue
        usable.append((stamp_ns, names, points, [float(vel[index[j]]) for j in ARM_JOINTS]))
    if not usable:
        return out
    if with_velocity:
        pairs = sorted(((stamp * 1e-9 + float(points[0][0]), v)
                        for stamp, _n, points, v in usable), key=lambda x: x[0])
        out['source'] = 'velocities'
        out['t'] = np.asarray([p[0] for p in pairs])
        out['v'] = np.asarray([p[1] for p in pairs])
        return out
    path = command_points([(stamp, names, points) for stamp, names, points, _v in usable])
    if len(path['t']):
        out['source'] = 'positions'
        out['t'] = path['t']
        out['v'] = joint_velocities(path['t'], path['p'])[0]
    return out


def still_windows(t: Sequence[float], p: Sequence[float], command_t: Sequence[float],
                  span: float = STILL_SPAN_S, move: float = STILL_MOVE_RAD,
                  min_len: float = STILL_MIN_S, max_gap: float = STILL_MAX_GAP_S
                  ) -> List[Tuple[float, float]]:
    """
    Return the (start, end) times of the stretches where one joint is still.

    A sample at ``t[i]`` is still when no entry of ``command_t`` (the times of nonzero
    joint-jog commands) falls in ``[t[i] - span, t[i]]`` and the positions ``p`` there differ
    by less than ``move`` (largest minus smallest; at least two samples). A stretch is a run of
    still samples with no gap above ``max_gap``; it starts ``span`` before its first sample
    (that look-back was still too) and is kept when at least ``min_len`` long. A look-back with
    a gap above ``max_gap`` in it, or none of whose samples is near its start, is not still.
    """
    tt = np.asarray(t, dtype=float)
    pp = np.asarray(p, dtype=float)
    cmd = np.sort(np.asarray(command_t, dtype=float))
    n = len(tt)
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return []
    lows = np.searchsorted(tt, tt - span, side='left')
    first_cmd = np.searchsorted(cmd, tt - span, side='left')
    lost = np.concatenate([[0], np.cumsum(np.diff(tt) > max_gap)])   # gaps up to each sample
    for i in range(n):
        if tt[i] - tt[0] < span or i - lows[i] < 1:
            continue
        # The look-back must be covered by samples: no gap in it, and one near its start.
        if lost[i] != lost[lows[i]] or tt[lows[i]] - (tt[i] - span) > max_gap:
            continue
        if first_cmd[i] < len(cmd) and cmd[first_cmd[i]] <= tt[i]:
            continue
        seg = pp[lows[i]:i + 1]
        if np.all(np.isfinite(seg)) and float(seg.max() - seg.min()) < move:
            flags[i] = True
    windows = []
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1] and tt[j + 1] - tt[j] <= max_gap:
            j += 1
        if tt[j] - (tt[i] - span) >= min_len:
            windows.append((float(tt[i] - span), float(tt[j])))
        i = j + 1
    return windows


def still_noise(t: Sequence[float], v: Sequence[float], windows, min_speed: float,
                window: float) -> dict:
    """
    Return the reported velocity's own noise over still ``windows`` (see ``still_windows``).

    ``{'seconds', 'windows', 'samples', 'rms', 'peak', 'sign_changes'}``: the total time, the
    root mean square and the largest magnitude of ``v`` in them, and the most sign changes
    (|v| above ``min_speed``) in any ``window`` s inside one stretch. rms, peak and
    sign_changes are None with no window.
    """
    tt = np.asarray(t, dtype=float)
    vv = np.asarray(v, dtype=float)
    out = {'seconds': float(sum(b - a for a, b in windows)), 'windows': len(windows),
           'samples': 0, 'rms': None, 'peak': None, 'sign_changes': None}
    chosen, changes = [], 0
    for start, end in windows:
        pick = (tt >= start) & (tt <= end) & np.isfinite(vv)
        chosen.append(vv[pick])
        changes = max(changes, sign_changes_per_window(tt[pick], vv[pick], min_speed, window))
    values = np.concatenate(chosen) if chosen else np.zeros(0)
    if len(values):
        out.update(samples=int(len(values)), rms=float(np.sqrt(np.mean(values ** 2))),
                   peak=float(np.max(np.abs(values))), sign_changes=changes)
    return out


_NUMBER = r'(\d+(?:\.\d+)?)'
_HALF = (r'(\d+) steps(?:, median ' + _NUMBER + r' ms, p99 ' + _NUMBER + r' ms, longest '
         + _NUMBER + r' ms)?')
_SELF_PAIR = re.compile(r'link keep-out by self pair:\s+active\s+' + _HALF
                        + r'\s*;\s*not active\s+' + _HALF)
_CYCLES = re.compile(r'teleop cycle computation:\s+(\d+) cycles(?:, median ' + _NUMBER
                     + r' ms, p99 ' + _NUMBER + r' ms, longest ' + _NUMBER + r' ms, (\d+) over '
                     r'the ' + _NUMBER + r' ms period)?')


def _half(groups) -> dict:
    steps, median_ms, p99_ms, longest_ms = groups
    return {'steps': int(steps), 'median_ms': None if median_ms is None else float(median_ms),
            'p99_ms': None if p99_ms is None else float(p99_ms),
            'longest_ms': None if longest_ms is None else float(longest_ms)}


def parse_teleop_summary(text: str) -> dict:
    """
    Return what the teleop node's periodic log line says about its step times.

    Looks for the sentences ``link keep-out by self pair: active N steps, median A ms, p99 B
    ms, longest C ms; not active M steps, median D ms, p99 E ms, longest F ms.`` (a half with
    no steps has no numbers after its count) and ``teleop cycle computation: K cycles, median
    G ms, p99 H ms, longest I ms, J over the P ms period.``, wherever they stand in ``text``.
    Returns ``{'pair': None or {'active': half, 'inactive': half}, 'cycles': None or {...}}``
    with ``half = {'steps', 'median_ms', 'p99_ms', 'longest_ms'}`` (None where absent).
    """
    out = {'pair': None, 'cycles': None}
    found = _SELF_PAIR.search(text)
    if found:
        g = found.groups()
        out['pair'] = {'active': _half(g[:4]), 'inactive': _half(g[4:])}
    found = _CYCLES.search(text)
    if found:
        k, median_ms, p99_ms, longest_ms, over, period = found.groups()
        out['cycles'] = {
            'cycles': int(k), 'median_ms': None if median_ms is None else float(median_ms),
            'p99_ms': None if p99_ms is None else float(p99_ms),
            'longest_ms': None if longest_ms is None else float(longest_ms),
            'over': None if over is None else int(over),
            'period_ms': None if period is None else float(period)}
    return out
