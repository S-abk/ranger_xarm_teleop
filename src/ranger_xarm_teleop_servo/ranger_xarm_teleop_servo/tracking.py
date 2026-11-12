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
Tracking error, delay and stopping distance from a recorded teleop run. No ROS imports.

Input is the list of records ``tools/measure_tracking.py record`` writes, one dict per
received message, each with ``t`` (monotonic seconds at reception in the recorder) and
``k``:

* ``input``   right controller pose: ``p``
* ``target``  ``/teleop/target_pose``: ``stamp`` (ns), ``p``, ``q``
* ``state``   ``/teleop/state``: ``stamp``, ``engaged``, ``hold``, ``tool`` (p, q) or None
* ``joints``  arm joint positions: ``q``

All times compared here come from the recorder's one monotonic clock.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .rotations import q_between

STOPPED_RAD_S = 0.005        # joint speed below this counts as stopped
MIN_MOTION = 1e-4            # a curve that moves less than this has no measurable lag


def _rms(values: Sequence[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else math.nan


def _interp(times: np.ndarray, points: np.ndarray, at: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(at, times, points[:, i]) for i in range(points.shape[1])], axis=1)


def _polyline_distance(point: np.ndarray, line: np.ndarray) -> float:
    """Return the distance from ``point`` to the polyline through the rows of ``line``."""
    if len(line) == 1:
        return float(np.linalg.norm(point - line[0]))
    a, b = line[:-1], line[1:]
    ab = b - a
    length = np.einsum('ij,ij->i', ab, ab)
    s = np.clip(np.einsum('ij,ij->i', point - a, ab) / np.where(length > 0, length, 1.0), 0, 1)
    nearest = a + s[:, None] * ab
    return float(np.min(np.linalg.norm(point - nearest, axis=1)))


def engagements(records: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Return ``{'start', 'end'}`` (recorder time) of every run of published targets."""
    spans = []
    current = None
    for record in records:
        if record['k'] != 'state':
            continue
        if record['engaged'] and current is None:
            current = {'start': record['t'], 'end': record['t']}
        elif record['engaged']:
            current['end'] = record['t']
        elif current is not None:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    return spans


def analyze(records: List[Dict[str, Any]], settle: float = 1.0) -> Dict[str, Any]:
    """Return the figures for every engagement in ``records``."""
    states = [r for r in records if r['k'] == 'state' and r.get('tool')]
    targets = [r for r in records if r['k'] == 'target']
    joints = [r for r in records if r['k'] == 'joints']
    inputs = [r for r in records if r['k'] == 'input']
    by_stamp = {r['stamp']: r for r in states}
    tool_t = np.array([r['t'] for r in states])
    tool_p = np.array([r['tool'][0] for r in states]) if states else np.zeros((0, 3))
    joint_t = np.array([r['t'] for r in joints])
    joint_q = np.array([r['q'] for r in joints]) if joints else np.zeros((0, 6))
    input_t = np.array([r['t'] for r in inputs])
    input_p = np.array([r['p'] for r in inputs]) if inputs else np.zeros((0, 3))
    spans = engagements(records)
    runs = []
    for number, span in enumerate(spans):
        # What happens after the release is measured up to the next engagement.
        until = spans[number + 1]['start'] if number + 1 < len(spans) else math.inf
        start, end = span['start'], span['end']
        run_targets = [r for r in targets if start - 0.03 <= r['t'] <= end + 0.03]
        if len(run_targets) < 2:
            continue
        target_t = np.array([r['t'] for r in run_targets])
        target_p = np.array([r['p'] for r in run_targets])
        out: Dict[str, Any] = {'start': start, 'duration': end - start,
                               'targets': len(run_targets),
                               'path_length': float(np.sum(np.linalg.norm(
                                   np.diff(target_p, axis=0), axis=1)))}

        # Error between each target and the tool pose measured in the same cycle.
        same_cycle, same_cycle_angle = [], []
        for record in run_targets:
            state = by_stamp.get(record['stamp'])
            if state is not None:
                same_cycle.append(float(np.linalg.norm(
                    np.array(record['p']) - np.array(state['tool'][0]))))
                same_cycle_angle.append(q_between(record['q'], state['tool'][1]))
        out['following_error_rms'] = _rms(same_cycle)
        out['following_error_max'] = max(same_cycle, default=math.nan)
        out['following_angle_max_deg'] = math.degrees(max(same_cycle_angle, default=math.nan))

        # Distance of the measured tip from the commanded path, until it has settled.
        during = (tool_t >= start) & (tool_t <= min(end + settle, until))
        cross = [_polyline_distance(p, target_p) for p in tool_p[during]]
        out['path_error_rms'] = _rms(cross)
        out['path_error_max'] = max(cross, default=math.nan)

        # Where the tip ends relative to the last target.
        settled = tool_p[(tool_t >= end + settle) & (tool_t <= min(end + settle + 0.2, until))]
        final = settled[-1] if len(settled) else (tool_p[during][-1] if during.any() else None)
        out['final_error'] = (float(np.linalg.norm(final - target_p[-1]))
                              if final is not None else math.nan)

        # Lag: the shift of the measured tip that best matches the target.
        best = (math.inf, math.nan)
        if during.sum() > 5:
            for shift in np.arange(0.0, 0.5, 0.005):
                moved = _interp(tool_t, tool_p, target_t + shift)
                error = _rms(np.linalg.norm(moved - target_p, axis=1))
                if error < best[0]:
                    best = (error, float(shift))
        out['lag'] = best[1]
        out['error_rms_at_lag'] = best[0]

        # Delays, as the time shift between normalised motion curves: how far the
        # controller, the target, the joints and the tip have moved from where they
        # were at engagement, each divided by its own maximum.
        input_curve = _progress(input_t, input_p, start, end) if len(input_t) else None
        target_curve = _progress(target_t, target_p, start, end)
        joint_curve = _progress(joint_t, joint_q, start, end) if len(joint_t) else None
        tip_curve = _progress(tool_t, tool_p, start, end) if len(tool_t) else None
        out['input_to_target_delay'] = progress_lag(input_curve, target_curve)
        out['input_to_joint_delay'] = progress_lag(input_curve, joint_curve)
        out['input_to_tip_delay'] = progress_lag(input_curve, tip_curve)

        # After the last target: how far the tip still travels and for how long.
        after = (tool_t >= target_t[-1]) & (tool_t < until)
        if after.any():
            at_release = _interp(tool_t, tool_p, np.array([target_t[-1]]))[0]
            travel = np.linalg.norm(tool_p[after] - at_release, axis=1)
            out['stop_distance'] = float(travel[-1]) if len(travel) else math.nan
            out['stop_distance_max'] = float(travel.max()) if len(travel) else math.nan
        out['stop_time'] = _stop_time(joint_t, joint_q, float(target_t[-1]), until)
        runs.append(out)
    return {'engagements': runs, 'records': len(records)}


def _progress(times: np.ndarray, values: np.ndarray, start: float, end: float):
    """
    Return ``(t, s)``: how far a signal has moved from its value at ``start``, 0 to 1.

    ``values`` are positions or joint angles; the distance from the first sample is
    divided by its largest value, so signals in different units can be compared.
    """
    inside = (times >= start) & (times <= end)
    if inside.sum() < 5:
        return None
    t, v = times[inside], values[inside]
    distance = np.linalg.norm(v - v[0], axis=1)
    if distance.max() < MIN_MOTION:
        return None
    return t, distance / distance.max()


def progress_lag(first, second, limit: float = 0.5) -> float:
    """
    Return the time shift (s) by which ``second`` follows ``first``.

    Both are ``(t, s)`` progress curves; the shift minimises the rms difference.
    """
    if first is None or second is None:
        return math.nan
    best = (math.inf, math.nan)
    for shift in np.arange(-0.1, limit, 0.002):
        at = first[0] + shift
        inside = (at >= second[0][0]) & (at <= second[0][-1])
        if inside.sum() < 5:
            continue
        error = _rms(np.interp(at[inside], second[0], second[1]) - first[1][inside])
        if error < best[0]:
            best = (error, float(shift))
    return best[1]


def _stop_time(joint_t, joint_q, release: float, until: float) -> float:
    after = (joint_t >= release) & (joint_t < until)
    if after.sum() < 3:
        return math.nan
    t, q = joint_t[after], joint_q[after]
    speed = np.max(np.abs(np.diff(q, axis=0)), axis=1) / np.maximum(np.diff(t), 1e-6)
    moving = np.nonzero(speed > STOPPED_RAD_S)[0]
    return 0.0 if len(moving) == 0 else float(t[moving[-1] + 1] - release)


def format_report(result: Dict[str, Any]) -> str:
    """Return the figures as text."""
    lines = [f"records: {result['records']}, engagements: {len(result['engagements'])}"]
    for i, run in enumerate(result['engagements']):
        lines += [
            f'engagement {i + 1}: {run["duration"]:.2f} s, {run["targets"]} targets, '
            f'commanded path {run["path_length"] * 1000:.1f} mm',
            f'  following error (target vs tip in the same cycle): rms '
            f'{run["following_error_rms"] * 1000:.2f} mm, max '
            f'{run["following_error_max"] * 1000:.2f} mm, max angle '
            f'{run["following_angle_max_deg"]:.2f} deg',
            f'  path error (tip to the commanded path): rms {run["path_error_rms"] * 1000:.2f} mm,'
            f' max {run["path_error_max"] * 1000:.2f} mm',
            f'  final error after settling: {run["final_error"] * 1000:.2f} mm',
            f'  lag target -> tip: {run["lag"] * 1000:.0f} ms '
            f'(rms error {run["error_rms_at_lag"] * 1000:.2f} mm at that shift)',
            f'  delay from the controller motion: target {run["input_to_target_delay"] * 1000:.0f}'
            f' ms, joints {run["input_to_joint_delay"] * 1000:.0f} ms, tip '
            f'{run["input_to_tip_delay"] * 1000:.0f} ms',
            f'  after the last target: tip travels {run.get("stop_distance", math.nan) * 1000:.2f}'
            f' mm (max {run.get("stop_distance_max", math.nan) * 1000:.2f} mm), joints stopped '
            f'after {run["stop_time"] * 1000:.0f} ms',
        ]
    return '\n'.join(lines)
