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
* ``head``    headset pose: ``p``, ``q`` (recorded, not analysed)
* ``target``  ``/teleop/target_pose``: ``stamp`` (ns), ``p``, ``q``
* ``state``   ``/teleop/state``: ``stamp``, ``engaged``, ``hold``, ``servo`` (MoveIt Servo
  status code), ``tool`` (p, q) or None; and, from the joint-space solver on, ``conditioning``,
  ``damped``, ``damping``, ``lead_translation`` (m), ``lead_rotation`` (rad)
* ``joints``  arm joint positions: ``stamp`` (ns), ``q``, ``v`` (joint velocities or None)
* ``command`` the joint command of the node (``/servo_node/delta_joint_cmds``): ``stamp``,
  ``names``, ``v`` (rad/s)

All times compared here come from the recorder's one monotonic clock; speeds and
accelerations are computed over header stamps where the rows carry them. A figure whose
rows are missing (a recording from before the solver) is NaN, and the report says n/a.
"""

import math
from typing import Any, Dict, List, Sequence

import numpy as np

from .rotations import q_between

STOPPED_RAD_S = 0.005        # joint speed below this counts as stopped
MIN_MOTION = 1e-4            # a curve that moves less than this has no measurable lag
HALT_CODES = (2, 5, 6)       # moveit_msgs/ServoStatus: singularity, collision, joint bound
SMOOTHING_SAMPLES = 3        # moving average for the smoothed acceleration


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


def _seconds(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Return the time of each row: its header stamp in seconds when all have one, else ``t``."""
    if rows and all(isinstance(r.get('stamp'), int) and r['stamp'] > 0 for r in rows):
        return np.array([r['stamp'] for r in rows], dtype=float) * 1e-9
    return np.array([r['t'] for r in rows], dtype=float)


def _moving_average(values: np.ndarray, samples: int = SMOOTHING_SAMPLES) -> np.ndarray:
    """Return the centred moving average over ``samples`` rows (the ends use what exists)."""
    if len(values) < samples:
        return values
    kernel = np.ones(samples) / samples
    full = np.stack([np.convolve(values[:, i], kernel, mode='valid')
                     for i in range(values.shape[1])], axis=1)
    return full


def joint_accelerations(times: np.ndarray, positions: np.ndarray):
    """
    Return ``(raw, smoothed)`` peak joint acceleration (rad/s^2) of a joint position series.

    Raw: second difference of the positions at the sample times (joint-state rate), largest
    over joints and samples. Smoothed: the same series after a 3-sample moving average of
    the acceleration. Rows with a repeated time are dropped; NaN when fewer than 3 remain.
    """
    keep = np.concatenate([[True], np.diff(times) > 1e-6])
    times, positions = times[keep], positions[keep]
    if len(times) < 3:
        return math.nan, math.nan
    dt = np.diff(times)
    velocity = np.diff(positions, axis=0) / dt[:, None]
    middle = (times[1:] + times[:-1]) / 2.0
    acceleration = np.diff(velocity, axis=0) / np.diff(middle)[:, None]
    smooth = _moving_average(acceleration)
    return float(np.abs(acceleration).max()), float(np.abs(smooth).max())


def tool_speeds(rows: List[Dict[str, Any]]):
    """
    Return the peak linear (m/s) and angular (rad/s) speed of the measured tool pose.

    ``rows`` are state rows with a ``tool`` pose. The tool pose is forward kinematics of
    the newest joint state, so successive rows can repeat a pose: speed is taken between
    consecutive *different* poses over the time between them.
    """
    linear = angular = 0.0
    seen = False
    times = _seconds(rows)
    previous = None
    for row, t in zip(rows, times):
        if previous is not None and (row['tool'][0] != previous[0][0]
                                     or row['tool'][1] != previous[0][1]):
            dt = t - previous[1]
            if dt > 1e-6:
                seen = True
                linear = max(linear, float(np.linalg.norm(
                    np.array(row['tool'][0]) - np.array(previous[0][0]))) / dt)
                angular = max(angular, q_between(row['tool'][1], previous[0][1]) / dt)
        if previous is None or row['tool'][0] != previous[0][0] \
                or row['tool'][1] != previous[0][1]:
            previous = (row['tool'], t)
    return (linear, angular) if seen else (math.nan, math.nan)


def _command_step(rows: List[Dict[str, Any]]) -> float:
    """
    Return the largest change of any joint's commanded velocity between two rows, rad/s.

    A joint missing from a row counts as zero there (the node sends zeros on release).
    """
    if len(rows) < 2:
        return math.nan
    names = sorted({name for row in rows for name in row['names']})
    series = np.array([[dict(zip(row['names'], row['v'])).get(name, 0.0) for name in names]
                       for row in rows])
    return float(np.abs(np.diff(series, axis=0)).max())


def _solver_figures(engaged_states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the figures that come from the solver fields of the state rows (NaN if absent)."""
    def column(name: str) -> List[float]:
        return [float(r[name]) for r in engaged_states if r.get(name) is not None]
    lead_t, lead_r = column('lead_translation'), column('lead_rotation')
    conditioning = column('conditioning')
    damped = [bool(r['damped']) for r in engaged_states if r.get('damped') is not None]
    codes = [r['servo'] for r in engaged_states if r.get('servo') is not None]
    entered = sum(1 for before, now in zip([None] + codes, codes)
                  if now in HALT_CODES and before not in HALT_CODES)
    return {
        'lead_translation_max': max(lead_t, default=math.nan),
        'lead_rotation_max_deg': math.degrees(max(lead_r, default=math.nan)),
        'damped_share': sum(damped) / len(damped) if damped else math.nan,
        'conditioning_min': min(conditioning, default=math.nan),
        'servo_halts': entered if codes else None}


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
    commands = [r for r in records if r['k'] == 'command']
    all_states = [r for r in records if r['k'] == 'state']
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

        # Peaks of the motion itself, from the start of the engagement until the tip has
        # settled (the same window as the path error).
        moving = [r for r in joints if start <= r['t'] <= min(end + settle, until)]
        out['peak_joint_speed'] = _peak_joint_speed(moving)
        raw, smoothed = (joint_accelerations(_seconds(moving), np.array([r['q'] for r in moving]))
                         if len(moving) >= 3 else (math.nan, math.nan))
        out['peak_joint_acceleration'] = raw
        out['peak_joint_acceleration_smoothed'] = smoothed
        window_states = [r for r in states if start <= r['t'] <= min(end + settle, until)]
        out['peak_tool_speed'], out['peak_tool_angular_speed'] = (
            tool_speeds(window_states) if len(window_states) >= 2 else (math.nan, math.nan))
        out['command_step_max'] = _command_step([r for r in commands
                                                 if start <= r['t'] <= end])
        out.update(_solver_figures([r for r in all_states if start <= r['t'] <= end]))
        runs.append(out)
    return {'engagements': runs, 'records': len(records)}


def _peak_joint_speed(rows: List[Dict[str, Any]]) -> float:
    """
    Return the largest joint speed (rad/s) over the rows.

    From the differences of the joint positions over the row times: a joint state's own
    velocity field is not a measurement everywhere (mock hardware with a position
    interface reports zero). The reported velocities are used only without two positions.
    """
    if not rows:
        return math.nan
    if len(rows) < 2:
        if all(r.get('v') is not None for r in rows):
            return float(np.abs(np.array([r['v'] for r in rows])).max())
        return math.nan
    times = _seconds(rows)
    q = np.array([r['q'] for r in rows])
    dt = np.diff(times)
    ok = dt > 1e-6
    if not ok.any():
        return math.nan
    return float((np.abs(np.diff(q, axis=0))[ok] / dt[ok][:, None]).max())


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


def _fmt(value: Any, scale: float = 1.0, spec: str = '.2f') -> str:
    """Return ``value * scale`` formatted, or ``n/a`` for None and NaN."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return 'n/a'
    return format(value * scale, spec)


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
            f'  joints: peak speed {_fmt(run.get("peak_joint_speed"))} rad/s, peak acceleration '
            f'{_fmt(run.get("peak_joint_acceleration"), spec=".1f")} rad/s^2 raw, '
            f'{_fmt(run.get("peak_joint_acceleration_smoothed"), spec=".1f")} rad/s^2 after a '
            f'{SMOOTHING_SAMPLES}-sample moving average; largest step of the commanded joint '
            f'velocity {_fmt(run.get("command_step_max"))} rad/s per cycle',
            f'  tool: peak speed {_fmt(run.get("peak_tool_speed"), 1000.0, ".0f")} mm/s, '
            f'{_fmt(run.get("peak_tool_angular_speed"), math.degrees(1.0), ".1f")} deg/s',
            f'  solver: max lead {_fmt(run.get("lead_translation_max"), 1000.0, ".1f")} mm / '
            f'{_fmt(run.get("lead_rotation_max_deg"), 1.0, ".1f")} deg, damped '
            f'{_fmt(run.get("damped_share"), 100.0, ".0f")} % of the engaged time, minimum '
            f'conditioning {_fmt(run.get("conditioning_min"), 1.0, ".3f")}, Servo halt statuses '
            f'{"n/a" if run.get("servo_halts") is None else run["servo_halts"]}',
        ]
    return '\n'.join(lines)
