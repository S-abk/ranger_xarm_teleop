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
Map the singularity measure over the tool poses reachable from a posture (numpy, no ROS).

A measuring device for offline analysis, not a way to drive the arm: from a start posture
the tool is walked along straight lines (or turned in place) in small steps, each solved
by a few pseudo-inverse iterations from the previous posture, the way a servo loop
continues from where the arm is. Along the way it records the condition number that
MoveIt Servo thresholds, the margin to the joint limits and the joint speed a given tool
speed would need.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ranger_xarm_teleop_servo import kinematics

STEP_M = 0.005
STEP_RAD = math.radians(1.0)
STOP_CONDITION = 500.0          # give up walking: numerically at the singularity
LIMIT_MARGIN_RAD = 0.1          # MoveIt Servo halts a joint this close to its bound


@dataclass(frozen=True)
class Sample:
    """One step of a walk."""

    travel: float               # m along a line, or rad of a turn
    q: Tuple[float, ...]
    condition: float
    margin: float               # rad to the nearest joint limit
    joint_speed: float          # largest joint speed (rad/s) for the walk's tool speed


@dataclass(frozen=True)
class Walk:
    """A walk from the start posture and why it ended."""

    samples: Tuple[Sample, ...]
    ended: str                  # 'complete', 'joint_limit', 'singular', 'unreachable'

    def reach(self, threshold: float) -> float:
        """Return the travel before the condition number first reaches ``threshold``."""
        for sample in self.samples:
            if sample.condition >= threshold:
                return sample.travel
        return self.samples[-1].travel

    @property
    def worst(self) -> float:
        """Return the largest condition number met on the way."""
        return max(sample.condition for sample in self.samples)


def _axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Return the rotation vector of a rotation matrix (angle below pi)."""
    cosine = max(-1.0, min(1.0, (float(np.trace(matrix)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3)
    vector = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0],
                       matrix[1, 0] - matrix[0, 1]])
    return vector / (2.0 * math.sin(angle)) * angle


def _rotation(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _solve(chain: kinematics.Chain, q: np.ndarray, position: np.ndarray,
           rotation: np.ndarray) -> Optional[np.ndarray]:
    """Move ``q`` so the tip is at the pose; None when it does not get there."""
    for _ in range(8):
        tip, _, _ = kinematics.forward(chain, q)
        error = np.concatenate([position - tip[:3, 3],
                                _axis_angle(rotation @ tip[:3, :3].T)])
        if float(np.linalg.norm(error)) < 1e-7:
            return q
        step = np.linalg.pinv(kinematics.jacobian(chain, q)) @ error
        if float(np.max(np.abs(step))) > 0.5:
            return None
        q = q + step
    return None


def _sample(chain, q, travel: float, twist: np.ndarray) -> Sample:
    rates = np.linalg.pinv(kinematics.jacobian(chain, q)) @ twist
    return Sample(travel, tuple(float(v) for v in q), kinematics.condition_number(chain, q),
                  kinematics.limit_margin(chain, q)[1], float(np.max(np.abs(rates))))


def walk(chain: kinematics.Chain, start: Sequence[float], direction: Sequence[float],
         distance: float, speed: float = 0.25, turn: bool = False,
         step: Optional[float] = None) -> Walk:
    """
    Walk the tool from the start posture and record the measure along the way.

    ``direction`` is in the base frame. Straight line of ``distance`` metres at fixed
    orientation, or with ``turn`` a rotation in place by ``distance`` radians about it.
    ``speed`` (m/s or rad/s) is the tool speed the joint speeds are reported for.
    """
    unit = np.array(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    step = step or (STEP_RAD if turn else STEP_M)
    twist = np.concatenate([np.zeros(3), unit * speed] if turn else [unit * speed, np.zeros(3)])
    q = np.array(start, dtype=float)
    origin, _, _ = kinematics.forward(chain, q)
    samples = [_sample(chain, q, 0.0, twist)]
    count = int(round(distance / step))
    for index in range(1, count + 1):
        travel = index * step
        if turn:
            position = origin[:3, 3]
            rotation = _rotation(unit * travel) @ origin[:3, :3]
        else:
            position = origin[:3, 3] + unit * travel
            rotation = origin[:3, :3]
        solved = _solve(chain, q, position, rotation)
        if solved is None:
            return Walk(tuple(samples), 'unreachable')
        q = solved
        samples.append(_sample(chain, q, travel, twist))
        if samples[-1].margin < LIMIT_MARGIN_RAD:
            return Walk(tuple(samples), 'joint_limit')
        if samples[-1].condition >= STOP_CONDITION:
            return Walk(tuple(samples), 'singular')
    return Walk(tuple(samples), 'complete')


AXES = (('+x', (1, 0, 0)), ('-x', (-1, 0, 0)), ('+y', (0, 1, 0)), ('-y', (0, -1, 0)),
        ('+z', (0, 0, 1)), ('-z', (0, 0, -1)))


def rays(chain: kinematics.Chain, start: Sequence[float], distance: float = 0.30,
         angle: float = math.radians(60.0), lower: float = 17.0, hard_stop: float = 30.0,
         linear_speed: float = 0.25, angular_speed: float = 1.0) -> List[Dict[str, Any]]:
    """Walk along and turn about each base axis; report how far before each threshold."""
    rows = []
    for turn, reach, speed in ((False, distance, linear_speed), (True, angle, angular_speed)):
        for name, direction in AXES:
            result = walk(chain, start, direction, reach, speed, turn)
            below = [s for s in result.samples if s.condition < hard_stop]
            rows.append({
                'motion': ('turn ' if turn else 'move ') + name, 'turn': turn,
                'asked': reach, 'ended': result.ended, 'travel': result.samples[-1].travel,
                'to_lower': result.reach(lower), 'to_hard_stop': result.reach(hard_stop),
                'worst': result.worst,
                'max_joint_speed_below_hard_stop': max(s.joint_speed for s in below)
                if below else None})
    return rows


def grid(chain: kinematics.Chain, start: Sequence[float], extent: float = 0.30,
         spacing: float = 0.10, step: float = 0.01) -> List[Dict[str, Any]]:
    """
    Measure a cube of tool positions around the start, each reached by a straight line.

    Per grid point: the offset, whether the line got there, the condition number there
    and the worst one on the way (what an operator moving straight there would meet).
    """
    count = int(round(extent / spacing))
    offsets = [i * spacing for i in range(-count, count + 1)]
    rows = []
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                offset = np.array([dx, dy, dz])
                length = float(np.linalg.norm(offset))
                if length < 1e-9:
                    here = kinematics.condition_number(chain, start)
                    rows.append({'offset': [0.0, 0.0, 0.0], 'ended': 'complete',
                                 'condition': here, 'worst': here,
                                 'margin': kinematics.limit_margin(chain, start)[1]})
                    continue
                result = walk(chain, start, offset, length, step=length / max(
                    1, int(math.ceil(length / step))))
                done = result.ended == 'complete'
                last = result.samples[-1]
                rows.append({'offset': [dx, dy, dz], 'ended': result.ended,
                             'condition': last.condition if done else None,
                             'worst': result.worst, 'margin': last.margin})
    return rows


def grid_summary(rows: Sequence[Dict[str, Any]],
                 levels: Sequence[float] = (17.0, 30.0, 50.0, 75.0, 100.0)) -> Dict[str, Any]:
    """Return the share of grid points a straight line reaches below each level."""
    total = len(rows)
    done = [row for row in rows if row['ended'] == 'complete']
    ended: Dict[str, int] = {}
    for row in rows:
        ended[row['ended']] = ended.get(row['ended'], 0) + 1
    return {
        'points': total, 'ended': ended,
        'reached_below': {str(level): sum(1 for row in done if row['worst'] < level) / total
                          for level in levels},
        'median_condition': float(np.median([row['condition'] for row in done]))
        if done else None}


def evaluate(chain: kinematics.Chain, posture: Sequence[float], extent: float = 0.30,
             spacing: float = 0.10) -> Dict[str, Any]:
    """Return the figures by which ready postures are compared."""
    tip, _, _ = kinematics.forward(chain, posture)
    shape = kinematics.resemblance(chain, posture)
    name, margin = kinematics.limit_margin(chain, posture)
    return {
        'posture': [float(v) for v in posture],
        'tool_position': [float(v) for v in tip[:3, 3]],
        'tool_axis': [float(v) for v in tip[:3, 2]],
        'condition': kinematics.condition_number(chain, posture),
        'singular_values': [float(v) for v in kinematics.singular_values(chain, posture)],
        'nearest_singularity': shape.kind, 'wrist_deg': shape.wrist_deg,
        'elbow_deg': shape.elbow_deg,
        'shoulder_m': shape.shoulder_m,
        'nearest_limit': {'joint': name, 'margin_rad': margin},
        'rays': rays(chain, posture, extent),
        'grid': grid_summary(grid(chain, posture, extent, spacing))}


def format_evaluation(result: Dict[str, Any]) -> str:
    """Render ``evaluate``'s result as text."""
    lines = [
        'posture (rad): ' + ' '.join(f'{v:+.3f}' for v in result['posture']),
        'tool position (m): ' + ' '.join(f'{v:+.3f}' for v in result['tool_position'])
        + '   tool z axis: ' + ' '.join(f'{v:+.2f}' for v in result['tool_axis']),
        f"condition number {result['condition']:.1f}   singular values "
        + ' '.join(f'{v:.3f}' for v in result['singular_values']),
        f"nearest singularity: {result['nearest_singularity']}; axes 4-6 "
        f"{result['wrist_deg']:.1f} deg apart, elbow {result['elbow_deg']:.1f} deg from "
        f"straight, wrist {result['shoulder_m'] * 1000:.0f} mm from axis 1; "
        f"nearest limit {result['nearest_limit']['joint']} "
        f"{result['nearest_limit']['margin_rad']:.2f} rad",
        '',
        f"{'motion':10s} {'ended':12s} {'travel':>8s} {'to lower':>9s} {'to stop':>8s} "
        f"{'worst':>7s} {'joint rad/s':>11s}"]
    for row in result['rays']:
        scale, unit = (math.degrees(1.0), 'deg') if row['turn'] else (1000.0, 'mm')
        speed = row['max_joint_speed_below_hard_stop']
        lines.append(
            f"{row['motion']:10s} {row['ended']:12s} {row['travel'] * scale:5.0f} {unit:3s}"
            f"{row['to_lower'] * scale:6.0f} {unit:3s}{row['to_hard_stop'] * scale:5.0f} {unit:3s}"
            f"{row['worst']:7.1f} {'n/a' if speed is None else format(speed, '11.2f')}")
    summary = result['grid']
    lines.append('')
    lines.append(f"grid of {summary['points']} tool positions, each reached by a straight "
                 f"line: {summary['ended']}")
    lines.append('share reached with the condition number staying below: ' + '  '.join(
        f'{level}: {share * 100:.0f} %' for level, share in summary['reached_below'].items()))
    return '\n'.join(lines)


# ---- A model of MoveIt Servo's singularity rule, to compare threshold values ----

NO_WARNING, APPROACHING, HALT, LEAVING = 0, 1, 2, 3     # moveit_msgs/ServoStatus codes


def servo_scale(chain: kinematics.Chain, q: Sequence[float], delta_x: np.ndarray,
                lower: float, hard_stop: float, leaving_multiplier: float = 2.0,
                step_scale: float = 0.01) -> Tuple[float, int]:
    """
    Return the velocity scale and status MoveIt Servo would apply at posture ``q``.

    The rule of ``velocityScalingFactorForSingularity`` (moveit_servo 2.12,
    ``utils/common.cpp``): whether the commanded Cartesian step ``delta_x`` leads toward
    the singularity is judged from its component along the weakest task direction; the
    scale falls linearly from 1 at ``lower`` to 0 at ``hard_stop`` when approaching, and
    to 0 at ``lower + multiplier * (hard_stop - lower)`` when leaving; at or beyond
    ``hard_stop`` it is 0 in every direction.

    The approaching/leaving judgement compares the condition number before and after a
    probe step along the weakest task direction. Where that step hardly changes the
    condition number (near this arm's wrist singularity it changes it by less than
    0.1 %), the outcome rests on rounding and on the sign the SVD happens to return, so
    this model and MoveIt Servo can disagree there on approaching versus leaving.
    """
    left, values, right = np.linalg.svd(kinematics.jacobian(chain, q))
    condition = values[0] / values[-1]
    if condition <= lower:
        return 1.0, NO_WARNING
    if condition >= hard_stop:
        return 0.0, HALT
    toward = left[:, -1]
    inverse = right.T @ np.diag(1.0 / values) @ left.T
    probe = np.array(q, dtype=float) + inverse @ (toward * step_scale)
    if kinematics.condition_number(chain, probe) <= condition:
        toward = -toward
    approaching = float(np.dot(toward, delta_x)) > 0.0
    upper = hard_stop if approaching else lower + (hard_stop - lower) * leaving_multiplier
    return (1.0 - (condition - lower) / (upper - lower),
            APPROACHING if approaching else LEAVING)


@dataclass(frozen=True)
class Drive:
    """Result of driving the tool with a constant twist under the singularity rule."""

    outcome: str                # 'free', 'slowed', 'stalled', 'halted', 'joint_limit'
    progress: float             # share of the asked motion that was made
    peak_joint_speed: float     # rad/s, after the rule and the joint speed limit
    peak_condition: float
    slowed_share: float         # share of the cycles in which the rule scaled the motion
    q: Tuple[float, ...]


def drive(chain: kinematics.Chain, start: Sequence[float], twist: Sequence[float],
          seconds: float, lower: float, hard_stop: float, period: float = 0.02,
          joint_speed_limit: float = 2.0, leaving_multiplier: float = 2.0) -> Drive:
    """
    Step the arm as a servo loop would for a constant tool twist (m/s and rad/s, base frame).

    Each cycle: joint step from the pseudo-inverse, scaled by ``servo_scale`` and then
    down to ``joint_speed_limit``. A halt ends the run; so does a joint within the limit
    margin. ``stalled`` means the rule held the arm to under half of the asked motion
    without ever reporting a halt.
    """
    twist = np.array(twist, dtype=float)
    q = np.array(start, dtype=float)
    cycles = int(round(seconds / period))
    made = 0.0
    peak_speed = 0.0
    peak_condition = kinematics.condition_number(chain, q)
    slowed = 0
    outcome = ''
    for _ in range(cycles):
        delta_x = twist * period
        scale, status = servo_scale(chain, q, delta_x, lower, hard_stop, leaving_multiplier)
        if status == HALT:
            outcome = 'halted'
            break
        step = np.linalg.pinv(kinematics.jacobian(chain, q)) @ delta_x * scale
        speed = float(np.max(np.abs(step))) / period
        if speed > joint_speed_limit:
            scale *= joint_speed_limit / speed
            step *= joint_speed_limit / speed
            speed = joint_speed_limit
        slowed += scale < 0.999
        q = q + step
        made += scale
        peak_speed = max(peak_speed, speed)
        peak_condition = max(peak_condition, kinematics.condition_number(chain, q))
        if kinematics.limit_margin(chain, q)[1] < LIMIT_MARGIN_RAD:
            outcome = 'joint_limit'
            break
    progress = made / cycles if cycles else 1.0
    if not outcome:
        outcome = 'free' if slowed == 0 else ('stalled' if progress < 0.5 else 'slowed')
    return Drive(outcome, progress, peak_speed, peak_condition,
                 slowed / cycles if cycles else 0.0, tuple(float(v) for v in q))


def sample_twists(count: int, linear_speed: float = 0.25, angular_speed: float = 1.0,
                  seed: int = 0) -> List[np.ndarray]:
    """Return ``count`` reproducible tool twists: random directions, speeds up to the caps."""
    random = np.random.default_rng(seed)
    twists = []
    for _ in range(count):
        linear = random.normal(size=3)
        angular = random.normal(size=3)
        twists.append(np.concatenate([
            linear / np.linalg.norm(linear) * linear_speed * random.uniform(0.3, 1.0),
            angular / np.linalg.norm(angular) * angular_speed * random.uniform(0.0, 1.0)]))
    return twists


def survey(chain: kinematics.Chain, start: Sequence[float], lower: float, hard_stop: float,
           count: int = 200, seconds: float = 1.0, seed: int = 0,
           joint_speed_limit: float = 2.0) -> Dict[str, Any]:
    """
    Drive ``count`` sampled twists from the start posture; return how the runs ended.

    With the default caps and one second a run covers up to 0.25 m and 57 deg, inside
    the offset limits of the simulation values.
    """
    runs = [drive(chain, start, twist, seconds, lower, hard_stop,
                  joint_speed_limit=joint_speed_limit)
            for twist in sample_twists(count, seed=seed)]
    outcomes: Dict[str, float] = {}
    for run in runs:
        outcomes[run.outcome] = outcomes.get(run.outcome, 0.0) + 1.0 / count
    return {
        'lower': lower, 'hard_stop': hard_stop, 'runs': count, 'outcomes': outcomes,
        'mean_progress': float(np.mean([run.progress for run in runs])),
        'peak_joint_speed': float(max(run.peak_joint_speed for run in runs)),
        'runs_at_joint_speed_limit': sum(
            1 for run in runs if run.peak_joint_speed >= joint_speed_limit - 1e-9) / count,
        'peak_condition': float(max(run.peak_condition for run in runs))}


def wander(chain: kinematics.Chain, start: Sequence[float], lower: float, hard_stop: float,
           sessions: int = 40, segments: int = 20, seconds: float = 1.0, seed: int = 0,
           box: float = 0.30, max_turn: float = math.radians(60.0),
           joint_speed_limit: float = 2.0) -> Dict[str, Any]:
    """
    Model teleoperation sessions: a chain of sampled one-second motions from the start.

    Each motion continues from where the last one ended, as successive clutch
    engagements do. The tool stays within ``box`` metres and ``max_turn`` of where it
    started: a sampled motion that would leave is turned back toward the start. A halt
    ends the session (the arm then needs MoveIt to get out). Returns how many sessions
    halted, after how many motions, and what share of the motions were slowed or held.
    """
    origin, _, _ = kinematics.forward(chain, start)
    first_halt: List[int] = []
    outcomes: Dict[str, int] = {}
    peak_speed = 0.0
    total = 0
    for session in range(sessions):
        q = tuple(start)
        twists = sample_twists(segments, seed=seed * 100003 + session)
        for index, twist in enumerate(twists):
            tip, _, _ = kinematics.forward(chain, q)
            ahead = tip[:3, 3] + twist[:3] * seconds - origin[:3, 3]
            for axis in range(3):
                if abs(ahead[axis]) > box:
                    twist[axis] = -abs(twist[axis]) * np.sign(ahead[axis])
            turned = _axis_angle(tip[:3, :3] @ origin[:3, :3].T)
            if float(np.linalg.norm(turned + twist[3:] * seconds)) > max_turn:
                speed = float(np.linalg.norm(twist[3:]))
                size = float(np.linalg.norm(turned))
                twist[3:] = -turned / size * speed if size > 1e-9 else 0.0
            run = drive(chain, q, twist, seconds, lower, hard_stop,
                        joint_speed_limit=joint_speed_limit)
            outcomes[run.outcome] = outcomes.get(run.outcome, 0) + 1
            peak_speed = max(peak_speed, run.peak_joint_speed)
            total += 1
            q = run.q
            if run.outcome == 'halted':
                first_halt.append(index + 1)
                break
            if run.outcome == 'joint_limit':
                break               # needs MoveIt as well; counted, not a singularity halt
    return {
        'lower': lower, 'hard_stop': hard_stop, 'sessions': sessions, 'segments': segments,
        'sessions_halted': len(first_halt) / sessions,
        'median_motions_before_halt': float(np.median(first_halt)) if first_halt else None,
        'motions': total,
        'outcomes': {name: count / total for name, count in sorted(outcomes.items())},
        'peak_joint_speed': peak_speed}
