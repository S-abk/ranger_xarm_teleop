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
