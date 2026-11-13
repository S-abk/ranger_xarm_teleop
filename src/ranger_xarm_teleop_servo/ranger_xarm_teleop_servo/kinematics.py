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
Arm kinematics from a URDF, for diagnostics only (numpy, no ROS).

The singularity measure here is the one MoveIt Servo applies its thresholds to: the ratio
of the largest to the smallest singular value of the 6 x n geometric Jacobian of the
group's tip link (linear rows in metres first, angular rows in radians), expressed in the
group's root link, reference point at the tip link origin. Nothing in this module is used
to drive the arm.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ElementTree

import numpy as np


@dataclass(frozen=True)
class Segment:
    """One joint of the chain: the fixed transform to its frame and, if it moves, its axis."""

    name: str
    origin: np.ndarray                  # 4 x 4, parent link frame to joint frame
    axis: Optional[np.ndarray]          # unit axis in the joint frame; None for a fixed joint
    lower: float = -math.inf
    upper: float = math.inf
    velocity: float = math.inf


@dataclass(frozen=True)
class Chain:
    """A serial chain of revolute and fixed joints from ``base`` to ``tip``."""

    base: str
    tip: str
    segments: Tuple[Segment, ...]

    @property
    def joints(self) -> Tuple[Segment, ...]:
        """Return the moving joints, base first."""
        return tuple(s for s in self.segments if s.axis is not None)

    @property
    def joint_names(self) -> List[str]:
        """Return the names of the moving joints, base first."""
        return [s.name for s in self.joints]


class ChainError(ValueError):
    """The URDF does not contain the chain asked for, or a joint type is not supported."""


def _floats(text: Optional[str], default: Sequence[float]) -> List[float]:
    return [float(v) for v in text.split()] if text else list(default)


def _transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                   [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                   [-sp, cp * sr, cp * cr]]
    out[:3, 3] = xyz
    return out


def chain_from_urdf(urdf: str, base: str = 'xarm_link_base',
                    tip: str = 'xarm_link_tcp') -> Chain:
    """Read the chain ``base`` -> ``tip`` from URDF text."""
    try:
        root = ElementTree.fromstring(urdf)
    except ElementTree.ParseError as err:
        raise ChainError(f'not a URDF: {err}') from err
    by_child = {}
    for joint in root.findall('joint'):
        child = joint.find('child')
        if child is not None:
            by_child[child.get('link')] = joint
    segments: List[Segment] = []
    link = tip
    while link != base:
        joint = by_child.get(link)
        if joint is None:
            raise ChainError(f'no chain from {base} to {tip} (stopped at {link})')
        origin = joint.find('origin')
        fixed = _transform(_floats(origin.get('xyz') if origin is not None else None, (0, 0, 0)),
                           _floats(origin.get('rpy') if origin is not None else None, (0, 0, 0)))
        kind = joint.get('type')
        if kind == 'fixed':
            segments.append(Segment(joint.get('name'), fixed, None))
        elif kind in ('revolute', 'continuous'):
            axis_element = joint.find('axis')
            axis = np.array(_floats(axis_element.get('xyz') if axis_element is not None
                                    else None, (1, 0, 0)))
            limit = joint.find('limit')
            bounded = kind == 'revolute' and limit is not None
            segments.append(Segment(
                joint.get('name'), fixed, axis / np.linalg.norm(axis),
                float(limit.get('lower', '-inf')) if bounded else -math.inf,
                float(limit.get('upper', 'inf')) if bounded else math.inf,
                float(limit.get('velocity', 'inf')) if limit is not None else math.inf))
        else:
            raise ChainError(f'joint {joint.get("name")} is {kind}: not supported')
        link = joint.find('parent').get('link')
    return Chain(base, tip, tuple(reversed(segments)))


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    out = np.eye(4)
    out[:3, :3] = [[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                   [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                   [z * x * k - y * s, z * y * k + x * s, c + z * z * k]]
    return out


def forward(chain: Chain, q: Sequence[float]):
    """
    Return the tip pose and, per moving joint, its axis and a point on it (base frame).

    ``(tip 4 x 4, axes n x 3, points n x 3)``.
    """
    joints = chain.joints
    if len(q) != len(joints):
        raise ValueError(f'{len(joints)} joint angles are needed, got {len(q)}')
    pose = np.eye(4)
    axes, points = [], []
    index = 0
    for segment in chain.segments:
        pose = pose @ segment.origin
        if segment.axis is not None:
            axes.append(pose[:3, :3] @ segment.axis)
            points.append(pose[:3, 3].copy())
            pose = pose @ _rotation(segment.axis, float(q[index]))
            index += 1
    return pose, np.array(axes), np.array(points)


def jacobian(chain: Chain, q: Sequence[float]) -> np.ndarray:
    """Return the 6 x n geometric Jacobian of the tip origin: linear rows, then angular."""
    tip, axes, points = forward(chain, q)
    out = np.zeros((6, len(axes)))
    for i, (axis, point) in enumerate(zip(axes, points)):
        out[:3, i] = np.cross(axis, tip[:3, 3] - point)
        out[3:, i] = axis
    return out


def singular_values(chain: Chain, q: Sequence[float]) -> np.ndarray:
    """Return the singular values of the Jacobian, largest first."""
    return np.linalg.svd(jacobian(chain, q), compute_uv=False)


def condition_number(chain: Chain, q: Sequence[float]) -> float:
    """Return MoveIt Servo's singularity measure: largest over smallest singular value."""
    values = singular_values(chain, q)
    return math.inf if values[-1] <= 0.0 else float(values[0] / values[-1])


def limit_margin(chain: Chain, q: Sequence[float]) -> Tuple[str, float]:
    """Return the joint nearest to a position limit and its distance to it (rad)."""
    best = ('', math.inf)
    for segment, angle in zip(chain.joints, q):
        margin = min(angle - segment.lower, segment.upper - angle)
        if margin < best[1]:
            best = (segment.name, float(margin))
    return best


@dataclass(frozen=True)
class Resemblance:
    """
    Which classic singularity of a six-joint arm a posture is nearest to.

    ``kind`` comes from the joint motion that moves the tool least (the right singular
    vector of the smallest singular value): at a wrist singularity it is joints 4 and 6
    turning against each other, at a shoulder singularity joint 1 takes part as well, at
    an elbow singularity it is the in-plane joints 2, 3 and 5. This holds for arms whose
    wrist axes do not meet in a point, where the textbook postures are only approximate.
    The three geometric figures describe the posture in the operator's terms.
    """

    condition: float
    kind: str               # 'wrist', 'shoulder' or 'elbow'
    wrist_share: float      # squared parts of that joint motion: joints 4 and 6,
    shoulder_share: float   # joint 1,
    elbow_share: float      # joints 2, 3 and 5
    wrist_deg: float        # angle between joint axes 4 and 6 (0: forearm and tool flange in line)
    elbow_deg: float        # angle of the elbow away from straight (0: stretched out)
    shoulder_m: float       # distance of the wrist (joint 5) from joint axis 1 (0: overhead)

    def label(self, threshold: float = 17.0) -> str:
        """Return ``kind`` when the condition number is at or above ``threshold``, else ''."""
        return self.kind if self.condition >= threshold else ''


SHOULDER_SHARE = 0.15       # joint 1's part of the motion from which it is called shoulder


def resemblance(chain: Chain, q: Sequence[float]) -> Resemblance:
    """Classify the nearest singularity of a six-joint arm (joints 2 and 3 parallel)."""
    _, axes, points = forward(chain, q)
    if len(axes) != 6:
        raise ValueError('a six-joint arm is needed')
    _, values, right = np.linalg.svd(jacobian(chain, q))
    parts = right[-1] ** 2
    shoulder_share = float(parts[0])
    elbow_share = float(parts[1] + parts[2] + parts[4])
    wrist_share = float(parts[3] + parts[5])
    if elbow_share > 0.5:
        kind = 'elbow'
    elif shoulder_share >= SHOULDER_SHARE:
        kind = 'shoulder'
    else:
        kind = 'wrist'
    condition = math.inf if values[-1] <= 0.0 else float(values[0] / values[-1])
    cos_wrist = abs(float(np.dot(axes[3], axes[5])))
    wrist = math.degrees(math.acos(min(1.0, cos_wrist)))
    centre = points[4]
    normal = axes[1]

    def in_plane(vector: np.ndarray) -> np.ndarray:
        return vector - np.dot(vector, normal) * normal

    upper = in_plane(points[2] - points[1])
    fore = in_plane(centre - points[2])
    cross = float(np.dot(np.cross(upper, fore), normal))
    elbow = abs(math.degrees(math.atan2(cross, float(np.dot(upper, fore)))))
    elbow = min(elbow, 180.0 - elbow)       # folded flat counts as well
    offset = centre - points[0]
    shoulder = float(np.linalg.norm(offset - np.dot(offset, axes[0]) * axes[0]))
    return Resemblance(condition, kind, wrist_share, shoulder_share, elbow_share,
                       wrist, elbow, shoulder)
