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
Quaternion and pose helpers (numpy only, no ROS).

Quaternions are ``(x, y, z, w)`` tuples of floats, as on the wire and in ROS messages.
Every function returns plain tuples so results are hashable and comparable.
"""

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

IDENTITY_Q: Quat = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Pose:
    """A rigid pose: position ``p`` and orientation ``q`` (xyzw)."""

    p: Vec3
    q: Quat


IDENTITY_POSE = Pose((0.0, 0.0, 0.0), IDENTITY_Q)


def vec(values: Sequence[float]) -> Vec3:
    """Return three floats as a tuple."""
    x, y, z = (float(v) for v in values)
    return (x, y, z)


def q_normalize(q: Sequence[float]) -> Quat:
    """Return ``q`` scaled to unit norm; raise ``ValueError`` for a zero quaternion."""
    a = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(a))
    if a.shape != (4,) or not math.isfinite(n) or n < 1e-12:
        raise ValueError(f'not a quaternion: {q!r}')
    a = a / n
    return (float(a[0]), float(a[1]), float(a[2]), float(a[3]))


def q_mul(a: Sequence[float], b: Sequence[float]) -> Quat:
    """Hamilton product ``a * b``: the rotation ``b`` followed by ``a``."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def q_conj(q: Sequence[float]) -> Quat:
    """Return the conjugate (the inverse of a unit quaternion)."""
    return (-q[0], -q[1], -q[2], q[3])


def q_rotate(q: Sequence[float], v: Sequence[float]) -> Vec3:
    """Rotate vector ``v`` by unit quaternion ``q``."""
    x, y, z, _ = q_mul(q_mul(q, (v[0], v[1], v[2], 0.0)), q_conj(q))
    return (x, y, z)


def q_angle(q: Sequence[float]) -> float:
    """Return the rotation angle of ``q`` in ``[0, pi]``."""
    s = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2])
    return 2.0 * math.atan2(s, abs(q[3]))


def q_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the angle of the rotation taking orientation ``a`` to ``b``."""
    return q_angle(q_mul(b, q_conj(a)))


def q_from_axis_angle(axis: Sequence[float], angle: float) -> Quat:
    """Return the quaternion of a rotation by ``angle`` about ``axis``."""
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError('zero rotation axis')
    a = a / n * math.sin(angle / 2.0)
    return (float(a[0]), float(a[1]), float(a[2]), math.cos(angle / 2.0))


def q_from_yaw(angle: float) -> Quat:
    """Return the quaternion of a rotation by ``angle`` about +z."""
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def q_slerp(a: Sequence[float], b: Sequence[float], t: float) -> Quat:
    """Interpolate from ``a`` (t=0) to ``b`` (t=1) along the shorter arc."""
    qa = np.asarray(a, dtype=float)
    qb = np.asarray(b, dtype=float)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        out = qa + t * (qb - qa)
    else:
        theta = math.acos(min(1.0, dot))
        out = (math.sin((1.0 - t) * theta) * qa + math.sin(t * theta) * qb) / math.sin(theta)
    return q_normalize(out)


def v_add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    """Return ``a + b``."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    """Return ``a - b``."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(a: Sequence[float], s: float) -> Vec3:
    """Return ``a * s``."""
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the dot product."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_norm(a: Sequence[float]) -> float:
    """Return the Euclidean length."""
    return math.sqrt(v_dot(a, a))


def compose(a: Pose, b: Pose) -> Pose:
    """Return ``a * b``: pose ``b`` given in the frame of ``a``, expressed in a's parent."""
    return Pose(v_add(a.p, q_rotate(a.q, b.p)), q_normalize(q_mul(a.q, b.q)))


def inverse(a: Pose) -> Pose:
    """Return the inverse transform of ``a``."""
    qi = q_conj(a.q)
    return Pose(v_scale(q_rotate(qi, a.p), -1.0), qi)
