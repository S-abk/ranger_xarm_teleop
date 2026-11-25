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
How far the robot is from starting to tip (contract, "Stability monitor"), as pure numpy.

* **The figure.** The floor's reaction acts at the centre of pressure ``p`` on the contact
  plane. Newton-Euler over every body i (mass m_i at c_i, gravito-inertial vector
  w_i = g - a_i per unit mass) and the rate of the bodies' own angular momentum H' gives,
  about ``p``: sum (c_i - p) x m_i w_i - H' = 0 (x and y). The figure is the signed distance
  from ``p`` to the nearest edge of the convex hull of the wheel contacts, positive inside,
  with that edge named. On a level floor at rest it is the distance from the point under the
  centre of mass to that edge.
* **The gravito-inertial vectors from an IMU** (``imu_transfer``): an accelerometer measures
  the specific force f = a - g at its own point. A rigid body carries it to every other point:
  w_i = -f - alpha x (r_i - r_imu) - omega x (omega x (r_i - r_imu)), with the gyro's rate omega
  and its derivative alpha. The IMU's height enters through r_imu; the base's acceleration, the
  floor's tilt, the body's lean on its springs and its ring are all inside f, omega and alpha.
  Only the IMU's acceleration and rate are used, never an orientation it may publish.
* **The pessimistic figure** (``pessimistic_margin``): the smallest figure over the bounds of
  the mass model, for each edge: every centre of mass moved toward that edge by its bound
  (lengthened along the gravito-inertial line), every mass at the end of its bound that lowers
  the figure.
* **If the base stopped now** (``stop_vectors``): the gravito-inertial vectors of a stop from
  the present motion at a stated deceleration, amplified by the suspension (quasi-static lean
  and the overshoot of a sudden change).
* **The energy reserve** (``energy_to_tip``): the energy to roll the robot over an edge, minus
  the kinetic energy of the present motion. Positive: a stop from this motion cannot roll the
  robot over that edge (sufficient only).

Accelerations the figure counts: the base's linear acceleration, the body's rotation (roll,
pitch and the turn), each arm body's own. A force on the tool is not seen.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

G = 9.80665

# Edge names by the direction of the edge's outward normal in the base frame, 45-degree sectors.
EDGE_NAMES = ('front', 'front-left', 'left', 'rear-left', 'rear', 'rear-right', 'right',
              'front-right')


@dataclass(frozen=True)
class Edge:
    """One edge of the support polygon: a point on it and its inward unit normal (x, y)."""

    point: np.ndarray
    inward: np.ndarray
    name: str
    code: int                               # index into EDGE_NAMES, + 1 (0 = none)

    @property
    def outward(self) -> np.ndarray:
        return -self.inward


def edge_name(outward: np.ndarray) -> Tuple[str, int]:
    """Return the sector name of an outward normal and its code (1..8)."""
    angle = math.degrees(math.atan2(outward[1], outward[0])) % 360.0
    index = int(((angle + 22.5) % 360.0) // 45.0)
    return EDGE_NAMES[index], index + 1


def support_polygon(contacts: Sequence[np.ndarray]) -> List[Edge]:
    """Return the edges of the convex hull of the contacts' (x, y), counter-clockwise."""
    pts = sorted({(round(float(c[0]), 12), round(float(c[1]), 12)) for c in contacts})
    if len(pts) < 3:
        raise ValueError('fewer than three distinct contacts')

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower: List[tuple] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[tuple] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError('the contacts are on one line')
    edges = []
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        e = np.array([b[0] - a[0], b[1] - a[1]])
        inward = np.array([-e[1], e[0]]) / np.linalg.norm(e)      # left of a CCW edge
        name, code = edge_name(-inward)
        edges.append(Edge(np.array(a), inward, name, code))
    return edges


def distance(p: np.ndarray, edge: Edge) -> float:
    """Return the signed distance of ``p`` (x, y) to ``edge``: positive inside."""
    return float(np.dot(p - edge.point, edge.inward))


def nearest(p: np.ndarray, edges: Sequence[Edge]) -> Tuple[float, Edge]:
    """Return the figure (smallest signed distance) and its edge."""
    best = min(edges, key=lambda e: distance(p, e))
    return distance(p, best), best


def centre_of_pressure(positions: np.ndarray, masses: np.ndarray, w: np.ndarray,
                       plane_z: float, hdot: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return the centre of pressure (x, y) on the plane z = ``plane_z``.

    ``w`` (N, 3) are the gravito-inertial vectors per unit mass (g - a); ``hdot`` the rate of
    the bodies' angular momentum about their own centres of mass (base axes).
    """
    W = masses[:, None] * w
    h = positions[:, 2] - plane_z
    wz = W[:, 2].sum()
    if not wz < 0.0:
        raise ValueError('the gravito-inertial force does not press the robot onto the floor')
    hx, hy = (0.0, 0.0) if hdot is None else (float(hdot[0]), float(hdot[1]))
    px = ((positions[:, 0] * W[:, 2] - h * W[:, 0]).sum() + hy) / wz
    py = ((positions[:, 1] * W[:, 2] - h * W[:, 1]).sum() - hx) / wz
    return np.array([px, py])


def angular_momentum_rate(inertias: np.ndarray, omega: np.ndarray,
                          alpha: np.ndarray) -> np.ndarray:
    """Return sum I_i alpha + omega x I_i omega over the bodies (base axes)."""
    total = inertias.sum(axis=0)
    return total @ alpha + np.cross(omega, total @ omega)


def imu_transfer(positions: np.ndarray, r_imu: np.ndarray, f: np.ndarray, omega: np.ndarray,
                 alpha: np.ndarray) -> np.ndarray:
    """Return each body's gravito-inertial vector from the IMU's specific force and rates."""
    dr = positions - r_imu
    return (-f[None, :] - np.cross(alpha, dr) - np.cross(omega, np.cross(omega[None, :], dr)))


def gravity_vectors(n: int, g_body: np.ndarray) -> np.ndarray:
    """Return the gravity vector (base axes) for every body: the static case."""
    return np.tile(g_body, (n, 1))


@dataclass(frozen=True)
class Figure:
    """A figure: signed distance, its edge, the centre of pressure."""

    margin: float
    edge: Edge
    cop: np.ndarray


def figure(positions, masses, w, plane_z, edges, hdot=None) -> Figure:
    """Return the figure for one set of gravito-inertial vectors."""
    p = centre_of_pressure(positions, masses, w, plane_z, hdot)
    m, e = nearest(p, edges)
    return Figure(m, e, p)


def pessimistic_margin(positions: np.ndarray, masses: np.ndarray, mass_low: np.ndarray,
                       mass_high: np.ndarray, com_bound: np.ndarray, w: np.ndarray,
                       plane_z: float, edges: Sequence[Edge],
                       hdot: Optional[np.ndarray] = None) -> Tuple[float, Edge]:
    """
    Return the smallest figure over the mass model's bounds, and its edge.

    For an edge, each body i acts on the plane at the point q_i where its gravito-inertial
    line meets it, weighted by m_i (-w_iz) > 0; the centre of pressure is the weighted mean
    (plus the angular-momentum term). Moving the centre of mass by up to b_i moves q_i toward
    the edge by up to b_i * |(n, n . w_xy / w_z)|; the masses then take, exactly, the end of
    their bound that lowers the weighted mean (a threshold found by iteration).
    """
    h = positions[:, 2] - plane_z
    wz = w[:, 2]
    if np.any(wz >= 0.0):
        raise ValueError('a gravito-inertial vector does not press onto the floor')
    q = positions[:, :2] - h[:, None] * w[:, :2] / wz[:, None]
    unit = -wz
    best: Optional[Tuple[float, Edge]] = None
    for e in edges:
        n = e.inward
        slope = (w[:, :2] @ n) / wz
        s = (q - e.point) @ n - com_bound * np.sqrt(1.0 + slope ** 2)
        # The angular-momentum term adds (h_y, -h_x) / sum(m w_z) to the centre of pressure:
        # a constant c over the weights' sum, so the figure is (sum m u s + c) / sum m u.
        c = 0.0 if hdot is None else -float(np.dot(np.array([hdot[1], -hdot[0]]), n))
        lam = float((np.sum(masses * unit * s) + c) / np.sum(masses * unit))
        for _ in range(100):
            m = np.where(s < lam, mass_high, mass_low)
            new = float((np.sum(m * unit * s) + c) / np.sum(m * unit))
            if abs(new - lam) < 1e-12:
                break
            lam = new
        if best is None or lam < best[0]:
            best = (lam, e)
    return best


def overshoot(zeta: float) -> float:
    """Return the overshoot of a damped mode's step response (0 when critically damped)."""
    if not zeta < 1.0:
        return 0.0
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))


def lean_amplification(stiffness: float, sprung_mass: float, sprung_height: float) -> float:
    """Return 1 / (1 - m g h / K), the quasi-static amplification of a load shift by lean."""
    if math.isinf(stiffness):
        return 1.0
    ratio = sprung_mass * G * sprung_height / stiffness
    if ratio >= 1.0:
        return math.inf
    return 1.0 / (1.0 - ratio)


def stop_vectors(positions: np.ndarray, g_body: np.ndarray, velocity: np.ndarray,
                 turn_rate: float, decel: float, turn_decel: float,
                 factor_x: float, factor_y: float) -> np.ndarray:
    """
    Return the gravito-inertial vectors of a stop from the present motion.

    The base decelerates at ``decel`` against its velocity (x, y, base frame) and its turn at
    ``turn_decel``; the inertial part is scaled by ``factor_x`` and ``factor_y`` (the
    suspension's amplification along pitch and roll).
    """
    n = len(positions)
    a = np.zeros((n, 3))
    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed > 1e-6:
        a[:, :2] -= decel * velocity[:2] / speed
    if abs(turn_rate) > 1e-6:
        alpha = np.array([0.0, 0.0, -math.copysign(turn_decel, turn_rate)])
        omega = np.array([0.0, 0.0, turn_rate])
        r = positions.copy()
        r[:, 2] = 0.0
        a += np.cross(alpha, r) + np.cross(omega, np.cross(omega[None, :], r))
    a[:, 0] *= factor_x
    a[:, 1] *= factor_y
    return g_body[None, :] - a


def tilt_toward(g_body: np.ndarray, outward: np.ndarray) -> float:
    """Return the floor's tilt (rad) toward an edge: positive when the edge is downhill."""
    return math.atan2(float(np.dot(g_body[:2], outward)), float(-g_body[2]))


def energy_to_tip(com_foot: np.ndarray, com_height: float, mass: float, edge: Edge,
                  tilt: float) -> float:
    """
    Return the energy (J) to lift the centre of mass over ``edge``.

    rho = sqrt(d0^2 + h^2) is the distance of the centre of mass from the edge, d0 the
    in-plane distance of its foot from the edge (geometry, not the dynamic figure); the centre
    of mass is now d0 sin(tilt) + h cos(tilt) above the edge, at most rho when over it.
    """
    d0 = distance(com_foot, edge)
    if d0 <= 0.0:
        return 0.0
    rho = math.hypot(d0, com_height)
    rise = rho - (d0 * math.sin(tilt) + com_height * math.cos(tilt))
    return max(0.0, mass * G * rise)


def kinetic_energy(mass: float, velocity: np.ndarray, turn_rate: float,
                   yaw_inertia: float) -> float:
    """Return 1/2 M v^2 + 1/2 I_z w^2."""
    linear = 0.5 * mass * float(np.dot(velocity[:2], velocity[:2]))
    return linear + 0.5 * yaw_inertia * turn_rate ** 2


def yaw_inertia(positions: np.ndarray, masses: np.ndarray, inertias: np.ndarray) -> float:
    """Return the inertia about the vertical through the centre of mass."""
    c = (masses[:, None] * positions).sum(axis=0) / masses.sum()
    r2 = (positions[:, 0] - c[0]) ** 2 + (positions[:, 1] - c[1]) ** 2
    return float((masses * r2).sum() + inertias[:, 2, 2].sum())


def edges_toward(edges: Sequence[Edge], velocity: np.ndarray) -> List[Edge]:
    """Return the edges the motion heads toward (all of them when the base is not translating)."""
    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed < 1e-3:
        return list(edges)
    v = velocity[:2] / speed
    toward = [e for e in edges if float(np.dot(e.outward, v)) > 0.0]
    return toward or list(edges)


def energy_reserve(positions: np.ndarray, masses: np.ndarray, inertias: np.ndarray,
                   plane_z: float, edges: Sequence[Edge], g_body: np.ndarray,
                   velocity: np.ndarray, turn_rate: float,
                   com_shift: float = 0.0) -> Tuple[float, Edge]:
    """
    Return the smallest energy reserve (J) over the edges the motion heads toward.

    ``com_shift`` moves the centre of mass toward each edge and up by that much (the mass
    model's bound) before the energy is taken.
    """
    total = float(masses.sum())
    c = (masses[:, None] * positions).sum(axis=0) / total
    ke = kinetic_energy(total, velocity, turn_rate, yaw_inertia(positions, masses, inertias))
    best: Optional[Tuple[float, Edge]] = None
    for e in edges_toward(edges, velocity):
        foot = c[:2] + com_shift * e.outward
        height = c[2] - plane_z + com_shift
        reserve = energy_to_tip(foot, height, total, e, tilt_toward(g_body, e.outward)) - ke
        if best is None or reserve < best[0]:
            best = (reserve, e)
    return best


def combined_com_bound(masses: np.ndarray, com_bound: np.ndarray) -> float:
    """Return the bound on the whole centre of mass from the bodies' own bounds (mass-weighted)."""
    return float((masses * com_bound).sum() / masses.sum())


def finite(*arrays) -> bool:
    """Return whether every value is finite."""
    return all(np.all(np.isfinite(np.asarray(a, dtype=float))) for a in arrays)


def as_dict(fig: Figure) -> Dict[str, object]:
    """Return a figure for a log line or a test."""
    return {'margin': fig.margin, 'edge': fig.edge.name, 'cop': [float(v) for v in fig.cop]}
