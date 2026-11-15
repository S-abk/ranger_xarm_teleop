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
Tool target to joint velocities: damped least squares with position first (G-8). No ROS.

One call of ``solve`` turns the tool target of a cycle into the joint velocity command:

1. **Error and command twist.** ``y = feedforward + gain * error``, where the error is the
   target minus the tool pose at the reference joint state (translation in metres,
   rotation as a rotation vector in radians, both in the arm base frame).
2. **Units.** The Jacobian is normalised by the limits the result must respect:
   ``J~ = diag(1/v_cap, 1/w_cap) J diag(qdot_max)``. It is dimensionless; its singular
   values are the share of the speed cap the tool reaches in that direction with the
   joints at their velocity limits. This replaces a measure that mixes metres and radians.
3. **Priority.** Position first, orientation in what position leaves free
   (singularity-robust task priority)::

       z = Jp# yp + P (Jo P)# (yo - Jo Jp# yp),   P = I - Jp# Jp

   ``#`` is the damped pseudo-inverse of its level. For this arm the tool position stays
   well conditioned through the wrist and shoulder regions, so near those the accuracy
   that is lost is orientation. Where the position level itself is near singular (the arm
   stretched out to its reach) the space it leaves free turns quickly from one posture to
   the next, and correcting orientation in it makes the joints chatter from cycle to
   cycle; there the orientation term fades out with the position level's damping,
   ``(1 - (lambda_p / max_damping)^2)``. Each level's command is first limited to its
   speed cap (normalised length 1): more would be scaled away anyway, and an oversized
   demand there flips the damped solution from one cycle to the next.
4. **Damping law,** per level, from that level's smallest singular value ``s``:
   ``lambda^2 = lambda_max^2 (1 - (s / threshold)^2)`` for ``s < threshold``, else 0, with
   ``lambda_max`` = ``max_damping`` for the position level and ``orientation_max_damping``
   for the orientation level. A command of one speed cap along the weakest direction asks
   for ``s / (s^2 + lambda^2)`` joint velocity limits: at most ``1 / damping_threshold`` = 2
   at the threshold, where damping starts, before the one scale factor below. Below the
   threshold a larger ``lambda_max`` makes it fall faster. Near the wrist singularity,
   where joints 4 and 6 nearly share an axis, that is what keeps them from turning fast
   against each other for a small tool rotation (at ``s`` = 0.3 the orientation level asks
   for 0.41 limits with ``lambda_max`` 1.0, 1.2 with 0.5).
5. **Joint position limits.** A joint may move toward a limit at most
   ``(distance - margin) / tau`` (a velocity damper), ``tau`` chosen so the bound reaches
   the joint velocity limit at ``limit_zone`` from the limit. A joint that would exceed its
   bound is fixed at it and the rest is solved again without it.
5b. **Link keep-out (G-15),** when a model is given (``keepout.py``): for every sphere
   pair of the model within its influence distance, the clearance ``d`` may fall at most
   at ``max(d - margin - 0.5 mm, 0) / keepout_time``: a row
   ``n^T J_c qdot >= -max(d - m', 0) / tau``. If the solution above breaks a row it is
   replaced by the nearest command (in tool motion, orientation weighted 0.3, plus a small
   weight on joint motion) that keeps every row and every joint position bound: a small
   convex problem solved exactly (least distance programming). Only the approach is
   removed; motion along and away from the structure keeps its speed. Rows that cannot
   all hold give zero joint velocity. The 0.5 mm inside the margin covers the second-order
   drift of a step along a curved boundary, which the linear rows do not see.
6. **One scale factor** for the whole vector, so the direction of motion is kept: every
   joint within ``velocity_fraction`` of its velocity limit, and the tool's linear and
   angular speed within the caps, for the twist ``J qdot`` at the reference and for the
   actual step of one cycle (forward kinematics), which on a curved path can be longer.
"""

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .kinematics import Chain, chain_from_urdf, forward, jacobian
from .rotations import q_conj, q_from_matrix, q_mul, q_to_rotvec

# z^4 - z^3 + gain * period has only real roots up to 27/256: the loop does not ring with
# the reference three cycles old (see ``SolverParams.gain``).
STABLE_GAIN_PERIOD = 27.0 / 256.0
LIMIT_NONE = ''
LIMIT_JOINT_POSITION = 'joint_position'
LIMIT_JOINT_VELOCITY = 'joint_velocity'
LIMIT_TOOL_SPEED = 'tool_speed'
LIMIT_LINK_KEEPOUT = 'link_keepout'
# m: the damper aims this far outside the margin, so the second-order drift of a step
# along a curved boundary (the linear rows see only the first order) stays outside it.
KEEPOUT_TARGET_INSIDE = 0.0005


@dataclass(frozen=True)
class SolverParams:
    """Solver settings. The two caps are safety limits; the rest are tuning values."""

    linear_speed_cap: float             # m/s (limits file)
    angular_speed_cap: float            # rad/s (limits file)
    period: float = 0.02                # s, the target cycle
    # 1/s, error feedback. The loop runs on Servo's commanded state, predicted one step past
    # its newest output; when the two loops slip it is a cycle or two older. With d cycles
    # of staleness z^(d+1) - z^d + gain*period has only real roots (no ringing) for
    # gain*period <= 1/4 (d = 1), 4/27 (d = 2), 27/256 (d = 3): 5 /s stays below all three.
    gain: float = 5.0
    damping_threshold: float = 0.5      # normalised singular value below which damping starts
    max_damping: float = 0.5            # normalised damping at a singularity, position level
    orientation_max_damping: float = 1.0    # the same for the orientation level
    limit_margin: float = 0.15          # rad: no motion toward a position limit closer than this
    limit_zone: float = 0.35            # rad: the damper starts slowing a joint from here
    velocity_fraction: float = 0.95     # share of each joint velocity limit that is used
    # s: the link keep-out lets a clearance d fall at most at (d - margin) / keepout_time.
    # Like the gain, period / keepout_time <= 27/256: no ringing at the boundary with the
    # reference three cycles old.
    keepout_time: float = 0.2

    def __post_init__(self) -> None:
        for name in ('linear_speed_cap', 'angular_speed_cap', 'period', 'gain',
                     'damping_threshold', 'max_damping', 'orientation_max_damping',
                     'limit_margin', 'keepout_time'):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
                raise ValueError(f'{name}: expected a finite positive number, got {value!r}')
        if not self.limit_zone > self.limit_margin:
            raise ValueError('limit_zone must be larger than limit_margin')
        if not 0.0 < self.velocity_fraction <= 1.0:
            raise ValueError('velocity_fraction: expected a value in (0, 1]')
        if self.period / self.keepout_time > STABLE_GAIN_PERIOD:
            raise ValueError(f'period / keepout_time must be at most {STABLE_GAIN_PERIOD:.4f}')
        if self.gain * self.period > STABLE_GAIN_PERIOD:
            raise ValueError(f'gain * period must be at most {STABLE_GAIN_PERIOD:.4f}: no '
                             'ringing with the reference up to three cycles old')


@dataclass(frozen=True)
class Arm:
    """The kinematic chain, the fixed tool offset and the joint limits the solver respects."""

    chain: Chain
    tip_to_tool: np.ndarray            # 4 x 4: tool frame in the chain's tip frame
    lower: np.ndarray                  # rad
    upper: np.ndarray                  # rad
    velocity: np.ndarray               # rad/s, the smallest of every source given

    @property
    def joint_names(self):
        """Return the joint names, base first (the order of every vector here)."""
        return self.chain.joint_names


def make_arm(chain: Chain, tip_to_tool: Optional[np.ndarray] = None,
             velocity_limits: Optional[Dict[str, float]] = None) -> Arm:
    """
    Return the arm of ``chain`` with ``tip_to_tool`` (identity by default).

    Joint velocity limits are the smaller of the URDF value and ``velocity_limits[name]``
    (for example MoveIt's ``joint_limits.yaml``); a joint without any finite limit is refused.
    """
    joints = chain.joints
    velocity = []
    for joint in joints:
        limit = joint.velocity
        if velocity_limits and joint.name in velocity_limits:
            limit = min(limit, float(velocity_limits[joint.name]))
        if not (math.isfinite(limit) and limit > 0.0):
            raise ValueError(f'{joint.name}: no finite velocity limit')
        velocity.append(limit)
    tool = np.eye(4) if tip_to_tool is None else np.asarray(tip_to_tool, dtype=float)
    return Arm(chain, tool, np.array([j.lower for j in joints]),
               np.array([j.upper for j in joints]), np.array(velocity))


def arm_from_urdf(urdf: str, base: str, tip: str, tip_to_tool: Optional[np.ndarray] = None,
                  velocity_limits: Optional[Dict[str, float]] = None) -> Arm:
    """Return the arm of the chain ``base`` -> ``tip`` in URDF text."""
    return make_arm(chain_from_urdf(urdf, base, tip), tip_to_tool, velocity_limits)


def tool_pose(arm: Arm, q: Sequence[float]) -> np.ndarray:
    """Return the 4 x 4 tool pose in the chain's base frame."""
    tip, _, _ = forward(arm.chain, q)
    return tip @ arm.tip_to_tool


def tool_jacobian(arm: Arm, q: Sequence[float]) -> np.ndarray:
    """Return the 6 x n Jacobian of the tool point (linear rows, then angular), base frame."""
    tip, _, _ = forward(arm.chain, q)
    jac = jacobian(arm.chain, q)
    lever = tip[:3, :3] @ arm.tip_to_tool[:3, 3]        # tip origin -> tool point, base frame
    out = jac.copy()
    out[:3] = jac[:3] + np.cross(jac[3:].T, lever).T    # v_tool = v_tip + w x lever
    return out


def pose_error(target: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(translation, rotation vector)`` taking ``current`` to ``target``, base frame."""
    dp = target[:3, 3] - current[:3, 3]
    turn = q_mul(q_from_matrix(target[:3, :3]), q_conj(q_from_matrix(current[:3, :3])))
    return dp, np.array(q_to_rotvec(turn))


def damping_for(sigma: float, params: SolverParams,
                maximum: Optional[float] = None) -> float:
    """
    Return the damping ``lambda`` for a level whose smallest singular value is ``sigma``.

    ``maximum`` is the level's damping at a singularity (``params.max_damping`` if None).
    """
    if sigma >= params.damping_threshold:
        return 0.0
    ratio = max(0.0, sigma) / params.damping_threshold
    top = params.max_damping if maximum is None else maximum
    return top * math.sqrt(1.0 - ratio * ratio)


def _damped_pinv(a: np.ndarray, lam: float) -> Tuple[np.ndarray, float]:
    """Return the damped pseudo-inverse of ``a`` and its smallest singular value."""
    rows, cols = a.shape
    if cols == 0:
        return np.zeros((0, rows)), 0.0
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    smallest = float(s[-1]) if len(s) >= rows else 0.0
    if lam > 0.0:
        inv = s / (s * s + lam * lam)
    else:
        inv = np.where(s > 1e-12, 1.0 / np.maximum(s, 1e-12), 0.0)
    return (vt.T * inv) @ u.T, smallest


def _smallest_rank(a: np.ndarray, rank: int) -> float:
    """Return the ``rank``-th singular value of ``a`` (0 when ``a`` has fewer)."""
    if a.size == 0:
        return 0.0
    s = np.linalg.svd(a, compute_uv=False)
    return float(s[rank - 1]) if len(s) >= rank else 0.0


@dataclass(frozen=True)
class Levels:
    """Conditioning of the two levels in normalised units, and the damping each gets."""

    position: float             # smallest singular value of the position rows
    orientation: float          # third singular value of the orientation rows given position
    position_damping: float
    orientation_damping: float

    @property
    def conditioning(self) -> float:
        """Return the weaker of the two levels (>= 1: unrestricted, 0: singular)."""
        return min(self.position, self.orientation)

    @property
    def damping(self) -> float:
        """Return the larger damping of the two levels."""
        return max(self.position_damping, self.orientation_damping)


def _priority(jn: np.ndarray, yn: np.ndarray, params: SolverParams):
    """Return ``(z, levels)`` of the two-level damped solution for normalised ``jn``, ``yn``."""
    cols = jn.shape[1]
    jp, jo = jn[:3], jn[3:]
    sp = _smallest_rank(jp, 3)
    lam_p = damping_for(sp, params)
    jp_inv, _ = _damped_pinv(jp, lam_p)
    zp = jp_inv @ yn[:3]
    projector = np.eye(cols) - jp_inv @ jp
    jop = jo @ projector
    so = _smallest_rank(jop, 3)
    lam_o = damping_for(so, params, params.orientation_max_damping)
    jop_inv, _ = _damped_pinv(jop, lam_o)
    fade = 1.0 - (lam_p / params.max_damping) ** 2
    z = zp + fade * (projector @ (jop_inv @ (yn[3:] - jo @ zp)))
    return z, Levels(sp, so, lam_p, lam_o)


def levels(arm: Arm, q: Sequence[float], params: SolverParams) -> Levels:
    """Return the conditioning of both levels at ``q`` (all joints free)."""
    jn = _normalised(arm, tool_jacobian(arm, q), params)
    return _priority(jn, np.zeros(6), params)[1]


def _normalised(arm: Arm, jac: np.ndarray, params: SolverParams) -> np.ndarray:
    rows = np.array([1.0 / params.linear_speed_cap] * 3 + [1.0 / params.angular_speed_cap] * 3)
    return rows[:, None] * jac * arm.velocity[None, :]


def position_bounds(arm: Arm, q: Sequence[float],
                    params: SolverParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the lowest and highest joint velocity (rad/s) the position limits allow at ``q``.

    Toward a limit: at most ``(distance - margin) / tau``, never below zero (no joint is
    pushed away by the solver). These are position bounds only; they exceed the velocity
    limit outside ``limit_zone``. Velocity is limited afterwards by one factor for the
    whole vector, so a joint that is merely fast is never held on its own.
    """
    q = np.asarray(q, dtype=float)
    # At least one cycle, so one step never covers more than the remaining distance.
    tau = np.maximum((params.limit_zone - params.limit_margin) / arm.velocity, params.period)
    with np.errstate(invalid='ignore'):
        up = np.maximum((arm.upper - q - params.limit_margin) / tau, 0.0)
        down = np.maximum((q - arm.lower - params.limit_margin) / tau, 0.0)
    up = np.where(np.isfinite(arm.upper), up, np.inf)
    down = np.where(np.isfinite(arm.lower), down, np.inf)
    return -down, up


@dataclass(frozen=True)
class Solution:
    """The joint velocity command of one cycle and what shaped it."""

    qdot: np.ndarray                    # rad/s, joint order of the arm
    conditioning: float                 # normalised (see ``Levels``)
    damping: float                      # normalised lambda, the larger of the two levels
    error_translation: float            # m: target vs the tool at the reference state
    error_rotation: float               # rad
    linear_speed: float                 # m/s: tool speed of one cycle's step of ``qdot``
    angular_speed: float                # rad/s
    scale: float                        # the one factor applied to the whole vector
    limit: str                          # LIMIT_*: what limited the command, '' if nothing
    fixed: Tuple[str, ...] = ()         # joints held at a position bound
    # Link keep-out (G-15). Clearance: the model's smallest clearance at the reference (m;
    # inf without a model; a lower bound when nothing is within the model's influence).
    keepout_clearance: float = math.inf
    keepout_pair: Tuple[str, str] = ('', '')
    keepout_bound: bool = False         # the keep-out shaped this command
    # m/s, base frame: tool velocity the position-type constraints (joint position bounds,
    # keep-out) took out of the command, before the speed scaling. Zero when none bound.
    blocked: np.ndarray = None

    @property
    def damped(self) -> bool:
        """Return whether either level was damped, which reduces accuracy."""
        return self.damping > 0.0


def solve(arm: Arm, q: Sequence[float], target: np.ndarray, params: SolverParams,
          feedforward: Optional[Sequence[float]] = None, keepout=None,
          extra: Optional[Mapping[str, float]] = None) -> Solution:
    """
    Return the joint velocities that drive the tool from reference ``q`` toward ``target``.

    ``target`` is a 4 x 4 tool pose in the base frame. ``feedforward`` is the target's own
    twist ``(vx, vy, vz, wx, wy, wz)`` in m/s and rad/s, base frame (zero by default).
    ``keepout`` is a link keep-out model (``keepout.Model``) on the same joints, ``extra``
    the measured values of the joints it needs that are not arm joints.
    """
    q = np.asarray(q, dtype=float)
    target = np.asarray(target, dtype=float)
    ff = np.zeros(6) if feedforward is None else np.asarray(feedforward, dtype=float)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(target)) and np.all(np.isfinite(ff))):
        raise ValueError('reference, target or feedforward is not finite')
    current = tool_pose(arm, q)
    dp, dr = pose_error(target, current)
    y = ff + params.gain * np.concatenate([dp, dr])
    jac = tool_jacobian(arm, q)
    jn = _normalised(arm, jac, params)
    yn = y / np.array([params.linear_speed_cap] * 3 + [params.angular_speed_cap] * 3)
    # Each level's command is limited to its speed cap (normalised length 1). The tool's
    # direction can change by this (linear and angular shrink by different factors); it is
    # reported as the speed cap having limited the command.
    capped = False
    for level in (slice(0, 3), slice(3, 6)):
        size = float(np.linalg.norm(yn[level]))
        if size > 1.0:
            yn[level] /= size
            capped = True
    low, high = position_bounds(arm, q, params)
    low_n, high_n = low / arm.velocity, high / arm.velocity

    rows = None
    clearance, pair = math.inf, ('', '')
    if keepout is not None:
        if list(keepout.joint_names) != list(arm.joint_names):
            raise ValueError("the keep-out model is not on the arm's joints")
        rows = keepout.evaluate(q, extra)
        clearance, pair = rows.clearance, rows.nearest
    result = _constrained(arm, jn, yn, low_n, high_n, rows, keepout, params)
    z, z_first, first_levels, fixed, bound, infeasible = result
    if infeasible:
        qdot, scale, limit, linear, angular = np.zeros_like(q), 0.0, LIMIT_LINK_KEEPOUT, 0.0, 0.0
    else:
        qdot, scale, limit, linear, angular = _scaled(
            arm, q, jac, current, z, fixed, bound, capped, params)
    blocked = jac[:3] @ ((z_first - z) * arm.velocity) if (fixed or bound) else np.zeros(3)
    return Solution(qdot=qdot, conditioning=first_levels.conditioning,
                    damping=first_levels.damping,
                    error_translation=float(np.linalg.norm(dp)),
                    error_rotation=float(np.linalg.norm(dr)),
                    linear_speed=linear, angular_speed=angular,
                    scale=scale, limit=limit, fixed=fixed,
                    keepout_clearance=clearance, keepout_pair=pair, keepout_bound=bound,
                    blocked=blocked)


def _rows_normalised(arm: Arm, rows, keepout, params: SolverParams):
    """Return ``(A, L)``: the keep-out rows as ``A z >= L`` in normalised joint units."""
    a = rows.grad * arm.velocity[None, :]
    target = keepout.margin + KEEPOUT_TARGET_INSIDE
    lower = -np.maximum(rows.distance - target, 0.0) / params.keepout_time
    return a, lower


def _constrained(arm: Arm, jn: np.ndarray, yn: np.ndarray, low_n: np.ndarray,
                 high_n: np.ndarray, rows, keepout, params: SolverParams):
    """
    Return the solution under the position-type constraints (joint bounds, keep-out rows).

    ``(z, z_first, first_levels, fixed, bound, infeasible)``: ``z_first`` is the first pass
    (no constraint), ``bound`` whether the keep-out changed the solution, ``infeasible``
    whether the rows cannot all hold (the command is then zero).

    The joint bounds are handled as before (a joint over its bound is fixed and the rest
    solved again). If that solution breaks a keep-out row, it is corrected by the smallest
    change in tool position (orientation weighted ``KEEPOUT_ORIENTATION_WEIGHT``) that
    satisfies every row and every joint bound: a small convex problem solved exactly.
    """
    n = jn.shape[1]
    free = np.ones(n, dtype=bool)
    z = np.zeros(n)
    first_levels, z_first = None, None
    for _ in range(n + 1):
        rhs = yn - jn[:, ~free] @ z[~free]
        sub, lv = _priority(jn[:, free], rhs, params)
        if first_levels is None:
            first_levels = lv
        z[free] = sub
        if z_first is None:
            z_first = z.copy()
        over = free & ((z > high_n + 1e-12) | (z < low_n - 1e-12))
        if not over.any():
            break
        z[over] = np.clip(z[over], low_n[over], high_n[over])
        free &= ~over
    z = np.clip(z, low_n, high_n)        # only reached if every pass fixed a joint
    fixed = tuple(name for name, f in zip(arm.joint_names, free) if not f)
    if rows is None or len(rows.distance) == 0:
        return z, z_first, first_levels, fixed, False, False
    a, lower = _rows_normalised(arm, rows, keepout, params)
    if not np.any(lower - a @ z > 1e-9 * (1.0 + np.linalg.norm(a, axis=1))):
        return z, z_first, first_levels, fixed, False, False
    # min (z - z*)^T H (z - z*)  s.t.  a z >= lower,  low <= z <= high.
    weights = np.array([1.0] * 3 + [KEEPOUT_ORIENTATION_WEIGHT] * 3)
    h = (jn * weights[:, None] ** 2).T @ jn + KEEPOUT_REGULARISATION * np.eye(n)
    g = np.vstack([a, np.eye(n), -np.eye(n)])
    bound = np.concatenate([lower, low_n, -high_n])
    finite = np.isfinite(bound)
    solved = _least_change(h, z, g[finite], bound[finite])
    if solved is None:
        return np.zeros(n), z_first, first_levels, fixed, True, True
    return solved, z_first, first_levels, fixed, True, False


KEEPOUT_ORIENTATION_WEIGHT = 0.3     # orientation yields to position in the correction
KEEPOUT_REGULARISATION = 0.05       # weight of joint motion itself (normalised): keeps the
#                                      correction from trading large joint swings for tool motion


def _least_change(h: np.ndarray, start: np.ndarray, g: np.ndarray,
                  bound: np.ndarray) -> Optional[np.ndarray]:
    """
    Return ``z`` minimising ``(z - start)^T h (z - start)`` with ``g z >= bound``, or None.

    None when the constraints contradict each other. Least distance programming
    (Lawson and Hanson): with ``h = R^T R`` and ``x = R (z - start)`` it is the shortest
    ``x`` with ``g R^-1 x >= bound - g start``, found by non-negative least squares.
    """
    r = np.linalg.cholesky(h).T
    rinv = np.linalg.inv(r)
    gg = g @ rinv
    hh = bound - g @ start
    n = gg.shape[1]
    e = np.vstack([gg.T, hh[None, :]])
    f = np.zeros(n + 1)
    f[n] = 1.0
    u = _nnls(e, f)
    residual = e @ u - f
    if np.linalg.norm(residual) < 1e-10 or abs(residual[n]) < 1e-12:
        return None
    x = -residual[:n] / residual[n]
    z = start + rinv @ x
    if np.any(g @ z < bound - 1e-7 * (1.0 + np.abs(bound))):
        return None
    return z


def _nnls(e: np.ndarray, f: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Return ``u >= 0`` minimising ``|e u - f|`` (Lawson and Hanson's active set)."""
    m = e.shape[1]
    u = np.zeros(m)
    passive = np.zeros(m, dtype=bool)
    w = e.T @ (f - e @ u)
    tol = 10.0 * np.finfo(float).eps * np.abs(e).sum(axis=0).max() * max(e.shape)
    for _ in range(max_iter):
        if passive.all() or not np.any(w[~passive] > tol):
            break
        j = int(np.argmax(np.where(passive, -np.inf, w)))
        passive[j] = True
        while True:
            s = np.zeros(m)
            s[passive] = np.linalg.lstsq(e[:, passive], f, rcond=None)[0]
            if np.all(s[passive] > tol):
                break
            mask = passive & (s <= tol)
            alpha = np.min(u[mask] / np.maximum(u[mask] - s[mask], 1e-300))
            u = u + alpha * (s - u)
            passive &= u > tol
            u[~passive] = 0.0
        u = s
        w = e.T @ (f - e @ u)
    return u


def _scaled(arm: Arm, q: np.ndarray, jac: np.ndarray, current: np.ndarray, z: np.ndarray,
            fixed, bound: bool, capped: bool, params: SolverParams):
    """Return ``(qdot, scale, limit, linear, angular)`` after the one scale factor."""
    limit = LIMIT_JOINT_POSITION if fixed else (
        LIMIT_LINK_KEEPOUT if bound else (LIMIT_TOOL_SPEED if capped else LIMIT_NONE))
    qdot = z * arm.velocity
    scale = 1.0
    joint_scale = float(np.min(np.where(np.abs(qdot) > 1e-12,
                                        arm.velocity * params.velocity_fraction
                                        / np.maximum(np.abs(qdot), 1e-12), np.inf)))
    twist = jac @ qdot
    linear, angular = float(np.linalg.norm(twist[:3])), float(np.linalg.norm(twist[3:]))
    tool_scale = min(params.linear_speed_cap / linear if linear > 1e-12 else math.inf,
                     params.angular_speed_cap / angular if angular > 1e-12 else math.inf)
    if joint_scale < 1.0 and joint_scale <= tool_scale:
        scale, limit = joint_scale, LIMIT_JOINT_VELOCITY
    elif tool_scale < 1.0:
        scale, limit = tool_scale, LIMIT_TOOL_SPEED
    qdot = qdot * scale
    # The caps hold for the twist J qdot; the step of one cycle follows the arm's curved
    # path, which can be slightly longer. Check the step itself and scale once more.
    linear, angular = _step_speeds(arm, q, qdot, current, params)
    over = max(linear / params.linear_speed_cap, angular / params.angular_speed_cap)
    if over > 1.0:
        qdot = qdot / over
        scale, limit = scale / over, LIMIT_TOOL_SPEED
        linear, angular = linear / over, angular / over
    return qdot, scale, limit, linear, angular


def _step_speeds(arm: Arm, q: np.ndarray, qdot: np.ndarray, current: np.ndarray,
                 params: SolverParams) -> Tuple[float, float]:
    after = tool_pose(arm, q + qdot * params.period)
    step_p, step_r = pose_error(after, current)
    return (float(np.linalg.norm(step_p)) / params.period,
            float(np.linalg.norm(step_r)) / params.period)


def twist_between(previous: np.ndarray, current: np.ndarray, period: float) -> np.ndarray:
    """Return the twist ``(v, w)`` that takes pose ``previous`` to ``current`` in ``period``."""
    dp, dr = pose_error(current, previous)
    return np.concatenate([dp, dr]) / period
