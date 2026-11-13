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
   ``(1 - (lambda_p / max_damping)^2)``.
4. **Damping law,** per level, from that level's smallest singular value ``s``:
   ``lambda^2 = max_damping^2 (1 - (s / threshold)^2)`` for ``s < threshold``, else 0.
   With ``max_damping = 0.5`` a command of one speed cap along the weakest direction asks
   for at most ``1 / damping_threshold`` = 2 joint velocity limits (at the threshold, where
   damping starts; below it ``s / (s^2 + lambda^2)`` is smaller), before the one scale
   factor below.
5. **Joint position limits.** A joint may move toward a limit at most
   ``(distance - margin) / tau`` (a velocity damper), ``tau`` chosen so the bound reaches
   the joint velocity limit at ``limit_zone`` from the limit. A joint that would exceed its
   bound is fixed at it and the rest is solved again without it.
6. **One scale factor** for the whole vector, so the direction of motion is kept: every
   joint within ``velocity_fraction`` of its velocity limit, and the tool's linear and
   angular speed within the caps, for the twist ``J qdot`` at the reference and for the
   actual step of one cycle (forward kinematics), which on a curved path can be longer.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

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
    max_damping: float = 0.5            # normalised damping at a singularity
    limit_margin: float = 0.15          # rad: no motion toward a position limit closer than this
    limit_zone: float = 0.35            # rad: the damper starts slowing a joint from here
    velocity_fraction: float = 0.95     # share of each joint velocity limit that is used

    def __post_init__(self) -> None:
        for name in ('linear_speed_cap', 'angular_speed_cap', 'period', 'gain',
                     'damping_threshold', 'max_damping', 'limit_margin'):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
                raise ValueError(f'{name}: expected a finite positive number, got {value!r}')
        if not self.limit_zone > self.limit_margin:
            raise ValueError('limit_zone must be larger than limit_margin')
        if not 0.0 < self.velocity_fraction <= 1.0:
            raise ValueError('velocity_fraction: expected a value in (0, 1]')
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


def damping_for(sigma: float, params: SolverParams) -> float:
    """Return the damping ``lambda`` for a level whose smallest singular value is ``sigma``."""
    if sigma >= params.damping_threshold:
        return 0.0
    ratio = max(0.0, sigma) / params.damping_threshold
    return params.max_damping * math.sqrt(1.0 - ratio * ratio)


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
    lam_o = damping_for(so, params)
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

    @property
    def damped(self) -> bool:
        """Return whether either level was damped, which reduces accuracy."""
        return self.damping > 0.0


def solve(arm: Arm, q: Sequence[float], target: np.ndarray, params: SolverParams,
          feedforward: Optional[Sequence[float]] = None) -> Solution:
    """
    Return the joint velocities that drive the tool from reference ``q`` toward ``target``.

    ``target`` is a 4 x 4 tool pose in the base frame. ``feedforward`` is the target's own
    twist ``(vx, vy, vz, wx, wy, wz)`` in m/s and rad/s, base frame (zero by default).
    """
    q = np.asarray(q, dtype=float)
    current = tool_pose(arm, q)
    dp, dr = pose_error(target, current)
    ff = np.zeros(6) if feedforward is None else np.asarray(feedforward, dtype=float)
    y = ff + params.gain * np.concatenate([dp, dr])
    jac = tool_jacobian(arm, q)
    jn = _normalised(arm, jac, params)
    yn = y / np.array([params.linear_speed_cap] * 3 + [params.angular_speed_cap] * 3)
    low, high = position_bounds(arm, q, params)
    low_n, high_n = low / arm.velocity, high / arm.velocity

    n = len(q)
    free = np.ones(n, dtype=bool)
    z = np.zeros(n)
    first_levels = None
    for _ in range(n + 1):
        rhs = yn - jn[:, ~free] @ z[~free]
        sub, lv = _priority(jn[:, free], rhs, params)
        if first_levels is None:
            first_levels = lv
        z[free] = sub
        over = free & ((z > high_n + 1e-12) | (z < low_n - 1e-12))
        if not over.any():
            break
        z[over] = np.clip(z[over], low_n[over], high_n[over])
        free &= ~over
    z = np.clip(z, low_n, high_n)        # only reached if every pass fixed a joint
    fixed = tuple(name for name, f in zip(arm.joint_names, free) if not f)
    limit = LIMIT_JOINT_POSITION if fixed else LIMIT_NONE

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
    after = tool_pose(arm, q + qdot * params.period)
    step_p, step_r = pose_error(after, current)
    linear = float(np.linalg.norm(step_p)) / params.period
    angular = float(np.linalg.norm(step_r)) / params.period
    over = max(linear / params.linear_speed_cap, angular / params.angular_speed_cap)
    if over > 1.0:
        qdot = qdot / over
        scale, limit = scale / over, LIMIT_TOOL_SPEED
        linear, angular = linear / over, angular / over
    return Solution(qdot=qdot, conditioning=first_levels.conditioning,
                    damping=first_levels.damping,
                    error_translation=float(np.linalg.norm(dp)),
                    error_rotation=float(np.linalg.norm(dr)),
                    linear_speed=linear, angular_speed=angular,
                    scale=scale, limit=limit, fixed=fixed)


def twist_between(previous: np.ndarray, current: np.ndarray, period: float) -> np.ndarray:
    """Return the twist ``(v, w)`` that takes pose ``previous`` to ``current`` in ``period``."""
    dp, dr = pose_error(current, previous)
    return np.concatenate([dp, dr]) / period
