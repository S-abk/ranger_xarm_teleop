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
Compare the joint-space solver with MoveIt Servo's singularity rule, offline (numpy, no ROS).

Three views, all on the same arm and ready posture:

* ``cube``: straight tool motions at fixed orientation to every point of a cube of tool
  positions, the target moving at the speed caps and then held. The old rule is the model
  in ``workspace`` (singularity scaling and halt, joint velocity limit); the new one is
  ``solver.solve``. Both are kinematic: the arm is where the last command put it.
* ``sessions``: the session model of ``workspace.wander`` (chained one-second motions of
  random direction, kept near where the session started), driven by the solver.
* ``replay``: the recorded targets of each engagement of a recorded session, through the
  solver with the delayed arm model of ``teleop_model`` (without the gate).
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import kinematics, workspace
from .rotations import q_from_axis_angle, q_to_matrix
from .solver import (Arm, make_arm, pose_error, Solution, solve, SolverParams, tool_pose,
                     twist_between)

STILL_RAD_S = 0.05                  # a joint faster than this at the end of a hold is moving
FAST_RAD_S = 0.5                    # a joint that is this fast in both directions is reversing
STILL_WINDOW_S = 0.5                # the end of a hold, in seconds
REACHED_M = 0.005                   # a motion "reached" its target within 5 mm ...
REACHED_RAD = math.radians(2.0)     # ... and 2 deg
LIMIT_MARGIN = workspace.LIMIT_MARGIN_RAD


def arm_from_description(urdf: str, joint_velocity: float = 2.0) -> Arm:
    """Return the arm of a robot description with every joint limited to ``joint_velocity``."""
    chain = kinematics.chain_from_urdf(urdf)
    return make_arm(chain, velocity_limits={name: joint_velocity for name in chain.joint_names})


# ---- the loop: solver against a delayed arm ----

def reference_index(count: int, stale_ticks: int) -> int:
    """
    Return which of ``count`` stored joint states the solver uses as its reference.

    The states are the commanded state at the end of each cycle, oldest first. One tick of
    staleness is the newest state; the node's realistic case is three (two cycles old).
    """
    return max(0, count - 1 - (max(stale_ticks, 1) - 1))


def measured_index(count: int, delay_ticks: int) -> int:
    """Return which stored joint state the measured arm is: ``delay_ticks`` cycles old."""
    return max(0, count - 1 - delay_ticks)


@dataclass(frozen=True)
class Track:
    """What the solver did over a series of targets (one row per cycle)."""

    qdot: np.ndarray                    # cycles x joints, rad/s
    q: np.ndarray                       # (cycles + 1) x joints: commanded state, start first
    solver_error: np.ndarray            # cycles x 2: target vs the reference, m and rad
    tool_error: np.ndarray              # cycles x 2: target vs the measured tool, m and rad
    conditioning: np.ndarray            # cycles
    damped: np.ndarray                  # cycles, bool
    fixed: np.ndarray                   # cycles, bool: a joint was held at a position bound
    limited: Tuple[str, ...]            # the solver's limit of each cycle ('' if none)

    @property
    def peak_joint_speed(self) -> float:
        """Return the largest joint speed commanded, rad/s."""
        return float(np.abs(self.qdot).max()) if self.qdot.size else 0.0


def track(arm: Arm, params: SolverParams, q0: Sequence[float], targets: Sequence[np.ndarray],
          feedforward: Optional[Sequence[Optional[Sequence[float]]]] = None,
          delay_ticks: int = 0, stale_ticks: int = 1) -> Track:
    """
    Run the solver over ``targets`` (4 x 4 tool poses, one per cycle) from ``q0``.

    The arm integrates each command exactly (Servo's own commanded state); the measured arm
    is that state ``delay_ticks`` cycles old. The first cycle's reference is the measured
    state (the engagement is seeded from the measurement); later ones are ``stale_ticks``
    old. With ``delay_ticks=0`` and ``stale_ticks=1`` this is plain kinematic integration.
    This is ``teleop_model.run`` without the gate.
    """
    period = params.period
    commanded = np.array(q0, dtype=float)
    history = [commanded.copy()]
    qdots, solver_error, tool_error, conditioning = [], [], [], []
    damped, fixed, limited = [], [], []
    for k, target in enumerate(targets):
        measured = history[measured_index(len(history), delay_ticks)]
        reference = measured if k == 0 else history[reference_index(len(history), stale_ticks)]
        ff = None if feedforward is None else feedforward[k]
        solution: Solution = solve(arm, reference, target, params, ff)
        dp, dr = pose_error(target, tool_pose(arm, measured))
        commanded = commanded + solution.qdot * period
        history.append(commanded.copy())
        qdots.append(solution.qdot)
        solver_error.append((solution.error_translation, solution.error_rotation))
        tool_error.append((float(np.linalg.norm(dp)), float(np.linalg.norm(dr))))
        conditioning.append(solution.conditioning)
        damped.append(solution.damped)
        fixed.append(bool(solution.fixed))
        limited.append(solution.limit)
    n = len(q0)
    return Track(np.array(qdots).reshape(-1, n), np.array(history),
                 np.array(solver_error).reshape(-1, 2), np.array(tool_error).reshape(-1, 2),
                 np.array(conditioning), np.array(damped, dtype=bool),
                 np.array(fixed, dtype=bool), tuple(limited))


# ---- targets ----

def pose_at(origin: np.ndarray, offset: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    """Return ``origin`` moved by ``offset`` and turned by the rotation vector ``rotation``."""
    out = origin.copy()
    out[:3, 3] = origin[:3, 3] + np.asarray(offset, dtype=float)
    angle = float(np.linalg.norm(rotation))
    if angle > 1e-12:
        turn = q_to_matrix(q_from_axis_angle(np.asarray(rotation) / angle, angle))
        out[:3, :3] = turn @ origin[:3, :3]
    return out


def twists(targets: Sequence[np.ndarray], first: np.ndarray, period: float) -> List[np.ndarray]:
    """Return the twist of each target relative to the one before (``first`` precedes all)."""
    out, previous = [], first
    for target in targets:
        out.append(twist_between(previous, target, period))
        previous = target
    return out


def straight_targets(origin: np.ndarray, offset: Sequence[float], speed: float, hold: float,
                     period: float) -> List[np.ndarray]:
    """
    Return targets that move along ``offset`` from ``origin`` at ``speed``, then hold ``hold`` s.

    The orientation stays that of ``origin``.
    """
    offset = np.asarray(offset, dtype=float)
    length = float(np.linalg.norm(offset))
    moving = int(math.ceil(length / speed / period - 1e-9)) if length > 0 else 0
    holding = int(round(hold / period))
    out = []
    for k in range(1, moving + holding + 1):
        travelled = min(speed * k * period, length)
        along = offset / length * travelled if length > 0 else offset
        out.append(pose_at(origin, along, (0, 0, 0)))
    return out


# ---- 1. the cube ----

def cube_offsets(extent: float = 0.30, spacing: float = 0.10) -> List[np.ndarray]:
    """Return the tool offsets of the cube, in the order of ``workspace.grid``."""
    count = int(round(extent / spacing))
    values = [i * spacing for i in range(-count, count + 1)]
    return [np.array([dx, dy, dz]) for dx in values for dy in values for dz in values]


@dataclass(frozen=True)
class Run:
    """One motion: how it went, for either method."""

    reached: bool                       # ended within 5 mm and 2 deg of its target
    final_error: Tuple[float, float]    # m, rad
    max_error: Tuple[float, float]      # m, rad: target vs tool over the whole motion
    peak_joint_speed: float             # rad/s
    affected: bool                      # damping engaged (solver) / the rule slowed or halted
    halted: bool                        # the old rule's hard stop
    joint_limit: bool                   # solver: damper holding a joint at a bound; rule: stop
    min_margin: float = math.inf        # rad: closest any joint came to a position bound
    moving_at_end: Optional[bool] = None    # solver: a joint still above 0.05 rad/s at the end
    reversed_fast: Optional[bool] = None    # solver: a joint above 0.5 rad/s in both directions


def moving_at_end(qdot: np.ndarray, period: float) -> bool:
    """Return whether any joint is faster than 0.05 rad/s in the last 0.5 s of the run."""
    window = max(1, int(round(STILL_WINDOW_S / period)))
    return bool(qdot.size and np.abs(qdot[-window:]).max() > STILL_RAD_S)


def reversed_fast(qdot: np.ndarray) -> bool:
    """Return whether any joint moved faster than 0.5 rad/s in both directions."""
    for v in qdot.T:
        fast = v[np.abs(v) > FAST_RAD_S]
        if len(fast) and (fast > 0).any() and (fast < 0).any():
            return True
    return False


def _margin(arm: Arm, q: np.ndarray) -> float:
    return float(np.minimum(q - arm.lower, arm.upper - q).min())


def _run_of(arm: Arm, arm_track: Track, final: Tuple[float, float], period: float) -> Run:
    err = arm_track.tool_error
    return Run(reached=final[0] < REACHED_M and final[1] < REACHED_RAD, final_error=final,
               max_error=(float(err[:, 0].max()), float(err[:, 1].max())) if len(err)
               else (0.0, 0.0),
               peak_joint_speed=arm_track.peak_joint_speed,
               affected=bool(arm_track.damped.any()), halted=False,
               joint_limit=bool(arm_track.fixed.any()),
               min_margin=min(_margin(arm, q) for q in arm_track.q),
               moving_at_end=moving_at_end(arm_track.qdot, period),
               reversed_fast=reversed_fast(arm_track.qdot))


def solver_motion(arm: Arm, params: SolverParams, q0: Sequence[float], targets, ff,
                  delay_ticks: int = 0, stale_ticks: int = 1) -> Run:
    """Run the solver over ``targets`` and judge the motion by where the tool ends."""
    result = track(arm, params, q0, targets, ff, delay_ticks, stale_ticks)
    final_pose = tool_pose(arm, result.q[-1])
    dp, dr = pose_error(targets[-1], final_pose)
    return _run_of(arm, result, (float(np.linalg.norm(dp)), float(np.linalg.norm(dr))),
                   params.period)


def rule_motion(chain: kinematics.Chain, q0: Sequence[float], targets: Sequence[np.ndarray],
                period: float, linear_cap: float, angular_cap: float, lower: float = 17.0,
                hard_stop: float = 30.0, joint_limit: float = 2.0) -> Run:
    """
    Follow ``targets`` as MoveIt Servo's pose tracking would with its singularity rule.

    Each cycle the step toward the target is limited to the speed caps, scaled by
    ``workspace.servo_scale`` (a hard stop ends the motion: nothing moves again), then to
    the joint velocity limit; a joint within the Servo margin of its bound ends it too.
    One cycle of lag is part of the model: the tool steps to where the target was.
    """
    q = np.array(q0, dtype=float)
    max_p = max_r = 0.0
    peak = 0.0
    slowed = halted = at_limit = False
    closest = math.inf
    for target in targets:
        tip, _, _ = kinematics.forward(chain, q)
        dp, dr = pose_error(target, tip)
        max_p, max_r = max(max_p, float(np.linalg.norm(dp))), max(max_r, float(np.linalg.norm(dr)))
        if halted or at_limit:
            continue
        step_p = dp if np.linalg.norm(dp) <= linear_cap * period \
            else dp / np.linalg.norm(dp) * linear_cap * period
        step_r = dr if np.linalg.norm(dr) <= angular_cap * period \
            else dr / np.linalg.norm(dr) * angular_cap * period
        delta = np.concatenate([step_p, step_r])
        scale, status = workspace.servo_scale(chain, q, delta, lower, hard_stop)
        if status == workspace.HALT:
            halted = True
            continue
        slowed = slowed or scale < 0.999
        step = np.linalg.pinv(kinematics.jacobian(chain, q)) @ delta * scale
        speed = float(np.max(np.abs(step))) / period
        if speed > joint_limit:
            step *= joint_limit / speed
            speed = joint_limit
        q = q + step
        peak = max(peak, speed)
        closest = min(closest, kinematics.limit_margin(chain, q)[1])
        if closest < LIMIT_MARGIN:
            at_limit = True
    tip, _, _ = kinematics.forward(chain, q)
    dp, dr = pose_error(targets[-1], tip)
    final = (float(np.linalg.norm(dp)), float(np.linalg.norm(dr)))
    return Run(reached=final[0] < REACHED_M and final[1] < REACHED_RAD, final_error=final,
               max_error=(max_p, max_r), peak_joint_speed=peak, affected=slowed or halted,
               halted=halted, joint_limit=at_limit, min_margin=closest)


def cube(arm: Arm, params: SolverParams, start: Sequence[float], offsets: Sequence[np.ndarray],
         hold: float = 1.0, lower: float = 17.0, hard_stop: float = 30.0
         ) -> Dict[str, List[Run]]:
    """Run the old rule and the solver to every offset: ``{'rule': runs, 'solver': runs}``."""
    origin = tool_pose(arm, start)
    runs: Dict[str, List[Run]] = {'rule': [], 'solver': []}
    for offset in offsets:
        if float(np.linalg.norm(offset)) < 1e-9:
            targets = [origin.copy()] * int(round(hold / params.period))
        else:
            targets = straight_targets(origin, offset, params.linear_speed_cap, hold,
                                       params.period)
        ff = twists(targets, origin, params.period)
        runs['solver'].append(solver_motion(arm, params, start, targets, ff))
        runs['rule'].append(rule_motion(arm.chain, start, targets, params.period,
                                        params.linear_speed_cap, params.angular_speed_cap,
                                        lower, hard_stop, float(arm.velocity.min())))
    return runs


# ---- 2. the session model ----

def confine(twist: np.ndarray, tool: np.ndarray, origin: np.ndarray, seconds: float,
            box: float, max_turn: float) -> np.ndarray:
    """
    Return ``twist`` turned back where it would take the tool out of the session's bounds.

    The rule of ``workspace.wander``: per axis a motion that would leave ``box`` metres of
    where the session started is reversed, and a turn beyond ``max_turn`` is pointed back.
    """
    out = twist.copy()
    ahead = tool[:3, 3] + out[:3] * seconds - origin[:3, 3]
    for axis in range(3):
        if abs(ahead[axis]) > box:
            out[axis] = -abs(out[axis]) * np.sign(ahead[axis])
    turned = pose_error(tool, origin)[1]
    if float(np.linalg.norm(turned + out[3:] * seconds)) > max_turn:
        speed = float(np.linalg.norm(out[3:]))
        size = float(np.linalg.norm(turned))
        out[3:] = -turned / size * speed if size > 1e-9 else 0.0
    return out


def sessions(arm: Arm, params: SolverParams, start: Sequence[float], count: int = 100,
             segments: int = 30, seconds: float = 1.0, seed: int = 0, box: float = 0.30,
             max_turn: float = math.radians(60.0)) -> Dict[str, Any]:
    """
    Drive the session model of ``workspace.wander`` with the solver.

    The same twists and the same confinement as ``wander``; each motion is a target that
    moves with the twist for ``seconds`` from where the tool is, and the solver follows it
    (kinematically). No motion ever ends a session: the solver has no halt. Returns the
    runs of all motions and, per session, whether a joint was held at a position bound.
    """
    origin = tool_pose(arm, start)
    cycles = int(round(seconds / params.period))
    runs: List[Run] = []
    limit_sessions = 0
    for session in range(count):
        q = np.array(start, dtype=float)
        hit = False
        for twist in workspace.sample_twists(segments, seed=seed * 100003 + session):
            tool = tool_pose(arm, q)
            twist = confine(twist, tool, origin, seconds, box, max_turn)
            targets = [pose_at(tool, twist[:3] * k * params.period,
                               twist[3:] * k * params.period) for k in range(1, cycles + 1)]
            ff = twists(targets, tool, params.period)
            result = track(arm, params, q, targets, ff)
            dp, dr = pose_error(targets[-1], tool_pose(arm, result.q[-1]))
            runs.append(_run_of(arm, result, (float(np.linalg.norm(dp)),
                                              float(np.linalg.norm(dr))), params.period))
            hit = hit or bool(result.fixed.any())
            q = result.q[-1]
        limit_sessions += hit
    return {'sessions': count, 'segments': segments, 'runs': runs,
            'sessions_with_joint_limit': limit_sessions}


# ---- 3. replay of a recorded session ----

@dataclass
class Engagement:
    """One engagement of a recorded session: where the arm was and the targets it was given."""

    start: float
    end: float
    q: Optional[List[float]] = None             # joint state nearest the start
    targets: Optional[List[Tuple[float, List[float], List[float]]]] = None   # t, p, q (xyzw)

    def __post_init__(self) -> None:
        """Start with no targets."""
        if self.targets is None:
            self.targets = []


class EngagementReader:
    """
    Cut the rows of a recorded session into engagements, one row at a time (streaming).

    Rows are dicts with ``k`` and ``t``: ``state`` (``engaged``), ``target`` (``p``, ``q``)
    and ``joints`` (``q``). An engagement starts at the first engaged state; its seed is the
    joint row nearest in time to that state (the one before or the first one after); its
    targets are the target rows received while engaged.
    """

    def __init__(self) -> None:
        """Start outside any engagement."""
        self._last_joints: Optional[Dict[str, Any]] = None
        self._open: Optional[Engagement] = None
        self._need_seed = False
        self.done: List[Engagement] = []

    def feed(self, row: Dict[str, Any]) -> None:
        """Take in one row."""
        kind = row.get('k')
        if kind == 'joints':
            if self._open is not None and self._need_seed:
                before = self._last_joints
                start = self._open.start
                nearest = before if before is not None and (
                    start - before['t'] <= row['t'] - start) else row
                self._open.q = [float(v) for v in nearest['q']]
                self._need_seed = False
            self._last_joints = row
        elif kind == 'state':
            if row.get('engaged') and self._open is None:
                self._open = Engagement(row['t'], row['t'])
                self._need_seed = True
            elif row.get('engaged'):
                self._open.end = row['t']
            elif self._open is not None:
                self._close()
        elif kind == 'target' and self._open is not None:
            self._open.targets.append((row['t'], row['p'], row['q']))

    def _close(self) -> None:
        engagement = self._open
        if self._need_seed and self._last_joints is not None:
            engagement.q = [float(v) for v in self._last_joints['q']]
        self._open = None
        self._need_seed = False
        self.done.append(engagement)

    def finish(self) -> List[Engagement]:
        """Close a last open engagement and return all."""
        if self._open is not None:
            self._close()
        return self.done


def read_engagements(rows: Iterable[Dict[str, Any]]) -> List[Engagement]:
    """Return the engagements of the rows of a session."""
    reader = EngagementReader()
    for row in rows:
        reader.feed(row)
    return reader.finish()


def _matrix(p: Sequence[float], q: Sequence[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = q_to_matrix(q)
    out[:3, 3] = p
    return out


def recorded_targets(engagement: Engagement, period: float):
    """
    Return the cycles of an engagement: ``(targets, feedforward)``, one per ``period``.

    Each recorded target is one cycle; a gap in the stream repeats the previous target
    (with no feedforward) for the cycles it covered. The first target has no feedforward.
    """
    targets: List[np.ndarray] = []
    ff: List[Optional[np.ndarray]] = []
    previous: Optional[Tuple[float, np.ndarray]] = None
    for t, p, q in engagement.targets:
        pose = _matrix(p, q)
        if previous is None:
            targets.append(pose)
            ff.append(np.zeros(6))
        else:
            dt = max(t - previous[0], 1e-6)
            for _ in range(max(1, int(round(dt / period))) - 1):
                targets.append(previous[1])
                ff.append(np.zeros(6))
            targets.append(pose)
            ff.append(twist_between(previous[1], pose, dt))
        previous = (t, pose)
    return targets, ff


def replay(arm: Arm, params: SolverParams, engagements: Sequence[Engagement],
           delay_ticks: int = 8, stale_ticks: int = 3, minimum_targets: int = 5
           ) -> Dict[str, Any]:
    """
    Run the solver over the recorded targets of every engagement.

    Each engagement is seeded from its nearest joint row. ``delay_ticks`` is the measured
    arm's age (8 cycles = 150 ms); the reference is ``stale_ticks`` old. The recorded
    targets were made for the old driving path (no lead clamp), so the solver's error
    here includes the lag of the arm behind a hand that outran the old limits.
    """
    results = []
    for engagement in engagements:
        if engagement.q is None or len(engagement.targets) < minimum_targets:
            continue
        targets, ff = recorded_targets(engagement, params.period)
        result = track(arm, params, engagement.q, targets, ff, delay_ticks, stale_ticks)
        first = pose_error(targets[0], tool_pose(arm, engagement.q))
        results.append({
            'start': engagement.start, 'duration': engagement.end - engagement.start,
            'cycles': len(targets),
            'seed_error': (float(np.linalg.norm(first[0])), float(np.linalg.norm(first[1]))),
            'damped_s': float(result.damped.sum()) * params.period,
            'solver_error_max': (float(result.solver_error[:, 0].max()),
                                 float(result.solver_error[:, 1].max())),
            'tool_error_max': (float(result.tool_error[:, 0].max()),
                               float(result.tool_error[:, 1].max())),
            'peak_joint_speed': result.peak_joint_speed,
            'joint_limit_cycles': int(result.fixed.sum()),
            'min_margin': min(_margin(arm, q) for q in result.q),
            'reversed_fast': reversed_fast(result.qdot),
            'conditioning_min': float(result.conditioning.min())})
    return {'engagements': results}


# ---- statistics and the report ----

def share(flags: Iterable[bool]) -> float:
    """Return the fraction of true values (NaN when there are none)."""
    flags = list(flags)
    return sum(flags) / len(flags) if flags else math.nan


def _percentile(values: Sequence[float], fraction: float) -> float:
    return float(np.percentile(values, fraction * 100.0)) if len(values) else math.nan


def summarize_runs(runs: Sequence[Run]) -> Dict[str, float]:
    """Return the figures of a set of motions."""
    return {
        'count': len(runs),
        'reached': share(r.reached for r in runs),
        'affected': share(r.affected for r in runs),
        'halted': share(r.halted for r in runs),
        'joint_limit': share(r.joint_limit for r in runs),
        'min_margin': min((r.min_margin for r in runs), default=math.nan),
        'moving_at_end': share(r.moving_at_end for r in runs if r.moving_at_end is not None),
        'reversed_fast': share(r.reversed_fast for r in runs if r.reversed_fast is not None),
        'peak_joint_speed_max': max((r.peak_joint_speed for r in runs), default=math.nan),
        'peak_joint_speed_p95': _percentile([r.peak_joint_speed for r in runs], 0.95),
        'max_error_mm_p95': _percentile([r.max_error[0] * 1000.0 for r in runs], 0.95),
        'max_error_mm_max': max((r.max_error[0] * 1000.0 for r in runs), default=math.nan),
        'max_error_deg_p95': _percentile([math.degrees(r.max_error[1]) for r in runs], 0.95),
        'max_error_deg_max': max((math.degrees(r.max_error[1]) for r in runs),
                                 default=math.nan),
        'final_error_mm_p95': _percentile([r.final_error[0] * 1000.0 for r in runs], 0.95),
        'final_error_deg_p95': _percentile([math.degrees(r.final_error[1]) for r in runs],
                                           0.95)}


def _pct(value: float) -> str:
    return 'n/a' if not math.isfinite(value) else f'{value * 100:.1f} %'


def _num(value: float, spec: str = '.2f') -> str:
    return 'n/a' if not math.isfinite(value) else format(value, spec)


def _pair(first: float, second: float, spec: str) -> str:
    return f'{_num(first, spec)} / {_num(second, spec)}'


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Return the rows as aligned text lines: the first column left, the others right."""
    widths = [max(len(str(r[i])) for r in [header] + list(rows)) for i in range(len(header))]
    out = []
    for row in [header] + list(rows):
        out.append('  '.join(str(cell).ljust(widths[0]) if i == 0 else str(cell).rjust(widths[i])
                             for i, cell in enumerate(row)).rstrip())
    return out


def format_cube(result: Dict[str, List[Run]], reference: Optional[Dict[str, float]] = None
                ) -> str:
    """Return the cube table. ``reference`` is the workspace map's grid summary, if computed."""
    rule, sol = summarize_runs(result['rule']), summarize_runs(result['solver'])
    lines = [f"cube of {int(sol['count'])} tool positions, straight line at fixed orientation, "
             'target at the speed caps then held 1 s;',
             'reached = final error under 5 mm and 2 deg']
    if reference:
        lines.append('workspace map (kinematic walk; the condition number staying below 17 / '
                     f"30 / 100): {_pct(reference['17.0'])} / {_pct(reference['30.0'])} / "
                     f"{_pct(reference['100.0'])}; walk completed at all: "
                     f"{_pct(reference['complete'])}")
    rows = [
        ('share reached', _pct(rule['reached']), _pct(sol['reached'])),
        ('share where the rule slowed / damping engaged', _pct(rule['affected']),
         _pct(sol['affected'])),
        ('share halted by the rule', _pct(rule['halted']), '-'),
        ('joint limit: rule stopped within 0.1 rad / solver damper holding',
         _pct(rule['joint_limit']), _pct(sol['joint_limit'])),
        ('closest approach to a joint bound, rad (min)', _num(rule['min_margin'], '.3f'),
         _num(sol['min_margin'], '.3f')),
        ('still moving in the last 0.5 s of the hold (a joint above 0.05 rad/s)', '-',
         _pct(sol['moving_at_end'])),
        ('a joint above 0.5 rad/s in both directions', '-', _pct(sol['reversed_fast'])),
        ('peak joint speed, rad/s (max / 95th pct)',
         _pair(rule['peak_joint_speed_max'], rule['peak_joint_speed_p95'], '.2f'),
         _pair(sol['peak_joint_speed_max'], sol['peak_joint_speed_p95'], '.2f')),
        ('max position error on the way, mm (95th pct / max)',
         _pair(rule['max_error_mm_p95'], rule['max_error_mm_max'], '.0f'),
         _pair(sol['max_error_mm_p95'], sol['max_error_mm_max'], '.0f')),
        ('max orientation error on the way, deg (95th pct / max)',
         _pair(rule['max_error_deg_p95'], rule['max_error_deg_max'], '.0f'),
         _pair(sol['max_error_deg_p95'], sol['max_error_deg_max'], '.0f'))]
    return '\n'.join(lines + table(('', 'old rule (model)', 'DLS solver'), rows))


def format_sessions(result: Dict[str, Any], old: Optional[Dict[str, Any]] = None) -> str:
    """Return the session-model table. ``old`` is ``workspace.wander``'s result."""
    runs: List[Run] = result['runs']
    unaffected = share(not r.joint_limit and not r.affected for r in runs)
    limit = share(r.joint_limit for r in runs)
    damped = share(r.affected and not r.joint_limit for r in runs)
    with_limit = result['sessions_with_joint_limit'] / result['sessions']
    stats = summarize_runs(runs)
    lines = [f"{result['sessions']} sessions of {result['segments']} one-second motions "
             f'({len(runs)} motions with the solver): random direction, 30-100 % of the speed '
             'caps, chained from where the last ended, kept within 0.30 m and 60 deg of the start']
    out = old['outcomes'] if old else {}
    rows = [
        ('motions unaffected', _pct(out.get('free', math.nan)), _pct(unaffected)),
        ('motions slowed / damped', _pct(out.get('slowed', math.nan)), _pct(damped)),
        ('motions held to a standstill', _pct(out.get('stalled', math.nan)), '-'),
        ('motions ending in a halt', _pct(out.get('halted', math.nan)), '0 (no halt state)'),
        ('motions ending at a joint limit (rule: within 0.1 rad)',
         _pct(out.get('joint_limit', math.nan)), '-'),
        ('motions with the joint-limit damper active', '-', _pct(limit)),
        ('closest approach to a joint bound, rad (min)', '-', _num(stats['min_margin'], '.3f')),
        ('motions with a joint above 0.5 rad/s in both directions', '-',
         _pct(stats['reversed_fast'])),
        ('sessions with any halt', _pct(old['sessions_halted']) if old else 'n/a',
         '0 (no halt state)'),
        ('sessions with the joint-limit damper active', '-', _pct(with_limit))]
    lines += table(('', 'old rule (17 / 30)', 'DLS solver'), rows)
    lines += [
        f"DLS peak joint speed {_num(stats['peak_joint_speed_max'])} rad/s",
        'DLS error against the moving target, per motion: max position error (95th pct / max) '
        f"{_pair(stats['max_error_mm_p95'], stats['max_error_mm_max'], '.1f')} mm, max "
        f"orientation error {_pair(stats['max_error_deg_p95'], stats['max_error_deg_max'], '.1f')}"
        ' deg',
        'DLS error at the end of the motion (95th pct): '
        f"{_num(stats['final_error_mm_p95'], '.1f')} mm, "
        f"{_num(stats['final_error_deg_p95'], '.1f')} deg; motions that ended outside 5 mm / "
        f"2 deg: {_pct(1.0 - stats['reached'])}"]
    return '\n'.join(lines)


def format_replay(result: Dict[str, Any], frame_check: Optional[Tuple[float, float]] = None,
                  recorded: Optional[Dict[str, Any]] = None) -> str:
    """Return the replay summary."""
    runs = result['engagements']
    lines = []
    if frame_check is not None:
        lines.append('frame check: tool at the seed joints vs the first recorded target, median '
                     f'{frame_check[0] * 1000:.1f} mm, max {frame_check[1] * 1000:.1f} mm')
    lines.append(f'engagements replayed: {len(runs)}, engaged time '
                 f"{sum(r['duration'] for r in runs):.1f} s")
    if not runs:
        return '\n'.join(lines)
    damped = sum(r['damped_s'] for r in runs)
    total = sum(r['cycles'] for r in runs) * 0.02
    translation = [r['solver_error_max'][0] * 1000.0 for r in runs]
    rotation = [math.degrees(r['solver_error_max'][1]) for r in runs]
    lines += [
        f'time damped: {damped:.1f} s ({_pct(damped / total)} of the replayed cycles); '
        f"engagements with any damping: {sum(1 for r in runs if r['damped_s'] > 0)}",
        f"minimum conditioning {min(r['conditioning_min'] for r in runs):.3f}",
        'max solver error (target vs the reference state): '
        f'{max(translation):.0f} mm, {max(rotation):.0f} deg '
        f'(95th pct of the engagements {_percentile(translation, 0.95):.0f} mm, '
        f'{_percentile(rotation, 0.95):.0f} deg)',
        'max error, target vs the measured tool (with the lag): '
        f"{max(r['tool_error_max'][0] for r in runs) * 1000:.0f} mm, "
        f"{math.degrees(max(r['tool_error_max'][1] for r in runs)):.0f} deg",
        f"peak joint speed {max(r['peak_joint_speed'] for r in runs):.2f} rad/s",
        f"joint-limit damper holding a joint: {sum(r['joint_limit_cycles'] for r in runs)} "
        f"cycles in {sum(1 for r in runs if r['joint_limit_cycles'] > 0)} engagements; closest "
        f"approach to a bound {min(r['min_margin'] for r in runs):.3f} rad",
        'engagements with a joint above 0.5 rad/s in both directions: '
        f"{sum(1 for r in runs if r['reversed_fast'])} (a real reversal of the hand counts "
        'too)']
    if recorded:
        lines.append(f"for comparison, the recording itself: {recorded['halts']} MoveIt Servo "
                     f"halt statuses, {recorded['decelerating']} decelerate statuses")
    return '\n'.join(lines)


def seed_errors(arm: Arm, engagements: Sequence[Engagement], minimum_targets: int = 5
                ) -> Tuple[float, float]:
    """
    Return the median and largest distance (m) from each seed to its first recorded target.

    The tool at an engagement's seed joints should be where its first target is: the check
    that the arm model and the recorded frames agree.
    """
    distances = []
    for engagement in engagements:
        if engagement.q is None or len(engagement.targets) < minimum_targets:
            continue
        first = engagement.targets[0][1]
        distances.append(float(np.linalg.norm(
            tool_pose(arm, engagement.q)[:3, 3] - np.asarray(first))))
    return (float(np.median(distances)), max(distances)) if distances else (math.nan, math.nan)


def iter_rows(lines: Iterable[str], kinds: Optional[Sequence[str]] = None
              ) -> Iterator[Dict[str, Any]]:
    """Yield the JSON rows of the lines (blank lines skipped), only of ``kinds`` if given."""
    import json
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if kinds is not None and not any(f'"k": "{kind}"' in line[:120] for kind in kinds):
            continue
        yield json.loads(line)
