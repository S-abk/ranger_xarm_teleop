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
Retreat after a collision halt: back along the joint path the arm came in on. No ROS.

The teleop node records the measured joint path while teleop is armed
(``PathRecorder``). After a halt the retreat tool fetches it and ``plan`` decides how far
back to go: from the arm's posture, which must be the newest point of the path, it walks
the path backwards and stops at the first point that is

* clear of the robot's fixed structure by the clearance asked for (at least the link
  keep-out margin) according to the keep-out model, and
* valid for MoveIt (``valid``, the tool asks move_group, whose padding is included), so
  MoveIt plans from there again.

Whether teleop can be armed there (the arming check refuses postures near the wrist
singularity) is reported with the plan; where it cannot, the message says to move to the
ready posture with MoveIt first.

It never leaves the recorded path, never goes past ``max_tool_travel`` of tool path or
``max_joint_travel`` on any joint, and moves nothing at all (no partial move) when the path
goes deeper into contact on the way, or no point within those bounds qualifies.
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, List, Sequence, Tuple

import numpy as np

POINT_SPACING = 0.002          # rad: a new point when any joint moved this much
KEEP_SECONDS = 60.0            # the recorder keeps this much of the path
KEEP_POINTS = 3000
ON_PATH = 0.01                 # rad: the arm counts as at the path's newest point
MAX_GAP = 0.08                 # rad: two recorded points further apart are not one motion
DEEPER = 0.001                 # m: a point this much less clear than the start is deeper
MAX_TOOL_TRAVEL = 0.15         # m of tool path, at most
MAX_JOINT_TRAVEL = 0.5         # rad on any joint, at most
JOINT_SPEED = 0.1              # rad/s, at most
TOOL_SPEED = 0.03              # m/s, at most
RAMP = 0.5                     # s to reach and to leave the top speed


class PathRecorder:
    """The measured joint path while teleop is armed: bounded, timed, decimated."""

    def __init__(self, spacing: float = POINT_SPACING, keep_seconds: float = KEEP_SECONDS,
                 keep_points: int = KEEP_POINTS) -> None:
        self.spacing = spacing
        self.keep_seconds = keep_seconds
        self.points: deque = deque(maxlen=keep_points)     # (time s, q array)
        self.dropped_for_age = False

    def add(self, t: float, q: Sequence[float]) -> bool:
        """Add the posture at time ``t`` (s) if it moved enough; return whether it did."""
        q = np.asarray(q, dtype=float)
        if not np.all(np.isfinite(q)):
            return False
        if self.points and np.max(np.abs(q - self.points[-1][1])) < self.spacing:
            return False
        self.points.append((float(t), q.copy()))
        while self.points and t - self.points[0][0] > self.keep_seconds:
            self.points.popleft()
            self.dropped_for_age = True
        return True

    def resume(self, q: Sequence[float]) -> bool:
        """
        Start recording again after a pause (teleop was disarmed).

        If the arm moved while nothing was recorded, the old path does not lead to where
        the arm is: it is dropped. Return whether it was.
        """
        q = np.asarray(q, dtype=float)
        if self.points and float(np.max(np.abs(q - self.points[-1][1]))) > ON_PATH:
            self.points.clear()
            return True
        return False

    def path(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(times, postures)``, oldest first."""
        if not self.points:
            return np.zeros(0), np.zeros((0, 0))
        return (np.array([t for t, _ in self.points]),
                np.array([q for _, q in self.points]))


@dataclass(frozen=True)
class Plan:
    """What the retreat will do, or why it does nothing."""

    ok: bool
    message: str
    postures: np.ndarray = None        # from the arm's posture back to the stop, inclusive
    times: np.ndarray = None           # s from the start, one per posture
    start_clearance: float = math.nan
    stop_clearance: float = math.nan
    tool_travel: float = 0.0           # m along the path to the stop
    joint_travel: float = 0.0          # rad, largest on any joint
    armable: bool = True               # the stop passes teleop's arming check


NEXT_STEP = ('Next: tools/move_arm.py <six joint values> moves the arm directly on '
             "MoveIt's controller (no planning, simulation only) to a posture you choose clear "
             'of the structure; or restart the simulation.')


Point = Callable[[np.ndarray], np.ndarray]


def plan(times: np.ndarray, postures: np.ndarray, measured: Sequence[float],
         clearance: Callable[[np.ndarray], float], tool_point: Point,
         valid: Callable[[np.ndarray], bool], conditioning: Callable[[np.ndarray], float],
         wanted: float, min_conditioning: float, max_tool_travel: float = MAX_TOOL_TRAVEL,
         max_joint_travel: float = MAX_JOINT_TRAVEL) -> Plan:
    """
    Return the retreat along the recorded path ``postures`` (oldest first) from ``measured``.

    ``clearance(q)`` is the keep-out model's clearance (m), ``tool_point(q)`` the tool
    position, ``valid(q)`` MoveIt's verdict, ``conditioning(q)`` the arming measure.
    ``wanted`` is the clearance to reach (at least the margin; checked by the caller).
    """
    measured = np.asarray(measured, dtype=float)
    if len(postures) == 0:
        return Plan(False, 'no recorded path: the teleop node has none (it was restarted, or '
                    'teleop has not moved the arm since it started). No motion goal sent. '
                    + NEXT_STEP)
    newest = postures[-1]
    off = float(np.max(np.abs(newest - measured)))
    if off > ON_PATH:
        return Plan(False, f'the arm is {off:.3f} rad from the end of the recorded path: it '
                    'has moved since teleop moved it. No motion goal sent. ' + NEXT_STEP)
    start = clearance(measured)
    if start >= wanted and valid(measured):
        return Plan(False, f'already clear: clearance {start * 1000.0:.1f} mm, valid for '
                    'MoveIt. Nothing to do.', start_clearance=start)
    tool_travel, best = 0.0, (start, 0.0)
    previous = measured
    for i in range(len(postures) - 1, -1, -1):
        q = postures[i]
        if float(np.max(np.abs(q - previous))) > MAX_GAP:
            break                      # not a recorded motion: the walk stops here
        tool_travel += float(np.linalg.norm(tool_point(q) - tool_point(previous)))
        joint_travel = float(np.max(np.abs(q - measured)))
        previous = q
        if tool_travel > max_tool_travel or joint_travel > max_joint_travel:
            break
        c = clearance(q)
        if c < start - DEEPER:
            return Plan(False, f'the recorded path goes deeper on the way back (clearance '
                        f'{c * 1000.0:.1f} mm, {start * 1000.0:.1f} mm now, at '
                        f'{tool_travel * 1000.0:.0f} mm of tool path): it is not followed. '
                        'No motion goal sent. ' + NEXT_STEP, start_clearance=start)
        if c > best[0]:
            best = (c, tool_travel)
        if c >= wanted and valid(q):
            path = np.vstack([measured, postures[i:][::-1]])
            keep = np.concatenate([[True], np.max(np.abs(np.diff(path, axis=0)), axis=1)
                                   > 1e-9])
            path = path[keep]
            armable = conditioning(q) >= min_conditioning
            message = (f'back {tool_travel * 1000.0:.0f} mm of tool path along the recorded '
                       f'path to clearance {c * 1000.0:.1f} mm (from {start * 1000.0:.1f} mm)')
            if not armable:
                message += ('; teleop arming is refused there (the wrist is near alignment): '
                            'move to the ready posture with tools/recover_arm.py first')
            return Plan(True, message, path, timing(path, tool_point), start, c, tool_travel,
                        joint_travel, armable)
    return Plan(False, f'no point on the recorded path within {max_tool_travel * 1000.0:.0f} '
                f'mm of tool path and {max_joint_travel:.2f} rad per joint is clear by '
                f'{wanted * 1000.0:.1f} mm and valid for MoveIt; the best clearance is '
                f'{best[0] * 1000.0:.1f} mm, at '
                f'{best[1] * 1000.0:.0f} mm of tool path. No motion goal sent. ' + NEXT_STEP,
                start_clearance=start)


def timing(path: np.ndarray, tool_point: Callable[[np.ndarray], np.ndarray],
           joint_speed: float = JOINT_SPEED, tool_speed: float = TOOL_SPEED,
           ramp: float = RAMP) -> np.ndarray:
    """
    Return the time (s) of each posture: at most ``joint_speed`` and ``tool_speed``.

    Each segment takes as long as its slowest limit allows; a trapezoidal profile over the
    whole path ramps up and down in ``ramp`` seconds, which only lowers the speeds.
    """
    lengths = [0.0]
    for a, b in zip(path[:-1], path[1:]):
        lengths.append(max(float(np.max(np.abs(b - a))) / joint_speed,
                           float(np.linalg.norm(tool_point(b) - tool_point(a))) / tool_speed))
    s = np.cumsum(lengths)                     # time at the top speed
    total = float(s[-1])
    if total <= 0.0:
        return np.zeros(len(path))
    # Trapezoid in s: speed 1 after ``ramp``, from 0; the same down at the end.
    out = np.empty(len(s))
    if total >= ramp:
        duration = total + ramp
        for k, v in enumerate(s):
            if v <= ramp / 2.0:
                out[k] = math.sqrt(2.0 * ramp * v)
            elif v >= total - ramp / 2.0:
                out[k] = duration - math.sqrt(2.0 * ramp * (total - v))
            else:
                out[k] = v + ramp / 2.0
    else:                                      # too short to reach the top speed
        duration = 2.0 * math.sqrt(ramp * total)
        for k, v in enumerate(s):
            out[k] = (math.sqrt(2.0 * ramp * v) if v <= total / 2.0
                      else duration - math.sqrt(2.0 * ramp * (total - v)))
    return out


def speeds(path: np.ndarray, times: np.ndarray,
           tool_point: Callable[[np.ndarray], np.ndarray]) -> Tuple[float, float]:
    """Return the largest joint speed (rad/s) and tool speed (m/s) between postures."""
    joint, tool = 0.0, 0.0
    for a, b, ta, tb in zip(path[:-1], path[1:], times[:-1], times[1:]):
        dt = tb - ta
        if dt <= 0.0:
            continue
        joint = max(joint, float(np.max(np.abs(b - a))) / dt)
        tool = max(tool, float(np.linalg.norm(tool_point(b) - tool_point(a))) / dt)
    return joint, tool


def distance_from_path(path: np.ndarray, q: Sequence[float]) -> float:
    """Return how far (rad, largest joint) ``q`` is from the polyline ``path``."""
    q = np.asarray(q, dtype=float)
    best = math.inf
    for a, b in zip(path[:-1], path[1:]):
        d = b - a
        denom = float(d @ d)
        t = 0.0 if denom <= 0.0 else min(1.0, max(0.0, float((q - a) @ d) / denom))
        best = min(best, float(np.max(np.abs(q - (a + t * d)))))
    if len(path) == 1:
        best = float(np.max(np.abs(q - path[0])))
    return best


def as_lists(plan_: Plan) -> List[List[float]]:
    """Return the plan's postures as plain lists (for a trajectory message)."""
    return [[float(v) for v in q] for q in plan_.postures]
