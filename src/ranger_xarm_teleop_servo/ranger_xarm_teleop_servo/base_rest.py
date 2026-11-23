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
Whether the base is at rest, and how far it turned since the arm's calibration (G-19).

Both the base node and the arm's teleop node judge the base from odometry with ``RestWatch``,
each from its own subscription: the arm does not take the base node's word for it, so a base
node that has died cannot report a rest.

* **At rest:** over the last ``window_s`` every odometry sample has a linear speed of at most
  ``linear_mps`` and a turning rate of at most ``turn_rps``, there are at least
  ``min_samples`` of them, and the newest arrived no more than ``fresh_s`` ago. Ages are taken
  on arrival (monotonic). Measured in Gazebo (base task, step 1): with no command the wheel
  odometry reads exactly zero, also during arm strokes with the base free to move; a stop from
  the caps comes to rest by ground truth 0.11 to 0.24 s after the zero.
* **Unknown:** no odometry, or none fresh. Neither node takes an unknown base for a resting one.
* **Turn since the calibration** (``HeadingWatch``): the calibration's heading is relative to
  the arm base, which rides on the vehicle, so a turn of the base turns every later mapping. The
  odometry yaw is taken when a calibration is made; once it differs by more than the set angle
  the calibration no longer holds. Without odometry to measure it, any commanded turn makes the
  calibration invalid.

No ROS imports.
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Optional

AT_REST, MOVING, UNKNOWN = 0, 1, 2
REST_NAMES = ('AT_REST', 'MOVING', 'UNKNOWN')


@dataclass(frozen=True)
class RestParams:
    """The rule for "at rest" (G-19, INV-2)."""

    linear_mps: float = 0.01        # speed at or below this counts as still
    turn_rps: float = 0.02          # turning rate at or below this counts as still
    window_s: float = 0.3           # still over this long
    min_samples: int = 3            # with at least this many samples in the window
    fresh_s: float = 0.2            # the newest sample no older than this

    def check(self) -> None:
        """Raise ValueError for settings that cannot work."""
        for name in ('linear_mps', 'turn_rps', 'window_s', 'fresh_s'):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f'rest {name} must be finite and above zero')
        if self.min_samples < 1:
            raise ValueError('rest min_samples must be at least 1')


class RestWatch:
    """Odometry samples of the last window and the verdict on them."""

    def __init__(self, params: RestParams = RestParams()) -> None:
        """Watch with ``params``."""
        params.check()
        self.params = params
        self._arrivals: Deque[float] = deque()      # arrivals within the window
        self.newest: Optional[float] = None
        self.still_since: Optional[float] = None    # first still sample of an unbroken run
        self.yaw: Optional[float] = None            # newest odometry yaw, rad

    def add(self, arrival: float, vx: float, vy: float, wz: float,
            yaw: Optional[float] = None) -> None:
        """Take one odometry sample: arrival (monotonic s), body twist, pose yaw."""
        if self.newest is not None and arrival < self.newest:
            return                          # out of order: the newer verdict stands
        still = (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(wz)
                 and math.hypot(vx, vy) <= self.params.linear_mps
                 and abs(wz) <= self.params.turn_rps)
        gap = self.newest is None or arrival - self.newest > self.params.fresh_s
        if not still:
            self.still_since = None
        elif self.still_since is None or gap:
            self.still_since = arrival      # a run of stillness starts (again, after a gap)
        self._arrivals.append(arrival)
        self.newest = arrival
        self.yaw = yaw if yaw is not None and math.isfinite(yaw) else None
        while self._arrivals and self._arrivals[0] < arrival - self.params.window_s:
            self._arrivals.popleft()

    def age(self, now: float) -> float:
        """Return seconds since the newest sample arrived (inf when none)."""
        return math.inf if self.newest is None else max(0.0, now - self.newest)

    def fresh(self, now: float) -> bool:
        """Return True while the newest sample is fresh."""
        return self.age(now) <= self.params.fresh_s

    def state(self, now: float) -> int:
        """Return AT_REST, MOVING or UNKNOWN at ``now``."""
        if not self.fresh(now):
            return UNKNOWN
        if self.still_since is None or self.newest - self.still_since < self.params.window_s:
            return MOVING                   # moving, or not still for a whole window yet
        if len(self._arrivals) < self.params.min_samples:
            return MOVING                   # too few samples to say
        return AT_REST

    def fresh_yaw(self, now: float) -> Optional[float]:
        """Return the newest odometry yaw while fresh, else None."""
        return self.yaw if self.fresh(now) else None


class CommandWatch:
    """
    Whether the base is being commanded, from the base command as the arm's node hears it.

    Commanded while the newest command is nonzero and no older than ``fresh_s``: the base node
    publishes every cycle while it drives and one zero when it stops. A base node that falls
    silent mid-drive stops counting after ``fresh_s``; odometry then shows whether the base
    still moves.
    """

    def __init__(self, fresh_s: float = 0.1) -> None:
        """Count a nonzero command as in force for ``fresh_s`` after it arrived."""
        self.fresh_s = fresh_s
        self.nonzero_at: Optional[float] = None     # arrival of the newest command, if nonzero

    def on_command(self, arrival: float, vx: float, vy: float, wz: float) -> None:
        """Take one base command (arrival on the monotonic clock)."""
        zero = vx == 0.0 and vy == 0.0 and wz == 0.0
        self.nonzero_at = None if zero else arrival

    def commanded(self, now: float) -> bool:
        """Return True while the newest command is nonzero and fresh."""
        return self.nonzero_at is not None and now - self.nonzero_at <= self.fresh_s


def motion(rest: 'RestWatch', commands: 'CommandWatch', now: float) -> int:
    """Return the base's motion for the arm: commanded counts as moving; unknown stays."""
    state = rest.state(now)
    if state == AT_REST and commands.commanded(now):
        return MOVING
    return state


def yaw_difference(a: float, b: float) -> float:
    """Return the absolute angle between two yaws, wrapped to [0, pi]."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


class HeadingWatch:
    """How far the base turned since the calibration in force (G-19)."""

    def __init__(self, limit_rad: float) -> None:
        """Refuse the calibration once the base turned more than ``limit_rad``."""
        if not (math.isfinite(limit_rad) and limit_rad > 0.0):
            raise ValueError('the turn that invalidates the calibration must be above zero')
        self.limit = limit_rad
        self.calibrated = False
        self.yaw0: Optional[float] = None       # odometry yaw at the calibration
        self.unmeasured_turn = False            # a turn was commanded with no odometry for it
        self.turned = 0.0                       # rad since the calibration, as last measured

    def on_calibration(self, yaw: Optional[float]) -> None:
        """Take a new calibration, made while odometry said ``yaw`` (None: no fresh odometry)."""
        self.calibrated = True
        self.yaw0 = yaw if yaw is not None and math.isfinite(yaw) else None
        self.unmeasured_turn = False
        self.turned = 0.0

    def on_odometry(self, yaw: Optional[float]) -> None:
        """
        Take a fresh odometry yaw.

        A calibration made before any odometry takes the first yaw that follows as its
        reference, unless a turn was commanded in between (the base turns only when commanded;
        a turn without odometry has already made the calibration invalid).
        """
        if not self.calibrated or yaw is None or not math.isfinite(yaw):
            return
        if self.yaw0 is None:
            if not self.unmeasured_turn:
                self.yaw0 = yaw
            return
        self.turned = max(self.turned, yaw_difference(yaw, self.yaw0))

    def on_turn_command(self, wz: float, odometry_fresh: bool) -> None:
        """Take a commanded turning rate seen on the base command."""
        if not self.calibrated or wz == 0.0:
            return
        if not odometry_fresh or self.yaw0 is None:
            self.unmeasured_turn = True

    def invalid(self) -> bool:
        """Return True when the calibration no longer holds because of the base's turn."""
        return self.calibrated and (self.unmeasured_turn or self.turned > self.limit)

    def describe(self) -> str:
        """Return plain words for the log: how far, or why unknown."""
        if self.unmeasured_turn:
            return 'the base was commanded to turn with no odometry to measure it'
        return (f'the base turned {math.degrees(self.turned):.1f} deg since the calibration '
                f'(limit {math.degrees(self.limit):.1f})')
