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
Ownership logic of the command mux (G-9). No ROS imports.

Two trajectory controllers claim the arm joints; ros2_control allows only one of them
to be active, so the owner is whoever feeds the active one. The mux asks for a switch
only when the arm is stopped, no trajectory goal is executing and teleop is idle. A
refused or failed switch leaves the owner as it was.
"""

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Mapping, Optional, Sequence


class Owner(IntEnum):
    """Values match ``ArmOwner.OWNER_*``."""

    NONE = 0
    MOVEIT = 1
    TELEOP = 2
    SWITCHING = 3


class MuxStatus(IntEnum):
    """Values match ``ArmOwner.STATUS_*``."""

    OK = 0
    REFUSED_MOVING = 1
    REFUSED_GOAL_ACTIVE = 2
    REFUSED_TELEOP_ACTIVE = 3
    STALE_JOINT_STATE = 4
    SWITCH_FAILED = 5


# action_msgs/GoalStatus values that mean a trajectory goal is in progress.
GOAL_ACCEPTED, GOAL_EXECUTING, GOAL_CANCELING = 1, 2, 3
ACTIVE_GOAL_STATES = (GOAL_ACCEPTED, GOAL_EXECUTING, GOAL_CANCELING)


@dataclass(frozen=True)
class MuxParams:
    """Thresholds for "the arm is stopped"."""

    still_speed: float = 0.02          # rad/s, every arm joint below this
    still_duration: float = 0.3        # s, for at least this long
    joint_state_timeout: float = 0.2   # s, newest joint state no older than this


def owner_of(states: Mapping[str, str], moveit_controller: str,
             teleop_controller: str) -> Owner:
    """Return the owner implied by the controller manager's controller states."""
    moveit = states.get(moveit_controller) == 'active'
    teleop = states.get(teleop_controller) == 'active'
    if moveit and not teleop:
        return Owner.MOVEIT
    if teleop and not moveit:
        return Owner.TELEOP
    return Owner.NONE


def any_goal_active(statuses: Sequence[int]) -> bool:
    """Return True when any goal status is accepted, executing or canceling."""
    return any(int(s) in ACTIVE_GOAL_STATES for s in statuses)


class StillnessTracker:
    """
    Tracks, from joint states, since when every arm joint has been still.

    Speeds and gaps are computed between consecutive samples from their own stamps
    (one source clock, so no clocks are compared). The local arrival time on a
    monotonic clock is used only to tell how old the newest sample is. A sample
    sequence with a hole longer than the timeout cannot show stillness across it.
    """

    def __init__(self, params: MuxParams) -> None:
        self.params = params
        self.last_arrival: Optional[float] = None
        self.last_stamp: Optional[float] = None
        self.last_positions: Optional[Sequence[float]] = None
        self.still_since: Optional[float] = None    # stamp of the first still sample
        self.last_motion = 'no joint state yet'     # why stillness last restarted

    def update(self, arrival: float, stamp: float, positions: Sequence[float],
               velocities: Optional[Sequence[float]] = None) -> None:
        """
        Take in one joint state: local ``arrival`` (monotonic s) and its ``stamp`` (s).

        The speed is the larger of the reported velocity and the position difference
        over the stamp interval, so a backend that reports zero velocity is still
        seen moving.
        """
        speed = 0.0
        if velocities:
            speed = max(abs(float(v)) for v in velocities)
        comparable = (self.last_stamp is not None and self.last_positions is not None
                      and len(self.last_positions) == len(positions))
        if comparable:
            dt = stamp - self.last_stamp
            if dt <= 0.0:
                if dt < 0.0:
                    self.still_since = None
                    self.last_motion = f'joint state stamps went back by {-dt:.3f} s'
                    self.last_stamp = stamp
                    self.last_positions = tuple(float(p) for p in positions)
                self.last_arrival = arrival
                return
            if dt > self.params.joint_state_timeout:
                self.still_since = None     # a hole: stillness has to be shown again
                self.last_motion = f'gap of {dt:.2f} s in the joint states'
            else:
                step = max(abs(float(a) - float(b))
                           for a, b in zip(positions, self.last_positions))
                speed = max(speed, step / dt)
        if not math.isfinite(speed) or speed > self.params.still_speed:
            self.still_since = None
            self.last_motion = f'joint speed {speed:.3f} rad/s'
        elif self.still_since is None:
            self.still_since = stamp
        self.last_arrival = arrival
        self.last_stamp = stamp
        self.last_positions = tuple(float(p) for p in positions)

    def age(self, now: float) -> float:
        """Return the age of the newest joint state by local arrival; inf when none."""
        return math.inf if self.last_arrival is None else now - self.last_arrival

    def still_for(self) -> float:
        """Return for how long (by sample stamps) the arm has been still; 0 if moving."""
        if self.still_since is None or self.last_stamp is None:
            return 0.0
        return self.last_stamp - self.still_since


def check_switch(now: float, tracker: StillnessTracker, goal_active: bool,
                 teleop_active: bool) -> MuxStatus:
    """Return OK when a controller switch may be requested now, else why not."""
    if teleop_active:
        return MuxStatus.REFUSED_TELEOP_ACTIVE
    if goal_active:
        return MuxStatus.REFUSED_GOAL_ACTIVE
    if tracker.age(now) > tracker.params.joint_state_timeout:
        return MuxStatus.STALE_JOINT_STATE
    if tracker.still_for() < tracker.params.still_duration:
        return MuxStatus.REFUSED_MOVING
    return MuxStatus.OK
