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

With gripper teleop, a second pair of controllers claims the gripper joint, and the
owner is one for both: MoveIt's pair or teleop's pair, swapped in the same switch. A
mixed state (the arm with one side, the gripper with the other, or both of a pair
active) is a split: no owner, never switched to teleop, only back to MoveIt. A pair with
neither controller active (still loading) is incomplete: no owner either.
"""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import math
from typing import List, Mapping, Optional, Sequence, Tuple


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


def _pair_owner(states: Mapping[str, str], moveit_controller: str,
                teleop_controller: str) -> Owner:
    moveit = states.get(moveit_controller) == 'active'
    teleop = states.get(teleop_controller) == 'active'
    if moveit and not teleop:
        return Owner.MOVEIT
    if teleop and not moveit:
        return Owner.TELEOP
    return Owner.NONE


def owner_of(states: Mapping[str, str], moveit_controller: str, teleop_controller: str,
             moveit_gripper: str = '', teleop_gripper: str = '') -> Owner:
    """
    Return the owner implied by the controller manager's controller states.

    With a gripper pair, the arm's and the gripper's owner must agree, else NONE.
    """
    arm = _pair_owner(states, moveit_controller, teleop_controller)
    if not (moveit_gripper and teleop_gripper):
        return arm
    gripper = _pair_owner(states, moveit_gripper, teleop_gripper)
    return arm if arm == gripper else Owner.NONE


def split_of(states: Mapping[str, str], pairs: Sequence[Tuple[str, str]]) -> str:
    """
    Return why the controllers are split between owners, or '' when they are not.

    ``pairs``: (MoveIt's controller, teleop's controller) per joint group. Split: both
    controllers of a pair active, or one pair on MoveIt's side and another on teleop's. A
    pair with neither active (for example still loading) is incomplete, not split: there
    is no owner, and a switch activates what is missing.
    """
    both = any(states.get(m) == 'active' and states.get(t) == 'active' for m, t in pairs)
    owners = {_pair_owner(states, m, t) for m, t in pairs} - {Owner.NONE}
    if not both and len(owners) <= 1:
        return ''
    described = ', '.join(f'{name} {states.get(name, "not loaded")}'
                          for pair in pairs for name in pair)
    return f'controllers split between owners: {described}'


def switch_lists(states: Mapping[str, str], pairs: Sequence[Tuple[str, str]],
                 wanted: Owner) -> Tuple[Optional[Tuple[List[str], List[str]]], str]:
    """
    Return ``((activate, deactivate), '')`` for one strict switch to ``wanted``.

    Return ``(None, why)`` when the switch must not be made: a split is never switched
    to teleop, only resolved back to MoveIt.
    """
    side = 0 if wanted == Owner.MOVEIT else 1
    split = split_of(states, pairs)
    if split and wanted == Owner.TELEOP:
        return None, split
    activate = [pair[side] for pair in pairs if states.get(pair[side]) != 'active']
    deactivate = [pair[1 - side] for pair in pairs if states.get(pair[1 - side]) == 'active']
    return (activate, deactivate), ''


def outside_activation(states: Mapping[str, str], teleop_controllers: Sequence[str],
                       switched_by_mux: bool) -> List[str]:
    """
    Return the teleop controllers that are active although the mux did not switch them there.

    The robot's driver switches every controller off after an arm error and asks for all of
    them to be activated again once the arm is ready, bypassing the mux (G-17). Such a teleop
    controller is reported and, once the arm is still, deactivated; nothing is activated in
    response.
    """
    if switched_by_mux:
        return []
    return [name for name in teleop_controllers if states.get(name) == 'active']


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

    def __init__(self, params: MuxParams, names: Sequence[str] = ()) -> None:
        self.params = params
        self.names = tuple(names)                   # joint names, for the motion report
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
        fastest = None
        if velocities:
            speeds = [abs(float(v)) for v in velocities]
            speed = max(speeds)
            fastest = speeds.index(speed)
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
                steps = [abs(float(a) - float(b)) / dt
                         for a, b in zip(positions, self.last_positions)]
                if max(steps) > speed:
                    speed = max(steps)
                    fastest = steps.index(speed)
        if not math.isfinite(speed) or speed > self.params.still_speed:
            self.still_since = None
            name = (self.names[fastest] if fastest is not None and fastest < len(self.names)
                    else '')
            self.last_motion = f'joint speed {speed:.3f} rad/s' + (f' ({name})' if name else '')
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


class BandTracker:
    """
    Tracks whether the gripper joints have stayed within a band over the stillness window.

    In Gazebo the arm's rule (sample-to-sample speed) read the gripper's driven joint as
    moving at about 0.16 rad/s, again and again, and refused every switch back to MoveIt;
    a finger squeezing an object might read so too (not measured). Its "stopped" is
    therefore judged by net travel: every joint's positions over
    the last ``still_duration`` (by sample stamps) span at most ``band``. A ramp of the
    gripper at its speed limit spans far more than the band in that window. Holes in the
    stream and stamps that go back restart the window, as for the arm.
    """

    def __init__(self, params: MuxParams, band: float, names: Sequence[str] = ()) -> None:
        if not (math.isfinite(band) and band > 0.0):
            raise ValueError('gripper_still_band: expected a finite positive number, got '
                             f'{band!r}')
        self.params = params
        self.band = band
        self.names = tuple(names)
        self.samples: deque = deque()               # (stamp, positions)
        self.last_arrival: Optional[float] = None
        self.last_motion = 'no joint state yet'

    def update(self, arrival: float, stamp: float, positions: Sequence[float]) -> None:
        """Take in one sample of the gripper joints: local ``arrival`` and its ``stamp``."""
        positions = tuple(float(p) for p in positions)
        if self.samples:
            dt = stamp - self.samples[-1][0]
            if dt == 0.0:
                self.last_arrival = arrival     # the same sample again: nothing new to judge
                return
            if dt < 0.0 or dt > self.params.joint_state_timeout:
                self.samples.clear()
                self.last_motion = ('gripper joint state stamps went back' if dt < 0.0
                                    else f'gap of {dt:.2f} s in the gripper joint states')
        self.samples.append((stamp, positions))
        # Keep the newest sample at or before the window's start, and everything after it.
        while len(self.samples) > 1 and self.samples[1][0] <= stamp - self.params.still_duration:
            self.samples.popleft()
        self.last_arrival = arrival

    def age(self, now: float) -> float:
        """Return the age of the newest sample by local arrival; inf when none."""
        return math.inf if self.last_arrival is None else now - self.last_arrival

    def still(self) -> bool:
        """Return True when the window is covered and every joint stayed within the band."""
        if not self.samples:
            return False
        covered = self.samples[-1][0] - self.samples[0][0]
        if covered < self.params.still_duration - 1e-6:
            if covered > 0.0:
                self.last_motion = (f'gripper joint states cover only {covered:.2f} s of '
                                    f'{self.params.still_duration:.2f} s')
            return False
        for k in range(len(self.samples[-1][1])):
            values = [s[1][k] for s in self.samples]
            finite = all(math.isfinite(v) for v in values)
            span = max(values) - min(values) if finite else math.inf
            if span > self.band:
                name = self.names[k] if k < len(self.names) else f'joint {k}'
                self.last_motion = (f'gripper moved {span:.4f} within '
                                    f'{self.params.still_duration:.2f} s ({name})')
                return False
        return True


def check_switch(now: float, tracker: StillnessTracker, goal_active: bool,
                 teleop_active: bool, gripper: Optional[BandTracker] = None) -> MuxStatus:
    """Return OK when a controller switch may be requested now, else why not."""
    if teleop_active:
        return MuxStatus.REFUSED_TELEOP_ACTIVE
    if goal_active:
        return MuxStatus.REFUSED_GOAL_ACTIVE
    if tracker.age(now) > tracker.params.joint_state_timeout:
        return MuxStatus.STALE_JOINT_STATE
    if gripper is not None and gripper.age(now) > gripper.params.joint_state_timeout:
        return MuxStatus.STALE_JOINT_STATE
    if tracker.still_for() < tracker.params.still_duration:
        return MuxStatus.REFUSED_MOVING
    if gripper is not None and not gripper.still():
        return MuxStatus.REFUSED_MOVING
    return MuxStatus.OK
