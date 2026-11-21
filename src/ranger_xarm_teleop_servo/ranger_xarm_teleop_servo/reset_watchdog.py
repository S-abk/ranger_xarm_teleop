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
The reset watchdog's rule (no ROS): cancel the reset's goal when the reset node falls silent.

The reset node (G-16, hold-to-run) cancels its own goal when the operator lets go. If that
node dies or stalls, nothing would: move_group does not cancel a goal when its client goes.
So a separate process watches the reset node's heartbeat, which names the goal while one is
out, and when the heartbeat is older than ``heartbeat_timeout_s`` while that goal is
unfinished, it cancels that goal by its id and keeps cancelling until the goal has ended and
the joints are still (``goal_stop.GoalStop``, without a limit on the requests).

* It only ever cancels the goal the heartbeat named, by its own non-zero id. It never starts
  or changes a motion and never cancels anything else.
* A goal it has cancelled is not watched again, whatever later heartbeats say.
* A process that joins the ROS graph can stall a publisher for about a third of a second,
  longer than the heartbeat limit; such a stall cancels a reset in progress. That fails safe
  and is reported as such (the reset node says "heartbeat late"); the limit is not widened.
"""

import math
from typing import List, Optional, Sequence, Set

from .calls import JointWatch
from .goal_stop import GoalStop, TERMINAL

HEARTBEAT_TIMEOUT_S = 0.2       # the reset node's heartbeat (50 Hz) is late after this
CANCEL_ANSWER_S = 0.5           # a cancel request without an answer is asked again after this


class Watchdog:
    """What the watchdog has heard, and the stop it is carrying out."""

    def __init__(self, joint_names: Sequence[str],
                 heartbeat_timeout_s: float = HEARTBEAT_TIMEOUT_S,
                 cancel_answer_s: float = CANCEL_ANSWER_S) -> None:
        """Watch the joints ``joint_names`` while stopping."""
        if not (math.isfinite(heartbeat_timeout_s) and heartbeat_timeout_s > 0.0):
            raise ValueError('the heartbeat timeout must be a positive number of seconds')
        self.names = list(joint_names)
        self.timeout, self.cancel_answer_s = heartbeat_timeout_s, cancel_answer_s
        self.beat_at: Optional[float] = None    # arrival of the newest heartbeat
        self.beat_seq: Optional[int] = None
        self.goal_out = False                   # the newest heartbeat says a goal is out
        self.goal_id: Optional[bytes] = None    # and names it
        self.status: Optional[int] = None       # the newest status of that goal
        self.done: Set[bytes] = set()           # goals cancelled: never watched again
        self.stop: Optional[GoalStop] = None
        self.stop_id: Optional[bytes] = None    # the goal being cancelled or cancelled last
        self.stop_at: Optional[float] = None
        self.stops = 0
        self.last_report = ''
        self.watch = JointWatch(self.names)
        self.log: List[str] = []

    @property
    def stopping(self) -> bool:
        """Return True while a goal is being cancelled."""
        return self.stop is not None

    # ---- input ----

    def on_heartbeat(self, now: float, seq: int, goal_out: bool, goal_id: bytes) -> None:
        """Take one heartbeat of the reset node."""
        goal_id = bytes(goal_id)
        if self.beat_seq is not None and seq != self.beat_seq + 1:
            self.log.append(f'heartbeat seq {self.beat_seq} -> {seq}')
        self.beat_at, self.beat_seq = now, int(seq)
        if goal_out and not any(goal_id):
            self.log.append('a heartbeat says a goal is out but names none: ignored')
            goal_out = False
        if goal_id != self.goal_id:
            self.status = None
        self.goal_out = bool(goal_out)
        self.goal_id = goal_id if goal_out else self.goal_id

    def on_status(self, now: float, goal_id: bytes, status: int) -> None:
        """Take a status of a goal from move_group's status topic."""
        goal_id = bytes(goal_id)
        if goal_id == self.goal_id:
            self.status = int(status)
        if self.stop is not None:
            if goal_id == self.stop_id:
                self.stop.status(now, status)
            else:
                self.stop.other_status()

    def on_joints(self, when: float, positions) -> None:
        """Take one joint state (a mapping of joint name to position)."""
        self.watch.sample(when, positions)

    # ---- the cycle ----

    def tick(self, now: float) -> None:
        """Start a stop when the heartbeat is late while its goal is unfinished; end it after."""
        if self.stop is not None:
            self.stop.tick(now)
            motion = self.watch.motion(now)
            if self.stop.finished(motion):
                self.last_report = self.stop.report(motion)
                self.log.append(f'reset goal {self.stop_id.hex()} stopped: {self.last_report}')
                self.stop = None
            return
        if (self.beat_at is None or not self.goal_out or self.goal_id is None
                or self.goal_id in self.done or self.status in TERMINAL):
            return
        late = now - self.beat_at
        if late <= self.timeout:
            return
        self.stop = GoalStop(self.goal_id, self.cancel_answer_s)
        if self.status is not None:
            self.stop.status(now, self.status)
        self.stop_id, self.stop_at = self.goal_id, now
        self.done.add(self.goal_id)
        self.stops += 1
        self.watch.mark(now)
        self.log.append(f'the reset heartbeat is {late * 1000.0:.0f} ms old while goal '
                        f'{self.goal_id.hex()} is unfinished: cancelling it')

    # ---- cancel requests (the node sends them) ----

    def cancel_due(self, now: float) -> Optional[bytes]:
        """Return the goal id to send a cancel request for now, or None."""
        if self.stop is not None and self.stop.due(now):
            return self.stop_id
        return None

    def cancel_sent(self, now: float) -> None:
        """Note that a cancel request went out."""
        self.stop.sent(now)

    def cancel_answered(self, now: float, code: int, cancelling: int) -> None:
        """Take the answer to a cancel request."""
        if self.stop is not None:
            self.stop.answered(now, code, cancelling)

    def cancel_timed_out(self) -> None:
        """Note that the open cancel request got no answer in time."""
        if self.stop is not None:
            self.stop.timed_out()

    def take_log(self) -> List[str]:
        """Return the lines to log since the last call."""
        lines, self.log = self.log, []
        return lines
