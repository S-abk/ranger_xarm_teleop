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
Stopping a motion goal by its own id, by the rules of ``calls.py`` (no ROS).

One stop, used by the arm tools (``tools/_calls.py``), the reset node and its watchdog:

* the cancel request names the goal's own id, never the zero id (which asks the server to
  cancel every goal it has), so that asking again can cancel nothing else;
* a request without an answer within ``answer_s`` counts as unanswered, and the next one may
  go out: a reply can be lost without the caller being told;
* the caller watches the goal's status and the joints, and reports what it saw.

How long to keep asking is the caller's choice. The tools ask at most ``attempts`` times and
then report. The reset node and its watchdog pass no limit: they keep asking until the goal
has ended (``ended``), and their stop is over only once the joints are also still
(``finished``).

When a goal counts as ended: its status was seen terminal (succeeded, cancelled, aborted); or
the server has answered for ``settle_s`` that it cannot cancel it (code 3: it has ended or is
already being cancelled) or that it knows no goal with this id (code 2: it never arrived, or
ended and was forgotten), with no status of the goal seen active meanwhile. A goal still on
its way when the server answers "unknown" would be listed active once it arrives; that
restarts the wait and the requests go on.
"""

from typing import List, Optional, Tuple

from .calls import CANCEL_NO_ERROR, cancel_text, GOAL_STATUS, Motion, status_text

STATUS_ACCEPTED, STATUS_EXECUTING, STATUS_CANCELING = 1, 2, 3   # action_msgs/msg/GoalStatus
TERMINAL = (4, 5, 6)                                               # succeeded, canceled, aborted
ACTIVE = (STATUS_ACCEPTED, STATUS_EXECUTING, STATUS_CANCELING)
CODE_UNKNOWN_GOAL, CODE_TERMINATED = 2, 3                          # CancelGoal return codes
SETTLE_S = 2.0          # answers that the goal cannot be cancelled, for this long: it has ended


class GoalStop:
    """The cancel requests for one goal, the answers and statuses seen, and when it is over."""

    def __init__(self, goal_id: bytes, answer_s: float, attempts: Optional[int] = None,
                 settle_s: float = SETTLE_S) -> None:
        """
        Stop the goal ``goal_id`` (its 16 bytes).

        ``answer_s``: a request without an answer for this long is unanswered. ``attempts``:
        the most requests to send (None: until the goal has ended).
        """
        goal_id = bytes(goal_id)
        if len(goal_id) != 16 or not any(goal_id):
            raise ValueError('a goal is cancelled by its own 16-byte id; the zero id would '
                             'cancel every goal of the server')
        self.goal_id = goal_id
        self.answer_s, self.attempts, self.settle_s = answer_s, attempts, settle_s
        self.sent_count = 0
        self.unanswered_count = 0
        self.outstanding: Optional[float] = None        # when the open request went out
        self.answer: Optional[Tuple[int, int]] = None   # the newest (return code, goals)
        self.statuses: List[int] = []                   # of this goal, in the order seen
        self.any_status = False
        self.ended_by = ''                              # why the goal counts as ended
        self._cannot_since: Optional[float] = None      # answers "cannot cancel" since then

    # ---- the requests ----

    def due(self, now: float) -> bool:
        """Return True when a cancel request should go out now."""
        if self.outstanding is not None:
            if now - self.outstanding < self.answer_s:
                return False
            self.timed_out()
        if self.attempts is not None:
            return self.answer is None and self.sent_count < self.attempts
        return not self.ended_by

    def sent(self, now: float) -> None:
        """Note that a request went out (one still open counts as unanswered)."""
        self.timed_out()
        self.sent_count += 1
        self.outstanding = now

    def timed_out(self) -> None:
        """Note that the open request got no answer in time (it may never come)."""
        if self.outstanding is not None:
            self.outstanding = None
            self.unanswered_count += 1

    def answered(self, now: float, code: int, cancelling: int) -> None:
        """Take the answer to a request: its return code and how many goals it cancels."""
        self.outstanding = None
        self.answer = (int(code), int(cancelling))
        if code in (CODE_UNKNOWN_GOAL, CODE_TERMINATED):
            if self._cannot_since is None:
                self._cannot_since = now
            self._check(now)
        else:
            self._cannot_since = None

    # ---- what is seen ----

    def status(self, now: float, status: int) -> None:
        """Take a status of this goal from the server's status topic."""
        self.any_status = True
        status = int(status)
        if not self.statuses or self.statuses[-1] != status:
            self.statuses.append(status)
        if status in TERMINAL:
            self.ended_by = self.ended_by or f'the server listed it as {GOAL_STATUS[status]}'
        elif status in ACTIVE:
            self._cannot_since = None       # listed: whatever was answered, it is there
        self._check(now)

    def other_status(self) -> None:
        """Note a status message of the server that did not list this goal."""
        self.any_status = True

    def tick(self, now: float) -> None:
        """Let time pass (the settle wait ends without a new answer)."""
        self._check(now)

    def _check(self, now: float) -> None:
        if self.ended_by or self._cannot_since is None:
            return
        if now - self._cannot_since >= self.settle_s:
            code = self.answer[0] if self.answer else None
            self.ended_by = ('the server answered for {:g} s that it knew no goal with this id'
                             if code == CODE_UNKNOWN_GOAL else
                             'the server answered for {:g} s that the goal could no longer be '
                             'cancelled').format(self.settle_s)

    # ---- the outcome ----

    def finished(self, motion: Motion) -> bool:
        """Return True when the goal has ended and the joints are seen still."""
        return bool(self.ended_by) and motion.known and motion.still

    def cancel_answer(self):
        """Return the newest answer as ``(return code, goals)``, None if none came."""
        return self.answer

    def accepted(self) -> bool:
        """Return True when the server answered that it is cancelling the goal."""
        return self.answer is not None and self.answer[0] == CANCEL_NO_ERROR

    def report(self, motion: Motion) -> str:
        """Say what the requests achieved, what the server listed and what the joints did."""
        asked = (f"{self.sent_count} cancel request(s) for this goal's own id, "
                 f'{self.unanswered_count} unanswered')
        return (f'{asked}; {cancel_text(self.answer)}; '
                f'{status_text(self.statuses, self.any_status)}; {motion.text()}')
