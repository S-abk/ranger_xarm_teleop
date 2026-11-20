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
Service calls whose reply may never come (no ROS).

A reply can be lost without the caller being told. Two ways were measured in this stack:

* the server has not yet matched a new client and drops the reply;
* every reply of a service goes to every client of it, and a client keeps only the last
  ``depth`` of them. While another client asks in a burst, a client that is busy for a few
  milliseconds finds its own reply pushed out of that queue. Nobody logs it.

So nothing waits for ever on a reply: every call has a timeout and a next step, a read is
asked again, and a client of a service that others ask in bursts (the parameter services)
has a deeper reply queue than the default of 10.
"""

from typing import Any, Optional

# Reply queue of a client of a parameter service. One reading of all the parameters of
# MoveIt Servo is about 150 requests when asked name by name; 1000 leaves room for several
# such readers at once. The queue holds references to replies, most of them a few bytes.
PARAMETER_REPLY_DEPTH = 1000


class Reread:
    """
    One read asked of a service until an answer arrives.

    This keeps the state only: whether an answer is still wanted, which request is out,
    since when, and how often it was asked. The owner sends the request and drops a lost
    one from its client. ``what`` names the read in log lines.
    """

    def __init__(self, what: str, timeout_s: float) -> None:
        self.what = what
        self.timeout_s = float(timeout_s)
        self.wanted = False
        self.request: Optional[Any] = None
        self.sent_at = 0.0
        self.attempts = 0

    def start(self) -> None:
        """Want an answer from now on (asked from scratch: the attempts count again)."""
        if not self.wanted:
            self.wanted = True
            self.attempts = 0

    def due(self) -> bool:
        """Return True when an answer is wanted and no request is out."""
        return self.wanted and self.request is None

    def sent(self, request: Any, now: float) -> None:
        """Note the request that is out now."""
        self.request = request
        self.sent_at = now
        self.attempts += 1

    def lost(self, now: float) -> Optional[Any]:
        """
        Return the request that is out for longer than the timeout, and give it up.

        The answer stays wanted, so the read is due again. None while no request is out or
        the one that is out is younger than the timeout.
        """
        if self.request is None or now - self.sent_at < self.timeout_s:
            return None
        request, self.request = self.request, None
        return request

    def answered(self, request: Any) -> bool:
        """
        Take the answer to ``request``; return False for a request that was given up.

        An answer to a request given up earlier is not used: the same read is out again, and
        its answer is the one that counts.
        """
        if request is not self.request:
            return False
        self.request = None
        self.wanted = False
        return True

    def drop(self) -> None:
        """Stop wanting an answer (the owner asks from scratch later)."""
        self.wanted = False
        self.request = None


# ---- a motion goal without an answer ----
#
# A goal is never sent twice, and a missing answer is not a rejection: the controller may be
# moving. The caller cancels, watches the goal's status and the joints, and reports what it
# saw. "Nothing moved" is said only when the joints were watched and did not move.

GOAL_STATUS = {0: 'unknown', 1: 'accepted', 2: 'executing', 3: 'cancelling', 4: 'succeeded',
               5: 'cancelled', 6: 'aborted'}        # action_msgs/msg/GoalStatus
GOAL_ACTIVE = (1, 2, 3)
STILL_S = 1.0               # joints are still when none changed by STILL_RAD over this long
STILL_RAD = 0.001
CANCEL_NO_ERROR = 0         # action_msgs/srv/CancelGoal return codes
CANCEL_REASONS = {1: 'the server rejected the cancel request',
                  2: 'the server knows no such goal',
                  3: 'the goal had already ended'}


class Motion:
    """What the joint states showed between two moments: how far, and whether it ended."""

    def __init__(self, known: bool, moved: float = 0.0, still: bool = False,
                 watched_s: float = 0.0) -> None:
        self.known, self.moved, self.still, self.watched_s = known, moved, still, watched_s

    def text(self) -> str:
        """Say it in words for a report line."""
        if not self.known:
            return 'no joint states arrived, so whether the arm moved is not known'
        if self.moved < STILL_RAD and self.still:
            return (f'the joints did not move (largest change {self.moved:.4f} rad in '
                    f'{self.watched_s:.1f} s)')
        end = 'are still now' if self.still else 'ARE STILL MOVING'
        return f'the joints moved by up to {self.moved:.3f} rad and {end}'


def motion_seen(samples, names, since: float, now: float, still_s: float = STILL_S,
                still_rad: float = STILL_RAD) -> Motion:
    """
    Return the motion in ``samples`` ((time, {joint: position}), oldest first) since ``since``.

    The reference is the last sample at or before ``since`` (else the first after it).
    ``still`` needs samples that cover the last ``still_s`` up to ``now``: joint states that
    stopped arriving do not show an arm at rest.
    """
    rows = [(t, [positions[n] for n in names]) for t, positions in samples
            if all(n in positions for n in names)]
    if not rows:
        return Motion(False)
    before = [row for row in rows if row[0] <= since]
    reference = before[-1] if before else rows[0]
    after = [row for row in rows if row[0] >= reference[0]]
    moved = max(max(abs(a - b) for a, b in zip(q, reference[1])) for _, q in after)
    recent = [q for t, q in rows if t >= now - still_s]
    covered = (rows[-1][0] >= now - 0.5 * still_s and rows[0][0] <= now - still_s
               and len(recent) >= 2)
    spread = max((max(q[j] for q in recent) - min(q[j] for q in recent)
                  for j in range(len(names))), default=0.0) if recent else 0.0
    return Motion(True, moved, covered and spread < still_rad, now - reference[0])


def cancel_text(answer) -> str:
    """Say what a cancel request achieved (``answer``: return code and goals, or None)."""
    if answer is None:
        return 'the cancel request was NOT ANSWERED'
    code, cancelling = answer
    if code != CANCEL_NO_ERROR:
        return f'cancel refused: {CANCEL_REASONS.get(code, f"return code {code}")}'
    if cancelling == 0:
        return 'the server had no goal to cancel'
    return f'the server is cancelling {cancelling} goal{"" if cancelling == 1 else "s"}'


def status_text(statuses) -> str:
    """Say what the action server's status topic showed of the goal, in the order seen."""
    if not statuses:
        return 'the server never listed the goal'
    return 'the server listed the goal as ' + ', then '.join(
        str(GOAL_STATUS.get(status, status)) for status in statuses)


def unanswered_goal(what: str, seconds: float, statuses, cancel, motion: Motion) -> str:
    """Return the report for a request about a goal that got no answer."""
    return (f'no answer to {what} within {seconds:g} s, so its state is UNKNOWN: '
            f'{status_text(statuses)}; {cancel_text(cancel)}; {motion.text()}')
