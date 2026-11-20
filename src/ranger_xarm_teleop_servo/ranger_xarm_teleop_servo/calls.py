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
# moving. The caller cancels its own goal, watches the goal's status and the joints, and
# reports what it saw. "The joints did not move" is said only when they were watched from
# before the goal went out and did not move.

GOAL_STATUS = {0: 'unknown', 1: 'accepted', 2: 'executing', 3: 'cancelling', 4: 'succeeded',
               5: 'cancelled', 6: 'aborted'}        # action_msgs/msg/GoalStatus
STILL_S = 1.0               # joints are still when none changed by STILL_RAD over this long
STILL_RAD = 0.001
CANCEL_NO_ERROR = 0         # action_msgs/srv/CancelGoal return codes
# What a return code shows, and no more. An unknown id does not show that the goal never
# reached the server: a server forgets a goal some seconds after it ended (10 s by default),
# and a goal request still on its way is unknown until it arrives. A goal that is already being
# cancelled is answered like one that has ended.
CANCEL_REASONS = {1: 'the server rejected the cancel request',
                  2: ('the server knew no goal with this id when asked to cancel it (the goal '
                      'never reached it, or had ended and been forgotten)'),
                  3: ('the server answered that the goal can no longer be cancelled (it has '
                      'ended, or is already being cancelled)')}


class Motion:
    """
    What the joint states showed since a mark: how far, and whether it has ended.

    ``from_before`` is False when no joint state from before the mark was at hand: the
    motion is then counted from the first one after it, and may be larger. ``stale_s`` is
    the age of the newest joint state when joint states have stopped arriving, else None:
    what the joints do now is then not known, neither that they rest nor that they move.
    """

    def __init__(self, known: bool, moved: float = 0.0, still: bool = False,
                 watched_s: float = 0.0, from_before: bool = True, stale_s=None) -> None:
        self.known, self.moved, self.still = known, moved, still
        self.watched_s, self.from_before, self.stale_s = watched_s, from_before, stale_s

    def text(self) -> str:
        """Say it in words for a report line."""
        if not self.known:
            return ("no joint state with all of the arm's joints arrived, so whether the arm "
                    'moved is not known')
        amount = (f'up to {self.moved:.3f} rad' if self.from_before else
                  f'{self.moved:.3f} rad or more (no joint state from before the request was '
                  'at hand)')
        if self.stale_s is not None:
            return (f'the joints had moved by {amount} when the joint states stopped arriving '
                    f'{self.stale_s:.1f} s ago: what they do now is not known')
        if self.from_before and self.moved < STILL_RAD and self.still:
            return (f'the joints did not move (largest change {self.moved:.4f} rad in '
                    f'{self.watched_s:.1f} s)')
        return f'the joints moved by {amount} and ' + (
            'are still now' if self.still else 'ARE STILL MOVING')


class JointWatch:
    """
    The joints since a mark: the largest excursion of any of them, and whether they rest.

    Fed with every joint state. The excursion is kept as a running maximum against the
    positions at the mark, so it does not depend on how many joint states are kept; only
    the last few seconds are, to judge rest.
    """

    def __init__(self, names, still_s: float = STILL_S, still_rad: float = STILL_RAD) -> None:
        self.names = list(names)
        self.still_s, self.still_rad = still_s, still_rad
        self.recent = []            # (time, positions) of the last 2 * still_s
        self.reference = None       # positions at the mark
        self.marked_at = None
        self.from_before = False
        self.moved = 0.0

    def sample(self, when: float, positions) -> None:
        """Take one joint state (a mapping of joint name to position)."""
        if not all(name in positions for name in self.names):
            return
        q = [float(positions[name]) for name in self.names]
        self.recent.append((when, q))
        while self.recent and self.recent[0][0] < when - 2.0 * self.still_s:
            del self.recent[0]
        if self.marked_at is None:
            return
        if self.reference is None:
            self.reference = q      # the first one after the mark: motion before it is unseen
        self.moved = max(self.moved, max(abs(a - b) for a, b in zip(q, self.reference)))

    def mark(self, when: float, fresh_s: float = 0.5) -> None:
        """Start counting from now: the reference is the newest joint state, if it is fresh."""
        self.marked_at, self.moved = when, 0.0
        fresh = bool(self.recent) and when - self.recent[-1][0] <= fresh_s
        self.reference = list(self.recent[-1][1]) if fresh else None
        self.from_before = fresh

    def motion(self, now: float) -> Motion:
        """Return what was seen since the mark. Rest needs joint states up to ``now``."""
        if self.marked_at is None or self.reference is None:
            return Motion(False)
        age = now - self.recent[-1][0]
        stale = age > 0.5 * self.still_s        # joint states have stopped arriving
        recent = [q for when, q in self.recent if when >= now - self.still_s]
        covered = len(recent) >= 2 and not stale and self.recent[0][0] <= now - self.still_s
        spread = max(max(q[j] for q in recent) - min(q[j] for q in recent)
                     for j in range(len(self.names))) if recent else 0.0
        return Motion(True, self.moved, covered and spread < self.still_rad,
                      now - self.marked_at, self.from_before, age if stale else None)


def cancel_text(answer) -> str:
    """Say what the request to cancel the goal achieved (return code and goals, or None)."""
    if answer is None:
        return 'the cancel request was NOT ANSWERED'
    code, cancelling = answer
    if code == CANCEL_NO_ERROR:
        return ('the server is cancelling the goal' if cancelling else
                'the server accepted the cancel request and named no goal that it cancels')
    if code in (2, 3):
        return CANCEL_REASONS[code]
    return f'not cancelled: {CANCEL_REASONS.get(code, f"return code {code}")}'


def status_text(statuses, any_status: bool = True) -> str:
    """
    Say what the action server's status topic showed of the goal, in the order seen.

    ``any_status`` is False when no status message arrived at all: then nothing is known
    about what the server lists.
    """
    if statuses:
        return 'the server listed the goal as ' + ', then '.join(
            str(GOAL_STATUS.get(status, status)) for status in statuses)
    if not any_status:
        return 'no status message of the server arrived, so what it lists is not known'
    return 'the server did not list the goal'


def unanswered_goal(what: str, seconds: float, statuses, cancel, motion: Motion,
                    any_status: bool = True) -> str:
    """Return the report for a request about a goal that got no answer."""
    return (f'no answer to {what} within {seconds:g} s, so its state is not known: '
            f'{status_text(statuses, any_status)}; {cancel_text(cancel)}; {motion.text()}')
