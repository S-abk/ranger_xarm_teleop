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
