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
Reading every parameter of another node in three requests (no ROS).

A parameter declared without a value makes a whole ``get_parameters`` reply empty, so the
values cannot be asked for in one request by name alone. Asking name by name to find those
parameters costs one request per name: about 150 for MoveIt Servo, within a few tens of
milliseconds. Every reply of a service goes to every client of it, so such a burst pushes
the replies other nodes are waiting for out of their queues (measured: the teleop node's
start-up request to MoveIt Servo went unanswered that way).

So the read is: the names (``list_parameters``), their types (``get_parameter_types``; a
parameter without a value has the type "not set"), then the values of the names that have
one, in one request. A step without an answer, or with an answer that does not fit, starts
the read again from the names; it never waits for ever.
"""

from typing import Any, Dict, List, Optional, Sequence

NOT_SET = 0                 # rcl_interfaces/msg/ParameterType.PARAMETER_NOT_SET
NAMES, TYPES, VALUES = 'names', 'types', 'values'       # the steps, in order
READ_TIMEOUT_S = 3.0        # a step without an answer for this long is asked again from the start


class ParameterRead:
    """
    The state of one such read; the owner sends the requests.

    ``step`` is the request that is out (None: none), ``since`` when it went out.
    ``values`` and ``unset`` are complete once ``done``.
    """

    def __init__(self, timeout_s: float = READ_TIMEOUT_S) -> None:
        self.timeout_s = float(timeout_s)
        self.step: Optional[str] = None
        self.since = 0.0
        self.done = False
        self.restarts = 0           # reads started again after silence
        self.asked: List[str] = []  # the names of the step that is out
        self.unset: List[str] = []
        self.values: Dict[str, Any] = {}

    def due(self, now: float) -> bool:
        """Return True when the names should be asked for (again)."""
        return not self.done and (self.step is None or now - self.since >= self.timeout_s)

    def begin(self, now: float) -> Optional[str]:
        """
        Start the read from the names; return the step that got no answer, or None.

        None means a first start, or a start after an answer that did not fit.
        """
        silent = self.step
        if silent is not None:
            self.restarts += 1
        self.step, self.since, self.asked = NAMES, now, []
        return silent

    def _again(self) -> None:
        self.step, self.asked = None, []

    def failed(self) -> None:
        """Take a failed request: the read starts again when next due."""
        self._again()

    def listed(self, names: Sequence[str], now: float) -> Optional[List[str]]:
        """
        Take the names; return the names to ask the types of, or None to start again.

        A node that reports two names or fewer (``use_sim_time`` and the like) has not
        declared its own parameters yet.
        """
        names = list(names)
        if self.step != NAMES or len(names) <= 2:
            self._again()
            return None
        self.step, self.since, self.asked = TYPES, now, names
        return names

    def typed(self, types: Sequence[int], now: float) -> Optional[List[str]]:
        """
        Take the types; return the names that have a value, or None to start again.

        An empty list is a valid answer here: no parameter has a value, and the read is done.
        """
        types = list(types)
        if self.step != TYPES or len(types) != len(self.asked):
            self._again()           # a name was undeclared in between: the names changed
            return None
        self.unset = [name for name, kind in zip(self.asked, types) if kind == NOT_SET]
        having = [name for name, kind in zip(self.asked, types) if kind != NOT_SET]
        if not having:
            self.step, self.asked, self.values, self.done = None, [], {}, True
            return []
        self.step, self.since, self.asked = VALUES, now, having
        return having

    def valued(self, values: Sequence[Any]) -> bool:
        """Take the values; return True when the read is done, False to start again."""
        values = list(values)
        if self.step != VALUES or len(values) != len(self.asked):
            self._again()           # a value was unset in between: the reply came back empty
            return False
        self.values = dict(zip(self.asked, values))
        self.step, self.asked, self.done = None, [], True
        return True
