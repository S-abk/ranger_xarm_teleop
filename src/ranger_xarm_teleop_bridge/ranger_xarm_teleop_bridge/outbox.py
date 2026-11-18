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
What the bridge has to send to the headset, waiting for its relay connection (no ROS).

Filled from ROS callbacks, emptied by the connection's sender task, under a lock. Nothing
grows while the link is down: vibration requests are at most ``haptic_max`` (the oldest is
dropped on overflow) and each is dropped once older than its own maximum age; the status is
one slot holding only the newest, dropped once older than ``status_max_age_s``. Ages count
from the moment the item was made (its age when put, plus the time it waited here), so an item
is sent with its true age or not at all.
"""

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable, Deque, List, Optional, Tuple


@dataclass
class _Item:
    payload: dict
    origin: float       # clock time at which the item's age was zero
    max_age: float


class Outbox:
    """Bounded, age-limited queue of vibration requests plus a newest-only status slot."""

    def __init__(self, haptic_max: int = 8, status_max_age_s: float = 0.4,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._haptics: Deque[_Item] = deque()
        self._haptic_max = max(1, int(haptic_max))
        self._status: Optional[_Item] = None
        self.status_max_age_s = float(status_max_age_s)
        self.clock = clock
        self.dropped_overflow = 0
        self.dropped_old_haptic = 0
        self.dropped_old_status = 0
        self.replaced_status = 0

    def put_haptic(self, payload: dict, age_s: float, max_age_s: float) -> None:
        """Queue a vibration request that is ``age_s`` old now."""
        now = self.clock()
        with self._lock:
            if len(self._haptics) >= self._haptic_max:
                self._haptics.popleft()
                self.dropped_overflow += 1
            self._haptics.append(_Item(payload, now - max(0.0, age_s), float(max_age_s)))

    def set_status(self, payload: dict, age_s: float) -> None:
        """Replace the status with a newer one that is ``age_s`` old now."""
        now = self.clock()
        with self._lock:
            if self._status is not None:
                self.replaced_status += 1
            self._status = _Item(payload, now - max(0.0, age_s), self.status_max_age_s)

    def take(self) -> List[Tuple[dict, float]]:
        """Return (payload, age s) of everything still young enough, oldest request first."""
        now = self.clock()
        out: List[Tuple[dict, float]] = []
        with self._lock:
            while self._haptics:
                item = self._haptics.popleft()
                age = now - item.origin
                if age <= item.max_age:
                    out.append((item.payload, age))
                else:
                    self.dropped_old_haptic += 1
            if self._status is not None:
                item, self._status = self._status, None
                age = now - item.origin
                if age <= item.max_age:
                    out.append((item.payload, age))
                else:
                    self.dropped_old_status += 1
        return out

    def expire(self) -> None:
        """Drop what has become too old without sending anything (link stalled)."""
        now = self.clock()
        with self._lock:
            kept = deque(i for i in self._haptics if now - i.origin <= i.max_age)
            self.dropped_old_haptic += len(self._haptics) - len(kept)
            self._haptics = kept
            if self._status is not None and now - self._status.origin > self._status.max_age:
                self._status = None
                self.dropped_old_status += 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._haptics) + (self._status is not None)
