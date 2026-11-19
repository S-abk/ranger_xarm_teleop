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
Episode events to vibration requests, exactly once per event (no ROS).

The episode node counts its events (``EpisodeState.event_count``) and repeats its state at
5 Hz. A request is made only when the count goes up, for the newest event; events skipped
between two state messages are counted as missed, never played or invented. The first
message seen is the baseline: it is the latched state of events that happened before.
"""

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

# EpisodeState.EVENT_* (checked against the message definition by the unit tests).
EVENT_NAMES = {0: 'NONE', 1: 'START', 2: 'STOP', 3: 'KEEP', 4: 'DISCARD', 5: 'MARK',
               6: 'REFUSED', 7: 'AUTO_STOP'}
EVENTS = tuple(name for code, name in sorted(EVENT_NAMES.items()) if code != 0)

MAX_PATTERN_MS = 5000
MAX_PULSES = 16


@dataclass(frozen=True)
class Pattern:
    """A vibration pattern: pulses of (intensity 0..1, on ms, off ms) and its priority."""

    pulses: Tuple[Tuple[float, int, int], ...]
    priority: int

    @staticmethod
    def from_flat(flat: Sequence[float], priority: int) -> 'Pattern':
        """
        Build a pattern from a flat [intensity, on_ms, off_ms, ...] list (a ROS parameter).

        An empty list, or pulses all at intensity 0, is a valid pattern that vibrates nothing:
        the event is not felt and no request is made.
        Raises ValueError for anything else that is not a sensible pattern.
        """
        values = [float(v) for v in flat]
        if len(values) % 3:
            raise ValueError(f'a pattern is [intensity, on_ms, off_ms, ...]; got {len(values)} '
                             'numbers')
        pulses = []
        for i in range(0, len(values), 3):
            intensity, on_ms, off_ms = values[i:i + 3]
            if not all(math.isfinite(v) for v in (intensity, on_ms, off_ms)):
                raise ValueError('pattern values must be finite')
            if not 0.0 <= intensity <= 1.0:
                raise ValueError(f'intensity {intensity} outside 0..1')
            if on_ms < 0 or off_ms < 0 or on_ms != int(on_ms) or off_ms != int(off_ms):
                raise ValueError('on_ms and off_ms must be whole, non-negative milliseconds')
            pulses.append((intensity, int(on_ms), int(off_ms)))
        if len(pulses) > MAX_PULSES:
            raise ValueError(f'more than {MAX_PULSES} pulses')
        if sum(on + off for _, on, off in pulses) > MAX_PATTERN_MS:
            raise ValueError(f'pattern longer than {MAX_PATTERN_MS} ms')
        if not 0 <= int(priority) <= 255:
            raise ValueError('priority must be 0..255')
        return Pattern(tuple(pulses), int(priority))

    def flat(self) -> Tuple[float, ...]:
        """Return the flat parameter form."""
        return tuple(v for pulse in self.pulses for v in pulse)

    @property
    def silent(self) -> bool:
        """Return True when the pattern vibrates nothing (no pulses, or all at intensity 0)."""
        return not any(intensity > 0.0 and on > 0 for intensity, on, _ in self.pulses)

    @property
    def duration_ms(self) -> int:
        return sum(on + off for _, on, off in self.pulses)


# Distinct by count and length. The two the operator must not miss (REFUSED, AUTO_STOP) are
# long and full strength, and outrank other sources' requests; a newer episode event still
# replaces them (the page plays the newest request of a source).
DEFAULT_PATTERNS: Dict[str, Pattern] = {
    'MARK': Pattern(((0.5, 40, 0),), 1),
    'START': Pattern(((0.8, 200, 0),), 2),
    'STOP': Pattern(((0.8, 120, 120), (0.8, 120, 0)), 2),
    'KEEP': Pattern(((0.7, 500, 0),), 2),
    'DISCARD': Pattern(((0.8, 80, 100), (0.8, 80, 100), (0.8, 80, 0)), 2),
    'REFUSED': Pattern(((1.0, 350, 200), (1.0, 350, 0)), 3),
    'AUTO_STOP': Pattern(((1.0, 120, 80),) * 4 + ((1.0, 120, 400),)
                         + ((1.0, 120, 80),) * 4 + ((1.0, 120, 0),), 3),
}


@dataclass(frozen=True)
class EpisodeSample:
    """The fields of one EpisodeState message this module uses."""

    session: str
    event_count: int
    last_event: int
    detail: str
    stamp: float              # header.stamp, s (ROS time)
    last_event_stamp: float   # s (ROS time)
    wall_time: float          # host wall clock at publication, s

    @property
    def event_wall_time(self) -> float:
        """Host wall clock of the newest event: publication time less its ROS-time age."""
        return self.wall_time - max(0.0, self.stamp - self.last_event_stamp)


@dataclass(frozen=True)
class HapticRequest:
    """One vibration request for an episode event."""

    event: str
    pattern: Pattern
    source_count: int
    missed: int
    wall_time: float          # of the event
    detail: str


class EpisodeEvents:
    """Turns successive EpisodeState samples into at most one request per new event."""

    def __init__(self, patterns: Optional[Mapping[str, Pattern]] = None,
                 max_age_s: float = 0.75) -> None:
        self.patterns = dict(DEFAULT_PATTERNS if patterns is None else patterns)
        self.max_age_s = float(max_age_s)
        self._key: Optional[Tuple[str, int]] = None
        self.missed_total = 0
        self.too_old = 0
        self.baselines = 0

    def update(self, s: EpisodeSample, wall_now: float) -> Optional[HapticRequest]:
        """Take one state message; return the request for its new event, if any."""
        key = (s.session, int(s.event_count))
        last = self._key
        self._key = key
        if last is None or s.session != last[0] or key[1] < last[1]:
            # First message, a restarted episode node (new session) or a count going back:
            # a new baseline. Nothing in it is new to this node.
            self.baselines += 1
            return None
        steps = key[1] - last[1]
        if steps <= 0:
            return None                                   # a repeat
        missed = steps - 1
        self.missed_total += missed
        name = EVENT_NAMES.get(int(s.last_event), '')
        pattern = self.patterns.get(name)
        if not name or name == 'NONE' or pattern is None or pattern.silent:
            return None
        if wall_now - s.event_wall_time > self.max_age_s:
            self.too_old += 1                             # never played late
            return None
        return HapticRequest(name, pattern, key[1], missed, s.event_wall_time, s.detail)
