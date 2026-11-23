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
Base driving state to vibration requests and the panel line, exactly once per event (no ROS).

The base node counts its events (``BaseState.event_count``) and repeats its state at 10 Hz. A
request is made only when the count goes up, for the newest event; events skipped between two
state messages are counted as missed, never played or invented. The first message seen is the
baseline: it is the state of events that happened before this node started. A count going back
means the base node restarted: a new baseline. An event older than ``max_age_s`` on arrival is
not played (never felt late). A release is deliberate: it has no vibration. The arrival at rest
(the base stopped after a drive, the arm may be used) is felt on the other hand than the drive's
own events: it marks the switch to arm work.

``base_text`` is the panel line for a state, in plain words. Refusals and stops are shown for
``RECENT_S`` after their event, with their age; an idle or switched-off base shows no line.
"""

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional

from .episode_feedback import Pattern

# BaseState.STATE_*, REASON_*, EVENT_*, LIMIT_* (checked against the message by the tests).
STATE_NAMES = {0: 'OFF', 1: 'IDLE', 2: 'DRIVING', 3: 'STOPPING', 4: 'REFUSED', 5: 'STOPPED'}
REASON_NAMES = {
    0: 'NONE', 1: 'OFF', 2: 'NOT_SIMULATION', 3: 'RELEASED', 4: 'INPUT_STALE',
    5: 'CONTROLLER_MISSING', 6: 'SESSION_CHANGED', 7: 'SEQ_BACKWARDS', 8: 'CLUTCH_ENGAGED',
    9: 'RIGHT_GRIP_HELD', 10: 'TELEOP_UNKNOWN', 11: 'ODOM_MISSING', 12: 'RESET_ACTIVE',
    13: 'RESET_UNKNOWN', 14: 'OTHER_PUBLISHER', 15: 'TOOL_POSE_UNKNOWN', 16: 'SHUTDOWN',
}
EVENT_NAMES = {0: 'NONE', 1: 'START', 2: 'REFUSED', 3: 'STOPPED', 4: 'RELEASED', 5: 'AT_REST'}
LIMIT_TOOL_SPEED = 3
# The events that have a vibration pattern (a release has none).
EVENTS = ('START', 'REFUSED', 'STOPPED', 'AT_REST')

RECENT_S = 30.0          # a refusal or stop stays on the panel this long
MAX_PUBLISHERS_CHARS = 40

DEFAULT_PATTERNS: Dict[str, Pattern] = {
    'START': Pattern(((0.6, 100, 0),), 2),                          # one short pulse
    'REFUSED': Pattern(((1.0, 350, 200), (1.0, 350, 0)), 3),        # as the reset's refusal
    'STOPPED': Pattern(((1.0, 350, 200), (1.0, 350, 0)), 3),        # a stop by a fault: as REFUSED
    'AT_REST': Pattern(((0.5, 60, 0),), 2),                         # one short pulse, other hand
}
# The drive is the left hand's; the arm is the right's: the at-rest pulse is felt where the
# arm work starts.
DEFAULT_HANDS: Dict[str, str] = {
    'START': 'left', 'REFUSED': 'left', 'STOPPED': 'left', 'AT_REST': 'right',
}

DRIVING_TEXT = 'Base: driving, release left grip to stop (stick up = robot front)'
TURN_IGNORED_TEXT = 'Base: driving; turn ignored, centre the left stick to turn'
SLOWED_SUFFIX = ' (slowed: tool speed)'
STOPPING_TEXT = 'Base: stopping…'
# The words after "Base refused: " or "Base stopped: ", per BaseState.REASON_*.
REASON_WORDS = {
    'OFF': 'off (parameter)',
    'NOT_SIMULATION': 'simulation only',
    'RELEASED': 'released',
    'INPUT_STALE': 'headset input late or lost',
    'CONTROLLER_MISSING': 'left controller missing',
    'SESSION_CHANGED': 'headset session changed',
    'SEQ_BACKWARDS': 'input out of order',
    'CLUTCH_ENGAGED': 'release the arm clutch (right grip) first',
    'RIGHT_GRIP_HELD': 'release the right grip (arm clutch) first',
    'TELEOP_UNKNOWN': 'arm state unknown',
    'ODOM_MISSING': 'base odometry missing',
    'RESET_ACTIVE': 'reset running',
    'RESET_UNKNOWN': 'reset state unknown',
    'OTHER_PUBLISHER': 'another node publishes the base command',
    'TOOL_POSE_UNKNOWN': 'arm posture unknown (tool speed limit)',
    'SHUTDOWN': 'base node stopping',
}


def _publishers(other_text: str) -> str:
    names = ' '.join(str(other_text).split())
    if len(names) > MAX_PUBLISHERS_CHARS:
        names = names[:MAX_PUBLISHERS_CHARS - 1] + '…'
    return names


def base_text(state: int, reason: int, mode: int, turn_ignored: bool, active_limit: int,
              event_age_s: float, other_text: str = '') -> str:
    """
    Return the panel line for a base state ('' for no line).

    ``mode`` is accepted with the state; the words do not depend on it. ``other_text`` names
    what else publishes the base command (REASON_OTHER_PUBLISHER); it is clipped. A refusal
    and a stop are shown for ``RECENT_S`` after their event, with " (N s ago)"; with no known
    event age, or after that time, there is no line. An idle or switched-off base has no line.
    A code this module does not know gives a generic line, never an error.
    """
    state_name = STATE_NAMES.get(int(state))
    if state_name in ('OFF', 'IDLE'):
        return ''
    if state_name == 'DRIVING':
        text = TURN_IGNORED_TEXT if turn_ignored else DRIVING_TEXT
        return text + (SLOWED_SUFFIX if int(active_limit) == LIMIT_TOOL_SPEED else '')
    if state_name == 'STOPPING':
        return STOPPING_TEXT
    kind = {'REFUSED': 'refused', 'STOPPED': 'stopped'}.get(state_name)
    words = REASON_WORDS.get(REASON_NAMES.get(int(reason), ''))
    if kind is None:
        return f'Base: state {int(state)}, reason {int(reason)}'
    if words is None:
        text = f'Base: state {int(state)}, reason {int(reason)}'
    else:
        text = f'Base {kind}: {words}'
        names = _publishers(other_text) if REASON_NAMES.get(int(reason)) == 'OTHER_PUBLISHER' \
            else ''
        if names:
            text += f': {names}'
    age = float(event_age_s)
    if not math.isfinite(age) or age >= RECENT_S:
        return ''
    return f'{text} ({int(max(0.0, age))} s ago)'


@dataclass(frozen=True)
class BaseSample:
    """The fields of one BaseState message the event logic uses."""

    event_count: int
    last_event: int
    event_wall_time: float      # host wall clock of the newest event; NaN when none


@dataclass(frozen=True)
class BaseRequest:
    """One vibration request for a base event."""

    event: str
    hand: str
    pattern: Pattern
    source_count: int
    missed: int
    wall_time: float            # of the event


class BaseEvents:
    """Turns successive BaseState samples into at most one request per new event."""

    def __init__(self, patterns: Optional[Mapping[str, Pattern]] = None,
                 max_age_s: float = 0.75, hands: Optional[Mapping[str, str]] = None) -> None:
        self.patterns = dict(DEFAULT_PATTERNS if patterns is None else patterns)
        self.hands = {**DEFAULT_HANDS, **(hands or {})}
        for event, hand in self.hands.items():
            if hand not in ('left', 'right'):
                raise ValueError(f"base hand for {event} must be 'left' or 'right', not {hand!r}")
        self.max_age_s = float(max_age_s)
        self._count: Optional[int] = None
        self.missed_total = 0
        self.too_old = 0
        self.baselines = 0

    def update(self, s: BaseSample, wall_now: float) -> Optional[BaseRequest]:
        """Take one state message; return the request for its new event, if any."""
        count = int(s.event_count)
        last = self._count
        self._count = count
        if last is None or count < last:
            # First message, or a count going back (the base node restarted): a new baseline.
            # Nothing in it is new to this node.
            self.baselines += 1
            return None
        steps = count - last
        if steps <= 0:
            return None                                   # a repeat
        missed = steps - 1
        self.missed_total += missed
        name = EVENT_NAMES.get(int(s.last_event), '')
        pattern = self.patterns.get(name)
        hand = self.hands.get(name)
        if pattern is None or pattern.silent or hand is None:
            return None                                   # NONE, a release, an unknown event
        wall = float(s.event_wall_time)
        if not math.isfinite(wall) or wall_now - wall > self.max_age_s:
            self.too_old += 1                             # never played late (or age unknown)
            return None
        return BaseRequest(name, hand, pattern, count, missed, wall)
