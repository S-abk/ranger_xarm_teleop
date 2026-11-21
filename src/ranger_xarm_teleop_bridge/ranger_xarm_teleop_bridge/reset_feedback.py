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
Reset events to vibration requests and the panel line, exactly once per event (no ROS).

The reset node counts its events (``ResetState.event_count``) and repeats its state at 10 Hz. A
request is made only when the count goes up, for the newest event; events skipped between two
state messages are counted as missed, never played or invented. The first message seen is the
baseline: it is the state of events that happened before this node started. A count going back
means the reset node restarted: a new baseline. An event older than ``max_age_s`` on arrival is
not played (never felt late). A release is deliberate: it has no vibration.

``reset_text`` is the panel line for a state, in plain words. Refusals, stops and arrivals are
shown for ``RECENT_S`` after their event, with their age; an idle reset shows no line.
"""

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional

from .episode_feedback import Pattern

# ResetState.STATE_*, REASON_*, EVENT_* (checked against the message definition by the tests).
STATE_NAMES = {0: 'IDLE', 1: 'WAITING', 2: 'REFUSED', 3: 'PLANNING', 4: 'MOVING',
               5: 'STOPPING', 6: 'STOPPED', 7: 'ARRIVED'}
REASON_NAMES = {
    0: 'NONE', 1: 'OFF', 2: 'NOT_SIMULATION', 3: 'ARMED', 4: 'NOT_MOVEIT_OWNER',
    5: 'ARMED_UNKNOWN', 6: 'RECORDING', 7: 'REVIEW', 8: 'EPISODE_UNKNOWN',
    9: 'WATCHDOG_ABSENT', 10: 'MOVEIT_NOT_READY', 11: 'PLAN_FAILED', 12: 'START_IN_COLLISION',
    13: 'BUSY', 14: 'JOINTS_UNKNOWN', 20: 'REACHED', 21: 'ALREADY_THERE', 30: 'RELEASED',
    31: 'INPUT_STALE', 32: 'CONTROLLER_MISSING', 33: 'SESSION_CHANGED', 34: 'SEQ_BACKWARDS',
    35: 'TELEOP_ARMED', 36: 'OWNER_CHANGED', 37: 'EPISODE_STARTED', 38: 'WATCHDOG_LOST',
    39: 'HEARTBEAT_LATE', 40: 'CANCELLED_ELSEWHERE', 41: 'MOVEIT_FAILED', 42: 'NO_ANSWER',
    43: 'STATE_UNKNOWN',
}
EVENT_NAMES = {0: 'NONE', 1: 'START', 2: 'ARRIVED', 3: 'REFUSED', 4: 'STOPPED', 5: 'RELEASED'}
# The events that have a vibration pattern (a release has none).
EVENTS = ('START', 'ARRIVED', 'REFUSED', 'STOPPED')

RECENT_S = 30.0          # a refusal, stop or arrival stays on the panel this long
MAX_POSTURE_CHARS = 32

DEFAULT_PATTERNS: Dict[str, Pattern] = {
    'START': Pattern(((0.8, 200, 0),), 2),
    'ARRIVED': Pattern(((0.7, 500, 0),), 2),
    'REFUSED': Pattern(((1.0, 350, 200), (1.0, 350, 0)), 3),
    'STOPPED': Pattern(((1.0, 350, 200), (1.0, 350, 0)), 3),     # a stop by a fault: as REFUSED
}

# {posture} is the posture name; {code} is move_group's error code.
MOVING_TEXT = {
    'WAITING': 'Reset: keep holding\u2026',
    'PLANNING': 'Reset: planning to {posture}\u2026',
    'MOVING': 'Reset: moving to {posture}; release to stop',
    'STOPPING': 'Reset: stopping\u2026',
}
ARRIVED_TEXT = {
    'REACHED': 'Reset: at {posture}. Arm with B',
    'ALREADY_THERE': 'Reset: already at {posture}',
}
STOPPED_TEXT = {
    'RELEASED': 'Reset stopped: released. Pull back again to go on',
    'INPUT_STALE': 'Reset stopped: headset input late or lost',
    'CONTROLLER_MISSING': 'Reset stopped: right controller lost',
    'SESSION_CHANGED': 'Reset stopped: new headset session',
    'SEQ_BACKWARDS': 'Reset stopped: input out of order',
    'TELEOP_ARMED': 'Reset stopped: teleop was armed',
    'OWNER_CHANGED': 'Reset stopped: arm handed over',
    'EPISODE_STARTED': 'Reset stopped: an episode started',
    'WATCHDOG_LOST': 'Reset stopped: watchdog lost',
    'HEARTBEAT_LATE': 'Reset stopped: heartbeat late',
    'CANCELLED_ELSEWHERE': 'Reset stopped: cancelled elsewhere',
    'MOVEIT_FAILED': 'Reset stopped: MoveIt stopped the motion (code {code})',
    'NO_ANSWER': 'Reset stopped: MoveIt did not answer',
    'STATE_UNKNOWN': 'Reset stopped: arm or teleop state lost',
}
REFUSED_TEXT = {
    'OFF': 'Reset refused: off (parameter)',
    'NOT_SIMULATION': 'Reset refused: simulation only',
    'ARMED': 'Reset refused: disarm first (B)',
    'NOT_MOVEIT_OWNER': 'Reset refused: arm not handed to MoveIt yet',
    'ARMED_UNKNOWN': 'Reset refused: teleop state unknown',
    'RECORDING': 'Reset refused: recording, stop the episode (X)',
    'REVIEW': 'Reset refused: keep (Y) or discard (hold Y) first',
    'EPISODE_UNKNOWN': 'Reset refused: recording state unknown',
    'WATCHDOG_ABSENT': 'Reset refused: watchdog not running',
    'MOVEIT_NOT_READY': 'Reset refused: MoveIt not ready',
    'PLAN_FAILED': 'Reset refused: no path to {posture}; nothing moved',
    'START_IN_COLLISION': 'Reset refused: arm in collision; nothing moved. Disarm, use the '
                          'retreat tool',
    'BUSY': 'Reset refused: still stopping, wait',
    'JOINTS_UNKNOWN': 'Reset refused: arm position unknown',
}


def _posture_name(posture: str) -> str:
    name = ' '.join(str(posture).split())
    if not name:
        return 'the posture'
    return name if len(name) <= MAX_POSTURE_CHARS else name[:MAX_POSTURE_CHARS - 1] + '\u2026'


def reset_text(state: int, reason: int, posture: str, moveit_error_code: int,
               event_age_s: float) -> str:
    """
    Return the panel line for a reset state ('' for no line).

    A refusal, a stop and an arrival are shown for ``RECENT_S`` after their event, with
    " (N s ago)"; with no known event age, or after that time, there is no line. An idle
    reset has no line. A code this module does not know gives a generic line, never an error.
    """
    state_name = STATE_NAMES.get(int(state))
    reason_name = REASON_NAMES.get(int(reason))
    name = _posture_name(posture)
    if state_name == 'IDLE':
        return ''
    if state_name in MOVING_TEXT:
        return MOVING_TEXT[state_name].format(posture=name)
    table = {'REFUSED': REFUSED_TEXT, 'STOPPED': STOPPED_TEXT,
             'ARRIVED': ARRIVED_TEXT}.get(state_name)
    if table is None:
        return f'Reset: state {int(state)}, reason {int(reason)}'
    template = table.get(reason_name)
    text = (template.format(posture=name, code=int(moveit_error_code)) if template
            else f'Reset: state {int(state)}, reason {int(reason)}')
    age = float(event_age_s)
    if not math.isfinite(age) or age >= RECENT_S:
        return ''
    return f'{text} ({int(max(0.0, age))} s ago)'


@dataclass(frozen=True)
class ResetSample:
    """The fields of one ResetState message the event logic uses."""

    event_count: int
    last_event: int
    event_wall_time: float      # host wall clock of the newest event; NaN when none


@dataclass(frozen=True)
class ResetRequest:
    """One vibration request for a reset event."""

    event: str
    pattern: Pattern
    source_count: int
    missed: int
    wall_time: float            # of the event


class ResetEvents:
    """Turns successive ResetState samples into at most one request per new event."""

    def __init__(self, patterns: Optional[Mapping[str, Pattern]] = None,
                 max_age_s: float = 0.75) -> None:
        self.patterns = dict(DEFAULT_PATTERNS if patterns is None else patterns)
        self.max_age_s = float(max_age_s)
        self._count: Optional[int] = None
        self.missed_total = 0
        self.too_old = 0
        self.baselines = 0

    def update(self, s: ResetSample, wall_now: float) -> Optional[ResetRequest]:
        """Take one state message; return the request for its new event, if any."""
        count = int(s.event_count)
        last = self._count
        self._count = count
        if last is None or count < last:
            # First message, or a count going back (the reset node restarted): a new
            # baseline. Nothing in it is new to this node.
            self.baselines += 1
            return None
        steps = count - last
        if steps <= 0:
            return None                                   # a repeat
        missed = steps - 1
        self.missed_total += missed
        name = EVENT_NAMES.get(int(s.last_event), '')
        pattern = self.patterns.get(name)
        if pattern is None or pattern.silent:
            return None                                   # NONE, a release, an unknown event
        wall = float(s.event_wall_time)
        if not math.isfinite(wall) or wall_now - wall > self.max_age_s:
            self.too_old += 1                             # never played late (or age unknown)
            return None
        return ResetRequest(name, pattern, count, missed, wall)
