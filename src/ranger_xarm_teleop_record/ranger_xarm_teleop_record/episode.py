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
Episode control from the left controller (no ROS).

Buttons (contract, Recording): X starts or stops an episode; after a stop, Y keeps it and Y
held for ``discard_hold_s`` discards it (marked, never deleted); the left trigger marks a
moment inside an episode. The left thumbstick and grip are reserved for the base and the
thumbstick click is unused, so none of them is read here.

Presses are read as levels: an edge is a change between two received Joy messages. A Joy
message that never arrives is not a release, so one lost message inside a press neither
loses nor doubles it. Hold times use the caller's monotonic clock.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import List, Optional, Sequence

from ranger_xarm_teleop_bridge.joy_layout import AXIS_NAMES, BUTTON_NAMES

X_BUTTON = BUTTON_NAMES.index('left_primary')
Y_BUTTON = BUTTON_NAMES.index('left_secondary')
MARK_AXIS = AXIS_NAMES.index('left_trigger')


class State(IntEnum):
    """Episode states (EpisodeState.STATE_*)."""

    NOT_READY = 0
    IDLE = 1
    RECORDING = 2
    REVIEW = 3


class Event(IntEnum):
    """Episode events (EpisodeState.EVENT_*)."""

    NONE = 0
    START = 1
    STOP = 2
    KEEP = 3
    DISCARD = 4
    MARK = 5
    REFUSED = 6
    AUTO_STOP = 7


@dataclass(frozen=True)
class Buttons:
    """The left controller's episode inputs in one Joy message."""

    x: bool
    y: bool
    trigger: float          # 0..1; NaN when the controller has no trigger value


def buttons_from_joy(buttons: Sequence[int], axes: Sequence[float]) -> Optional[Buttons]:
    """Return the episode inputs of one Joy message, or None when it is too short."""
    if len(buttons) <= max(X_BUTTON, Y_BUTTON) or len(axes) <= MARK_AXIS:
        return None
    trigger = float(axes[MARK_AXIS])
    return Buttons(bool(buttons[X_BUTTON]), bool(buttons[Y_BUTTON]), trigger)


@dataclass(frozen=True)
class Outcome:
    """One event; ``command`` is what the caller must do (start/stop the recorder, ...)."""

    event: Event
    detail: str = ''


@dataclass
class Control:
    """The episode state machine. The caller performs the recorder calls."""

    discard_hold_s: float = 1.0
    mark_on: float = 0.6
    mark_off: float = 0.25
    state: State = State.NOT_READY
    busy: bool = False                      # a recorder call is in flight
    _x: bool = False
    _y: bool = False
    _trigger_high: bool = False
    _y_down_at: Optional[float] = None      # a Y press that began in REVIEW, not yet used
    _seen: bool = False
    pending: List[Outcome] = field(default_factory=list)

    def set_ready(self, ready: bool, detail: str = '') -> List[Outcome]:
        """Move between NOT_READY and IDLE; never interrupts an episode."""
        if self.state == State.NOT_READY and ready:
            self.state = State.IDLE
        elif self.state == State.IDLE and not ready:
            self.state = State.NOT_READY
        return []

    @property
    def discard_armed(self) -> bool:
        """Return True while Y is held in REVIEW and the discard has not happened yet."""
        return self.state == State.REVIEW and self._y and self._y_down_at is not None

    def joy(self, b: Buttons, now: float, start_problems: Sequence[str] = ()) -> List[Outcome]:
        """Take one Joy message received at ``now``; return the events it causes."""
        out: List[Outcome] = []
        first = not self._seen
        self._seen = True
        x_rise = b.x and not self._x and not first
        y_rise = b.y and not self._y and not first
        y_fall = self._y and not b.y
        self._x, self._y = b.x, b.y
        if x_rise:
            out.append(self._on_x(start_problems))
        if y_rise:
            if self.state == State.REVIEW and not self.busy:
                self._y_down_at = now
            else:
                self._y_down_at = None
        if y_fall and self._y_down_at is not None:
            held = now - self._y_down_at
            self._y_down_at = None
            if self.state == State.REVIEW:
                out.append(self._resolve(Event.DISCARD if held >= self.discard_hold_s
                                         else Event.KEEP))
        if math.isfinite(b.trigger):
            if not self._trigger_high and b.trigger >= self.mark_on:
                self._trigger_high = True
                if self.state == State.RECORDING and not self.busy:
                    out.append(Outcome(Event.MARK))
            elif self._trigger_high and b.trigger <= self.mark_off:
                self._trigger_high = False
        out.extend(self.tick(now))
        return out

    def tick(self, now: float) -> List[Outcome]:
        """Discard once Y has been held long enough in REVIEW."""
        if self.discard_armed and now - self._y_down_at >= self.discard_hold_s:
            self._y_down_at = None
            return [self._resolve(Event.DISCARD)]
        return []

    def auto_stop(self, reason: str) -> List[Outcome]:
        """Stop a running episode by itself (disarm, lost teleop state, disk, recorder)."""
        if self.state != State.RECORDING or self.busy:
            return []
        self.state = State.REVIEW
        self.busy = True
        return [Outcome(Event.AUTO_STOP, reason)]

    def done(self, ok: bool, detail: str = '') -> List[Outcome]:
        """Report the end of the recorder call for the last START, STOP or AUTO_STOP."""
        self.busy = False
        if ok:
            return []
        if self.state == State.RECORDING:            # resume failed: no episode
            self.state = State.IDLE
            return [Outcome(Event.REFUSED, f'recorder did not start: {detail}')]
        return []                                    # a failed stop stays in REVIEW, flagged

    def _on_x(self, start_problems: Sequence[str]) -> Outcome:
        if self.busy:
            return Outcome(Event.REFUSED, 'the recorder is busy with the last press')
        if self.state == State.IDLE:
            if start_problems:
                return Outcome(Event.REFUSED, '; '.join(start_problems))
            self.state = State.RECORDING
            self.busy = True
            return Outcome(Event.START)
        if self.state == State.RECORDING:
            self.state = State.REVIEW
            self.busy = True
            return Outcome(Event.STOP)
        if self.state == State.REVIEW:
            return Outcome(Event.REFUSED, 'keep (Y) or discard (hold Y) the last episode first')
        return Outcome(Event.REFUSED, 'not ready: ' + ('; '.join(start_problems) or 'starting'))

    def _resolve(self, event: Event) -> Outcome:
        self.state = State.IDLE
        return Outcome(event)
