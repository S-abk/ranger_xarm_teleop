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
What the operator's panel shows, in plain words (no ROS).

``StatusTracker`` takes every teleop state (50 Hz) and episode state message and keeps what a
5 Hz sample would lose: the newest fault hold (latched for ``fault_latch_s`` even if it
lasted one cycle) and the most severe limit (held ``limit_hold_s`` after it clears, so the
panel does not flicker). ``snapshot`` gives the fields of OperatorStatus, the reset part too. A
source older than its staleness limit is reported as not valid; its fields are then defaults,
never old values.

``SimSpeed`` takes the (ROS stamp, wall time) pair of each status and says, in words, when the
ROS clock runs much slower than real time (a slow simulator: arm data then arrives late).
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict, Optional

from .episode_feedback import EVENT_NAMES
from .reset_feedback import reset_text

NAN = float('nan')

# TeleopState.HOLD_* (checked against the message definition by the unit tests).
HOLD_NAMES = {
    0: 'NONE', 1: 'MOTION_DISABLED', 2: 'NOT_ARMED', 3: 'NOT_OWNER', 4: 'NOT_CALIBRATED',
    5: 'CLUTCH_RELEASED', 6: 'INPUT_STALE', 7: 'POSE_ABSENT', 8: 'POSE_EMULATED',
    9: 'OUT_OF_VIEW', 10: 'HEAD_ABSENT', 11: 'HEAD_EMULATED', 12: 'RESET_CHANGED',
    13: 'SESSION_CHANGED', 14: 'SEQ_BACKWARDS', 15: 'INPUT_JUMP', 16: 'STEP_REJECTED',
    17: 'TOOL_POSE_STALE', 18: 'SERVO_NOT_READY', 19: 'SERVO_HALTED', 20: 'SOLVER',
}
HOLD_TEXT = {
    'NONE': '',
    'MOTION_DISABLED': 'Motion disabled (parameter)',
    'NOT_ARMED': 'Not armed: press B',
    'NOT_OWNER': 'Waiting for the arm (MoveIt owns it)',
    'NOT_CALIBRATED': 'Not calibrated: point right hand forward, press A',
    'CLUTCH_RELEASED': 'Ready: hold right grip to move',
    'INPUT_STALE': 'Headset input late or lost',
    'POSE_ABSENT': 'Right controller not tracked',
    'POSE_EMULATED': 'Right controller tracking poor',
    'OUT_OF_VIEW': 'Right controller out of view: look at it',
    'HEAD_ABSENT': 'Headset tracking lost',
    'HEAD_EMULATED': 'Headset tracking poor',
    'RESET_CHANGED': 'View was reset: release grip, recalibrate (A)',
    'SESSION_CHANGED': 'New headset session: release grip, recalibrate (A)',
    'SEQ_BACKWARDS': 'Input out of order: release and re-grip',
    'INPUT_JUMP': 'Hand moved too fast: release and re-grip',
    'STEP_REJECTED': 'Target jumped too far: release and re-grip',
    'TOOL_POSE_STALE': 'Arm position late: wait',
    'SERVO_NOT_READY': 'Arm controller starting: wait',
    'SERVO_HALTED': 'Arm halted near a collision or joint limit: disarm (B), then use the '
                    'retreat tool',
    'SOLVER': 'Arm state disagrees with command: release, check arm',
}
# Holds that are the system's state, not a fault; every other hold is a fault and is latched.
STATE_HOLDS = frozenset({'NONE', 'MOTION_DISABLED', 'NOT_ARMED', 'NOT_OWNER',
                         'NOT_CALIBRATED', 'CLUTCH_RELEASED'})

# TeleopState.LIMIT_*.
LIMIT_NAMES = {
    0: 'NONE', 1: 'OFFSET_TRANSLATION', 2: 'OFFSET_ROTATION', 3: 'KEEP_OUT', 4: 'SERVO',
    5: 'LEAD', 6: 'TOOL_SPEED', 7: 'JOINT_VELOCITY', 8: 'JOINT_POSITION', 9: 'LINK_KEEPOUT',
}
LIMIT_TEXT = {
    'NONE': '',
    'OFFSET_TRANSLATION': 'reach range',
    'OFFSET_ROTATION': 'rotation range',
    'KEEP_OUT': 'keep-out zone',
    'SERVO': 'arm controller slowing',
    'LEAD': 'arm lagging hand',
    'TOOL_SPEED': 'speed',
    'JOINT_VELOCITY': 'joint speed',
    'JOINT_POSITION': 'joint at its end',
    'LINK_KEEPOUT': 'near robot body',
}
# Higher is more severe; the panel shows the most severe limit of the hold window.
LIMIT_SEVERITY = {
    'NONE': 0, 'TOOL_SPEED': 1, 'JOINT_VELOCITY': 1, 'LEAD': 2, 'OFFSET_TRANSLATION': 3,
    'OFFSET_ROTATION': 3, 'SERVO': 4, 'JOINT_POSITION': 5, 'KEEP_OUT': 6, 'LINK_KEEPOUT': 6,
}

# TeleopState.GRIPPER_HOLD_*.
GRIPPER_HOLD_NAMES = {
    0: 'NONE', 1: 'DISABLED', 2: 'NOT_ARMED', 3: 'NOT_OWNER', 4: 'INPUT', 5: 'NO_REFERENCE',
    6: 'AWAIT_TRIGGER', 7: 'KEEP_OUT', 8: 'JOINTS_STALE',
}
GRIPPER_HOLD_TEXT = {
    'NONE': '',
    'DISABLED': 'off',
    'NOT_ARMED': 'not armed',
    'NOT_OWNER': 'waiting for the gripper',
    'INPUT': 'input lost: holding',
    'NO_REFERENCE': 'waiting for its opening',
    'AWAIT_TRIGGER': 'pull the trigger past 60 % to take over',
    'KEEP_OUT': 'stopped: near robot body',
    'JOINTS_STALE': 'joint states late: holding',
}
# GripperCommand.REQUEST_*.
GRIPPER_REQUEST_TEXT = {0: 'still', 1: 'opening', 2: 'closing'}

# EpisodeState.STATE_*.
EPISODE_STATE_NAMES = {0: 'NOT_READY', 1: 'IDLE', 2: 'RECORDING', 3: 'REVIEW'}

REFUSAL_TEXT = (
    # (start of the episode node's detail, operator words)
    ('teleop is not armed', 'arm first (B)'),
    ('teleop does not own the arm yet', 'arm not handed over yet'),
    ('keep (Y) or discard (hold Y) the last episode first',
     'keep (Y) or discard (hold Y) first'),
    ('the recorder is busy with the last press', 'recorder busy, press again'),
)
AUTO_STOP_TEXT = {
    'disarmed': 'teleop was disarmed',
    'teleop state lost': 'teleop state lost',
    'disk low': 'disk almost full',
}
MAX_TEXT = 160


def _clip(text: str) -> str:
    text = ' '.join(str(text).split())
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT - 1] + '…'


def refusal_words(detail: str) -> str:
    """Return the operator's words for a refusal detail of the episode node."""
    detail = str(detail).strip()
    if detail.startswith('not ready:'):
        detail = detail[len('not ready:'):].strip()
    first = detail.split('; ')[0] if detail else ''
    for prefix, words in REFUSAL_TEXT:
        if first.startswith(prefix):
            return words
    rest = detail.count('; ')
    return _clip(first + (f' (+{rest} more)' if rest else '')) if first else 'not ready'


def auto_stop_words(detail: str) -> str:
    """Return the operator's words for why an episode stopped by itself."""
    detail = str(detail).strip()
    return AUTO_STOP_TEXT.get(detail, _clip(detail) or 'unknown reason')


def event_text(event: str, index: int, detail: str) -> str:
    """Return the panel words for the newest episode event."""
    if event == 'START':
        return f'Episode {index} started'
    if event == 'STOP':
        return f'Episode {index} stopped'
    if event == 'KEEP':
        return f'Episode {index} kept'
    if event == 'DISCARD':
        return f'Episode {index} discarded'
    if event == 'MARK':
        return 'Mark set'
    if event == 'REFUSED':
        return 'Refused: ' + refusal_words(detail)
    if event == 'AUTO_STOP':
        return f'Episode {index} stopped by itself: ' + auto_stop_words(detail)
    return ''


def code_name(names: Dict[int, str], code: int) -> str:
    return names.get(int(code), f'UNKNOWN_{int(code)}')


def hold_text(code: int) -> str:
    name = code_name(HOLD_NAMES, code)
    return HOLD_TEXT.get(name, f'Hold {name}')


def limit_text(code: int) -> str:
    name = code_name(LIMIT_NAMES, code)
    return LIMIT_TEXT.get(name, f'limit {name}')


@dataclass(frozen=True)
class TeleopSample:
    """The fields of one TeleopState message this module uses."""

    motion_enabled: bool
    armed: bool
    clutch_engaged: bool
    calibrated: bool
    translation_scale: float
    hold_reason: int
    active_limit: int
    gripper_enabled: bool
    gripper_active: bool
    gripper_hold: int
    gripper_request: int
    gripper_measured_valid: bool
    gripper_measured: float


@dataclass(frozen=True)
class EpisodeView:
    """The fields of one EpisodeState message the panel shows."""

    state: int
    episode_index: int
    elapsed: float
    kept_count: int
    discarded_count: int
    discard_armed: bool
    last_event: int
    detail: str
    event_wall_time: float      # host wall clock of the newest event
    event_count: int
    disk_minutes_left: float


@dataclass(frozen=True)
class ResetView:
    """The fields of one ResetState message the panel shows."""

    state: int
    reason: int
    posture: str
    moveit_error_code: int
    last_event: int
    event_wall_time: float      # host wall clock of the newest event; NaN when none


@dataclass
class Snapshot:
    """The fields of one OperatorStatus message (same names)."""

    teleop_valid: bool = False
    teleop_age: float = NAN
    motion_enabled: bool = False
    armed: bool = False
    clutch_engaged: bool = False
    calibrated: bool = False
    translation_scale: float = NAN
    hold_reason: int = 0
    hold_text: str = ''
    last_fault: int = 0
    last_fault_text: str = ''
    last_fault_age: float = NAN
    limit: int = 0
    limit_text: str = ''
    gripper_enabled: bool = False
    gripper_active: bool = False
    gripper_hold: int = 0
    gripper_text: str = ''
    gripper_closed_fraction: float = NAN
    episode_valid: bool = False
    episode_age: float = NAN
    episode_state: int = 0
    episode_index: int = 0
    elapsed: float = 0.0
    kept_count: int = 0
    discarded_count: int = 0
    discard_armed: bool = False
    last_event: int = 0
    last_event_text: str = ''
    last_event_age: float = NAN
    last_refusal_text: str = ''
    last_refusal_age: float = NAN
    events_missed: int = 0
    disk_minutes_left: float = NAN
    reset_valid: bool = False
    reset_age: float = NAN
    reset_state: int = 0
    reset_reason: int = 0
    reset_text: str = ''
    reset_event_age: float = NAN


@dataclass
class StatusTracker:
    """Keeps the latched panel state from every teleop and episode state message."""

    teleop_stale_s: float = 0.5
    episode_stale_s: float = 1.0
    fault_latch_s: float = 10.0
    limit_hold_s: float = 1.0
    reset_stale_s: float = 0.5
    gripper_open: float = NAN
    gripper_closed: float = NAN
    _teleop: Optional[TeleopSample] = None
    _teleop_at: float = NAN
    _fault: int = 0
    _fault_at: float = NAN
    _limit: int = 0
    _limit_at: float = NAN
    _episode: Optional[EpisodeView] = None
    _episode_at: float = NAN
    _refusal: str = ''
    _refusal_wall: float = NAN
    _refusal_count: int = -1
    _reset: Optional[ResetView] = None
    _reset_at: float = NAN
    events_missed: int = 0
    counts: Dict[str, int] = field(default_factory=lambda: {'teleop': 0, 'episode': 0})

    def on_teleop(self, s: TeleopSample, now: float) -> None:
        """Take one TeleopState (``now``: monotonic arrival time, s)."""
        self.counts['teleop'] += 1
        self._teleop, self._teleop_at = s, now
        if code_name(HOLD_NAMES, s.hold_reason) not in STATE_HOLDS:
            self._fault, self._fault_at = int(s.hold_reason), now
        limit = int(s.active_limit)
        if limit:
            held = self._limit
            expired = not (now - self._limit_at <= self.limit_hold_s)
            if (expired or limit == held
                    or self._severity(limit) >= self._severity(held)):
                self._limit, self._limit_at = limit, now

    def on_episode(self, e: EpisodeView, now: float) -> None:
        """Take one EpisodeState (``now``: monotonic arrival time, s)."""
        self.counts['episode'] += 1
        self._episode, self._episode_at = e, now
        name = EVENT_NAMES.get(int(e.last_event), '')
        if name == 'REFUSED' and e.event_count != self._refusal_count:
            self._refusal = refusal_words(e.detail)
            self._refusal_wall = e.event_wall_time
            self._refusal_count = e.event_count

    def on_reset(self, r: ResetView, now: float) -> None:
        """Take one ResetState (``now``: monotonic arrival time, s)."""
        self._reset, self._reset_at = r, now

    @staticmethod
    def _severity(code: int) -> int:
        return LIMIT_SEVERITY.get(code_name(LIMIT_NAMES, code), 1)

    def gripper_fraction(self, s: TeleopSample) -> float:
        span = self.gripper_closed - self.gripper_open
        if not (s.gripper_measured_valid and math.isfinite(s.gripper_measured)
                and math.isfinite(span) and abs(span) > 1e-9):
            return NAN
        return min(1.0, max(0.0, (s.gripper_measured - self.gripper_open) / span))

    def gripper_words(self, s: TeleopSample) -> str:
        if not s.gripper_enabled:
            return ''
        hold = code_name(GRIPPER_HOLD_NAMES, s.gripper_hold)
        if hold != 'NONE':
            words = GRIPPER_HOLD_TEXT.get(hold, hold.lower())
        else:
            words = GRIPPER_REQUEST_TEXT.get(int(s.gripper_request), 'still')
        fraction = self.gripper_fraction(s)
        if math.isfinite(fraction):
            words += f', {round(100 * fraction)} % closed'
        elif s.gripper_measured_valid and math.isfinite(s.gripper_measured):
            words += f', at {s.gripper_measured:.2f}'
        return words

    def snapshot(self, now: float, wall_now: float) -> Snapshot:
        """Return the panel state at ``now`` (monotonic) and ``wall_now`` (host wall clock)."""
        out = Snapshot(events_missed=int(self.events_missed))
        s = self._teleop
        if s is not None:
            out.teleop_age = now - self._teleop_at
        if s is not None and out.teleop_age <= self.teleop_stale_s:
            out.teleop_valid = True
            out.motion_enabled = bool(s.motion_enabled)
            out.armed = bool(s.armed)
            out.clutch_engaged = bool(s.clutch_engaged)
            out.calibrated = bool(s.calibrated)
            out.translation_scale = float(s.translation_scale)
            out.hold_reason = int(s.hold_reason)
            out.hold_text = hold_text(s.hold_reason)
            out.gripper_enabled = bool(s.gripper_enabled)
            out.gripper_active = bool(s.gripper_active)
            out.gripper_hold = int(s.gripper_hold)
            out.gripper_text = self.gripper_words(s)
            out.gripper_closed_fraction = self.gripper_fraction(s)
            if self._fault and now - self._fault_at <= self.fault_latch_s:
                out.last_fault = self._fault
                out.last_fault_text = hold_text(self._fault)
                out.last_fault_age = max(0.0, now - self._fault_at)
            if self._limit and now - self._limit_at <= self.limit_hold_s:
                out.limit = self._limit
                out.limit_text = limit_text(self._limit)
        e = self._episode
        if e is not None:
            out.episode_age = now - self._episode_at
        if e is not None and out.episode_age <= self.episode_stale_s:
            out.episode_valid = True
            out.episode_state = int(e.state)
            out.episode_index = int(e.episode_index)
            out.elapsed = float(e.elapsed)
            out.kept_count = int(e.kept_count)
            out.discarded_count = int(e.discarded_count)
            out.discard_armed = bool(e.discard_armed)
            out.last_event = int(e.last_event)
            name = EVENT_NAMES.get(int(e.last_event), '')
            out.last_event_text = event_text(name, e.episode_index, e.detail)
            if name and name != 'NONE':
                out.last_event_age = max(0.0, wall_now - e.event_wall_time)
            out.disk_minutes_left = float(e.disk_minutes_left)
            if self._refusal:
                out.last_refusal_text = self._refusal
                out.last_refusal_age = max(0.0, wall_now - self._refusal_wall)
        r = self._reset
        if r is not None:
            out.reset_age = now - self._reset_at
        if r is not None and out.reset_age <= self.reset_stale_s:
            out.reset_valid = True
            out.reset_state = int(r.state)
            out.reset_reason = int(r.reason)
            if int(r.last_event) != 0 and math.isfinite(r.event_wall_time):
                out.reset_event_age = max(0.0, wall_now - r.event_wall_time)
            out.reset_text = reset_text(r.state, r.reason, r.posture, r.moveit_error_code,
                                        out.reset_event_age)
        return out


class SimSpeed:
    """
    How fast the ROS clock runs against the host wall clock, from successive status messages.

    The factor is measured over the last ``window_s`` of wall time. It is "slow" once it falls
    below ``slow_below`` and stays so until it rises above ``clear_above`` (no flicker). A ROS
    clock that goes back (simulation restarted) or a wall-clock gap or jump of ``max_gap_s``
    starts the measurement again. On hardware the ROS clock is the wall clock: never slow.
    """

    SHOWN_STEP = 0.1        # the shown factor changes only when the measured one moved this much
    PAUSED_BELOW = 0.05     # below this the simulator is said to be paused or stalled

    def __init__(self, window_s: float = 3.0, slow_below: float = 0.6,
                 clear_above: float = 0.7, max_gap_s: float = 2.0) -> None:
        if not 0.0 < slow_below < clear_above <= 1.0:
            raise ValueError('need 0 < slow_below < clear_above <= 1')
        if not window_s > 0.0:
            raise ValueError('window_s must be positive')
        self.window_s, self.slow_below, self.clear_above = window_s, slow_below, clear_above
        self.max_gap_s = max_gap_s
        self._samples: deque = deque()
        self.factor: Optional[float] = None
        self.slow = False
        self.shown: Optional[float] = None   # the factor the text gives, held (no flicker)

    def reset(self) -> None:
        self._samples.clear()
        self.factor = None
        self.slow = False
        self.shown = None

    def update(self, ros_s: float, wall_s: float) -> Optional[float]:
        """Take one (ROS time, wall time) pair, s; return the factor, None until measured."""
        if not (math.isfinite(ros_s) and math.isfinite(wall_s)) or ros_s <= 0.0:
            return self.factor          # no time yet (an unset stamp, or no /clock yet)
        if self._samples:
            last_ros, last_wall = self._samples[-1]
            if ros_s < last_ros or not 0.0 < wall_s - last_wall < self.max_gap_s:
                self.reset()
        self._samples.append((ros_s, wall_s))
        while len(self._samples) > 2 and wall_s - self._samples[1][1] >= self.window_s:
            self._samples.popleft()
        first_ros, first_wall = self._samples[0]
        span = wall_s - first_wall
        if span < 0.8 * self.window_s:
            return None                 # not yet a full window (start, or after a reset)
        self.factor = (ros_s - first_ros) / span
        if self.factor < self.slow_below:
            self.slow = True
        elif self.factor > self.clear_above:
            self.slow = False
        if not self.slow:
            self.shown = None
        elif self.shown is None or abs(self.factor - self.shown) >= self.SHOWN_STEP:
            # A bursty clock makes the measured factor jitter; the text (and so the panel
            # texture) changes only when it really moved.
            self.shown = max(0.0, self.factor)
        return self.factor

    def text(self) -> str:
        """Return the panel line while slow, else ''; it changes only when the factor moved 0.1."""
        if not self.slow or self.shown is None:
            return ''
        if self.shown < self.PAUSED_BELOW:
            return 'Simulator paused or stalled'
        return f'Simulator slow: {self.shown:.1f}x real time'
