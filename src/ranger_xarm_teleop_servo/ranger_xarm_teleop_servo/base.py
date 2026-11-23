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
Driving the base from the headset (G-19), as pure logic.

* **The controls (contract, "Base driving").** Hold the left grip to drive. While it is held:
  left thumbstick up = robot forward, right = robot right, a diagonal = sideways travel along the
  stick, in the robot's own axes (not the operator's view); right thumbstick left or right =
  turn in place, taken only while the left stick is centred. The base runs one motion mode per
  command, so the command carries exactly one: when both sticks are deflected the left stick
  wins and ``turn_ignored`` says so. The right stick's x axis counts only while it dominates its
  y axis (a pull back, the reset's control, never turns the base).
* **Shaping.** Radial dead zone (output exactly 0.0 inside it), quadratic response, and a snap
  to the pure axis within ``snap_deg`` so that straight forward or sideways is exactly one axis.
  The linear cap bounds forward and sideways together (a diagonal is not faster); the turning cap
  is separate. A change of mode passes through zero.
* **Acceleration.** The command changes no faster than the acceleration limits, also when the
  stick returns to centre with the grip held (the command keeps being published, ramping down).
* **Hold-to-drive and the stop (G-19, INV-1).** Released grip, stale input (G-3's limit), a
  missing left controller, a session or sequence fault, the arm's clutch engaged or its state
  unknown, a reset under way or its state unknown, no fresh odometry, another publisher of the
  base command, an unknown arm posture while the tool speed limit is set, and the node's own
  shutdown: one zero command at once, then nothing. Exempt from the acceleration limit, like
  G-18's immediate stops; the base's own controller brakes, and its own command timeout backs
  this up. Pose-only conditions (controller out of view, emulated or absent pose) do not stop
  the base: the control is not a pose.
* **The arm and the base exclude each other (INV-2), decided on the input.** Both nodes see the
  same headset frame. A drive starts only if the right grip (the arm's clutch) is not held in the
  frame it starts on and the arm's fresh state says the clutch is released and no stop ramp
  runs; the arm's gate refuses the clutch while the left grip is held. The first to be held
  wins; both first held in one frame: neither moves. Teleop need not be armed to drive.
* **Never by itself.** A start needs a press: the grip seen released, then held. After any
  refusal or stop the grip must be released and pressed again.
* **Tool speed in the room (optional).** With ``tool_speed_cap`` set, the whole command is
  scaled down (direction kept) so that the tool, at its measured position in the base frame,
  moves no faster than the cap in the room: a turn moves it at the turning rate times its
  horizontal distance from the turning centre, a translation at the base's own speed.

The node (``base_node.py``) feeds this class with the input samples and what it knows of the
world at every cycle, and publishes what ``tick`` returns. No ROS imports.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

from ranger_xarm_teleop_bridge.joy_layout import AXIS_NAMES, BUTTON_NAMES

from .base_rest import AT_REST, UNKNOWN
from .gate import frame_fault, Hold
from .params import INPUT_TIMEOUT_S

LEFT_GRIP_BUTTON = BUTTON_NAMES.index('left_grip')
RIGHT_GRIP_BUTTON = BUTTON_NAMES.index('right_grip')
LEFT_TRIGGER_AXIS = AXIS_NAMES.index('left_trigger')        # NaN when there is no controller
LEFT_X_AXIS = AXIS_NAMES.index('left_thumbstick_x')         # negative to the left
LEFT_Y_AXIS = AXIS_NAMES.index('left_thumbstick_y')         # negative when pushed up (forward)
RIGHT_X_AXIS = AXIS_NAMES.index('right_thumbstick_x')
RIGHT_Y_AXIS = AXIS_NAMES.index('right_thumbstick_y')       # positive when pulled back (reset)
RIGHT_TRIGGER_AXIS = AXIS_NAMES.index('right_trigger')

# BaseState.STATE_* (checked against the message file by the unit tests).
OFF, IDLE, DRIVING, STOPPING, REFUSED, STOPPED = range(6)
STATE_NAMES = ('OFF', 'IDLE', 'DRIVING', 'STOPPING', 'REFUSED', 'STOPPED')
# BaseState.MODE_*.
MODE_NONE, MODE_TRANSLATE, MODE_TURN = 0, 1, 2
# BaseState.LIMIT_*.
LIMIT_NONE, LIMIT_CAP, LIMIT_ACCELERATION, LIMIT_TOOL_SPEED = 0, 1, 2, 3
# BaseState.REASON_*.
NONE = 0
(R_OFF, NOT_SIMULATION, RELEASED, INPUT_STALE, CONTROLLER_MISSING, SESSION_CHANGED,
 SEQ_BACKWARDS, CLUTCH_ENGAGED, RIGHT_GRIP_HELD, TELEOP_UNKNOWN, ODOM_MISSING, RESET_ACTIVE,
 RESET_UNKNOWN, OTHER_PUBLISHER, TOOL_POSE_UNKNOWN, SHUTDOWN) = range(1, 17)
REASON_NAMES = ('NONE', 'OFF', 'NOT_SIMULATION', 'RELEASED', 'INPUT_STALE', 'CONTROLLER_MISSING',
                'SESSION_CHANGED', 'SEQ_BACKWARDS', 'CLUTCH_ENGAGED', 'RIGHT_GRIP_HELD',
                'TELEOP_UNKNOWN', 'ODOM_MISSING', 'RESET_ACTIVE', 'RESET_UNKNOWN',
                'OTHER_PUBLISHER', 'TOOL_POSE_UNKNOWN', 'SHUTDOWN')
REASON_TEXT = {
    NONE: '',
    R_OFF: 'base driving is off by configuration',
    NOT_SIMULATION: "the robot description is not a simulator's (simulation only)",
    RELEASED: 'the left grip was released',
    INPUT_STALE: 'no fresh headset input',
    CONTROLLER_MISSING: 'no value for the left controller: it is missing',
    SESSION_CHANGED: 'the XR session changed',
    SEQ_BACKWARDS: "the input's sequence or time went backwards",
    CLUTCH_ENGAGED: "the arm's clutch is engaged or its stop still runs: release the right grip",
    RIGHT_GRIP_HELD: "the right grip (the arm's clutch) is held: release it first",
    TELEOP_UNKNOWN: "no fresh state of the arm's teleop node",
    ODOM_MISSING: 'no fresh odometry of the base',
    RESET_ACTIVE: 'a return to the ready posture is under way',
    RESET_UNKNOWN: 'a reset node runs and no fresh state of it is at hand',
    OTHER_PUBLISHER: 'something else publishes the base command',
    TOOL_POSE_UNKNOWN: "the tool speed limit is set and the arm's posture is unknown",
    SHUTDOWN: 'the base node is stopping',
}
# BaseState.EVENT_*.
EVENT_NONE, EVENT_START, EVENT_REFUSED, EVENT_STOPPED, EVENT_RELEASED, EVENT_AT_REST = range(6)

# ResetState.STATE_* in which a reset is under way (WAITING: its control is being held).
RESET_BUSY_STATES = (1, 3, 4, 5)

Twist3 = Tuple[float, float, float]
ZERO: Twist3 = (0.0, 0.0, 0.0)


def left_controller_missing(joy) -> bool:
    """
    Return True when the frame carries no value for the base's controls.

    The one place that reads the bridge's "no controller" mark for the left hand: no Joy, too
    few buttons or axes, the left trigger axis NaN (the bridge sends NaN there for a hand without
    a controller and 0.0 on its stick axes), or a stick value that is not finite. If the bridge
    later marks the stick axes the same way as the trigger, only this changes.
    """
    if joy is None or len(joy.axes) <= max(LEFT_TRIGGER_AXIS, LEFT_X_AXIS, LEFT_Y_AXIS):
        return True
    if len(joy.buttons) <= max(LEFT_GRIP_BUTTON, RIGHT_GRIP_BUTTON):
        return True
    return not all(math.isfinite(joy.axes[i])
                   for i in (LEFT_TRIGGER_AXIS, LEFT_X_AXIS, LEFT_Y_AXIS))


def _right_stick(joy) -> Tuple[float, float]:
    """Return the right stick (x, y); (0, 0) when the right controller gives no value."""
    if len(joy.axes) <= max(RIGHT_X_AXIS, RIGHT_Y_AXIS, RIGHT_TRIGGER_AXIS):
        return 0.0, 0.0
    x, y = float(joy.axes[RIGHT_X_AXIS]), float(joy.axes[RIGHT_Y_AXIS])
    if not (math.isfinite(joy.axes[RIGHT_TRIGGER_AXIS]) and math.isfinite(x)
            and math.isfinite(y)):
        return 0.0, 0.0
    return x, y


@dataclass(frozen=True)
class Sample:
    """One headset frame as the base sees it."""

    arrival: float
    session_id: int
    seq: int
    xr_time: float
    missing: bool               # no value for the left controller
    left_grip: bool
    right_grip: bool
    left: Tuple[float, float] = (0.0, 0.0)      # left stick (x, y), raw
    right: Tuple[float, float] = (0.0, 0.0)     # right stick (x, y), raw; (0, 0) without one


def sample_of(status, joy, arrival: float) -> Sample:
    """Return the base's sample of one paired frame (status and its joy, or None)."""
    missing = left_controller_missing(joy)
    left_grip = right_grip = False
    left = right = (0.0, 0.0)
    if joy is not None and not missing:
        left_grip = bool(joy.buttons[LEFT_GRIP_BUTTON])
        right_grip = bool(joy.buttons[RIGHT_GRIP_BUTTON])
        left = (float(joy.axes[LEFT_X_AXIS]), float(joy.axes[LEFT_Y_AXIS]))
        right = _right_stick(joy)
    elif joy is not None and len(joy.buttons) > RIGHT_GRIP_BUTTON:
        right_grip = bool(joy.buttons[RIGHT_GRIP_BUTTON])
    return Sample(arrival, int(status.session_id), int(status.seq),
                  float(status.xr_time_ms) / 1000.0, missing, left_grip, right_grip, left, right)


@dataclass(frozen=True)
class Params:
    """The base's settings; the caps and limits are required (G-19 values, PENDING)."""

    linear_cap: float               # m/s, forward and sideways together
    turn_cap: float                 # rad/s
    linear_accel: float             # m/s^2
    turn_accel: float               # rad/s^2
    tool_speed_cap: Optional[float] = None      # m/s of the tool in the room; None: no limit
    dead_zone: float = 0.15
    snap_deg: float = 10.0
    input_timeout_s: float = INPUT_TIMEOUT_S    # G-3, the gate's own value
    max_dt_s: float = 0.1           # a longer gap between cycles counts as this long

    def check(self) -> None:
        """Raise ValueError for settings that cannot work."""
        for name in ('linear_cap', 'turn_cap', 'linear_accel', 'turn_accel'):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f'base_{name} must be finite and above zero')
        if self.tool_speed_cap is not None and not (math.isfinite(self.tool_speed_cap)
                                                    and self.tool_speed_cap > 0.0):
            raise ValueError('base_tool_speed_cap must be finite and above zero when set')
        if not 0.0 <= self.dead_zone < 0.9:
            raise ValueError('the stick dead zone must be within [0, 0.9)')
        if not 0.0 <= self.snap_deg < 45.0:
            raise ValueError('the axis snap must be within [0, 45) degrees')


def shape(magnitude: float, dead_zone: float) -> float:
    """Return the response 0..1 for a stick deflection: dead zone, then quadratic."""
    m = min(1.0, abs(magnitude))
    if m <= dead_zone:
        return 0.0
    return ((m - dead_zone) / (1.0 - dead_zone)) ** 2


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def translation(stick: Tuple[float, float], p: Params) -> Tuple[float, float]:
    """Return (vx, vy) in the robot's axes for the left stick (x, y raw)."""
    forward, left = -stick[1], -stick[0]
    s = shape(math.hypot(forward, left), p.dead_zone)
    if s == 0.0:
        return 0.0, 0.0
    speed = p.linear_cap * s
    angle = math.atan2(left, forward)
    snap = math.radians(p.snap_deg)
    for axis, out in ((0.0, (speed, 0.0)), (math.pi, (-speed, 0.0)),
                      (math.pi / 2.0, (0.0, speed)), (-math.pi / 2.0, (0.0, -speed))):
        if abs(_wrap(angle - axis)) <= snap:
            return out
    return speed * math.cos(angle), speed * math.sin(angle)


def turn(stick: Tuple[float, float], p: Params) -> float:
    """Return wz for the right stick (x, y raw): left is positive; only while x dominates."""
    x, y = stick
    if not abs(x) > abs(y):
        return 0.0
    s = shape(x, p.dead_zone)
    return 0.0 if s == 0.0 else -math.copysign(p.turn_cap * s, x)


def desired(sample: Sample, p: Params) -> Tuple[Twist3, int, bool, bool]:
    """Return (command, mode, turn ignored, at a cap) the sticks ask for: one mode only."""
    vx, vy = translation(sample.left, p)
    wz = turn(sample.right, p)
    if vx != 0.0 or vy != 0.0:
        at_cap = shape(math.hypot(*sample.left), p.dead_zone) >= 1.0
        return (vx, vy, 0.0), MODE_TRANSLATE, wz != 0.0, at_cap
    if wz != 0.0:
        return (0.0, 0.0, wz), MODE_TURN, False, shape(sample.right[0], p.dead_zone) >= 1.0
    return ZERO, MODE_NONE, False, False


def tool_speed(command: Twist3, tool_xy: Tuple[float, float]) -> float:
    """Return the speed in the room of a point at ``tool_xy`` (base frame) under ``command``."""
    vx, vy, wz = command
    return math.hypot(vx - wz * tool_xy[1], vy + wz * tool_xy[0])


def tool_scale(command: Twist3, tool_xy: Optional[Tuple[float, float]],
               cap: Optional[float]) -> float:
    """Return the factor (<= 1) that keeps the tool within ``cap``; 1 without a cap."""
    if cap is None or tool_xy is None:
        return 1.0
    speed = tool_speed(command, tool_xy)
    return 1.0 if speed <= cap else cap / speed


def mode_of(command: Twist3) -> int:
    """Return the motion mode a command is in."""
    if command[0] != 0.0 or command[1] != 0.0:
        return MODE_TRANSLATE
    return MODE_TURN if command[2] != 0.0 else MODE_NONE


def ramp(current: Twist3, target: Twist3, dt: float, p: Params) -> Tuple[Twist3, bool]:
    """
    Return the command one step from ``current`` toward ``target`` and whether it was limited.

    A change of mode first ramps the current command to zero. Forward and sideways change
    together, as one vector, at the linear limit; turning at its own limit.
    """
    if mode_of(current) not in (MODE_NONE, mode_of(target)) and target != ZERO:
        target = ZERO                       # through zero: one mode per command
    limited = False
    dx, dy = target[0] - current[0], target[1] - current[1]
    step = p.linear_accel * dt
    d = math.hypot(dx, dy)
    if d > step:
        k = step / d
        vx, vy = current[0] + dx * k, current[1] + dy * k
        limited = True
    else:
        vx, vy = target[0], target[1]
    dw = target[2] - current[2]
    step = p.turn_accel * dt
    if abs(dw) > step:
        wz = current[2] + math.copysign(step, dw)
        limited = True
    else:
        wz = target[2]
    return (vx, vy, wz), limited


@dataclass(frozen=True)
class TeleopView:
    """What the base needs of the arm's fresh state."""

    clutch_engaged: bool
    stopping: bool                  # the arm's G-18 stop ramp runs


@dataclass(frozen=True)
class World:
    """What the node knows at a cycle besides the input. Freshness is judged by the node."""

    now: float
    enabled: bool
    simulator: bool
    teleop: Optional[TeleopView]                # None: no fresh teleop state
    reset_state: Optional[int] = None           # a fresh ResetState.state (None: none fresh)
    reset_running: bool = False                 # a reset node is (or was) heard
    rest: int = UNKNOWN                         # base_rest.AT_REST, MOVING or UNKNOWN
    other_publishers: Tuple[str, ...] = ()      # nodes other than this one on a base command
    tool_xy: Optional[Tuple[float, float]] = None   # tool position in the base frame (m)


@dataclass(frozen=True)
class Output:
    """Result of one cycle."""

    publish: Optional[Twist3]       # the command to publish now; None: publish nothing
    state: int
    mode: int
    reason: int
    driving: bool                   # a command is published this cycle
    turn_ignored: bool
    active_limit: int
    tool_scale: float               # NaN without a tool speed cap


def world_reason(world: World, p: Params) -> int:
    """Return what in the world forbids driving now (a REASON_*), or NONE; input aside."""
    if not world.enabled:
        return R_OFF
    if not world.simulator:
        return NOT_SIMULATION
    if world.teleop is None:
        return TELEOP_UNKNOWN
    if world.teleop.clutch_engaged or world.teleop.stopping:
        return CLUTCH_ENGAGED
    if world.reset_state is None:
        if world.reset_running:
            return RESET_UNKNOWN
    elif world.reset_state in RESET_BUSY_STATES:
        return RESET_ACTIVE
    if world.rest == UNKNOWN:
        return ODOM_MISSING
    if world.other_publishers:
        return OTHER_PUBLISHER
    if p.tool_speed_cap is not None and world.tool_xy is None:
        return TOOL_POSE_UNKNOWN
    return NONE


class Base:
    """The base's state machine; see the module text."""

    def __init__(self, params: Params) -> None:
        """Start idle, driving with ``params``."""
        params.check()
        self.params = params
        self.state, self.reason, self.mode = IDLE, NONE, MODE_NONE
        self.command: Twist3 = ZERO
        self.driving = False
        self.need_release = True            # a start needs the grip seen released first
        self.last: Optional[Sample] = None
        self.fault = NONE                   # a session or sequence fault since the last cycle
        self.last_tick: Optional[float] = None
        self.await_rest = False             # a drive ended; the at-rest event is due
        self.other_text = ''
        self.event_count, self.last_event, self.last_event_reason = 0, EVENT_NONE, NONE
        self.last_event_wall = math.nan
        self.log: List[str] = []

    # ---- input ----

    def on_sample(self, sample: Sample) -> None:
        """Take one headset frame (every frame, in order)."""
        prev = self.last
        if prev is not None:
            fault = frame_fault(prev, sample)
            if fault != Hold.NONE and self.fault == NONE:
                self.fault = (SESSION_CHANGED if fault == Hold.SESSION_CHANGED
                              else SEQ_BACKWARDS)
        self.last = sample

    def _input(self, now: float) -> Tuple[int, Optional[Sample]]:
        """Return (what breaks the hold now or NONE, the sample)."""
        if self.last is None or not now - self.last.arrival <= self.params.input_timeout_s:
            return INPUT_STALE, None
        if self.fault != NONE:
            return self.fault, None
        if self.last.missing:
            return CONTROLLER_MISSING, None
        return NONE, self.last

    # ---- the cycle ----

    def tick(self, now: float, world: World, wall: float = math.nan) -> Output:
        """Run one cycle at ``now`` (monotonic s); return what to publish."""
        dt = 0.0 if self.last_tick is None else min(max(now - self.last_tick, 0.0),
                                                    self.params.max_dt_s)
        self.last_tick = now
        problem, sample = self._input(now)
        self.fault = NONE                   # consumed: it breaks this cycle's hold only
        grip = sample is not None and sample.left_grip
        if sample is not None and not sample.left_grip:
            self.need_release = False
        if self.driving:
            return self._driving(now, world, problem, sample, dt, wall)
        if not (world.enabled and world.simulator):
            self.state, self.reason = OFF, (R_OFF if not world.enabled else NOT_SIMULATION)
            return self._idle_out()
        if self.state == OFF:
            self.state, self.reason = IDLE, NONE
        self._watch_rest(world, wall)
        if grip and not self.need_release:
            return self._press(now, world, sample, wall)
        return self._idle_out()

    def _press(self, now: float, world: World, sample: Sample, wall: float) -> Output:
        """Start a drive on a press of the grip, or refuse it."""
        self.need_release = True            # whatever happens, the next start needs a press
        reason = RIGHT_GRIP_HELD if sample.right_grip else world_reason(world, self.params)
        if reason != NONE:
            self.state, self.reason = REFUSED, reason
            self._note_other(world, reason)
            self._event(EVENT_REFUSED, reason, wall)
            return self._idle_out()
        self.driving, self.state, self.reason = True, DRIVING, NONE
        self.command = ZERO
        self.await_rest = False
        self._event(EVENT_START, NONE, wall)
        return self._step(world, sample, 0.0)

    def _driving(self, now: float, world: World, problem: int, sample: Optional[Sample],
                 dt: float, wall: float) -> Output:
        reason = problem
        if reason == NONE and not sample.left_grip:
            reason = RELEASED
        if reason == NONE:
            reason = world_reason(world, self.params)
        if reason != NONE:
            return self.stop(reason, wall, world)
        return self._step(world, sample, dt)

    def _step(self, world: World, sample: Sample, dt: float) -> Output:
        target, mode, ignored, at_cap = desired(sample, self.params)
        scale = tool_scale(target, world.tool_xy, self.params.tool_speed_cap)
        if scale < 1.0:
            target = (target[0] * scale, target[1] * scale, target[2] * scale)
        self.command, limited = ramp(self.command, target, dt, self.params)
        self.mode = mode_of(self.command)
        limit = LIMIT_NONE
        if scale < 1.0:
            limit = LIMIT_TOOL_SPEED
        elif limited:
            limit = LIMIT_ACCELERATION
        elif at_cap and self.command == target:
            limit = LIMIT_CAP
        return Output(self.command, DRIVING, self.mode, NONE, True, ignored, limit,
                      scale if self.params.tool_speed_cap is not None else math.nan)

    def stop(self, reason: int, wall: float = math.nan, world: Optional[World] = None) -> Output:
        """End a drive now: one zero command, then nothing (also the node's shutdown)."""
        if not self.driving:
            return self._idle_out()
        self.driving = False
        self.need_release = reason != RELEASED      # released already: the next hold is a press
        self.command, self.mode = ZERO, MODE_NONE
        self.state = STOPPING if reason == RELEASED else STOPPED
        self.reason = reason
        if world is not None:
            self._note_other(world, reason)
        self._event(EVENT_RELEASED if reason == RELEASED else EVENT_STOPPED, reason, wall)
        self.await_rest = True
        return Output(ZERO, self.state, MODE_NONE, reason, False, False, LIMIT_NONE,
                      self._scale_unset())

    def _watch_rest(self, world: World, wall: float) -> None:
        """After a drive: the at-rest event once odometry says so; STOPPING becomes IDLE."""
        if self.await_rest and world.rest == AT_REST:
            self.await_rest = False
            self._event(EVENT_AT_REST, self.reason, wall)
            if self.state == STOPPING:
                self.state = IDLE

    def _idle_out(self) -> Output:
        return Output(None, self.state, MODE_NONE, self.reason, False, False, LIMIT_NONE,
                      self._scale_unset())

    def _scale_unset(self) -> float:
        return 1.0 if self.params.tool_speed_cap is not None else math.nan

    def _note_other(self, world: World, reason: int) -> None:
        self.other_text = (', '.join(world.other_publishers)
                           if reason == OTHER_PUBLISHER else '')

    def _event(self, kind: int, reason: int, wall: float) -> None:
        self.event_count += 1
        self.last_event, self.last_event_reason, self.last_event_wall = kind, reason, wall
        names = ('NONE', 'START', 'REFUSED', 'STOPPED', 'RELEASED', 'AT_REST')
        self.log.append(f'{names[kind]}' + (f': {self.reason_text(reason)}' if reason else ''))

    def reason_text(self, reason: Optional[int] = None) -> str:
        """Return plain words for ``reason`` (default: the current one)."""
        reason = self.reason if reason is None else reason
        text = REASON_TEXT.get(reason, '')
        if reason == OTHER_PUBLISHER and self.other_text:
            text += f': {self.other_text}'
        return text

    def take_log(self) -> List[str]:
        """Return and clear the lines for the node to log."""
        lines, self.log = self.log, []
        return lines
