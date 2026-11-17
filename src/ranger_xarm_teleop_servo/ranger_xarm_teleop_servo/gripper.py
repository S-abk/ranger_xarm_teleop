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
Gripper teleoperation: the right trigger opens and closes the gripper. No ROS imports.

Mapping: two states with hysteresis. The request becomes CLOSE when the trigger rises
through the close threshold and OPEN when it falls through the open threshold; the
command then moves to the closed or the open position at the speed limit. The command,
not the trigger, is the recorded action.

When the gripper may be commanded ("active"): gripper teleop is enabled, teleop is
armed, teleop owns the gripper (the mux swaps it with the arm), the input is valid
(the gate's clutch-independent condition holds and no frame since the last cycle tripped
anything) and the newest frame carries a trigger value in [0, 1]. The clutch is not
part of it: the trigger works while the clutch is released.

Freeze: any cycle that is not active sends nothing, so the gripper holds the last
command it was given. On becoming active again (arming, or the end of a freeze):

* the command starts at the held reference: teleop's own last command when it has owned
  the gripper since (after a freeze), else the opening the controller holds (after an
  owner switch: what the previous owner last commanded), else, after
  ``reference_timeout`` without one, the measured opening;
* the trigger has no effect until it crosses a threshold (``AWAIT_TRIGGER``). A trigger
  held through the freeze does nothing until it is released (or pressed) again.

Keep-out: a step of the command that would bring the fingers nearer the robot's fixed
structure than the margin, and nearer than they are at the current command, is not
taken (``KEEP_OUT``). The request stays, so the motion resumes by itself once the
clearance allows (for example after the arm moved away).
"""

from dataclasses import dataclass, replace
from enum import IntEnum
import math
from typing import Any, Callable, Mapping, Optional, Tuple
from xml.etree import ElementTree

# Safety values: required, no default, when gripper teleop is enabled.
REQUIRED_WHEN_ENABLED = ('gripper_open_position', 'gripper_closed_position',
                         'gripper_max_speed')

DEFAULTS = {
    'gripper_enabled': False,               # G-1; off is safe: nothing is ever published
    'gripper_joint': 'xarm_drive_joint',    # the driven joint; followers come from <mimic>
    'gripper_close_threshold': 0.6,
    'gripper_open_threshold': 0.25,
    # After an owner switch, how long to wait for the controller's held reference before
    # the measured opening is used instead.
    'gripper_reference_timeout_s': 0.5,
}

# A step smaller than this is no motion (joint units).
STEP_EPS = 1e-9
# Clearance comparisons: a step that keeps the clearance within this is not "nearer".
CLEARANCE_EPS = 1e-6


class Hold(IntEnum):
    """Why the gripper is not commanded, or not moving; values match ``GRIPPER_HOLD_*``."""

    NONE = 0
    DISABLED = 1
    NOT_ARMED = 2
    NOT_OWNER = 3
    INPUT = 4               # the input is not valid (gate condition, trip, no trigger)
    NO_REFERENCE = 5        # waiting for the opening the controller holds
    AWAIT_TRIGGER = 6       # active, holding the reference until the trigger crosses
    KEEP_OUT = 7            # a step toward the robot's fixed structure was not taken


class Request(IntEnum):
    """The operator's request; values match ``GripperCommand.REQUEST_*``."""

    NONE = 0
    OPEN = 1
    CLOSE = 2


class Source(IntEnum):
    """Where the first command after becoming active came from; ``SOURCE_*``."""

    NONE = 0
    HELD = 1
    OWN = 2
    MEASURED = 3


class MissingGripperValue(ValueError):
    """A required value of an enabled gripper was not supplied."""


@dataclass(frozen=True)
class GripperParams:
    """Validated gripper parameters (joint units: rad for a revolute joint)."""

    enabled: bool
    joint: str = ''
    open_position: float = math.nan
    closed_position: float = math.nan
    max_speed: float = math.nan
    close_threshold: float = DEFAULTS['gripper_close_threshold']
    open_threshold: float = DEFAULTS['gripper_open_threshold']
    reference_timeout: float = DEFAULTS['gripper_reference_timeout_s']
    period: float = 0.02

    @property
    def step(self) -> float:
        """Largest change of the command per cycle."""
        return self.max_speed * self.period


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}: expected a number, got {value!r}')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name}: expected a finite number, got {value!r}')
    return value


def build_gripper_params(values: Mapping[str, Any], period: float) -> GripperParams:
    """
    Build the gripper parameters from a name-to-value mapping.

    Disabled (the default), nothing else is read. Enabled, a missing safety value raises
    ``MissingGripperValue`` naming every one; nothing is substituted.
    """
    def opt(name: str) -> Any:
        value = values.get(name)
        return DEFAULTS[name] if value is None else value

    enabled = bool(opt('gripper_enabled'))
    if not enabled:
        return GripperParams(enabled=False, period=period)
    missing = [name for name in REQUIRED_WHEN_ENABLED if values.get(name) is None]
    if missing:
        raise MissingGripperValue(
            'gripper teleop is enabled but these required values have no default and are '
            'not set: ' + ', '.join(missing))
    joint = str(opt('gripper_joint'))
    if not joint:
        raise ValueError("gripper_joint: must name the gripper's driven joint")
    open_position = _finite('gripper_open_position', values['gripper_open_position'])
    closed_position = _finite('gripper_closed_position', values['gripper_closed_position'])
    if abs(open_position - closed_position) < 1e-6:
        raise ValueError('gripper_open_position and gripper_closed_position are equal')
    speed = _finite('gripper_max_speed', values['gripper_max_speed'])
    if speed <= 0.0:
        raise ValueError(f'gripper_max_speed: expected a positive number, got {speed!r}')
    close = _finite('gripper_close_threshold', opt('gripper_close_threshold'))
    open_ = _finite('gripper_open_threshold', opt('gripper_open_threshold'))
    if not 0.0 < open_ < close < 1.0:
        raise ValueError(f'gripper thresholds: expected 0 < open ({open_!r}) < close '
                         f'({close!r}) < 1')
    timeout = _finite('gripper_reference_timeout_s', opt('gripper_reference_timeout_s'))
    if timeout < 0.0:
        raise ValueError('gripper_reference_timeout_s: expected a number >= 0')
    return GripperParams(enabled=True, joint=joint, open_position=open_position,
                         closed_position=closed_position, max_speed=speed,
                         close_threshold=close, open_threshold=open_,
                         reference_timeout=timeout, period=period)


def joint_limits(urdf: str, joint: str) -> Tuple[float, float]:
    """Return the position limits of ``joint`` in a URDF; ``ValueError`` when it has none."""
    try:
        root = ElementTree.fromstring(urdf)
    except ElementTree.ParseError as err:
        raise ValueError(f'the robot description is not valid XML: {err}') from err
    # Only the robot's own joints: a <ros2_control> block names joints too, without a type.
    for element in root.findall('joint'):
        if element.get('name') != joint:
            continue
        if element.get('type') not in ('revolute', 'prismatic'):
            raise ValueError(f'{joint} is {element.get("type")!r}, not a revolute or '
                             'prismatic joint')
        limit = element.find('limit')
        if limit is None or limit.get('lower') is None or limit.get('upper') is None:
            raise ValueError(f'{joint} has no position limits')
        return float(limit.get('lower')), float(limit.get('upper'))
    raise ValueError(f'the robot description has no joint {joint}')


def check_joint_range(params: GripperParams, lower: float, upper: float) -> None:
    """Raise ``ValueError`` when the open or closed position is outside the joint's limits."""
    for name, value in (('gripper_open_position', params.open_position),
                        ('gripper_closed_position', params.closed_position)):
        if not lower - 1e-9 <= value <= upper + 1e-9:
            raise ValueError(f'{name} {value!r} is outside the limits of {params.joint} '
                             f'[{lower!r}, {upper!r}]')


def valid_trigger(value: Optional[float]) -> Optional[float]:
    """Return the trigger value when it is a finite number in [0, 1], else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


@dataclass(frozen=True)
class GripperState:
    """What the gripper logic remembers between cycles."""

    active: bool = False
    # Teleop's own last command, kept only while teleop keeps owning the gripper.
    command: Optional[float] = None
    source: Source = Source.NONE
    request: Request = Request.NONE         # NONE: waiting for a threshold crossing
    last_trigger: Optional[float] = None    # trigger of the previous active cycle
    waiting_since: Optional[float] = None   # since when a held reference is awaited


@dataclass(frozen=True)
class GripperInputs:
    """One cycle's inputs."""

    now: float                              # monotonic s
    armed: bool
    owner: bool                             # teleop owns the gripper
    input_ok: bool                          # gate condition holds, no trip since the last cycle
    trigger: Optional[float]                # newest frame's trigger, None when missing
    held: Optional[float] = None            # controller's held reference after the switch
    measured: Optional[float] = None        # measured opening, None when unknown
    # Clearance (m) of the gripper at a commanded position, arm at its measured joints;
    # None when there is no keep-out. NaN (cannot be evaluated) holds the command.
    clearance: Optional[Callable[[float], float]] = None
    margin: float = 0.0


@dataclass(frozen=True)
class GripperOutput:
    """What one cycle publishes."""

    publish: bool               # send ``command`` this cycle
    command: float              # the command in force (NaN when teleop holds none)
    request: Request
    source: Source
    hold: Hold
    events: Tuple[str, ...] = ()


def _toward(value: float, goal: float, step: float) -> float:
    if abs(goal - value) <= step:
        return goal
    return value + math.copysign(step, goal - value)


def _crossing(previous: float, now: float, params: GripperParams) -> Request:
    if previous < params.close_threshold <= now:
        return Request.CLOSE
    if previous > params.open_threshold >= now:
        return Request.OPEN
    return Request.NONE


def _in_force(state: GripperState) -> float:
    return state.command if state.command is not None else math.nan


def step(state: GripperState, inputs: GripperInputs,
         params: GripperParams) -> Tuple[GripperState, GripperOutput]:
    """Run one cycle; return the new state and what to publish."""
    if not params.enabled:
        return GripperState(), GripperOutput(False, math.nan, Request.NONE, Source.NONE,
                                             Hold.DISABLED)
    events = []
    if not inputs.owner:
        # Someone else may command the gripper now: teleop's last command is no longer
        # what the gripper holds.
        if state.command is not None:
            events.append('gripper: teleop no longer owns the gripper')
        state = GripperState()
    trigger = valid_trigger(inputs.trigger)
    hold = Hold.NONE
    if not inputs.armed:
        hold = Hold.NOT_ARMED
    elif not inputs.owner:
        hold = Hold.NOT_OWNER
    elif not inputs.input_ok or trigger is None:
        hold = Hold.INPUT
    if hold != Hold.NONE:
        if state.active:
            events.append(f'gripper frozen: {hold.name}')
        # Freeze: nothing is sent; teleop's command (if it owns) stays in force.
        state = replace(state, active=False, request=Request.NONE, last_trigger=None,
                        waiting_since=None)
        return state, GripperOutput(False, _in_force(state), Request.NONE, state.source, hold,
                                    tuple(events))

    if not state.active:
        if state.command is not None:
            reference, source = state.command, Source.OWN
        elif inputs.held is not None and math.isfinite(inputs.held):
            reference, source = float(inputs.held), Source.HELD
        else:
            since = state.waiting_since if state.waiting_since is not None else inputs.now
            measured = inputs.measured
            if (inputs.now - since < params.reference_timeout or measured is None
                    or not math.isfinite(measured)):
                state = replace(state, waiting_since=since)
                return state, GripperOutput(False, math.nan, Request.NONE, Source.NONE,
                                            Hold.NO_REFERENCE, tuple(events))
            reference, source = float(measured), Source.MEASURED
            events.append(f'gripper: no held reference within '
                          f'{params.reference_timeout:.2f} s; the measured opening is used')
        events.append(f'gripper active: command {reference:.4f} from {source.name}; '
                      'the trigger acts after it crosses a threshold')
        state = GripperState(active=True, command=reference, source=source,
                             request=Request.NONE, last_trigger=trigger)
    else:
        crossed = _crossing(state.last_trigger, trigger, params)
        request = state.request
        if crossed != Request.NONE:
            if crossed != request:
                events.append(f'gripper: {crossed.name.lower()}')
            request = crossed
        state = replace(state, request=request, last_trigger=trigger)

    command = state.command
    if state.request == Request.NONE:
        hold = Hold.AWAIT_TRIGGER
    else:
        goal = (params.closed_position if state.request == Request.CLOSE
                else params.open_position)
        proposed = _toward(command, goal, params.step)
        if abs(proposed - command) > STEP_EPS and inputs.clearance is not None:
            ahead = inputs.clearance(proposed)
            if not math.isfinite(ahead):
                hold = Hold.KEEP_OUT        # cannot be evaluated: do not move
            elif ahead < inputs.margin:
                here = inputs.clearance(command)
                if not (math.isfinite(here) and ahead >= here - CLEARANCE_EPS):
                    hold = Hold.KEEP_OUT
        if hold == Hold.NONE:
            command = proposed
    state = replace(state, command=command)
    return state, GripperOutput(True, command, state.request, state.source, hold,
                                tuple(events))
