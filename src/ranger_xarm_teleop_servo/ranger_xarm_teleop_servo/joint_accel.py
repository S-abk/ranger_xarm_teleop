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
The joint acceleration limit of the teleop command (G-18). No ROS imports.

The joint command (rad/s, one per target cycle) never changes by more than
``acceleration * period`` on any joint from one cycle to the next:

* while a target is followed, the change the solver asks for is scaled down, all joints by
  the same factor (``limit_step``): the limited change is parallel to the requested one, so
  when the request keeps its direction the tool keeps it too, and no joint is ever clipped on
  its own;
* when the target ends (a release, a gate fault, a disarm), the last command is ramped to zero
  at the limit, all joints by the same factor (``ramp_step``), so the tool keeps its direction
  while it stops.

The only state is the last command sent. Nothing the limiter could not deliver is kept
anywhere: the next cycle starts from what was sent. A ramp from the joint speed cap takes
``speed_cap / acceleration``, and a configuration where that exceeds ``STOP_TIME_LIMIT_S`` is
refused (``stop_time_refusal``).
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

# G-18: the longest a stop ramp may take. Fixed here, not a parameter.
STOP_TIME_LIMIT_S = 0.2

# The two limits-file values (G-18). Both are required; neither has a default.
ACCELERATION_NAME = 'joint_acceleration_limit_rad_s2'
SPEED_CAP_NAME = 'joint_speed_cap_rad_s'
NAMES = (ACCELERATION_NAME, SPEED_CAP_NAME)

# The teleop node's log line when it sends one zero at once instead of a ramp (teleop lost the
# arm, or MoveIt Servo halted); the check tool counts these lines on /rosout.
IMMEDIATE_STOP_LOG = 'joint command stopped at once'

# Relative slack for floating-point rounding when deciding whether a step was limited.
_ROUNDING = 1e-9


@dataclass(frozen=True)
class Step:
    """One cycle's limited joint command."""

    velocity: np.ndarray        # rad/s, the command to send
    scale: float                # share of the requested change that was sent (1: all of it)
    limited: bool               # the limit acted this cycle


def check_values(acceleration, speed_cap) -> None:
    """Raise ``ValueError`` unless both values are finite positive numbers."""
    for name, value in ((ACCELERATION_NAME, acceleration), (SPEED_CAP_NAME, speed_cap)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name}: expected a number, got {value!r}')
        if not (math.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f'{name}: expected a finite positive number, got {value!r}')


def stop_time_refusal(acceleration: float, speed_cap: float) -> str:
    """
    Return one line naming the value to change when a stop could exceed the stop time, else ''.

    A stop ramp from the joint speed cap takes ``speed_cap / acceleration``. The joint speed cap
    is chosen first (well under the arm's rated joint speed), so the line names the
    acceleration limit as the value to change, and says what lowering the cap would need.
    """
    seconds = speed_cap / acceleration
    if seconds <= STOP_TIME_LIMIT_S * (1.0 + _ROUNDING):
        return ''
    return (f'{SPEED_CAP_NAME} / {ACCELERATION_NAME} = {speed_cap:g} / {acceleration:g} = '
            f'{seconds:.3f} s, above the {STOP_TIME_LIMIT_S:g} s a stop may take (G-18): raise '
            f'{ACCELERATION_NAME} to at least {speed_cap / STOP_TIME_LIMIT_S:.4g}, or lower '
            f'{SPEED_CAP_NAME} to at most {acceleration * STOP_TIME_LIMIT_S:.4g}')


def limit_step(previous: Sequence[float], desired: Sequence[float], acceleration: float,
               period: float) -> Step:
    """
    Return the command to send toward ``desired`` from the last command sent, ``previous``.

    The change ``desired - previous`` is sent whole when no joint's part of it exceeds
    ``acceleration * period``; otherwise all of it is scaled by one factor so that the largest
    joint change is exactly that much.
    """
    previous = np.asarray(previous, dtype=float)
    desired = np.asarray(desired, dtype=float)
    change = desired - previous
    largest = float(np.max(np.abs(change))) if change.size else 0.0
    allowed = acceleration * period
    if largest <= allowed * (1.0 + _ROUNDING):
        return Step(desired.copy(), 1.0, False)
    scale = allowed / largest
    return Step(previous + change * scale, scale, True)


def ramp_step(previous: Sequence[float], acceleration: float, period: float) -> Step:
    """
    Return the next command of a stop ramp from ``previous``: toward zero at the limit.

    All joints are scaled by one factor, so the command keeps its direction until it is zero.
    ``limited`` is True while the ramp is still moving (this step is not yet zero).
    """
    previous = np.asarray(previous, dtype=float)
    largest = float(np.max(np.abs(previous))) if previous.size else 0.0
    allowed = acceleration * period
    if largest <= allowed * (1.0 + _ROUNDING):
        return Step(np.zeros_like(previous), 0.0, False)
    keep = 1.0 - allowed / largest
    return Step(previous * keep, keep, True)


def flush_cycles(window: float, period: float) -> int:
    """
    Return how many zero commands follow a stop ramp's zero before MoveIt Servo is paused.

    Servo keeps ``window`` seconds of points (its max_expected_latency) and publishes every
    one but the newest: the window in command periods, rounded up, brings it to rest; one
    more covers a cycle Servo skips (its loop runs without phase to this node's).
    """
    if not (window > 0.0 and period > 0.0):
        raise ValueError(f'window {window} s and period {period} s must be positive')
    return math.ceil(window / period - _ROUNDING) + 1


def stop_time(velocity: Sequence[float], acceleration: float) -> float:
    """Return how long a stop ramp from ``velocity`` takes (s)."""
    velocity = np.asarray(velocity, dtype=float)
    return (float(np.max(np.abs(velocity))) / acceleration) if velocity.size else 0.0


def stop_travel_bound(tool_speed: float, joint_speed: float, acceleration: float) -> float:
    """
    Return the bound on the extra tool travel of a stop ramp (m): ``v * q / (2 a)``.

    ``tool_speed`` (m/s) and ``joint_speed`` (rad/s, the fastest joint) at the start of the
    ramp. The ramp lasts ``q / a`` and the tool speed falls linearly to zero over it; the
    tool's path is taken as straight, with the Jacobian constant over the ramp. Points already
    on their way to the controller come on top of this.
    """
    return tool_speed * joint_speed / (2.0 * acceleration)
