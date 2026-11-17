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
What the gripper relay sends and reports, for each backend. No ROS imports.

The teleop node publishes one gripper command per cycle on ``/teleop/gripper/command``
(the recorded action) and never branches on the backend. A relay carries it on:

* ``trajectory`` (mock execution, Gazebo, Isaac Sim): each command becomes a one-point
  trajectory, reached one target period later, on the teleop gripper controller's topic.
  The relay also reports the opening that controller holds (the reference in its state),
  which is the held reference the teleop node starts from after an owner switch: the
  controller activates holding the last command of the controller it replaced.
* ``action`` (hardware, not built): would send ``control_msgs/GripperCommand`` goals and
  report its last goal as the held reference. The relay refuses this mode.
"""

import math
from typing import Optional, Sequence, Tuple

MODES = ('trajectory',)
NOT_BUILT = {'action': 'the hardware adapter (GripperCommand action) is not built yet'}


def check_mode(mode: str) -> None:
    """Raise ``ValueError`` for a mode this relay does not provide."""
    if mode in MODES:
        return
    raise ValueError(NOT_BUILT.get(mode, f'unknown mode {mode!r}; expected one of {MODES}'))


def duration(seconds: float) -> Tuple[int, int]:
    """Return ``(sec, nanosec)`` of a non-negative duration."""
    nanoseconds = int(round(seconds * 1e9))
    return nanoseconds // 1_000_000_000, nanoseconds % 1_000_000_000


def command_point(joint: str, name: str, position: float,
                  period: float) -> Optional[Tuple[float, Tuple[int, int]]]:
    """
    Return ``(position, time_from_start)`` of the trajectory point for one command.

    ``None`` (nothing is sent) when the command names another joint or is not finite.
    """
    if name != joint or not math.isfinite(position):
        return None
    return float(position), duration(period)


def held_reference(joint: str, names: Sequence[str],
                   reference: Sequence[float]) -> Optional[float]:
    """Return the controller's reference position of ``joint``, or None."""
    names = list(names)
    if joint not in names:
        return None
    index = names.index(joint)
    if index >= len(reference) or not math.isfinite(reference[index]):
        return None
    return float(reference[index])
