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
One-button heading calibration between ``quest_world`` and the arm base (contract, G-13).

Both frames have z up and the mapping is clutch-relative, so the calibration is one
angle: the rotation about z that takes ``quest_world`` directions into the arm base
frame. The operator points the controller along "arm forward" and presses the
calibration button; the horizontal projection of the pointing axis becomes base +x.
A calibration is valid only for the session id and reset count it was made under.
No ROS imports.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence

from .rotations import q_rotate


@dataclass(frozen=True)
class Calibration:
    """A heading and the ``quest_world`` frame (session id, reset count) it belongs to."""

    heading: float
    session_id: int
    reset_count: int

    def valid_for(self, session_id: int, reset_count: int) -> bool:
        """Return True when the calibration was made under exactly this frame."""
        return self.session_id == session_id and self.reset_count == reset_count


def heading_from_pointing(orientation: Sequence[float], pointing_axis: Sequence[float],
                          min_horizontal: float) -> Optional[float]:
    """
    Return the heading for a controller ``orientation`` (xyzw, ``quest_world``).

    ``pointing_axis`` is the controller-frame axis the operator points with. Returns
    ``None`` when its horizontal projection is shorter than ``min_horizontal`` times its
    length (pointing too close to vertical to define a heading).
    """
    d = q_rotate(orientation, pointing_axis)
    length = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    horizontal = math.hypot(d[0], d[1])
    if length < 1e-9 or horizontal < min_horizontal * length:
        return None
    return -math.atan2(d[1], d[0])


def calibrate(orientation: Sequence[float], pointing_axis: Sequence[float],
              min_horizontal: float, session_id: int, reset_count: int) -> Optional[Calibration]:
    """Return a calibration bound to ``(session_id, reset_count)``, or ``None`` if refused."""
    heading = heading_from_pointing(orientation, pointing_axis, min_horizontal)
    if heading is None:
        return None
    return Calibration(heading, int(session_id), int(reset_count))
