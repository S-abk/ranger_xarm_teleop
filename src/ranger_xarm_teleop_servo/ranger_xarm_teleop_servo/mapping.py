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
Clutch-relative motion mapping in a fixed frame (interface contract, "Motion mapping").

With the controller pose ``(p, q)`` in ``quest_world``, the calibration heading ``h``
(rotation about z taking ``quest_world`` directions into the arm base frame) and the
translation scale ``s`` (fine or coarse, chosen by the operator)::

    dp = s Rz(h) (p_now - p_clutch)
    dq = Rz(h) (q_now * inverse(q_clutch)) Rz(h)^-1
    target = (p_tool0 + dp, dq * q_tool0)

where ``tool0`` is the tool pose captured at clutch engagement. Rotation is never
scaled. The motion is taken in the fixed room frame, never as
``inverse(T_clutch) * T_now``, so how the controller is held at engagement does not
change the mapping. No ROS imports.
"""

from typing import Sequence

from .rotations import (compose, Pose, q_conj, q_from_yaw, q_mul, q_normalize, q_rotate,
                        v_add, v_scale, v_sub)


def apply_pivot(controller: Pose, pivot_offset: Sequence[float]) -> Pose:
    """
    Move the tracked point from the grip origin to ``pivot_offset`` (controller frame).

    With the offset at the operator's wrist pivot, a pure wrist twist no longer
    translates the tracked point. The orientation is unchanged. Zero offset is a no-op.
    """
    return Pose(v_add(controller.p, q_rotate(controller.q, pivot_offset)), controller.q)


def clutch_relative(now: Pose, clutch: Pose, heading: float, tool0: Pose,
                    scale: float = 1.0) -> Pose:
    """Return the tool target for controller pose ``now``, engaged at ``clutch``."""
    cal = q_from_yaw(heading)
    dp = v_scale(q_rotate(cal, v_sub(now.p, clutch.p)), scale)
    dq_room = q_mul(now.q, q_conj(clutch.q))
    dq = q_mul(q_mul(cal, dq_room), q_conj(cal))
    return Pose(v_add(tool0.p, dp), q_normalize(q_mul(dq, tool0.q)))


def rescale(anchor: Pose, now: Pose, clutch: Pose, heading: float, old: float,
            new: float) -> Pose:
    """
    Return the anchor that keeps the target where it is when the scale changes.

    For controller pose ``now``, ``clutch_relative(now, clutch, heading, result, new)``
    equals ``clutch_relative(now, clutch, heading, anchor, old)``: the anchor moves by
    ``(old - new) Rz(h) (p_now - p_clutch)``. Rotation is not scaled, so the anchor's
    orientation is kept; further hand motion is taken at the new scale from here on.
    """
    moved = q_rotate(q_from_yaw(heading), v_sub(now.p, clutch.p))
    return Pose(v_add(anchor.p, v_scale(moved, old - new)), anchor.q)


def tool_to_tip(tool_target: Pose, tool_to_tip_transform: Pose) -> Pose:
    """
    Return the pose of the servo group's tip link for a tool-frame target.

    ``tool_to_tip_transform`` is the fixed pose of the tip link in the tool frame, so a
    rotation of the target happens about the tool point, not about the tip.
    """
    return compose(tool_target, tool_to_tip_transform)
