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
WebXR to ROS axis conversion (interface contract, "Frames").

This is the one place where axes are swapped (INV-1). WebXR ``local-floor`` is
x right, y up, z toward the user; ROS is x forward, y left, z up. The basis change

    x_ros = -z_xr,  y_ros = -x_xr,  z_ros = y_xr

is the proper rotation ``C = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]`` (det +1). It is
applied to the position and, as the similarity transform ``C R C^T``, to the
rotation. For a unit quaternion that is the same basis change applied to the
vector part: ``q_ros = (-qz, -qx, qy, qw)``. No ROS imports here (INV-6).
"""

from typing import Sequence, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def xr_pose_to_ros(position: Sequence[float],
                   orientation_xyzw: Sequence[float]) -> Tuple[Vec3, Quat]:
    """
    Convert one WebXR pose (position, xyzw quaternion) to ROS axes.

    Returns ``((x, y, z), (qx, qy, qz, qw))`` in ROS axes, as plain floats.
    Raises ``ValueError`` for inputs of the wrong length; callers validate
    values (finite, unit norm) before calling.
    """
    if len(position) != 3:
        raise ValueError(f'position must have 3 components, got {len(position)}')
    if len(orientation_xyzw) != 4:
        raise ValueError(f'orientation must have 4 components, got {len(orientation_xyzw)}')
    x, y, z = (float(v) for v in position)
    qx, qy, qz, qw = (float(v) for v in orientation_xyzw)
    return (-z, -x, y), (-qz, -qx, qy, qw)
