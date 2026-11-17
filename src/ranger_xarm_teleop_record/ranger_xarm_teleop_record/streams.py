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
Stream tables: one recorded episode as plain numpy arrays (no ROS).

The bag reader (``bag_io``, ROS) fills these; the checker, the consistency check and the
export (pure) only read them. Unit tests build them synthetically.

Conventions
-----------
- Times are float64 seconds.
- ``stamp`` is the message's own ``header.stamp`` as recorded, never changed. A message
  type without a header (``geometry_msgs/Twist`` on ``/cmd_vel``) has ``stamp`` equal to
  ``recv`` and ``has_header`` False.
- ``recv`` is the recorder's receive time, on the recording clock: ROS time of the
  recording host (simulation time when the profile says ``use_sim_time``).
- Rows are in the order the recorder wrote them (receive order). Nothing is sorted,
  deduplicated, resampled or filtered here.
- ``fields`` holds per-message values; the first dimension of every array is the row
  count. Poses are ``(n, 7)``: x, y, z, qx, qy, qz, qw.

Field names per stream kind (``kind`` is the profile's ``kind`` of the stream)
--------------------------------------------------------------------------------
``teleop_state`` (TeleopState): bool ``armed``, ``motion_enabled``, ``clutch_engaged``,
``calibrated``, ``measured_tool_pose_valid``, ``damped``, ``keepout_bound``,
``gripper_enabled``, ``gripper_active``, ``gripper_measured_valid``; int ``owner``,
``mux_status``, ``hold_reason``, ``active_limit``, ``servo_status``, ``gripper_hold``,
``gripper_request``, ``gripper_reference_source``, ``calibration_session_id``,
``calibration_reset_count``, ``input_seq``; float ``calibration_heading``, ``input_age``,
``damping``, ``lead_translation``, ``lead_rotation``, ``translation_scale``,
``gripper_command``, ``gripper_measured``, ``gripper_measured_stamp``;
pose ``measured_tool_pose``.
``pose`` (PoseStamped): pose ``pose``.
``gripper_command`` (GripperCommand): float ``position``; int ``request``,
``reference_source``.
``joint_command`` (JointJog): float ``velocities`` ``(n, J)`` in ``joint_names`` order.
``joint_states`` (JointState): float ``position``, ``velocity``, ``effort`` ``(n, J)`` in
``joint_names`` order (the first message's order; a joint missing from a message is NaN).
``image`` (Image): int ``width``, ``height``, ``step``; str list ``encoding`` (in ``labels``).
Pixels are not held here; ``Episode.images[name]`` loads one frame by row index.
``camera_info`` (CameraInfo): int ``width``, ``height``; float ``k`` ``(n, 9)``,
``d`` (first message's length) ; ``labels['distortion_model']``.
``input_status`` (InputStatus): int ``session_id``, ``seq``, ``reset_count``; float
``xr_time_ms``, ``client_time_ms``; bool ``right_present``, ``left_present``.
``joy`` (Joy): float ``axes`` ``(n, A)``; int ``buttons`` ``(n, B)``.
``episode_state`` (EpisodeState): int ``state``; float ``wall_time`` (host wall clock at
``stamp``, seconds since the epoch).
``odometry`` (Odometry): pose ``pose``; float ``twist`` ``(n, 6)``.
``twist`` (Twist, no header): float ``twist`` ``(n, 6)``.
``owner`` (ArmOwner): int ``owner``, ``status``.
Kinds not listed (tf, strings, logs) carry no fields; only their times are used.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class Stream:
    """One recorded topic of one episode."""

    name: str                      # the profile's stream key, e.g. 'state', 'camera_gantry'
    topic: str
    kind: str                      # see the module docstring
    stamp: np.ndarray              # (n,) float64 s, header stamp as recorded
    recv: np.ndarray               # (n,) float64 s, recorder receive time
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    frame_id: List[str] = field(default_factory=list)       # per row ('' without a header)
    joint_names: List[str] = field(default_factory=list)    # joint kinds only
    labels: Dict[str, List[str]] = field(default_factory=dict)   # per-row strings
    has_header: bool = True

    def __post_init__(self):
        """Check the shapes, so a malformed table fails here and not deep in a check."""
        self.stamp = np.asarray(self.stamp, dtype=float).reshape(-1)
        self.recv = np.asarray(self.recv, dtype=float).reshape(-1)
        n = len(self.stamp)
        if len(self.recv) != n:
            raise ValueError(f'{self.name}: {n} stamps but {len(self.recv)} receive times')
        for key, value in self.fields.items():
            value = np.asarray(value)
            if value.shape[:1] != (n,):
                raise ValueError(f'{self.name}.{key}: {value.shape[:1]} rows, expected {n}')
            self.fields[key] = value
        if self.frame_id and len(self.frame_id) != n:
            raise ValueError(f'{self.name}: {len(self.frame_id)} frame ids, expected {n}')

    def __len__(self) -> int:
        """Return the number of recorded messages."""
        return len(self.stamp)


@dataclass
class Episode:
    """One episode: its stream tables, its sidecar and the profile it was recorded under."""

    streams: Dict[str, Stream]
    sidecar: dict                  # the episode JSON (see metadata.py)
    profile: dict                  # the deployment profile (see profile.py), as in the sidecar
    images: Dict[str, Callable[[int], np.ndarray]] = field(default_factory=dict)

    def get(self, name: str) -> Optional[Stream]:
        """Return the stream ``name``, or None when it was not recorded."""
        return self.streams.get(name)
