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
``gripper_enabled``, ``gripper_active``, ``gripper_measured_valid``,
``joint_acceleration_limited``, ``stopping`` (G-18; False in recordings made before
TeleopState carried them, when no such limit existed); int ``owner``,
``mux_status``, ``hold_reason``, ``active_limit``, ``servo_status``, ``gripper_hold``,
``gripper_request``, ``gripper_reference_source``, ``calibration_session_id``,
``calibration_reset_count``, ``input_seq``; float ``calibration_heading``, ``input_age``,
``damping``, ``lead_translation``, ``lead_rotation``, ``translation_scale``,
``gripper_command``, ``gripper_measured``, ``gripper_measured_stamp``,
``measured_tool_pose_stamp`` (the stamp of the joint state the tool pose came from; 0 while
the pose is not valid; NaN in recordings made before TeleopState carried it);
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
``twist_stamped`` (TwistStamped): float ``twist`` ``(n, 6)``; the stamp is the header's.
``base_state`` (BaseState): int ``state``, ``mode``, ``reason``, ``active_limit``,
``event_count``; bool ``driving``, ``at_rest``, ``enabled``; float ``odom_age`` (NaN when no
odometry), ``tool_speed_scale`` (NaN when no such limit), ``command`` ``(n, 6)`` (the last
command published, as a twist); the stamp is the header's.
``stability_state`` (StabilityState, information only): bool ``available``, ``at_rest``; int
``reason``, ``tilt_source``, ``edge``, ``edge_pessimistic``, ``stop_edge``, ``energy_edge``;
float ``margin_m``, ``margin_pessimistic_m``, ``stop_margin_m``, ``stop_margin_pessimistic_m``,
``stop_decel_m_s2``, ``energy_reserve_j``, ``speed_m_s`` (the figures are NaN while not
``available``); str list ``reason_text`` and ``model_sha256`` (in ``labels``); the stamp is
the header's.
Kinds not listed (tf, strings, logs) carry no fields; only their times are used.

Absent fields
-------------
Some set-ups leave a field of a state message empty or NaN (a simulator whose joints declare
no effort interface publishes ``effort`` as NaN). ``OPTIONAL_FIELDS`` lists, per kind, the
fields that may be absent this way. A field is *absent* in an episode when no row has a
finite value in it. Absence is reported as information; a profile's ``required_fields`` on a
stream makes it fail the episode.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

OPTIONAL_FIELDS: Dict[str, tuple] = {
    'joint_states': ('position', 'velocity', 'effort'),
    'odometry': ('pose', 'twist'),
}


def any_finite(values: Sequence[float]) -> bool:
    """Return True when at least one value is a finite number (an empty sequence has none)."""
    return bool(np.any(np.isfinite(np.asarray(values, dtype=float))))


class FieldPresence:
    """Which optional fields had a finite value over the messages added (the episode node)."""

    def __init__(self, keys: Sequence[str]):
        """Track ``keys`` (a kind's OPTIONAL_FIELDS)."""
        self.keys = tuple(keys)
        self.n = 0
        self.finite = {k: False for k in self.keys}
        self.nonempty = {k: False for k in self.keys}
        self.last: Dict[str, bool] = {}            # finite in the newest message

    def add(self, values: Dict[str, Sequence[float]]) -> None:
        """Note one message's values per field."""
        self.n += 1
        for key in self.keys:
            value = values.get(key, ())
            finite = len(value) > 0 and any_finite(value)
            self.last[key] = finite
            self.finite[key] = self.finite[key] or finite
            self.nonempty[key] = self.nonempty[key] or len(value) > 0

    def absent(self) -> List[str]:
        """Return the fields with no finite value in any message added."""
        return [k for k in self.keys if not self.finite[k]]

    def report(self) -> dict:
        """Return ``{messages, absent, how}`` for a sidecar (how: empty, NaN, no message)."""
        how = {k: ('no message' if self.n == 0 else 'empty' if not self.nonempty[k]
                   else 'NaN') for k in self.absent()}
        return {'messages': self.n, 'absent': self.absent(), 'how': how}


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

    def absent_fields(self) -> List[str]:
        """Return this kind's optional fields with no finite value in any row."""
        out = []
        for key in OPTIONAL_FIELDS.get(self.kind, ()):
            value = self.fields.get(key)
            if value is None or not len(self) or not any_finite(np.ravel(value)):
                out.append(key)
        return out


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
