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
Parameters of the safety gate (SAFETY_GATE.md, "Values"). No ROS imports.

Values the gate document marks PENDING have **no default**: ``build_params`` raises
``MissingValue`` naming every one that is absent, and the node refuses to start.
Lengths are metres, angles are given in degrees in the parameter names that say so and
stored in radians.
"""

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence, Tuple

from .rotations import Vec3

# PENDING in the gate document: required, no default (INV-10).
REQUIRED = (
    'max_offset_translation_m',
    'max_offset_rotation_deg',
    'view_cone_half_angle_deg',
    'step_translation_limit_m',
    'step_rotation_limit_deg',
    'linear_speed_cap_m_s',
    'angular_speed_cap_rad_s',
    'max_lead_translation_m',       # G-14
    'max_lead_rotation_deg',
)

# Set in the gate document.
INPUT_TIMEOUT_S = 0.1
TARGET_RATE_HZ = 50.0

# Not safety limits: defaults that a deployment may tune.
DEFAULTS = {
    'motion_enabled': False,            # G-1
    'filter_cutoff_hz': 8.0,            # 0 disables the low-pass filter
    'view_cone_tilt_deg': 0.0,          # cone axis tilted down from the headset's forward axis
    'keep_out_planes': (),              # flat list, 4 numbers per plane: nx, ny, nz, d
    'wrist_offset': (0.0, 0.0, 0.0),    # wrist point in the tool frame (keep-out check)
    'wrist_pivot_offset': (0.0, 0.0, 0.0),
    'pointing_axis': (1.0, 0.0, 0.0),
    'calibration_min_horizontal': 0.5,  # refuse a pointing axis within 30 deg of vertical
    'tool_pose_timeout_s': 0.1,
    'servo_ready_timeout_s': 0.5,
}


class MissingValue(ValueError):
    """A required (PENDING) value was not supplied."""


@dataclass(frozen=True)
class Plane:
    """A keep-out half-space in the arm base frame: allowed where ``n . x >= d``."""

    n: Vec3
    d: float


@dataclass(frozen=True)
class GateParams:
    """Validated gate parameters; angles in radians."""

    motion_enabled: bool
    input_timeout: float
    target_period: float
    max_offset_translation: float
    max_offset_rotation: float
    view_cone_angle: float
    view_cone_tilt: float
    step_translation_limit: float
    step_rotation_limit: float
    linear_speed_cap: float
    angular_speed_cap: float
    max_lead_translation: float
    max_lead_rotation: float
    filter_cutoff_hz: float
    keep_out_planes: Tuple[Plane, ...]
    wrist_offset: Vec3
    pivot_offset: Vec3
    pointing_axis: Vec3
    calibration_min_horizontal: float
    tool_pose_timeout: float
    servo_ready_timeout: float

    @property
    def raw_speed_limit(self) -> float:
        """Largest raw controller speed (m/s): the per-cycle step over one target period."""
        return self.step_translation_limit / self.target_period

    @property
    def raw_rate_limit(self) -> float:
        """Largest raw controller angular rate (rad/s)."""
        return self.step_rotation_limit / self.target_period


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}: expected a number, got {value!r}')
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f'{name}: expected a finite positive number, got {value!r}')
    return value


def _vec3(name: str, value: Sequence[float]) -> Vec3:
    if len(value) != 3 or not all(math.isfinite(float(v)) for v in value):
        raise ValueError(f'{name}: expected 3 finite numbers, got {value!r}')
    return (float(value[0]), float(value[1]), float(value[2]))


def parse_planes(flat: Sequence[float]) -> Tuple[Plane, ...]:
    """Return the planes of a flat ``[nx, ny, nz, d, ...]`` list, normals normalised."""
    flat = [float(v) for v in flat]
    if len(flat) % 4 != 0:
        raise ValueError('keep_out_planes: expected 4 numbers per plane (nx, ny, nz, d)')
    planes = []
    for i in range(0, len(flat), 4):
        nx, ny, nz, d = flat[i:i + 4]
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if not math.isfinite(length) or length < 1e-9 or not math.isfinite(d):
            raise ValueError(f'keep_out_planes: plane {i // 4} is not a valid half-space')
        planes.append(Plane((nx / length, ny / length, nz / length), d / length))
    return tuple(planes)


def build_params(values: Mapping[str, Any]) -> GateParams:
    """
    Build the gate parameters from a name-to-value mapping.

    A required name that is missing or ``None`` raises ``MissingValue`` listing all of
    them; nothing is substituted. Optional names fall back to ``DEFAULTS``.
    """
    missing = [name for name in REQUIRED if values.get(name) is None]
    if missing:
        raise MissingValue(
            'required safety values without a default are not set: ' + ', '.join(missing))

    def opt(name: str) -> Any:
        value = values.get(name)
        return DEFAULTS[name] if value is None else value

    cutoff = float(opt('filter_cutoff_hz'))
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError(f'filter_cutoff_hz: expected a number >= 0, got {cutoff!r}')
    pointing = _vec3('pointing_axis', opt('pointing_axis'))
    if math.sqrt(sum(v * v for v in pointing)) < 1e-9:
        raise ValueError('pointing_axis: must not be zero')
    min_horizontal = float(opt('calibration_min_horizontal'))
    if not 0.0 < min_horizontal <= 1.0:
        raise ValueError('calibration_min_horizontal: expected a value in (0, 1]')
    tilt_deg = float(opt('view_cone_tilt_deg'))
    if not math.isfinite(tilt_deg) or not -90.0 <= tilt_deg <= 90.0:
        raise ValueError(f'view_cone_tilt_deg: expected -90 to 90, got {tilt_deg!r}')
    tilt = math.radians(tilt_deg)
    cone = math.radians(_positive('view_cone_half_angle_deg', values['view_cone_half_angle_deg']))
    if cone > math.pi:
        raise ValueError('view_cone_half_angle_deg: expected at most 180')
    return GateParams(
        motion_enabled=bool(opt('motion_enabled')),
        input_timeout=INPUT_TIMEOUT_S,
        target_period=1.0 / TARGET_RATE_HZ,
        max_offset_translation=_positive('max_offset_translation_m',
                                         values['max_offset_translation_m']),
        max_offset_rotation=math.radians(_positive('max_offset_rotation_deg',
                                                   values['max_offset_rotation_deg'])),
        view_cone_angle=cone,
        view_cone_tilt=tilt,
        step_translation_limit=_positive('step_translation_limit_m',
                                         values['step_translation_limit_m']),
        step_rotation_limit=math.radians(_positive('step_rotation_limit_deg',
                                                   values['step_rotation_limit_deg'])),
        linear_speed_cap=_positive('linear_speed_cap_m_s', values['linear_speed_cap_m_s']),
        angular_speed_cap=_positive('angular_speed_cap_rad_s',
                                    values['angular_speed_cap_rad_s']),
        max_lead_translation=_positive('max_lead_translation_m',
                                       values['max_lead_translation_m']),
        max_lead_rotation=math.radians(_positive('max_lead_rotation_deg',
                                                 values['max_lead_rotation_deg'])),
        filter_cutoff_hz=cutoff,
        keep_out_planes=parse_planes(opt('keep_out_planes')),
        wrist_offset=_vec3('wrist_offset', opt('wrist_offset')),
        pivot_offset=_vec3('wrist_pivot_offset', opt('wrist_pivot_offset')),
        pointing_axis=pointing,
        calibration_min_horizontal=min_horizontal,
        tool_pose_timeout=_positive('tool_pose_timeout_s', opt('tool_pose_timeout_s')),
        servo_ready_timeout=_positive('servo_ready_timeout_s', opt('servo_ready_timeout_s')),
    )


def replace_wrist_offset(params: GateParams, offset: Sequence[float]) -> GateParams:
    """Return params with the wrist point (tool frame) set; not a limit, a geometry fact."""
    return replace(params, wrist_offset=_vec3('wrist_offset', offset))
