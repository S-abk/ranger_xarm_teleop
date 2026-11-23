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
"""Helpers shared by the launch files: limit values and the simulation-only refusal."""

import math
import re

from ranger_xarm_teleop_servo import joint_accel
import yaml

SIMULATION_ONLY_KEY = 'limits_are_simulation_only'


class SimulationOnlyRefused(RuntimeError):
    """A simulation-only limits file was given to a launch path that is not a simulation."""


class HardwareLimitsRefused(ValueError):
    """A limits file is not acceptable for the hardware path; the message lists every problem."""


def load_limits(path: str) -> dict:
    """Return the ``ros__parameters`` of a limits file (any node-name key)."""
    with open(path, encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    merged = {}

    def collect(section) -> None:
        # ROS parameter files may nest namespaces and node names above ros__parameters.
        if not isinstance(section, dict):
            return
        for key, value in section.items():
            if key == 'ros__parameters' and isinstance(value, dict):
                merged.update(value)
            else:
                collect(value)

    collect(data)
    return merged


# What an extra parameter file of the teleop node may not set: the limit values and the
# switches that guard them come from the limits file and the launch arguments only.
RESERVED = ('limits_are_simulation_only', 'motion_enabled', 'max_offset_translation_m',
            'max_offset_rotation_deg', 'view_cone_half_angle_deg', 'step_translation_limit_m',
            'step_rotation_limit_deg', 'linear_speed_cap_m_s', 'angular_speed_cap_rad_s',
            'max_lead_translation_m', 'max_lead_rotation_deg', 'max_translation_scale',
            'link_keepout_margin_m', 'link_keepout_cell_m', 'tool_pose_timeout_s',
            'joint_velocity_limit_names', 'joint_velocity_limits',
            # Gripper teleop: its switch and its safety values.
            'gripper_enabled', 'gripper_joint', 'gripper_open_position',
            'gripper_closed_position', 'gripper_max_speed',
            # The reset (G-16): its speed scaling.
            'reset_speed_scaling',
            # The joint command's speed cap and acceleration limit (G-18).
            'joint_speed_cap_rad_s', 'joint_acceleration_limit_rad_s2',
            # The hardware path's guards: the node's check of the hardware values, and the
            # mux's refusal of teleop while motion is disabled.
            'hardware_limits', 'refuse_teleop')

# Required in the limits file when gripper teleop is enabled (no default).
GRIPPER_REQUIRED = ('gripper_open_position', 'gripper_closed_position', 'gripper_max_speed')


def check_extra_params_file(path: str) -> None:
    """Raise ``ValueError`` when an extra parameter file sets a limit or a guard switch."""
    found = sorted(set(load_limits(path)) & set(RESERVED))
    if found:
        raise ValueError(
            f'{path} sets {", ".join(found)}: limit values and their guards may only come '
            'from the limits file and the launch arguments')


def check_limits_file(path: str, simulation: bool) -> dict:
    """
    Return the limit values of ``path``; refuse simulation-only values outside simulation.

    Raises ``SimulationOnlyRefused`` when the file is marked simulation-only and
    ``simulation`` is false, and ``ValueError`` when a speed cap, a lead limit, the
    largest translation scale, the link keep-out margin, the joint speed cap or the joint
    acceleration limit is missing (the teleop node enforces them itself; they have no
    default), or when a stop from the joint speed cap would take longer than G-18 allows.
    """
    values = load_limits(path)
    if values.get(SIMULATION_ONLY_KEY) and not simulation:
        raise SimulationOnlyRefused(
            f'{path} holds simulation-only limit values; it is refused unless the launch '
            'path is a simulation (simulation:=true, set only by the mock and simulator '
            'launch files)')
    for name in ('linear_speed_cap_m_s', 'angular_speed_cap_rad_s', 'max_lead_translation_m',
                 'max_lead_rotation_deg', 'max_translation_scale', 'link_keepout_margin_m',
                 *joint_accel.NAMES):
        if not isinstance(values.get(name), (int, float)) or isinstance(values.get(name), bool):
            raise ValueError(f'{path}: {name} is required and has no default')
    try:
        joint_accel.check_values(values[joint_accel.ACCELERATION_NAME],
                                 values[joint_accel.SPEED_CAP_NAME])
    except ValueError as err:
        raise ValueError(f'{path}: {err}') from err
    _check_stop_time(path, values)
    return values


def _check_stop_time(path: str, values: dict) -> None:
    """Raise ``ValueError`` when both G-18 values are valid and a stop would take too long."""
    try:
        joint_accel.check_values(values[joint_accel.ACCELERATION_NAME],
                                 values[joint_accel.SPEED_CAP_NAME])
    except (KeyError, ValueError):
        return      # missing or bad: reported as such, not as a stop time
    line = joint_accel.stop_time_refusal(values[joint_accel.ACCELERATION_NAME],
                                         values[joint_accel.SPEED_CAP_NAME])
    if line:
        raise ValueError(f'{path}: {line}')


def check_gripper_limits(path: str, values: dict) -> None:
    """Raise ``ValueError`` when gripper teleop is enabled and a gripper value is missing."""
    for name in GRIPPER_REQUIRED:
        if not isinstance(values.get(name), (int, float)) or isinstance(values.get(name), bool):
            raise ValueError(f'{path}: {name} is required with gripper teleop and has no '
                             'default')


# The values a hardware limits file must supply, in the order of the shipped template: the
# gate's PENDING values (ranger_xarm_teleop_servo.params.REQUIRED) with the joint speed cap and
# the joint acceleration limit (G-18) after the angular speed cap, then the link keep-out
# margin and the tool pose timeout. None has a default or a shipped number.
HARDWARE_REQUIRED = (
    'max_offset_translation_m',
    'max_offset_rotation_deg',
    'view_cone_half_angle_deg',
    'step_translation_limit_m',
    'step_rotation_limit_deg',
    'linear_speed_cap_m_s',
    'angular_speed_cap_rad_s',
    joint_accel.SPEED_CAP_NAME,             # G-18: the cap is chosen first
    joint_accel.ACCELERATION_NAME,
    'max_lead_translation_m',
    'max_lead_rotation_deg',
    'max_translation_scale',
    'link_keepout_margin_m',
    'tool_pose_timeout_s',
)

# The first line of the shipped template. A copy that still has it is the unfilled template.
HARDWARE_TEMPLATE_MARKER = re.compile(r'^[ \t]*#[ \t]*HARDWARE LIMITS TEMPLATE[ \t]*$',
                                      re.MULTILINE)
_PROCEDURE = 'docs/HARDWARE_PROCEDURE.md, "Values you supply"'
_SCALES = ('translation_scale_coarse', 'translation_scale_fine')
_DEFAULT_SCALE = 1.0      # params.DEFAULTS of the teleop node, for a scale the file leaves out


def _positive_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def check_hardware_limits_file(path: str) -> dict:
    """
    Return the limit values of ``path``, which must be a hardware limits file; else refuse.

    Raises ``SimulationOnlyRefused`` for a file marked simulation-only (the hardware path
    never loads simulation-only values) and ``HardwareLimitsRefused``, with one line per
    problem and every problem listed, when the file cannot be read, still has the template's
    marker line, leaves any of ``HARDWARE_REQUIRED`` missing, ``PENDING`` or not a positive
    finite number, enables gripper teleop or sets the reset's speed scaling (both are
    simulation only), sets a translation scale above ``max_translation_scale``, or has a joint
    speed cap and acceleration limit whose stop from the cap would take longer than G-18
    allows (``joint_accel.stop_time_refusal``).
    """
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
        values = load_limits(path)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as err:
        raise HardwareLimitsRefused(f'{path}: cannot be read as a limits file: {err}') from err
    if values.get(SIMULATION_ONLY_KEY):
        raise SimulationOnlyRefused(
            f'{path} holds simulation-only limit values; the hardware path never loads '
            f'them: write your own file from the hardware template ({_PROCEDURE})')
    problems = []
    if HARDWARE_TEMPLATE_MARKER.search(text):
        problems.append(
            f'{path}: this is the unfilled template: copy it outside the checkout, fill in '
            'every value and delete the marker line (the first line, "HARDWARE LIMITS '
            'TEMPLATE")')
    for name in HARDWARE_REQUIRED + _SCALES:
        if name not in values:
            if name in HARDWARE_REQUIRED:
                problems.append(f'{path}: {name} is missing: add it to your copy of the '
                                f'template ({_PROCEDURE})')
            continue
        value = values[name]
        if isinstance(value, str) and value.strip().upper() == 'PENDING':
            problems.append(f'{path}: {name} is PENDING: set it in your copy of the template '
                            f'({_PROCEDURE})')
        elif not _positive_number(value):
            problems.append(f'{path}: {name} is not a positive number (got {value!r}): set it '
                            f'to a finite number above zero ({_PROCEDURE})')
    largest = values.get('max_translation_scale')
    if _positive_number(largest):
        for name in _SCALES:
            if _positive_number(values.get(name)) and values[name] > largest:
                problems.append(f'{path}: {name} ({values[name]!r}) is above '
                                f'max_translation_scale ({largest!r}): lower it or raise '
                                'max_translation_scale (a scale is never clamped into range)')
            elif name not in values and _DEFAULT_SCALE > largest:
                # The teleop node's default scale (1.0) applies when the file sets none.
                problems.append(f'{path}: max_translation_scale ({largest!r}) is below the '
                                f'default {name} ({_DEFAULT_SCALE:g}): set {name} in your file '
                                'to at most max_translation_scale')
    try:
        _check_stop_time(path, values)
    except ValueError as err:
        problems.append(str(err))
    if values.get('gripper_enabled') not in (None, False):
        problems.append(f'{path}: gripper_enabled is {values["gripper_enabled"]!r}: gripper '
                        'teleop is simulation only; remove it from the file')
    if 'reset_speed_scaling' in values:
        problems.append(f'{path}: reset_speed_scaling is set: reset is simulation only; '
                        'remove it from the file')
    if problems:
        raise HardwareLimitsRefused('\n'.join(problems))
    return values
