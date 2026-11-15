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

import yaml

SIMULATION_ONLY_KEY = 'limits_are_simulation_only'


class SimulationOnlyRefused(RuntimeError):
    """A simulation-only limits file was given to a launch path that is not a simulation."""


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
            'max_lead_translation_m', 'max_lead_rotation_deg', 'tool_pose_timeout_s',
            'joint_velocity_limit_names', 'joint_velocity_limits')


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
    ``simulation`` is false, and ``ValueError`` when a speed cap or a lead limit is missing
    (the teleop node enforces them itself; they have no default).
    """
    values = load_limits(path)
    if values.get(SIMULATION_ONLY_KEY) and not simulation:
        raise SimulationOnlyRefused(
            f'{path} holds simulation-only limit values; it is refused unless the launch '
            'path is a simulation (simulation:=true, set only by the mock and simulator '
            'launch files)')
    for name in ('linear_speed_cap_m_s', 'angular_speed_cap_rad_s', 'max_lead_translation_m',
                 'max_lead_rotation_deg'):
        if not isinstance(values.get(name), (int, float)) or isinstance(values.get(name), bool):
            raise ValueError(f'{path}: {name} is required and has no default')
    return values
