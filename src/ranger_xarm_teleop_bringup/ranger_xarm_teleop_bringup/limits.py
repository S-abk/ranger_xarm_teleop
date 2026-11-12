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
    for section in data.values():
        if isinstance(section, dict):
            merged.update(section.get('ros__parameters') or {})
    return merged


def check_limits_file(path: str, simulation: bool) -> dict:
    """
    Return the limit values of ``path``; refuse simulation-only values outside simulation.

    Raises ``SimulationOnlyRefused`` when the file is marked simulation-only and
    ``simulation`` is false, and ``ValueError`` when a speed cap is missing (MoveIt Servo
    needs both; they have no default).
    """
    values = load_limits(path)
    if values.get(SIMULATION_ONLY_KEY) and not simulation:
        raise SimulationOnlyRefused(
            f'{path} holds simulation-only limit values; it is refused unless the launch '
            'path is a simulation (simulation:=true, set only by the mock and simulator '
            'launch files)')
    for name in ('linear_speed_cap_m_s', 'angular_speed_cap_rad_s'):
        if not isinstance(values.get(name), (int, float)) or isinstance(values.get(name), bool):
            raise ValueError(f'{path}: {name} is required and has no default')
    return values
