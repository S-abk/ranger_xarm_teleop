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
"""Base driving's values (G-19): required with base:=true, simulation only, refused on hardware."""

import os

import pytest
from ranger_xarm_teleop_bringup.limits import (BASE_REQUIRED, check_base_limits,
                                               check_extra_params_file,
                                               check_hardware_limits_file,
                                               HardwareLimitsRefused, load_limits)
from test_hardware_limits import good_values, SIM_LIMITS, TEMPLATE, write


def test_the_simulation_file_holds_every_required_base_value():
    values = check_base_limits(SIM_LIMITS, load_limits(SIM_LIMITS))
    assert sorted(values) == sorted(BASE_REQUIRED)
    assert values['base_linear_cap'] < 0.25 and values['base_turn_cap'] < 0.5
    assert values['base_linear_accel'] < 1.0
    assert values['base_turn_recalibrate_deg'] == 5.0


@pytest.mark.parametrize('name', BASE_REQUIRED)
@pytest.mark.parametrize('case', ['missing', 0.0, -1.0, 'fast', True])
def test_each_base_value_is_required_and_positive(tmp_path, name, case):
    values = dict(load_limits(SIM_LIMITS))
    if case == 'missing':
        values.pop(name)
    else:
        values[name] = case
    with pytest.raises(ValueError, match=name):
        check_base_limits(write(tmp_path, values), values)


def test_the_tool_speed_cap_is_optional_and_positive_when_set(tmp_path):
    values = dict(load_limits(SIM_LIMITS))
    assert 'base_tool_speed_cap' not in check_base_limits(SIM_LIMITS, values)
    values['base_tool_speed_cap'] = 0.15
    assert check_base_limits(SIM_LIMITS, values)['base_tool_speed_cap'] == 0.15
    values['base_tool_speed_cap'] = 0.0
    with pytest.raises(ValueError, match='base_tool_speed_cap'):
        check_base_limits(SIM_LIMITS, values)


@pytest.mark.parametrize('name', BASE_REQUIRED + ('base_driving', 'base_tool_speed_cap'))
def test_the_hardware_path_refuses_every_base_value(tmp_path, name):
    values = {**good_values(), name: True if name == 'base_driving' else 0.1}
    with pytest.raises(HardwareLimitsRefused, match='base driving'):
        check_hardware_limits_file(write(tmp_path, values))


def test_the_template_names_no_base_value():
    with open(TEMPLATE, encoding='utf-8') as handle:
        text = handle.read()
    for name in BASE_REQUIRED + ('base_tool_speed_cap',):
        assert f'{name}:' not in text


@pytest.mark.parametrize('name', BASE_REQUIRED + ('base_driving', 'base_tool_speed_cap'))
def test_an_extra_parameter_file_may_not_set_a_base_value(tmp_path, name):
    with pytest.raises(ValueError):
        check_extra_params_file(write(tmp_path, {name: 1.0}))


def test_the_simulation_file_says_simulation_only():
    with open(SIM_LIMITS, encoding='utf-8') as handle:
        assert handle.readline().startswith('# SIMULATION ONLY')
    assert os.path.basename(SIM_LIMITS).startswith('SIMULATION_ONLY')
