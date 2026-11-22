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
The hardware limits refusal (no ROS): no value has a default, none is shipped.

Every number in this file is generated into ``tmp_path`` here, from the simulation-only
file's values; no file under ``src/*/config/`` other than that one may pass the hardware
check, so no acceptable hardware limits file is ever part of the repository.
"""

import math
import os

import pytest
from ranger_xarm_teleop_bringup import limits
from ranger_xarm_teleop_bringup.limits import (check_hardware_limits_file, HARDWARE_REQUIRED,
                                               HardwareLimitsRefused, load_limits,
                                               SimulationOnlyRefused)
from ranger_xarm_teleop_servo import params
import yaml

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.dirname(PKG)
CONFIG = os.path.join(PKG, 'config')
TEMPLATE = os.path.join(CONFIG, 'hardware_limits_TEMPLATE.yaml')
SIM_LIMITS = os.path.join(CONFIG, 'SIMULATION_ONLY_limits.yaml')


def good_values():
    """Numbers for every hardware value, generated from the simulation file; test-only."""
    sim = load_limits(SIM_LIMITS)
    values = {name: sim[name] * 0.5 for name in HARDWARE_REQUIRED if name in sim}
    values['tool_pose_timeout_s'] = 1.0 * 0.1
    assert sorted(values) == sorted(HARDWARE_REQUIRED)
    return {name: values[name] for name in HARDWARE_REQUIRED}


def write(tmp_path, values, name='limits.yaml'):
    path = tmp_path / name
    path.write_text(yaml.safe_dump({'/**': {'ros__parameters': values}}), encoding='utf-8')
    return str(path)


def lines_of(err):
    return str(err.value).splitlines()


def test_the_required_names_are_the_gates_pending_values_and_two_more():
    assert HARDWARE_REQUIRED[:len(params.REQUIRED)] == params.REQUIRED
    assert HARDWARE_REQUIRED[len(params.REQUIRED):] == ('link_keepout_margin_m',
                                                        'tool_pose_timeout_s')
    assert len(HARDWARE_REQUIRED) == 12


def test_the_template_holds_the_twelve_names_in_order_all_pending():
    values = load_limits(TEMPLATE)
    assert tuple(values) == HARDWARE_REQUIRED
    assert set(values.values()) == {'PENDING'}
    assert limits.SIMULATION_ONLY_KEY not in values


def test_the_template_carries_no_number_in_a_value():
    with open(TEMPLATE, encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    for value in data['/**']['ros__parameters'].values():
        assert isinstance(value, str)


def test_the_template_has_the_marker_on_its_first_line_and_only_there():
    with open(TEMPLATE, encoding='utf-8') as handle:
        text = handle.read()
    assert text.splitlines()[0] == '# HARDWARE LIMITS TEMPLATE'
    assert len(limits.HARDWARE_TEMPLATE_MARKER.findall(text)) == 1


def test_the_template_is_refused_and_says_what_it_is():
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(TEMPLATE)
    first = lines_of(err)[0]
    assert 'unfilled template' in first and 'delete the marker line' in first
    assert 'copy it outside the checkout' in first


@pytest.mark.parametrize('name', HARDWARE_REQUIRED)
def test_the_template_refusal_names_each_value(name):
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(TEMPLATE)
    line = f'{TEMPLATE}: {name} is PENDING: set it in your copy of the template ' \
           '(docs/HARDWARE_PROCEDURE.md, "Values you supply")'
    assert line in lines_of(err)


def test_the_template_refusal_has_one_line_per_problem():
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(TEMPLATE)
    assert len(lines_of(err)) == 1 + len(HARDWARE_REQUIRED)
    assert all(line.startswith(f'{TEMPLATE}: ') for line in lines_of(err))


def test_a_complete_file_is_accepted_and_its_values_returned(tmp_path):
    values = good_values()
    assert check_hardware_limits_file(write(tmp_path, values)) == values


def test_a_complete_file_with_scales_within_the_largest_is_accepted(tmp_path):
    values = dict(good_values(), translation_scale_fine=0.5, translation_scale_coarse=1.0)
    values['max_translation_scale'] = 1.0
    assert check_hardware_limits_file(write(tmp_path, values)) == values


def test_a_copy_of_the_template_with_the_marker_kept_is_refused(tmp_path):
    path = tmp_path / 'copy.yaml'
    with open(TEMPLATE, encoding='utf-8') as handle:
        marker = handle.read().splitlines()[0]
    path.write_text(marker + '\n' + yaml.safe_dump(
        {'/**': {'ros__parameters': good_values()}}), encoding='utf-8')
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(str(path))
    assert lines_of(err) == [lines_of(err)[0]] and 'unfilled template' in lines_of(err)[0]


@pytest.mark.parametrize('name', HARDWARE_REQUIRED)
@pytest.mark.parametrize('case', ['missing', 'PENDING', 'abc', 0, math.nan, math.inf, -1.0,
                                  True, None])
def test_one_bad_value_is_refused_with_a_line_naming_it(tmp_path, name, case):
    values = good_values()
    if case == 'missing':
        del values[name]
    else:
        values[name] = case
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(write(tmp_path, values))
    assert len(lines_of(err)) == 1
    line = lines_of(err)[0]
    path = str(tmp_path / 'limits.yaml')
    assert line.startswith(f'{path}: {name} is ')
    assert 'docs/HARDWARE_PROCEDURE.md, "Values you supply"' in line
    if case == 'missing':
        assert f'{name} is missing: ' in line
    elif case == 'PENDING':
        assert f'{name} is PENDING: ' in line
    else:
        assert f'{name} is not a positive number (got ' in line
        assert f'(got {case!r})' in line


def test_every_problem_is_listed_not_only_the_first(tmp_path):
    values = good_values()
    del values['max_offset_translation_m']
    values['linear_speed_cap_m_s'] = 'PENDING'
    values['tool_pose_timeout_s'] = 0
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(write(tmp_path, values))
    text = str(err.value)
    assert len(lines_of(err)) == 3
    for name in ('max_offset_translation_m', 'linear_speed_cap_m_s', 'tool_pose_timeout_s'):
        assert name in text


def test_a_missing_or_unreadable_file_is_refused(tmp_path):
    with pytest.raises(HardwareLimitsRefused, match='cannot be read'):
        check_hardware_limits_file(str(tmp_path / 'absent.yaml'))
    bad = tmp_path / 'bad.yaml'
    bad.write_text('a: [unclosed', encoding='utf-8')
    with pytest.raises(HardwareLimitsRefused, match='cannot be read'):
        check_hardware_limits_file(str(bad))


def test_a_file_without_parameters_names_every_value_as_missing(tmp_path):
    path = tmp_path / 'empty.yaml'
    path.write_text('', encoding='utf-8')
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(str(path))
    assert len(lines_of(err)) == len(HARDWARE_REQUIRED)
    assert all(' is missing: ' in line for line in lines_of(err))


def test_the_simulation_only_file_is_refused():
    with pytest.raises(SimulationOnlyRefused, match='never loads them'):
        check_hardware_limits_file(SIM_LIMITS)


def test_a_marked_copy_of_good_values_is_refused(tmp_path):
    values = dict(good_values(), limits_are_simulation_only=True)
    with pytest.raises(SimulationOnlyRefused):
        check_hardware_limits_file(write(tmp_path, values))


def test_gripper_teleop_enabled_is_refused(tmp_path):
    values = dict(good_values(), gripper_enabled=True)
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(write(tmp_path, values))
    assert len(lines_of(err)) == 1 and 'gripper_enabled is True' in lines_of(err)[0]
    assert 'simulation only' in lines_of(err)[0]


def test_gripper_teleop_explicitly_off_is_accepted(tmp_path):
    values = dict(good_values(), gripper_enabled=False)
    assert check_hardware_limits_file(write(tmp_path, values)) == values


@pytest.mark.parametrize('value', [0.2, 'PENDING', None])
def test_the_reset_speed_scaling_is_refused(tmp_path, value):
    values = dict(good_values(), reset_speed_scaling=value)
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(write(tmp_path, values))
    assert len(lines_of(err)) == 1
    assert 'reset_speed_scaling is set: reset is simulation only' in lines_of(err)[0]


@pytest.mark.parametrize('name', ['translation_scale_coarse', 'translation_scale_fine'])
def test_a_scale_above_the_largest_is_refused(tmp_path, name):
    values = good_values()
    values[name] = values['max_translation_scale'] * 2
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(write(tmp_path, values))
    assert len(lines_of(err)) == 1
    assert f'{name} ' in lines_of(err)[0] and 'above max_translation_scale' in lines_of(err)[0]


@pytest.mark.parametrize('name', ['translation_scale_coarse', 'translation_scale_fine'])
def test_a_scale_that_is_not_a_positive_number_is_refused(tmp_path, name):
    values = dict(good_values(), **{name: 'abc'})
    with pytest.raises(HardwareLimitsRefused) as err:
        check_hardware_limits_file(write(tmp_path, values))
    assert f"{name} is not a positive number (got 'abc')" in lines_of(err)[0]


def test_check_limits_file_is_unchanged_for_the_simulation_file():
    assert limits.check_limits_file(SIM_LIMITS, simulation=True)['max_translation_scale'] == 2.0
    with pytest.raises(SimulationOnlyRefused):
        limits.check_limits_file(SIM_LIMITS, simulation=False)


def shipped_yaml_files():
    found = []
    for root, _, names in os.walk(SRC):
        if f'{os.sep}config' not in root + os.sep or 'build' in root.split(os.sep):
            continue
        found.extend(os.path.join(root, n) for n in names if n.endswith(('.yaml', '.yml')))
    return sorted(found)


def test_the_walk_finds_the_shipped_config_files():
    names = {os.path.basename(p) for p in shipped_yaml_files()}
    assert {'SIMULATION_ONLY_limits.yaml', 'hardware_limits_TEMPLATE.yaml', 'servo.yaml',
            'checks.yaml'} <= names


def test_no_shipped_file_would_pass_the_hardware_check():
    for path in shipped_yaml_files():
        with pytest.raises((HardwareLimitsRefused, SimulationOnlyRefused)):
            check_hardware_limits_file(path)
