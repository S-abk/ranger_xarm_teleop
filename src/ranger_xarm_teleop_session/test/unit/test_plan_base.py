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
"""The session launcher's --base option: off by default, simulation only (no ROS)."""

import os

import pytest
from ranger_xarm_teleop_session import cli
from ranger_xarm_teleop_session import plan as pl
import yaml

PROFILES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                            'ranger_xarm_teleop_record', 'profiles')


def record_profile(name):
    with open(os.path.join(PROFILES_DIR, f'{name}.yaml'), encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def steps_of(**kwargs):
    opts = pl.Options(profile='gazebo', **kwargs)
    return {s.name: s for s in pl.build_plan(opts, checkout='/co', bringup_share='/share',
                                             session='S', log_dir='/logs', gpu='nvidia')}


def test_the_base_is_off_by_default_and_adds_nothing():
    steps = steps_of()
    assert pl.Options().base is False
    assert 'drive_base' not in pl.args_of(steps['simulator'].argv)
    assert 'base' not in pl.args_of(steps['stack'].argv)
    assert 'base' not in pl.args_of(steps['bridge'].argv)


def test_base_adds_drive_base_to_the_simulator_and_base_to_the_teleop_launch():
    steps = steps_of(base=True)
    sim = pl.args_of(steps['simulator'].argv)
    assert sim['drive_base'] == 'true'
    assert steps['simulator'].argv[:4] == ('ros2', 'launch', 'ranger_xarm_gazebo',
                                           'sim.launch.py')
    stack = pl.args_of(steps['stack'].argv)
    assert stack['base'] == 'true' and stack['reset'] == 'true' and stack['record'] == 'true'
    assert 'base' not in pl.args_of(steps['bridge'].argv)
    # Nothing else of the plan changes.
    plain = steps_of()
    assert {k: v for k, v in sim.items() if k != 'drive_base'} == pl.args_of(
        plain['simulator'].argv)
    assert {k: v for k, v in stack.items() if k != 'base'} == pl.args_of(plain['stack'].argv)
    assert steps['probe'].argv == plain['probe'].argv


def test_base_is_accepted_for_the_simulation_profile():
    assert pl.check_options(pl.Options(profile='gazebo', base=True),
                            record_profile('gazebo')) == []


def test_base_on_hardware_is_refused_in_one_line():
    hardware = record_profile('hardware')
    with pytest.raises(pl.PlanError, match='simulated base only') as raised:
        pl.check_options(pl.Options(profile='gazebo', base=True), hardware)
    assert '\n' not in str(raised.value)
    with pytest.raises(pl.PlanError, match='simulated base only'):
        pl.check_options(pl.Options(profile='gazebo', base=True), None)
    with pytest.raises(pl.PlanError, match='hardware bring-up is started by hand'):
        pl.check_options(pl.Options(profile='hardware', base=True), hardware)


def test_base_with_a_profile_that_starts_no_simulator_is_refused():
    with pytest.raises(pl.PlanError, match='--base needs a simulator'):
        pl.check_options(pl.Options(profile='mock', base=True), record_profile('mock'))
    assert pl.check_options(pl.Options(profile='mock'), record_profile('mock')) == []


def parsed(*argv):
    return cli.options_from(cli.parser().parse_args(['start', *argv]))


def test_the_command_line_option_is_off_unless_typed():
    assert parsed('--profile', 'gazebo').base is False
    opts = parsed('--profile', 'gazebo', '--base')
    assert opts.base is True and 'base' in opts.explicit


def test_the_relay_role_ignores_the_option_with_a_note():
    opts = pl.Options(role='relay', base=True, explicit=('role', 'base'))
    assert any('--base' in w for w in pl.check_options(opts, None))
