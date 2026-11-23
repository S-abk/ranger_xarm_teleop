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
"""The base command and state streams are in the simulation profiles (no ROS)."""

import pytest
from ranger_xarm_teleop_record import profile


@pytest.mark.parametrize('name', ['gazebo', 'mock'])
def test_the_simulation_profiles_record_the_base_command_and_state_optionally(name):
    streams = profile.load_profile(name)['streams']
    cmd, state = streams['base_cmd'], streams['base_state']
    assert (cmd['topic'], cmd['kind'], cmd['type']) == (
        '/teleop/base/cmd_vel', 'twist_stamped', 'geometry_msgs/msg/TwistStamped')
    assert (state['topic'], state['kind'], state['type']) == (
        '/teleop/base/state', 'base_state',
        'ranger_xarm_teleop_interfaces/msg/BaseState')
    for stream in (cmd, state):
        assert stream['required'] is False and stream['rate_hz'] is None
        assert stream['clock'] == 'robot' and stream['record'] is True
    assert cmd['base'] is True and state['base'] is False
    assert profile.period(cmd) is None and profile.period(state) is None


@pytest.mark.parametrize('name', ['gazebo', 'mock'])
def test_the_existing_base_streams_are_unchanged(name):
    streams = profile.load_profile(name)['streams']
    assert streams['cmd_vel']['topic'] == '/cmd_vel' and streams['cmd_vel']['kind'] == 'twist'
    assert streams['odom']['topic'] == '/odom' and streams['odom']['base'] is True
    assert streams['ground_truth_odom']['topic'] == '/ground_truth/odom'
    for stream in ('cmd_vel', 'odom', 'ground_truth_odom'):
        assert streams[stream]['required'] is False


def test_the_hardware_profile_still_loads_without_the_base_command_streams():
    streams = profile.load_profile('hardware')['streams']
    assert 'base_cmd' not in streams and 'base_state' not in streams


def test_the_new_kinds_are_known_to_the_profile_schema():
    assert 'twist_stamped' in profile.KINDS and 'base_state' in profile.KINDS
