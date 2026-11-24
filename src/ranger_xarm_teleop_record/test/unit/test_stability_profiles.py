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
"""The stability state stream is in the simulation profiles only, and optional (no ROS)."""

import pytest
from ranger_xarm_teleop_record import profile


@pytest.mark.parametrize('name', ['gazebo', 'mock'])
def test_the_simulation_profiles_record_the_stability_state_optionally(name):
    loaded = profile.load_profile(name)
    stream = loaded['streams']['stability_state']
    assert stream['topic'] == '/teleop/stability/state'
    assert stream['kind'] == 'stability_state'
    assert stream['type'] == 'ranger_xarm_teleop_interfaces/msg/StabilityState'
    assert stream['required'] is False and stream['rate_hz'] == 50
    assert stream['clock'] == 'robot' and stream.get('record', True) is True
    assert stream['gate'] is None and stream['count'] is False
    assert stream['clock'] == loaded['streams']['base_state']['clock']
    assert 'stability_state' in profile.recorded(loaded)


def test_the_hardware_profile_does_not_record_it():
    streams = profile.load_profile('hardware')['streams']
    assert 'stability_state' not in streams
    assert all(s['kind'] != 'stability_state' for s in streams.values())


def test_the_kind_is_known_and_added_at_the_end_of_the_kinds():
    assert profile.KINDS.index('stability_state') > profile.KINDS.index('other')
    assert profile.KINDS.index('base_state') < profile.KINDS.index('other')
