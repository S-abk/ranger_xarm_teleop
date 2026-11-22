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
"""The packaged hardware deployment profile validates and says what it is (no ROS)."""

import os

from ranger_xarm_teleop_record import profile


def test_the_hardware_profile_loads_and_validates():
    loaded = profile.load_profile('hardware')
    assert loaded['name'] == 'hardware'
    assert loaded['simulation'] is False
    assert loaded['use_sim_time'] is False
    assert loaded['low_rtf'] is None
    assert loaded['data_dir'] == '~/teleop_data'
    assert os.path.isfile(profile.resolve('hardware'))


def test_the_hardware_profile_validates_from_its_file_by_path():
    path = profile.resolve('hardware')
    assert profile.validate(profile.load_yaml(path))['name'] == 'hardware'


def test_the_hardware_profile_has_the_real_joint_state_and_camera_settings():
    streams = profile.load_profile('hardware')['streams']
    assert streams['joint_states']['rate_hz'] == 150
    assert streams['joint_states']['required_fields'] == []
    for name in ('camera_gantry', 'camera_wrist', 'depth_gantry', 'depth_wrist'):
        assert streams[name]['required'] is False
        assert streams[name]['rate_hz'] == 30
        assert (streams[name]['image']['width'], streams[name]['image']['height']) == (640, 480)
    assert streams['camera_gantry']['image']['encoding'] == 'rgb8'
    assert streams['depth_wrist']['image']['encoding'] == '16UC1'


def test_the_hardware_profile_keeps_the_mock_frames_and_core_streams():
    hardware = profile.load_profile('hardware')
    mock = profile.load_profile('mock')
    assert hardware['frames'] == mock['frames']
    for name in profile.CORE:
        assert hardware['streams'][name]['topic'] == mock['streams'][name]['topic']
