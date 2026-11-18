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
"""The shipped feedback parameter file: valid, and the same as the code's defaults (no ROS)."""

import os

from ranger_xarm_teleop_bridge.episode_feedback import DEFAULT_PATTERNS, Pattern
from ranger_xarm_teleop_bridge.relay_messages import panel_config
import yaml

CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config',
                                      'operator_feedback.yaml'))


def load():
    with open(CONFIG, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def test_patterns_are_valid_and_equal_the_defaults():
    params = load()['operator_feedback']['ros__parameters']
    assert set(params['haptic']) == set(DEFAULT_PATTERNS) == set(params['haptic_priority'])
    for event, flat in params['haptic'].items():
        # ROS parameter arrays hold one type: every number is a float.
        assert all(isinstance(v, float) for v in flat), event
        assert Pattern.from_flat(flat, params['haptic_priority'][event]) == \
            DEFAULT_PATTERNS[event], event


def test_panel_parameters_are_valid():
    params = load()['teleop_input_bridge']['ros__parameters']
    panel_config(params['panel_anchor'], params['panel_offset'], params['panel_width_m'],
                 params['panel_visible_default'])
    assert all(isinstance(v, float) for v in params['panel_offset'])
