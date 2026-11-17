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
"""The right trigger for the gripper: a missing value is NaN, never 0.0 (released)."""

import json
import math

from parse_helpers import button, hand, valid_frame
from ranger_xarm_teleop_bridge.frame_parse import parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.joy_layout import AXIS_NAMES, build_joy_arrays

TRIGGER = AXIS_NAMES.index('right_trigger')


def axes_of(frame):
    result = parse_text(json.dumps(frame))
    assert isinstance(result, ParsedFrame)
    return build_joy_arrays(result)[0]


def test_trigger_value_passes_through():
    frame = valid_frame()
    frame['controllers']['right'] = hand(buttons=[button(True, 0.8), button(False, 0.0)],
                                         axes=[0.0, 0.0, 0.0, 0.0])
    assert axes_of(frame)[TRIGGER] == 0.8


def test_trigger_without_an_analog_value_is_nan():
    frame = valid_frame()
    frame['controllers']['right'] = hand(buttons=[{'p': True}, button(False, 0.0)],
                                         axes=[0.0, 0.0, 0.0, 0.0])
    assert math.isnan(axes_of(frame)[TRIGGER])


def test_controller_without_buttons_has_a_nan_trigger():
    frame = valid_frame()
    frame['controllers']['right'] = hand(buttons=[], axes=[0.0, 0.0, 0.0, 0.0])
    axes = axes_of(frame)
    assert math.isnan(axes[TRIGGER])
    assert axes[TRIGGER + 1] == 0.0          # the grip value keeps its zero filling


def test_absent_or_hand_tracked_right_hand_has_a_nan_trigger():
    # Released would be 0.0: losing the controller must not open the gripper.
    frame = valid_frame()
    del frame['controllers']['right']
    axes = axes_of(frame)
    assert math.isnan(axes[TRIGGER]) and axes[TRIGGER + 1:] == (0.0,) * 3
    frame = valid_frame()
    frame['controllers']['right'] = {'hand_tracked': True}
    axes = axes_of(frame)
    assert math.isnan(axes[TRIGGER]) and axes[TRIGGER + 1:] == (0.0,) * 3
