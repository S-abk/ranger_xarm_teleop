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
"""Fixed Joy layout: names, index mapping and zero filling."""

import json

from parse_helpers import button, hand, valid_frame
from ranger_xarm_teleop_bridge import joy_layout
from ranger_xarm_teleop_bridge.frame_parse import parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.joy_layout import build_joy_arrays


def parsed(frame):
    result = parse_text(json.dumps(frame))
    assert isinstance(result, ParsedFrame)
    return result


def test_layout_lengths_and_names():
    assert len(joy_layout.BUTTON_NAMES) == 10
    assert len(joy_layout.AXIS_NAMES) == 8
    assert joy_layout.BUTTON_NAMES[:5] == (
        'left_trigger', 'left_grip', 'left_thumbstick', 'left_primary', 'left_secondary')
    assert joy_layout.BUTTON_NAMES[5:] == tuple(
        n.replace('left', 'right') for n in joy_layout.BUTTON_NAMES[:5])
    assert joy_layout.AXIS_NAMES == (
        'left_trigger', 'left_grip', 'left_thumbstick_x', 'left_thumbstick_y',
        'right_trigger', 'right_grip', 'right_thumbstick_x', 'right_thumbstick_y')


def test_xr_index_constants():
    assert (joy_layout.XR_BUTTON_TRIGGER, joy_layout.XR_BUTTON_GRIP,
            joy_layout.XR_BUTTON_THUMBSTICK, joy_layout.XR_BUTTON_PRIMARY,
            joy_layout.XR_BUTTON_SECONDARY) == (0, 1, 3, 4, 5)
    assert (joy_layout.XR_AXIS_THUMBSTICK_X, joy_layout.XR_AXIS_THUMBSTICK_Y) == (2, 3)


def test_output_lengths_and_types():
    axes, buttons = build_joy_arrays(parsed(valid_frame()))
    assert len(axes) == 8 and len(buttons) == 10
    assert all(type(a) is float for a in axes)
    assert all(type(b) is int for b in buttons)
    assert set(buttons) == {0}


def test_right_hand_mapping():
    frame = valid_frame()
    right = frame['controllers']['right']
    right['buttons'][0] = button(True, 0.75)
    right['buttons'][1] = button(True, 0.25)
    right['buttons'][2] = button(True, 1.0)   # touchpad placeholder: unused
    right['buttons'][3] = button(True, 1.0)
    right['buttons'][5] = button(True, 1.0)
    right['axes'] = [0.9, 0.8, 0.5, -0.5]
    axes, buttons = build_joy_arrays(parsed(frame))
    assert axes == (0.0, 0.0, 0.0, 0.0, 0.75, 0.25, 0.5, -0.5)
    names = dict(zip(joy_layout.BUTTON_NAMES, buttons))
    assert names == {
        'left_trigger': 0, 'left_grip': 0, 'left_thumbstick': 0, 'left_primary': 0,
        'left_secondary': 0, 'right_trigger': 1, 'right_grip': 1, 'right_thumbstick': 1,
        'right_primary': 0, 'right_secondary': 1}


def test_left_hand_mapping_and_stick_not_flipped():
    frame = valid_frame()
    left = frame['controllers']['left']
    left['buttons'][1] = button(True, 1.0)
    left['buttons'][4] = button(True, 1.0)
    left['axes'] = [0.0, 0.0, -0.3, 0.7]
    axes, buttons = build_joy_arrays(parsed(frame))
    assert axes == (0.0, 1.0, -0.3, 0.7, 0.0, 0.0, 0.0, 0.0)
    assert buttons == (0, 1, 0, 1, 0, 0, 0, 0, 0, 0)


def test_analog_value_is_independent_of_pressed():
    frame = valid_frame()
    frame['controllers']['right']['buttons'][0] = {'p': False, 'v': 0.4}
    axes, buttons = build_joy_arrays(parsed(frame))
    assert axes[4] == 0.4 and buttons[5] == 0


def test_missing_hand_gives_zero_block():
    frame = valid_frame()
    frame['controllers']['left']['buttons'][0] = button(True, 1.0)
    frame['controllers']['right']['buttons'][0] = button(True, 1.0)
    frame['controllers']['right']['axes'] = [0.0, 0.0, 0.5, 0.5]
    del frame['controllers']['right']
    axes, buttons = build_joy_arrays(parsed(frame))
    assert axes == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert buttons == (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame['controllers'] = {}
    assert build_joy_arrays(parsed(frame)) == ((0.0,) * 8, (0,) * 10)


def test_hand_tracked_gives_zero_block():
    frame = valid_frame()
    frame['controllers']['left'] = {'hand_tracked': True}
    axes, buttons = build_joy_arrays(parsed(frame))
    assert axes[:4] == (0.0,) * 4 and buttons[:5] == (0,) * 5


def test_short_arrays_give_zeros():
    frame = valid_frame()
    frame['controllers']['right'] = hand(
        buttons=[button(True, 1.0), button(True, 1.0)], axes=[0.1, 0.2, 0.3])
    axes, buttons = build_joy_arrays(parsed(frame))
    assert axes[4:] == (1.0, 1.0, 0.3, 0.0)
    assert buttons[5:] == (1, 1, 0, 0, 0)
    frame['controllers']['right'] = hand(buttons=[], axes=[])
    axes, buttons = build_joy_arrays(parsed(frame))
    assert axes[4:] == (0.0,) * 4 and buttons[5:] == (0,) * 5
