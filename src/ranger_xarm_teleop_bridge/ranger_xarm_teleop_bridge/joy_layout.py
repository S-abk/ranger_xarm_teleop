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
Fixed order of the ``sensor_msgs/Joy`` arrays built from a frame (no ROS, INV-6).

Quest ``xr-standard`` gamepad: ``buttons[0]`` trigger, ``[1]`` squeeze (grip),
``[2]`` touchpad placeholder (unused), ``[3]`` thumbstick press, ``[4]`` A or X,
``[5]`` B or Y; ``axes[0], [1]`` touchpad (unused), ``axes[2], [3]`` thumbstick
x and y, raw and not flipped.

Output, left block then right block. A missing hand, a source that is not a
real controller (``ParsedFrame.controller``), or a missing button or axis, gives
zeros.

Buttons (10 ints, 1 when ``pressed``):

=====  ================  =====  =================
Index  Name              Index  Name
=====  ================  =====  =================
0      left_trigger      5      right_trigger
1      left_grip         6      right_grip
2      left_thumbstick   7      right_thumbstick
3      left_primary      8      right_primary
4      left_secondary    9      right_secondary
=====  ================  =====  =================

Axes (8 floats): per hand, trigger and grip are the analog ``value`` of
``buttons[0]`` and ``buttons[1]``; the thumbstick x and y are ``axes[2]`` and
``axes[3]``.

=====  ===================  =====  ====================
Index  Name                 Index  Name
=====  ===================  =====  ====================
0      left_trigger         4      right_trigger
1      left_grip            5      right_grip
2      left_thumbstick_x    6      right_thumbstick_x
3      left_thumbstick_y    7      right_thumbstick_y
=====  ===================  =====  ====================
"""

from typing import Optional, Tuple

from .frame_parse import HandSample, ParsedFrame

XR_BUTTON_TRIGGER = 0
XR_BUTTON_GRIP = 1
XR_BUTTON_THUMBSTICK = 3
XR_BUTTON_PRIMARY = 4
XR_BUTTON_SECONDARY = 5
XR_AXIS_THUMBSTICK_X = 2
XR_AXIS_THUMBSTICK_Y = 3

BUTTON_NAMES = (
    'left_trigger', 'left_grip', 'left_thumbstick', 'left_primary', 'left_secondary',
    'right_trigger', 'right_grip', 'right_thumbstick', 'right_primary', 'right_secondary',
)
AXIS_NAMES = (
    'left_trigger', 'left_grip', 'left_thumbstick_x', 'left_thumbstick_y',
    'right_trigger', 'right_grip', 'right_thumbstick_x', 'right_thumbstick_y',
)

_BUTTON_ORDER = (XR_BUTTON_TRIGGER, XR_BUTTON_GRIP, XR_BUTTON_THUMBSTICK,
                 XR_BUTTON_PRIMARY, XR_BUTTON_SECONDARY)


def _hand_buttons(hand: Optional[HandSample]) -> Tuple[int, ...]:
    result = []
    for index in _BUTTON_ORDER:
        pressed = hand is not None and index < len(hand.buttons) and hand.buttons[index].pressed
        result.append(1 if pressed else 0)
    return tuple(result)


def _hand_axes(hand: Optional[HandSample]) -> Tuple[float, ...]:
    def analog(index: int) -> float:
        if hand is None or index >= len(hand.buttons):
            return 0.0
        return float(hand.buttons[index].value)

    def axis(index: int) -> float:
        if hand is None or index >= len(hand.axes):
            return 0.0
        return float(hand.axes[index])

    return (analog(XR_BUTTON_TRIGGER), analog(XR_BUTTON_GRIP),
            axis(XR_AXIS_THUMBSTICK_X), axis(XR_AXIS_THUMBSTICK_Y))


def build_joy_arrays(frame: ParsedFrame) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    """Return ``(axes, buttons)`` in the fixed layout above; never raises."""
    left, right = frame.controller('left'), frame.controller('right')
    axes = _hand_axes(left) + _hand_axes(right)
    buttons = _hand_buttons(left) + _hand_buttons(right)
    return axes, buttons
