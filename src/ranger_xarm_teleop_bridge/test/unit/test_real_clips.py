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
Clips of real Quest 3 recordings (cut verbatim with ``tools/clip_recording.py``).

``clip_hands_head_turn.jsonl``: first 0.5 s of ``recordings/head_turn_a.jsonl``. The Touch
controllers were put down, so the headset presents both hands as controller-like sources
(10 buttons, 0 axes, no ``hand_tracked`` flag from the then-current client). The bridge
must publish them as not present and hand tracked.

``clip_controller_occlusion.jsonl``: 4.4 .. 5.0 s of ``recordings/occlusion.jsonl``, the
real right Touch controller (12 buttons, 4 axes) going from tracked to emulated at 4.62 s.

``clip_controller_buttons.jsonl``: nine 0.1 s cuts (9 frames each, concatenated) of
``recordings/button_clip.jsonl``, right Touch Plus controller, one control operated per cut in
the order trigger, grip, thumbstick click, A, B, stick full left, right, up, down. It fixes the
Joy layout and the stick signs against the real headset.
"""

import os

from parse_helpers import FIXTURE_DIR
import pytest
from ranger_xarm_teleop_bridge.analysis import analyze, format_report
from ranger_xarm_teleop_bridge.frame_parse import is_controller, parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.joy_layout import AXIS_NAMES, build_joy_arrays, BUTTON_NAMES
from ranger_xarm_teleop_bridge.jsonl_io import read_records


def clip(name):
    return list(read_records(os.path.join(FIXTURE_DIR, name + '.jsonl')))


def frames(name):
    out = [parse_text(r.text()) for r in clip(name)]
    return [f for f in out if isinstance(f, ParsedFrame)]


def test_hands_clip_is_what_the_headset_sent():
    parsed = frames('clip_hands_head_turn')
    assert len(parsed) == 45
    for f in parsed:
        for side in ('left', 'right'):
            raw = getattr(f, side)
            assert raw is not None
            assert len(raw.buttons) == 10 and len(raw.axes) == 0 and raw.profiles == ()
            assert getattr(f, f'{side}_hand_tracked') is False   # client did not flag it
            source = getattr(f, f'{side}_source')
            assert (source.button_count, source.axis_count) == (10, 0)
            assert source.client_hand_tracked is False


def test_hands_clip_is_not_present_for_the_bridge():
    for f in frames('clip_hands_head_turn'):
        for side in ('left', 'right'):
            assert not is_controller(getattr(f, side))
            assert f.controller(side) is None
            assert f.is_hand_tracked(side) is True
        axes, buttons = build_joy_arrays(f)
        assert axes == (0.0,) * 8 and buttons == (0,) * 10


def test_hands_clip_contains_a_pinch_that_never_reaches_joy():
    pinched = [f for f in frames('clip_hands_head_turn') if f.right.buttons[0].pressed]
    assert len(pinched) >= 1
    for f in pinched:
        assert build_joy_arrays(f)[1] == (0,) * 10


def test_controller_clip_is_a_real_controller():
    parsed = frames('clip_controller_occlusion')
    assert len(parsed) == 54
    for f in parsed:
        assert f.left is None and f.left_source is None and not f.is_hand_tracked('left')
        assert f.right is not None and is_controller(f.right)
        assert f.controller('right') is f.right
        assert f.is_hand_tracked('right') is False
        assert len(f.right.buttons) == 12 and len(f.right.axes) == 4
        assert f.session_id == 128324305 and f.reset_count == 2
    emulated = [f.right.emulated_position for f in parsed]
    assert emulated[0] is False and emulated[-1] is True
    assert emulated == sorted(emulated)   # one transition, tracked -> emulated


def test_analysis_warns_on_hands_and_not_on_the_controller():
    hands = analyze(clip('clip_hands_head_turn'), hand='right')
    assert hands['bridge_not_present_frames'] == 45
    assert [s['controller'] for s in hands['sources']['right']] == [False]
    assert hands['sources']['right'][0]['buttons'] == 10
    assert len(hands['sources']['warnings']) == 2
    assert hands['spread_in_headset'] is not None
    assert hands['gamepad']['buttons'][0]['pressed_frames'] >= 1
    text = format_report(hands)
    assert 'HAND-LIKE' in text and 'WARNING right:' in text

    ctrl = analyze(clip('clip_controller_occlusion'), hand='right')
    assert ctrl['bridge_not_present_frames'] == 0
    assert ctrl['sources']['warnings'] == []
    assert [s['controller'] for s in ctrl['sources']['right']] == [True]
    assert ctrl['sources']['left'] == []
    assert len(ctrl['emulated_intervals']) == 1
    assert ctrl['emulated_intervals'][0]['start'] == pytest.approx(0.22, abs=0.03)
    assert len(ctrl['gamepad']['buttons']) == 12 and len(ctrl['gamepad']['axes']) == 4
    assert 'WARNING' not in format_report(ctrl)


# One cut per control, in the order the human operated them (the stick labels come from
# that stated order; the y sign is provisional until a lone stick-up push is recorded).
# For each: the Joy button that must be 1, the Joy axis that must be large and its sign,
# and the raw XR index pressed.
BUTTON_CLIP_ORDER = (
    ('trigger', 'right_trigger', 'right_trigger', +1, 0),
    ('grip', 'right_grip', 'right_grip', +1, 1),
    ('thumbstick click', 'right_thumbstick', None, 0, 3),
    ('A', 'right_primary', None, 0, 4),
    ('B', 'right_secondary', None, 0, 5),
    ('stick left', None, 'right_thumbstick_x', -1, None),
    ('stick right', None, 'right_thumbstick_x', +1, None),
    ('stick up', None, 'right_thumbstick_y', +1, None),
    ('stick down', None, 'right_thumbstick_y', -1, None),
)
TOUCH_ONLY_XR_INDICES = (7, 8, 9, 10, 11)


def test_buttons_clip_is_the_real_right_controller():
    parsed = frames('clip_controller_buttons')
    assert len(parsed) == 9 * 9
    for f in parsed:
        assert f.left is None and f.right is not None and is_controller(f.right)
        assert len(f.right.buttons) == 12 and len(f.right.axes) == 4
        assert f.right.emulated_position is False
        assert f.session_id == 3915310367 and f.reset_count == 4
        assert f.right.axes[0] == 0.0 and f.right.axes[1] == 0.0   # touchpad axes unused


def test_buttons_clip_confirms_the_joy_layout_in_the_recorded_order():
    parsed = frames('clip_controller_buttons')
    for cut, (_, button, axis, sign, xr_index) in zip(range(9), BUTTON_CLIP_ORDER):
        for f in parsed[cut * 9:(cut + 1) * 9]:
            axes, buttons = build_joy_arrays(f)
            pressed = {BUTTON_NAMES[i] for i, v in enumerate(buttons) if v}
            assert pressed == ({button} if button else set())
            large = {AXIS_NAMES[i]: v for i, v in enumerate(axes) if abs(v) > 0.5}
            assert set(large) == ({axis} if axis else set())
            if axis:
                assert large[axis] * sign > 0.5
            if xr_index is not None:
                assert [i for i, b in enumerate(f.right.buttons) if b.pressed] == [xr_index]
            assert buttons[:5] == (0,) * 5 and axes[:4] == (0.0,) * 4


def test_buttons_clip_stick_signs():
    parsed = frames('clip_controller_buttons')
    x = [f.right.axes[2] for f in parsed]
    y = [f.right.axes[3] for f in parsed]
    assert min(x[45:54]) < -0.9 and max(x[54:63]) > 0.85     # left negative, right positive
    assert min(y[63:72]) > 0.9 and max(y[72:81]) < -0.9      # up positive, down negative


def test_buttons_clip_indices_7_to_11_are_touch_sensors_that_never_press():
    parsed = frames('clip_controller_buttons')
    for index in TOUCH_ONLY_XR_INDICES:
        assert not any(f.right.buttons[index].pressed for f in parsed)
        assert any(f.right.buttons[index].touched for f in parsed)
    for index in (2, 6):   # touchpad placeholder and an unused slot: silent
        assert not any(f.right.buttons[index].pressed or f.right.buttons[index].touched
                       for f in parsed)
    # Whatever the sensors say, they never reach Joy: only the five mapped indices do.
    activity = analyze(clip('clip_controller_buttons'), hand='right')['gamepad']['buttons']
    assert [b['pressed_frames'] > 0 for b in activity] == [
        i in (0, 1, 3, 4, 5) for i in range(12)]
