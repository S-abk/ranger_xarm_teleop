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

``clip_controller_buttons.jsonl`` / ``clip_controller_buttons_left.jsonl``: nine 0.1 s cuts
(9 frames each, concatenated) of ``recordings/button_clip.jsonl`` (right Touch Plus controller,
12 buttons) and ``recordings/button_clip_left.jsonl`` (left, 13 buttons), one control operated
per cut in the order trigger, grip, thumbstick click, A/X, B/Y, stick full left, right, back,
forward. They fix the Joy layout against the real headset.

``clip_controller_stick_up.jsonl``: 9 frames of ``recordings/stick-up.jsonl``, the right stick
pushed fully up (forward, away from the user) and nothing else: ``axes[3]`` is -1.0 (-0.998), which
settles the sign of stick y (negative up, as in the Gamepad API convention).
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


# One cut per control, in the order the human operated them, for both hands. For each: the
# Joy button that must be 1, the Joy axis that must be large and its sign, and the raw XR index
# pressed. The stick-y sign comes from the lone stick-up recording (see below); in the
# button-by-button clips the back push came before the forward push.
BUTTON_CLIP_ORDER = (
    ('trigger', 'trigger', 'trigger', +1, 0),
    ('grip', 'grip', 'grip', +1, 1),
    ('thumbstick click', 'thumbstick', None, 0, 3),
    ('A / X', 'primary', None, 0, 4),
    ('B / Y', 'secondary', None, 0, 5),
    ('stick left', None, 'thumbstick_x', -1, None),
    ('stick right', None, 'thumbstick_x', +1, None),
    ('stick back (down)', None, 'thumbstick_y', +1, None),
    ('stick forward (up)', None, 'thumbstick_y', -1, None),
)
TOUCH_ONLY_XR_INDICES = (7, 8, 9, 10, 11)
BUTTON_CLIPS = (('clip_controller_buttons', 'right', 12),
                ('clip_controller_buttons_left', 'left', 13))


@pytest.mark.parametrize('name, hand, button_count', BUTTON_CLIPS)
def test_buttons_clip_is_one_real_controller(name, hand, button_count):
    parsed = frames(name)
    assert len(parsed) == 9 * 9
    other = 'left' if hand == 'right' else 'right'
    for f in parsed:
        assert getattr(f, other) is None
        sample = getattr(f, hand)
        assert sample is not None and is_controller(sample)
        assert len(sample.buttons) == button_count and len(sample.axes) == 4
        assert sample.emulated_position is False
        assert f.session_id == 3915310367
        assert sample.axes[0] == 0.0 and sample.axes[1] == 0.0   # touchpad axes unused


@pytest.mark.parametrize('name, hand, button_count', BUTTON_CLIPS)
def test_buttons_clip_confirms_the_joy_layout_in_the_recorded_order(name, hand, button_count):
    parsed = frames(name)
    for cut, (_, button, axis, sign, xr_index) in zip(range(9), BUTTON_CLIP_ORDER):
        for f in parsed[cut * 9:(cut + 1) * 9]:
            axes, buttons = build_joy_arrays(f)
            pressed = {BUTTON_NAMES[i] for i, v in enumerate(buttons) if v}
            assert pressed == ({f'{hand}_{button}'} if button else set())
            large = {AXIS_NAMES[i]: v for i, v in enumerate(axes) if abs(v) > 0.5}
            assert set(large) == ({f'{hand}_{axis}'} if axis else set())
            if axis:
                assert large[f'{hand}_{axis}'] * sign > 0.5
            sample = getattr(f, hand)
            if xr_index is not None:
                assert [i for i, b in enumerate(sample.buttons) if b.pressed] == [xr_index]
            # The other hand's block stays at zero; only this hand's block is used.
            other_buttons = buttons[:5] if hand == 'right' else buttons[5:]
            other_axes = axes[:4] if hand == 'right' else axes[4:]
            assert other_buttons == (0,) * 5 and other_axes == (0.0,) * 4


@pytest.mark.parametrize('name, hand, button_count', BUTTON_CLIPS)
def test_buttons_clip_stick_signs(name, hand, button_count):
    parsed = frames(name)
    x = [getattr(f, hand).axes[2] for f in parsed]
    y = [getattr(f, hand).axes[3] for f in parsed]
    assert min(x[45:54]) < -0.9 and max(x[54:63]) > 0.85     # left negative, right positive
    assert min(y[63:72]) > 0.9 and max(y[72:81]) < -0.9      # back positive, forward negative


def test_stick_up_clip_settles_the_y_sign():
    parsed = frames('clip_controller_stick_up')
    assert len(parsed) == 9
    for f in parsed:
        assert f.left is None and is_controller(f.right) and len(f.right.buttons) == 12
        assert f.right.axes[:3] == (0.0, 0.0, 0.0) and f.right.axes[3] < -0.99
        assert not any(b.pressed for b in f.right.buttons)
        axes, buttons = build_joy_arrays(f)
        assert buttons == (0,) * 10
        assert axes[:7] == (0.0,) * 7 and axes[7] < -0.99
        assert AXIS_NAMES[7] == 'right_thumbstick_y'


@pytest.mark.parametrize('name, hand, button_count', BUTTON_CLIPS)
def test_buttons_clip_indices_7_to_11_are_touch_sensors_that_never_press(name, hand, button_count):
    parsed = frames(name)
    samples = [getattr(f, hand) for f in parsed]
    for index in TOUCH_ONLY_XR_INDICES:
        assert not any(s.buttons[index].pressed for s in samples)
        assert any(s.buttons[index].touched for s in samples)
    for index in (2, 6) + ((12,) if button_count == 13 else ()):   # placeholders: silent
        assert not any(s.buttons[index].pressed or s.buttons[index].touched for s in samples)
    # Whatever the sensors say, they never reach Joy: only the five mapped indices do.
    activity = analyze(clip(name), hand=hand)['gamepad']['buttons']
    assert len(activity) == button_count
    assert [b['pressed_frames'] > 0 for b in activity] == [
        i in (0, 1, 3, 4, 5) for i in range(button_count)]
