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
"""Episode control from the left controller: presses, holds, marks and refusals (no ROS)."""

import math

import pytest

from ranger_xarm_teleop_bridge.joy_layout import AXIS_NAMES, BUTTON_NAMES
from ranger_xarm_teleop_record.episode import (
    Buttons, buttons_from_joy, Control, Event, MARK_AXIS, State, X_BUTTON, Y_BUTTON)

DT = 1.0 / 90.0          # headset frame period
NAN = math.nan


def ready():
    c = Control()
    c.set_ready(True)
    c.joy(Buttons(False, False, 0.0), 0.0)
    return c


def run(c, frames, t0=0.0, drop=None, problems=()):
    """Feed (x, y, trigger) frames at the headset rate; ``drop`` skips one frame index."""
    events, t = [], t0
    for i, (x, y, trig) in enumerate(frames):
        t += DT
        if i == drop:
            continue
        for o in c.joy(Buttons(x, y, trig), t, problems):
            events.append(o)
            if o.event in (Event.START, Event.STOP, Event.AUTO_STOP):
                c.done(True)
    return events, t


def press(button, n, before=3, after=3):
    on = (button == 'x', button == 'y', 0.0)
    return [(False, False, 0.0)] * before + [on] * n + [(False, False, 0.0)] * after


def kinds(events):
    return [o.event for o in events]


def test_indices_are_the_left_controllers_buttons():
    assert BUTTON_NAMES[X_BUTTON] == 'left_primary'
    assert BUTTON_NAMES[Y_BUTTON] == 'left_secondary'
    assert AXIS_NAMES[MARK_AXIS] == 'left_trigger'
    for name in ('right_grip', 'right_primary', 'right_secondary', 'right_thumbstick',
                 'left_thumbstick', 'left_grip'):
        assert BUTTON_NAMES.index(name) not in (X_BUTTON, Y_BUTTON)


def test_buttons_from_joy_reads_levels_and_refuses_short_messages():
    buttons = [0] * len(BUTTON_NAMES)
    axes = [0.0] * len(AXIS_NAMES)
    buttons[X_BUTTON] = 1
    axes[MARK_AXIS] = 0.7
    b = buttons_from_joy(buttons, axes)
    assert b == Buttons(True, False, 0.7)
    assert buttons_from_joy(buttons[:2], axes) is None


def test_start_stop_keep():
    c = ready()
    ev, t = run(c, press('x', 5))
    assert kinds(ev) == [Event.START] and c.state == State.RECORDING
    ev, t = run(c, press('x', 5), t)
    assert kinds(ev) == [Event.STOP] and c.state == State.REVIEW
    ev, t = run(c, press('y', 20), t)            # 0.22 s: keep
    assert kinds(ev) == [Event.KEEP] and c.state == State.IDLE


def test_y_held_discards_at_the_hold_time_while_still_held():
    c = ready()
    _, t = run(c, press('x', 4) + press('x', 4))
    assert c.state == State.REVIEW
    c.joy(Buttons(False, True, 0.0), t + DT)
    assert c.discard_armed
    assert c.tick(t + DT + 0.99) == []
    out = c.tick(t + DT + 1.0)
    assert kinds(out) == [Event.DISCARD] and c.state == State.IDLE
    # the release afterwards does nothing more
    assert c.joy(Buttons(False, False, 0.0), t + 1.5) == []


def test_y_release_after_the_hold_time_discards_even_without_a_tick():
    c = ready()
    _, t = run(c, press('x', 4) + press('x', 4))
    c.joy(Buttons(False, True, 0.0), t)
    c.state, c._y_down_at = State.REVIEW, t       # no tick in between
    out = c.joy(Buttons(False, False, 0.0), t + 1.2)
    assert kinds(out) == [Event.DISCARD]


@pytest.mark.parametrize('n', [2, 3, 5, 9])
@pytest.mark.parametrize('button', ['x', 'y'])
def test_one_dropped_joy_message_neither_loses_nor_doubles_a_press(button, n):
    frames = press(button, n)
    for drop in range(len(frames)):
        c = ready()
        if button == 'y':
            _, t0 = run(c, press('x', 4) + press('x', 4))
            assert c.state == State.REVIEW
        else:
            t0 = 0.0
        ev, _ = run(c, frames, t0, drop=drop)
        want = [Event.START] if button == 'x' else [Event.KEEP]
        assert kinds(ev) == want, (drop, kinds(ev))


def test_x_in_review_is_refused_with_the_reason():
    c = ready()
    _, t = run(c, press('x', 4) + press('x', 4))
    ev, _ = run(c, press('x', 4), t)
    assert kinds(ev) == [Event.REFUSED] and 'keep' in ev[0].detail
    assert c.state == State.REVIEW


def test_start_is_refused_while_a_required_stream_is_missing():
    c = ready()
    ev, _ = run(c, press('x', 4), problems=['camera_wrist: missing'])
    assert kinds(ev) == [Event.REFUSED] and 'camera_wrist' in ev[0].detail
    assert c.state == State.IDLE


def test_not_ready_refuses_and_says_why():
    c = Control()
    c.joy(Buttons(False, False, 0.0), 0.0)
    ev, _ = run(c, press('x', 4), problems=['no recorder'])
    assert kinds(ev) == [Event.REFUSED] and 'no recorder' in ev[0].detail


def test_presses_while_the_recorder_is_busy_are_refused():
    c = ready()
    c.joy(Buttons(True, False, 0.0), 1.0)          # START, recorder call not done
    assert c.busy
    c.joy(Buttons(False, False, 0.0), 1.1)
    out = c.joy(Buttons(True, False, 0.0), 1.2)
    assert kinds(out) == [Event.REFUSED] and c.state == State.RECORDING


def test_failed_resume_returns_to_idle_with_a_refusal():
    c = ready()
    c.joy(Buttons(True, False, 0.0), 1.0)
    out = c.done(False, 'timeout')
    assert kinds(out) == [Event.REFUSED] and c.state == State.IDLE


def test_a_button_held_at_start_up_is_not_a_press():
    c = Control()
    c.set_ready(True)
    assert c.joy(Buttons(True, False, 0.0), 0.0) == []
    assert c.joy(Buttons(True, False, 0.0), 0.1) == []
    assert c.state == State.IDLE


def test_y_pressed_while_recording_does_nothing_then_or_after_the_stop():
    c = ready()
    _, t = run(c, press('x', 4))
    c.joy(Buttons(False, True, 0.0), t)            # Y down while recording
    out = c.joy(Buttons(True, True, 0.0), t + 0.1)   # X: stop, Y still held
    assert kinds(out) == [Event.STOP]
    c.done(True)
    assert c.tick(t + 3.0) == []                   # the hold began before REVIEW
    assert c.joy(Buttons(False, False, 0.0), t + 3.1) == []
    assert c.state == State.REVIEW


def test_mark_with_hysteresis_only_while_recording():
    c = ready()
    frames = [(False, False, v) for v in (0.0, 0.7, 0.5, 0.65, 0.2, 0.7, NAN, 0.1)]
    ev, t = run(c, frames)
    assert Event.MARK not in kinds(ev)              # idle: no marks
    _, t = run(c, press('x', 4), t)
    ev, _ = run(c, frames, t)
    assert kinds(ev) == [Event.MARK, Event.MARK]    # 0.7 / (0.5, 0.65 no) / 0.2 re-arms / 0.7


def test_auto_stop_only_while_recording():
    c = ready()
    assert c.auto_stop('disarmed') == []
    _, t = run(c, press('x', 4))
    out = c.auto_stop('disarmed')
    assert kinds(out) == [Event.AUTO_STOP] and out[0].detail == 'disarmed'
    assert c.state == State.REVIEW


def test_ready_changes_never_interrupt_an_episode():
    c = ready()
    run(c, press('x', 4))
    c.set_ready(False)
    assert c.state == State.RECORDING
