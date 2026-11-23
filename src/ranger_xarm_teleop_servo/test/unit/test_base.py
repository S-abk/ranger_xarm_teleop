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
"""Base driving from the headset (G-19): mapping, limits, the stop, the exclusion with the arm."""

import math
import os
import re
from types import SimpleNamespace as NS

import pytest

from ranger_xarm_teleop_servo import base as bs
from ranger_xarm_teleop_servo.base_rest import AT_REST, MOVING, UNKNOWN

MSG = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ranger_xarm_teleop_interfaces',
                   'msg', 'BaseState.msg')
P = bs.Params(linear_cap=0.2, turn_cap=0.4, linear_accel=0.4, turn_accel=0.8)


def joy(grip=False, right_grip=False, left=(0.0, 0.0), right=(0.0, 0.0), left_missing=False,
        right_missing=False):
    axes = [0.0] * 8
    buttons = [0] * 10
    axes[bs.LEFT_TRIGGER_AXIS] = math.nan if left_missing else 0.0
    axes[bs.RIGHT_TRIGGER_AXIS] = math.nan if right_missing else 0.0
    if not left_missing:
        buttons[bs.LEFT_GRIP_BUTTON] = int(grip)
        axes[bs.LEFT_X_AXIS], axes[bs.LEFT_Y_AXIS] = left
    if not right_missing:
        buttons[bs.RIGHT_GRIP_BUTTON] = int(right_grip)
        axes[bs.RIGHT_X_AXIS], axes[bs.RIGHT_Y_AXIS] = right
    return NS(axes=axes, buttons=buttons)


def status(seq, session=1, xr=None):
    return NS(session_id=session, seq=seq, xr_time_ms=1000.0 + seq * 11.0 if xr is None else xr)


def world(**kw):
    base = {'now': 0.0, 'enabled': True, 'simulator': True,
            'teleop': bs.TeleopView(clutch_engaged=False, stopping=False),
            'reset_state': 0, 'reset_running': True, 'rest': AT_REST}
    base.update(kw)
    return bs.World(**base)


class Rig:
    """A Base with a fake clock: one frame and one cycle every 20 ms."""

    def __init__(self, params=P):
        self.b = bs.Base(params)
        self.now = 100.0
        self.seq = 0
        self.session = 1
        self.out = None

    def step(self, frame=True, st=None, **kw):
        w = {k: kw.pop(k) for k in list(kw) if k in bs.World.__dataclass_fields__}
        self.now += 0.02
        if frame:
            self.seq += 1
            self.b.on_sample(bs.sample_of(st or status(self.seq, self.session), joy(**kw),
                                          self.now))
        self.out = self.b.tick(self.now, world(now=self.now, **w), wall=self.now)
        return self.out

    def run(self, n, **kw):
        outs = [self.step(**kw) for _ in range(n)]
        return outs

    def drive(self, **kw):
        """Release, press, and drive a second at full forward."""
        self.step(grip=False)
        self.run(50, grip=True, left=(0.0, -1.0), **kw)
        assert self.b.driving
        return self.out


# ---- the message ----

def test_constants_match_the_message_file():
    with open(MSG, encoding='utf-8') as handle:
        text = handle.read()
    consts = dict(re.findall(r'^uint8 (\w+) = (\d+)', text, re.MULTILINE))
    for i, name in enumerate(bs.STATE_NAMES):
        assert int(consts[f'STATE_{name}']) == i
    for i, name in enumerate(bs.REASON_NAMES):
        assert int(consts[f'REASON_{name}']) == i
    assert int(consts['MODE_TRANSLATE']) == bs.MODE_TRANSLATE
    assert int(consts['MODE_TURN']) == bs.MODE_TURN
    assert int(consts['LIMIT_TOOL_SPEED']) == bs.LIMIT_TOOL_SPEED
    assert int(consts['LIMIT_ACCELERATION']) == bs.LIMIT_ACCELERATION
    assert int(consts['EVENT_AT_REST']) == bs.EVENT_AT_REST
    assert int(consts['EVENT_RELEASED']) == bs.EVENT_RELEASED
    assert set(bs.REASON_TEXT) == set(range(len(bs.REASON_NAMES)))


def test_required_values_have_no_default():
    with pytest.raises(TypeError):
        bs.Params()                                         # noqa: the caps are required
    for name in ('linear_cap', 'turn_cap', 'linear_accel', 'turn_accel'):
        values = {'linear_cap': 0.2, 'turn_cap': 0.4, 'linear_accel': 0.4, 'turn_accel': 0.8}
        for bad in (0.0, -1.0, math.nan, math.inf):
            values[name] = bad
            with pytest.raises(ValueError):
                bs.Params(**values).check()
    with pytest.raises(ValueError):
        bs.Params(0.2, 0.4, 0.4, 0.8, tool_speed_cap=0.0).check()


# ---- mapping ----

def test_stick_up_is_robot_forward_and_right_is_robot_right():
    vx, vy = bs.translation((0.0, -1.0), P)                 # pushed up: y negative
    assert vx == pytest.approx(0.2) and vy == 0.0
    vx, vy = bs.translation((1.0, 0.0), P)                  # right: x positive
    assert vx == 0.0 and vy == pytest.approx(-0.2)          # robot right is -y
    vx, vy = bs.translation((0.0, 1.0), P)
    assert vx == pytest.approx(-0.2) and vy == 0.0


def test_dead_zone_gives_exact_zero_and_the_response_is_quadratic():
    assert bs.translation((0.1, -0.1), P) == (0.0, 0.0)     # |0.14| inside 0.15
    assert bs.shape(0.15, 0.15) == 0.0
    half = bs.shape(0.575, 0.15)                            # halfway out of the dead zone
    assert half == pytest.approx(0.25)
    assert bs.shape(1.4, 0.15) == 1.0


def test_near_an_axis_snaps_to_exactly_that_axis():
    angle = math.radians(8.0)
    vx, vy = bs.translation((-math.sin(angle), -math.cos(angle)), P)
    assert vy == 0.0 and vx > 0.0
    angle = math.radians(12.0)
    vx, vy = bs.translation((-math.sin(angle), -math.cos(angle)), P)
    assert vy > 0.0 and vx > 0.0                            # left of forward: +y


def test_a_diagonal_is_not_faster_than_the_cap():
    vx, vy = bs.translation((0.9, -0.9), P)
    assert math.hypot(vx, vy) == pytest.approx(0.2)
    assert vx > 0.0 and vy < 0.0                            # forward and to the right


def test_right_stick_x_turns_in_place_left_positive():
    assert bs.turn((-1.0, 0.0), P) == pytest.approx(0.4)
    assert bs.turn((1.0, 0.0), P) == pytest.approx(-0.4)
    assert bs.turn((0.1, 0.0), P) == 0.0


def test_a_pull_back_never_turns_the_base():
    """The reset's control (right stick pulled back) dominates its x: no turn."""
    assert bs.turn((0.7, 0.9), P) == 0.0
    assert bs.turn((0.7, 0.7), P) == 0.0
    assert bs.turn((0.9, 0.7), P) != 0.0


def test_left_stick_wins_and_the_turn_is_said_to_be_ignored():
    s = bs.Sample(0, 1, 1, 0.0, False, True, False, left=(0.0, -1.0), right=(-1.0, 0.0))
    command, mode, ignored, _ = bs.desired(s, P)
    assert mode == bs.MODE_TRANSLATE and command[2] == 0.0 and ignored
    s = bs.Sample(0, 1, 1, 0.0, False, True, False, left=(0.0, 0.0), right=(-1.0, 0.0))
    command, mode, ignored, _ = bs.desired(s, P)
    assert mode == bs.MODE_TURN and command[:2] == (0.0, 0.0) and not ignored


def test_one_mode_per_command_throughout_a_switch():
    r = Rig()
    r.drive()
    outs = r.run(60, grip=True, left=(0.0, 0.0), right=(-1.0, 0.0))
    for o in outs:
        vx, vy, wz = o.publish
        assert not ((vx != 0.0 or vy != 0.0) and wz != 0.0)
    assert outs[-1].publish[2] == pytest.approx(0.4)
    assert outs[-1].mode == bs.MODE_TURN


# ---- acceleration ----

def test_the_command_changes_no_faster_than_the_limits():
    r = Rig()
    r.step(grip=False)
    outs = r.run(80, grip=True, left=(0.0, -1.0))
    outs += r.run(60, grip=True, left=(0.0, 0.0))           # back to centre, grip held
    outs += r.run(60, grip=True, right=(1.0, 0.0))
    prev = (0.0, 0.0, 0.0)
    for o in outs:
        c = o.publish
        assert math.hypot(c[0] - prev[0], c[1] - prev[1]) <= 0.4 * 0.02 + 1e-12
        assert abs(c[2] - prev[2]) <= 0.8 * 0.02 + 1e-12
        prev = c
    assert outs[79].publish[0] == pytest.approx(0.2)
    assert outs[79].active_limit == bs.LIMIT_CAP


def test_centring_the_stick_with_the_grip_held_ramps_down_and_keeps_publishing():
    r = Rig()
    r.drive()
    outs = r.run(40, grip=True, left=(0.0, 0.0))
    assert all(o.publish is not None and o.driving for o in outs)
    assert outs[0].active_limit == bs.LIMIT_ACCELERATION
    assert outs[0].publish[0] == pytest.approx(0.2 - 0.008)
    assert outs[-1].publish == (0.0, 0.0, 0.0)


# ---- the stop: one zero at once, then nothing (INV-1), one test per trigger ----

def stopped_by(r, expect, **kw):
    first = r.step(**kw)
    assert first.publish == (0.0, 0.0, 0.0), 'zero at once'
    assert first.reason == expect and not first.driving
    rest = r.run(10, **kw)
    assert all(o.publish is None for o in rest), 'then publishing stops'
    return first


def test_release_sends_zero_at_once_and_then_stops_publishing():
    r = Rig()
    r.drive(rest=MOVING)
    out = stopped_by(r, bs.RELEASED, grip=False, rest=MOVING)
    assert out.state == bs.STOPPING
    assert r.b.last_event == bs.EVENT_RELEASED


def test_stale_input_stops_the_base():
    r = Rig()
    r.drive(rest=MOVING)
    r.run(5, frame=False, rest=MOVING)                      # 100 ms: still fresh (G-3: older)
    assert r.b.driving
    stopped_by(r, bs.INPUT_STALE, frame=False, rest=MOVING)
    assert r.b.last_event == bs.EVENT_STOPPED


def test_a_missing_left_controller_stops_the_base():
    r = Rig()
    r.drive()
    stopped_by(r, bs.CONTROLLER_MISSING, left_missing=True)


def test_a_nan_stick_counts_as_a_missing_controller():
    j = joy(grip=True)
    j.axes[bs.LEFT_X_AXIS] = math.nan
    assert bs.left_controller_missing(j)
    assert bs.left_controller_missing(None)


def test_a_session_change_stops_the_base():
    r = Rig()
    r.drive()
    r.session = 2
    stopped_by(r, bs.SESSION_CHANGED, grip=True, left=(0.0, -1.0))


def test_a_sequence_going_backwards_stops_the_base():
    r = Rig()
    r.drive()
    r.seq -= 5
    stopped_by(r, bs.SEQ_BACKWARDS, grip=True, left=(0.0, -1.0))


@pytest.mark.parametrize('change, reason', [
    ({'teleop': bs.TeleopView(clutch_engaged=True, stopping=False)}, bs.CLUTCH_ENGAGED),
    ({'teleop': bs.TeleopView(clutch_engaged=False, stopping=True)}, bs.CLUTCH_ENGAGED),
    ({'teleop': None}, bs.TELEOP_UNKNOWN),
    ({'reset_state': 4}, bs.RESET_ACTIVE),
    ({'reset_state': None}, bs.RESET_UNKNOWN),
    ({'rest': UNKNOWN}, bs.ODOM_MISSING),
    ({'other_publishers': ('/someone',)}, bs.OTHER_PUBLISHER),
    ({'enabled': False}, bs.R_OFF),
])
def test_each_world_condition_stops_a_drive(change, reason):
    r = Rig()
    r.drive()
    out = stopped_by(r, reason, grip=True, left=(0.0, -1.0), **change)
    assert out.state == bs.STOPPED


def test_the_node_shutdown_sends_zero_once():
    r = Rig()
    r.drive()
    out = r.b.stop(bs.SHUTDOWN)
    assert out.publish == (0.0, 0.0, 0.0) and r.b.reason == bs.SHUTDOWN
    assert r.b.stop(bs.SHUTDOWN).publish is None


def test_shutdown_while_idle_sends_nothing():
    r = Rig()
    r.step(grip=False)
    assert r.b.stop(bs.SHUTDOWN).publish is None


def test_the_other_publisher_is_named():
    r = Rig()
    r.drive()
    r.step(grip=True, other_publishers=('/teleop_twist',))
    assert '/teleop_twist' in r.b.reason_text()


def test_pose_only_conditions_do_not_exist_for_the_base():
    """The base's sample has no pose: out of view, emulated or absent poses cannot stop it."""
    assert 'pose' not in ' '.join(bs.Sample.__dataclass_fields__)


# ---- starting: a press, refusals, the exclusion with the arm ----

def test_a_held_grip_never_starts_by_itself():
    r = Rig()
    outs = r.run(20, grip=True, left=(0.0, -1.0))           # held from the first frame
    assert all(o.publish is None for o in outs) and not r.b.driving
    r.step(grip=False)
    r.step(grip=True)
    assert r.b.driving


@pytest.mark.parametrize('change, reason', [
    ({'teleop': bs.TeleopView(clutch_engaged=True, stopping=False)}, bs.CLUTCH_ENGAGED),
    ({'teleop': bs.TeleopView(clutch_engaged=False, stopping=True)}, bs.CLUTCH_ENGAGED),
    ({'teleop': None}, bs.TELEOP_UNKNOWN),
    ({'reset_state': 1}, bs.RESET_ACTIVE),
    ({'reset_state': None}, bs.RESET_UNKNOWN),
    ({'rest': UNKNOWN}, bs.ODOM_MISSING),
    ({'other_publishers': ('/x',)}, bs.OTHER_PUBLISHER),
])
def test_each_refusal_says_why_and_needs_a_new_press(change, reason):
    r = Rig()
    r.step(grip=False, **change)
    out = r.step(grip=True, **change)
    assert out.publish is None and out.state == bs.REFUSED and out.reason == reason
    assert r.b.last_event == bs.EVENT_REFUSED
    r.run(5, grip=True)                                     # the condition clears, grip held
    assert not r.b.driving
    r.step(grip=False)
    r.step(grip=True)
    assert r.b.driving


def test_off_by_configuration_does_nothing():
    r = Rig()
    outs = [r.step(grip=g, enabled=False) for g in (False, True, True, False, True)]
    assert all(o.publish is None and o.state == bs.OFF and o.reason == bs.R_OFF for o in outs)


def test_not_a_simulator_does_nothing():
    r = Rig()
    outs = [r.step(grip=g, simulator=False) for g in (False, True, True, False, True)]
    assert all(o.publish is None and o.state == bs.OFF and o.reason == bs.NOT_SIMULATION
               for o in outs)


def test_no_reset_node_needs_no_reset_state():
    r = Rig()
    r.step(grip=False, reset_state=None, reset_running=False)
    r.step(grip=True, reset_state=None, reset_running=False)
    assert r.b.driving


def test_driving_needs_no_armed_teleop():
    """The base needs the arm's fresh state with the clutch released, not an armed teleop."""
    r = Rig()
    r.step(grip=False)
    r.step(grip=True)                                       # TeleopView has no 'armed' at all
    assert r.b.driving
    assert 'armed' not in bs.TeleopView.__dataclass_fields__


def test_right_grip_held_in_the_starting_frame_refuses_the_drive():
    r = Rig()
    r.step(grip=False, right_grip=True)
    out = r.step(grip=True, right_grip=True)
    assert out.state == bs.REFUSED and out.reason == bs.RIGHT_GRIP_HELD


def test_both_grips_first_held_in_one_frame_neither_drives():
    r = Rig()
    r.step(grip=False)
    out = r.step(grip=True, right_grip=True, left=(0.0, -1.0))
    assert out.publish is None and out.reason == bs.RIGHT_GRIP_HELD


def test_first_holder_wins_a_right_grip_during_a_drive_does_not_stop_it():
    r = Rig()
    r.drive()
    outs = r.run(10, grip=True, right_grip=True, left=(0.0, -1.0))
    assert all(o.driving and o.publish[0] > 0.0 for o in outs)


def test_a_fault_while_the_grip_is_held_idle_needs_a_new_press():
    r = Rig()
    r.step(grip=False)
    r.step(grip=True, teleop=None)                          # refused
    r.session = 2
    r.run(3, grip=True)
    assert not r.b.driving


# ---- rest ----

def test_at_rest_event_after_a_drive_and_stopping_becomes_idle():
    r = Rig()
    r.drive(rest=MOVING)
    r.step(grip=False, rest=MOVING)
    assert r.b.state == bs.STOPPING
    r.run(3, grip=False, rest=MOVING)
    assert r.b.last_event == bs.EVENT_RELEASED
    r.step(grip=False, rest=AT_REST)
    assert r.b.last_event == bs.EVENT_AT_REST and r.b.state == bs.IDLE
    count = r.b.event_count
    r.run(5, grip=False, rest=AT_REST)
    assert r.b.event_count == count                         # once


def test_a_fault_stop_stays_shown_after_rest():
    r = Rig()
    r.drive(rest=MOVING)
    r.step(frame=False, rest=MOVING)
    r.run(10, frame=False, rest=MOVING)
    assert r.b.state == bs.STOPPED and r.b.reason == bs.INPUT_STALE
    r.step(grip=False, rest=AT_REST)
    assert r.b.last_event == bs.EVENT_AT_REST and r.b.state == bs.STOPPED


def test_starting_while_still_coasting_is_allowed():
    r = Rig()
    r.drive(rest=MOVING)
    r.step(grip=False, rest=MOVING)
    r.step(grip=True, rest=MOVING)
    assert r.b.driving


# ---- the tool's speed in the room ----

def test_tool_speed_of_a_turn_is_rate_times_reach():
    assert bs.tool_speed((0.0, 0.0, 0.4), (0.7, 0.0)) == pytest.approx(0.28)
    assert bs.tool_speed((0.2, 0.0, 0.0), (0.7, 0.3)) == pytest.approx(0.2)


def test_tool_speed_cap_scales_the_whole_command_keeping_its_direction():
    p = bs.Params(0.2, 0.4, 0.4, 0.8, tool_speed_cap=0.1)
    r = Rig(p)
    r.step(grip=False, tool_xy=(0.7, 0.0))
    r.run(80, grip=True, right=(-1.0, 0.0), tool_xy=(0.7, 0.0))
    out = r.out
    assert out.active_limit == bs.LIMIT_TOOL_SPEED
    assert out.publish[2] == pytest.approx(0.1 / 0.7)
    assert out.tool_scale == pytest.approx(0.1 / 0.28)
    r2 = Rig(p)
    r2.step(grip=False, tool_xy=(0.7, 0.0))
    r2.run(80, grip=True, left=(0.9, -0.9), tool_xy=(0.7, 0.0))
    vx, vy, _ = r2.out.publish
    assert math.hypot(vx, vy) == pytest.approx(0.1)
    assert vx == pytest.approx(-vy)                          # the diagonal kept


def test_without_a_cap_nothing_is_scaled_and_the_posture_is_not_needed():
    r = Rig()
    r.drive()
    assert math.isnan(r.out.tool_scale) and r.out.active_limit != bs.LIMIT_TOOL_SPEED


def test_tool_speed_cap_needs_the_posture():
    p = bs.Params(0.2, 0.4, 0.4, 0.8, tool_speed_cap=0.1)
    r = Rig(p)
    r.step(grip=False)
    out = r.step(grip=True)
    assert out.reason == bs.TOOL_POSE_UNKNOWN
    r.step(grip=False, tool_xy=(0.3, 0.0))
    r.step(grip=True, tool_xy=(0.3, 0.0))
    assert r.b.driving
    stopped_by(r, bs.TOOL_POSE_UNKNOWN, grip=True)
