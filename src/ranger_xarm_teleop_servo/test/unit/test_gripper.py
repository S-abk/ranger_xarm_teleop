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
"""Gripper teleoperation logic (gripper task INV-1, -2, -3, -5, keep-out; pure)."""

import math

import pytest
from ranger_xarm_teleop_servo.gripper import (build_gripper_params, check_joint_range,
                                              GripperInputs, GripperParams, GripperState,
                                              Hold, input_ok, MissingGripperValue, Request,
                                              Source, step, valid_trigger)

OPEN, CLOSED, SPEED, PERIOD = 0.0, 0.849, 1.0, 0.02
PARAMS = GripperParams(enabled=True, joint='xarm_drive_joint', open_position=OPEN,
                       closed_position=CLOSED, max_speed=SPEED, period=PERIOD)
STEP = SPEED * PERIOD


def inputs(trigger=0.0, now=0.0, armed=True, owner=True, input_ok=True, held=None,
           measured=None, clearance=None, margin=0.025, joints_fresh=True):
    return GripperInputs(now=now, armed=armed, owner=owner, input_ok=input_ok,
                         trigger=trigger, held=held, measured=measured, clearance=clearance,
                         margin=margin, joints_fresh=joints_fresh)


class Run:
    """Feeds cycles at the target period and keeps every output."""

    def __init__(self, params=PARAMS, state=None):
        self.params = params
        self.state = state if state is not None else GripperState()
        self.now = 0.0
        self.outputs = []

    def cycle(self, **kw):
        kw.setdefault('now', self.now)
        self.state, out = step(self.state, inputs(**kw), self.params)
        self.now += PERIOD
        self.outputs.append(out)
        return out

    def cycles(self, n, **kw):
        return [self.cycle(**kw) for _ in range(n)]


def active_run(reference=OPEN, trigger=0.0, **kw):
    run = Run(**kw)
    out = run.cycle(trigger=trigger, held=reference)
    assert out.publish and out.source == Source.HELD
    return run


# ---- parameters ----

def test_disabled_by_default_and_nothing_else_is_read():
    params = build_gripper_params({}, PERIOD)
    assert not params.enabled
    state, out = step(GripperState(), inputs(trigger=1.0), params)
    assert not out.publish and out.hold == Hold.DISABLED and math.isnan(out.command)


def test_enabled_needs_every_safety_value_and_names_them():
    with pytest.raises(MissingGripperValue) as err:
        build_gripper_params({'gripper_enabled': True}, PERIOD)
    for name in ('gripper_open_position', 'gripper_closed_position', 'gripper_max_speed'):
        assert name in str(err.value)


@pytest.mark.parametrize('change', [
    {'gripper_max_speed': 0.0}, {'gripper_max_speed': -1.0}, {'gripper_max_speed': math.inf},
    {'gripper_closed_position': 0.0}, {'gripper_open_position': math.nan},
    {'gripper_close_threshold': 0.2}, {'gripper_open_threshold': 0.0},
    {'gripper_close_threshold': 1.0}, {'gripper_joint': ''},
    {'gripper_reference_timeout_s': -0.1}, {'gripper_max_speed': True},
])
def test_invalid_values_are_refused(change):
    values = {'gripper_enabled': True, 'gripper_open_position': 0.0,
              'gripper_closed_position': 0.849, 'gripper_max_speed': 1.0}
    values.update(change)
    with pytest.raises(ValueError):
        build_gripper_params(values, PERIOD)


def test_valid_values_build():
    params = build_gripper_params(
        {'gripper_enabled': True, 'gripper_open_position': 0.0,
         'gripper_closed_position': 0.849, 'gripper_max_speed': 1.0}, PERIOD)
    assert params.enabled and params.joint == 'xarm_drive_joint'
    assert params.step == pytest.approx(0.02)
    assert (params.open_threshold, params.close_threshold) == (0.25, 0.6)


def test_range_must_lie_within_the_joint_limits():
    check_joint_range(PARAMS, 0.0, 0.85)
    with pytest.raises(ValueError):
        check_joint_range(PARAMS, 0.0, 0.8)
    with pytest.raises(ValueError):
        check_joint_range(PARAMS, 0.1, 0.85)


@pytest.mark.parametrize('value, expected', [
    (0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (None, None), (math.nan, None), (-0.01, None),
    (1.01, None), (math.inf, None), (True, None), ('x', None)])
def test_trigger_validity(value, expected):
    assert valid_trigger(value) == expected


# ---- mapping ----

def test_close_and_open_ramp_at_the_speed_limit():
    run = active_run(OPEN)
    run.cycle(trigger=0.0)
    out = run.cycle(trigger=0.7)                # rising crossing of 0.6
    assert out.request == Request.CLOSE and out.command == pytest.approx(STEP)
    commands = [o.command for o in run.cycles(60, trigger=0.7)]
    steps = [b - a for a, b in zip([out.command] + commands, commands)]
    assert max(steps) <= STEP + 1e-12
    assert commands[-1] == CLOSED
    out = run.cycle(trigger=0.1)                # falling crossing of 0.25
    assert out.request == Request.OPEN and out.command == pytest.approx(CLOSED - STEP)
    run.cycles(60, trigger=0.1)
    assert run.outputs[-1].command == OPEN


def test_hysteresis_band_keeps_the_request():
    run = active_run(OPEN)
    run.cycle(trigger=0.7)
    run.cycles(50, trigger=0.7)
    # A re-grip relaxes the index finger into the band: the squeeze stays.
    for value in (0.5, 0.3, 0.26, 0.4, 0.59, 0.3):
        out = run.cycle(trigger=value)
        assert out.request == Request.CLOSE and out.command == CLOSED
    out = run.cycle(trigger=0.25)
    assert out.request == Request.OPEN


def test_a_press_that_stays_in_the_band_does_nothing():
    run = active_run(OPEN, trigger=0.3)
    for value in (0.3, 0.55, 0.3, 0.4, 0.26, 0.59):
        out = run.cycle(trigger=value)
        assert out.command == OPEN and out.hold == Hold.AWAIT_TRIGGER


# ---- INV-3 and the re-pickup rule ----

def test_arming_never_moves_the_gripper_the_first_command_is_the_held_reference():
    run = Run()
    out = run.cycle(trigger=0.0, held=0.4, measured=0.3)
    assert out.publish and out.command == 0.4 and out.source == Source.HELD
    assert out.hold == Hold.AWAIT_TRIGGER


def test_arm_with_trigger_released_while_the_held_reference_is_closed_stays_closed():
    run = Run()
    for _ in range(100):
        out = run.cycle(trigger=0.0, held=CLOSED)
        assert out.command == CLOSED and out.hold == Hold.AWAIT_TRIGGER
    # Press: a rising crossing asks for CLOSE, already there. Release: now it opens.
    out = run.cycle(trigger=0.9)
    assert out.request == Request.CLOSE and out.command == CLOSED
    out = run.cycle(trigger=0.0)
    assert out.request == Request.OPEN and out.command == pytest.approx(CLOSED - STEP)


def test_arm_with_trigger_held_does_not_close_until_it_is_released_and_pressed_again():
    run = Run()
    run.cycle(trigger=1.0, held=OPEN)
    for _ in range(20):
        assert run.cycle(trigger=1.0).command == OPEN
    run.cycle(trigger=0.0)                      # falling crossing: OPEN, already open
    assert run.outputs[-1].command == OPEN
    out = run.cycle(trigger=1.0)
    assert out.request == Request.CLOSE and out.command > OPEN


def test_intermediate_reference_is_kept_until_a_crossing():
    run = Run()
    run.cycle(trigger=0.4, held=0.33)
    for value in (0.4, 0.5, 0.3):
        assert run.cycle(trigger=value).command == 0.33


def test_no_held_reference_waits_then_uses_measured():
    run = Run()
    out = run.cycle(trigger=0.0, measured=0.2)
    assert not out.publish and out.hold == Hold.NO_REFERENCE and math.isnan(out.command)
    outs = run.cycles(24, trigger=0.0, measured=0.2)
    assert not any(o.publish for o in outs)
    out = run.cycle(trigger=0.0, measured=0.2)  # 0.5 s after the first cycle
    assert out.publish and out.command == 0.2 and out.source == Source.MEASURED


def test_held_reference_arriving_while_waiting_is_used():
    run = Run()
    run.cycles(5, trigger=0.0, measured=0.2)
    out = run.cycle(trigger=0.0, measured=0.2, held=0.7)
    assert out.publish and out.command == 0.7 and out.source == Source.HELD


def test_without_reference_and_measurement_nothing_is_sent():
    run = Run()
    outs = run.cycles(100, trigger=0.0)
    assert not any(o.publish for o in outs)
    assert outs[-1].hold == Hold.NO_REFERENCE


# ---- INV-1: freezes, one per trigger ----

FREEZES = {
    'disarm': {'armed': False},
    'stale input, session or sequence fault': {'input_ok': False},
    'joint states stale': {'joints_fresh': False},
    'missing trigger': {'trigger': None},
    'trigger not finite': {'trigger': math.nan},
    'trigger out of range': {'trigger': 1.5},
    'owner lost': {'owner': False},
}


@pytest.mark.parametrize('name', list(FREEZES))
def test_a_freeze_sends_nothing_and_the_trigger_has_no_effect(name):
    run = active_run(OPEN)
    run.cycle(trigger=0.7)
    run.cycles(10, trigger=0.7)                # closing, part-way
    part = run.outputs[-1].command
    assert OPEN < part < CLOSED
    for value in (0.7, 0.0, 1.0, 0.0):
        kw = dict(FREEZES[name])
        kw.setdefault('trigger', value)
        out = run.cycle(**kw)
        assert not out.publish
        expected_hold = {'disarm': Hold.NOT_ARMED, 'owner lost': Hold.NOT_OWNER,
                         'joint states stale': Hold.JOINTS_STALE}.get(name, Hold.INPUT)
        assert out.hold == expected_hold
        if name == 'owner lost':
            assert math.isnan(out.command)     # someone else may command it now
        else:
            assert out.command == part          # the command in force


@pytest.mark.parametrize('name', [n for n in FREEZES if n != 'owner lost'])
def test_after_a_freeze_the_own_command_is_kept_and_a_new_crossing_is_needed(name):
    run = active_run(OPEN)
    run.cycle(trigger=0.7)
    run.cycles(10, trigger=0.7)
    part = run.outputs[-1].command
    kw = dict(FREEZES[name])
    kw.setdefault('trigger', 0.7)
    run.cycles(3, **kw)
    # Recovery with the trigger still pressed: part-closed, awaiting a crossing.
    out = run.cycle(trigger=0.7, held=0.0)
    assert out.publish and out.command == part and out.source == Source.OWN
    assert out.hold == Hold.AWAIT_TRIGGER and out.request == Request.NONE
    for _ in range(20):
        assert run.cycle(trigger=0.7).command == part
    run.cycle(trigger=0.0)                      # released: opens
    assert run.outputs[-1].command < part
    run.cycle(trigger=0.9)                      # pressed again: closes
    assert run.outputs[-1].request == Request.CLOSE


def test_after_owner_loss_the_reference_is_held_again_not_the_old_own_command():
    run = active_run(OPEN)
    run.cycle(trigger=0.7)
    run.cycles(60, trigger=0.7)
    run.cycles(3, trigger=0.0, owner=False, armed=False)
    out = run.cycle(trigger=0.0, held=0.5)
    assert out.source == Source.HELD and out.command == 0.5


def test_disarm_and_rearm_while_still_owner_keeps_the_own_command():
    run = active_run(OPEN)
    run.cycle(trigger=0.7)
    run.cycles(60, trigger=0.7)
    run.cycles(5, trigger=0.0, armed=False)
    out = run.cycle(trigger=0.0, held=0.1, measured=0.6)
    assert out.source == Source.OWN and out.command == CLOSED


def test_a_crossing_during_a_freeze_is_not_acted_on():
    run = active_run(OPEN)
    run.cycle(trigger=0.0)
    run.cycle(trigger=0.9, input_ok=False)     # pressed during the freeze
    out = run.cycle(trigger=0.9)
    assert out.command == OPEN and out.hold == Hold.AWAIT_TRIGGER


# ---- INV-2: the clutch is not an input of this module ----

def test_the_command_follows_the_trigger_only_whatever_the_clutch_does():
    # The module has no clutch input, and input_ok() takes none: a release changes
    # nothing here (node test).
    assert 'clutch' not in GripperInputs.__dataclass_fields__


# ---- input_ok: what about the input may freeze the gripper ----

@pytest.mark.parametrize('present, age, tripped, expected', [
    (True, 0.0, False, True),
    (True, 0.1, False, True),           # at the timeout: still fresh
    (True, 0.1001, False, False),       # stale by local arrival
    (True, math.inf, False, False),
    (True, math.nan, False, False),
    (False, 0.0, False, False),         # no frame yet
    (True, 0.0, True, False),           # a session or sequence fault since the last cycle
])
def test_input_ok(present, age, tripped, expected):
    assert input_ok(present, age, 0.1, tripped) is expected


def test_input_ok_has_no_pose_or_arm_side_inputs():
    # The trigger is a button: pose-only and arm-side conditions are not inputs (G-4,
    # solver, MoveIt Servo, tool pose), so they cannot freeze the gripper.
    import inspect
    assert list(inspect.signature(input_ok).parameters) == [
        'frame_present', 'input_age', 'input_timeout', 'tripped']


def test_stale_joint_states_freeze_after_the_input_checks():
    run = active_run(OPEN)
    out = run.cycle(trigger=0.0, joints_fresh=False, input_ok=False)
    assert out.hold == Hold.INPUT and not out.publish
    out = run.cycle(trigger=0.0, joints_fresh=False)
    assert out.hold == Hold.JOINTS_STALE and not out.publish
    out = run.cycle(trigger=0.9)
    assert out.publish and out.hold == Hold.AWAIT_TRIGGER     # re-pickup unchanged


# ---- keep-out ----

def clearance_model(limit_at):
    """Clearance falls below the margin when the opening drops below ``limit_at``."""
    return lambda position: 0.010 + (position - limit_at)


def test_opening_toward_the_structure_stops_at_the_margin():
    run = active_run(CLOSED)
    clear = clearance_model(limit_at=0.6)       # opening below 0.615 is within 25 mm
    run.cycle(trigger=0.9, clearance=clear)
    out = run.cycle(trigger=0.0, clearance=clear)
    assert out.request == Request.OPEN
    run.cycles(30, trigger=0.0, clearance=clear)
    last = run.outputs[-1]
    assert last.hold == Hold.KEEP_OUT and last.publish
    assert clear(last.command) >= 0.025 - 1e-9
    assert clear(last.command - STEP) < 0.025


def test_after_a_keep_out_hold_the_ramp_resumes_by_itself_once_the_arm_moved_away():
    run = active_run(CLOSED)
    near = clearance_model(limit_at=0.6)
    run.cycle(trigger=0.9, clearance=near)
    run.cycles(40, trigger=0.0, clearance=near)
    held_at = run.outputs[-1].command
    assert run.outputs[-1].hold == Hold.KEEP_OUT
    # The arm moved away: more clearance at every opening. No new crossing (trigger stays 0).
    far = clearance_model(limit_at=-1.0)
    out = run.cycle(trigger=0.0, clearance=far)
    assert out.hold == Hold.NONE and out.request == Request.OPEN
    assert out.command == pytest.approx(held_at - STEP)
    run.cycles(60, trigger=0.0, clearance=far)
    assert run.outputs[-1].command == OPEN


def test_motion_away_from_the_structure_is_never_held():
    # Inside the margin already; closing increases the clearance.
    run = active_run(0.3)
    clear = clearance_model(limit_at=0.6)
    run.cycle(trigger=0.0, clearance=clear)
    out = run.cycle(trigger=0.9, clearance=clear)
    assert out.hold == Hold.NONE and out.command == pytest.approx(0.3 + STEP)


def test_a_nearest_pair_elsewhere_does_not_hold_the_gripper():
    # Another link is within the margin; the fingers do not change the clearance.
    run = active_run(CLOSED)
    run.cycle(trigger=0.9, clearance=lambda p: 0.012)
    out = run.cycle(trigger=0.0, clearance=lambda p: 0.012)
    assert out.hold == Hold.NONE and out.command < CLOSED


def test_clearance_that_cannot_be_evaluated_holds_the_command():
    run = active_run(CLOSED)
    run.cycle(trigger=0.9, clearance=lambda p: math.nan)
    out = run.cycle(trigger=0.0, clearance=lambda p: math.nan)
    assert out.hold == Hold.KEEP_OUT and out.command == CLOSED


def test_no_keep_out_check_while_the_command_does_not_change():
    calls = []

    def clear(position):
        calls.append(position)
        return 0.0

    run = active_run(CLOSED)
    run.cycles(10, trigger=0.0, clearance=clear)
    run.cycle(trigger=0.9, clearance=clear)     # CLOSE, already closed
    assert calls == []


# ---- the held reference counts only when produced after teleop asked for the arm ----

def test_a_held_reference_stamped_after_the_ownership_request_is_usable():
    from ranger_xarm_teleop_servo.gripper import held_usable
    request, tol = 100_000_000_000, 50_000_000
    assert held_usable(request, request, request + 10_000_000, tol) == (True, '')
    assert held_usable(request + 5_000_000, request, request + 10_000_000, tol)[0]
    # Up to the tolerance ahead of this clock (the controller's clock runs a little ahead).
    assert held_usable(request + 60_000_000, request, request + 10_000_000, tol)[0]


def test_a_held_reference_from_before_the_request_is_refused_however_late_it_arrives():
    from ranger_xarm_teleop_servo.gripper import held_usable
    request, tol = 100_000_000_000, 50_000_000
    usable, why = held_usable(request - 1, request, request + 30_000_000_000, tol)
    assert not usable and why == 'stamped before the ownership request'


def test_without_a_request_or_a_stamp_nothing_is_usable():
    from ranger_xarm_teleop_servo.gripper import held_usable
    assert held_usable(5, None, 10, 0) == (False, 'teleop has not asked for the arm')
    assert held_usable(0, 1, 10, 0) == (False, 'not stamped')


def test_a_clock_that_went_back_refuses_the_held_reference():
    from ranger_xarm_teleop_servo.gripper import held_usable
    tol = 50_000_000
    # A simulation restarted after teleop asked for the arm (request at 300 s, the clock now
    # at 2 s): a new reference is stamped before the request, and the clock is behind it.
    usable, why = held_usable(1_500_000_000, 300_000_000_000, 2_000_000_000, tol)
    assert not usable and why == 'the clock went back since the ownership request'
    # Teleop asked after the restart (request at 1 s, now at 2 s); a reference from before
    # the restart (stamped 299 s) arrives late: ahead of this clock, refused.
    usable, why = held_usable(299_000_000_000, 1_000_000_000, 2_000_000_000, tol)
    assert not usable and why == 'stamped ahead of this clock (the clock went back?)'
