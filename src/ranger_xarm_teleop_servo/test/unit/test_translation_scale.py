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
"""Fine and coarse translation scale: mapping, the switch, arming, and the checks around it."""

from dataclasses import replace
import math

from arm_urdf import arm_urdf, READY
from helpers import frame, HAND, params, ready_state, run, same_pose, TOOL_START, VALUES
import numpy as np
import pytest
from ranger_xarm_teleop_servo import kinematics as kin
from ranger_xarm_teleop_servo import solver as sv
from ranger_xarm_teleop_servo.calibration import Calibration
from ranger_xarm_teleop_servo.gate import Env, GateState, Hold, on_frame, tick
from ranger_xarm_teleop_servo.mapping import apply_pivot, clutch_relative, rescale
from ranger_xarm_teleop_servo.params import build_params, MissingValue, REQUIRED
from ranger_xarm_teleop_servo.rotations import (Pose, q_between, q_from_axis_angle, q_from_yaw,
                                                q_mul, q_rotate, v_norm, v_scale, v_sub)
from ranger_xarm_teleop_servo.teleop_model import run as model_run

RATE = 90.0
# Frames come at 90 Hz and cycles at 50 Hz: a cycle takes in one or two frames of motion.
FRAMES_PER_CYCLE = 2
PIVOT = (0.0, -0.04, 0.08)
HEADING = 0.6
# The grip held rotated 70 deg about a tilted axis, so the pivot offset points somewhere
# other than along a room axis.
GRIP = q_mul(q_from_axis_angle((0.3, -0.5, 0.8), math.radians(70.0)), HAND.q)
COARSE = {'translation_scale_coarse': 2.0}


def _close(a, b, tol=1e-9):
    return v_norm(v_sub(a, b)) <= tol


def _clicks(n, motion, clicks=(), grip=lambda k: True):
    """Frames with the thumbstick held during the frame ranges in ``clicks``."""
    return [frame(k, controller=motion(k), grip=grip(k),
                  scale_toggle=any(a <= k < b for a, b in clicks)) for k in range(n)]


def _state(heading=HEADING):
    return ready_state(heading=heading)


def _targets(steps):
    return [(s.t, s.output.target) for s in steps if s.output.target is not None]


def _at(steps, t):
    """Return the step closest to time ``t``."""
    return min(steps, key=lambda s: abs(s.t - t))


# ---- the mapping ----

def test_scale_one_is_the_unscaled_mapping():
    clutch = Pose((0.4, -0.1, 1.2), q_from_axis_angle((0.2, 1, 0.3), 0.5))
    now = Pose((0.47, -0.05, 1.31), q_mul(q_from_axis_angle((1, -0.4, 0.2), 0.7), clutch.q))
    tool0 = Pose((0.3, 0.05, 0.25), q_from_axis_angle((0.3, 0.1, 1), 2.4))
    assert clutch_relative(now, clutch, HEADING, tool0) == clutch_relative(
        now, clutch, HEADING, tool0, 1.0)


def test_scale_multiplies_translation_only():
    clutch = Pose((0.4, -0.1, 1.2), q_from_axis_angle((0.2, 1, 0.3), 0.5))
    now = Pose((0.47, -0.05, 1.31), q_mul(q_from_axis_angle((1, -0.4, 0.2), 0.7), clutch.q))
    tool0 = Pose((0.3, 0.05, 0.25), q_from_axis_angle((0.3, 0.1, 1), 2.4))
    one = clutch_relative(now, clutch, HEADING, tool0, 1.0)
    two = clutch_relative(now, clutch, HEADING, tool0, 2.0)
    assert _close(v_sub(two.p, tool0.p), v_scale(v_sub(one.p, tool0.p), 2.0), 1e-12)
    assert q_between(one.q, two.q) < 1e-9


@pytest.mark.parametrize('old, new', [(1.0, 2.0), (2.0, 1.0), (1.0, 1.5), (0.5, 2.0)])
def test_rescale_keeps_the_target_and_scales_only_later_motion(old, new):
    """Non-zero pivot, a rotated grip and a heading: the switch moves nothing."""
    clutch = apply_pivot(Pose((0.4, -0.1, 1.2), GRIP), PIVOT)
    now = apply_pivot(Pose((0.47, -0.05, 1.31), q_mul(q_from_axis_angle((1, -0.4, 0.2), 0.7),
                                                      GRIP)), PIVOT)
    anchor = Pose((0.3, 0.05, 0.25), q_from_axis_angle((0.3, 0.1, 1), 2.4))
    before = clutch_relative(now, clutch, HEADING, anchor, old)
    moved_anchor = rescale(anchor, now, clutch, HEADING, old, new)
    after = clutch_relative(now, clutch, HEADING, moved_anchor, new)
    assert _close(after.p, before.p, 1e-12) and q_between(after.q, before.q) < 1e-9
    # Further hand motion, wrist turn included, is taken at the new scale; rotation unscaled.
    later_raw = Pose((0.52, -0.02, 1.27), q_mul(q_from_axis_angle((0, 1, 0), 0.4), now.q))
    later = apply_pivot(later_raw, PIVOT)
    target = clutch_relative(later, clutch, HEADING, moved_anchor, new)
    hand = q_rotate(q_from_yaw(HEADING), v_sub(later.p, now.p))
    assert _close(v_sub(target.p, before.p), v_scale(hand, new), 1e-12)
    unscaled = clutch_relative(later, clutch, HEADING, anchor, 1.0)
    assert q_between(target.q, unscaled.q) < 1e-9


# ---- the switch in the gate ----

def _still_then_moving(k):
    """Hand still for 1 s, then 0.10 m along x over 1 s, still again; the grip rotated."""
    t = k / RATE
    dx = 0.10 * min(max(t - 1.0, 0.0), 1.0)
    return Pose((HAND.p[0] + dx, HAND.p[1], HAND.p[2]), GRIP)


HAND_END = (HAND.p[0] + 0.10, HAND.p[1], HAND.p[2])


def test_switch_with_the_hand_still_does_not_move_the_target():
    """The hand moved 0.06 m and turned since engagement, then stopped: the click moves nothing."""
    def motion(k):
        t = k / RATE
        dx = 0.06 * min(max(t - 0.2, 0.0) / 0.4, 1.0)
        turn = q_from_axis_angle((0.0, 1.0, 0.2), 0.5 * min(max(t - 0.2, 0.0) / 0.4, 1.0))
        return Pose((HAND.p[0] + dx, HAND.p[1] - dx, HAND.p[2]), q_mul(turn, GRIP))

    frames = _clicks(int(2.0 * RATE), motion, clicks=[(120, 126)])       # at 1.33 s
    steps = run(frames, params(wrist_pivot_offset=PIVOT, **COARSE), state=_state(), tail=0.0)
    after_rest = [p for t, p in _targets(steps) if t >= 1.1]
    assert len(after_rest) > 40
    assert all(same_pose(p, after_rest[0], 1e-9) for p in after_rest)
    assert v_norm(v_sub(after_rest[0].p, TOOL_START.p)) > 0.05     # it had moved


def test_switch_mid_stroke_keeps_the_target_continuous_and_scales_the_rest():
    """A click in the middle of the move: no step at the switch, double speed after it."""
    n = int(3.0 * RATE)
    clicked = _clicks(n, _still_then_moving, clicks=[(135, 140)])       # at 1.5 s
    plain = _clicks(n, _still_then_moving)
    gate = params(wrist_pivot_offset=PIVOT, filter_cutoff_hz=0.0, **COARSE)
    with_switch = run(clicked, gate, state=_state(), tail=0.0)
    without = run(plain, gate, state=_state(), tail=0.0)
    a, b = _targets(with_switch), _targets(without)
    assert [t for t, _ in a] == [t for t, _ in b]
    switch = next(i for i, (t, _) in enumerate(a) if t >= 135 / RATE)
    # Up to and including the cycle of the switch the two runs agree exactly.
    for (_, p), (_, q) in zip(a[:switch + 1], b[:switch + 1]):
        assert same_pose(p, q)
    # No step at the switch beyond the hand's own motion at the new scale (0.1 m/s x 2).
    steps = [v_norm(v_sub(q.p, p.p)) for (_, p), (_, q) in zip(a, a[1:])]
    assert max(steps) <= 2.0 * 0.10 * FRAMES_PER_CYCLE / RATE + 1e-9
    # After the switch, target motion is twice the hand's (rotated by the heading).
    rest_hand = q_rotate(q_from_yaw(HEADING), (0.10 * (2.0 - 1.5), 0.0, 0.0))
    rest_target = v_sub(a[-1][1].p, a[switch][1].p)
    assert _close(rest_target, v_scale(rest_hand, 2.0), 2e-3)
    # The run without the switch moved the hand's own distance.
    assert _close(v_sub(b[-1][1].p, b[0][1].p),
                  q_rotate(q_from_yaw(HEADING), (0.10, 0.0, 0.0)), 2e-3)
    assert all(q_between(p.q, q.q) < 1e-9 for (_, p), (_, q) in zip(a, b))


def test_switching_back_and_forth_with_the_filter_on_has_no_jump():
    n = int(3.0 * RATE)
    frames = _clicks(n, _still_then_moving, clicks=[(110, 115), (140, 145), (170, 175)])
    gate = params(wrist_pivot_offset=PIVOT, **COARSE)
    steps = run(frames, gate, state=_state(), tail=0.0)
    targets = [p for _, p in _targets(steps)]
    moves = [v_norm(v_sub(q.p, p.p)) for p, q in zip(targets, targets[1:])]
    assert all(s.output.hold == Hold.NONE for s in steps[1:])     # never released
    assert max(moves) <= 2.0 * 0.10 * FRAMES_PER_CYCLE / RATE + 1e-9


def test_a_switch_while_released_applies_from_the_next_engagement():
    """Grip open, click, grip again: engagement at the coarse scale, no jump at it."""
    def grip(k):
        return not 60 <= k < 120

    frames = _clicks(int(3.0 * RATE), _still_then_moving, clicks=[(80, 85)], grip=grip)
    steps = run(frames, params(**COARSE), state=_state(), tail=0.0)
    spans = [s for s in steps if s.output.target is not None]
    second = [s for s in spans if s.t > 120 / RATE]
    assert same_pose(second[0].output.target, second[0].tool)       # engages where it is
    first_move = v_norm(v_sub(second[1].output.target.p, second[0].output.target.p))
    assert first_move <= 2.0 * 0.10 * FRAMES_PER_CYCLE / RATE + 1e-9
    # The target moves twice the hand travel from the frame the engagement used.
    engaged_at = _still_then_moving(second[0].output.input_seq - 1000)
    hand = v_norm(v_sub(HAND_END, engaged_at.p))
    travel = v_norm(v_sub(second[-1].output.target.p, second[0].output.target.p))
    assert travel == pytest.approx(2.0 * hand, abs=1e-4)


# ---- arming ----

def _press(state, k, gate, **buttons):
    return on_frame(state, frame(k, **buttons), gate)


def _accept(state, gate):
    """Run one cycle, in which a pending arm request passes its posture check."""
    state, _ = tick(state, Env(now=0.0, owner_is_teleop=True, tool_pose=TOOL_START,
                               tool_pose_age=0.0, servo_active=True), gate)
    assert not state.arm_pending
    return state


def test_the_click_counts_only_while_armed_and_every_arming_starts_fine():
    gate = params(**COARSE)
    state = GateState(calibration=Calibration(0.0, 7, 0))
    state = _press(state, 0, gate)
    state = _press(state, 1, gate, scale_toggle=True)
    assert not state.coarse                               # not armed: ignored
    state = _press(state, 2, gate, arm=True)
    assert state.want_armed and not state.coarse
    state = _accept(state, gate)                          # the posture check passes
    state = _press(state, 3, gate)
    state = _press(state, 4, gate, scale_toggle=True)
    assert state.coarse and 'translation scale: coarse (2.00)' in state.events
    state = _press(state, 5, gate, scale_toggle=True)      # held: no second edge
    assert state.coarse
    state = _press(state, 6, gate, arm=True)              # disarm
    assert not state.want_armed and not state.coarse
    state = _press(state, 7, gate)
    state = _press(state, 8, gate, arm=True)              # arm again: fine
    assert state.want_armed and not state.coarse


def test_a_click_on_the_frame_that_arms_does_not_start_at_coarse():
    gate = params(**COARSE)
    state = GateState(calibration=Calibration(0.0, 7, 0))
    state = _press(state, 0, gate)
    state = _press(state, 1, gate, arm=True, scale_toggle=True)
    assert state.want_armed and not state.coarse
    state = _press(state, 2, gate, arm=True, scale_toggle=True)         # both still held
    assert state.want_armed and not state.coarse
    state = _press(state, 3, gate)
    state = _press(state, 4, gate, arm=True, scale_toggle=True)         # disarm + click
    assert not state.want_armed and not state.coarse
    # Armed, the posture check still pending (no tick yet): a click is ignored too.
    state = _press(state, 5, gate)
    state = _press(state, 6, gate, arm=True)
    state = _press(state, 7, gate, scale_toggle=True)
    assert state.want_armed and state.arm_pending and not state.coarse


def test_a_disarm_by_any_other_path_resets_the_selection():
    gate = params(**COARSE)
    state = replace(_state(), coarse=True, want_armed=False)
    state = on_frame(state, frame(0), gate)
    state, _ = tick(state, Env(now=0.0, owner_is_teleop=True, tool_pose=TOOL_START,
                               tool_pose_age=0.0, servo_active=True), gate)
    assert not state.coarse


def test_rotation_is_never_scaled():
    def turning(k):
        angle = math.radians(30.0) * min(k / RATE, 1.0)
        return Pose(HAND.p, q_mul(q_from_axis_angle((0.2, 0.4, 1.0), angle), HAND.q))

    frames = [frame(k, controller=turning(k)) for k in range(int(1.5 * RATE))]
    fine = run(frames, params(**COARSE), state=_state(), tail=0.0)
    coarse = run(frames, params(**COARSE), state=replace(_state(), coarse=True), tail=0.0)
    for a, b in zip(_targets(fine), _targets(coarse)):
        assert same_pose(a[1], b[1])


# ---- the checks around the scale ----

def _constant_speed(speed):
    def motion(k):
        t = k / RATE
        return Pose((HAND.p[0] + speed * max(t - 0.5, 0.0), HAND.p[1], HAND.p[2]), HAND.q)
    return motion


def test_raw_jump_check_stays_on_the_unscaled_hand():
    """1.0 m/s of hand is below the 1.5 m/s raw limit at either scale; 1.6 m/s is not."""
    frames = [frame(k, controller=_constant_speed(1.6)(k)) for k in range(int(0.8 * RATE))]
    for coarse in (False, True):
        steps = run(frames, params(**COARSE), state=replace(_state(), coarse=coarse), tail=0.0)
        assert Hold.INPUT_JUMP in [s.output.hold for s in steps]


def test_target_step_check_sees_the_scaled_target():
    """At 1.0 m/s of hand the coarse target steps 40 mm per cycle: rejected, not clipped."""
    frames = [frame(k, controller=_constant_speed(1.0)(k)) for k in range(int(1.0 * RATE))]
    fine = run(frames, params(**COARSE), state=_state(), tail=0.0)
    coarse = run(frames, params(**COARSE), state=replace(_state(), coarse=True), tail=0.0)
    assert Hold.STEP_REJECTED not in [s.output.hold for s in fine]
    assert Hold.INPUT_JUMP not in [s.output.hold for s in coarse]
    holds = [s.output.hold for s in coarse]
    assert Hold.STEP_REJECTED in holds
    rejected = holds.index(Hold.STEP_REJECTED)
    assert all(h != Hold.NONE for h in holds[rejected:])          # latched until the grip opens


# ---- parameters ----

def test_largest_translation_scale_is_required_and_has_no_default():
    assert 'max_translation_scale' in REQUIRED
    values = {k: v for k, v in VALUES.items() if k != 'max_translation_scale'}
    with pytest.raises(MissingValue, match='max_translation_scale'):
        build_params(values)


def test_scales_default_to_one_and_are_refused_above_the_largest_or_not_positive():
    gate = params()
    assert gate.translation_scale_fine == 1.0 and gate.translation_scale_coarse == 1.0
    assert params(translation_scale_coarse=2.0).translation_scale(True) == 2.0
    for name in ('translation_scale_fine', 'translation_scale_coarse'):
        with pytest.raises(ValueError, match=name):
            params(**{name: 2.5})
        for bad in (0.0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match=name):
                params(**{name: bad})
    with pytest.raises(ValueError, match='max_translation_scale'):
        params(max_translation_scale=0.0)


# ---- with the delayed arm (lead clamp absorbing) ----

@pytest.fixture(scope='module')
def arm():
    return sv.make_arm(kin.chain_from_urdf(arm_urdf()),
                       velocity_limits={f'xarm_joint{i}': 2.0 for i in range(1, 7)})


def test_switch_while_the_lead_clamp_absorbs_moves_nothing_and_stays_stable(arm):
    """
    Click to coarse mid-stroke while the lead clamp absorbs: no jump, no instability.

    The hand runs at 0.3 m/s, above the 0.25 m/s cap, so the lead clamp is absorbing: the
    target does not jump at the switch, the lead stays within the limit, and the joint
    command does not alternate in sign once the hand has stopped.
    """
    def motion(k):
        t = k / RATE
        dx = 0.30 * min(max(t - 0.5, 0.0), 0.8)
        return Pose((HAND.p[0] + dx, HAND.p[1], HAND.p[2]), GRIP)

    n = int(2.5 * RATE)
    frames = _clicks(n, motion, clicks=[(81, 86)])                   # at 0.9 s
    gate = params(wrist_pivot_offset=PIVOT, **COARSE)
    solver = sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0)
    steps = model_run(frames, gate, arm, solver, READY, delay=0.15, state=_state(), tail=0.0)
    engaged = [s for s in steps if s.output.target is not None]
    assert all(s.output.hold == Hold.NONE for s in steps[1:])       # never released
    switch = next(i for i, s in enumerate(engaged) if s.t >= 81 / RATE)
    leads = [v_norm(v_sub(s.output.target.p, s.tool.p)) for s in engaged]
    assert max(leads) <= gate.max_lead_translation + 1e-9
    moves = [v_norm(v_sub(b.output.target.p, a.output.target.p))
             for a, b in zip(engaged, engaged[1:])]
    # The step of the switch cycle is no larger than the steps around it (all clamp-bound).
    assert moves[switch - 1] <= max(moves[switch - 6:switch - 1]) + 1e-6
    qdot = np.array([s.qdot for s in engaged])
    change = np.abs(np.diff(qdot, axis=0)).max(axis=1)
    assert change[switch - 2:switch + 3].max() <= change.max() + 1e-12
    # No sign-alternating joint velocity once the hand has stopped (INV-7 style).
    rest = qdot[int(1.8 * 50):]
    flips = np.sum(np.diff(np.sign(np.round(rest, 6)), axis=0) != 0, axis=0)
    assert flips.max() <= 2
