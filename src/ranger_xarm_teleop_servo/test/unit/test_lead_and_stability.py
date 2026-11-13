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
"""G-14 (INV-6) and loop stability with the measured delay (INV-7), whole driving path."""

import math

from arm_urdf import arm_urdf, READY
from helpers import HAND, params, ready_state, stream
import numpy as np
import pytest
from ranger_xarm_teleop_servo import kinematics as kin
from ranger_xarm_teleop_servo import solver as sv
from ranger_xarm_teleop_servo.gate import absorb, Limit
from ranger_xarm_teleop_servo.mapping import apply_pivot, clutch_relative
from ranger_xarm_teleop_servo.rotations import (Pose, q_between, q_from_axis_angle, q_mul,
                                                v_norm, v_sub)
from ranger_xarm_teleop_servo.teleop_model import run

SOLVER = sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0)
RATE = 90.0


@pytest.fixture(scope='module')
def arm():
    return sv.make_arm(kin.chain_from_urdf(arm_urdf()),
                       velocity_limits={f'xarm_joint{i}': 2.0 for i in range(1, 7)})


def _hand_path(points):
    """Return ``motion(k)``: piecewise-linear hand x offset over ``(t, dx)`` points."""
    times = [t for t, _ in points]
    values = [d for _, d in points]

    def motion(k):
        dx = float(np.interp(k / RATE, times, values))
        return Pose((HAND.p[0] + dx, HAND.p[1], HAND.p[2]), HAND.q)
    return motion


def _run(arm, points, seconds, delay=0.15, stale=1, **gate_overrides):
    frames = stream(int(seconds * RATE), _hand_path(points))
    return run(frames, params(**gate_overrides), arm, SOLVER, READY, delay=delay,
               stale_ticks=stale, state=ready_state(), tail=0.0)


def _engaged(steps):
    return [s for s in steps if s.output.target is not None]


def test_absorbing_reproduces_the_clamped_target_for_the_same_controller_pose():
    """The anchor update is a conjugation: same controller pose, clamped target exactly."""
    pivot = (0.0, 0.02, 0.09)
    clutch = apply_pivot(Pose((0.4, -0.1, 1.2), q_from_axis_angle((0.2, 1, 0.3), 0.5)), pivot)
    turned = q_mul(q_from_axis_angle((1, -0.4, 0.2), 0.7), clutch.q)
    now = apply_pivot(Pose((0.47, -0.05, 1.31), turned), pivot)
    heading = 0.6
    anchor = Pose((0.3, 0.05, 0.25), q_from_axis_angle((0.3, 0.1, 1), 2.4))
    before = clutch_relative(now, clutch, heading, anchor)
    after = Pose((before.p[0] - 0.03, before.p[1] + 0.01, before.p[2]),
                 q_mul(q_from_axis_angle((0.5, -1, 0.2), -0.3), before.q))
    turn = clutch_relative(now, clutch, heading, Pose(anchor.p, (0, 0, 0, 1))).q
    new_anchor, new_filter = absorb(anchor, before, turn, before, after)
    again = clutch_relative(now, clutch, heading, new_anchor)
    assert v_norm(v_sub(again.p, after.p)) < 1e-12
    assert q_between(again.q, after.q) < 1e-7
    assert v_norm(v_sub(new_filter.p, after.p)) < 1e-12
    assert q_between(new_filter.q, after.q) < 1e-7


def test_hand_faster_than_the_caps_never_leads_by_more_than_the_limit(arm):
    """INV-6: hand at 0.6 m/s (2.4 x the cap) for 0.4 s, then still."""
    steps = _run(arm, [(0.0, 0.0), (0.5, 0.0), (0.9, 0.24)], 3.5)
    engaged = _engaged(steps)
    assert len(engaged) > 100
    gp = params()
    for s in engaged:
        lead = v_norm(v_sub(s.output.target.p, s.tool.p))
        assert lead <= gp.max_lead_translation + 1e-9
        assert q_between(s.output.target.q, s.tool.q) <= gp.max_lead_rotation + 1e-7
    assert any(s.output.limit == Limit.LEAD for s in engaged)
    # Tool speed (measured, forward kinematics) never above the cap.
    xs = np.array([s.tool.p for s in engaged])
    speed = np.linalg.norm(np.diff(xs, axis=0), axis=1) / gp.target_period
    assert speed.max() <= SOLVER.linear_speed_cap * 1.01


def test_when_the_hand_stops_the_arm_stops_within_the_lead_and_never_catches_up(arm):
    """INV-6: no catch-up after the hand stops; the excess was absorbed, not stored."""
    steps = _run(arm, [(0.0, 0.0), (0.5, 0.0), (0.9, 0.24)], 3.5)
    engaged = _engaged(steps)
    gp = params()
    stop = min(range(len(engaged)), key=lambda i: abs(engaged[i].t - 0.9))
    at_stop = engaged[stop].tool.p
    final = engaged[-1].tool.p
    travelled_after = v_norm(v_sub(final, at_stop))
    # The lead limit, plus the hand motion still inside the input low-pass filter when the
    # hand stops (hand speed times the filter's time constant): real motion, not stored lead.
    filter_lag = 0.6 / (2.0 * math.pi * gp.filter_cutoff_hz)
    assert travelled_after <= gp.max_lead_translation + filter_lag + 1e-3
    # The arm ended where the target ended, and far short of the hand's 0.24 m.
    assert v_norm(v_sub(final, engaged[-1].output.target.p)) < 1e-3
    start = engaged[0].tool.p
    assert final[0] - start[0] < 0.24 - 0.05
    # Stopped: the last second is still.
    last = np.array([s.tool.p for s in engaged if s.t > 2.5])
    assert np.ptp(last, axis=0).max() < 1e-4


def test_after_absorbing_the_arm_moves_back_as_soon_as_the_hand_does(arm):
    """INV-6: no dead zone after a clamp; reversing the hand reverses the target at once."""
    steps = _run(arm, [(0.0, 0.0), (0.5, 0.0), (0.9, 0.24), (2.0, 0.24), (2.2, 0.20)], 3.0)
    engaged = _engaged(steps)
    before = [s for s in engaged if s.t <= 2.0][-1].output.target.p[0]
    after = [s for s in engaged if 2.0 < s.t <= 2.1]
    assert after[2].output.target.p[0] < before - 1e-3     # within three cycles


def _reversal(after):
    """Return the largest opposite-sign joint speed after the step, relative to the peak."""
    worst = 0.0
    for v in after.T:
        peak = np.abs(v).max()
        if peak < 1e-9:
            continue
        sign = np.sign(v[np.argmax(np.abs(v))])
        worst = max(worst, float(np.max(np.maximum(-sign * v, 0.0)) / peak))
    return worst


@pytest.mark.parametrize('delay', [0.15, 0.30])
@pytest.mark.parametrize('stale', [1, 2, 3, 4])
def test_no_oscillation_at_rest_or_after_a_step(arm, delay, stale):
    """
    INV-7: 40 mm hand step in 0.1 s, held; delay 150 ms and twice that, stale reference.

    ``stale`` 1 is Servo's state including the last command: what the node uses (Servo's
    newest output plus one step of the last command, the window point Servo does not
    publish). 2 happens when Servo's and the node's cycles slip. Margin: up to 3 no joint
    ever reverses; at 4 a decaying reversal under 5 % of the peak speed.
    """
    steps = _run(arm, [(0.0, 0.0), (1.0, 0.0), (1.1, 0.04)], 4.0, delay=delay, stale=stale)
    engaged = _engaged(steps)
    # At rest before the step: no joint motion at all.
    rest = [s for s in engaged if s.t < 1.0]
    assert max(np.abs(s.qdot).max() for s in rest) < 1e-9
    # After the step: the measured tool overshoots the final target by under 2 % of 40 mm.
    final = engaged[-1].output.target.p[0]
    xs = np.array([s.tool.p[0] for s in engaged if s.t > 1.1])
    assert xs.max() <= final + 0.02 * 0.04
    assert abs(xs[-1] - final) < 2e-4
    # No oscillation: the x component of the joint command of the dominant joint never
    # changes sign after the step (velocity of each joint stays one-signed or decays to 0).
    after = np.array([s.qdot for s in engaged if s.t > 1.0])
    assert _reversal(after) <= (1e-4 if stale <= 3 else 0.05)


def test_lead_limit_values_have_no_default():
    from ranger_xarm_teleop_servo.params import build_params, MissingValue, REQUIRED
    from helpers import VALUES
    assert 'max_lead_translation_m' in REQUIRED and 'max_lead_rotation_deg' in REQUIRED
    for name in ('max_lead_translation_m', 'max_lead_rotation_deg'):
        values = dict(VALUES)
        del values[name]
        with pytest.raises(MissingValue, match=name):
            build_params(values)


def test_lead_clamp_does_not_hide_a_jump(arm):
    """G-5 stays ahead of the lead clamp: a jump is rejected, not clipped into the lead."""
    steps = _run(arm, [(0.0, 0.0), (0.5, 0.0), (0.52, 0.05)], 1.0)
    holds = [s.output.hold.name for s in steps]
    assert 'INPUT_JUMP' in holds or 'STEP_REJECTED' in holds
    assert math.isfinite(steps[-1].tool.p[0])
