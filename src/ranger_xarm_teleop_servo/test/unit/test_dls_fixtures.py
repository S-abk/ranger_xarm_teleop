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
The solver fixtures through the whole driving path: gate, solver, delayed model arm.

Each fixture starts at the ready posture and runs with the realistic delay (150 ms) and
reference age (3 cycles). Common to all: no gate hold but a plain clutch release (and the
start-up holds before the first target), the lead inside its limit, every joint speed
inside 95 % of its limit, and no joint nearer than the solver's margin to a position bound.
Then, per fixture, that the region or bound it was made for is reached.
"""

import math

from arm_urdf import arm_urdf, READY
from helpers import engaged_spans, fixture, params
import numpy as np
import pytest
from ranger_xarm_teleop_servo import kinematics as kin
from ranger_xarm_teleop_servo import solver as sv
from ranger_xarm_teleop_servo.gate import Hold, Limit
from ranger_xarm_teleop_servo.rotations import q_between, v_norm, v_sub
from ranger_xarm_teleop_servo.teleop_model import run
from test_lead_and_stability import _reversal

SOLVER = sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0)
JOINT_LIMIT = 2.0
START_UP = {Hold.NOT_ARMED, Hold.NOT_CALIBRATED, Hold.CLUTCH_RELEASED, Hold.SERVO_NOT_READY}
# The velocity damper works from a reference that is two cycles old, so a joint approaching
# a bound overshoots its 0.15 rad margin by a few mrad. The margin is where the damper
# starts to hold, not the bound: MoveIt Servo halts at 0.1 rad.
MARGIN = 0.15
MARGIN_SLACK = 0.005
SERVO_HALT_MARGIN = 0.1
READY_TOOL_Z = 0.246
NAMES = ('clutch_shoulder', 'clutch_fast', 'clutch_step', 'clutch_joint_limit', 'clutch_reach')


@pytest.fixture(scope='module')
def arm():
    return sv.make_arm(kin.chain_from_urdf(arm_urdf()),
                       velocity_limits={f'xarm_joint{i}': JOINT_LIMIT for i in range(1, 7)})


@pytest.fixture(scope='module')
def runs(arm):
    out = {}
    for name in NAMES:
        out[name] = run(fixture(name), params(), arm, SOLVER, READY, delay=0.15,
                        stale_ticks=3, tail=0.0)
    return out


def _quiet_reversal(speeds, floor=1e-3):
    """
    Return ``_reversal`` of the joints that move at all (peak above ``floor`` rad/s).

    The fixtures round hand positions to a micrometre, which moves joints 1, 4 and 6 by
    some 1e-5 rad/s with changing sign: a different thing from a joint that rings.
    """
    moving = speeds[:, np.abs(speeds).max(axis=0) > floor]
    return _reversal(moving) if moving.size else 0.0


def _engaged(steps):
    return [s for s in steps if s.output.target is not None]


def _target_x(steps, t):
    return [s for s in _engaged(steps) if s.t <= t][-1].output.target.p[0]


@pytest.mark.parametrize('name', NAMES)
def test_only_a_plain_clutch_release_ever_stops_the_target(runs, name):
    steps = runs[name]
    first = next(i for i, s in enumerate(steps) if s.output.target is not None)
    assert {s.output.hold for s in steps[:first]} <= START_UP | {Hold.NONE}
    allowed = {Hold.NONE, Hold.CLUTCH_RELEASED} | {Hold.SERVO_NOT_READY}
    assert {s.output.hold for s in steps[first:]} <= allowed
    # Servo-not-ready is the one cycle between a grip and the first target, nothing else.
    for i, s in enumerate(steps[first:], first):
        if s.output.hold == Hold.SERVO_NOT_READY:
            assert steps[i + 1].output.target is not None or \
                steps[i + 1].output.hold == Hold.SERVO_NOT_READY
    assert steps[-1].output.hold == Hold.CLUTCH_RELEASED
    assert len(engaged_spans(steps)) == {'clutch_shoulder': 2, 'clutch_reach': 3}.get(name, 1)


@pytest.mark.parametrize('name', NAMES)
def test_lead_joint_speed_and_joint_margin_stay_inside_their_limits(runs, arm, name):
    steps = runs[name]
    gp = params()
    engaged = _engaged(steps)
    for s in engaged:
        assert v_norm(v_sub(s.output.target.p, s.tool.p)) <= gp.max_lead_translation + 1e-9
        assert q_between(s.output.target.q, s.tool.q) <= gp.max_lead_rotation + 1e-7
    assert max(np.abs(s.qdot).max() for s in steps) <= 0.95 * JOINT_LIMIT + 1e-9
    positions = np.array([s.commanded for s in steps] + [s.measured for s in steps])
    margin = np.minimum(positions - arm.lower, arm.upper - positions).min(axis=0)
    assert margin.min() >= MARGIN - MARGIN_SLACK
    assert margin.min() > SERVO_HALT_MARGIN
    solutions = [s.solution for s in engaged]
    assert all(math.isfinite(x.conditioning) and x.conditioning >= 0.0 for x in solutions)


def test_shoulder_pull_takes_the_wrist_over_the_axis_of_joint_1(runs, arm):
    steps = runs['clutch_shoulder']
    conditioning = [sv.levels(arm, s.commanded, SOLVER).conditioning for s in steps]
    assert min(conditioning) < 0.15
    assert any(s.solution.damped for s in _engaged(steps))
    tool = _engaged(steps)[-1].tool.p
    assert tool[0] < 0.05                   # 0.31 m at the start: 0.29 m back, tool above
    assert tool[2] > READY_TOOL_Z + 0.15    # risen, so it passes over the shoulder
    # It never goes below the ready height, so never into the shoulder housing.
    assert min(s.tool.p[2] for s in _engaged(steps)) >= READY_TOOL_Z - 1e-3
    # Both engagements follow their target: the lost accuracy is orientation, not position.
    assert max(s.solution.error_translation for s in _engaged(steps)) < 0.03


def test_fast_hand_leads_by_the_limit_and_the_target_follows_it_back(runs):
    steps = runs['clutch_fast']
    engaged = _engaged(steps)
    assert any(s.output.limit == Limit.LEAD for s in engaged)
    # The cap holds for the commanded tool speed. The measured one is the same motion seen
    # through the Jacobian at the reference, which is two cycles old: a little above it.
    assert max(s.solution.linear_speed for s in engaged) <= SOLVER.linear_speed_cap + 1e-9
    speeds = np.linalg.norm(np.diff([s.tool.p for s in engaged], axis=0), axis=1) / 0.02
    assert speeds.max() <= SOLVER.linear_speed_cap * 1.03
    # Hand rests 1.9-4.4 s: the arm stops and stays stopped.
    rest = [s for s in engaged if 3.4 < s.t < 4.4]
    assert max(np.abs(s.qdot).max() for s in rest) < 0.01
    # The hand comes back 40 mm from 4.4 s: the target turns within three cycles.
    before = _target_x(steps, 4.4)
    after = [s for s in engaged if 4.4 < s.t <= 4.5]
    assert after[2].output.target.p[0] < before - 1e-3


def test_step_stays_at_rest_then_settles_without_reversing(runs):
    steps = runs['clutch_step']
    engaged = _engaged(steps)
    assert max(np.abs(s.qdot).max() for s in engaged if s.t < 4.2) < 1e-9
    final = engaged[-1].output.target.p[0]
    xs = np.array([s.tool.p[0] for s in engaged if s.t > 4.3])
    assert xs.max() <= final + 0.02 * 0.04
    assert abs(xs[-1] - final) < 2e-4
    after = np.array([s.qdot for s in engaged if s.t >= 4.25])
    assert _quiet_reversal(after) <= 1e-4


def test_joint_limit_tilt_holds_a_joint_off_its_bound(runs, arm):
    steps = runs['clutch_joint_limit']
    held = {name for s in _engaged(steps) for name in s.solution.fixed}
    assert held == {'xarm_joint3'}
    assert any(s.solution.limit == sv.LIMIT_JOINT_POSITION for s in _engaged(steps))
    margin = min(float(arm.upper[2] - s.commanded[2]) for s in steps)
    assert margin < MARGIN + MARGIN_SLACK            # the bound of joint 3 was reached
    # Position first: the tool point stays where it was; the lost accuracy is the tilt.
    last = _engaged(steps)[-1]
    assert v_norm(v_sub(last.tool.p, _engaged(steps)[0].tool.p)) < 2e-3
    assert 0.1 < last.solution.error_rotation < math.radians(40.0)
    # At the bound the joint is still, not chattering.
    rest = [s for s in _engaged(steps) if s.t > 5.0]
    assert max(abs(s.qdot[2]) for s in rest) < 1e-3


def test_reach_runs_out_at_the_elbow_and_the_target_comes_back_at_once(runs, arm):
    steps = runs['clutch_reach']
    engaged = _engaged(steps)
    conditioning = [s.solution.conditioning for s in engaged]
    assert min(conditioning) < 0.01                  # elbow straight: orientation level is lost
    far = max(s.tool.p[0] for s in engaged)
    assert far > 0.88                                # the position reach, well past 0.68 m
    # Past the reach the target is held at the lead limit and the excess is absorbed.
    assert any(s.output.limit == Limit.LEAD for s in engaged if s.t > 11.0)
    assert max(v_norm(v_sub(s.output.target.p, s.tool.p)) for s in engaged) > 0.049
    # It is the elbow, not the bound of joint 3.
    assert min(float(arm.upper[2] - s.commanded[2]) for s in steps) > 0.5
    assert min(float(s.commanded[2] - arm.lower[2]) for s in steps) > 1.0
    # At rest at the far point (hand still from 13.1 s): no creep, and nothing restarts.
    rest = np.array([s.qdot for s in engaged if 14.1 <= s.t <= 16.4])
    assert np.abs(rest).max() < 0.005
    # The hand comes back 0.10 m from 16.5 s: the target turns within three cycles.
    before = _target_x(steps, 16.5)
    after = [s for s in engaged if 16.5 < s.t <= 16.6]
    assert after[2].output.target.p[0] < before - 1e-3
    # The return settles without reversing.
    back = np.array([s.qdot for s in engaged if s.t >= 17.6])
    assert _quiet_reversal(back) <= 1e-4
