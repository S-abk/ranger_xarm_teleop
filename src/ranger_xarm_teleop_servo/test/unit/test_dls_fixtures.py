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
the reference the node uses (MoveIt Servo's state, ``stale_ticks`` 1). Common to all: no
gate hold but a plain clutch release (and the start-up holds before the first target),
the lead inside its limit, every joint speed
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

SOLVER = sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0)
JOINT_LIMIT = 2.0
START_UP = {Hold.NOT_ARMED, Hold.NOT_CALIBRATED, Hold.CLUTCH_RELEASED, Hold.SERVO_NOT_READY}
# The solver holds a joint 0.15 rad from its bound. With the reference the node uses
# (Servo's state, ``stale_ticks`` 1) it is exact; with the reference one cycle older the
# commanded state can go up to 2 mrad into the margin (``test_joint_margin_with_an_older_
# reference``), still 48 mrad outside MoveIt Servo's.
MARGIN = 0.15
MARGIN_SLACK = 0.0
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
                        stale_ticks=1, tail=0.0)
    return out


def _counter_motion(speeds, floor=1e-3):
    """
    Return the largest joint speed (rad/s) against the direction a joint mainly moves in.

    Only joints whose peak speed exceeds ``floor`` count: the fixtures round hand positions
    to a micrometre, which moves joints 1, 4 and 6 by some 1e-5 rad/s with changing sign,
    a different thing from a joint that rings. 1e-4 rad/s is what ``_reversal`` allows the
    1 rad/s peak of the 40 mm step in test_lead_and_stability.
    """
    worst = 0.0
    for v in speeds.T:
        peak = np.abs(v).max()
        if peak > floor:
            worst = max(worst, float(np.max(np.maximum(-np.sign(v[np.argmax(np.abs(v))]) * v,
                                                       0.0))))
    return worst


def _rest_onset(engaged, stops, moves_again, floor=0.005):
    """
    Return when the arm came to rest, or None if it did not.

    The hand is still from ``stops`` to ``moves_again``; the arm is at rest from the first
    cycle on after which no joint speed exceeds ``floor`` (rad/s) until the hand moves.
    """
    window = [s for s in engaged if stops <= s.t < moves_again]
    for i in range(len(window)):
        if all(np.abs(w.qdot).max() <= floor for w in window[i:]):
            return window[i].t
    return None


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
    offset = (Limit.OFFSET_TRANSLATION, Limit.OFFSET_ROTATION)
    assert not [s for s in engaged if s.output.limit in offset]
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


def test_fast_hand_leads_by_the_limit_and_the_target_follows_it_back(runs, arm):
    steps = runs['clutch_fast']
    engaged = _engaged(steps)
    assert any(s.output.limit == Limit.LEAD for s in engaged)
    # The cap holds for each commanded step, and so for the commanded state's tool path.
    assert max(s.solution.linear_speed for s in engaged) <= SOLVER.linear_speed_cap + 1e-9
    path = np.array([sv.tool_pose(arm, s.commanded)[:3, 3] for s in steps])
    speed = np.linalg.norm(np.diff(path, axis=0), axis=1) / params().target_period
    assert speed.max() <= SOLVER.linear_speed_cap + 1e-6
    # The hand rests from 1.9 s to 4.4 s: the arm stops within a second and stays stopped.
    onset = _rest_onset(engaged, 1.9, 4.4)
    assert onset is not None and onset - 1.9 < 1.0
    assert _counter_motion(np.array([s.qdot for s in engaged if onset <= s.t < 4.4])) <= 1e-4
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
    assert _counter_motion(after) <= 1e-4


def test_joint_limit_tilt_holds_a_joint_off_its_bound(runs, arm):
    steps = runs['clutch_joint_limit']
    held = {name for s in _engaged(steps) for name in s.solution.fixed}
    assert held == {'xarm_joint3'}
    assert any(s.solution.limit == sv.LIMIT_JOINT_POSITION for s in _engaged(steps))
    margin = min(float(arm.upper[2] - s.commanded[2]) for s in steps)
    assert margin <= MARGIN + 1e-9                   # the bound of joint 3 was reached
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
    # The hand comes back 0.10 m from 16.5 s: the target turns within three cycles ...
    before = _target_x(steps, 16.5)
    after = [s for s in engaged if 16.5 < s.t <= 16.6]
    assert after[2].output.target.p[0] < before - 1e-3
    # ... and the measured arm (0.15 s late in this model) soon after the target has come
    # back past it (up to the lead limit ahead of it, the target first has to cover that):
    # it was kept off the stretched singularity, where the way back is the direction it
    # lacks and it used to stay put for half a second.
    at_turn = [s.tool.p[0] for s in engaged if s.t <= 16.5][-1]
    crossed = next((s.t for s in engaged if s.t > 16.5 and s.output.target.p[0] < at_turn),
                   None)
    moved = [s.t for s in engaged if s.t > 16.5 and s.tool.p[0] < at_turn - 0.002]
    assert crossed is not None and moved
    assert moved[0] - crossed < 0.3
    # From the hand's turn: the lead (here 0.26 s at the hand's speed) plus the 0.15 s
    # measurement delay; it used to be 0.8 s.
    assert moved[0] - 16.5 < 0.5
    # The return settles without reversing.
    back = np.array([s.qdot for s in engaged if s.t >= 17.6])
    assert _counter_motion(back) <= 1e-4


def test_reach_arm_comes_to_rest_at_the_far_point(runs):
    """
    Past the reach the arm comes to rest and stays still.

    Before the orientation term faded out with the position level's damping, q2 and q3
    chattered here from cycle to cycle at up to the joint velocity limit.
    """
    engaged = _engaged(runs['clutch_reach'])
    onset = _rest_onset(engaged, 13.1, 16.5)       # hand still from 13.1 s to 16.5 s
    assert onset is not None and onset - 13.1 < 1.0
    rest = np.array([s.qdot for s in engaged if onset <= s.t < 16.5])
    assert _counter_motion(rest, floor=1e-4) <= 1e-4


@pytest.mark.parametrize('stale', [2, 3])
def test_reach_rest_and_joint_margin_with_an_older_reference(arm, stale):
    """Robustness: with Servo's state one or two cycles older than the node predicts."""
    steps = run(fixture('clutch_reach'), params(), arm, SOLVER, READY, delay=0.15,
                stale_ticks=stale, tail=0.0)
    engaged = _engaged(steps)
    onset = _rest_onset(engaged, 13.1, 16.5)
    assert onset is not None and onset - 13.1 < 1.0
    rest = np.array([s.qdot for s in engaged if onset <= s.t < 16.5])
    assert _counter_motion(rest, floor=1e-4) <= 1e-4
    limited = run(fixture('clutch_joint_limit'), params(), arm, SOLVER, READY, delay=0.15,
                  stale_ticks=stale, tail=0.0)
    positions = np.array([s.commanded for s in limited])
    margin = np.minimum(positions - arm.lower, arm.upper - positions).min()
    assert margin >= MARGIN - 0.002 and margin > SERVO_HALT_MARGIN
