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
The keep-out on the whole driving path (gate, solver, delayed arm): G-15 with G-14.

The hand pulls the tool back into the screen behind the arm, keeps pulling, then turns
back. The arm stops at the margin, its joints come to rest, and it follows the hand back
at once: the gate does not keep the target's lead inside the blocked direction.
"""

from helpers import HAND, params, ready_state, stream
from keepout_robot import READY, resolve, robot_urdf
import numpy as np
import pytest
from ranger_xarm_teleop_servo import gate
from ranger_xarm_teleop_servo import keepout as ko
from ranger_xarm_teleop_servo import solver as sv
from ranger_xarm_teleop_servo.rotations import Pose
from ranger_xarm_teleop_servo.teleop_model import run

RATE = 90.0
MARGIN = 0.025
SOLVER = sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0)
# The hand comes close to the head: the simulation's view cone keeps it in view.
VIEW = {'view_cone_half_angle_deg': 65.0}
PULL_END, TURN = 2.5, 5.0          # s: the hand is still from PULL_END to TURN


@pytest.fixture(scope='module')
def robot():
    urdf = robot_urdf()
    model = ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN)
    arm = sv.arm_from_urdf(urdf, 'xarm_link_base', 'xarm_link_tcp',
                           velocity_limits={f'xarm_joint{i}': 2.0 for i in range(1, 7)})
    return arm, model, model.extra_values({'finger_left_joint': 0.02})


def _hand(points):
    times = [t for t, _ in points]
    values = [d for _, d in points]

    def motion(k):
        return Pose((HAND.p[0] + float(np.interp(k / RATE, times, values)), HAND.p[1],
                     HAND.p[2]), HAND.q)
    return motion


PATH = [(0.0, 0.0), (0.3, 0.0), (PULL_END, -0.30), (TURN, -0.30), (TURN + 1.0, -0.18)]


def _run(robot, keepout=True, stale=1):
    arm, model, extra = robot
    frames = stream(int((TURN + 1.6) * RATE), _hand(PATH))
    return run(frames, params(**VIEW), arm, SOLVER, READY, delay=0.15, stale_ticks=stale,
               state=ready_state(), tail=0.0, keepout=model if keepout else None,
               extra=extra)


def _commanded_x(arm, steps):
    return np.array([sv.tool_pose(arm, s.commanded)[0, 3] for s in steps])


def _response_after_turn(arm, steps):
    """Seconds from the hand's turn until the commanded tool moves forward at 20 mm/s."""
    t = np.array([s.t for s in steps])
    x = _commanded_x(arm, steps)
    v = np.gradient(x, t)
    after = np.nonzero((t >= TURN) & (v > 0.02))[0]
    return float(t[after[0]] - TURN) if len(after) else np.inf


def test_pulled_into_the_screen_the_arm_keeps_the_margin_rests_and_turns_back_at_once(robot):
    arm, model, extra = robot
    steps = _run(robot)
    engaged = [s for s in steps if s.solution is not None]
    assert any(s.solution.keepout_bound for s in engaged)
    # INV-1 on the commanded state: never below the margin.
    clearance = [model.clearance(s.commanded, extra)[0] for s in steps]
    assert min(clearance) >= MARGIN - 1e-9
    # While the hand is held back the arm may still creep along the boundary toward it,
    # but it does not chatter: no joint changes direction more than once, and it slows.
    resting = np.array([s.qdot for s in steps if TURN - 1.0 <= s.t < TURN])
    for v in resting.T:
        moving = v[np.abs(v) > 1e-4]
        assert len(moving) < 2 or np.sum(np.diff(np.sign(moving)) != 0) <= 1
    speed = np.abs(resting).max(axis=1)
    assert speed[-1] <= speed[0] + 1e-6 and speed[-1] < 0.05
    # Back at once: within a few cycles of the hand's turn (the 8 Hz filter's lag
    # included).
    assert _response_after_turn(arm, steps) <= 0.12


def test_without_the_gate_absorbing_the_blocked_lead_the_turn_waits(robot, monkeypatch):
    """Mutation check of the test above: the absorption is what makes the turn immediate."""
    arm, _, _ = robot
    with_absorption = _response_after_turn(arm, _run(robot))
    monkeypatch.setattr(gate, 'absorb_blocked', lambda pose, measured, blocked: (pose, False))
    without = _response_after_turn(arm, _run(robot))
    assert without > with_absorption + 0.1


def test_lead_along_the_blocked_direction_only_is_absorbed():
    measured = Pose((0.3, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0))
    target = Pose((0.26, 0.02, 0.31), (0.0, 0.0, 0.0, 1.0))
    out, absorbed = gate.absorb_blocked(target, measured, (-0.2, 0.0, 0.0))
    assert absorbed and out.p == pytest.approx((0.3, 0.02, 0.31))
    out, absorbed = gate.absorb_blocked(target, measured, (0.2, 0.0, 0.0))
    assert not absorbed and out == target                # leading out of it: kept
    out, absorbed = gate.absorb_blocked(target, measured, (0.0, 0.0, 0.0))
    assert not absorbed and out == target


def test_far_from_the_structure_the_driving_path_is_unchanged(robot):
    """The keep-out changes nothing on a path that stays clear of the structure."""
    arm, model, extra = robot
    frames = stream(int(3.0 * RATE), _hand([(0.0, 0.0), (0.3, 0.0), (2.0, 0.12)]))
    with_model = run(frames, params(**VIEW), arm, SOLVER, READY, delay=0.15, stale_ticks=1,
                     state=ready_state(), tail=0.0, keepout=model, extra=extra)
    plain = run(frames, params(**VIEW), arm, SOLVER, READY, delay=0.15, stale_ticks=1,
                state=ready_state(), tail=0.0)
    assert not any(s.solution is not None and s.solution.keepout_bound for s in with_model)
    for a, b in zip(with_model, plain):
        assert np.array_equal(a.qdot, b.qdot)
