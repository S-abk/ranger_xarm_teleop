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
"""The link keep-out in the solver (G-15): INV-1 and INV-2, settling, infeasible rows."""

import itertools
import math

from keepout_robot import READY, resolve, robot_urdf
import numpy as np
import pytest
from ranger_xarm_teleop_servo import keepout as ko
from ranger_xarm_teleop_servo import solver as sv

MARGIN = 0.025
PARAMS = sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0)
# Pulled back: link 2 within the slowing zone of the screen behind the arm.
NEAR_SCREEN = [0.0, -1.05, -0.55, 0.0, 1.5, 0.0]


@pytest.fixture(scope='module')
def robot():
    urdf = robot_urdf()
    model = ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN)
    arm = sv.arm_from_urdf(urdf, 'xarm_link_base', 'xarm_link_tcp')
    return arm, model, model.extra_values({'finger_left_joint': 0.02})


def _moved(pose, dx=0.0, dy=0.0, dz=0.0):
    out = np.array(pose, dtype=float)
    out[:3, 3] += (dx, dy, dz)
    return out


def _drive(arm, model, extra, q, target, cycles, delay=0, params=PARAMS):
    """Integrate the solver ``cycles`` times; return postures, commands and solutions."""
    history = [np.array(q, dtype=float)]
    commands, solutions = [], []
    for _ in range(cycles):
        reference = history[max(0, len(history) - 1 - delay)]
        solution = sv.solve(arm, reference, target, params, keepout=model, extra=extra)
        history.append(history[-1] + solution.qdot * params.period)
        commands.append(solution.qdot)
        solutions.append(solution)
    return history, np.array(commands), solutions


def _floor_kept(model, extra, history):
    """Return the largest amount (m) any pair fell below min(its clearance, margin)."""
    worst = 0.0
    floor = np.minimum(model.pair_minima(history[0], extra), model.margin)
    for q in history[1:]:
        now = model.pair_minima(q, extra)
        worst = max(worst, float(np.max(floor - now)))
        floor = np.maximum(floor, np.minimum(now, model.margin))
    return worst


# --- unchanged where it does not bind ----------------------------------------------------------

def test_far_from_the_structure_the_command_is_bit_identical(robot):
    arm, model, extra = robot
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = np.array(READY) + rng.normal(0.0, 0.2, 6)
        target = _moved(sv.tool_pose(arm, q), *rng.normal(0.0, 0.05, 3))
        with_model = sv.solve(arm, q, target, PARAMS, keepout=model, extra=extra)
        if with_model.keepout_bound:
            continue
        plain = sv.solve(arm, q, target, PARAMS)
        assert np.array_equal(with_model.qdot, plain.qdot)
        assert with_model.limit == plain.limit and with_model.scale == plain.scale
        assert np.array_equal(with_model.blocked, np.zeros(3)) or plain.fixed


def test_inv2_motion_away_from_the_structure_is_never_reduced(robot):
    arm, model, extra = robot
    for q in (NEAR_SCREEN, [0.3, -1.1, -0.5, 0.2, 1.4, 0.1]):
        assert model.clearance(q, extra)[0] < MARGIN + 0.06      # inside the slowing zone
        target = _moved(sv.tool_pose(arm, q), dx=0.04, dz=0.02)  # forward: away
        with_model = sv.solve(arm, q, target, PARAMS, keepout=model, extra=extra)
        plain = sv.solve(arm, q, target, PARAMS)
        assert not with_model.keepout_bound
        assert np.array_equal(with_model.qdot, plain.qdot)


# --- INV-1 -----------------------------------------------------------------------------------

def test_inv1_no_pair_falls_below_the_margin_for_random_motions(robot):
    """Random start postures clear of the structure, random targets (many beyond it)."""
    arm, model, extra = robot
    rng = np.random.default_rng(2)
    lower = np.array([-3.0, -1.9, -3.7, -3.0, -1.5, -3.0])
    upper = np.array([3.0, 1.9, 0.0, 3.0, 3.0, 3.0])
    bound_runs = 0
    for _ in range(16):
        while True:
            q = rng.uniform(lower, upper)
            if model.clearance(q, extra)[0] > MARGIN + 0.005:
                break
        target = _moved(sv.tool_pose(arm, q), *rng.normal(0.0, 0.25, 3))
        history, _, solutions = _drive(arm, model, extra, q, target, 120)
        assert _floor_kept(model, extra, history) <= 1e-9
        bound_runs += any(s.keepout_bound for s in solutions)
    assert bound_runs >= 4


def test_inv1_pulling_back_into_the_screen_stops_at_the_margin_and_slides(robot):
    arm, model, extra = robot
    start = np.array(READY)
    target = _moved(sv.tool_pose(arm, start), dx=-0.25)
    history, commands, solutions = _drive(arm, model, extra, start, target, 300)
    assert _floor_kept(model, extra, history) <= 1e-9
    assert any(s.limit == sv.LIMIT_LINK_KEEPOUT for s in solutions)
    clearance, _ = model.clearance(history[-1], extra)
    assert clearance == pytest.approx(MARGIN, abs=2e-4)          # it got there
    # ... and the tool still went back by other joints (the hand's direction is kept).
    tool = sv.tool_pose(arm, history[-1])[:3, 3]
    assert tool[0] < sv.tool_pose(arm, start)[0, 3] - 0.15


# --- INV-2 along the boundary ------------------------------------------------------------------

def test_inv2_motion_along_the_boundary_keeps_its_speed(robot):
    arm, model, extra = robot
    start = np.array(READY)
    target = _moved(sv.tool_pose(arm, start), dx=-0.25)
    history, _, _ = _drive(arm, model, extra, start, target, 300)
    q = history[-1]
    tool = sv.tool_pose(arm, q)
    for dy in (0.05, -0.05):
        sideways = _moved(tool, dy=dy)
        with_model = sv.solve(arm, q, sideways, PARAMS, keepout=model, extra=extra)
        plain = sv.solve(arm, q, sideways, PARAMS)
        v_model = (sv.tool_jacobian(arm, q) @ with_model.qdot)[:3]
        v_plain = (sv.tool_jacobian(arm, q) @ plain.qdot)[:3]
        assert v_model[1] * np.sign(dy) >= 0.95 * abs(v_plain[1])
    # Sliding for a second keeps the clearance and moves the tool sideways at speed.
    history, _, _ = _drive(arm, model, extra, q, _moved(tool, dy=0.3), 50)
    assert _floor_kept(model, extra, history) <= 1e-9
    moved = sv.tool_pose(arm, history[-1])[1, 3] - tool[1, 3]
    assert moved > 0.9 * 0.25 * 1.0 - 0.05


def test_the_blocked_direction_points_where_the_hand_could_not_go(robot):
    arm, model, extra = robot
    q = NEAR_SCREEN
    history, _, _ = _drive(arm, model, extra, q, _moved(sv.tool_pose(arm, q), dx=-0.3), 150)
    q = history[-1]
    target = _moved(sv.tool_pose(arm, q), dx=-0.05)
    solution = sv.solve(arm, q, target, PARAMS, keepout=model, extra=extra)
    assert solution.keepout_bound
    assert solution.blocked[0] < 0.0                  # toward the back, as asked
    away = sv.solve(arm, q, _moved(sv.tool_pose(arm, q), dx=0.05), PARAMS, keepout=model,
                    extra=extra)
    assert not away.keepout_bound and np.array_equal(away.blocked, np.zeros(3))


# --- settling ----------------------------------------------------------------------------------

def _direction_changes(commands, floor=1e-3):
    out = []
    for v in commands.T:
        moving = v[np.abs(v) > floor]
        out.append(int(np.sum(np.diff(np.sign(moving)) != 0)) if len(moving) > 1 else 0)
    return out


@pytest.mark.parametrize('delay', [0, 1, 2, 3])
def test_held_beyond_the_boundary_the_joints_settle(robot, delay):
    """The failure the task-space boundary had before: the joints must come to rest."""
    arm, model, extra = robot
    start = np.array(READY)
    target = _moved(sv.tool_pose(arm, start), dx=-0.30)
    history, commands, _ = _drive(arm, model, extra, start, target, 400, delay=delay)
    assert np.abs(commands[-1]).max() < 1e-3
    assert max(_direction_changes(commands[-100:])) <= 1
    if delay == 0:
        assert _floor_kept(model, extra, history) <= 1e-9


def test_keepout_time_is_held_to_the_stability_bound():
    with pytest.raises(ValueError, match='keepout_time'):
        sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0, keepout_time=0.15)
    sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0, keepout_time=0.19)


# --- infeasible and over-determined rows ---------------------------------------------------------

def test_least_change_is_the_constrained_optimum():
    """Against every active set of small random problems (brute force)."""
    rng = np.random.default_rng(4)
    for _ in range(60):
        n, k = 4, int(rng.integers(1, 6))
        m = rng.normal(size=(n, n))
        h = m.T @ m + 0.1 * np.eye(n)
        start = rng.normal(size=n)
        g = rng.normal(size=(k, n))
        bound = g @ rng.normal(size=n) * 0 + rng.normal(size=k) * 0.5
        z = sv._least_change(h, start, g, bound)
        best = None
        for size in range(k + 1):
            for active in itertools.combinations(range(k), size):
                a = g[list(active)]
                if size:
                    kkt = np.block([[h, a.T], [a, np.zeros((size, size))]])
                    rhs = np.concatenate([h @ start, bound[list(active)]])
                    try:
                        cand = np.linalg.solve(kkt, rhs)[:n]
                    except np.linalg.LinAlgError:
                        continue
                else:
                    cand = start
                if np.all(g @ cand >= bound - 1e-9):
                    cost = (cand - start) @ h @ (cand - start)
                    if best is None or cost < best[0]:
                        best = (cost, cand)
        if best is None:
            assert z is None
        else:
            assert z is not None and np.all(g @ z >= bound - 1e-7)
            assert (z - start) @ h @ (z - start) <= best[0] + 1e-7


def test_contradictory_rows_give_zero_velocity_and_the_keepout_limit(robot):
    arm, model, extra = robot
    q = NEAR_SCREEN
    rows = model.evaluate(q, extra)
    assert len(rows.distance) >= 2
    # Ask every row to move away by more than the joints can do together, both ways.
    contradictory = ko.Rows(np.vstack([rows.grad[:1], -rows.grad[:1]]),
                            np.array([0.0, 0.0]), np.array([rows.pair[0]] * 2), 0.0,
                            rows.nearest)
    bias = np.array([0.05, 0.05])
    jn = sv._normalised(arm, sv.tool_jacobian(arm, q), PARAMS)
    low, high = sv.position_bounds(arm, q, PARAMS)
    z, _, _, _, bound, infeasible = sv._constrained(
        arm, jn, np.array([0.5, 0, 0, 0, 0, 0]), low / arm.velocity, high / arm.velocity,
        contradictory, bias, model, PARAMS)
    assert infeasible and bound and np.array_equal(z, np.zeros(6))


def _between_robot():
    """The test robot with a second box in front of link 2: link 2 between two obstacles."""
    wall = ('<link name="front_wall"><collision><geometry><box size="0.03 0.40 0.30"/>'
            '</geometry></collision></link><joint name="base_to_front_wall" type="fixed">'
            '<parent link="xarm_link_base"/><child link="front_wall"/>'
            '<origin xyz="0.075 0 0.40"/></joint></robot>')
    urdf = robot_urdf().replace('</robot>', wall)
    model = ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN)
    return sv.arm_from_urdf(urdf, 'xarm_link_base', 'xarm_link_tcp'), model


def test_between_two_obstacles_neither_is_approached_and_the_free_direction_is_open():
    arm, model = _between_robot()
    extra = model.extra_values({'finger_left_joint': 0.02})
    # Lean back until link 2 is held off the screen behind, with the wall just in front.
    q = np.array([0.0, -1.0, -0.6, 0.0, 1.5, 0.0])
    pairs = model.pair_clearances(q, extra)
    assert pairs[('l2', 'front_wall')] < MARGIN + 0.06
    for dx in (-0.2, 0.2):                    # toward either obstacle
        target = _moved(sv.tool_pose(arm, q), dx=dx)
        history, _, solutions = _drive(arm, model, extra, q, target, 100)
        assert _floor_kept(model, extra, history) <= 1e-9
    # Sideways, along both: free, at speed.
    target = _moved(sv.tool_pose(arm, q), dy=0.3)
    history, commands, _ = _drive(arm, model, extra, q, target, 50)
    assert _floor_kept(model, extra, history) <= 1e-9
    assert abs(sv.tool_pose(arm, history[-1])[1, 3] - sv.tool_pose(arm, q)[1, 3]) > 0.15


def test_command_is_finite_and_never_breaks_a_row_at_contact(robot):
    """Starting in contact (as after a collision halt): no row is ever made worse."""
    arm, model, extra = robot
    q = np.array([0.0, -1.35, -0.3, 0.0, 1.6, 0.0])
    assert model.clearance(q, extra)[0] < 0.0
    for target in (_moved(sv.tool_pose(arm, q), dx=-0.1), _moved(sv.tool_pose(arm, q), dx=0.1)):
        history, commands, _ = _drive(arm, model, extra, q, target, 60)
        assert np.all(np.isfinite(commands))
        assert _floor_kept(model, extra, history) <= 1e-9
    # Forward (away) moves it out.
    assert model.clearance(history[-1], extra)[0] > model.clearance(q, extra)[0] + 0.01
    assert math.isfinite(model.clearance(history[-1], extra)[0])
