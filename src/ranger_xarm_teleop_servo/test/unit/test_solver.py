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
"""G-8 and INV-2..5: the damped least-squares solver, over random postures and poor regions."""

import math

from arm_urdf import arm_urdf, READY
import numpy as np
import pytest
from ranger_xarm_teleop_servo import kinematics as kin
from ranger_xarm_teleop_servo import solver as sv
from ranger_xarm_teleop_servo.rotations import q_from_axis_angle, q_to_matrix

PLANNING = {f'xarm_joint{i}': 2.0 for i in range(1, 7)}    # the robot's joint_limits.yaml
PARAMS = sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0)


@pytest.fixture(scope='module')
def arm():
    return sv.make_arm(kin.chain_from_urdf(arm_urdf()), velocity_limits=PLANNING)


def _random_posture(rng, arm):
    # Inside the limits, away from the far ends of the continuous joints.
    lower = np.maximum(arm.lower, -3.0)
    upper = np.minimum(arm.upper, 3.0)
    return rng.uniform(lower, upper)


def _offset(pose, rng, reach=0.3, turn=1.0):
    out = pose.copy()
    out[:3, 3] += rng.uniform(-reach, reach, 3)
    axis = rng.normal(size=3)
    out[:3, :3] = q_to_matrix(q_from_axis_angle(axis, rng.uniform(0, turn))) @ pose[:3, :3]
    return out


def _simulate(arm, q0, target_at, seconds, params=PARAMS, feedforward=False):
    """Integrate the command kinematically; return the joint path and the solutions."""
    q = np.array(q0, dtype=float)
    path, solutions = [q.copy()], []
    previous = target_at(0.0)
    for i in range(int(round(seconds / params.period))):
        target = target_at(i * params.period)
        ff = sv.twist_between(previous, target, params.period) if feedforward else None
        previous = target
        s = sv.solve(arm, q, target, params, ff)
        q = q + s.qdot * params.period
        path.append(q.copy())
        solutions.append(s)
    return np.array(path), solutions


def test_params_refuse_nonsense():
    with pytest.raises(ValueError):
        sv.SolverParams(linear_speed_cap=0.0, angular_speed_cap=1.0)
    with pytest.raises(ValueError):
        sv.SolverParams(0.25, 1.0, limit_zone=0.1, limit_margin=0.15)
    with pytest.raises(ValueError):
        sv.SolverParams(0.25, 1.0, gain=30.0)        # gain * period >= 0.5


def test_velocity_limits_are_the_smaller_of_urdf_and_planning(arm):
    assert np.allclose(arm.velocity, 2.0)
    loose = sv.make_arm(kin.chain_from_urdf(arm_urdf()), velocity_limits={'xarm_joint1': 9.0})
    assert loose.velocity[0] == pytest.approx(3.14)


def test_tool_jacobian_matches_finite_differences_with_a_tool_offset():
    tool = np.eye(4)
    tool[:3, :3] = q_to_matrix(q_from_axis_angle((1, 1, 0), 0.4))
    tool[:3, 3] = (0.03, -0.02, 0.05)
    arm = sv.make_arm(kin.chain_from_urdf(arm_urdf()), tool, PLANNING)
    q = np.array([0.3, -0.5, -0.9, 0.4, 1.0, -0.3])
    jac = sv.tool_jacobian(arm, q)
    eps = 1e-6
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        dp, dr = sv.pose_error(sv.tool_pose(arm, q + d), sv.tool_pose(arm, q - d))
        assert np.allclose(dp / (2 * eps), jac[:3, i], atol=1e-6)
        assert np.allclose(dr / (2 * eps), jac[3:, i], atol=1e-6)


def test_no_error_and_no_feedforward_gives_no_motion(arm):
    s = sv.solve(arm, READY, sv.tool_pose(arm, READY), PARAMS)
    assert np.allclose(s.qdot, 0.0)
    assert s.error_translation == 0.0 and s.limit == sv.LIMIT_NONE


def test_well_conditioned_and_slow_matches_the_exact_inverse(arm):
    """Away from singularities and limits the damped solution is the exact one."""
    twist = np.array([0.02, -0.01, 0.015, 0.05, -0.03, 0.08])
    s = sv.solve(arm, READY, sv.tool_pose(arm, READY), PARAMS, twist)
    assert not s.damped and s.scale == 1.0
    exact = np.linalg.solve(sv.tool_jacobian(arm, READY), twist)
    assert np.allclose(s.qdot, exact, atol=1e-9)


@pytest.mark.parametrize('seed', range(40))
def test_random_postures_and_targets_keep_every_limit(arm, seed):
    """INV-2, INV-3 (one step), INV-4 over random postures and targets."""
    rng = np.random.default_rng(seed)
    q = _random_posture(rng, arm)
    target = _offset(sv.tool_pose(arm, q), rng)
    ff = rng.uniform(-1.0, 1.0, 6) * [0.6, 0.6, 0.6, 3, 3, 3]
    s = sv.solve(arm, q, target, PARAMS, ff)
    assert np.all(np.isfinite(s.qdot))
    # INV-2: within the velocity limits (with the used share).
    assert np.all(np.abs(s.qdot) <= arm.velocity * PARAMS.velocity_fraction + 1e-9)
    # INV-4: tool speed at the reference within the caps.
    twist = sv.tool_jacobian(arm, q) @ s.qdot
    assert np.linalg.norm(twist[:3]) <= PARAMS.linear_speed_cap + 1e-9
    assert np.linalg.norm(twist[3:]) <= PARAMS.angular_speed_cap + 1e-9
    # ... and so is the actual step of one cycle (the arm's path is curved).
    dp, dr = sv.pose_error(sv.tool_pose(arm, q + s.qdot * PARAMS.period), sv.tool_pose(arm, q))
    assert np.linalg.norm(dp) / PARAMS.period <= PARAMS.linear_speed_cap + 1e-9
    assert np.linalg.norm(dr) / PARAMS.period <= PARAMS.angular_speed_cap + 1e-9
    # INV-3: one step never crosses the hard margin from outside it.
    after = q + s.qdot * PARAMS.period
    for i in range(6):
        room_up = arm.upper[i] - q[i] - PARAMS.limit_margin
        room_down = q[i] - arm.lower[i] - PARAMS.limit_margin
        if room_up >= 0:
            assert arm.upper[i] - after[i] >= PARAMS.limit_margin - 1e-12
        if room_down >= 0:
            assert after[i] - arm.lower[i] >= PARAMS.limit_margin - 1e-12
    assert 0.0 <= s.conditioning and 0.0 < s.scale <= 1.0


@pytest.mark.parametrize('seed', range(20))
def test_scaling_keeps_the_direction_of_motion(arm, seed):
    """INV-2: velocity and speed limits scale the whole vector, never one joint."""
    rng = np.random.default_rng(100 + seed)
    q = np.array(READY) + rng.uniform(-0.15, 0.15, 6)
    ff = rng.uniform(-1, 1, 6) * [1, 1, 1, 4, 4, 4]      # beyond the caps
    fast = sv.solve(arm, q, sv.tool_pose(arm, q), PARAMS, ff)
    # Scaled, unless limiting each level's command to its cap already brought it within.
    assert fast.scale < 1.0 or fast.limit == sv.LIMIT_NONE
    if fast.scale < 1.0:
        assert fast.limit in (sv.LIMIT_JOINT_VELOCITY, sv.LIMIT_TOOL_SPEED)
    # The same solve with every limit 100 times wider: the normalised Jacobian, and so the
    # damping, is identical, and nothing limits, so this is the unscaled direction. Each
    # level's command is limited to its cap before the solve (the solver does that too).
    ff = np.array(ff, dtype=float)
    for level, cap in ((slice(0, 3), PARAMS.linear_speed_cap),
                       (slice(3, 6), PARAMS.angular_speed_cap)):
        size = np.linalg.norm(ff[level]) / cap
        if size > 1.0:
            ff[level] /= size
    wide = sv.SolverParams(25.0, 100.0)
    big = sv.make_arm(arm.chain, velocity_limits={n: 200.0 for n in arm.joint_names})
    big = sv.Arm(big.chain, big.tip_to_tool, big.lower, big.upper, arm.velocity * 100.0)
    unlimited = sv.solve(big, q, sv.tool_pose(arm, q), wide, ff)
    assert unlimited.scale == 1.0 and unlimited.damping == fast.damping
    if fast.fixed:
        pytest.skip('a joint at a position bound: that one joint is held, not scaled')
    cosine = np.dot(fast.qdot, unlimited.qdot) / (
        np.linalg.norm(fast.qdot) * np.linalg.norm(unlimited.qdot))
    assert cosine == pytest.approx(1.0, abs=1e-9)


def test_driving_into_a_joint_limit_stops_short_of_servo_margin(arm):
    """INV-3: joint 3 driven to its upper limit stays outside the margin, wider than 0.1."""
    q0 = [0.0, -0.3, -0.25, 0.0, 0.5, 0.0]
    target = sv.tool_pose(arm, [0.0, -0.3, 0.6, 0.0, 0.5, 0.0])    # needs joint 3 past 0.19
    path, solutions = _simulate(arm, q0, lambda t: target, 6.0)
    assert np.max(path[:, 2]) <= arm.upper[2] - PARAMS.limit_margin + 1e-9
    assert np.max(path[:, 2]) > arm.upper[2] - PARAMS.limit_margin - 0.02    # it got there
    assert any(s.limit == sv.LIMIT_JOINT_POSITION for s in solutions)
    for q in path:
        assert np.all(q <= arm.upper - 0.1) and np.all(q >= arm.lower + 0.1)


def test_a_joint_inside_its_margin_may_move_away_but_not_further_in(arm):
    q = np.array(READY, dtype=float)
    q[2] = arm.upper[2] - 0.05                          # already inside the margin
    low, high = sv.position_bounds(arm, q, PARAMS)
    assert high[2] == 0.0 and low[2] < 0.0


def _rotation_about(pose, axis, angle):
    out = pose.copy()
    out[:3, :3] = q_to_matrix(q_from_axis_angle(axis, angle)) @ pose[:3, :3]
    return out


@pytest.mark.parametrize('feedforward', [False, True])
def test_wrist_singularity_is_crossed_without_stopping(arm, feedforward):
    """INV-5 (wrist): tilt from q5 0.4 through the true singularity near -0.25 and beyond."""
    q0 = [0.0, -0.6, -0.7, 0.0, 0.4, 0.0]
    start = sv.tool_pose(arm, q0)

    def target_at(t):
        return _rotation_about(start, (0, -1, 0), math.radians(50) * min(t, 2.0) / 2.0)

    path, solutions = _simulate(arm, q0, target_at, 3.0, feedforward=feedforward)
    assert min(s.conditioning for s in solutions) < 0.05        # it went through
    assert any(s.damped for s in solutions)
    speeds = np.abs(np.diff(path, axis=0)) / PARAMS.period
    assert speeds.max() <= 2.0 * PARAMS.velocity_fraction + 1e-9
    # Never stalls: while the target moves, every cycle moves the arm.
    moving = [np.linalg.norm(s.qdot) for s in solutions[:100]]
    assert min(moving[1:]) > 1e-3
    # Position stays exact (priority), orientation error bounded, and reported.
    assert max(s.error_translation for s in solutions) < 1e-3
    assert max(s.error_rotation for s in solutions) < math.radians(8)
    final = sv.pose_error(target_at(3.0), sv.tool_pose(arm, path[-1]))
    assert np.linalg.norm(final[0]) < 1e-4 and np.linalg.norm(final[1]) < math.radians(0.5)


def test_shoulder_region_is_crossed_without_stopping(arm):
    """INV-5 (shoulder): pull the tool back 0.30 m from ready toward the base axis."""
    start = sv.tool_pose(arm, READY)

    def target_at(t):
        out = start.copy()
        out[0, 3] -= 0.30 * min(t, 2.0) / 2.0
        return out

    path, solutions = _simulate(arm, READY, target_at, 3.0, feedforward=True)
    assert min(s.conditioning for s in solutions) < 0.1
    speeds = np.abs(np.diff(path, axis=0)) / PARAMS.period
    assert speeds.max() <= 2.0 * PARAMS.velocity_fraction + 1e-9
    # One cycle of motion (3 mm) at the start and the end of the ramp, where the
    # feedforward of this simulation is one cycle late; in between well under a millimetre.
    assert max(s.error_translation for s in solutions) < 3.5e-3
    assert max(s.error_translation for s in solutions[20:95]) < 0.5e-3
    final = sv.pose_error(target_at(3.0), sv.tool_pose(arm, path[-1]))
    assert np.linalg.norm(final[0]) < 1e-3


def test_at_an_exact_singularity_the_command_is_finite_and_moves(arm):
    """INV-5: at the wrist singularity a command along the lost direction gives finite motion."""
    q = np.array([0.0, -0.6, -0.7, 0.0, 0.0, 0.0])
    # Find the true singular q5 near -0.3 on this family.
    q5s = np.linspace(-0.4, -0.15, 501)
    q[4] = q5s[int(np.argmin([sv.levels(arm, [0, -0.6, -0.7, 0, v, 0], PARAMS).orientation
                              for v in q5s]))]
    lv = sv.levels(arm, q, PARAMS)
    assert lv.orientation < 0.01 and lv.orientation_damping > 0.49
    for axis in np.eye(3):
        target = _rotation_about(sv.tool_pose(arm, q), axis, 0.2)
        s = sv.solve(arm, q, target, PARAMS)
        assert np.all(np.isfinite(s.qdot))
        assert np.abs(s.qdot).max() <= 2.0 * PARAMS.velocity_fraction + 1e-9
        assert s.damped and s.error_rotation == pytest.approx(0.2, abs=1e-9)


def test_conditioning_matches_the_planned_table(arm):
    """The figures behind the choice of the damping threshold."""
    ready = sv.levels(arm, READY, PARAMS)
    assert ready.position == pytest.approx(1.91, abs=0.01)
    assert ready.orientation == pytest.approx(1.09, abs=0.01)
    aligned = sv.levels(arm, [0, -0.6, -0.7, 0, 0.0, 0], PARAMS)
    assert aligned.orientation == pytest.approx(0.20, abs=0.01)
    assert aligned.position > 1.5           # position is not what the wrist loses


def test_damping_law_is_continuous_and_bounded():
    assert sv.damping_for(0.6, PARAMS) == 0.0
    assert sv.damping_for(0.5, PARAMS) == 0.0
    assert sv.damping_for(0.4999, PARAMS) == pytest.approx(0.0, abs=0.02)
    assert sv.damping_for(0.0, PARAMS) == PARAMS.max_damping
    values = [sv.damping_for(s, PARAMS) for s in np.linspace(0, 1, 101)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    # Worst joint demand of a unit command along the weakest direction: s / (s^2 + lambda^2).
    gains = [s / (s * s + sv.damping_for(s, PARAMS) ** 2) for s in np.linspace(1e-3, 0.5, 500)]
    assert max(gains) <= 2.0 + 1e-9        # 1/s at the threshold; never more


def test_twist_between_inverts_a_constant_motion(arm):
    a = sv.tool_pose(arm, READY)
    b = _rotation_about(a, (0, 0, 1), 0.02)
    b[:3, 3] += (0.004, 0, 0)
    twist = sv.twist_between(a, b, 0.02)
    assert np.allclose(twist[:3], (0.2, 0, 0)) and np.allclose(twist[3:], (0, 0, 1.0))


def test_gain_beyond_the_ringing_bound_is_refused():
    sv.SolverParams(0.25, 1.0, gain=5.0)
    with pytest.raises(ValueError, match='gain'):
        sv.SolverParams(0.25, 1.0, gain=6.0)        # 0.12 > 27/256


def test_a_narrow_damper_zone_still_never_steps_into_the_margin(arm):
    """INV-3 when limit_zone is set barely wider than limit_margin (tau below one cycle)."""
    params = sv.SolverParams(0.25, 1.0, limit_margin=0.15, limit_zone=0.151)
    q0 = [0.0, -0.3, -0.25, 0.0, 0.5, 0.0]
    target = sv.tool_pose(arm, [0.0, -0.3, 0.6, 0.0, 0.5, 0.0])
    path, _ = _simulate(arm, q0, lambda t: target, 4.0, params=params)
    assert np.max(path[:, 2]) <= arm.upper[2] - params.limit_margin + 1e-9


def _out_of_reach(arm, x, seconds, params=PARAMS, rotation_lead=True):
    """Drive toward a target out of reach, the target kept within the lead limits."""
    from ranger_xarm_teleop_servo.gate import clamp_around
    from ranger_xarm_teleop_servo.rotations import pose_from_matrix, pose_to_matrix
    q = np.array(READY, dtype=float)
    goal = sv.tool_pose(arm, q)
    goal[0, 3] = x
    qdots = []
    for _ in range(int(seconds / params.period)):
        current = pose_from_matrix(sv.tool_pose(arm, q))
        target, _, _ = clamp_around(pose_from_matrix(goal), current, 0.05,
                                    math.radians(12.0) if rotation_lead else math.pi)
        s = sv.solve(arm, q, pose_to_matrix(target), params)
        q = q + s.qdot * params.period
        qdots.append(s.qdot)
    return q, np.array(qdots)


@pytest.mark.parametrize('x', [0.93, 0.98, 1.2])
def test_out_of_reach_the_arm_stops_short_of_the_stretched_singularity(arm, x):
    """Past the reach the position level is held at its floor and the arm comes to rest."""
    q, qdots = _out_of_reach(arm, x, 12.0)
    assert sv.position_conditioning(arm, q, PARAMS) >= PARAMS.boundary_floor - 5e-3
    assert np.abs(qdots[-100:]).max() < 1e-3
    flips = sum(1 for a, b in zip(qdots[-100:], qdots[-99:]) if a @ b < 0)
    assert flips == 0
    # From there, a target 20 mm back is followed at once: the way back is not lost. (At
    # the stretched singularity itself the arm took 0.5 s to start moving back.)
    start = sv.tool_pose(arm, q)[0, 3]
    back = sv.tool_pose(arm, q)
    back[0, 3] -= 0.02
    travel = []
    for _ in range(25):
        s = sv.solve(arm, q, back, PARAMS)
        q = q + s.qdot * PARAMS.period
        travel.append(start - sv.tool_pose(arm, q)[0, 3])
    assert travel[4] > 1.0e-3 and travel[24] > 0.010     # after 0.1 s and 0.5 s


@pytest.mark.parametrize('x', [0.93, 0.98, 1.2])
def test_an_oversized_orientation_error_does_not_make_the_joints_chatter(arm, x):
    """Without the rotation lead (an error of 60 deg) each level's command is still capped."""
    _, qdots = _out_of_reach(arm, x, 12.0, rotation_lead=False)
    flips = sum(1 for a, b in zip(qdots[-100:], qdots[-99:]) if a @ b < 0)
    assert flips == 0 and np.abs(qdots[-100:]).max() < 1e-2
