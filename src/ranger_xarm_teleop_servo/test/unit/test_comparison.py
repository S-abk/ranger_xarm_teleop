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
"""The offline comparison of the solver with the old singularity rule (small counts)."""

import math

from arm_urdf import arm_urdf, READY
from helpers import fixture, params
import numpy as np
import pytest
from ranger_xarm_teleop_servo import comparison as cmp
from ranger_xarm_teleop_servo import kinematics as kin
from ranger_xarm_teleop_servo import solver as sv
from ranger_xarm_teleop_servo import workspace
from ranger_xarm_teleop_servo.rotations import pose_to_matrix, q_from_matrix
from ranger_xarm_teleop_servo.teleop_model import run

SOLVER = sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0)


@pytest.fixture(scope='module')
def arm():
    return cmp.arm_from_description(arm_urdf())


@pytest.fixture(scope='module')
def origin(arm):
    return sv.tool_pose(arm, READY)


def test_arm_from_description_limits_every_joint():
    arm = cmp.arm_from_description(arm_urdf(), joint_velocity=1.5)
    assert np.allclose(arm.velocity, 1.5)
    assert list(arm.joint_names) == [f'xarm_joint{i}' for i in range(1, 7)]


def test_reference_and_measured_index():
    assert cmp.reference_index(10, 1) == 9 and cmp.reference_index(10, 3) == 7
    assert cmp.reference_index(2, 3) == 0 and cmp.reference_index(1, 3) == 0
    assert cmp.measured_index(10, 0) == 9 and cmp.measured_index(10, 8) == 1
    assert cmp.measured_index(3, 8) == 0


def test_track_with_no_delay_is_plain_kinematic_integration(arm, origin):
    targets = cmp.straight_targets(origin, [0.0, 0.08, 0.04], 0.25, 0.3, 0.02)
    ff = cmp.twists(targets, origin, 0.02)
    result = cmp.track(arm, SOLVER, READY, targets, ff)
    q = np.array(READY, dtype=float)
    for k, target in enumerate(targets):
        solution = sv.solve(arm, q, target, SOLVER, ff[k])
        assert np.allclose(result.qdot[k], solution.qdot, atol=1e-12)
        q = q + solution.qdot * SOLVER.period
    assert np.allclose(result.q[-1], q, atol=1e-12) and len(result.q) == len(targets) + 1
    assert result.tool_error[-1, 0] < 0.002           # it got there


def test_track_matches_the_delay_model_of_the_node_driven_by_the_gate(arm):
    """The same targets, feedforward, delay and reference age give the same joint commands."""
    steps = run(fixture('clutch_step'), params(), arm, SOLVER, READY, delay=0.15,
                stale_ticks=3, tail=0.0)
    engaged = [s for s in steps if s.output.target is not None]
    targets = [pose_to_matrix(s.output.target) for s in engaged]
    ff = [s.output.feedforward for s in engaged]
    result = cmp.track(arm, SOLVER, READY, targets, ff, delay_ticks=8, stale_ticks=3)
    assert np.allclose(result.qdot, [s.qdot for s in engaged], atol=1e-9)
    assert np.allclose(result.q[1:], [s.commanded for s in engaged], atol=1e-9)
    assert result.peak_joint_speed == pytest.approx(max(np.abs(s.qdot).max() for s in engaged))


def test_straight_targets_move_at_the_cap_then_hold(origin):
    targets = cmp.straight_targets(origin, [0.10, 0.0, 0.0], 0.25, 0.2, 0.02)
    assert len(targets) == 20 + 10                    # 0.4 s moving, 0.2 s holding
    steps = np.diff([t[0, 3] for t in targets])
    assert np.allclose(steps[:19], 0.005) and np.allclose(steps[19:], 0.0)
    assert targets[-1][0, 3] == pytest.approx(origin[0, 3] + 0.10)
    assert all(np.array_equal(t[:3, :3], origin[:3, :3]) for t in targets)
    assert [t[1, 3] for t in targets] == [origin[1, 3]] * len(targets)


def test_cube_offsets_are_the_grid_of_the_workspace_map():
    offsets = cmp.cube_offsets(0.30, 0.10)
    assert len(offsets) == 343
    chain = kin.chain_from_urdf(arm_urdf())
    rows = workspace.grid(chain, READY, extent=0.10, spacing=0.10, step=0.05)
    assert [list(np.round(o, 9)) for o in cmp.cube_offsets(0.10, 0.10)] == \
        [list(np.round(r['offset'], 9)) for r in rows]


def test_free_motion_is_the_same_for_both_and_a_pull_to_the_shoulder_halts_only_the_rule(
        arm, origin):
    fine = cmp.cube(arm, SOLVER, READY, [np.array([0.0, 0.10, 0.0]),
                                         np.array([0.0, 0.0, 0.0])], hold=0.5)
    for method in ('rule', 'solver'):
        first, second = fine[method]
        assert first.reached and not first.affected and not first.halted
        assert second.reached and second.max_error == (0.0, 0.0)
        assert first.peak_joint_speed < 2.0
    pull = cmp.cube(arm, SOLVER, READY, [np.array([-0.30, 0.0, 0.0])], hold=0.5)
    (rule,), (solver,) = pull['rule'], pull['solver']
    assert rule.halted and rule.affected and not rule.reached
    assert rule.final_error[0] > 0.05
    # The solver damps there (and is slow to settle) but gets to the target.
    assert solver.affected and not solver.joint_limit and not solver.halted
    assert solver.reached and solver.final_error[0] < 0.005


def test_rule_motion_stops_a_joint_at_its_margin_and_counts_it(arm, origin):
    # A turn about -y of 55 deg runs a joint into its limit under the rule as well.
    target = cmp.pose_at(origin, (0, 0, 0), (0.0, -math.radians(55.0), 0.0))
    run_ = cmp.rule_motion(arm.chain, READY, [target] * 100, 0.02, 0.25, 1.0, 100.0, 200.0, 2.0)
    assert run_.joint_limit or not run_.reached
    assert not run_.halted


def test_confine_turns_a_motion_back_that_would_leave_the_box(origin):
    inside = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.array_equal(cmp.confine(inside, origin, origin, 1.0, 0.30, 1.0), inside)
    far = cmp.pose_at(origin, (0.28, 0.0, 0.0), (0, 0, 0))
    out = cmp.confine(np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0]), far, origin, 1.0, 0.30, 1.0)
    assert out[0] == pytest.approx(-0.2) and out[1] == pytest.approx(0.1)
    turned = cmp.pose_at(origin, (0, 0, 0), (0.0, 0.0, 1.0))
    out = cmp.confine(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5]), turned, origin, 1.0, 0.30, 1.1)
    assert out[5] == pytest.approx(-0.5)                # pointed back toward the start


def test_sessions_small_count_is_reproducible_and_inside_the_joint_speed(arm):
    first = cmp.sessions(arm, SOLVER, READY, count=2, segments=3, seed=7)
    second = cmp.sessions(arm, SOLVER, READY, count=2, segments=3, seed=7)
    assert len(first['runs']) == 6 and first['runs'] == second['runs']
    assert max(r.peak_joint_speed for r in first['runs']) <= 0.95 * 2.0 + 1e-9
    assert 0 <= first['sessions_with_joint_limit'] <= 2
    text = cmp.format_sessions(first, {'outcomes': {'free': 0.6, 'slowed': 0.3, 'stalled': 0.05,
                                                    'halted': 0.01, 'joint_limit': 0.01},
                                       'sessions_halted': 0.27})
    for wanted in ('2 sessions of 3', 'motions unaffected', '60.0 %', 'sessions with any halt',
                   '27.0 %', 'sessions with the joint-limit damper active',
                   'DLS peak joint speed', 'motions with a joint above 0.5 rad/s'):
        assert wanted in text, wanted


def rows_for(engaged_spans, joints, targets):
    rows = []
    for t, q in joints:
        rows.append({'t': t, 'k': 'joints', 'q': q})
    for start, end in engaged_spans:
        for t in np.arange(start, end + 1e-9, 0.02):
            rows.append({'t': float(t), 'k': 'state', 'engaged': True})
        rows.append({'t': end + 0.02, 'k': 'state', 'engaged': False})
    rows += targets
    return sorted(rows, key=lambda r: (r['t'], r['k'] != 'state'))


def test_reader_cuts_engagements_and_seeds_from_the_nearest_joint_row():
    joints = [(1.0, [1.0] * 6), (1.995, [2.0] * 6), (2.004, [3.0] * 6), (5.0, [4.0] * 6),
              (5.5, [5.0] * 6)]
    targets = [{'t': 2.0 + 0.02 * k, 'k': 'target', 'p': [0.0, 0.0, k], 'q': [0, 0, 0, 1]}
               for k in range(1, 4)]
    targets.append({'t': 3.5, 'k': 'target', 'p': [9.0, 0, 0], 'q': [0, 0, 0, 1]})   # released
    rows = rows_for([(2.0, 2.1), (5.45, 5.55)], joints, targets)
    first, second = cmp.read_engagements(rows)
    assert first.q == [3.0] * 6                     # 4 ms after is nearer than 5 ms before
    assert [t[1][2] for t in first.targets] == [1, 2, 3]
    assert second.q == [5.0] * 6 and second.targets == []


def test_reader_without_joint_rows_before_takes_the_first_one_after():
    rows = [{'t': 1.0, 'k': 'state', 'engaged': True}, {'t': 1.01, 'k': 'joints', 'q': [7.0] * 6},
            {'t': 1.02, 'k': 'state', 'engaged': False}]
    (engagement,) = cmp.read_engagements(rows)
    assert engagement.q == [7.0] * 6
    (open_one,) = cmp.read_engagements([{'t': 1.0, 'k': 'joints', 'q': [1.0] * 6},
                                        {'t': 1.1, 'k': 'state', 'engaged': True}])
    assert open_one.q == [1.0] * 6 and open_one.end == 1.1       # still open at the end


def test_recorded_targets_repeat_across_a_gap_and_carry_the_feedforward(origin):
    def row(t, x):
        pose = cmp.pose_at(origin, (x, 0, 0), (0, 0, 0))
        return (t, list(pose[:3, 3]), list(q_from_matrix(pose[:3, :3])))
    engagement = cmp.Engagement(0.0, 1.0, READY,
                                [row(0.00, 0.0), row(0.02, 0.005), row(0.08, 0.020)])
    targets, ff = cmp.recorded_targets(engagement, 0.02)
    assert len(targets) == 1 + 1 + 3                  # the gap of 60 ms covers three cycles
    assert np.allclose(ff[0], 0.0) and ff[1][0] == pytest.approx(0.25)
    assert np.allclose(ff[2], 0.0) and np.allclose(ff[3], 0.0)
    assert ff[4][0] == pytest.approx(0.015 / 0.06)
    assert targets[2][0, 3] == targets[1][0, 3] == targets[3][0, 3]


def test_replay_follows_recorded_targets_and_reports_each_figure(arm, origin):
    poses = [cmp.pose_at(origin, (0.0, 0.002 * k, 0.0), (0, 0, 0)) for k in range(60)]
    targets = [(0.02 * k, list(p[:3, 3]), list(q_from_matrix(p[:3, :3])))
               for k, p in enumerate(poses)]
    engagement = cmp.Engagement(0.0, 1.2, list(READY), targets)
    short = cmp.Engagement(5.0, 5.0, list(READY), targets[:2])
    unseeded = cmp.Engagement(6.0, 7.0, None, targets)
    result = cmp.replay(arm, SOLVER, [engagement, short, unseeded], delay_ticks=8,
                        stale_ticks=3)
    (one,) = result['engagements']                    # the short and the unseeded are skipped
    assert one['cycles'] == 60 and one['seed_error'][0] < 1e-9
    assert one['solver_error_max'][0] < 0.01 and one['tool_error_max'][0] < 0.03
    assert one['damped_s'] == 0.0 and one['joint_limit_cycles'] == 0
    assert 0.0 < one['peak_joint_speed'] < 1.0 and one['conditioning_min'] > 0.5
    median, largest = cmp.seed_errors(arm, [engagement, short, unseeded])
    assert median == largest and median < 1e-9
    text = cmp.format_replay(result, (median, largest))
    for wanted in ('frame check', 'engagements replayed: 1', 'time damped: 0.0 s',
                   'minimum conditioning', 'max solver error', 'peak joint speed',
                   'joint-limit damper holding a joint: 0 cycles in 0 engagements',
                   'closest approach to a bound 0.892 rad', 'in both directions: 0'):
        assert wanted in text, wanted
    assert 'no engagement' not in cmp.format_replay({'engagements': []})


def test_a_target_beyond_the_reach_leaves_the_solver_moving_and_a_reachable_one_does_not(arm,
                                                                                         origin):
    for x, moving in ((0.40, False), (0.95, True)):
        offset = np.array([x - origin[0, 3], 0.0, 0.0])
        run_ = cmp.cube(arm, SOLVER, READY, [offset], hold=1.0)['solver'][0]
        assert run_.moving_at_end is moving
        assert run_.reversed_fast is moving
    assert not cmp.moving_at_end(np.zeros((10, 6)), 0.02)
    chatter = np.zeros((10, 6))
    chatter[2, 1], chatter[6, 1] = 1.0, -1.0
    assert cmp.reversed_fast(chatter) and not cmp.reversed_fast(np.abs(chatter))
    slow = np.zeros((10, 6))
    slow[2, 0], slow[6, 0] = 0.4, -0.4
    assert not cmp.reversed_fast(slow)


def test_runs_report_the_closest_approach_to_a_joint_bound(arm, origin):
    run_ = cmp.cube(arm, SOLVER, READY, [np.array([0.0, 0.1, 0.0])], hold=0.2)['solver'][0]
    assert run_.min_margin == pytest.approx(0.892, abs=0.01)         # joint 3 at -0.7, bound 0.19
    held = cmp.pose_at(origin, (0, 0, 0), (0.0, -math.radians(55.0), 0.0))
    ff = cmp.twists([held] * 100, origin, 0.02)
    tilted = cmp.solver_motion(arm, SOLVER, READY, [held] * 100, ff)
    assert tilted.joint_limit and 0.14 < tilted.min_margin < 0.152


def test_cube_report_carries_both_columns_and_the_reference(arm):
    result = cmp.cube(arm, SOLVER, READY, [np.array([0.0, 0.1, 0.0])], hold=0.2)
    text = cmp.format_cube(result, {'17.0': 0.59, '30.0': 0.78, '100.0': 0.82,
                                    'complete': 0.83})
    for wanted in ('cube of 1 tool positions', '59.0 % / 78.0 % / 82.0 %',
                   'completed at all: 83.0 %', 'old rule (model)', 'DLS solver', 'share reached',
                   'peak joint speed', 'max position error'):
        assert wanted in text, wanted


def test_rows_are_streamed_and_filtered_by_kind():
    lines = ['{"t": 1.0, "k": "joints", "q": [0]}', '', '{"t": 2.0, "k": "input", "p": [0]}',
             '{"t": 3.0, "k": "state", "engaged": true}']
    assert [r['k'] for r in cmp.iter_rows(iter(lines))] == ['joints', 'input', 'state']
    assert [r['k'] for r in cmp.iter_rows(iter(lines), ('joints', 'state'))] == ['joints',
                                                                                 'state']
