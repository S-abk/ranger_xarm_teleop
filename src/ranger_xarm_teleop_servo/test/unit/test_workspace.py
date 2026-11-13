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
"""INV-4: the workspace map: the singularity measure along tool motions from a posture."""

import math

from arm_urdf import arm_urdf, READY
import numpy as np
import pytest
from ranger_xarm_teleop_servo import kinematics as kin
from ranger_xarm_teleop_servo import workspace

CHAIN = kin.chain_from_urdf(arm_urdf())


def test_straight_walk_keeps_the_line_and_the_orientation():
    start, _, _ = kin.forward(CHAIN, READY)
    result = workspace.walk(CHAIN, READY, (0, 1, 0), 0.10)
    assert result.ended == 'complete' and len(result.samples) == 21
    for sample in result.samples:
        tip, _, _ = kin.forward(CHAIN, sample.q)
        assert np.allclose(tip[:3, 3], start[:3, 3] + [0, sample.travel, 0], atol=1e-6)
        assert np.allclose(tip[:3, :3], start[:3, :3], atol=1e-6)
        assert sample.condition == pytest.approx(kin.condition_number(CHAIN, sample.q))


def test_joint_speed_is_for_the_given_tool_speed():
    slow = workspace.walk(CHAIN, READY, (1, 0, 0), 0.02, speed=0.1).samples[-1]
    fast = workspace.walk(CHAIN, READY, (1, 0, 0), 0.02, speed=0.25).samples[-1]
    assert fast.joint_speed == pytest.approx(2.5 * slow.joint_speed)
    rates = np.linalg.pinv(kin.jacobian(CHAIN, READY)) @ [0.25, 0, 0, 0, 0, 0]
    assert workspace.walk(CHAIN, READY, (1, 0, 0), 0.0).samples[0].joint_speed == \
        pytest.approx(float(np.max(np.abs(rates))))


def test_pulling_the_tool_toward_the_base_meets_the_thresholds_before_30_cm():
    result = workspace.walk(CHAIN, READY, (-1, 0, 0), 0.30)
    assert 0.10 < result.reach(17.0) < result.reach(30.0) < 0.28
    assert result.worst > 100.0
    sideways = workspace.walk(CHAIN, READY, (0, 1, 0), 0.30)
    assert sideways.worst < 17.0 and sideways.reach(17.0) == pytest.approx(0.30)


def test_turn_in_place_keeps_the_tool_point_and_stops_at_a_joint_limit():
    start, _, _ = kin.forward(CHAIN, READY)
    result = workspace.walk(CHAIN, READY, (0, -1, 0), math.radians(60), turn=True)
    assert result.ended == 'joint_limit'
    assert result.samples[-1].margin < workspace.LIMIT_MARGIN_RAD
    assert math.radians(40) < result.samples[-1].travel < math.radians(60)
    tip, _, _ = kin.forward(CHAIN, result.samples[-1].q)
    assert np.allclose(tip[:3, 3], start[:3, 3], atol=1e-6)
    other = workspace.walk(CHAIN, READY, (0, 0, 1), math.radians(60), turn=True)
    assert other.ended == 'complete'
    assert all(s.condition == pytest.approx(other.samples[0].condition) for s in other.samples)


def test_out_of_reach_ends_the_walk():
    result = workspace.walk(CHAIN, READY, (1, 0, 0), 0.60)
    assert result.ended in ('unreachable', 'singular', 'joint_limit')
    assert result.samples[-1].travel < 0.60


def test_rays_and_grid_summary():
    rows = workspace.rays(CHAIN, READY, distance=0.10, angle=math.radians(10))
    assert [row['motion'] for row in rows][:2] == ['move +x', 'move -x'] and len(rows) == 12
    assert all(row['to_lower'] <= row['to_hard_stop'] <= row['asked'] + 1e-9 for row in rows)
    cells = workspace.grid(CHAIN, READY, extent=0.10, spacing=0.10, step=0.02)
    assert len(cells) == 27
    centre = [c for c in cells if c['offset'] == [0.0, 0.0, 0.0]][0]
    assert centre['condition'] == pytest.approx(kin.condition_number(CHAIN, READY))
    summary = workspace.grid_summary(cells)
    assert summary['points'] == 27 and sum(summary['ended'].values()) == 27
    assert summary['reached_below']['17.0'] <= summary['reached_below']['30.0'] <= 1.0


def test_evaluation_text_has_the_figures():
    text = workspace.format_evaluation(workspace.evaluate(CHAIN, READY, 0.10, 0.10))
    for wanted in ('condition number 9.3', 'move -x', 'turn +z', 'grid of 27 tool positions',
                   'nearest limit xarm_joint3'):
        assert wanted in text, wanted


def test_singularity_rule_scale_and_status():
    step = np.array([0.0, 0.001, 0, 0, 0, 0])
    assert workspace.servo_scale(CHAIN, READY, step, 17.0, 30.0) == (1.0, workspace.NO_WARNING)
    near = READY[:4] + [0.2, 0.0]                   # condition number 24.0
    measure = kin.condition_number(CHAIN, near)
    scale, status = workspace.servo_scale(CHAIN, near, step, 17.0, 30.0)
    toward = 1.0 - (measure - 17.0) / 13.0
    away = 1.0 - (measure - 17.0) / 26.0
    assert (status, scale) in ((workspace.APPROACHING, pytest.approx(toward)),
                               (workspace.LEAVING, pytest.approx(away)))
    beyond = READY[:4] + [0.05, 0.0]                # 34.6
    for direction in (step, -step):
        assert workspace.servo_scale(CHAIN, beyond, direction, 17.0, 30.0) == \
            (0.0, workspace.HALT)
    assert workspace.servo_scale(CHAIN, beyond, step, 40.0, 80.0) == (1.0, workspace.NO_WARNING)


def test_probe_step_cannot_tell_approaching_from_leaving_near_the_wrist_singularity():
    # The rule probes along the weakest task direction; here that step leaves the
    # condition number where it was, so the judgement has nothing to go on.
    q = np.array(READY[:4] + [0.15, 0.0])
    left, values, right = np.linalg.svd(kin.jacobian(CHAIN, q))
    step = right.T @ np.diag(1.0 / values) @ left.T @ (left[:, -1] * 0.01)
    here = kin.condition_number(CHAIN, q)
    assert abs(kin.condition_number(CHAIN, q + step) - here) < 0.002 * here
    assert abs(kin.condition_number(CHAIN, q - step) - here) < 0.002 * here
    # While the tilt that does change it is a direction the arm moves well in.
    assert kin.condition_number(CHAIN, READY[:4] + [0.10, 0.0]) > here + 3.0


def test_drive_free_slowed_and_halted():
    sideways = workspace.drive(CHAIN, READY, [0, 0.2, 0, 0, 0, 0], 1.0, 17.0, 30.0)
    assert sideways.outcome == 'free' and sideways.progress == pytest.approx(1.0)
    tip, _, _ = kin.forward(CHAIN, sideways.q)
    start, _, _ = kin.forward(CHAIN, READY)
    assert np.allclose(tip[:3, 3] - start[:3, 3], [0, 0.2, 0], atol=2e-3)
    tilt = [0, 0, 0, 0, -0.5, 0]            # toward the wrist singularity
    held = workspace.drive(CHAIN, READY[:4] + [0.5, 0.0], tilt, 2.0, 17.0, 30.0)
    assert held.outcome in ('stalled', 'halted', 'slowed') and held.progress < 0.7
    assert held.peak_condition < 31.0 and held.slowed_share > 0.3
    # Looser thresholds let the same motion through, past where 17 / 30 held it ...
    loose = workspace.drive(CHAIN, READY[:4] + [0.5, 0.0], tilt, 0.7, 100.0, 200.0)
    assert loose.outcome == 'free' and 40.0 < loose.peak_condition < 100.0
    # ... and hold it in turn at the singularity itself.
    further = workspace.drive(CHAIN, READY[:4] + [0.5, 0.0], tilt, 1.2, 100.0, 200.0)
    assert further.outcome == 'slowed' and 190.0 < further.peak_condition <= 200.5
    at_once = workspace.drive(CHAIN, READY[:4] + [0.05, 0.0], tilt, 1.0, 17.0, 30.0)
    assert at_once.outcome == 'halted' and at_once.progress == 0.0


def test_drive_respects_the_joint_speed_limit():
    fast = workspace.drive(CHAIN, READY, [0.25, 0, 0, 0, 0, 0], 0.2, 1e9, 2e9,
                           joint_speed_limit=0.5)
    assert fast.peak_joint_speed == pytest.approx(0.5) and fast.progress < 1.0


def test_survey_and_wander_are_reproducible_and_looser_thresholds_do_not_add_trouble():
    twists = workspace.sample_twists(5, seed=3)
    assert all(np.allclose(a, b) for a, b in zip(twists, workspace.sample_twists(5, seed=3)))
    assert all(np.linalg.norm(t[:3]) <= 0.25 + 1e-9 and np.linalg.norm(t[3:]) <= 1.0 + 1e-9
               for t in twists)
    tight = workspace.survey(CHAIN, READY, 17.0, 30.0, count=12)
    assert tight == workspace.survey(CHAIN, READY, 17.0, 30.0, count=12)
    assert sum(tight['outcomes'].values()) == pytest.approx(1.0)
    session = workspace.wander(CHAIN, READY, 17.0, 30.0, sessions=3, segments=6)
    loose = workspace.wander(CHAIN, READY, 100.0, 200.0, sessions=3, segments=6)
    assert sum(session['outcomes'].values()) == pytest.approx(1.0)
    assert loose['outcomes'].get('free', 0.0) >= session['outcomes'].get('free', 0.0)
    assert 0.0 <= session['sessions_halted'] <= 1.0
