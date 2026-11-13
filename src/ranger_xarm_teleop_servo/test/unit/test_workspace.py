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
