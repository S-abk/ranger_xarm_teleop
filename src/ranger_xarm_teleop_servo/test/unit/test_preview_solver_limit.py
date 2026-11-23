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
The preview reports what the solver would do from the measured posture (motion disabled).

The link keep-out acts in the solver, not on the target, so a preview of the target alone
cannot show it. The node solves from the measured joints toward the previewed target (the
arm stands still; nothing is commanded) and reports the position-type limit. These tests use
the same solver and keep-out model on the made-up robot: a posture held at the structure
reports the link keep-out toward it and nothing away from it; a far target at a free posture
reports nothing, although the solver capped its speed.
"""

from keepout_robot import READY, resolve, robot_urdf
import numpy as np
import pytest
from ranger_xarm_teleop_servo import keepout as ko
from ranger_xarm_teleop_servo import preview
from ranger_xarm_teleop_servo import solver as sv
from ranger_xarm_teleop_servo.gate import Limit

MARGIN = 0.025
PARAMS = sv.SolverParams(linear_speed_cap=0.05, angular_speed_cap=0.2)
# Pulled back: link 2 near the screen behind the arm (as in test_keepout_solver).
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


def _reported(arm, model, extra, q, target):
    solution = sv.solve(arm, q, target, PARAMS, keepout=model, extra=extra)
    return preview.solver_limit(solution.fixed, solution.keepout_bound), solution


def _at_the_screen(arm, model, extra):
    """Return a posture with link 2 held at the margin by the screen (the arm pulled back)."""
    q = np.array(NEAR_SCREEN)
    target = _moved(sv.tool_pose(arm, q), dx=-0.3)
    for _ in range(600):
        q = q + sv.solve(arm, q, target, PARAMS, keepout=model, extra=extra).qdot * PARAMS.period
    clearance, pair = model.clearance(q, extra)
    assert pair == ('l2', 'screen') and clearance < MARGIN + 0.002
    return q


def test_toward_the_structure_from_a_held_posture_the_link_keep_out_is_reported(robot):
    arm, model, extra = robot
    q = _at_the_screen(arm, model, extra)
    limit, _ = _reported(arm, model, extra, q, _moved(sv.tool_pose(arm, q), dx=-0.10))
    assert limit == Limit.LINK_KEEPOUT


def test_away_from_the_structure_nothing_is_reported(robot):
    arm, model, extra = robot
    q = _at_the_screen(arm, model, extra)
    limit, _ = _reported(arm, model, extra, q, _moved(sv.tool_pose(arm, q), dx=0.10))
    assert limit == Limit.NONE


def test_a_far_target_at_a_free_posture_reports_no_speed_limit(robot):
    """The solver caps the speed toward a far target; the preview does not report it."""
    arm, model, extra = robot
    q = np.array(READY)
    limit, solution = _reported(arm, model, extra, q, _moved(sv.tool_pose(arm, q), dz=0.10))
    assert solution.limit in (sv.LIMIT_TOOL_SPEED, sv.LIMIT_JOINT_VELOCITY)
    assert limit == Limit.NONE


def test_a_joint_at_its_position_bound_is_reported(robot):
    """Joint 3 within the solver's margin of its upper bound, the target asking past it."""
    arm, model, extra = robot
    q = np.array(READY)
    q[2] = arm.upper[2] - 0.05
    tool = sv.tool_pose(arm, q)
    # Of the six axis directions, those that would drive joint 3 into its bound are reported.
    seen = 0
    for d in np.vstack([np.eye(3), -np.eye(3)]):
        limit, solution = _reported(arm, model, extra, q, _moved(tool, *(0.10 * d)))
        if solution.fixed:
            assert limit == Limit.JOINT_POSITION
            seen += 1
    assert seen, 'no direction drove joint 3 into its bound'
