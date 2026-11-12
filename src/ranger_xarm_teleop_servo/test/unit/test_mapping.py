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
"""INV-1: clutch-relative mapping in a fixed frame; wrist pivot; tool-to-tip conversion."""

import math

import numpy as np
import pytest
from ranger_xarm_teleop_servo.mapping import apply_pivot, clutch_relative, tool_to_tip
from ranger_xarm_teleop_servo.rotations import (compose, inverse, Pose, q_between, q_conj,
                                                q_from_axis_angle, q_from_yaw, q_mul,
                                                q_normalize, q_rotate, v_add, v_norm, v_sub)

TOOL0 = Pose((0.4, 0.1, 0.3), q_from_axis_angle((0.0, 1.0, 0.0), math.radians(90.0)))


def random_quat(rng):
    return q_normalize(rng.normal(size=4))


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_no_motion_gives_the_tool_pose_captured_at_engagement():
    rng = np.random.default_rng(1)
    for _ in range(20):
        clutch = Pose(tuple(rng.normal(size=3)), random_quat(rng))
        target = clutch_relative(clutch, clutch, rng.uniform(-3, 3), TOOL0)
        assert close(target.p, TOOL0.p)
        assert q_between(target.q, TOOL0.q) < 1e-9


@pytest.mark.parametrize('seed', range(8))
def test_mapping_does_not_depend_on_how_the_controller_is_held(seed):
    """The same room motion gives the same target for any controller start orientation."""
    rng = np.random.default_rng(seed)
    move = tuple(rng.normal(scale=0.1, size=3))
    turn = q_from_axis_angle(rng.normal(size=3), rng.uniform(0.1, 1.0))
    heading = rng.uniform(-math.pi, math.pi)
    results = []
    for _ in range(6):
        held = random_quat(rng)
        start = tuple(rng.normal(size=3))
        clutch = Pose(start, held)
        now = Pose(v_add(start, move), q_normalize(q_mul(turn, held)))
        results.append(clutch_relative(now, clutch, heading, TOOL0))
    for other in results[1:]:
        assert close(other.p, results[0].p)
        assert q_between(other.q, results[0].q) < 1e-9
    # And it is the room motion rotated by the calibration, applied to the tool pose.
    cal = q_from_yaw(heading)
    assert close(results[0].p, v_add(TOOL0.p, q_rotate(cal, move)))
    expected = q_mul(q_mul(q_mul(cal, turn), q_conj(cal)), TOOL0.q)
    assert q_between(results[0].q, expected) < 1e-9


def test_body_frame_formulation_would_depend_on_the_grip():
    """inverse(T_clutch) * T_now, which the contract forbids, differs between grips."""
    move = (0.1, 0.0, 0.0)
    answers = []
    for held in (q_from_yaw(0.0), q_from_yaw(math.radians(90.0))):
        clutch = Pose((0.0, 0.0, 1.0), held)
        now = Pose(v_add(clutch.p, move), held)
        answers.append(compose(inverse(clutch), now).p)
        fixed = clutch_relative(now, clutch, 0.0, TOOL0)
        assert close(v_sub(fixed.p, TOOL0.p), move)
    assert v_norm(v_sub(answers[0], answers[1])) > 0.1


def test_heading_rotates_room_directions_into_the_base_frame():
    clutch = Pose((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    now = Pose((0.1, 0.0, 1.0), q_from_yaw(math.radians(30.0)))
    target = clutch_relative(now, clutch, math.radians(90.0), TOOL0)
    assert close(v_sub(target.p, TOOL0.p), (0.0, 0.1, 0.0))
    # A rotation about the vertical stays a rotation about the vertical.
    assert q_between(target.q, q_mul(q_from_yaw(math.radians(30.0)), TOOL0.q)) < 1e-9
    # A roll about room +x becomes a roll about base +y.
    roll = Pose(clutch.p, q_from_axis_angle((1.0, 0.0, 0.0), 0.4))
    rolled = clutch_relative(roll, clutch, math.radians(90.0), TOOL0)
    assert q_between(rolled.q, q_mul(q_from_axis_angle((0.0, 1.0, 0.0), 0.4), TOOL0.q)) < 1e-9


def test_wrist_pivot_offset_removes_the_translation_of_a_wrist_twist():
    offset = (-0.08, 0.0, -0.03)        # grip origin to the wrist pivot, controller frame
    held = q_from_axis_angle((0.2, 1.0, 0.1), 0.6)
    grip = (0.3, -0.2, 1.1)
    pivot_point = v_add(grip, q_rotate(held, offset))
    twist = q_from_axis_angle((1.0, 0.3, 0.0), math.radians(50.0))
    twisted_q = q_normalize(q_mul(twist, held))
    twisted_grip = v_sub(pivot_point, q_rotate(twisted_q, offset))   # rotation about the pivot
    clutch, now = Pose(grip, held), Pose(twisted_grip, twisted_q)

    with_offset = clutch_relative(apply_pivot(now, offset), apply_pivot(clutch, offset),
                                  0.0, TOOL0)
    assert close(with_offset.p, TOOL0.p, 1e-9)
    assert q_between(with_offset.q, q_mul(twist, TOOL0.q)) < 1e-9

    zero = (0.0, 0.0, 0.0)
    without = clutch_relative(apply_pivot(now, zero), apply_pivot(clutch, zero), 0.0, TOOL0)
    # Without the offset the tool follows the grip origin, which the twist moved.
    assert close(v_sub(without.p, TOOL0.p), v_sub(twisted_grip, grip), 1e-12)
    assert v_norm(v_sub(twisted_grip, grip)) > 0.02
    assert apply_pivot(now, zero) == now


def test_tool_frame_offset_from_the_tip_rotates_about_the_tool_point():
    """A target rotation moves the tip on a circle around the tool point."""
    tip_in_tool = Pose((0.0, 0.0, -0.12), q_from_yaw(0.3))
    tool = Pose((0.5, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0))
    tip0 = tool_to_tip(tool, tip_in_tool)
    assert close(tip0.p, (0.5, 0.0, 0.18))
    for angle in (0.2, 0.9, 1.5):
        turned = Pose(tool.p, q_from_axis_angle((0.0, 1.0, 0.0), angle))
        tip = tool_to_tip(turned, tip_in_tool)
        assert abs(v_norm(v_sub(tip.p, tool.p)) - 0.12) < 1e-12
        assert v_norm(v_sub(tip.p, tip0.p)) > 0.01
        # Converting back gives the tool point unchanged: the rotation is about it.
        assert close(compose(tip, inverse(tip_in_tool)).p, tool.p, 1e-12)
        assert close(tip.p, v_add(tool.p, q_rotate(turned.q, tip_in_tool.p)), 1e-12)
    # With the tool at the tip the conversion is the identity.
    assert tool_to_tip(tool, Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))) == tool
