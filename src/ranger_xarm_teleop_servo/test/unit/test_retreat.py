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
"""Retreat after a collision halt (INV-8): back along the recorded path, or nothing at all."""

from keepout_robot import READY, resolve, robot_urdf
import numpy as np
import pytest
from ranger_xarm_teleop_servo import keepout as ko
from ranger_xarm_teleop_servo import retreat
from ranger_xarm_teleop_servo import solver as sv

MARGIN = 0.025
PARAMS = sv.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0)
PADDING = 0.010          # move_group's 2 x 5 mm: its verdict, emulated with the model


@pytest.fixture(scope='module')
def robot():
    urdf = robot_urdf()
    model = ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN)
    arm = sv.arm_from_urdf(urdf, 'xarm_link_base', 'xarm_link_tcp')
    return arm, model, model.extra_values({'finger_left_joint': 0.02})


def _into_contact(arm, model, extra, start, dx, dy=0.0, dz=0.0):
    """Drive without the keep-out into the screen, as teleop did before; record the path."""
    recorder = retreat.PathRecorder()
    q = np.array(start, dtype=float)
    target = sv.tool_pose(arm, q).copy()
    target[:3, 3] += (dx, dy, dz)
    for k in range(400):
        recorder.add(k * 0.02, q)
        if model.clearance(q, extra)[0] < -0.003:
            break
        q = q + sv.solve(arm, q, target, PARAMS).qdot * 0.02
    assert model.clearance(q, extra)[0] < 0.0, 'the test path did not reach the contact'
    times, postures = recorder.path()
    return times, postures, postures[-1].copy()


def _plan(robot, times, postures, measured, **kwargs):
    arm, model, extra = robot
    options = dict(wanted=MARGIN, min_conditioning=0.2)
    options.update(kwargs)
    return retreat.plan(times, postures, measured,
                        clearance=lambda q: model.clearance(q, extra)[0],
                        tool_point=lambda q: sv.tool_pose(arm, q)[:3, 3],
                        valid=lambda q: model.clearance(q, extra)[0] > PADDING,
                        conditioning=lambda q: sv.levels(arm, q, PARAMS).orientation,
                        **options)


@pytest.mark.parametrize('turned', [-0.5, 0.0, 0.25])
def test_retreat_follows_the_path_back_to_the_first_clear_point(robot, turned):
    """Link 2 pulled into the screen with the arm turned three ways, as in the halts."""
    start = [turned] + list(READY[1:])
    dx, dy, dz = -0.32 * np.cos(turned), -0.32 * np.sin(turned), 0.0
    arm, model, extra = robot
    times, postures, halted = _into_contact(arm, model, extra, start, dx, dy, dz)
    plan = _plan(robot, times, postures, halted)
    assert plan.ok, plan.message
    # It starts where the arm is and only visits recorded postures, newest first.
    assert np.array_equal(plan.postures[0], halted)
    recorded = {tuple(q) for q in postures}
    assert all(tuple(q) in recorded for q in plan.postures[1:])
    indices = [next(i for i, r in enumerate(postures) if np.array_equal(r, q))
               for q in plan.postures[1:]]
    assert indices == sorted(indices, reverse=True) and np.all(np.diff(indices) == -1)
    # It stops at the first point that qualifies, and that point is clear and valid.
    stop = plan.postures[-1]
    assert model.clearance(stop, extra)[0] >= MARGIN
    assert model.clearance(stop, extra)[0] > PADDING
    assert sv.levels(arm, stop, PARAMS).orientation >= 0.2
    for q in plan.postures[1:-1]:
        assert not (model.clearance(q, extra)[0] >= MARGIN
                    and sv.levels(arm, q, PARAMS).orientation >= 0.2)
    assert plan.tool_travel <= retreat.MAX_TOOL_TRAVEL
    assert plan.joint_travel <= retreat.MAX_JOINT_TRAVEL
    # Slowly.
    joint, tool = retreat.speeds(plan.postures, plan.times,
                                 lambda q: sv.tool_pose(arm, q)[:3, 3])
    assert joint <= retreat.JOINT_SPEED + 1e-9 and tool <= retreat.TOOL_SPEED + 1e-9
    assert plan.times[0] == 0.0 and np.all(np.diff(plan.times) > 0.0)


def test_more_clearance_asked_goes_further_back(robot):
    arm, model, extra = robot
    times, postures, halted = _into_contact(arm, model, extra, READY, -0.30)
    near = _plan(robot, times, postures, halted)
    far = _plan(robot, times, postures, halted, wanted=0.05)
    assert near.ok and far.ok and len(far.postures) > len(near.postures)
    assert model.clearance(far.postures[-1], extra)[0] >= 0.05


def test_no_recorded_path_moves_nothing_and_says_what_to_do(robot):
    plan = _plan(robot, np.zeros(0), np.zeros((0, 6)), READY)
    assert not plan.ok and plan.postures is None
    assert 'no recorded path' in plan.message and 'Nothing moved' in plan.message
    assert 'Next:' in plan.message


def test_an_arm_moved_since_the_path_moves_nothing(robot):
    arm, model, extra = robot
    times, postures, halted = _into_contact(arm, model, extra, READY, -0.30)
    moved = halted + np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    plan = _plan(robot, times, postures, moved)
    assert not plan.ok and 'has moved since' in plan.message and plan.postures is None


def test_a_path_that_goes_deeper_on_the_way_back_is_not_followed(robot):
    arm, model, extra = robot
    times, postures, halted = _into_contact(arm, model, extra, READY, -0.30)
    deeper = postures.copy()
    deeper[-4] = deeper[-4] + np.array([0.0, -0.08, 0.0, 0.0, 0.0, 0.0])   # further back
    assert model.clearance(deeper[-4], extra)[0] < model.clearance(halted, extra)[0] - 0.001
    plan = _plan(robot, times, deeper, halted)
    assert not plan.ok and 'deeper' in plan.message and plan.postures is None


def test_no_clear_point_within_the_bounds_moves_nothing_and_gives_the_best(robot):
    arm, model, extra = robot
    times, postures, halted = _into_contact(arm, model, extra, READY, -0.30)
    plan = _plan(robot, times, postures, halted, max_tool_travel=0.003)
    assert not plan.ok and plan.postures is None
    assert 'the best is' in plan.message and 'Nothing moved' in plan.message
    assert 'Next:' in plan.message


def test_already_clear_moves_nothing(robot):
    arm, model, extra = robot
    times, postures, _ = _into_contact(arm, model, extra, READY, -0.30)
    plan = _plan(robot, times[:3], postures[:3], postures[2])
    assert not plan.ok and plan.message.startswith('already clear')


def test_recorder_is_decimated_timed_and_bounded():
    recorder = retreat.PathRecorder(keep_points=50)
    assert not recorder.add(0.0, [np.nan] * 6)
    q = np.zeros(6)
    added = 0
    for k in range(400):
        q[1] -= 0.0007
        added += recorder.add(k * 0.02, q)
    times, postures = recorder.path()
    assert len(postures) == 50 and added > 50              # the count bound
    assert np.all(np.max(np.abs(np.diff(postures, axis=0)), axis=1) >= retreat.POINT_SPACING)
    assert np.all(np.diff(times) > 0.0)
    recorder.add(400 * 0.02 + 61.0, q - 1.0)               # a minute later: the rest is old
    assert len(recorder.path()[0]) == 1 and recorder.dropped_for_age


def test_timing_respects_both_speeds_and_ramps():
    path = np.array([[0.0] * 6, [0.05] + [0.0] * 5, [0.1] + [0.0] * 5, [0.3] + [0.0] * 5])
    times = retreat.timing(path, lambda q: np.array([q[0] * 0.5, 0.0, 0.0]))
    joint, tool = retreat.speeds(path, times, lambda q: np.array([q[0] * 0.5, 0.0, 0.0]))
    assert joint <= retreat.JOINT_SPEED + 1e-9 and tool <= retreat.TOOL_SPEED + 1e-9
    assert times[-1] >= 0.3 / min(retreat.JOINT_SPEED, retreat.TOOL_SPEED / 0.5)


def test_distance_from_path():
    path = np.array([[0.0] * 6, [1.0] + [0.0] * 5])
    assert retreat.distance_from_path(path, [0.5, 0.01, 0, 0, 0, 0]) == pytest.approx(0.01)
    assert retreat.distance_from_path(path, [1.5, 0, 0, 0, 0, 0]) == pytest.approx(0.5)
