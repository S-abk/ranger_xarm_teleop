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
"""INV-4: the singularity measure computed from joint angles, as MoveIt Servo defines it."""

import math

from arm_urdf import arm_urdf, READY
import numpy as np
import pytest
from ranger_xarm_teleop_servo import kinematics as kin


@pytest.fixture(scope='module')
def chain():
    return kin.chain_from_urdf(arm_urdf())


def test_chain_has_the_six_joints_and_their_limits(chain):
    assert chain.joint_names == [f'xarm_joint{i}' for i in range(1, 7)]
    assert len(chain.segments) == 7
    assert chain.joints[2].lower == -3.927 and chain.joints[2].upper == 0.19198
    assert chain.joints[0].velocity == 3.14


def test_ready_posture_puts_the_tool_in_front_pointing_down(chain):
    tip, axes, points = kin.forward(chain, READY)
    assert np.allclose(tip[:3, 3], [0.310, 0.0, 0.246], atol=2e-3)
    assert np.allclose(tip[:3, 2], [0, 0, -1], atol=1e-3)
    assert axes.shape == (6, 3) and points.shape == (6, 3)


@pytest.mark.parametrize('seed', range(5))
def test_jacobian_matches_finite_differences(chain, seed):
    q = np.random.default_rng(seed).uniform(-1.2, 1.2, 6)
    jac = kin.jacobian(chain, q)
    eps = 1e-6
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = eps
        ahead, _, _ = kin.forward(chain, q + delta)
        behind, _, _ = kin.forward(chain, q - delta)
        assert np.allclose((ahead[:3, 3] - behind[:3, 3]) / (2 * eps), jac[:3, i], atol=1e-6)
        turn = ahead[:3, :3] @ behind[:3, :3].T      # rotation between the two, base frame
        rate = np.array([turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0],
                         turn[1, 0] - turn[0, 1]]) / (4 * eps)
        assert np.allclose(rate, jac[3:, i], atol=1e-6)


@pytest.mark.parametrize('seed', range(3))
def test_jacobian_matches_an_independent_library(chain, seed):
    kdl = pytest.importorskip('PyKDL')
    arm = kdl.Chain()
    for segment in chain.segments:
        m = segment.origin
        frame = kdl.Frame(kdl.Rotation(*[float(v) for v in m[:3, :3].flatten()]),
                          kdl.Vector(*[float(v) for v in m[:3, 3]]))
        if segment.axis is None:
            arm.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.Fixed), frame))
            continue
        # KDL applies the joint first and the fixed frame after it: add them separately.
        arm.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.Fixed), frame))
        arm.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.RotZ), kdl.Frame()))
    q = np.random.default_rng(100 + seed).uniform(-1.2, 1.2, 6)
    angles = kdl.JntArray(6)
    for i in range(6):
        angles[i] = float(q[i])
    result = kdl.Jacobian(6)
    kdl.ChainJntToJacSolver(arm).JntToJac(angles, result)
    theirs = np.array([[result[r, c] for c in range(6)] for r in range(6)])
    assert np.allclose(theirs, kin.jacobian(chain, q), atol=1e-9)
    assert np.isclose(np.linalg.cond(theirs), kin.condition_number(chain, q), rtol=1e-9)


def test_measure_is_largest_over_smallest_singular_value(chain):
    values = kin.singular_values(chain, READY)
    assert list(values) == sorted(values, reverse=True)
    assert np.isclose(kin.condition_number(chain, READY), values[0] / values[-1])
    assert 9.0 < kin.condition_number(chain, READY) < 9.6


def test_measure_does_not_depend_on_the_first_joint(chain):
    turned = [1.1] + READY[1:]
    assert np.isclose(kin.condition_number(chain, turned), kin.condition_number(chain, READY))


def test_wrist_singularity_is_off_the_aligned_posture_and_is_named(chain):
    # Axes 4 and 6 are parallel at joint 5 = 0 but 76 mm apart, so the arm is not singular
    # there; the measure is already past MoveIt Servo's default hard stop (30) though.
    def measure(q5):
        return kin.condition_number(chain, READY[:4] + [q5, 0.0])

    values = [measure(q5) for q5 in (1.3, 0.8, 0.4, 0.2, 0.1, 0.0, -0.15, -0.25, -0.3)]
    assert values == sorted(values)
    assert 16.5 < measure(0.4) < 17.5 and 30.0 < measure(0.1) < 31.0
    assert 40.0 < measure(0.0) < 41.0 and measure(-0.3) > 500.0
    shape = kin.resemblance(chain, READY[:4] + [0.1, 0.0])
    assert shape.wrist_deg == pytest.approx(math.degrees(0.1), abs=0.01)
    assert shape.kind == 'wrist' and shape.label() == 'wrist' and shape.label(40.0) == ''
    assert shape.wrist_share > 0.9


def test_a_well_conditioned_posture_resembles_nothing(chain):
    assert kin.resemblance(chain, READY).label() == ''
    assert kin.resemblance(chain, READY).condition == pytest.approx(
        kin.condition_number(chain, READY))


def test_stretched_arm_is_named_elbow(chain):
    # Joint 3 at the angle where the forearm continues the upper arm.
    best = min(np.linspace(-3.2, -2.2, 4000),
               key=lambda q3: kin.resemblance(chain, [0, 0.3, q3, 0, 1.0, 0]).elbow_deg)
    shape = kin.resemblance(chain, [0, 0.3, float(best), 0, 1.0, 0])
    assert shape.elbow_deg < 0.2 and shape.condition > 1000.0
    assert shape.kind == 'elbow' and shape.elbow_share > 0.9
    near = kin.resemblance(chain, [0, 0.3, float(best) + 0.15, 0, 1.0, 0])
    assert near.label() == 'elbow' and 17.0 < near.condition < 1000.0


def test_wrist_over_the_first_axis_is_named_shoulder(chain):
    values = [kin.condition_number(chain, [0, q2, -1.2, 0, 1.0, 0])
              for q2 in (-0.6, -0.8, -1.0, -1.1, -1.15, -1.17)]
    assert values == sorted(values) and values[-1] > 500.0
    shape = kin.resemblance(chain, [0, -1.15, -1.2, 0, 1.0, 0])
    assert shape.kind == 'shoulder' and shape.shoulder_share > 0.25
    assert shape.shoulder_m < 0.06


def test_measure_depends_on_the_tool_length(chain):
    # Metres and radians are mixed, so a longer tool changes the measure of one posture.
    flange = kin.chain_from_urdf(arm_urdf(tool='0 0 0'))
    assert abs(kin.condition_number(flange, READY) - kin.condition_number(chain, READY)) > 0.3


def test_nearest_limit_and_margin(chain):
    name, margin = kin.limit_margin(chain, READY)
    assert name == 'xarm_joint3' and margin == pytest.approx(0.19198 + 0.7)
    name, margin = kin.limit_margin(chain, [0, 2.0, -0.7, 0, 1.3, 0])
    assert name == 'xarm_joint2' and margin == pytest.approx(0.0944)


def test_bad_descriptions_are_refused(chain):
    with pytest.raises(kin.ChainError):
        kin.chain_from_urdf('not xml <')
    with pytest.raises(kin.ChainError):
        kin.chain_from_urdf(arm_urdf(), tip='nowhere')
    with pytest.raises(kin.ChainError):
        kin.chain_from_urdf(arm_urdf().replace('type="revolute"', 'type="prismatic"', 1))
    with pytest.raises(ValueError):
        kin.forward(chain, [0.0] * 5)
