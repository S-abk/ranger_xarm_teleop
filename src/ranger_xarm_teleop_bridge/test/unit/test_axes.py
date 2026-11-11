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
"""INV-1: the axis conversion is one pure function with known vectors and rotations."""

import math

import numpy as np
import pytest

from ranger_xarm_teleop_bridge.axes import xr_pose_to_ros

# Basis change from the contract: x_ros = -z_xr, y_ros = -x_xr, z_ros = y_xr.
C = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)


def quat_to_mat(q):
    """Return the rotation matrix of a unit xyzw quaternion (hand-written, no scipy)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_q(axis, angle):
    """Return the unit xyzw quaternion for a rotation of ``angle`` about ``axis``."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    s = math.sin(angle / 2)
    return (a[0] * s, a[1] * s, a[2] * s, math.cos(angle / 2))


def test_basis_change_is_a_proper_rotation():
    assert np.allclose(C @ C.T, np.eye(3))
    assert math.isclose(np.linalg.det(C), 1.0)


@pytest.mark.parametrize('xr, ros', [
    ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),    # WebXR right  -> ROS -y (right)
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),     # WebXR up     -> ROS up
    ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),    # WebXR forward (-z) -> ROS forward (+x)
    ((0.3, -1.2, 2.5), (-2.5, -0.3, -1.2)),
])
def test_known_position_vectors(xr, ros):
    pos, quat = xr_pose_to_ros(xr, IDENTITY_Q)
    assert pos == pytest.approx(ros)
    assert quat == pytest.approx(IDENTITY_Q)
    assert pos == pytest.approx(tuple(C @ np.asarray(xr)))


def test_yaw_about_webxr_up_becomes_yaw_about_ros_up():
    q_xr = axis_angle_q((0, 1, 0), math.pi / 2)
    _, q_ros = xr_pose_to_ros((0, 0, 0), q_xr)
    assert q_ros == pytest.approx(axis_angle_q((0, 0, 1), math.pi / 2))
    # Rotating WebXR forward (-z) by +90 deg about +y gives WebXR -x; in ROS that is
    # +x (forward) rotated +90 deg about +z, i.e. +y (left).
    assert quat_to_mat(q_ros) @ np.array([1.0, 0, 0]) == pytest.approx([0, 1.0, 0])


def test_rotation_is_the_similarity_transform_c_r_ct():
    rng = np.random.default_rng(1234)
    for _ in range(50):
        q = rng.normal(size=4)
        q = tuple(q / np.linalg.norm(q))
        _, q_ros = xr_pose_to_ros((0, 0, 0), q)
        assert np.allclose(quat_to_mat(q_ros), C @ quat_to_mat(q) @ C.T, atol=1e-12)
        assert math.isclose(sum(c * c for c in q_ros), 1.0, abs_tol=1e-12)


def test_converted_pose_rotates_converted_vectors_consistently():
    """C (R_xr v) == R_ros (C v) for arbitrary v: the frame and its contents agree."""
    rng = np.random.default_rng(99)
    for _ in range(20):
        q = rng.normal(size=4)
        q = tuple(q / np.linalg.norm(q))
        v = rng.normal(size=3)
        _, q_ros = xr_pose_to_ros((0, 0, 0), q)
        assert np.allclose(C @ (quat_to_mat(q) @ v), quat_to_mat(q_ros) @ (C @ v))


def test_output_is_plain_floats():
    pos, quat = xr_pose_to_ros([1, 2, 3], [0, 0, 0, 1])
    assert all(type(v) is float for v in pos + quat)


def test_wrong_lengths_raise():
    with pytest.raises(ValueError):
        xr_pose_to_ros((1, 2), IDENTITY_Q)
    with pytest.raises(ValueError):
        xr_pose_to_ros((1, 2, 3), (0, 0, 1))
