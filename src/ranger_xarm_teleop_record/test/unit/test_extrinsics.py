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
"""Camera extrinsics from fixed joints and static transforms (no ROS)."""

import math

import pytest

from ranger_xarm_teleop_record.extrinsics import Frames, rpy_pose
from ranger_xarm_teleop_servo.rotations import Pose, q_from_yaw

URDF = """<robot name="r">
  <link name="world"/><link name="body"/><link name="arm_base"/><link name="l1"/>
  <link name="flange"/><link name="tcp"/><link name="cam_a_link"/><link name="cam_w_link"/>
  <joint name="j0" type="floating"><parent link="world"/><child link="body"/></joint>
  <joint name="mount" type="fixed"><parent link="body"/><child link="arm_base"/>
    <origin xyz="0.1 0 0.5"/></joint>
  <joint name="j1" type="revolute"><parent link="arm_base"/><child link="l1"/>
    <axis xyz="0 0 1"/></joint>
  <joint name="j2" type="fixed"><parent link="l1"/><child link="flange"/>
    <origin xyz="0 0 0.3"/></joint>
  <joint name="t" type="fixed"><parent link="flange"/><child link="tcp"/>
    <origin xyz="0 0 0.2"/></joint>
  <joint name="ca" type="fixed"><parent link="body"/><child link="cam_a_link"/>
    <origin xyz="-0.3 0 1.0" rpy="0 0.5 0"/></joint>
  <joint name="cw" type="fixed"><parent link="flange"/><child link="cam_w_link"/>
    <origin xyz="0.05 0 0.02" rpy="0 0 1.5707963"/></joint>
</robot>"""
OPTICAL = rpy_pose((0, 0, 0), (-math.pi / 2, 0, -math.pi / 2))


def frames():
    return Frames(URDF, [('cam_a_link', 'cam_a_optical', OPTICAL),
                         ('cam_w_link', 'cam_w_optical', OPTICAL)])


def test_rpy_matches_yaw():
    p = rpy_pose((1, 2, 3), (0, 0, 0.7))
    assert p.p == (1.0, 2.0, 3.0)
    assert p.q == pytest.approx(q_from_yaw(0.7))


def test_a_camera_on_the_body_rides_the_body_and_has_a_base_pose():
    e = frames().extrinsics('cam_a_optical', 'arm_base', 'tcp')
    assert e['mount_link'] == 'body'
    assert e['in_mount']['xyz'] == pytest.approx([-0.3, 0.0, 1.0])
    assert e['in_base']['xyz'] == pytest.approx([-0.4, 0.0, 0.5])
    assert e['in_tool'] is None                    # a joint lies between: not constant


def test_a_wrist_camera_rides_the_moving_link_and_has_a_tool_pose():
    e = frames().extrinsics('cam_w_optical', 'arm_base', 'tcp')
    assert e['mount_link'] == 'l1'
    assert e['in_mount']['xyz'] == pytest.approx([0.05, 0.0, 0.32])
    assert e['in_tool']['xyz'] == pytest.approx([0.05, 0.0, -0.18])
    assert e['in_base'] is None


def test_an_unknown_frame_is_reported_not_guessed():
    e = frames().extrinsics('nowhere', 'arm_base', 'tcp')
    assert e['known'] is False and e['mount_link'] is None and e['in_mount'] is None


def test_static_transform_replaces_nothing_in_the_description():
    f = Frames(URDF, [('tcp', 'extra', Pose((0, 0, 0.1), (0, 0, 0, 1)))])
    assert f.mount('extra') == 'l1'
