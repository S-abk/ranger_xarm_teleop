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
Two publishers on /joint_states, as on a real bring-up.

The joint state broadcaster publishes the arm (150 Hz); a gripper proxy publishes only the
gripper joint (10 Hz). A gripper-only message must neither take the arm's measurement away nor
make an old arm sample look fresh.
"""

from dataclasses import replace
import math
import os
import sys

import rclpy
import rclpy.duration
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_teleop_node import JOINTS, LIMITS, overrides, READY, ros, TeleopNode  # noqa: E402, I100
from arm_urdf import arm_urdf  # noqa: E402, I100

__all__ = ['ros']        # the module-scoped ROS fixture of the node tests

GRIPPER = 'xarm_drive_joint'


def message(node, names, positions, age=0.0):
    msg = JointState()
    msg.header.stamp = (node.get_clock().now()
                        - rclpy.duration.Duration(seconds=age)).to_msg()
    msg.name = list(names)
    msg.position = [float(p) for p in positions]
    return msg


def make_node():
    node = TeleopNode(node_name='teleop_joint_sources',
                      parameter_overrides=overrides(LIMITS))
    node._on_tf_static(TFMessage(transforms=[]))
    node._on_robot_description(String(data=arm_urdf()))
    assert node.arm is not None
    # Gripper teleop stays off; the opening is still measured (and reported) by joint name.
    node.gripper_params = replace(node.gripper_params, joint=GRIPPER)
    return node


def test_a_gripper_only_message_does_not_take_the_arm_measurement_away(ros):
    node = make_node()
    try:
        now = [100.0]
        node.mono = lambda: now[0]
        missing = []
        for i in range(150):            # 1 s: arm at 150 Hz, gripper at 10 Hz in between
            now[0] += 1.0 / 150.0
            node._on_joint_state(message(node, JOINTS, READY))
            if i % 15 == 7:
                node._on_joint_state(message(node, [GRIPPER], [0.3]))
            joints, pose, age = node.measured()
            if pose is None or not age < 0.02:
                missing.append(i)
        assert not missing, f'no fresh arm measurement after cycles {missing[:5]}'
        assert node.joint_state[2][GRIPPER] == 0.3
    finally:
        node.close()


def test_a_gripper_only_message_does_not_make_an_old_arm_sample_fresh(ros):
    node = make_node()
    try:
        node.mono = lambda: 100.0
        node._on_joint_state(message(node, JOINTS, READY, age=0.5))
        node.mono = lambda: 100.5          # the arm went quiet; the gripper proxy did not
        node._on_joint_state(message(node, [GRIPPER], [0.2]))
        _, pose, age = node.measured()
        assert pose is not None and age >= 0.5
    finally:
        node.close()


def test_the_gripper_opening_carries_the_stamp_of_its_own_message(ros):
    node = make_node()
    try:
        node.mono = lambda: 100.0
        arm = message(node, JOINTS, READY, age=0.004)
        node._on_joint_state(arm)
        grip = message(node, [GRIPPER], [0.25], age=0.001)
        node._on_joint_state(grip)
        value, stamp = node.gripper_measured()
        assert value == 0.25
        assert stamp == grip.header.stamp.sec * 1_000_000_000 + grip.header.stamp.nanosec
    finally:
        node.close()


def test_a_gripper_message_before_the_first_arm_sample_is_kept(ros):
    node = make_node()
    try:
        node.mono = lambda: 100.0
        node._on_joint_state(message(node, [GRIPPER], [0.1]))
        assert node.joint_state is None
        assert node.measured() == (None, None, math.inf)
        node._on_joint_state(message(node, JOINTS, READY))
        assert node.joint_state[2][GRIPPER] == 0.1
        assert node.measured()[1] is not None
    finally:
        node.close()


def test_one_message_with_arm_and_gripper_is_unchanged(ros):
    node = make_node()
    try:
        node.mono = lambda: 100.0
        both = message(node, JOINTS + [GRIPPER], READY + [0.4])
        node._on_joint_state(both)
        value, stamp = node.gripper_measured()
        assert value == 0.4 and stamp == node.joint_state[1]
    finally:
        node.close()
