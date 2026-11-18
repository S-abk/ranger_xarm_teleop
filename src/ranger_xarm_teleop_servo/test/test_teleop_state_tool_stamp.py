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
"""TeleopState.measured_tool_pose_stamp: the stamp of the joint state the pose came from."""

import math
import os
import sys

import rclpy
import rclpy.duration
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_teleop_node import (Harness, JOINTS, LIMITS, overrides, READY,  # noqa: E402, I100
                              ros, TeleopNode)
from arm_urdf import arm_urdf  # noqa: E402, I100

__all__ = ['ros']        # the module-scoped ROS fixture of the node tests


def ns(stamp):
    """Return a builtin_interfaces/Time in nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def test_measured_notes_the_stamp_of_the_joint_state_it_used(ros):
    node = TeleopNode(node_name='teleop_tool_stamp',
                      parameter_overrides=overrides({**LIMITS, 'motion_enabled': True}))
    try:
        node._on_tf_static(TFMessage(transforms=[]))
        node._on_robot_description(String(data=arm_urdf()))
        node.mono = lambda: 100.0
        positions = dict(zip(JOINTS, READY))
        stamp = node.get_clock().now() - rclpy.duration.Duration(seconds=0.01)
        node.joint_state = (100.0, stamp.nanoseconds, dict(positions))
        _, pose, _ = node.measured()
        assert pose is not None and node.measured_stamp_ns == stamp.nanoseconds
        # No measurement: no stamp either.
        node.joint_state = (100.0, 0, dict(positions))
        assert node.measured() == (None, None, math.inf)
        assert node.measured_stamp_ns == 0
    finally:
        node.close()


class _Stamped(Harness):
    """The ideal arm of the node tests, with a joint-state stamp per measurement."""

    def __init__(self, bridge):
        super().__init__(bridge)
        self.calls = 0
        self.node.measured = self._stamped

    def _stamped(self):
        self.calls += 1
        joints, pose, age = self._measured()
        # Real measured() sets this; a stamp unrelated to the cycle stamp, so the test can
        # tell them apart.
        self.node.measured_stamp_ns = 7_000_000_000 + self.calls * 1_000 if pose else 0
        return joints, pose, age


def test_each_published_state_carries_the_stamp_of_its_own_measurement(ros):
    h = _Stamped(ros)
    try:
        h.play('clutch_line')
        valid = [m for m in h.states if m.measured_tool_pose_valid]
        assert len(valid) > 100
        stamps = [ns(m.measured_tool_pose_stamp) for m in valid]
        assert all(s > 7_000_000_000 for s in stamps)
        assert len(set(stamps)) == len(stamps)            # one per cycle, never reused
        assert all(ns(m.measured_tool_pose_stamp) != ns(m.header.stamp) for m in valid)
        for m in h.states:
            if not m.measured_tool_pose_valid:
                assert ns(m.measured_tool_pose_stamp) == 0
    finally:
        h.close()
