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
Stand-in for the robot workspace's gripper action proxy, for the hardware-shaped mock.

Same interface as the workspace proxy: the action /xarm_gripper/gripper_action
(control_msgs/GripperCommand) and sensor_msgs/JointState on /joint_states carrying only
xarm_drive_joint. There is no native gripper behind it: every goal is accepted and the joint
moves to the clamped goal position at a fixed speed, with position-only feedback. The start
position is assumed (a parameter), as in the workspace proxy; nothing here is measured.
"""

import math
import threading
import time

from control_msgs.action import GripperCommand
from ranger_xarm_teleop_bringup.hwmock import (
    clamp_position, GRIPPER_JOINT, GRIPPER_MAX, GRIPPER_MIN, step_toward)
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

MOTION_PERIOD_S = 0.02


class HwmockGripperProxy(Node):
    """The gripper action server and joint-state publisher."""

    def __init__(self):
        super().__init__('xarm_gripper_action_proxy')
        self.declare_parameter('proxy_action_name', '/xarm_gripper/gripper_action')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('joint_name', GRIPPER_JOINT)
        self.declare_parameter('initial_position', 0.0)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('speed_rad_s', 1.0)
        self.declare_parameter('model_state_min', GRIPPER_MIN)
        self.declare_parameter('model_state_max', GRIPPER_MAX)

        self._joint_name = str(self.get_parameter('joint_name').value)
        self._min = float(self.get_parameter('model_state_min').value)
        self._max = float(self.get_parameter('model_state_max').value)
        self._speed = float(self.get_parameter('speed_rad_s').value)
        rate = float(self.get_parameter('publish_rate_hz').value)
        initial = float(self.get_parameter('initial_position').value)
        if not math.isfinite(initial):
            raise ValueError('initial_position must be finite')
        if rate <= 0.0:
            raise ValueError('publish_rate_hz must be > 0')
        if not self._min < self._max:
            raise ValueError('model_state_min must be < model_state_max')
        if self._speed <= 0.0:
            raise ValueError('speed_rad_s must be > 0')

        self._lock = threading.Lock()
        self._position = clamp_position(initial, self._min, self._max)
        self._goal_serial = 0

        group = ReentrantCallbackGroup()
        self._publisher = self.create_publisher(
            JointState, str(self.get_parameter('joint_state_topic').value), 10)
        self._server = ActionServer(
            self, GripperCommand, str(self.get_parameter('proxy_action_name').value),
            execute_callback=self._execute, goal_callback=self._on_goal,
            cancel_callback=self._on_cancel, callback_group=group)
        self.create_timer(1.0 / rate, self._publish, callback_group=group)
        self._publish()
        self.get_logger().warning(
            'hwmock gripper stand-in: initial position is assumed, not measured: '
            '%s=%.4f rad. No native gripper.' % (self._joint_name, self._position))

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self._joint_name]
        with self._lock:
            msg.position = [self._position]
        self._publisher.publish(msg)

    def _on_goal(self, request):
        self.get_logger().info(
            'gripper goal: position=%.4f' % request.command.position)
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        with self._lock:
            self._goal_serial += 1
            serial = self._goal_serial
        target = clamp_position(goal_handle.request.command.position, self._min, self._max)
        last = time.monotonic()
        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._result(False)
            with self._lock:
                superseded = serial != self._goal_serial
                position = self._position
            if superseded:
                goal_handle.abort()
                return self._result(False)
            if position == target:
                break
            now = time.monotonic()
            position = step_toward(position, target, self._speed, now - last)
            last = now
            with self._lock:
                self._position = position
            feedback = GripperCommand.Feedback()
            feedback.position = position
            goal_handle.publish_feedback(feedback)
            time.sleep(MOTION_PERIOD_S)
        goal_handle.succeed()
        return self._result(True)

    def _result(self, reached):
        result = GripperCommand.Result()
        with self._lock:
            result.position = self._position
        result.reached_goal = reached
        return result


def main(args=None):
    rclpy.init(args=args)
    node = HwmockGripperProxy()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
