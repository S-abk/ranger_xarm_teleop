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
Gripper relay: ``/teleop/gripper/command`` to the backend's gripper interface.

Mode ``trajectory``: one-point trajectories on ``/<controller>/joint_trajectory``, and the
controller's held reference from ``/<controller>/controller_state`` republished on
``/teleop/gripper/held`` (sensor_msgs/JointState, the controller state's stamp). The
controller ignores its topic while inactive, so nothing this relay sends reaches the
gripper unless the command mux gave the gripper to teleop. See ``gripper_relay.py``.
"""

from control_msgs.msg import JointTrajectoryControllerState
from ranger_xarm_teleop_interfaces.msg import GripperCommand
from ranger_xarm_teleop_servo.gripper_relay import check_mode, command_point, held_reference
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

COMMAND_TOPIC = '/teleop/gripper/command'
HELD_TOPIC = '/teleop/gripper/held'


class GripperRelay(Node):
    """Carries the teleop gripper command to the gripper's controller."""

    def __init__(self, node_name: str = 'gripper_relay') -> None:
        super().__init__(node_name)
        self.declare_parameter('mode', 'trajectory')
        self.declare_parameter('controller', 'teleop_gripper_controller')
        self.declare_parameter('joint', 'xarm_drive_joint')
        self.declare_parameter('period_s', 0.02)
        self.mode = str(self.get_parameter('mode').value)
        check_mode(self.mode)
        controller = str(self.get_parameter('controller').value).strip('/')
        self.joint = str(self.get_parameter('joint').value)
        self.period = float(self.get_parameter('period_s').value)
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, f'/{controller}/joint_trajectory', 10)
        self.held_pub = self.create_publisher(JointState, HELD_TOPIC, 10)
        self.create_subscription(GripperCommand, COMMAND_TOPIC, self._on_command, 10)
        self.create_subscription(JointTrajectoryControllerState,
                                 f'/{controller}/controller_state', self._on_state, 10)
        self.get_logger().info(f'gripper relay: {self.mode}, controller {controller}, '
                               f'joint {self.joint}')

    def _on_command(self, msg: GripperCommand) -> None:
        point = command_point(self.joint, msg.joint_name, msg.position, self.period)
        if point is None:
            self.get_logger().error(
                f'gripper command for {msg.joint_name!r} at {msg.position!r} not sent '
                f'(this relay drives {self.joint})', throttle_duration_sec=1.0)
            return
        out = JointTrajectory()
        out.joint_names = [self.joint]     # header stamp zero: start on arrival
        target = JointTrajectoryPoint()
        target.positions = [point[0]]
        target.time_from_start.sec, target.time_from_start.nanosec = point[1]
        out.points = [target]
        self.trajectory_pub.publish(out)

    def _on_state(self, msg: JointTrajectoryControllerState) -> None:
        value = held_reference(self.joint, msg.joint_names, msg.reference.positions)
        if value is None:
            return
        out = JointState()
        out.header.stamp = msg.header.stamp
        out.name = [self.joint]
        out.position = [value]
        self.held_pub.publish(out)


def main(args=None) -> None:
    """Run the gripper relay."""
    rclpy.init(args=args)
    node = GripperRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
