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
Stand-in for the vendor driver's /xarm_xarm/robot_states, for the hardware-shaped mock.

Publishes xarm_msgs/RobotMsg at 5 Hz: err 0, warn 0, mode 1 (servo), state 2 (ready), every
axis braked and enabled (mt_brake and mt_able 63), the arm angles from /joint_states when
they are known (empty otherwise), the header stamped now. err, mode and state are parameters,
so a test can set them. Nothing here is measured: it reports what it is told to.
"""

from ranger_xarm_teleop_hardware import hwmock
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from xarm_msgs.msg import RobotMsg

JOINT_STATES_DEPTH = 10
PUBLISH_DEPTH = 10


class RobotStatesStandin(Node):
    """The arm-state publisher."""

    def __init__(self):
        super().__init__('robot_states_standin')
        self.declare_parameter('err', 0)
        self.declare_parameter('mode', hwmock.SERVO_MODE)
        self.declare_parameter('state', hwmock.READY_STATE)
        self.declare_parameter('publish_rate_hz', hwmock.ROBOT_STATES_RATE_HZ)
        rate = float(self.get_parameter('publish_rate_hz').value)
        if rate <= 0.0:
            raise ValueError('publish_rate_hz must be > 0')
        self._angles = []
        self._publisher = self.create_publisher(
            RobotMsg, hwmock.ROBOT_STATES_TOPIC, PUBLISH_DEPTH)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states,
                                 JOINT_STATES_DEPTH)
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().warning(
            'hwmock arm-state stand-in: %s is a stand-in, not the vendor driver; it reports '
            'the err, mode and state it is given.' % hwmock.ROBOT_STATES_TOPIC)

    def _on_joint_states(self, msg):
        angles = hwmock.arm_angles(msg.name, msg.position)
        if angles:
            self._angles = angles

    def _publish(self):
        fields = hwmock.robot_state_fields(
            self.get_parameter('err').value, self.get_parameter('mode').value,
            self.get_parameter('state').value, self._angles)
        msg = RobotMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.err = fields['err']
        msg.warn = fields['warn']
        msg.mode = fields['mode']
        msg.state = fields['state']
        msg.mt_brake = fields['mt_brake']
        msg.mt_able = fields['mt_able']
        msg.angle = fields['angle']
        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotStatesStandin()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
