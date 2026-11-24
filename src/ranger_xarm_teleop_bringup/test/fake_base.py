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
A stand-in for a drivable base in the mock stack: /cmd_vel in, /odom out (test only).

The mock stack has no base; the base node needs one to command and odometry to judge its rest
by. This follows the commanded body twist at a limited acceleration (1.0 m/s^2, 2.0 rad/s^2,
as the simulated base's wheels), stops on its own command timeout (0.5 s, as the simulated
base), integrates the pose, and publishes odometry at 100 Hz with an exact zero twist at rest.
It does not model the real base's one-mode-per-command steering: the base node already sends
one mode only.
"""

import math
import sys

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node

ACCEL = (1.0, 1.0, 2.0)
TIMEOUT_S = 0.5
PERIOD_S = 0.01


class FakeBase(Node):

    def __init__(self):
        super().__init__('fake_base')
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_at = -1e9
        self.v = [0.0, 0.0, 0.0]
        self.x = self.y = self.yaw = 0.0
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)
        self.pub = self.create_publisher(Odometry, '/odom', 50)
        self.create_timer(PERIOD_S, self._tick)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.cmd_at = self._now()

    def _tick(self):
        target = self.cmd if self._now() - self.cmd_at <= TIMEOUT_S else (0.0, 0.0, 0.0)
        for i in range(3):
            step = ACCEL[i] * PERIOD_S
            d = target[i] - self.v[i]
            self.v[i] = target[i] if abs(d) <= step else self.v[i] + math.copysign(step, d)
        vx, vy, wz = self.v
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * vx - s * vy) * PERIOD_S
        self.y += (s * vx + c * vy) * PERIOD_S
        self.yaw += wz * PERIOD_S
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id, msg.child_frame_id = 'odom', 'base_footprint'
        msg.pose.pose.position.x, msg.pose.pose.position.y = self.x, self.y
        msg.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        msg.twist.twist.linear.x, msg.twist.twist.linear.y = vx, vy
        msg.twist.twist.angular.z = wz
        self.pub.publish(msg)


def main():
    rclpy.init(args=sys.argv)
    node = FakeBase()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
