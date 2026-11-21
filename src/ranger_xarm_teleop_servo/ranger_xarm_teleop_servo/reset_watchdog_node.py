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
Reset watchdog node: cancels the reset's goal when the reset node falls silent (G-16).

A separate process from the reset node, started with the stack. It hears the reset node's
heartbeat (``/teleop/reset/heartbeat``, which names the goal while one is out), move_group's
goal status and the joint states. When the heartbeat is older than ``heartbeat_timeout_s``
while the named goal is unfinished, it cancels that goal by its own id and keeps cancelling
until the goal has ended and the joints are still (``reset_watchdog.py``). It never starts or
changes a motion and never cancels any other goal.

Its own sign of life, ``/teleop/reset/watchdog`` at 10 Hz, is sent only once its cancel path
has completed one round trip (a cancel request for an id no goal has), so that the reset
node, which sends no goal without a fresh one, knows the stop path works.
"""

import sys
import time
import uuid

from action_msgs.msg import GoalStatusArray
from action_msgs.srv import CancelGoal
from ranger_xarm_teleop_interfaces.msg import ResetHeartbeat, ResetWatchdog
from ranger_xarm_teleop_servo import reset_watchdog as rw
from ranger_xarm_teleop_servo.reset_node import HEARTBEAT_TOPIC, LIVENESS_QOS, WATCHDOG_TOPIC
from ranger_xarm_teleop_servo.session import ARM_JOINTS
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_action_status_default
from sensor_msgs.msg import JointState
from unique_identifier_msgs.msg import UUID

MOVE_ACTION = '/move_action'
CYCLE_S = 0.01              # the heartbeat's age is judged 100 times a second
ALIVE_PERIOD_S = 0.1        # the watchdog's own sign of life, 10 Hz
READY_ASK_S = 1.0           # the harmless round trip is asked again after this


class ResetWatchdogNode(Node):
    """The watchdog's ROS side; the rule is in ``reset_watchdog.Watchdog``."""

    def __init__(self, node_name: str = 'reset_watchdog', parameter_overrides=None) -> None:
        """Make the clients and subscriptions; start the cycle."""
        super().__init__(node_name, parameter_overrides=parameter_overrides or [])
        timeout = float(self.declare_parameter('heartbeat_timeout_s',
                                               rw.HEARTBEAT_TIMEOUT_S).value)
        names = list(self.declare_parameter('arm_joints', list(ARM_JOINTS)).value)
        self.dog = rw.Watchdog(names, heartbeat_timeout_s=timeout)
        self.create_subscription(ResetHeartbeat, HEARTBEAT_TOPIC, self._on_beat, LIVENESS_QOS)
        self.create_subscription(GoalStatusArray, f'{MOVE_ACTION}/_action/status',
                                 self._on_goal_status, qos_profile_action_status_default)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 50)
        self.cancel_client = self.create_client(CancelGoal, f'{MOVE_ACTION}/_action/cancel_goal')
        self.alive_pub = self.create_publisher(ResetWatchdog, WATCHDOG_TOPIC, LIVENESS_QOS)
        self.alive = False
        self.ready_asked = None
        self.alive_seq = 0
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(CYCLE_S, self._cycle, clock=steady)
        self.create_timer(ALIVE_PERIOD_S, self._publish_alive, clock=steady)
        self.get_logger().info(f'reset watchdog: cancels the reset goal when its heartbeat is '
                               f'older than {timeout * 1000.0:.0f} ms')

    def _on_beat(self, msg: ResetHeartbeat) -> None:
        self.dog.on_heartbeat(time.monotonic(), int(msg.seq), bool(msg.goal_out),
                              bytes(msg.goal_id))

    def _on_goal_status(self, msg: GoalStatusArray) -> None:
        now = time.monotonic()
        for entry in msg.status_list:
            self.dog.on_status(now, bytes(entry.goal_info.goal_id.uuid), int(entry.status))

    def _on_joints(self, msg: JointState) -> None:
        self.dog.on_joints(time.monotonic(), dict(zip(msg.name, msg.position)))

    def _check_ready(self, now: float) -> None:
        if self.alive or not self.cancel_client.service_is_ready():
            return
        if self.ready_asked is not None and now - self.ready_asked < READY_ASK_S:
            return
        request = CancelGoal.Request()
        request.goal_info.goal_id = UUID(uuid=list(uuid.uuid4().bytes))
        self.ready_asked = now
        self.get_logger().info("asking move_group's cancel path once (a goal id nobody has)")
        self.cancel_client.call_async(request).add_done_callback(self._ready_answered)

    def _ready_answered(self, future) -> None:
        if future.result() is not None and not self.alive:
            self.alive = True
            self.get_logger().info('the cancel path answers: watching the reset heartbeat')

    def _cycle(self) -> None:
        now = time.monotonic()
        self._check_ready(now)
        self.dog.tick(now)
        goal_id = self.dog.cancel_due(now)
        if goal_id is not None:
            request = CancelGoal.Request()
            request.goal_info.goal_id = UUID(uuid=list(goal_id))     # that goal only
            self.dog.cancel_sent(now)
            self.cancel_client.call_async(request).add_done_callback(self._cancel_answered)
        for line in self.dog.take_log():
            self.get_logger().warning('reset watchdog: ' + line)

    def _cancel_answered(self, future) -> None:
        answer = future.result()
        if answer is not None:
            self.dog.cancel_answered(time.monotonic(), int(answer.return_code),
                                     len(answer.goals_canceling))

    def _publish_alive(self) -> None:
        if not self.alive:
            return
        self.alive_seq += 1
        msg = ResetWatchdog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.seq = self.alive_seq
        msg.stopping = self.dog.stopping
        msg.goal_id = list(self.dog.stop_id or bytes(16))
        msg.stops = self.dog.stops
        msg.last_report = self.dog.last_report
        self.alive_pub.publish(msg)


def main(args=None) -> None:
    """Run the reset watchdog."""
    rclpy.init(args=args)
    node = ResetWatchdogNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
