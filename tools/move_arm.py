#!/usr/bin/env python3
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
Move the simulated arm to a joint pose through a trajectory controller's action.

    tools/move_arm.py [--controller xarm6_traj_controller] [--seconds 4] q1 q2 q3 q4 q5 q6

Angles in radians. Used to put the arm in a posture with room to move before a teleop
run. Needs a sourced workspace. **Simulation only:** it reads ``/robot_description`` and
sends nothing unless every ros2_control hardware plugin in it is a simulator (exit
status 3 otherwise). On a robot, move the arm through MoveIt. The goal is rejected while teleop owns the arm (the
MoveIt controller is then inactive); that rejection is what the command mux relies on. A
rejection is printed with its reason; ``tools/recover_arm.py`` disarms and then moves.
Exit status 0 only when the goal was accepted and succeeded.
"""

import argparse
import os
import sys

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'src', 'ranger_xarm_teleop_servo'))
from ranger_xarm_teleop_servo.session import READY_POSTURE, rejection_reason  # noqa: E402
from ranger_xarm_teleop_servo.simcheck import NotASimulator, require_simulator  # noqa: E402

READY = list(READY_POSTURE)


def main() -> int:
    """Send one joint goal and report the result."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--controller', default='xarm6_traj_controller')
    parser.add_argument('--seconds', type=float, default=4.0)
    parser.add_argument('joints', nargs='*', type=float, default=READY)
    args = parser.parse_args()
    if len(args.joints) != 6:
        parser.error('six joint angles are needed')
    rclpy.init()
    node = rclpy.create_node('move_arm')
    client = ActionClient(node, FollowJointTrajectory,
                          f'/{args.controller}/follow_joint_trajectory')
    code = 1
    try:
        description = []
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, '/robot_description',
                                 lambda msg: description.append(msg.data), latched)
        # Who owns the arm, to explain a rejected goal (listening only; nothing is requested).
        owner = []
        try:
            from ranger_xarm_teleop_interfaces.msg import ArmOwner
            node.create_subscription(ArmOwner, '/teleop/mux/owner',
                                     lambda msg: owner.append(msg.owner), 10)
        except ImportError:
            pass
        for _ in range(100):
            if description:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
        try:
            require_simulator(description[0] if description else '')
        except NotASimulator as err:
            print(f'REFUSED, nothing sent: {err}')
            return 3
        if not client.wait_for_server(10.0):
            print('no action server')
            return 2
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [f'xarm_joint{i}' for i in range(1, 7)]
        seconds = int(args.seconds)
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=list(args.joints), velocities=[0.0] * 6,
            time_from_start=Duration(sec=seconds,
                                     nanosec=int((args.seconds - seconds) * 1e9)))]
        sent = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, sent, timeout_sec=10.0)
        handle = sent.result()
        if handle is None or not handle.accepted:
            for _ in range(20):         # the mux reports its owner several times a second
                if owner:
                    break
                rclpy.spin_once(node, timeout_sec=0.1)
            print(f'goal REJECTED: {rejection_reason(owner[-1] if owner else None)}')
            return 1
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result, timeout_sec=args.seconds + 30.0)
        done = result.result()
        ok = done is not None and done.result.error_code == 0 and done.status == 4
        print('goal succeeded' if ok else f'goal FAILED: {done}')
        code = 0 if ok else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
