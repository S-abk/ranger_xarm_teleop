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
run. Needs a sourced workspace. The goal is rejected while teleop owns the arm (the
MoveIt controller is then inactive); that rejection is what the command mux relies on.
Exit status 0 only when the goal was accepted and succeeded.
"""

import argparse
import sys

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint

READY = [0.0, -0.6, -0.7, 0.0, 1.3, 0.0]


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
            print('goal REJECTED')
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
