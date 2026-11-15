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

    tools/move_arm.py [--controller NAME] [--seconds 4] q1 q2 q3 q4 q5 q6

Angles in radians. Used to put the arm in a posture with room to move before a teleop
run. Needs a sourced workspace.

Without ``--controller`` the command mux is asked which controller is MoveIt's (its
``moveit_controller`` parameter); without a mux, the one joint trajectory controller for
the arm joints other than teleop's is taken. Teleop's own controller is never used (goals
to it would bypass the mux). When no controller is found, or its action server is
missing, the loaded controllers are listed (exit status 2).

**Simulation only:** it reads ``/robot_description`` and sends nothing unless every
ros2_control hardware plugin in it is a simulator (exit status 3 otherwise). On a robot,
move the arm through MoveIt. The goal is rejected while teleop owns the arm (the MoveIt
controller is then inactive); that rejection is what the command mux relies on. A
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
from ranger_xarm_teleop_servo.session import (choose_arm_controller,  # noqa: E402
                                              describe_controllers, READY_POSTURE,
                                              rejection_reason, TELEOP_CONTROLLER)
from ranger_xarm_teleop_servo.simcheck import NotASimulator, require_simulator  # noqa: E402

READY = list(READY_POSTURE)
MUX_NODE = '/arm_mux'


def _call(node, client, request, seconds: float):
    """Return the response of one service call, or None."""
    if not client.wait_for_service(timeout_sec=seconds):
        return None
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=seconds)
    return future.result() if future.done() else None


def mux_controllers(node):
    """Return the command mux's (moveit, teleop, controller manager) names, or Nones."""
    from rcl_interfaces.srv import GetParameters
    names = ['moveit_controller', 'teleop_controller', 'controller_manager']
    answer = _call(node, node.create_client(GetParameters, f'{MUX_NODE}/get_parameters'),
                   GetParameters.Request(names=names), 2.0)
    if answer is None or len(answer.values) != len(names):
        return None, None, None
    values = [v.string_value or None for v in answer.values]
    return values[0], values[1], values[2]


def loaded_controllers(node, manager: str):
    """Return (name, type, state, required command interfaces) per controller, or None."""
    from controller_manager_msgs.srv import ListControllers
    answer = _call(node, node.create_client(ListControllers,
                                            f'{manager.rstrip("/")}/list_controllers'),
                   ListControllers.Request(), 3.0)
    if answer is None:
        return None
    return [(c.name, c.type, c.state, list(c.required_command_interfaces))
            for c in answer.controller]


def main() -> int:
    """Send one joint goal and report the result."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--controller', default=None,
                        help="MoveIt's arm controller (default: asked from the command mux)")
    parser.add_argument('--seconds', type=float, default=4.0)
    parser.add_argument('joints', nargs='*', type=float, default=READY)
    args = parser.parse_args()
    if len(args.joints) != 6:
        parser.error('six joint angles are needed')
    rclpy.init()
    node = rclpy.create_node('move_arm')
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
        moveit, teleop, manager = mux_controllers(node)
        controllers = loaded_controllers(node, manager or '/controller_manager')
        name, why = (args.controller, '') if args.controller else choose_arm_controller(
            controllers, moveit, teleop)
        if name is None:
            print(f'no arm controller found: {why}. Loaded controllers:')
            print(describe_controllers(controllers))
            return 2
        if name == (teleop or TELEOP_CONTROLLER):
            print(f'REFUSED, nothing sent: {name} is the teleop controller; goals to it would '
                  "bypass the command mux. Use MoveIt's controller.")
            return 2
        print(f'controller: {name}' + ('' if args.controller else
                                       ' (from the command mux)' if moveit else
                                       " (the only arm trajectory controller besides teleop's)"))
        client = ActionClient(node, FollowJointTrajectory, f'/{name}/follow_joint_trajectory')
        if not client.wait_for_server(10.0):
            print(f'no action server /{name}/follow_joint_trajectory. Loaded controllers:')
            print(describe_controllers(controllers))
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
