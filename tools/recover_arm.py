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
Recover the simulated arm after a MoveIt Servo halt: disarm, then plan to a posture.

    tools/recover_arm.py [q1 q2 q3 q4 q5 q6]

One command for "the arm halted, get it back to a posture with room to move":

1. asks the teleop node to disarm (``/teleop/disarm``), which hands the arm back to MoveIt
   through the command mux once the arm is still;
2. waits until the command mux reports MoveIt as the owner and the teleop node reports
   disarmed;
3. asks move_group to plan (with collision checking) and execute a motion to the posture
   (default: the ready posture of ``tools/move_arm.py``).

It never switches controllers and never asks the mux for ownership itself, and it sends
no motion goal unless step 2 succeeded. **Simulation only**, like ``tools/move_arm.py``:
nothing is sent unless every ros2_control hardware plugin in ``/robot_description`` is a
simulator (exit status 3). Exit status 0 only when the arm reached the posture; 4 when
the arm was not handed back to MoveIt (nothing moved).
"""

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'ranger_xarm_teleop_servo'))

from ranger_xarm_teleop_servo.session import ARM_JOINTS, may_recover  # noqa: E402
from ranger_xarm_teleop_servo.session import READY_POSTURE, STATE_FRESH_S  # noqa: E402
from ranger_xarm_teleop_servo.simcheck import NotASimulator, require_simulator  # noqa: E402


def main() -> int:
    """Disarm, wait for MoveIt to own the arm, plan and move."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('joints', nargs='*', type=float, default=list(READY_POSTURE))
    parser.add_argument('--group', default='xarm6')
    parser.add_argument('--wait', type=float, default=10.0,
                        help='seconds to wait for MoveIt to own the arm')
    parser.add_argument('--disarm-service', default='/teleop/disarm')
    parser.add_argument('--speed', type=float, default=0.1,
                        help='velocity and acceleration scaling of the planned motion')
    args = parser.parse_args()
    if len(args.joints) != 6:
        parser.error('six joint angles are needed')

    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
    from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    rclpy.init()
    node = rclpy.create_node('recover_arm')
    seen = {'description': None, 'owner': None, 'owner_at': -1e9, 'armed': None,
            'state_at': -1e9}

    def on_owner(msg) -> None:
        seen.update(owner=int(msg.owner), owner_at=time.monotonic())

    def on_state(msg) -> None:
        seen.update(armed=bool(msg.armed), state_at=time.monotonic())

    def spin(seconds: float, until=lambda: False) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if until():
                return True
            rclpy.spin_once(node, timeout_sec=0.05)
        return until()

    def fresh():
        """Return the owner and the armed flag, None for each that is not fresh."""
        now = time.monotonic()
        owner = seen['owner'] if now - seen['owner_at'] <= STATE_FRESH_S else None
        armed = seen['armed'] if now - seen['state_at'] <= STATE_FRESH_S else None
        return owner, armed

    try:
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, '/robot_description',
                                 lambda msg: seen.update(description=msg.data), latched)
        node.create_subscription(ArmOwner, '/teleop/mux/owner', on_owner, 10)
        node.create_subscription(TeleopState, '/teleop/state', on_state, 10)
        disarm = node.create_client(Trigger, args.disarm_service)
        move = ActionClient(node, MoveGroup, '/move_action')
        spin(10.0, lambda: seen['description'] is not None)
        try:
            require_simulator(seen['description'] or '')
        except NotASimulator as err:
            print(f'REFUSED, nothing sent: {err}')
            return 3

        # 1. Disarm (the only thing this tool asks of the teleop side).
        if disarm.wait_for_service(timeout_sec=3.0):
            call = disarm.call_async(Trigger.Request())
            spin(5.0, call.done)
            answer = call.result() if call.done() else None
            print('disarm: ' + (answer.message if answer is not None else 'no answer'))
        else:
            print(f'disarm: no {args.disarm_service} service (teleop node not running?)')

        # 2. MoveIt must own the arm and teleop must be disarmed, by fresh reports.
        def handed_back() -> bool:
            return may_recover(*fresh())[0]

        if not spin(args.wait, handed_back):
            print(f'NOT RECOVERED, nothing moved: {may_recover(*fresh())[1]} after '
                  f'{args.wait:.0f} s. If the arm is still moving, wait and run this again.')
            return 4
        print('MoveIt owns the arm')

        # 3. Plan and execute through move_group.
        if not move.wait_for_server(10.0):
            print('NOT RECOVERED: no move_group action server (/move_action)')
            return 2
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = args.group
        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = args.speed
        request.max_acceleration_scaling_factor = args.speed
        request.start_state.is_diff = True
        request.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=name, position=float(value), tolerance_above=0.001,
                            tolerance_below=0.001, weight=1.0)
            for name, value in zip(ARM_JOINTS, args.joints)])]
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        allowed, why = may_recover(*fresh())      # once more, right before sending
        if not allowed:
            print(f'NOT RECOVERED, nothing moved: {why}')
            return 4
        sent = move.send_goal_async(goal)
        spin(10.0, sent.done)
        handle = sent.result() if sent.done() else None
        if handle is None or not handle.accepted:
            print('NOT RECOVERED: move_group rejected the goal')
            return 1
        result = handle.get_result_async()
        spin(120.0, result.done)
        done = result.result() if result.done() else None
        code = done.result.error_code.val if done is not None else None
        if code == MoveItErrorCodes.SUCCESS:
            print('recovered: the arm is at ' + ' '.join(f'{v:g}' for v in args.joints))
            return 0
        print(f'NOT RECOVERED: move_group error code {code} '
              '(1 success, -1 planning failed, -4 control failed, -10 start state in collision)')
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
