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
   that it is not armed. With a teleop node running that report must have been heard,
   freshly; with no teleop node at all the owner's report is enough, and the tool says
   which of the two it is;
3. asks move_group to plan (with collision checking) and execute a motion to the posture
   (default: the ready posture of ``tools/move_arm.py``), and stops the motion if teleop
   is armed during it.

It never switches controllers and never asks the mux for ownership itself, and it sends
no motion goal unless step 2 succeeded. **Simulation only**, like ``tools/move_arm.py``:
nothing is sent unless every ros2_control hardware plugin in ``/robot_description`` is a
simulator (exit status 3). Exit status 0 only when the arm reached the posture; 4 when
the arm was not handed back to MoveIt (no motion goal sent).

The goal is sent once. Without an answer to it, or to the request for its result, its
state is unknown, not rejected: move_group is asked to cancel it, and what move_group
listed and what the joints did is printed (exit status 1).
"""

import argparse
import os
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, os.path.join(REPO, 'src', 'ranger_xarm_teleop_servo'))
sys.path.insert(0, TOOLS)

from _calls import Calls, Goal, GOAL_ANSWER_S  # noqa: E402
from ranger_xarm_teleop_servo import calls as rules  # noqa: E402
from ranger_xarm_teleop_servo.session import ARM_JOINTS, may_recover  # noqa: E402
from ranger_xarm_teleop_servo.session import READY_POSTURE, recovery_case  # noqa: E402
from ranger_xarm_teleop_servo.session import STATE_FRESH_S  # noqa: E402
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

    def fresh():
        """Return the owner and the armed flag, None for each that is not fresh."""
        now = time.monotonic()
        owner = seen['owner'] if now - seen['owner_at'] <= STATE_FRESH_S else None
        armed = seen['armed'] if now - seen['state_at'] <= STATE_FRESH_S else None
        return owner, armed

    try:
        calls = Calls(node)
        spin = calls.spin
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, '/robot_description',
                                 lambda msg: seen.update(description=msg.data), latched)
        node.create_subscription(ArmOwner, '/teleop/mux/owner', on_owner, 1)   # newest only
        node.create_subscription(TeleopState, '/teleop/state', on_state, 1)   # newest only
        disarm = node.create_client(Trigger, args.disarm_service)
        move = Goal(calls, MoveGroup, '/move_action', ARM_JOINTS)
        spin(10.0, lambda: seen['description'] is not None)
        try:
            require_simulator(seen['description'] or '')
        except NotASimulator as err:
            print(f'REFUSED, nothing sent: {err}')
            return 3

        # 1. Disarm (the only thing this tool asks of the teleop side). Asked once: whether
        # it took effect is read back in step 2, with or without an answer.
        if spin(3.0, disarm.service_is_ready):
            call = disarm.call_async(Trigger.Request())
            spin(5.0, call.done)
            answer = call.result() if call.done() else None
            print('disarm: ' + (answer.message if answer is not None else
                                'no answer; reading the state back'))
        else:
            print(f'disarm: no {args.disarm_service} service (teleop node not running?)')

        # 2. MoveIt must own the arm and teleop must not be armed, by fresh reports. With
        # a teleop node running, "not armed" must have been heard; with none at all (its
        # service and its state topic absent, nothing heard) a recovery still works.
        def teleop_running() -> bool:
            return (seen['state_at'] > 0.0 or disarm.service_is_ready()
                    or node.count_publishers('/teleop/state') > 0)

        def allowed():
            return may_recover(*fresh(), teleop_running())

        def handed_back() -> bool:
            return allowed()[0]

        if not spin(args.wait, handed_back):
            print(f'NOT RECOVERED, no motion goal sent: {allowed()[1]} after '
                  f'{args.wait:.0f} s. If the arm is still moving, wait and run this again.')
            return 4
        print(f'MoveIt owns the arm, {recovery_case(fresh()[1])}')

        # 3. Plan and execute through move_group.
        if not move.ready(10.0):
            print('NOT RECOVERED, no motion goal sent: '
                  + ('move_group did not answer on the cancel path of /move_action, so a '
                     'motion could not be stopped' if move.client.server_is_ready() else
                     'no move_group action server (/move_action)'))
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
        # Once more, right before sending. Waiting for the action server processed no
        # reports, so they may have gone stale meanwhile: wait for fresh ones first.
        spin(2.0, handed_back)
        may, why = allowed()
        if not may:
            print(f'NOT RECOVERED, no motion goal sent: {why}')
            return 4
        answer = move.send(goal)
        if answer == 'unknown':
            print('NOT RECOVERED: ' + move.unanswered('the goal', GOAL_ANSWER_S))
            return 1
        if answer == 'rejected':
            print('NOT RECOVERED: move_group rejected the goal; '
                  + calls.watch(limit=0.0).text())
            return 1
        # Teleop armed during the motion stops it: the operator has taken the arm back.
        done, why = move.result(
            120.0, stop=lambda: 'teleop was armed during the motion' if fresh()[1] else '')
        if why == 'unknown':
            print('NOT RECOVERED: ' + move.unanswered("the request for the goal's result",
                                                      120.0))
            return 1
        if why:
            cancel, motion = move.stop()
            print(f'NOT RECOVERED, STOPPED: {why}; {rules.cancel_text(cancel)}; '
                  f'{motion.text()}')
            return 1
        code = done.result.error_code.val
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
