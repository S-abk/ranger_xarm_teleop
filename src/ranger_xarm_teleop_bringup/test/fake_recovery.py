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
A stand-in for what ``tools/recover_arm.py`` talks to, for the tests of its arming check.

    fake_recovery.py TELEOP [SECONDS]

It serves the action ``/move_action`` by hand (a goal moves the six arm joints in a straight
line to its joint constraints over SECONDS, default 1), publishes ``/joint_states`` at 50 Hz,
a simulator's ``/robot_description`` and the command mux's report that MoveIt owns the arm.
TELEOP says what there is of a teleop node:

    none      nothing: no ``/teleop/disarm`` service, no ``/teleop/state``
    disarmed  the service, and a state that says "not armed", 20 times a second
    silent    the service and the state topic, but no state is ever published
    arms      as ``disarmed``, and "armed" from one second after a goal was accepted
    arms_early  as ``disarmed``, and "armed" from the first request on the action's cancel
              path (the tool tries that path before it sends a goal); that request is
              answered 0.4 s later, when the reports have gone out

"Armed" is reported as the teleop node does: together with teleop as the owner, in the
state and in the mux's report.

Every request is printed.
"""

import sys
import time

from action_msgs.msg import GoalInfo, GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from fake_controller import by_hand, DESCRIPTION, JOINTS
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
import rclpy
from rclpy.qos import (DurabilityPolicy, qos_profile_action_status_default, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger


def main() -> int:  # noqa: C901 - one small state machine
    """Serve until killed."""
    teleop = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    action = '/move_action/_action'
    rclpy.init()
    node = rclpy.create_node('fake_recovery')
    state = {'q': [0.0] * 6, 'goal': None, 'waiting': [], 'later': []}

    def say(text: str) -> None:
        print(text, flush=True)

    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    description = node.create_publisher(String, '/robot_description', latched)
    description.publish(String(data=DESCRIPTION))
    joints = node.create_publisher(JointState, '/joint_states', 10)
    owner = node.create_publisher(ArmOwner, '/teleop/mux/owner', 10)
    statuses = node.create_publisher(GoalStatusArray, f'{action}/status',
                                     qos_profile_action_status_default)
    node.create_publisher(MoveGroup.Impl.FeedbackMessage, f'{action}/feedback', 10)
    teleop_state = None
    if teleop != 'none':
        teleop_state = node.create_publisher(TeleopState, '/teleop/state', 10)

        def on_disarm(request, response):
            say('disarm requested')
            response.success, response.message = True, 'was not armed'
            return response
        node.create_service(Trigger, '/teleop/disarm', on_disarm)

    def list_status(status: int) -> None:
        goal = state['goal']
        goal['status'] = status
        info = GoalInfo(goal_id=goal['id'], stamp=goal['stamp'])
        statuses.publish(GoalStatusArray(status_list=[GoalStatus(goal_info=info,
                                                                 status=status)]))

    def on_goal(request, response):
        response.accepted = True
        response.stamp = node.get_clock().now().to_msg()
        constraints = request.goal.request.goal_constraints[0].joint_constraints
        target = {c.joint_name: c.position for c in constraints}
        state['goal'] = {'id': request.goal_id, 'stamp': response.stamp, 'status': None,
                         'from': list(state['q']), 'to': [target[name] for name in JOINTS],
                         'start': time.monotonic(), 'seconds': max(seconds, 0.1)}
        list_status(GoalStatus.STATUS_ACCEPTED)
        list_status(GoalStatus.STATUS_EXECUTING)
        say('goal accepted')
        state['later'].append((answer_goal, response))
        return response

    def on_result(request, response):
        say('result requested')
        state['waiting'].append(response)
        return response

    def on_cancel(request, response):
        goal = state['goal']
        wanted = bytes(request.goal_info.goal_id.uuid)
        if goal is None or bytes(goal['id'].uuid) != wanted:
            response.return_code = CancelGoal.Response.ERROR_UNKNOWN_GOAL_ID
            say('cancel request: a goal the server never had')
            if teleop == 'arms_early' and 'tried' not in state:
                state['tried'] = time.monotonic()
                state['held'] = (response, time.monotonic() + 0.4)
                return response
        elif goal['status'] == GoalStatus.STATUS_EXECUTING:
            list_status(GoalStatus.STATUS_CANCELING)
            list_status(GoalStatus.STATUS_CANCELED)
            response.goals_canceling = [GoalInfo(goal_id=goal['id'], stamp=goal['stamp'])]
            say(f'cancel request: the goal, cancelled at {state["q"][0]:.3f} rad')
        else:
            response.return_code = CancelGoal.Response.ERROR_GOAL_TERMINATED
            say('cancel request: the goal, which had ended')
        state['later'].append((answer_cancel, response))
        return response

    answer_goal = by_hand(node.create_service(MoveGroup.Impl.SendGoalService,
                                              f'{action}/send_goal', on_goal))
    answer_result = by_hand(node.create_service(MoveGroup.Impl.GetResultService,
                                                f'{action}/get_result', on_result))
    answer_cancel = by_hand(node.create_service(CancelGoal, f'{action}/cancel_goal', on_cancel))
    ticks = [0]

    def tick() -> None:
        ticks[0] += 1
        if 'held' in state and time.monotonic() >= state['held'][1]:
            answer_cancel(state.pop('held')[0])
        later, state['later'] = state['later'], []
        for send, response in later:
            send(response)
        goal = state['goal']
        if goal is not None and goal['status'] == GoalStatus.STATUS_EXECUTING:
            share = min(1.0, (time.monotonic() - goal['start']) / goal['seconds'])
            state['q'] = [a + (b - a) * share for a, b in zip(goal['from'], goal['to'])]
            if share >= 1.0:
                list_status(GoalStatus.STATUS_SUCCEEDED)
                say('goal reached')
        if goal is not None and goal['status'] in (GoalStatus.STATUS_SUCCEEDED,
                                                   GoalStatus.STATUS_CANCELED):
            waiting, state['waiting'] = state['waiting'], []
            for response in waiting:
                response.status = goal['status']
                response.result.error_code.val = (
                    MoveItErrorCodes.SUCCESS if goal['status'] == GoalStatus.STATUS_SUCCEEDED
                    else MoveItErrorCodes.PREEMPTED)
                answer_result(response)
                say('result sent')
        stamp = node.get_clock().now().to_msg()
        joints.publish(JointState(header=Header(stamp=stamp), name=JOINTS,
                                  position=[float(v) for v in state['q']]))
        armed = ((teleop == 'arms' and goal is not None
                  and time.monotonic() - goal['start'] > 1.0)
                 or (teleop == 'arms_early' and 'tried' in state))
        owns = ArmOwner.OWNER_TELEOP if armed else ArmOwner.OWNER_MOVEIT
        if ticks[0] % 5 == 0:
            owner.publish(ArmOwner(owner=owns))
        if teleop in ('disarmed', 'arms', 'arms_early') and ticks[0] % 2 == 0:
            message = TeleopState(armed=armed, owner=owns)
            message.header.stamp = stamp
            teleop_state.publish(message)

    node.create_timer(0.02, tick)
    say(f'ready: teleop {teleop}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
