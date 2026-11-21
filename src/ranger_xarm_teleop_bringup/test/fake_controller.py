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
A stand-in trajectory controller that can lose its answers, for the arm tools' tests.

    fake_controller.py NAME MODE

It serves the action ``/NAME/follow_joint_trajectory`` by hand (its three services and two
topics), publishes ``/joint_states`` for the six arm joints at 50 Hz and a simulator's
``/robot_description``. A goal moves the joints in a straight line to its last point over
that point's time. MODE says which answer is lost:

    answer        none: accepted, executed, the result sent
    reject        the goal is rejected
    goal_moves    the goal is accepted and executed; the answer to it is never sent
    goal_ignored  the goal request is lost: no answer, nothing listed, nothing moves
    result_lost   accepted and executed; the result is never sent, nor is the answer to a
                  request to cancel that goal

A request to cancel a goal the controller never had is always answered (the tools try the
cancel path with one before they send a goal). Every request is printed.

Unlike a real action server, the stand-in keeps a finished goal for ever. A real one forgets
it some seconds after its end and then answers a cancel request for it as for a goal it never
had; what the tools say to that answer is tested without ROS (``test_calls.py``).
"""

import sys
import time

from action_msgs.msg import GoalInfo, GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.qos import (DurabilityPolicy, qos_profile_action_status_default, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, String

JOINTS = [f'xarm_joint{i}' for i in range(1, 7)]
DESCRIPTION = ('<robot name="stand_in"><ros2_control name="arm" type="system"><hardware>'
               '<plugin>mock_components/GenericSystem</plugin></hardware></ros2_control>'
               '</robot>')


def by_hand(service):
    """Stop ``service`` answering by itself; return the function that sends an answer."""
    send = service.send_response
    headers = {}
    service.send_response = lambda response, header: headers.__setitem__(id(response), header)
    return lambda response: send(response, headers.pop(id(response)))


def main() -> int:  # noqa: C901 - one small state machine
    """Serve the action until killed."""
    name, mode = sys.argv[1], sys.argv[2]
    action = f'/{name}/follow_joint_trajectory/_action'
    rclpy.init()
    node = rclpy.create_node('fake_controller')
    state = {'q': [0.0] * 6, 'goal': None, 'status': None, 'waiting': [], 'later': []}

    def say(text: str) -> None:
        print(text, flush=True)

    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    description = node.create_publisher(String, '/robot_description', latched)
    description.publish(String(data=DESCRIPTION))
    joints = node.create_publisher(JointState, '/joint_states', 10)
    statuses = node.create_publisher(GoalStatusArray, f'{action}/status',
                                     qos_profile_action_status_default)
    node.create_publisher(FollowJointTrajectory.Impl.FeedbackMessage, f'{action}/feedback', 10)

    def list_status(status: int) -> None:
        goal = state['goal']
        goal['status'] = status
        info = GoalInfo(goal_id=goal['id'], stamp=goal['stamp'])
        statuses.publish(GoalStatusArray(status_list=[GoalStatus(goal_info=info,
                                                                 status=status)]))

    def on_goal(request, response):
        uuid = bytes(request.goal_id.uuid).hex()[:8]
        if mode == 'goal_ignored':
            say(f'goal {uuid}: request lost')
            return response                         # never sent
        response.accepted = mode != 'reject'
        response.stamp = node.get_clock().now().to_msg()
        if response.accepted:
            point = request.goal.trajectory.points[-1]
            seconds = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            state['goal'] = {'id': request.goal_id, 'stamp': response.stamp, 'status': None,
                             'from': list(state['q']), 'to': list(point.positions),
                             'start': time.monotonic(), 'seconds': max(seconds, 0.1)}
            list_status(GoalStatus.STATUS_ACCEPTED)
            list_status(GoalStatus.STATUS_EXECUTING)
        say(f'goal {uuid}: ' + ('accepted' if response.accepted else 'rejected')
            + (', the answer is not sent' if mode == 'goal_moves' else ''))
        if mode != 'goal_moves':
            state['later'].append((answer_goal, response))
        return response

    def on_result(request, response):
        say('result requested' + (', never answered' if mode == 'result_lost' else ''))
        if mode != 'result_lost':
            state['waiting'].append(response)
        return response

    def on_cancel(request, response):
        wanted = bytes(request.goal_info.goal_id.uuid)
        goal = state['goal']
        mine = goal is not None and bytes(goal['id'].uuid) == wanted
        if not mine:
            # (A request without an id would mean every goal; the tools never send one.)
            response.return_code = CancelGoal.Response.ERROR_UNKNOWN_GOAL_ID
            say('cancel request: every goal' if not any(wanted) else
                'cancel request: a goal the controller never had')
        elif mode == 'result_lost':
            say('cancel request: the goal, never answered')
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

    answer_goal = by_hand(node.create_service(FollowJointTrajectory.Impl.SendGoalService,
                                              f'{action}/send_goal', on_goal))
    answer_result = by_hand(node.create_service(FollowJointTrajectory.Impl.GetResultService,
                                                f'{action}/get_result', on_result))
    answer_cancel = by_hand(node.create_service(CancelGoal, f'{action}/cancel_goal', on_cancel))

    def tick() -> None:
        # Answers are sent here, after the service has handed the response back.
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
        if goal is not None and goal['status'] == GoalStatus.STATUS_SUCCEEDED:
            waiting, state['waiting'] = state['waiting'], []
            for response in waiting:
                response.status = GoalStatus.STATUS_SUCCEEDED
                answer_result(response)
                say('result sent')
        joints.publish(JointState(header=Header(stamp=node.get_clock().now().to_msg()),
                                  name=JOINTS, position=[float(v) for v in state['q']]))

    node.create_timer(0.02, tick)
    say(f'ready: {name} {mode}')
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
