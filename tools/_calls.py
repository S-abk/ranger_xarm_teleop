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
Service and action calls of the arm tools, by the rules of ``calls.py`` in the servo package.

A reply can be lost without the caller being told. So in these tools:

* clients are made when the tool starts, not right before a call;
* a read has a timeout and is asked again, and the tool says that it is waiting;
* a motion goal is sent once. Without an answer its state is unknown: the tool cancels every
  goal of that action server, watches the goal's status and the joints, and reports what it
  saw. It says that the joints did not move only when it watched them not moving.
"""

from collections import deque
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'src', 'ranger_xarm_teleop_servo'))
from ranger_xarm_teleop_servo import calls as rules  # noqa: E402

READ_S = 3.0                # a read without an answer for this long is asked again
READ_ATTEMPTS = 3
GOAL_ANSWER_S = 10.0        # the action server answers a goal within this, or its state is unknown
SETTLE_S = 10.0             # how long the joints are watched for coming to rest after a cancel
NO_SERVICE, NO_ANSWER = 'no service', 'no answer'


class Calls:
    """The tool's node with its joint states, its clients and its reads."""

    def __init__(self, node, joint_states: bool = True) -> None:
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        self.node, self._rclpy = node, rclpy
        self.joints = deque(maxlen=20000)       # (time, {joint: position}), oldest first
        self.silent = ''                        # why the last read returned None
        self._parameter_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=rules.PARAMETER_REPLY_DEPTH)
        if joint_states:
            from sensor_msgs.msg import JointState
            node.create_subscription(JointState, '/joint_states', self._on_joints, 50)

    def _on_joints(self, msg) -> None:
        self.joints.append((time.monotonic(), dict(zip(msg.name, msg.position))))

    def spin(self, seconds: float, until=lambda: False) -> bool:
        """Process messages for ``seconds`` or until ``until()``; return ``until()``."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if until():
                return True
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
        return until()

    def parameter_client(self, target: str):
        """Return a client of ``target``'s parameter service, with the deep reply queue."""
        from rcl_interfaces.srv import GetParameters
        return self.node.create_client(GetParameters, f'{target.rstrip("/")}/get_parameters',
                                       qos_profile=self._parameter_qos)

    def read(self, client, request, seconds: float = READ_S, attempts: int = READ_ATTEMPTS,
             quiet: bool = False):
        """
        Return the answer to a request that changes nothing, asked again after each timeout.

        None without the service or without an answer to any attempt; ``silent`` then says
        which. Never use this for a request that moves or switches something.
        """
        self.silent = ''
        if not self.spin(seconds, client.service_is_ready):
            self.silent = NO_SERVICE
            return None
        for attempt in range(1, attempts + 1):
            future = client.call_async(request)
            if self.spin(seconds, future.done):
                return future.result()
            client.remove_pending_request(future)       # a late answer is not delivered
            if not quiet:
                print(f'no answer from {client.srv_name} within {seconds:g} s'
                      + ('; asking again' if attempt < attempts else f' ({attempts} requests)'),
                      flush=True)
        self.silent = NO_ANSWER
        return None

    def motion(self, since: float, names) -> 'rules.Motion':
        """Return what the joint states showed since ``since`` (a monotonic time)."""
        return rules.motion_seen(list(self.joints), names, since, time.monotonic())

    def watch(self, since: float, names, at_least: float = rules.STILL_S + 0.2,
              limit: float = SETTLE_S) -> 'rules.Motion':
        """Watch the joints until they are still (at most ``limit`` s); return what they did."""
        self.spin(at_least)
        self.spin(limit, lambda: self.motion(since, names).still)
        return self.motion(since, names)


class Goal:
    """One action server: its goal, status and cancel paths, all made at start-up."""

    def __init__(self, calls: Calls, action_type, name: str) -> None:
        from action_msgs.msg import GoalStatusArray
        from action_msgs.srv import CancelGoal
        from rclpy.action import ActionClient
        from rclpy.qos import qos_profile_action_status_default
        self.calls, self.name = calls, name
        self._cancel_type = CancelGoal
        self.client = ActionClient(calls.node, action_type, name)
        self.cancel_client = calls.node.create_client(CancelGoal, f'{name}/_action/cancel_goal')
        self.goal_id = None
        self.handle = None
        self.sent_at = 0.0
        self.statuses = []          # of this tool's goal, in the order the server listed them
        calls.node.create_subscription(GoalStatusArray, f'{name}/_action/status',
                                       self._on_status, qos_profile_action_status_default)

    def _on_status(self, msg) -> None:
        if self.goal_id is None:
            return
        mine = bytes(self.goal_id.uuid)
        for entry in msg.status_list:
            if bytes(entry.goal_info.goal_id.uuid) == mine and (
                    not self.statuses or self.statuses[-1] != entry.status):
                self.statuses.append(int(entry.status))

    def ready(self, seconds: float) -> bool:
        """
        Wait for the server, then do one harmless round trip on its cancel path.

        The request names a goal that does not exist, so nothing is cancelled. It shows that
        the path the tool may need to stop a motion carries a request and its answer.
        """
        end = time.monotonic() + seconds
        if not self.calls.spin(seconds, self.client.server_is_ready):
            return False
        request = self._cancel_type.Request()
        request.goal_info.goal_id.uuid = list(uuid.uuid4().bytes)
        left = max(1.0, end - time.monotonic())
        return self.calls.read(self.cancel_client, request, seconds=min(READ_S, left)) is not None

    def send(self, goal, seconds: float = GOAL_ANSWER_S) -> str:
        """Send the goal once; return 'accepted', 'rejected' or 'unknown' (no answer)."""
        from unique_identifier_msgs.msg import UUID
        self.goal_id = UUID(uuid=list(uuid.uuid4().bytes))
        self.statuses = []
        self.sent_at = time.monotonic()
        sent = self.client.send_goal_async(goal, goal_uuid=self.goal_id)
        if not self.calls.spin(seconds, sent.done):
            return 'unknown'
        self.handle = sent.result()
        return 'accepted' if self.handle is not None and self.handle.accepted else 'rejected'

    def result(self, seconds: float, stop=lambda: ''):
        """
        Wait for the goal's result; return ``(result, why)``.

        ``why`` is '' with a result, the text of ``stop()`` when that asked to stop the
        motion, and 'unknown' when no result came within ``seconds``.
        """
        future = self.handle.get_result_async()
        end = time.monotonic() + seconds
        while not future.done():
            self.calls.spin(0.02, future.done)
            why = stop()
            if why:
                return None, why
            if time.monotonic() > end:
                return None, 'unknown'
        return future.result(), ''

    def cancel_all(self):
        """Ask the server to cancel every goal; return (return code, goals) or None."""
        answer = self.calls.read(self.cancel_client, self._cancel_type.Request(), seconds=2.0)
        return None if answer is None else (int(answer.return_code), len(answer.goals_canceling))

    def stop(self, names):
        """Cancel every goal and watch the joints come to rest; return (cancel, motion)."""
        cancel = self.cancel_all()
        return cancel, self.calls.watch(self.sent_at, names)

    def unanswered(self, what: str, seconds: float, names) -> str:
        """Cancel, watch, and return the report for a request that got no answer."""
        listed = list(self.statuses)        # as the server listed the goal before the cancel
        cancel, motion = self.stop(names)
        return rules.unanswered_goal(what, seconds, listed + self.statuses[len(listed):],
                                     cancel, motion)
