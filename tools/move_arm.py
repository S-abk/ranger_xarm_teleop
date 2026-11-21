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

**Who owns the arm:** with a command mux running (its owner report or its parameters are
seen), a goal goes out only when the mux reports MoveIt as the owner and, where a teleop
node runs, teleop has freshly reported that it is not armed (``session.may_recover``, the
rule of ``tools/recover_arm.py``); a motion is stopped when teleop is armed during it.
Without a mux and without a teleop node, the goal goes out as before. Without a mux but
with a teleop node running, no goal goes out (exit status 4): nothing says who owns the
arm. A teleop node or a mux is believed absent only some seconds after the tool started.

The goal is sent once. Without an answer to it, or to the request for its result, its
state is unknown, not rejected: the controller is asked to cancel it, and what the
controller listed and what the joints did is printed (exit status 1).
"""

import argparse
import os
import sys
import time

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(TOOLS), 'src', 'ranger_xarm_teleop_servo'))
sys.path.insert(0, TOOLS)
from _calls import Calls, Goal, GOAL_ANSWER_S  # noqa: E402, I100
from ranger_xarm_teleop_servo import calls as rules  # noqa: E402
from ranger_xarm_teleop_servo.session import (ARM_JOINTS, choose_arm_controller,  # noqa: E402
                                              describe_controllers, direct_case,
                                              may_move_directly, NO_TELEOP_AFTER_S,
                                              READY_POSTURE, rejection_reason, STATE_FRESH_S,
                                              TELEOP_CONTROLLER)
from ranger_xarm_teleop_servo.simcheck import NotASimulator, require_simulator  # noqa: E402

READY = list(READY_POSTURE)
MUX_NODE = '/arm_mux'
CONTROLLER_MANAGER = '/controller_manager'


def list_client(node, manager: str = CONTROLLER_MANAGER):
    """Return a client of the controller manager's list service (made when a tool starts)."""
    from controller_manager_msgs.srv import ListControllers
    return node.create_client(ListControllers, f'{manager.rstrip("/")}/list_controllers')


def mux_controllers(calls: Calls, client):
    """Return the command mux's (moveit, teleop, controller manager) names, or Nones."""
    from rcl_interfaces.srv import GetParameters
    names = ['moveit_controller', 'teleop_controller', 'controller_manager']
    answer = calls.read(client, GetParameters.Request(names=names), 2.0)
    if answer is None or len(answer.values) != len(names):
        return None, None, None
    values = [v.string_value or None for v in answer.values]
    return values[0], values[1], values[2]


def loaded_controllers(calls: Calls, client):
    """Return (name, type, state, required command interfaces) per controller, or None."""
    from controller_manager_msgs.srv import ListControllers
    answer = calls.read(client, ListControllers.Request(), 3.0)
    if answer is None:
        return None
    return [(c.name, c.type, c.state, list(c.required_command_interfaces))
            for c in answer.controller]


def controllers_of(calls: Calls, mux_client, default_list_client):
    """Return (moveit, teleop, loaded controllers), asking the manager the mux names."""
    moveit, teleop, manager = mux_controllers(calls, mux_client)
    client = default_list_client
    if manager and manager.rstrip('/') != CONTROLLER_MANAGER:
        client = list_client(calls.node, manager)      # known only now; a read, asked again
    return moveit, teleop, loaded_controllers(calls, client)


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
    started = time.monotonic()
    code = 1
    try:
        calls = Calls(node)
        mux_client = calls.parameter_client(MUX_NODE)
        controllers_client = list_client(node)
        description = []
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, '/robot_description',
                                 lambda msg: description.append(msg.data), latched)
        # Who owns the arm and whether teleop is armed (listening only; nothing is requested).
        from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
        from std_srvs.srv import Trigger
        owner = []
        seen = {'owner_at': -1e9, 'armed': None, 'state_at': -1e9, 'mux_answered': False}

        def on_owner(msg) -> None:
            owner.append(int(msg.owner))
            seen['owner_at'] = time.monotonic()

        def on_state(msg) -> None:
            seen.update(armed=bool(msg.armed), state_at=time.monotonic())
        node.create_subscription(ArmOwner, '/teleop/mux/owner', on_owner, 10)
        node.create_subscription(TeleopState, '/teleop/state', on_state, 1)    # newest only
        disarm = node.create_client(Trigger, '/teleop/disarm')    # only to see a teleop node

        def fresh():
            """Return the owner and the armed flag, None for each that is not fresh."""
            now = time.monotonic()
            return (owner[-1] if owner and now - seen['owner_at'] <= STATE_FRESH_S else None,
                    seen['armed'] if now - seen['state_at'] <= STATE_FRESH_S else None)

        def mux_present() -> bool:
            return (bool(owner) or seen['mux_answered']
                    or node.count_publishers('/teleop/mux/owner') > 0)

        def teleop_running() -> bool:
            return (seen['state_at'] > 0.0 or disarm.service_is_ready()
                    or node.count_publishers('/teleop/state') > 0)

        def allowed():
            """Return (allowed, why not, the case it is allowed in)."""
            mux, running = mux_present(), teleop_running()
            if (not mux or not running) and time.monotonic() - started < NO_TELEOP_AFTER_S:
                looked = (('a command mux', mux), ('a teleop node', running))
                missing = ' and '.join(name for name, there in looked if not there)
                return False, f'still looking for {missing}', ''
            owner_now, armed = fresh()
            may, why = may_move_directly(mux, owner_now, armed, running)
            return may, why, direct_case(mux, armed)
        calls.spin(10.0, lambda: bool(description))
        try:
            require_simulator(description[0] if description else '')
        except NotASimulator as err:
            print(f'REFUSED, nothing sent: {err}')
            return 3
        moveit, teleop, controllers = controllers_of(calls, mux_client, controllers_client)
        seen['mux_answered'] = moveit is not None
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
        action = Goal(calls, FollowJointTrajectory, f'/{name}/follow_joint_trajectory',
                      ARM_JOINTS)
        if not action.ready(10.0):
            if action.client.server_is_ready():
                print(f'no goal sent: the action server /{name}/follow_joint_trajectory did '
                      'not answer on its cancel path, so a motion could not be stopped')
            else:
                print(f'no action server /{name}/follow_joint_trajectory. Loaded controllers:')
                print(describe_controllers(controllers))
            return 2
        calls.spin(max(0.0, NO_TELEOP_AFTER_S - (time.monotonic() - started)) + 2.0,
                   lambda: allowed()[0])
        may, why, case = allowed()
        if not may:
            print(f'REFUSED, no motion goal sent: {why}')
            return 4
        print(case)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        seconds = int(args.seconds)
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=list(args.joints), velocities=[0.0] * 6,
            time_from_start=Duration(sec=seconds,
                                     nanosec=int((args.seconds - seconds) * 1e9)))]
        # Once more, right before sending: reports may have gone stale meanwhile.
        calls.spin(2.0, lambda: allowed()[0])
        may, why, _ = allowed()
        if not may:
            print(f'REFUSED, no motion goal sent: {why}')
            return 4
        answer = action.send(goal)
        if answer == 'unknown':
            print('goal UNKNOWN: ' + action.unanswered('the goal', GOAL_ANSWER_S))
            return 1
        if answer == 'rejected':
            calls.spin(2.0, lambda: bool(owner))    # the mux reports it several times a second
            print(f'goal REJECTED: {rejection_reason(owner[-1] if owner else None)}; '
                  + calls.watch(limit=0.0).text())
            return 1
        wait = args.seconds + 30.0
        # Teleop armed during the motion stops it: the operator has taken the arm back.
        done, why = action.result(
            wait, stop=lambda: 'teleop was armed during the motion' if fresh()[1] else '')
        if why == 'unknown':
            print('goal UNKNOWN: ' + action.unanswered("the request for the goal's result", wait))
            return 1
        if why:
            cancel, motion = action.stop()
            print(f'goal STOPPED: {why}; {rules.cancel_text(cancel)}; {motion.text()}')
            return 1
        ok = done.result.error_code == 0 and done.status == 4
        print('goal succeeded' if ok else f'goal FAILED: {done}')
        code = 0 if ok else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
