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
Back the simulated arm out of a collision halt along the path it came in on.

    tools/retreat_arm.py [--clearance M] [--max-tool-travel M] [--max-joint-travel RAD]

One command after MoveIt Servo halted the arm in contact (move_group then refuses to plan
from there, so ``tools/recover_arm.py`` cannot help):

1. asks the teleop node to disarm (``/teleop/disarm``) and waits until the command mux
   reports MoveIt's controller as the owner and the teleop node reports disarmed;
2. fetches the joint path the teleop node recorded while teleop was armed
   (``/teleop/joint_path``); the arm must be at its newest point;
3. walks that path backwards (``retreat.plan``) to the first point that is clear of the
   robot's fixed structure by the link keep-out margin (or ``--clearance``, never less)
   according to the keep-out model, valid for move_group (``/check_state_validity``, with
   its padding) and away from the wrist singularity, within the travel bounds;
4. sends that stretch of the path, slowly (at most 0.1 rad/s per joint and 0.03 m/s at the
   tool), as one goal to MoveIt's trajectory controller (the one the command mux names),
   and cancels it if the arm leaves the path or teleop is armed again.

It never plans, never leaves the recorded path, never asks the mux for ownership and never
uses teleop's controller. Nothing moves when there is no path, the arm is not at its end,
the path goes deeper into contact on the way back, or no point within the bounds is clear.

**Simulation only**, like ``tools/move_arm.py`` (exit status 3 otherwise). Exit status 0:
moved and clear, or already clear; 5: refused, no motion goal sent; 4: not handed back to
MoveIt, no motion goal sent; 1: the motion failed, was stopped, or its state is unknown; 2:
something is missing, no motion goal sent.

Every read is asked again when no answer comes. The goal is sent once: without an answer to
it, or to the request for its result, its state is unknown, not rejected. The controller is
then asked to cancel it, and what the controller listed and what the joints did is printed.
"""

import argparse
import os
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, os.path.join(REPO, 'src', 'ranger_xarm_teleop_servo'))
sys.path.insert(0, TOOLS)

from _calls import Calls, Goal, GOAL_ANSWER_S, NO_SERVICE  # noqa: E402, I100
from ranger_xarm_teleop_servo import calls as rules  # noqa: E402
from ranger_xarm_teleop_servo import keepout as ko  # noqa: E402
from ranger_xarm_teleop_servo import retreat  # noqa: E402
from ranger_xarm_teleop_servo import solver as dls  # noqa: E402
from ranger_xarm_teleop_servo.params import DEFAULTS  # noqa: E402
from ranger_xarm_teleop_servo.session import (choose_arm_controller, may_recover,  # noqa: E402
                                              STATE_FRESH_S, TELEOP_CONTROLLER)
from ranger_xarm_teleop_servo.simcheck import NotASimulator, require_simulator  # noqa: E402

LEAVES_PATH = 0.02          # rad: the arm this far from the path stops the motion
STALE_JOINTS_S = 0.5        # joint states older than this stop the motion
TELEOP_NODE = '/teleop_servo'
DEFAULT_MIN_ARMING_CONDITIONING = DEFAULTS['min_arming_conditioning']


def main() -> int:  # noqa: C901 - one sequence of checks, kept in the order it runs
    """Disarm, fetch the path, plan the retreat, move."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--clearance', type=float, default=0.0,
                        help='clearance to reach (m); never less than the keep-out margin')
    parser.add_argument('--max-tool-travel', type=float, default=retreat.MAX_TOOL_TRAVEL,
                        help=f'm of tool path, at most {retreat.MAX_TOOL_TRAVEL}')
    parser.add_argument('--max-joint-travel', type=float, default=retreat.MAX_JOINT_TRAVEL,
                        help=f'rad on any joint, at most {retreat.MAX_JOINT_TRAVEL}')
    parser.add_argument('--wait', type=float, default=10.0,
                        help='seconds to wait for MoveIt to own the arm')
    parser.add_argument('--group', default='xarm6')
    args = parser.parse_args()
    if not 0.0 < args.max_tool_travel <= retreat.MAX_TOOL_TRAVEL:
        parser.error(f'--max-tool-travel must be in (0, {retreat.MAX_TOOL_TRAVEL}]')
    if not 0.0 < args.max_joint_travel <= retreat.MAX_JOINT_TRAVEL:
        parser.error(f'--max-joint-travel must be in (0, {retreat.MAX_JOINT_TRAVEL}]')

    from ament_index_python.packages import get_package_share_directory
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from moveit_msgs.msg import RobotState
    from moveit_msgs.srv import GetStateValidity
    from move_arm import controllers_of, list_client, MUX_NODE
    from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
    from ranger_xarm_teleop_interfaces.srv import GetJointPath
    from rcl_interfaces.srv import GetParameters
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectoryPoint

    rclpy.init()
    node = rclpy.create_node('retreat_arm')
    seen = {'description': None, 'owner': None, 'owner_at': -1e9, 'armed': None,
            'state_at': -1e9, 'margin': None, 'joints': None, 'joints_at': -1e9}

    def fresh():
        now = time.monotonic()
        return (seen['owner'] if now - seen['owner_at'] <= STATE_FRESH_S else None,
                seen['armed'] if now - seen['state_at'] <= STATE_FRESH_S else None)

    def on_state(msg) -> None:
        seen.update(armed=bool(msg.armed), state_at=time.monotonic(),
                    margin=float(msg.keepout_margin))

    try:
        # Every client is made here, before the first wait, so that its server has long
        # matched it when it is used.
        calls = Calls(node)
        spin = calls.spin
        disarm = node.create_client(Trigger, '/teleop/disarm')
        parameters = calls.parameter_client(TELEOP_NODE)
        path_client = node.create_client(GetJointPath, '/teleop/joint_path')
        validity = node.create_client(GetStateValidity, '/check_state_validity')
        mux_client = calls.parameter_client(MUX_NODE)
        controllers_client = list_client(node)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, '/robot_description',
                                 lambda msg: seen.update(description=msg.data), latched)
        node.create_subscription(ArmOwner, '/teleop/mux/owner', lambda msg: seen.update(
            owner=int(msg.owner), owner_at=time.monotonic()), 1)   # newest only
        node.create_subscription(TeleopState, '/teleop/state', on_state, 1)   # newest only
        node.create_subscription(JointState, '/joint_states', lambda msg: seen.update(
            joints=dict(zip(msg.name, msg.position)), joints_at=time.monotonic()), 10)
        spin(10.0, lambda: seen['description'] is not None)
        try:
            require_simulator(seen['description'] or '')
        except NotASimulator as err:
            print(f'REFUSED, nothing sent: {err}')
            return 3

        # 1. Disarm, and MoveIt's controller must own the arm. Disarming is asked once:
        # whether it took effect is read back below, with or without an answer.
        if spin(5.0, disarm.service_is_ready):
            asked = disarm.call_async(Trigger.Request())
            spin(5.0, asked.done)
            print('disarm: ' + (asked.result().message if asked.done() else
                                'no answer; reading the state back'))
        else:
            print('disarm: no /teleop/disarm service (teleop node not running?)')
        if not spin(args.wait, lambda: may_recover(*fresh())[0]):
            print(f'NOT HANDED BACK, no motion goal sent: {may_recover(*fresh())[1]} after '
                  f'{args.wait:.0f} s. If the arm is still moving, wait and run this again.')
            return 4
        print('MoveIt owns the arm, teleop is disarmed')

        # 2. The teleop node's settings and its recorded path.
        names = ['base_frame', 'servo_tip_frame', 'link_keepout_cell_m',
                 'joint_velocity_limit_names', 'joint_velocity_limits',
                 'linear_speed_cap_m_s', 'angular_speed_cap_rad_s']
        values = calls.read(parameters, GetParameters.Request(names=names))
        if values is None:
            print('MISSING, no motion goal sent: ' + (
                f'no {TELEOP_NODE}/get_parameters service (teleop node not running?)'
                if calls.silent == NO_SERVICE else
                'the teleop node did not answer the request for its settings'))
            return 2
        if len(values.values) != len(names):
            print('MISSING, no motion goal sent: the teleop node answered with '
                  f'{len(values.values)} of its {len(names)} settings ({", ".join(names)}): one '
                  'of them has no value')
            return 2
        if seen['margin'] is None:
            print('MISSING, no motion goal sent: no /teleop/state arrived, which carries the '
                  'keep-out margin')
            return 2
        v = values.values
        base, tip = v[0].string_value, v[1].string_value
        cell = v[2].double_value or ko.DEFAULT_CELL
        limits = dict(zip(v[3].string_array_value, v[4].double_array_value))
        # Optional, without a value unless set (then the service answers nothing for it).
        optional = calls.read(parameters,
                              GetParameters.Request(names=['min_arming_conditioning']))
        min_conditioning = (optional.values[0].double_value
                            if optional is not None and len(optional.values) == 1
                            else DEFAULT_MIN_ARMING_CONDITIONING)
        margin = seen['margin']
        wanted = max(margin, args.clearance)
        reply = calls.read(path_client, GetJointPath.Request())
        if reply is None:
            print('MISSING, no motion goal sent: ' + (
                'no /teleop/joint_path service (teleop node not running?)'
                if calls.silent == NO_SERVICE else
                'the teleop node did not answer the request for its recorded path'))
            return 2
        path = reply.path
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                          for p in path.points])
        postures = np.array([list(p.positions) for p in path.points])
        print(f'recorded path: {reply.message}')

        # 3. The keep-out model, MoveIt's verdict and the arming measure, then the plan.
        urdf = seen['description']
        model = ko.build(urdf, ko.make_resolver(get_package_share_directory), base, tip,
                         margin, cell=cell)
        arm = dls.arm_from_urdf(urdf, base, tip, velocity_limits=limits)
        params = dls.SolverParams(linear_speed_cap=v[5].double_value,
                                  angular_speed_cap=v[6].double_value)
        spin(2.0, lambda: seen['joints'] is not None)
        if seen['joints'] is None:
            print('MISSING, no motion goal sent: no /joint_states')
            return 2
        positions = dict(seen['joints'])
        try:
            measured = np.array([positions[name] for name in arm.joint_names])
            extra = model.extra_values(positions)
        except (KeyError, ko.ModelError) as err:
            print(f'MISSING, no motion goal sent: joint state without {err}')
            return 2

        def valid(q) -> bool:
            # Without an answer the state does not count as valid: the retreat stays short
            # of it, or is refused.
            state = RobotState()
            state.joint_state.name = list(arm.joint_names) + list(extra)
            state.joint_state.position = [float(x) for x in q] + list(extra.values())
            answer = calls.read(validity, GetStateValidity.Request(robot_state=state,
                                                                   group_name=args.group))
            return bool(answer is not None and answer.valid)

        plan = retreat.plan(
            times, postures, measured,
            clearance=lambda q: model.clearance(q, extra, within=max(model.influence,
                                                                     wanted + 0.01))[0],
            tool_point=lambda q: dls.tool_pose(arm, q)[:3, 3],
            valid=valid,
            conditioning=lambda q: dls.levels(arm, q, params).orientation,
            wanted=wanted, min_conditioning=min_conditioning,
            max_tool_travel=args.max_tool_travel, max_joint_travel=args.max_joint_travel)
        if not plan.ok:
            done = plan.message.startswith('already clear')
            print(('' if done else 'REFUSED: ') + plan.message)
            return 0 if done else 5
        print(f'plan: {plan.message}; {len(plan.postures)} points, '
              f'{plan.times[-1]:.1f} s; the largest joint motion {plan.joint_travel:.3f} rad')

        # 4. One goal to MoveIt's controller, watched.
        moveit, teleop, controllers = controllers_of(calls, mux_client, controllers_client)
        name, why = choose_arm_controller(controllers, moveit, teleop)
        if name is None or name == (teleop or TELEOP_CONTROLLER):
            print(f'MISSING, no motion goal sent: no MoveIt arm controller ({why or name})')
            return 2
        # The controller's name is known only now. Its cancel path is tried with a harmless
        # request before any goal goes out.
        action = Goal(calls, FollowJointTrajectory, f'/{name}/follow_joint_trajectory',
                      list(arm.joint_names))
        if not action.ready(10.0):
            print('MISSING, no motion goal sent: ' + (
                f'the action server /{name}/follow_joint_trajectory did not answer on its '
                'cancel path, so a motion could not be stopped'
                if action.client.server_is_ready() else
                f'no action server /{name}/follow_joint_trajectory'))
            return 2
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(arm.joint_names)
        speeds = np.gradient(plan.postures, plan.times, axis=0) if len(plan.times) > 2 \
            else np.zeros_like(plan.postures)
        speeds[0] = 0.0
        speeds[-1] = 0.0
        for q, t, qd in zip(plan.postures[1:], plan.times[1:], speeds[1:]):
            seconds = int(t)
            goal.trajectory.points.append(JointTrajectoryPoint(
                positions=[float(x) for x in q], velocities=[float(x) for x in qd],
                time_from_start=Duration(sec=seconds, nanosec=int((t - seconds) * 1e9))))
        # Once more, right before sending. Waiting for the action server processed no
        # reports, so they may have gone stale meanwhile: wait for fresh ones first.
        spin(2.0, lambda: may_recover(*fresh())[0])
        allowed, why = may_recover(*fresh())
        if not allowed:
            print(f'NOT HANDED BACK, no motion goal sent: {why}')
            return 4
        answer = action.send(goal)
        if answer == 'unknown':
            print('UNKNOWN: ' + action.unanswered('the goal', GOAL_ANSWER_S))
            return 1
        if answer == 'rejected':
            print('FAILED: the controller rejected the goal; ' + calls.watch(limit=0.0).text())
            return 1

        def leaves() -> str:
            """Return why the motion has to stop now, or ''."""
            positions = dict(seen['joints'])
            q = np.array([positions[n] for n in arm.joint_names])
            off = retreat.distance_from_path(plan.postures, q)
            if time.monotonic() - seen['joints_at'] > STALE_JOINTS_S:
                return 'no fresh joint states'
            if off > LEAVES_PATH:
                return f'the arm left the path by {off:.3f} rad'
            if fresh()[1]:
                return 'teleop was armed'
            return ''

        wait = plan.times[-1] + 10.0
        try:
            done, stopped = action.result(wait, stop=leaves)
        except KeyboardInterrupt:
            done, stopped = None, 'interrupted'
        if stopped == 'unknown':
            print('UNKNOWN: ' + action.unanswered("the request for the goal's result", wait))
            return 1
        if stopped:
            cancel, motion = action.stop()
            print(f'STOPPED: {stopped}; {rules.cancel_text(cancel)}; {motion.text()}')
            return 1
        spin(1.0)
        if done.result.error_code != 0:
            print(f'FAILED: the controller reported {done}')
            return 1
        positions = dict(seen['joints'])
        q = np.array([positions[n] for n in arm.joint_names])
        clearance, pair = model.clearance(q, extra)
        ok = valid(q)
        print(f'retreated: clearance {clearance * 1000.0:.1f} mm ({pair[0]} - {pair[1]}), '
              f'valid for MoveIt: {ok}')
        return 0 if ok and clearance >= wanted - 1e-3 else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
