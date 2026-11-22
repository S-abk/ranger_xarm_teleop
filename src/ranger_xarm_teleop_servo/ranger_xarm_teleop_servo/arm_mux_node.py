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
Command mux: gives either MoveIt or teleop exclusive ownership of the arm (G-9).

With gripper teleop configured (``moveit_gripper_controller`` and
``teleop_gripper_controller``), the gripper has the same owner as the arm: MoveIt's and
teleop's gripper controllers are swapped in the same strict switch, and a goal on MoveIt's
gripper controller refuses the switch. The gripper must be stopped too, judged by net
travel rather than sample-to-sample speed: over the stillness window its joints may span at
most ``gripper_still_band`` (``ownership.BandTracker``). With the arm's rule, Gazebo read
the driven joint as moving at about 0.16 rad/s again and again and every switch back to
MoveIt was refused.

Two trajectory controllers claim the arm joints: MoveIt's and the one MoveIt Servo
feeds. ros2_control lets only one be active, and an inactive trajectory controller
rejects action goals and ignores its command topic, so the active controller defines
the owner. This node reports the owner and performs the switch, and only when

* every arm joint has been still for a while, judged from fresh joint states,
* no trajectory goal is accepted, executing or canceling on either controller,
* teleop is idle (clutch released).

If the arm has only just stopped, or the joint-state stream had a hole, the request
waits up to ``still_wait_s`` for an unbroken run of still samples before it is decided.

The switch is one strict ``switch_controller`` call, confirmed by reading the
controller states back. A refused or failed switch leaves the owner as it was; the
reason is in the service response and in ``/teleop/mux/owner``.

Topics and services
  /teleop/mux/owner       ArmOwner, 10 Hz
  /teleop/mux/set_owner   SetArmOwner

Threads. "Stopped" is judged from the arrival times of joint states, so their callback
must never wait behind a service call. The process therefore runs three nodes, each on
its own executor thread: ``arm_mux`` (service and owner topic; its callbacks may block),
``arm_mux_observer`` (joint states, goal status, teleop state) and ``arm_mux_client``
(the controller manager clients).
"""

import threading
import time
from typing import Dict, Optional

from action_msgs.msg import GoalStatusArray
from controller_manager_msgs.msg import ControllerManagerActivity
from controller_manager_msgs.srv import ListControllers, SwitchController
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from ranger_xarm_teleop_interfaces.srv import SetArmOwner
from ranger_xarm_teleop_servo.ownership import (any_goal_active, BandTracker, check_switch,
                                                MuxParams,
                                                MuxStatus, outside_activation, Owner, owner_of,
                                                split_of,
                                                StillnessTracker, switch_lists)
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

OWNER_TOPIC = '/teleop/mux/owner'
SET_OWNER_SERVICE = '/teleop/mux/set_owner'
STATE_TOPIC = '/teleop/state'

ACTION_STATUS_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST, depth=1,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
TELEOP_STATE_TIMEOUT_S = 0.5
# As the controller manager publishes ~/activity (reliable, transient local). Its own history
# is one message deep, so a switch reported right after another can replace it before delivery.
ACTIVITY_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST, depth=10,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)
DEFAULT_JOINTS = [f'xarm_joint{i}' for i in range(1, 7)]


class ArmMuxNode(Node):
    """Reports and switches the owner of the arm."""

    def __init__(self, node_name: str = 'arm_mux') -> None:
        super().__init__(node_name)
        self.declare_parameter('moveit_controller', 'xarm6_traj_controller')
        self.declare_parameter('teleop_controller', 'teleop_arm_controller')
        self.declare_parameter('controller_manager', '/controller_manager')
        self.declare_parameter('joints', DEFAULT_JOINTS)
        self.declare_parameter('still_speed_rad_s', MuxParams.still_speed)
        self.declare_parameter('still_duration_s', MuxParams.still_duration)
        self.declare_parameter('joint_state_timeout_s', MuxParams.joint_state_timeout)
        self.declare_parameter('switch_timeout_s', 3.0)
        self.declare_parameter('still_wait_s', 1.0)
        # Gripper teleop: both names, or neither (the switch is then the arm's alone).
        self.declare_parameter('moveit_gripper_controller', '')
        self.declare_parameter('teleop_gripper_controller', '')
        self.declare_parameter('gripper_joints', ['xarm_drive_joint'])
        # Net travel (joint units) within the stillness window that still counts as stopped.
        self.declare_parameter('gripper_still_band', 0.01)
        # A guard, set by the launch whenever teleop motion is disabled (motion_enabled false):
        # every switch to teleop is refused, so the teleop controller stays inactive and ignores
        # anything sent to it. Not a parameter an extra file may set (limits.RESERVED).
        self.declare_parameter('refuse_teleop', False)

        self.moveit_controller = str(self.get_parameter('moveit_controller').value)
        self.teleop_controller = str(self.get_parameter('teleop_controller').value)
        manager = str(self.get_parameter('controller_manager').value).rstrip('/')
        self.joints = [str(j) for j in self.get_parameter('joints').value]
        self.moveit_gripper = str(self.get_parameter('moveit_gripper_controller').value)
        self.teleop_gripper = str(self.get_parameter('teleop_gripper_controller').value)
        if bool(self.moveit_gripper) != bool(self.teleop_gripper):
            raise ValueError('moveit_gripper_controller and teleop_gripper_controller: set '
                             'both (gripper teleop) or neither')
        self.gripper = bool(self.teleop_gripper)
        self.pairs = [(self.moveit_controller, self.teleop_controller)]
        if self.gripper:
            self.pairs.append((self.moveit_gripper, self.teleop_gripper))
        self.split = ''
        self.gripper_joints = []
        if self.gripper:
            self.gripper_joints = [str(j) for j in self.get_parameter('gripper_joints').value]
            if not self.gripper_joints:
                raise ValueError('gripper_joints: gripper teleop needs the gripper joints')
        self.refuse_teleop = bool(self.get_parameter('refuse_teleop').value)
        self.switch_timeout = float(self.get_parameter('switch_timeout_s').value)
        self.still_wait = float(self.get_parameter('still_wait_s').value)
        self.tracker = StillnessTracker(MuxParams(
            still_speed=float(self.get_parameter('still_speed_rad_s').value),
            still_duration=float(self.get_parameter('still_duration_s').value),
            joint_state_timeout=float(self.get_parameter('joint_state_timeout_s').value)),
            names=self.joints)
        self.gripper_tracker = None
        if self.gripper:
            self.gripper_tracker = BandTracker(
                self.tracker.params, float(self.get_parameter('gripper_still_band').value),
                names=self.gripper_joints)

        self.mono = time.monotonic
        self.lock = threading.Lock()
        self.owner = Owner.NONE
        self.status = MuxStatus.OK
        self.detail = ''
        self.switching = False
        self.goal_status: Dict[str, bool] = {}
        self.clutch_engaged = False
        self.teleop_state_arrival: Optional[float] = None
        # The teleop controller is active because this mux switched it there (G-17: anything
        # else is an outside switch, reported and undone once the arm is still).
        self.teleop_by_mux = False
        self.outside = ''               # the report of an outside activation, while it lasts

        sim_time = [self.get_parameter('use_sim_time')]
        self.observer = Node(f'{node_name}_observer', parameter_overrides=sim_time)
        self.client = Node(f'{node_name}_client', parameter_overrides=sim_time)
        self.switch_requested_ns = 0    # ROS time of this mux's newest switch request
        self.owner_pub = self.create_publisher(ArmOwner, OWNER_TOPIC, 10)
        # Deep enough that a stall of this process loses no sample (150 Hz for over 1 s).
        self.observer.create_subscription(JointState, '/joint_states', self._on_joint_state,
                                          200)
        for controller in [name for pair in self.pairs for name in pair]:
            self.observer.create_subscription(
                GoalStatusArray, f'/{controller}/follow_joint_trajectory/_action/status',
                lambda msg, name=controller: self._on_goal_status(name, msg),
                ACTION_STATUS_QOS)
        self.observer.create_subscription(TeleopState, STATE_TOPIC, self._on_teleop_state, 10)
        # Every controller switch, as the controller manager reports it (G-17 bookkeeping).
        # On the client node's thread (no new node: the stack's set of nodes is fixed), so
        # these reports are never queued behind the observer's 150 Hz joint states; that
        # thread is otherwise busy only with this mux's own calls.
        self.client.create_subscription(ControllerManagerActivity, f'{manager}/activity',
                                        self._on_activity, ACTIVITY_QOS)
        self.list_client = self.client.create_client(ListControllers,
                                                     f'{manager}/list_controllers')
        self.switch_client = self.client.create_client(SwitchController,
                                                       f'{manager}/switch_controller')
        self.create_service(SetArmOwner, SET_OWNER_SERVICE, self._on_set_owner)
        self.create_timer(0.1, self._on_timer, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f'controllers: MoveIt {self.moveit_controller}, teleop {self.teleop_controller}'
            + (f'; gripper: MoveIt {self.moveit_gripper}, teleop {self.teleop_gripper}'
               if self.gripper else '; no gripper'))

    # ---- observations ----

    def _on_joint_state(self, msg: JointState) -> None:
        index = {name: i for i, name in enumerate(msg.name)}
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.gripper_tracker is not None and all(j in index for j in self.gripper_joints):
            with self.lock:
                self.gripper_tracker.update(self.mono(), stamp,
                                            [msg.position[index[j]] for j in self.gripper_joints])
        if not all(joint in index for joint in self.joints):
            return
        positions = [msg.position[index[j]] for j in self.joints]
        velocities = None
        if len(msg.velocity) == len(msg.name):
            velocities = [msg.velocity[index[j]] for j in self.joints]
        with self.lock:
            self.tracker.update(self.mono(), stamp, positions, velocities)

    def _on_goal_status(self, controller: str, msg: GoalStatusArray) -> None:
        with self.lock:
            self.goal_status[controller] = any_goal_active(
                [s.status for s in msg.status_list])

    def _on_teleop_state(self, msg: TeleopState) -> None:
        with self.lock:
            self.clutch_engaged = bool(msg.clutch_engaged)
            self.teleop_state_arrival = self.mono()

    def teleop_active(self) -> bool:
        """Return True when the teleop node reports an engaged clutch (fresh report)."""
        if self.teleop_state_arrival is None:
            return False
        if self.mono() - self.teleop_state_arrival > TELEOP_STATE_TIMEOUT_S:
            return False
        return self.clutch_engaged

    # ---- controller manager ----

    def _call(self, client, request):
        """Call a service and wait for the answer; ``None`` on timeout or failure."""
        if not client.wait_for_service(timeout_sec=1.0):
            return None
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(self.switch_timeout):
            future.cancel()
            return None
        try:
            return future.result()
        except Exception as err:  # noqa: B902 - reported by the caller as a failed call
            self.get_logger().error(f'controller manager call failed: {err}')
            return None

    def read_states(self) -> Optional[Dict[str, str]]:
        """Return the controller states; ``None`` when the controller manager did not answer."""
        result = self._call(self.list_client, ListControllers.Request())
        if result is None:
            return None
        return {c.name: c.state for c in result.controller}

    def owner_from(self, states: Optional[Dict[str, str]]) -> Optional[Owner]:
        """Return the owner the controller states imply; ``None`` for no answer."""
        if states is None:
            return None
        split = split_of(states, self.pairs) if self.gripper else ''
        if split and split != self.split:
            self.get_logger().error(split)
        self.split = split
        return owner_of(states, self.moveit_controller, self.teleop_controller,
                        self.moveit_gripper, self.teleop_gripper)

    def read_owner(self) -> Optional[Owner]:
        """Return the owner from the controller manager; ``None`` when it did not answer."""
        return self.owner_from(self.read_states())

    def _on_timer(self) -> None:
        if not self.switching:
            states = self.read_states()
            owner = self.owner_from(states)
            with self.lock:
                if not self.switching:
                    self.owner = owner if owner is not None else Owner.NONE
            if states is not None:
                self.restore_order(states)
        msg = ArmOwner()
        msg.header.stamp = self.get_clock().now().to_msg()
        with self.lock:
            msg.owner = int(Owner.SWITCHING if self.switching else self.owner)
            if self.outside and msg.owner == int(Owner.TELEOP):
                # G-17: a teleop controller this mux did not activate is not teleop's arm.
                # Reported as no owner, so an arming teleop asks the mux, which refuses until
                # the controller is deactivated.
                msg.owner = int(Owner.NONE)
                msg.detail = self.outside
            msg.status = int(self.status)
            msg.detail = msg.detail or self.detail
        msg.gripper = self.gripper
        self.owner_pub.publish(msg)

    # ---- an outside switch (G-17) ----

    def note_states(self, states: Optional[Dict[str, str]]) -> None:
        """
        Forget that the mux switched the arm to teleop once any teleop controller is not active.

        A teleop ownership the mux made has every teleop controller active; one of them seen
        inactive means it ended, by the mux or not, and a later activation is not the mux's.
        """
        if not states:
            return
        # The arm's teleop controller only, as for the outside activation itself: a gripper
        # split stays the split rule's (reported, resolved to MoveIt on request).
        if states.get(self.teleop_controller) != 'active':
            with self.lock:
                self.teleop_by_mux = False

    def _on_activity(self, msg: ControllerManagerActivity) -> None:
        """
        Take in the controller manager's own report of every switch (client thread).

        Polling can miss a deactivation followed by a reactivation between two reads (a read
        not answered, or the timer held up by a request); the manager publishes each switch.
        """
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self.lock:
            older = stamp < self.switch_requested_ns
        if older:
            return                      # from before this mux's own switch: says nothing of it
        self.note_states({c.name: c.state.label for c in msg.controllers})

    def restore_order(self, states: Dict[str, str]) -> None:
        """
        Report a teleop controller activated outside the mux; deactivate it once still.

        Only a deactivation is ever requested here, strict, and only with every arm joint
        still and no goal executing; nothing is activated in response.
        """
        self.note_states(states)
        active = outside_activation(states, [self.teleop_controller], self.teleop_by_mux)
        if not active:
            if self.outside:
                self.get_logger().info('the outside activation of the teleop controller is '
                                       'over')
            self.outside = ''
            return
        report = (f'{", ".join(active)} active without a switch by the mux (an outside switch, '
                  'for example the arm driver after an error)')
        if report != self.outside:
            self.get_logger().error(report + '; it is deactivated once the arm is still; '
                                    'nothing is activated in response')
        self.outside = report
        with self.lock:
            if self.switching:
                return
            goal_active = any(self.goal_status.values())
            status = check_switch(self.mono(), self.tracker, goal_active, False,
                                  self.gripper_tracker)
            if status != MuxStatus.OK:
                self.detail = f'{report}: waiting for the arm to be still ({status.name})'
                return
            self.switching = True
        try:
            request = SwitchController.Request()
            request.activate_controllers = []
            request.deactivate_controllers = active
            request.strictness = SwitchController.Request.STRICT
            request.timeout.sec = int(self.switch_timeout)
            result = self._call(self.switch_client, request)
            ok = result is not None and result.ok
            after = self.read_owner()
            with self.lock:
                if after is not None:
                    self.owner = after
                self.detail = (f'{report}: deactivated' if ok else
                               f'{report}: deactivation failed '
                               f'({"no answer" if result is None else result.message})')
            (self.get_logger().warning if ok else self.get_logger().error)(self.detail)
        finally:
            with self.lock:
                self.switching = False

    # ---- the switch ----

    def _on_set_owner(self, request: SetArmOwner.Request,
                      response: SetArmOwner.Response) -> SetArmOwner.Response:
        status, detail = self.set_owner(int(request.owner))
        with self.lock:
            self.status, self.detail = status, detail
            response.owner = int(self.owner)
        response.status = int(status)
        response.detail = detail
        response.success = status == MuxStatus.OK and response.owner == request.owner
        if status != MuxStatus.OK:
            self.get_logger().error(f'ownership request refused: {detail}')
        return response

    def set_owner(self, wanted: int):
        """Switch to ``wanted``; return ``(status, detail)``. The owner changes only on OK."""
        if wanted not in (Owner.MOVEIT, Owner.TELEOP):
            return MuxStatus.SWITCH_FAILED, f'unknown owner {wanted}'
        if wanted == Owner.TELEOP and self.refuse_teleop:
            return MuxStatus.SWITCH_FAILED, (
                'refused: teleop motion is disabled (motion_enabled is false), so the arm is '
                'never switched to teleop; start the teleop launch with motion_enabled:=true '
                'to allow it')
        with self.lock:
            if self.switching:
                return MuxStatus.SWITCH_FAILED, 'another switch is in progress'
            self.switching = True
        try:
            return self._switch(Owner(wanted))
        finally:
            with self.lock:
                self.switching = False

    def _switch(self, wanted: Owner):
        states = self.read_states()
        current = self.owner_from(states)
        if current is None:
            return MuxStatus.SWITCH_FAILED, 'the controller manager did not answer'
        with self.lock:
            self.owner = current
        self.note_states(states)
        if wanted == Owner.TELEOP:
            # Decided from the states just read, not from the timer's last look: an outside
            # activation that happened since the last tick must be refused too (G-17).
            active = outside_activation(states, [self.teleop_controller], self.teleop_by_mux)
            if active:
                return MuxStatus.SWITCH_FAILED, (
                    f'{", ".join(active)} active without a switch by the mux (an outside '
                    'switch): it is deactivated before teleop may have the arm')
        if wanted == Owner.MOVEIT and outside_activation(states, [self.teleop_controller],
                                                         self.teleop_by_mux):
            # An outside activation is undone by deactivating it (restore_order), never
            # answered by activating MoveIt's controller.
            return MuxStatus.SWITCH_FAILED, (
                f'{self.teleop_controller} active without a switch by the mux (an outside '
                'switch): it is deactivated first, then MoveIt may have the arm')
        if current == wanted:
            return MuxStatus.OK, ''
        # "Stopped" needs an unbroken run of still joint-state samples. If that run is
        # not long enough yet (the arm has just stopped, or the stream had a hole), wait
        # for it a little; the conditions are checked again on every pass, and the
        # switch is only requested on a pass where all of them hold.
        deadline = self.mono() + self.still_wait
        while True:
            with self.lock:
                goal_active = any(self.goal_status.values())
                status = check_switch(self.mono(), self.tracker, goal_active,
                                      self.teleop_active(), self.gripper_tracker)
                still = self.tracker.still_for()
                motion = self.tracker.last_motion
                if (self.gripper_tracker is not None
                        and still >= self.tracker.params.still_duration):
                    motion = self.gripper_tracker.last_motion   # the gripper is what moves
            if status not in (MuxStatus.REFUSED_MOVING, MuxStatus.STALE_JOINT_STATE) \
                    or self.mono() >= deadline:
                break
            time.sleep(0.02)
        if status != MuxStatus.OK:
            return status, {
                MuxStatus.REFUSED_MOVING: (
                    f'the arm is moving or just stopped (still for {still:.2f} s; '
                    f'last motion: {motion})'),
                MuxStatus.REFUSED_GOAL_ACTIVE: 'a trajectory goal is executing',
                MuxStatus.REFUSED_TELEOP_ACTIVE: 'the teleop clutch is engaged',
                MuxStatus.STALE_JOINT_STATE: 'no fresh joint state',
            }[status]
        lists, why = switch_lists(states, self.pairs, wanted)
        if lists is None:
            return MuxStatus.SWITCH_FAILED, why
        request = SwitchController.Request()
        request.activate_controllers, request.deactivate_controllers = lists
        with self.lock:
            self.switch_requested_ns = self.get_clock().now().nanoseconds
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout.sec = int(self.switch_timeout)
        result = self._call(self.switch_client, request)
        after = self.read_owner()
        if after is not None:
            with self.lock:
                self.owner = after
        if result is None or not result.ok:
            reason = 'no answer' if result is None else (result.message or 'refused')
            return MuxStatus.SWITCH_FAILED, f'switch_controller failed: {reason}'
        if after != wanted:
            return MuxStatus.SWITCH_FAILED, (
                f'switch not confirmed: controllers report '
                f'{after.name if after is not None else "nothing"}')
        self.teleop_by_mux = wanted == Owner.TELEOP
        self.get_logger().info(f'arm owner: {wanted.name} (was {current.name})')
        return MuxStatus.OK, ''


def _spin(executor) -> None:
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


def main(args=None) -> None:
    """Run the command mux."""
    rclpy.init(args=args)
    node = ArmMuxNode()
    threads = []
    for other in (node.observer, node.client):
        executor = SingleThreadedExecutor()
        executor.add_node(other)
        thread = threading.Thread(target=_spin, args=(executor,), daemon=True)
        thread.start()
        threads.append(thread)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()
        for thread in threads:
            thread.join(2.0)


if __name__ == '__main__':
    main()
