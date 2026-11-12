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
Teleop servo node: ``/teleop/input/*`` in, tool target and MoveIt Servo pose command out.

A thin wrapper around the pure modules. ``gate`` decides each cycle whether a target
may be published, ``mapping`` computes it, ``calibration`` holds the heading.

Published
  /teleop/target_pose            geometry_msgs/PoseStamped  tool-frame target in the arm base
                                 frame (the recorded action), only while the clutch is engaged
  /teleop/state                  TeleopState, every cycle (50 Hz)
  <servo>/pose_target_cmds       the same target converted to the servo group's tip link by
                                 the fixed tool-to-tip transform, same stamp
  /tf_static                     arm base -> quest_world, at each calibration

Service
  /teleop/disarm                 std_srvs/Trigger: disarm (the clutch releases, the arm goes
                                 back to MoveIt once it has stopped). There is no service
                                 to arm: that is the arm button only.

Clocks. Input staleness, the target cycle and every timeout run on the monotonic clock
of this machine: a frame's age is measured from its local arrival, never by comparing
header stamps from another clock. ``header.stamp`` of the outputs is this node's ROS
time (simulation time under ``use_sim_time``), which is what MoveIt Servo checks.

Safety. ``motion_enabled`` defaults to false (G-1). The limit values have no defaults:
the node exits when one is missing. Values marked simulation-only are refused unless the
robot description shows simulator hardware only.
"""

from collections import deque
from dataclasses import replace
import math
import sys
import threading
import time
from typing import Dict, Optional

from geometry_msgs.msg import PoseStamped, TransformStamped
from moveit_msgs.msg import ServoStatus
from moveit_msgs.srv import ServoCommandType
from ranger_xarm_teleop_bridge.bridge_node import (HEAD_POSE_TOPIC, JOY_QOS, JOY_TOPIC,
                                                   POSE_TOPICS, STATUS_QOS, STATUS_TOPIC)
from ranger_xarm_teleop_interfaces.msg import ArmOwner, InputStatus, TeleopState
from ranger_xarm_teleop_interfaces.srv import SetArmOwner
from ranger_xarm_teleop_servo import params as gate_params
from ranger_xarm_teleop_servo.frames import ARM_BUTTON, CALIBRATE_BUTTON, CLUTCH_BUTTON
from ranger_xarm_teleop_servo.gate import Env, Frame, GateState, Limit, on_frame, tick
from ranger_xarm_teleop_servo.mapping import tool_to_tip
from ranger_xarm_teleop_servo.rotations import Pose, q_from_yaw
from ranger_xarm_teleop_servo.simcheck import NotASimulator, require_simulator
from ranger_xarm_teleop_servo.static_frames import StaticGraph
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_msgs.msg import TFMessage
import tf2_ros

TARGET_TOPIC = '/teleop/target_pose'
STATE_TOPIC = '/teleop/state'
OWNER_TOPIC = '/teleop/mux/owner'
SET_OWNER_SERVICE = '/teleop/mux/set_owner'
DISARM_SERVICE = '/teleop/disarm'

# Poses are published best effort, depth 1; a little depth here lets the pose, joy and
# status of one frame be paired by stamp whatever order they arrive in.
INPUT_POSE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=5,
                            durability=DurabilityPolicy.VOLATILE)
LATCHED_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
TF_STATIC_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                           history=HistoryPolicy.KEEP_LAST, depth=100,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)

PAIR_WAIT_S = 0.02          # how long a status waits for the pose and joy of its frame
OWNER_TIMEOUT_S = 0.5       # mux state older than this counts as "no owner"
SERVICE_TIMEOUT_S = 1.0     # a Servo or mux call without an answer is abandoned
VERIFY_PERIOD_S = 1.0       # how often MoveIt Servo's caps and collision checking are re-read
REQUEST_RETRY_S = 1.0       # pause between ownership requests after a refusal

# MoveIt Servo interface states.
SERVO_UNKNOWN, SERVO_PAUSED, SERVO_STARTING, SERVO_ACTIVE, SERVO_STOPPING = range(5)
SERVO_HALT_CODES = (ServoStatus.HALT_FOR_SINGULARITY, ServoStatus.HALT_FOR_COLLISION)

_REQUIRED_DOUBLES = gate_params.REQUIRED
_OPTIONAL = {
    'filter_cutoff_hz': Parameter.Type.DOUBLE,
    'keep_out_planes': Parameter.Type.DOUBLE_ARRAY,
    'wrist_pivot_offset': Parameter.Type.DOUBLE_ARRAY,
    'pointing_axis': Parameter.Type.DOUBLE_ARRAY,
    'calibration_min_horizontal': Parameter.Type.DOUBLE,
    'tool_pose_timeout_s': Parameter.Type.DOUBLE,
    'servo_ready_timeout_s': Parameter.Type.DOUBLE,
}


class StartupRefused(RuntimeError):
    """The node must not start with this configuration."""


def stamp_key(stamp) -> int:
    """Return a header stamp as integer nanoseconds (the pairing key)."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def pose_of(msg) -> Pose:
    """Return the gate pose of a ``geometry_msgs/Pose``."""
    p, q = msg.position, msg.orientation
    return Pose((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


def fill_pose(msg, pose: Pose) -> None:
    """Write a gate pose into a ``geometry_msgs/Pose``."""
    msg.position.x, msg.position.y, msg.position.z = pose.p
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = pose.q


class TeleopNode(Node):
    """Gate, mapping and the MoveIt Servo interface."""

    def __init__(self, node_name: str = 'teleop_servo', parameter_overrides=None) -> None:
        super().__init__(node_name, parameter_overrides=parameter_overrides)
        self.declare_parameter('motion_enabled', False)                 # G-1
        self.declare_parameter('limits_are_simulation_only', False)
        for name in _REQUIRED_DOUBLES:
            self.declare_parameter(name, Parameter.Type.DOUBLE)         # no default (INV-10)
        for name, kind in _OPTIONAL.items():
            self.declare_parameter(name, kind)
        self.declare_parameter('base_frame', 'xarm_link_base')
        self.declare_parameter('tool_frame', 'xarm_link_tcp')
        self.declare_parameter('servo_tip_frame', 'xarm_link_tcp')
        self.declare_parameter('wrist_frame', '')
        self.declare_parameter('quest_frame', 'quest_world')
        self.declare_parameter('servo_node', '/servo_node')
        self.declare_parameter('robot_description_timeout_s', 15.0)

        self.base_frame = str(self.get_parameter('base_frame').value)
        self.tool_frame = str(self.get_parameter('tool_frame').value)
        self.tip_frame = str(self.get_parameter('servo_tip_frame').value)
        self.wrist_frame = str(self.get_parameter('wrist_frame').value)
        self.quest_frame = str(self.get_parameter('quest_frame').value)
        servo = str(self.get_parameter('servo_node').value).rstrip('/')
        self.simulation_only = bool(self.get_parameter('limits_are_simulation_only').value)

        values = {'motion_enabled': bool(self.get_parameter('motion_enabled').value)}
        for name in list(_REQUIRED_DOUBLES) + list(_OPTIONAL):
            values[name] = self._value_or_none(name)
        try:
            self.params = gate_params.build_params(values)
        except ValueError as err:           # MissingValue included
            raise StartupRefused(str(err)) from err

        self.state = GateState()
        self.mono = time.monotonic
        self.steady = Clock(clock_type=ClockType.STEADY_TIME)

        # Input pairing: messages of one headset frame share one stamp.
        self._poses: Dict[int, Pose] = {}
        self._heads: Dict[int, Pose] = {}
        self._joys: Dict[int, Joy] = {}
        self._pending: deque = deque()

        # Frames. Fixed transforms come from /tf_static only.
        self.static = StaticGraph()
        self.tool_to_tip: Optional[Pose] = None
        self.wrist_known = not self.wrist_frame
        self.tf_buffer = tf2_ros.Buffer()
        # The transforms arrive at the controller rate. They get their own thread so
        # they never queue in front of the headset input in this node's executor.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        # Joint states (controller rate) only tell how fresh the robot side is. They
        # are received by a small node on its own thread for the same reason.
        self.joint_state_arrival: Optional[float] = None
        self.watcher = Node(f'{node_name}_watcher',
                            parameter_overrides=[self.get_parameter('use_sim_time')])
        self.watcher.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self._watch_executor = SingleThreadedExecutor()
        self._watch_executor.add_node(self.watcher)
        self._watch_thread = threading.Thread(target=self._watch, daemon=True)
        self._watch_thread.start()

        # Simulation check and MoveIt Servo verification.
        self.robot_description_checked = not self.simulation_only
        self.started = self.mono()
        self.servo_state = SERVO_UNKNOWN
        self.servo_verified = False
        self.servo_verified_at = -math.inf
        self.servo_call_since: Optional[float] = None
        self.servo_status = -1
        self.servo_refusal = ''

        # Ownership.
        self.owner = ArmOwner.OWNER_NONE
        self.mux_status = ArmOwner.STATUS_OK
        self.owner_arrival: Optional[float] = None
        self.owner_request_since: Optional[float] = None
        self.owner_retry_after = 0.0
        self.published_calibration = None
        self._cycle_started: Optional[float] = None
        self._cycle_took = 0.0
        self._cycle_slowest = ''
        self.last_hold = None

        self.target_pub = self.create_publisher(PoseStamped, TARGET_TOPIC, 10)
        self.state_pub = self.create_publisher(TeleopState, STATE_TOPIC, 10)
        self.servo_pub = self.create_publisher(PoseStamped, f'{servo}/pose_target_cmds', 10)
        self.tf_static_pub = self.create_publisher(TFMessage, '/tf_static', TF_STATIC_QOS)

        self.create_subscription(PoseStamped, POSE_TOPICS['right'], self._on_pose,
                                 INPUT_POSE_QOS)
        self.create_subscription(PoseStamped, HEAD_POSE_TOPIC, self._on_head, INPUT_POSE_QOS)
        self.create_subscription(Joy, JOY_TOPIC, self._on_joy, JOY_QOS)
        self.create_subscription(InputStatus, STATUS_TOPIC, self._on_status, STATUS_QOS)
        self.create_subscription(TFMessage, '/tf_static', self._on_tf_static, TF_STATIC_QOS)
        self.create_subscription(String, '/robot_description', self._on_robot_description,
                                 LATCHED_QOS)
        self.create_subscription(ArmOwner, OWNER_TOPIC, self._on_owner, 10)
        self.create_subscription(ServoStatus, f'{servo}/status', self._on_servo_status, 10)

        self.pause_client = self.create_client(SetBool, f'{servo}/pause_servo')
        self.type_client = self.create_client(ServoCommandType, f'{servo}/switch_command_type')
        self.servo_params_client = self.create_client(GetParameters, f'{servo}/get_parameters')
        self.owner_client = self.create_client(SetArmOwner, SET_OWNER_SERVICE)

        self.create_service(Trigger, DISARM_SERVICE, self._on_disarm)
        self.timer = self.create_timer(self.params.target_period, self._on_timer,
                                       clock=self.steady)
        self.get_logger().info(
            f'motion_enabled={self.params.motion_enabled} '
            f'simulation_only_limits={self.simulation_only} tool_frame={self.tool_frame} '
            f'servo tip={self.tip_frame} base={self.base_frame}')
        if not self.params.motion_enabled:
            self.get_logger().warning('motion_enabled is false: no target will be published')

    def _value_or_none(self, name: str):
        try:
            value = self.get_parameter(name).value
        except rclpy.exceptions.ParameterUninitializedException:
            return None
        if isinstance(value, (list, tuple)) or hasattr(value, 'tolist'):
            return [float(v) for v in value]
        return value

    # ---- input ----

    def _on_pose(self, msg: PoseStamped) -> None:
        self._poses[stamp_key(msg.header.stamp)] = pose_of(msg.pose)
        self._assemble()

    def _on_head(self, msg: PoseStamped) -> None:
        self._heads[stamp_key(msg.header.stamp)] = pose_of(msg.pose)
        self._assemble()

    def _on_joy(self, msg: Joy) -> None:
        self._joys[stamp_key(msg.header.stamp)] = msg
        self._assemble()

    def _on_status(self, msg: InputStatus) -> None:
        self._pending.append((self.mono(), msg))
        self._assemble()

    def _assemble(self) -> None:
        """Hand complete frames to the gate, in order; give up on a part after PAIR_WAIT_S."""
        now = self.mono()
        while self._pending:
            arrival, status = self._pending[0]
            key = stamp_key(status.header.stamp)
            complete = (key in self._joys
                        and (not status.right.pose_present or key in self._poses)
                        and (not status.viewer_present or key in self._heads))
            if not complete and now - arrival < PAIR_WAIT_S:
                return
            self._pending.popleft()
            self.state = on_frame(self.state, self.build_frame(status, arrival), self.params)
            for store in (self._poses, self._heads, self._joys):
                for old in [k for k in store if k <= key]:
                    del store[old]

    def build_frame(self, status: InputStatus, arrival: float) -> Frame:
        """
        Build the gate frame of one status message from the messages with its stamp.

        A pose or joy message that did not arrive with the same stamp is treated as
        absent: without the buttons the grip is unknown, so the controller counts as
        not present.
        """
        key = stamp_key(status.header.stamp)
        joy = self._joys.get(key)
        buttons = list(joy.buttons) if joy is not None else []
        have_buttons = len(buttons) > max(CLUTCH_BUTTON, CALIBRATE_BUTTON, ARM_BUTTON)
        controller = self._poses.get(key) if have_buttons else None
        head = self._heads.get(key)
        return Frame(
            arrival=arrival,
            xr_time=float(status.xr_time_ms) / 1000.0,
            session_id=int(status.session_id),
            seq=int(status.seq),
            reset_count=int(status.reset_count),
            controller=controller,
            controller_present=bool(status.right.pose_present) and controller is not None,
            controller_emulated=bool(status.right.emulated_position),
            head=head,
            head_present=bool(status.viewer_present) and head is not None,
            head_emulated=bool(status.viewer_emulated_position),
            grip=have_buttons and bool(buttons[CLUTCH_BUTTON]),
            calibrate=have_buttons and bool(buttons[CALIBRATE_BUTTON]),
            arm=have_buttons and bool(buttons[ARM_BUTTON]),
        )

    # ---- robot side ----

    def _on_tf_static(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            tr, ro = t.transform.translation, t.transform.rotation
            self.static.add(t.header.frame_id, t.child_frame_id,
                            Pose((tr.x, tr.y, tr.z), (ro.x, ro.y, ro.z, ro.w)))
        self._resolve_fixed_frames()

    def _resolve_fixed_frames(self) -> None:
        if self.tool_to_tip is None:
            found = self.static.transform(self.tool_frame, self.tip_frame)
            if found is not None:
                self.tool_to_tip = found
                self.get_logger().info(
                    f'fixed transform {self.tool_frame} -> {self.tip_frame}: '
                    f'{[round(v, 4) for v in found.p]} m')
        if not self.wrist_known:
            found = self.static.transform(self.tool_frame, self.wrist_frame)
            if found is not None:
                self.wrist_known = True
                self.params = gate_params.replace_wrist_offset(self.params, found.p)
                self.get_logger().info(
                    f'wrist point {self.wrist_frame} at {[round(v, 4) for v in found.p]} m '
                    f'in {self.tool_frame}')

    def close(self) -> None:
        """Stop the helper threads (joint-state watcher, transform listener) and their nodes."""
        for executor in (self._watch_executor, getattr(self.tf_listener, 'executor', None)):
            if executor is not None:
                try:
                    executor.shutdown(timeout_sec=1.0)
                except Exception:  # noqa: B902 - already shut down with the context
                    pass
        self._watch_thread.join(2.0)
        thread = getattr(self.tf_listener, 'dedicated_listener_thread', None)
        if thread is not None:
            thread.join(2.0)
        self.watcher.destroy_node()

    def _watch(self) -> None:
        try:
            self._watch_executor.spin()
        except Exception:  # noqa: B902 - ends with the context; staleness then stops motion
            pass

    def _on_joint_state(self, msg: JointState) -> None:
        self.joint_state_arrival = self.mono()      # one assignment: safe across threads

    def _on_robot_description(self, msg: String) -> None:
        if not self.simulation_only or self.robot_description_checked:
            return
        try:
            plugins = require_simulator(msg.data)
        except NotASimulator as err:
            raise StartupRefused(str(err)) from err
        self.robot_description_checked = True
        self.get_logger().info(f'simulation-only limits accepted: hardware {sorted(set(plugins))}')

    def _on_owner(self, msg: ArmOwner) -> None:
        self.owner = msg.owner
        self.owner_arrival = self.mono()

    def _on_servo_status(self, msg: ServoStatus) -> None:
        self.servo_status = int(msg.code)

    def measured_tool_pose(self):
        """Return ``(pose, age)`` of the tool in the base frame; ``(None, inf)`` if unknown."""
        if self.joint_state_arrival is None:
            return None, math.inf
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, self.tool_frame,
                                                rclpy.time.Time())
        except tf2_ros.TransformException as err:
            self.get_logger().warning(f'no tool pose: {err}', throttle_duration_sec=2.0)
            return None, math.inf
        tr, ro = t.transform.translation, t.transform.rotation
        # Two ages, the larger counts. (1) On the monotonic clock: the local arrival of
        # the newest joint state; a robot side that stops reporting ages in real time
        # whatever any clock says. (2) The transform's own stamp against this node's
        # ROS time (both from the clock the robot side runs on): a transform that is
        # old by its stamp is stale however recently a joint state arrived.
        stamp = rclpy.time.Time.from_msg(t.header.stamp)
        if stamp.nanoseconds <= 0:
            return None, math.inf           # a chain of static transforms only: no measurement
        age = max(self.mono() - self.joint_state_arrival,
                  (self.get_clock().now() - stamp).nanoseconds * 1e-9)
        return Pose((tr.x, tr.y, tr.z), (ro.x, ro.y, ro.z, ro.w)), age

    # ---- the target cycle ----

    def _on_timer(self) -> None:
        now = self.mono()
        # A cycle that starts late means this process was not running its callbacks
        # (a blocking call, or the machine was busy); the gate will see stale input.
        # Say so, with where the previous cycle spent its time (G-11).
        if self._cycle_started is not None and now - self._cycle_started > 3 * \
                self.params.target_period:
            self.get_logger().warning(
                f'late cycle: {(now - self._cycle_started) * 1000.0:.0f} ms since the previous '
                f'one, which took {self._cycle_took * 1000.0:.0f} ms '
                f'(longest step: {self._cycle_slowest})')
        self._cycle_started = now
        marks = [('input', now)]
        self._assemble()
        if self.simulation_only and not self.robot_description_checked:
            limit = float(self.get_parameter('robot_description_timeout_s').value)
            if now - self.started > limit:
                raise StartupRefused(
                    'simulation-only limit values, but no robot description arrived to show '
                    'that this is a simulation')
        owner = self.current_owner(now)
        marks.append(('robot description', self.mono()))
        tool, tool_age = self.measured_tool_pose()
        ready = (self.robot_description_checked and self.tool_to_tip is not None
                 and self.wrist_known)
        env = Env(now=now, owner_is_teleop=owner == ArmOwner.OWNER_TELEOP,
                  tool_pose=tool if ready else None, tool_pose_age=tool_age,
                  servo_active=self.servo_state == SERVO_ACTIVE and self.servo_verified,
                  servo_halted=self.servo_status in SERVO_HALT_CODES)
        marks.append(('tool pose', self.mono()))
        self.state, output = tick(self.state, env, self.params)
        marks.append(('gate', self.mono()))
        for event in output.events:
            self.get_logger().info(event)
        if output.hold != self.last_hold:
            self.get_logger().info(f'state: {"ENGAGED" if output.engaged else output.hold.name}')
            self.last_hold = output.hold
        if not ready:
            self.get_logger().warning(
                f'not ready: need static transforms {self.tool_frame} -> {self.tip_frame}'
                + (f' and -> {self.wrist_frame}' if self.wrist_frame else '')
                + ' and the robot description', throttle_duration_sec=5.0)

        stamp = self.get_clock().now().to_msg()
        if output.target is not None:
            self.publish_target(output.target, stamp)
        marks.append(('publish target', self.mono()))
        self.publish_calibration(stamp)
        self.manage_servo(now, output.want_servo_active)
        marks.append(('servo calls', self.mono()))
        self.manage_ownership(now, owner)
        marks.append(('mux calls', self.mono()))
        self.state_pub.publish(self.build_state(output, owner, tool, stamp))
        marks.append(('publish state', self.mono()))
        steps = [(b[1] - a[1], b[0]) for a, b in zip(marks, marks[1:])]
        self._cycle_took = marks[-1][1] - now
        self._cycle_slowest = '%s %.0f ms' % (max(steps)[1], max(steps)[0] * 1000.0)

    def publish_target(self, target: Pose, stamp) -> None:
        """Publish the tool-frame target and the same target at the servo tip, one stamp."""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        fill_pose(msg.pose, target)
        command = PoseStamped()
        command.header = msg.header
        fill_pose(command.pose, tool_to_tip(target, self.tool_to_tip))
        self.target_pub.publish(msg)
        self.servo_pub.publish(command)

    def publish_calibration(self, stamp) -> None:
        """Publish the static transform arm base -> quest_world when the calibration changed."""
        calibration = self.state.calibration
        if calibration is None or calibration == self.published_calibration:
            return
        self.published_calibration = calibration
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.base_frame
        t.child_frame_id = self.quest_frame
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
         t.transform.rotation.w) = q_from_yaw(calibration.heading)
        self.tf_static_pub.publish(TFMessage(transforms=[t]))

    def build_state(self, output, owner: int, tool: Optional[Pose], stamp) -> TeleopState:
        """Return the TeleopState of this cycle."""
        msg = TeleopState()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        msg.motion_enabled = self.params.motion_enabled
        msg.armed = (self.params.motion_enabled and self.state.want_armed
                     and owner == ArmOwner.OWNER_TELEOP)
        msg.clutch_engaged = output.engaged
        msg.owner = owner
        msg.mux_status = self.mux_status
        msg.hold_reason = int(output.hold)
        limit = output.limit
        if output.engaged and limit == Limit.NONE and self.servo_status > 0:
            limit = Limit.SERVO
        msg.active_limit = int(limit)
        msg.servo_status = self.servo_status
        calibration = self.state.calibration
        frame = self.state.frame
        msg.calibrated = (calibration is not None and frame is not None
                          and calibration.valid_for(frame.session_id, frame.reset_count))
        if calibration is not None:
            msg.calibration_session_id = calibration.session_id
            msg.calibration_reset_count = calibration.reset_count
            msg.calibration_heading = calibration.heading
        msg.input_age = output.input_age if math.isfinite(output.input_age) else -1.0
        msg.input_seq = output.input_seq
        msg.measured_tool_pose_valid = tool is not None
        if tool is not None:
            fill_pose(msg.measured_tool_pose, tool)
        return msg

    # ---- MoveIt Servo ----

    def manage_servo(self, now: float, want_active: bool) -> None:
        """
        Keep MoveIt Servo paused except while the gate wants targets followed.

        Before the first activation its parameters are checked: the speed caps must be
        the required limit values and collision checking must be on.
        """
        if self.servo_call_since is not None:
            if now - self.servo_call_since < SERVICE_TIMEOUT_S:
                return
            self.get_logger().error('MoveIt Servo did not answer; treating it as unknown')
            self.servo_call_since = None
            self.servo_state = SERVO_UNKNOWN
        if self.servo_state == SERVO_UNKNOWN:
            if self.pause_client.service_is_ready():
                self._call(self.pause_client, SetBool.Request(data=True), self._paused)
            return
        if not want_active and self.servo_state in (SERVO_ACTIVE, SERVO_STARTING):
            self.servo_state = SERVO_STOPPING
            self._call(self.pause_client, SetBool.Request(data=True), self._paused)
            return
        # MoveIt Servo's parameters can be changed while it runs, so they are read again
        # every VERIFY_PERIOD_S, also under an engaged clutch (G-7).
        if now - self.servo_verified_at >= VERIFY_PERIOD_S:
            if self.servo_params_client.service_is_ready():
                request = GetParameters.Request(names=[
                    'moveit_servo.scale.linear', 'moveit_servo.scale.rotational',
                    'moveit_servo.check_collisions'])
                self._call(self.servo_params_client, request, self._verified)
                return
            self.servo_verified = False     # cannot be read: not verified
        if not self.servo_verified:
            return
        if want_active and self.servo_state == SERVO_PAUSED:
            self.servo_state = SERVO_STARTING
            self._call(self.type_client,
                       ServoCommandType.Request(command_type=ServoCommandType.Request.POSE),
                       self._type_set)

    def _call(self, client, request, done) -> None:
        self.servo_call_since = self.mono()
        future = client.call_async(request)
        future.add_done_callback(lambda f: self._answered(f, done))

    def _answered(self, future, done) -> None:
        self.servo_call_since = None
        try:
            result = future.result()
        except Exception as err:  # noqa: B902 - a failed call is reported, state unknown
            self.get_logger().error(f'MoveIt Servo call failed: {err}')
            self.servo_state = SERVO_UNKNOWN
            return
        done(result)

    def _paused(self, result) -> None:
        if result.success:
            self.servo_state = SERVO_PAUSED
        else:
            self.get_logger().error(f'MoveIt Servo refused to pause: {result.message}')
            self.servo_state = SERVO_UNKNOWN

    def _verified(self, result) -> None:
        values = result.values
        refusal = ''
        if len(values) != 3:
            refusal = 'MoveIt Servo did not report its speed caps'
        else:
            refusal = self._servo_mismatch(values)
        self.servo_verified_at = self.mono()
        if refusal:
            if refusal != self.servo_refusal:
                self.get_logger().error(refusal + '; teleop is disabled until it is corrected')
            self.servo_refusal = refusal
            self.servo_verified = False     # the gate releases the clutch on the next cycle
            return
        if not self.servo_verified:
            self.get_logger().info(
                'MoveIt Servo verified: speed caps match, collision checking on')
        self.servo_refusal = ''
        self.servo_verified = True

    def _servo_mismatch(self, values) -> str:
        """Return why MoveIt Servo's parameters are not acceptable, or ''."""
        linear, rotational, collisions = (values[0].double_value, values[1].double_value,
                                          values[2].bool_value)
        if abs(linear - self.params.linear_speed_cap) > 1e-9 \
                or abs(rotational - self.params.angular_speed_cap) > 1e-9:
            return (f'MoveIt Servo speed caps ({linear} m/s, {rotational} rad/s) differ from '
                    f'the limit values ({self.params.linear_speed_cap} m/s, '
                    f'{self.params.angular_speed_cap} rad/s)')
        if not collisions:
            return 'MoveIt Servo collision checking is off'
        return ''

    def _type_set(self, result) -> None:
        if not result.success or self.servo_state != SERVO_STARTING:
            if not result.success:
                self.get_logger().error('MoveIt Servo refused pose command mode')
            self.servo_state = SERVO_UNKNOWN
            return
        self._call(self.pause_client, SetBool.Request(data=False), self._unpaused)

    def _unpaused(self, result) -> None:
        if result.success and self.servo_state == SERVO_STARTING:
            self.servo_status = -1          # a halt reported before this start is history
            self.servo_state = SERVO_ACTIVE
        else:
            self.servo_state = SERVO_UNKNOWN

    # ---- ownership ----

    def current_owner(self, now: float) -> int:
        """Return the owner reported by the mux; NONE when its state is too old."""
        if self.owner_arrival is None or now - self.owner_arrival > OWNER_TIMEOUT_S:
            return ArmOwner.OWNER_NONE
        return self.owner

    def manage_ownership(self, now: float, owner: int) -> None:
        """Ask the mux for teleop when armed and for MoveIt when disarmed."""
        if self.owner_request_since is not None:
            if now - self.owner_request_since < 5.0:
                return
            self.owner_request_since = None
        wanted = None
        if self.state.want_armed and self.params.motion_enabled:
            if owner != ArmOwner.OWNER_TELEOP:
                wanted = ArmOwner.OWNER_TELEOP
        elif (owner == ArmOwner.OWNER_TELEOP
              and self.servo_state in (SERVO_PAUSED, SERVO_UNKNOWN)):
            wanted = ArmOwner.OWNER_MOVEIT
        if self.state.want_armed and not self.params.motion_enabled:
            self._disarm('motion_enabled is false')
        if wanted is None or now < self.owner_retry_after:
            return
        if not self.owner_client.service_is_ready():
            self.get_logger().warning('command mux not available', throttle_duration_sec=2.0)
            if wanted == ArmOwner.OWNER_TELEOP:
                self._disarm('the command mux is not available')
            return
        self.owner_request_since = now
        future = self.owner_client.call_async(SetArmOwner.Request(owner=wanted))
        future.add_done_callback(lambda f: self._owner_answered(f, wanted))

    def _owner_answered(self, future, wanted: int) -> None:
        self.owner_request_since = None
        self.owner_retry_after = self.mono() + REQUEST_RETRY_S
        try:
            result = future.result()
        except Exception as err:  # noqa: B902 - reported, ownership unchanged
            self.get_logger().error(f'ownership request failed: {err}')
            self.mux_status = ArmOwner.STATUS_SWITCH_FAILED
            if wanted == ArmOwner.OWNER_TELEOP:
                self._disarm('the ownership request failed')
            return
        self.mux_status = result.status
        if result.success:
            self.owner = result.owner
            self.owner_arrival = self.mono()
            self.get_logger().info(
                'arm owner: ' + ('TELEOP' if wanted == ArmOwner.OWNER_TELEOP else 'MOVEIT'))
        else:
            self.get_logger().error(f'ownership request refused: {result.detail}')
            if wanted == ArmOwner.OWNER_TELEOP:
                # No retry loop: the operator presses the arm button again.
                self._disarm('ownership was refused')

    def _on_disarm(self, request, response):
        """Disarm on request; arming stays a deliberate press of the arm button."""
        was = self.state.want_armed
        self._disarm('requested on ' + DISARM_SERVICE)
        response.success = True
        response.message = 'disarmed' if was else 'was not armed'
        return response

    def _disarm(self, why: str) -> None:
        if self.state.want_armed:
            self.get_logger().warning(f'disarmed: {why}')
            self.state = replace(self.state, want_armed=False)


def main(args=None) -> None:
    """Run the teleop servo node; exit non-zero when it must not start."""
    rclpy.init(args=args)
    logger = rclpy.logging.get_logger('teleop_servo')
    try:
        node = TeleopNode()
    except StartupRefused as err:
        logger.fatal(f'refusing to start: {err}')
        rclpy.try_shutdown()
        sys.exit(2)
    code = 0
    try:
        rclpy.spin(node)
    except StartupRefused as err:
        logger.fatal(f'refusing to run: {err}')
        code = 2
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
