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
Teleop servo node: ``/teleop/input/*`` in, tool target and joint-jog command out.

A thin wrapper around the pure modules. ``gate`` decides each cycle whether a target
may be published, ``mapping`` computes it, ``calibration`` holds the heading, ``solver``
turns it into joint velocities (damped least squares, G-8).

Published
  /teleop/target_pose            geometry_msgs/PoseStamped  tool-frame target in the arm base
                                 frame (the recorded action), only while the clutch is engaged;
                                 after the lead clamp (G-14), so it is what the arm is driven to
  /teleop/state                  TeleopState, every cycle (50 Hz)
  <servo>/delta_joint_cmds       control_msgs/JointJog, joint velocities (rad/s) for the target,
                                 same stamp; one zero command in the cycle the clutch releases
  /tf_static                     arm base -> quest_world, at each calibration

The measured tool pose (engagement, G-14 lead, state) is the forward kinematics of the
newest ``/joint_states`` on the chain of ``/robot_description`` (base frame to the servo
tip link) and the fixed tool-to-tip transform. The solver's reference is MoveIt Servo's
own commanded state: the last point of its newest output on ``servo_output_topic``.

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

from control_msgs.msg import JointJog
from geometry_msgs.msg import PoseStamped, TransformStamped
from moveit_msgs.msg import ServoStatus
from moveit_msgs.srv import ServoCommandType
import numpy as np
from ranger_xarm_teleop_bridge.bridge_node import (HEAD_POSE_TOPIC, JOY_QOS, JOY_TOPIC,
                                                   POSE_TOPICS, STATUS_QOS, STATUS_TOPIC)
from ranger_xarm_teleop_interfaces.msg import ArmOwner, InputStatus, TeleopState
from ranger_xarm_teleop_interfaces.srv import SetArmOwner
from ranger_xarm_teleop_servo import params as gate_params
from ranger_xarm_teleop_servo import solver as dls
from ranger_xarm_teleop_servo.frames import ARM_BUTTON, CALIBRATE_BUTTON, CLUTCH_BUTTON
from ranger_xarm_teleop_servo.gate import Env, Frame, GateState, Limit, on_frame, tick
from ranger_xarm_teleop_servo.kinematics import ChainError
from ranger_xarm_teleop_servo.rotations import (inverse, Pose, pose_from_matrix, pose_to_matrix,
                                                q_between, q_from_yaw, v_norm, v_sub)
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
from trajectory_msgs.msg import JointTrajectory

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
# Statuses in which MoveIt Servo stops the arm (a joint at its bound halts all joints).
SERVO_HALT_CODES = (ServoStatus.HALT_FOR_SINGULARITY, ServoStatus.HALT_FOR_COLLISION,
                    ServoStatus.JOINT_BOUND)
# Joint-jog commands never go through Servo's singularity scaling: one of these statuses
# means Servo is not in joint-jog mode.
SERVO_NOT_JOINT_JOG_CODES = (ServoStatus.DECELERATE_FOR_APPROACHING_SINGULARITY,
                             ServoStatus.HALT_FOR_SINGULARITY,
                             ServoStatus.DECELERATE_FOR_LEAVING_SINGULARITY)
# Servo's output must follow the first command within this time and then never be older:
# the solver's reference is that output, never the delayed measurement (D2).
SERVO_OUTPUT_TIMEOUT_S = 0.1
# Commanded and measured tool pose further apart than this many lead limits: the arm is
# not following what Servo commands; the solver stops (HOLD_SOLVER).
DIVERGENCE_LEADS = 2.0
SOLVER_LIMITS = {dls.LIMIT_JOINT_POSITION: Limit.JOINT_POSITION,
                 dls.LIMIT_JOINT_VELOCITY: Limit.JOINT_VELOCITY,
                 dls.LIMIT_TOOL_SPEED: Limit.TOOL_SPEED}
# Solver tuning parameters (not safety limits) and their SolverParams fields.
SOLVER_TUNING = {'solver_gain': 'gain', 'damping_threshold': 'damping_threshold',
                 'max_damping': 'max_damping', 'joint_limit_margin': 'limit_margin',
                 'joint_limit_zone': 'limit_zone',
                 'joint_velocity_fraction': 'velocity_fraction'}
FUTURE_STAMP_TOLERANCE_S = 0.05     # a joint state stamped further ahead than this is not trusted
# MoveIt Servo parameters read and checked (``_servo_mismatch``), in this order.
SERVO_CHECKED = ('moveit_servo.check_collisions', 'moveit_servo.command_in_type',
                 'moveit_servo.joint_limit_margins', 'moveit_servo.incoming_command_timeout')

_REQUIRED_DOUBLES = gate_params.REQUIRED
_OPTIONAL = {
    'filter_cutoff_hz': Parameter.Type.DOUBLE,
    'view_cone_tilt_deg': Parameter.Type.DOUBLE,
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
        self.declare_parameter('servo_output_topic', '/teleop_arm_controller/joint_trajectory')
        self.declare_parameter('robot_description_timeout_s', 15.0)
        for name in SOLVER_TUNING:
            self.declare_parameter(name, Parameter.Type.DOUBLE)
        # Joint velocity limits besides the URDF's (MoveIt's joint_limits.yaml); the smaller
        # of the two is used.
        self.declare_parameter('joint_velocity_limit_names', Parameter.Type.STRING_ARRAY)
        self.declare_parameter('joint_velocity_limits', Parameter.Type.DOUBLE_ARRAY)

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
        tuning = {}
        for name, field in SOLVER_TUNING.items():
            value = self._value_or_none(name)
            if value is not None:
                tuning[field] = float(value)
        try:
            self.solver_params = dls.SolverParams(
                linear_speed_cap=self.params.linear_speed_cap,
                angular_speed_cap=self.params.angular_speed_cap,
                period=self.params.target_period, **tuning)
        except ValueError as err:
            raise StartupRefused(f'solver: {err}') from err
        names = self._value_or_none('joint_velocity_limit_names', strings=True) or []
        limits = self._value_or_none('joint_velocity_limits') or []
        if len(names) != len(limits):
            raise StartupRefused('joint_velocity_limit_names and joint_velocity_limits differ '
                                 'in length')
        self.velocity_limits = dict(zip(names, limits))

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
        # Kinematics: the chain from the robot description, with the tool offset.
        self.urdf: Optional[str] = None
        self.arm: Optional[dls.Arm] = None
        self.arm_error = ''
        # Joint states and MoveIt Servo's output arrive at the controller rate. A second
        # node on its own executor thread receives both, so they never queue in front of
        # the headset input in this node's executor. Each callback stores one tuple (a
        # single assignment, safe across the threads).
        self.joint_state_arrival: Optional[float] = None
        self.joint_state = None             # (arrival, stamp ns, {name: position})
        self.servo_output = None            # (arrival, {name: position}) of the last point
        self.watcher = Node(f'{node_name}_watcher',
                            parameter_overrides=[self.get_parameter('use_sim_time')])
        self.watcher.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.watcher.create_subscription(
            JointTrajectory, str(self.get_parameter('servo_output_topic').value),
            self._on_servo_output, 10)
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
        self.servo_status_at = -math.inf    # monotonic arrival of the latest status
        self.servo_refusal = ''
        self.servo_mode_wrong = False       # a status only possible outside joint-jog mode

        # Solver, per engagement (G-10).
        self.first_command_at: Optional[float] = None   # monotonic, first command sent
        self.output_confirmed = False       # Servo's output followed the first command
        self.confirmed_at = math.inf        # arrival of that first following output
        self.own_reference: Optional[np.ndarray] = None
        self.commanding = False             # a non-final command was sent last cycle
        self.solution: Optional[dls.Solution] = None
        self.lead = (0.0, 0.0)

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
        self.servo_pub = self.create_publisher(JointJog, f'{servo}/delta_joint_cmds', 10)
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

    def _value_or_none(self, name: str, strings: bool = False):
        try:
            value = self.get_parameter(name).value
        except rclpy.exceptions.ParameterUninitializedException:
            return None
        if isinstance(value, (list, tuple)) or hasattr(value, 'tolist'):
            return [str(v) if strings else float(v) for v in value]
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
        self._build_arm()

    def _build_arm(self) -> None:
        """Build the solver's arm once the robot description and the tool offset are known."""
        if self.arm is not None or self.urdf is None or self.tool_to_tip is None:
            return
        try:
            self.arm = dls.arm_from_urdf(self.urdf, self.base_frame, self.tip_frame,
                                         pose_to_matrix(inverse(self.tool_to_tip)),
                                         self.velocity_limits)
        except (ChainError, ValueError) as err:
            if str(err) != self.arm_error:
                self.get_logger().error(f'no kinematic chain for the solver: {err}')
            self.arm_error = str(err)
            return
        self.get_logger().info(
            f'solver chain {self.base_frame} -> {self.tip_frame}: {self.arm.joint_names}, '
            f'velocity limits {[round(v, 3) for v in self.arm.velocity]} rad/s')

    def close(self) -> None:
        """Stop the watcher thread (joint states and transforms) and its node."""
        try:
            self._watch_executor.shutdown(timeout_sec=1.0)
        except Exception:  # noqa: B902 - already shut down with the context
            pass
        self._watch_thread.join(2.0)
        self.watcher.destroy_node()

    def _watch(self) -> None:
        try:
            self._watch_executor.spin()
        except Exception:  # noqa: B902 - ends with the context; staleness then stops motion
            pass

    def _on_joint_state(self, msg: JointState) -> None:
        arrival = self.mono()
        self.joint_state = (arrival, stamp_key(msg.header.stamp),
                            dict(zip(msg.name, msg.position)))
        self.joint_state_arrival = arrival

    def _on_servo_output(self, msg: JointTrajectory) -> None:
        if msg.points:
            self.servo_output = (self.mono(), dict(zip(msg.joint_names, msg.points[-1].positions)))

    def _on_robot_description(self, msg: String) -> None:
        if self.urdf is None:
            self.urdf = msg.data
            self._build_arm()
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
        self.servo_status_at = self.mono()

    def measured(self):
        """
        Return ``(joints, tool pose, age)`` measured; ``(None, None, inf)`` if unknown.

        The tool pose is the forward kinematics of the newest joint state, not a lookup
        in the transform tree (which is published at a lower rate).
        """
        latest = self.joint_state
        if latest is None or self.arm is None:
            return None, None, math.inf
        arrival, stamp_ns, positions = latest
        try:
            q = np.array([positions[name] for name in self.arm.joint_names], dtype=float)
        except KeyError as err:
            self.get_logger().warning(f'joint state without {err}', throttle_duration_sec=2.0)
            return None, None, math.inf
        if not np.all(np.isfinite(q)):
            return None, None, math.inf
        # Two ages, the larger counts. (1) On the monotonic clock: the local arrival of
        # the newest joint state; a robot side that stops reporting ages in real time
        # whatever any clock says. (2) The joint state's own stamp against this node's
        # ROS time (both from the clock the robot side runs on).
        if stamp_ns <= 0:
            return None, None, math.inf
        stamp_age = (self.get_clock().now().nanoseconds - stamp_ns) * 1e-9
        if stamp_age < -FUTURE_STAMP_TOLERANCE_S:
            # Stamped ahead of this node's clock (clock reset or skew): its age cannot
            # be judged, so it is not used.
            self.get_logger().warning(
                f'joint state is stamped {-stamp_age:.2f} s in the future; not used',
                throttle_duration_sec=2.0)
            return None, None, math.inf
        age = max(self.mono() - arrival, stamp_age)
        return q, pose_from_matrix(dls.tool_pose(self.arm, q)), age

    def servo_reference(self, now: float) -> Optional[np.ndarray]:
        """Return MoveIt Servo's commanded state if it followed this engagement's commands."""
        output = self.servo_output
        if (output is None or self.first_command_at is None or output[0] <= self.first_command_at
                or now - output[0] > SERVO_OUTPUT_TIMEOUT_S or self.arm is None):
            return None
        try:
            return np.array([output[1][name] for name in self.arm.joint_names], dtype=float)
        except KeyError:
            return None

    def reference(self, now: float) -> Optional[np.ndarray]:
        """
        Return the solver's reference: Servo's commanded state, or within the grace our own.

        MoveIt Servo integrates each step from the last point of its rolling window and
        publishes every point but that last one, re-applying the newest joint command each
        cycle. The state the next command is added to is therefore the newest published
        point plus one step of the last command sent.
        """
        servo = self.servo_reference(now)
        if servo is not None:
            if not self.output_confirmed:
                self.output_confirmed = True
                self.confirmed_at = self.servo_output[0]
            if self.solution is not None:
                servo = servo + self.solution.qdot * self.params.target_period
            return servo
        if (not self.output_confirmed and self.first_command_at is not None
                and now - self.first_command_at <= SERVO_OUTPUT_TIMEOUT_S):
            return self.own_reference
        return None

    def solver_checks(self, now: float, tool: Optional[Pose]):
        """
        Return ``(solver_ready, servo_following)`` for this cycle.

        Only while engaged: MoveIt Servo's output must follow the commands (within the
        grace after the first one, then never older than the timeout), and the tool at
        Servo's commanded state must agree with the measured one.
        """
        ready = self.arm is not None
        if not self.state.engaged or self.first_command_at is None:
            return ready, True
        reference = self.reference(now)
        if reference is None:
            self.get_logger().error(
                'MoveIt Servo output stopped following the joint commands (last status '
                f'{self.servo_status}); clutch released', throttle_duration_sec=1.0)
            return ready, False
        if tool is not None:
            commanded = pose_from_matrix(dls.tool_pose(self.arm, reference))
            apart = v_norm(v_sub(commanded.p, tool.p))
            turned = q_between(commanded.q, tool.q)
            if (apart > DIVERGENCE_LEADS * self.params.max_lead_translation
                    or turned > DIVERGENCE_LEADS * self.params.max_lead_rotation):
                self.get_logger().error(
                    f'commanded and measured tool pose {apart * 1000.0:.0f} mm, '
                    f'{math.degrees(turned):.1f} deg apart: the arm is not following; '
                    'clutch released', throttle_duration_sec=1.0)
                return False, True
        return ready, True

    def servo_status_current(self) -> bool:
        """
        Return True when the latest status belongs to this engagement's commands.

        After an unpause the first status can still carry the code from before the pause.
        Servo publishes its status after its output in the same cycle, so a status counts
        only if it arrived after the first output that followed this engagement's commands.
        """
        return self.output_confirmed and self.servo_status_at >= self.confirmed_at

    def servo_halted(self) -> bool:
        """Return True on a halt status that belongs to this engagement's commands."""
        return self.servo_status_current() and self.servo_status in SERVO_HALT_CODES

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
        joints, tool, tool_age = self.measured()
        ready = (self.robot_description_checked and self.tool_to_tip is not None
                 and self.wrist_known)
        solver_ready, following = self.solver_checks(now, tool)
        if self.servo_status_current() and self.servo_status in SERVO_NOT_JOINT_JOG_CODES:
            if not self.servo_mode_wrong:
                self.get_logger().error(
                    f'MoveIt Servo reports status {self.servo_status}, which joint-jog '
                    'commands never cause: it is not in joint-jog mode; teleop is disabled '
                    'until it is activated again')
            self.servo_mode_wrong = True
        env = Env(now=now, owner_is_teleop=owner == ArmOwner.OWNER_TELEOP,
                  tool_pose=tool if ready else None, tool_pose_age=tool_age,
                  servo_active=(self.servo_state == SERVO_ACTIVE and self.servo_verified
                                and not self.servo_mode_wrong and following),
                  servo_halted=self.servo_halted(), solver_ready=solver_ready)
        marks.append(('tool pose', self.mono()))
        engaged_before = self.state.engaged
        self.state, output = tick(self.state, env, self.params)
        marks.append(('gate', self.mono()))
        for event in output.events:
            self.get_logger().info(event)
        if output.hold != self.last_hold:
            self.get_logger().info(f'state: {"ENGAGED" if output.engaged else output.hold.name}')
            self.last_hold = output.hold
        if not ready or self.arm is None:
            self.get_logger().warning(
                f'not ready: need static transforms {self.tool_frame} -> {self.tip_frame}'
                + (f' and -> {self.wrist_frame}' if self.wrist_frame else '')
                + ' and the robot description with the chain '
                + f'{self.base_frame} -> {self.tip_frame}', throttle_duration_sec=5.0)

        stamp = self.get_clock().now().to_msg()
        if output.target is not None:
            self.publish_target(output.target, stamp)
            self.command(output, joints, tool, stamp, now, engaged_before)
        else:
            self.stop_command(stamp)
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
        """Publish the tool-frame target (the recorded action)."""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        fill_pose(msg.pose, target)
        self.target_pub.publish(msg)

    def publish_joint_command(self, velocities, stamp) -> None:
        """Publish one joint-jog command (rad/s, the arm's joint order)."""
        msg = JointJog()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        msg.joint_names = list(self.arm.joint_names)
        msg.velocities = [float(v) for v in velocities]
        self.servo_pub.publish(msg)

    def command(self, output, joints, tool: Pose, stamp, now: float,
                engaged_before: bool) -> None:
        """Solve this cycle's target into joint velocities and send them, same stamp."""
        if not engaged_before or self.first_command_at is None:
            # Engagement: the reference starts at the measured joint state (G-10).
            self.first_command_at = now
            self.output_confirmed = False
            self.confirmed_at = math.inf
            reference = joints
        else:
            reference = self.reference(now)
        if reference is None or joints is None:
            # Cannot happen after the checks of this cycle; never send a guess.
            self.get_logger().error('no solver reference; sending a stop')
            self.stop_command(stamp, force=True)
            return
        target = pose_to_matrix(output.target)
        self.solution = dls.solve(self.arm, reference, target, self.solver_params,
                                  output.feedforward)
        self.own_reference = reference + self.solution.qdot * self.params.target_period
        self.lead = (v_norm(v_sub(output.target.p, tool.p)), q_between(output.target.q, tool.q))
        self.publish_joint_command(self.solution.qdot, stamp)
        self.commanding = True

    def stop_command(self, stamp, force: bool = False) -> None:
        """
        Send one zero command in the cycle the clutch releases (INV-8).

        Joint-jog commands are velocities: without it MoveIt Servo would keep the last
        one until its command timeout.
        """
        if (self.commanding or force) and self.arm is not None:
            self.publish_joint_command([0.0] * len(self.arm.joint_names), stamp)
        self.commanding = False
        self.first_command_at = None
        self.output_confirmed = False
        self.confirmed_at = math.inf
        self.own_reference = None
        self.solution = None
        self.lead = (0.0, 0.0)

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
        solution = self.solution if output.target is not None else None
        if limit == Limit.NONE and solution is not None:
            limit = SOLVER_LIMITS.get(solution.limit, Limit.NONE)
        if (output.engaged and limit == Limit.NONE and self.servo_status_current()
                and self.servo_status > 0):
            limit = Limit.SERVO
        msg.active_limit = int(limit)
        if solution is not None:
            msg.conditioning = float(solution.conditioning)
            msg.damped = bool(solution.damped)
            msg.damping = float(solution.damping)
            msg.solver_error_translation = float(solution.error_translation)
            msg.solver_error_rotation = float(solution.error_rotation)
            msg.lead_translation, msg.lead_rotation = (float(v) for v in self.lead)
            msg.commanded_linear_speed = float(solution.linear_speed)
            msg.commanded_angular_speed = float(solution.angular_speed)
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

        Before the first activation, and every VERIFY_PERIOD_S after, its parameters are
        checked (``_servo_mismatch``). Every activation sets joint-jog command mode.
        """
        if self.servo_call_since is not None:
            if now - self.servo_call_since < SERVICE_TIMEOUT_S:
                return
            self.get_logger().error('MoveIt Servo did not answer; treating it as unknown')
            self.servo_call_since = None
            self.servo_state = SERVO_UNKNOWN
            self.servo_verified = False
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
                request = GetParameters.Request(names=list(SERVO_CHECKED))
                self._call(self.servo_params_client, request, self._verified)
                return
            self.servo_verified = False     # cannot be read: not verified
        if not self.servo_verified:
            return
        if want_active and self.servo_state == SERVO_PAUSED:
            self.servo_state = SERVO_STARTING
            self._call(self.type_client,
                       ServoCommandType.Request(command_type=ServoCommandType.Request.JOINT_JOG),
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
        if len(values) != len(SERVO_CHECKED):
            refusal = 'MoveIt Servo did not report its parameters'
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
                'MoveIt Servo verified: collision checking on, speed units, joint limit '
                "margins inside the solver's, command timeout within the input timeout")
        self.servo_refusal = ''
        self.servo_verified = True

    def _servo_mismatch(self, values) -> str:
        """
        Return why MoveIt Servo's parameters are not acceptable, or ''.

        Collision checking on (G-6 layer 1); velocities in rad/s (``speed_units``);
        every joint limit margin narrower than the solver's, so Servo's halt is a backstop
        normal operation never reaches (INV-3); a command timeout no longer than the input
        timeout, so Servo stops by itself if this node stops sending.
        """
        collisions = values[0].bool_value
        units = values[1].string_value
        margins = list(values[2].double_array_value)
        timeout = values[3].double_value
        if not collisions:
            return 'MoveIt Servo collision checking is off'
        if units != 'speed_units':
            return f'MoveIt Servo command_in_type is {units!r}, not speed_units'
        if not margins or max(margins) >= self.solver_params.limit_margin:
            return (f'MoveIt Servo joint_limit_margins {margins} are not all narrower than '
                    f"the solver's {self.solver_params.limit_margin} rad")
        if not 0.0 < timeout <= self.params.input_timeout:
            return (f'MoveIt Servo incoming_command_timeout {timeout} s is not within the '
                    f'input timeout {self.params.input_timeout} s')
        return ''

    def _type_set(self, result) -> None:
        if not result.success or self.servo_state != SERVO_STARTING:
            if not result.success:
                self.get_logger().error('MoveIt Servo refused joint-jog command mode')
            self.servo_state = SERVO_UNKNOWN
            return
        self.servo_mode_wrong = False
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
