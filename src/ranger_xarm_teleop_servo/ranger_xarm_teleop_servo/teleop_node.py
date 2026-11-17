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
  /teleop/gripper/command        GripperCommand, only with gripper_enabled: the gripper command
                                 (part of the recorded action), in every cycle where gripper
                                 teleop is active, same stamp as that cycle's target and state

The measured tool pose (engagement, G-14 lead, state) is the forward kinematics of the
newest ``/joint_states`` on the chain of ``/robot_description`` (base frame to the servo
tip link) and the fixed tool-to-tip transform. The solver's reference is MoveIt Servo's
own commanded state: the last point of its newest output on ``servo_output_topic``.

Services
  /teleop/disarm                 std_srvs/Trigger: disarm (the clutch releases, the arm goes
                                 back to MoveIt once it has stopped). There is no service
                                 to arm: that is the arm button only.
  /teleop/joint_path             GetJointPath: the measured joint path recorded while armed
                                 (for the retreat tool)
  /teleop/keepout_info           std_srvs/Trigger: the link keep-out model in use (margin,
                                 fingerprint, spheres, pairs left out and why)

Link keep-out (G-15). A model of the arm's links and the robot's fixed structure is built
from ``/robot_description`` when it first arrives (``keepout.py``); a description it cannot
follow refuses the start. A later description with other geometry, or MoveIt Servo checking
a robot description with other geometry, disables teleop until restart.

Clocks. Input staleness, the target cycle and every timeout run on the monotonic clock
of this machine: a frame's age is measured from its local arrival, never by comparing
header stamps from another clock. ``header.stamp`` of the outputs is this node's ROS
time (simulation time under ``use_sim_time``), which is what MoveIt Servo checks.

Gripper (``gripper.py``). The right trigger opens and closes the gripper while armed, clutch
or not. ``gripper_enabled`` defaults to false; off, no gripper publisher exists and the
trigger is not read. The opening the gripper's controller holds arrives on
``/teleop/gripper/held`` (sensor_msgs/JointState, from the backend's gripper relay).

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

from ament_index_python.packages import get_package_share_directory
from control_msgs.msg import JointJog
from geometry_msgs.msg import PoseStamped, TransformStamped
from moveit_msgs.msg import ServoStatus
from moveit_msgs.srv import ServoCommandType
import numpy as np
from ranger_xarm_teleop_bridge.bridge_node import (HEAD_POSE_TOPIC, JOY_TOPIC,
                                                   POSE_TOPICS, STATUS_QOS, STATUS_TOPIC)
from ranger_xarm_teleop_interfaces.msg import ArmOwner, GripperCommand, InputStatus, TeleopState
from ranger_xarm_teleop_interfaces.srv import GetJointPath, SetArmOwner
from ranger_xarm_teleop_servo import gripper as grip
from ranger_xarm_teleop_servo import keepout as ko
from ranger_xarm_teleop_servo import params as gate_params
from ranger_xarm_teleop_servo import rigid_pairs
from ranger_xarm_teleop_servo import solver as dls
from ranger_xarm_teleop_servo.frames import (ARM_BUTTON, CALIBRATE_BUTTON, CLUTCH_BUTTON,
                                             SCALE_BUTTON, TRIGGER_AXIS)
from ranger_xarm_teleop_servo.gate import (Env, Frame, frame_fault, GateState, Hold, Limit,
                                           on_frame, tick)
from ranger_xarm_teleop_servo.kinematics import ChainError
from ranger_xarm_teleop_servo.retreat import PathRecorder
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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

TARGET_TOPIC = '/teleop/target_pose'
STATE_TOPIC = '/teleop/state'
OWNER_TOPIC = '/teleop/mux/owner'
SET_OWNER_SERVICE = '/teleop/mux/set_owner'
DISARM_SERVICE = '/teleop/disarm'
JOINT_PATH_SERVICE = '/teleop/joint_path'
KEEPOUT_INFO_SERVICE = '/teleop/keepout_info'
GRIPPER_COMMAND_TOPIC = '/teleop/gripper/command'
GRIPPER_HELD_TOPIC = '/teleop/gripper/held'
KEEPOUT_SLOW_S = 0.002      # a keep-out step longer than this is reported
KEEPOUT_FLOOR_SLACK = 1e-4  # m: the clearance falling more below its floor is reported
TARGET_PERIOD_MS = 20.0     # for the reference-age summary only
IDLE_CLEARANCE_CYCLES = 5   # the measured clearance is computed this seldom ...
ENGAGED_CLEARANCE_CYCLES = 25   # ... and this seldom while engaged (the solver has its own)

# Poses and joy are published best effort, depth 1; a little depth here lets the pose, joy
# and status of one frame be paired by stamp whatever order they arrive in, and keeps them
# while this node's executor is busy for longer than a frame period (frames come every
# 11 ms at 90 Hz). With depth 1 a stall of one frame period replaces one frame's buttons
# with the next frame's; the frame then counts as having no controller (POSE_ABSENT).
INPUT_POSE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=5,
                            durability=DurabilityPolicy.VOLATILE)
INPUT_JOY_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                           history=HistoryPolicy.KEEP_LAST, depth=5,
                           durability=DurabilityPolicy.VOLATILE)
LATCHED_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
TF_STATIC_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                           history=HistoryPolicy.KEEP_LAST, depth=100,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)

PAIR_WAIT_S = 0.02          # how long a status waits for the pose and joy of its frame
SLOW_CYCLE_S = 0.01         # a cycle longer than this is reported
LATE_START_S = 0.03         # a cycle starting this long after the previous one is counted
CYCLE_REPORT_S = 60.0       # how often the slow and late cycles are summed up in the log
INPUT_GAP_WINDOW_S = 0.2    # the longest gap between input callbacks is kept this long
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
# the solver's reference is that output, never the delayed measurement.
SERVO_OUTPUT_TIMEOUT_S = 0.1
# Commanded and measured tool pose further apart than this many lead limits: the arm is
# not following what Servo commands; the solver stops (HOLD_SOLVER).
DIVERGENCE_LEADS = 2.0
# Stuck near a collision. MoveIt Servo scales every joint command by one factor that
# depends only on the distance to the nearest collision, down to 0.001 at contact, so
# no direction leads away. While Servo reports that deceleration, a window of
# STUCK_WINDOW_S in which at least STUCK_COMMANDED_* of motion was commanded and the tool
# moved less than STUCK_MOVED_* and less than STUCK_FRACTION of what was commanded releases
# the clutch as a Servo halt. An arm that came to rest near a collision with a small error
# left (the hand still, Servo passing a few per cent) commands too little to count: it is
# not trapped, and moving the hand away still moves it.
STUCK_WINDOW_S = 0.5
STUCK_MOVED_M = 0.001
STUCK_MOVED_RAD = math.radians(1.0)
STUCK_COMMANDED_M = 0.02
STUCK_COMMANDED_RAD = 0.2
STUCK_FRACTION = 0.02
SOLVER_LIMITS = {dls.LIMIT_JOINT_POSITION: Limit.JOINT_POSITION,
                 dls.LIMIT_JOINT_VELOCITY: Limit.JOINT_VELOCITY,
                 dls.LIMIT_TOOL_SPEED: Limit.TOOL_SPEED,
                 dls.LIMIT_LINK_KEEPOUT: Limit.LINK_KEEPOUT}
# Solver tuning parameters (not safety limits) and their SolverParams fields.
SOLVER_TUNING = {'solver_gain': 'gain', 'damping_threshold': 'damping_threshold',
                 'max_damping': 'max_damping',
                 'orientation_max_damping': 'orientation_max_damping',
                 'joint_limit_margin': 'limit_margin',
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
    'min_arming_conditioning': Parameter.Type.DOUBLE,
    'translation_scale_fine': Parameter.Type.DOUBLE,
    'translation_scale_coarse': Parameter.Type.DOUBLE,
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


class CycleCounter:
    """Slow and late target cycles, summed up for the log once per CYCLE_REPORT_S."""

    def __init__(self) -> None:
        self.since: Optional[float] = None
        self.slow_count = 0
        self.slowest = (0.0, '', '')        # seconds, longest step, joint command parts
        self.late_count = 0
        self.latest = 0.0                   # longest interval between cycle starts
        self.keepout = []                   # seconds of each solver keep-out step (wall)
        self.keepout_cpu = []               # the same, CPU time of the node's thread
        self.measured_clearance = []        # seconds of each clearance at the measured joints
        self.reference_ages = []            # seconds: MoveIt Servo's output at each solve

    def reference_age(self, age: float) -> None:
        """Record how old MoveIt Servo's newest output was when the solver used it."""
        self.reference_ages.append(age)

    def keepout_step(self, took: float, cpu: float) -> None:
        """Record one solver keep-out step (model and, if bound, the fix): wall and CPU s."""
        self.keepout.append(took)
        self.keepout_cpu.append(cpu)

    def measured_clearance_step(self, took: float) -> None:
        """Record one keep-out clearance computed at the measured joints (s)."""
        self.measured_clearance.append(took)

    def started(self, now: float, interval: float) -> None:
        """Count a cycle that started ``interval`` seconds after the previous one."""
        if self.since is None:
            self.since = now
        if interval > LATE_START_S:
            self.late_count += 1
            self.latest = max(self.latest, interval)

    def slow(self, took: float, slowest: str, parts: str) -> None:
        """Count a cycle that took ``took`` seconds."""
        self.slow_count += 1
        if took > self.slowest[0]:
            self.slowest = (took, slowest, parts)

    def report(self, now: float, final: bool = False) -> str:
        """
        Return the summary when one is due and anything was counted, then start again.

        ``final``: at shutdown, whatever was counted since the last summary.
        """
        if self.since is None or (now - self.since < CYCLE_REPORT_S and not final):
            return ''
        text = ''
        if self.keepout:
            steps = np.array(self.keepout) * 1000.0
            cpu = np.array(self.keepout_cpu) * 1000.0
            text = (f'link keep-out: {len(steps)} steps, median {np.median(steps):.2f} ms, '
                    f'p99 {np.percentile(steps, 99):.2f} ms, longest {steps.max():.2f} ms '
                    f'(CPU of this thread: median {np.median(cpu):.2f} ms, p99 '
                    f'{np.percentile(cpu, 99):.2f} ms). ')
        if self.measured_clearance:
            steps = np.array(self.measured_clearance) * 1000.0
            text += (f'clearance at the measured joints: {len(steps)} times, median '
                     f'{np.median(steps):.2f} ms, p99 {np.percentile(steps, 99):.2f} ms. ')
        if self.reference_ages:
            ages = np.array(self.reference_ages) * 1000.0
            text += (f'solver reference (MoveIt Servo output) age: median {np.median(ages):.0f} '
                     f'ms, p99 {np.percentile(ages, 99):.0f} ms, longest {ages.max():.0f} ms, '
                     f'{int(np.sum(ages > 2.5 * TARGET_PERIOD_MS))} of {len(ages)} over 2.5 '
                     'cycles. ')
        if self.slow_count or self.late_count:
            took, slowest, parts = self.slowest
            text += (f'last {now - self.since:.0f} s: {self.slow_count} slow cycles '
                     f'(over {SLOW_CYCLE_S * 1000.0:.0f} ms'
                     + (f'; longest {took * 1000.0:.0f} ms, {slowest}'
                        + (f' ({parts})' if parts else '') if self.slow_count else '')
                     + f'), {self.late_count} cycles started over '
                     f'{LATE_START_S * 1000.0:.0f} ms after the previous one'
                     + (f' (longest {self.latest * 1000.0:.0f} ms)' if self.late_count
                        else ''))
        self.__init__()
        self.since = now
        return text


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
        # G-15: PENDING for hardware, so no default; the cell is a model resolution.
        self.declare_parameter('link_keepout_margin_m', Parameter.Type.DOUBLE)
        self.declare_parameter('link_keepout_cell_m', ko.DEFAULT_CELL)
        for name in SOLVER_TUNING:
            self.declare_parameter(name, Parameter.Type.DOUBLE)
        # Joint velocity limits besides the URDF's (MoveIt's joint_limits.yaml); the smaller
        # of the two is used.
        self.declare_parameter('joint_velocity_limit_names', Parameter.Type.STRING_ARRAY)
        self.declare_parameter('joint_velocity_limits', Parameter.Type.DOUBLE_ARRAY)
        # Gripper teleop: off by default (G-1); its range and speed have no default.
        self.declare_parameter('gripper_enabled', False)
        self.declare_parameter('gripper_joint', grip.DEFAULTS['gripper_joint'])
        for name in grip.REQUIRED_WHEN_ENABLED + ('gripper_close_threshold',
                                                  'gripper_open_threshold',
                                                  'gripper_reference_timeout_s'):
            self.declare_parameter(name, Parameter.Type.DOUBLE)

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
        gripper_values = {'gripper_enabled': bool(self.get_parameter('gripper_enabled').value),
                          'gripper_joint': str(self.get_parameter('gripper_joint').value)}
        for name in grip.REQUIRED_WHEN_ENABLED + ('gripper_close_threshold',
                                                  'gripper_open_threshold',
                                                  'gripper_reference_timeout_s'):
            gripper_values[name] = self._value_or_none(name)
        try:
            self.gripper_params = grip.build_gripper_params(gripper_values,
                                                            self.params.target_period)
        except ValueError as err:           # MissingGripperValue included
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
        margin = self._value_or_none('link_keepout_margin_m')
        if margin is None:
            raise StartupRefused('required safety values without a default are not set: '
                                 'link_keepout_margin_m')
        if not (math.isfinite(margin) and margin > 0.0):
            raise StartupRefused(f'link_keepout_margin_m: {margin!r} is not a positive number')
        self.keepout_margin = float(margin)
        self.keepout_cell = float(self.get_parameter('link_keepout_cell_m').value)
        if not (math.isfinite(self.keepout_cell) and 0.002 < self.keepout_cell <= 0.05):
            raise StartupRefused('link_keepout_cell_m must be within (0.002, 0.05] m')

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
        # Link keep-out (G-15): the model, what it was built from, and why it cannot be used.
        self.keepout: Optional[ko.Model] = None
        self.keepout_fault = ''
        self._keepout_tried = False
        self.keepout_floor = math.inf       # clearance the next cycle may not fall below
        self.servo_description_compared = False
        self.servo_pairs_checked = False    # rigid self-collision pairs reported once
        self.servo_rigid_pairs: Optional[list] = None   # None: not derived (yet)
        self.path = PathRecorder()
        self._idle_count = 0
        self._idle_clearance = None
        self._recording = False
        self.servo_description_pending = False
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
        self.solver_fault = ''                # why the last command could not be made

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
        self._command_steps = []            # (step, seconds) inside the last joint command
        self._cycles = CycleCounter()
        # Input callbacks: arrival of the last one and the longest gaps between them, to
        # tell an executor that was busy from input that did not arrive.
        self._input_at: Optional[float] = None
        self._input_gaps: deque = deque()   # (arrival, gap before it)
        self._incomplete = False            # the last frame lacked a part
        # Stuck near a collision: (start, tool pose, commanded travel m, rad) of the window.
        self._stuck_window = None
        self.stuck_near_collision = False
        self.last_hold = None

        # Gripper teleop.
        self.gripper_state = grip.GripperState()
        self.gripper_output: Optional[grip.GripperOutput] = None
        self.gripper_tripped = False        # a frame since the last cycle tripped (any clutch)
        self.owner_gripper = False          # the mux swaps the gripper with the arm
        self.gripper_owner_since: Optional[float] = None    # monotonic, teleop owns it since
        self.gripper_held = None            # (arrival, position) from the gripper relay
        self.gripper_keepout: Optional[ko.Model] = None     # the pairs the fingers move
        self.gripper_moved: frozenset = frozenset()          # the gripper joint and its mimics
        self.gripper_last_hold = None
        self.gripper_pub = None
        if self.gripper_params.enabled:
            self.gripper_pub = self.create_publisher(GripperCommand, GRIPPER_COMMAND_TOPIC, 10)
            self.watcher.create_subscription(JointState, GRIPPER_HELD_TOPIC,
                                             self._on_gripper_held, 10)

        self.target_pub = self.create_publisher(PoseStamped, TARGET_TOPIC, 10)
        self.state_pub = self.create_publisher(TeleopState, STATE_TOPIC, 10)
        self.servo_pub = self.create_publisher(JointJog, f'{servo}/delta_joint_cmds', 10)
        self.tf_static_pub = self.create_publisher(TFMessage, '/tf_static', TF_STATIC_QOS)

        self.create_subscription(PoseStamped, POSE_TOPICS['right'], self._on_pose,
                                 INPUT_POSE_QOS)
        self.create_subscription(PoseStamped, HEAD_POSE_TOPIC, self._on_head, INPUT_POSE_QOS)
        self.create_subscription(Joy, JOY_TOPIC, self._on_joy, INPUT_JOY_QOS)
        self.create_subscription(InputStatus, STATUS_TOPIC, self._on_status, STATUS_QOS)
        self.create_subscription(TFMessage, '/tf_static', self._on_tf_static, TF_STATIC_QOS)
        self.create_subscription(String, '/robot_description', self._on_robot_description,
                                 LATCHED_QOS)
        self.create_subscription(ArmOwner, OWNER_TOPIC, self._on_owner, 10)
        self.create_subscription(ServoStatus, f'{servo}/status', self._on_servo_status, 10)

        self.pause_client = self.create_client(SetBool, f'{servo}/pause_servo')
        self.type_client = self.create_client(ServoCommandType, f'{servo}/switch_command_type')
        self.servo_params_client = self.create_client(GetParameters, f'{servo}/get_parameters')
        self.servo_description_client = self.create_client(GetParameters,
                                                           f'{servo}/get_parameters')
        self.owner_client = self.create_client(SetArmOwner, SET_OWNER_SERVICE)

        self.create_service(Trigger, DISARM_SERVICE, self._on_disarm)
        self.create_service(GetJointPath, JOINT_PATH_SERVICE, self._on_joint_path)
        self.create_service(Trigger, KEEPOUT_INFO_SERVICE, self._on_keepout_info)
        self.timer = self.create_timer(self.params.target_period, self._on_timer,
                                       clock=self.steady)
        self.get_logger().info(
            f'motion_enabled={self.params.motion_enabled} '
            f'simulation_only_limits={self.simulation_only} tool_frame={self.tool_frame} '
            f'servo tip={self.tip_frame} base={self.base_frame} translation scale '
            f'fine {self.params.translation_scale_fine:g} coarse '
            f'{self.params.translation_scale_coarse:g} (largest allowed '
            f'{self.params.max_translation_scale:g})')
        if not self.params.motion_enabled:
            self.get_logger().warning('motion_enabled is false: no target will be published')
        gp = self.gripper_params
        if gp.enabled:
            self.get_logger().info(
                f'gripper teleop: joint {gp.joint}, open {gp.open_position:g}, closed '
                f'{gp.closed_position:g}, at most {gp.max_speed:g}/s; trigger closes above '
                f'{gp.close_threshold:g}, opens below {gp.open_threshold:g}')
        else:
            self.get_logger().info('gripper teleop disabled: the trigger has no effect')

    def _value_or_none(self, name: str, strings: bool = False):
        try:
            value = self.get_parameter(name).value
        except rclpy.exceptions.ParameterUninitializedException:
            return None
        if isinstance(value, (list, tuple)) or hasattr(value, 'tolist'):
            return [str(v) if strings else float(v) for v in value]
        return value

    # ---- input ----

    def _input_arrived(self) -> float:
        """Note an input callback; return its arrival (monotonic)."""
        now = self.mono()
        if self._input_at is not None:
            self._input_gaps.append((now, now - self._input_at))
        self._input_at = now
        while self._input_gaps and now - self._input_gaps[0][0] > INPUT_GAP_WINDOW_S:
            self._input_gaps.popleft()
        return now

    def _on_pose(self, msg: PoseStamped) -> None:
        self._input_arrived()
        self._poses[stamp_key(msg.header.stamp)] = pose_of(msg.pose)
        self._assemble()

    def _on_head(self, msg: PoseStamped) -> None:
        self._input_arrived()
        self._heads[stamp_key(msg.header.stamp)] = pose_of(msg.pose)
        self._assemble()

    def _on_joy(self, msg: Joy) -> None:
        self._input_arrived()
        self._joys[stamp_key(msg.header.stamp)] = msg
        self._assemble()

    def _on_status(self, msg: InputStatus) -> None:
        self._pending.append((self._input_arrived(), msg))
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
            self._diagnose_frame(status, key, now - arrival)
            frame = self.build_frame(status, arrival)
            if (self.gripper_params.enabled
                    and frame_fault(self.state.frame, frame) != Hold.NONE):
                # Every frame counts, clutch held or not: a session or sequence fault
                # between two cycles freezes the gripper in the next one.
                self.gripper_tripped = True
            self.state = on_frame(self.state, frame, self.params)
            for store in (self._poses, self._heads, self._joys):
                for old in [k for k in store if k <= key]:
                    del store[old]

    def _diagnose_frame(self, status: InputStatus, key: int, waited: float) -> None:
        """
        Log which part of a frame is missing, once per run of such frames, while it counts.

        It counts when the clutch is engaged or the grip was last seen held: then the frame
        releases the clutch, and the log says whether the input did not arrive (the status
        says so, or a message is missing after the wait) and how long this node's input
        callbacks had paused just before (a long pause: this process was busy).
        """
        missing = []
        if not status.right.pose_present:
            missing.append('the status reports no controller pose')
        else:
            if key not in self._poses:
                missing.append('no pose message')
            if key not in self._joys:
                missing.append('no joy message (buttons unknown)')
        if not missing:
            self._incomplete = False
            return
        held = self.state.engaged or bool(self.state.buttons and self.state.buttons[0])
        if held and not self._incomplete:
            gap = max((g for _, g in self._input_gaps), default=0.0)
            self.get_logger().warning(
                f'frame seq {status.seq} incomplete: {", ".join(missing)}; waited '
                f'{waited * 1000.0:.0f} ms; longest pause between input callbacks in the last '
                f'{INPUT_GAP_WINDOW_S * 1000.0:.0f} ms: {gap * 1000.0:.0f} ms',
                throttle_duration_sec=1.0)
        self._incomplete = True

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
        have_buttons = len(buttons) > max(CLUTCH_BUTTON, CALIBRATE_BUTTON, ARM_BUTTON,
                                          SCALE_BUTTON)
        controller = self._poses.get(key) if have_buttons else None
        head = self._heads.get(key)
        trigger = None
        if joy is not None and len(joy.axes) > TRIGGER_AXIS:
            trigger = float(joy.axes[TRIGGER_AXIS])
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
            scale_toggle=have_buttons and bool(buttons[SCALE_BUTTON]),
            trigger=trigger,
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

    def _resolver(self):
        return ko.make_resolver(get_package_share_directory)

    def _build_keepout(self) -> None:
        """Build the link keep-out model from the robot description (refuses the start)."""
        self._keepout_tried = True
        started = time.perf_counter()
        try:
            self.keepout = ko.build(self.urdf, self._resolver(), self.base_frame,
                                    self.tip_frame, self.keepout_margin,
                                    cell=self.keepout_cell)
        except ChainError as err:
            # No arm chain: the solver has none either and teleop stays not ready.
            self.get_logger().error(f'no link keep-out model: {err}')
            return
        except (ko.ModelError, ValueError) as err:
            raise StartupRefused(f'link keep-out: {err}') from err
        took = time.perf_counter() - started
        model = self.keepout
        self.get_logger().info(
            f'link keep-out: margin {model.margin * 1000.0:.1f} mm, {model.sphere_count} '
            f'spheres on {len(model.bodies)} moving links, largest radius '
            f'{model.largest_radius * 1000.0:.1f} mm (inflation below twice that per covered '
            f'side), {len(model.pair_list)} pairs, fixed '
            f'links {model.units.names}, built in {took:.2f} s, fingerprint '
            f'{model.fingerprint[:12]}')
        for item in model.excluded:
            extra = ('' if item.reason == 'adjacent' else
                     f' (clearance {item.low * 1000.0:.1f} to {item.high * 1000.0:.1f} mm over '
                     'the sampled postures)')
            self.get_logger().warning(
                f'link keep-out: {item.moving} - {item.other} left out: {item.reason}{extra}')
        if model.not_covered:
            self.get_logger().warning(
                'link keep-out: not covered (behind joints that are neither fixed nor arm '
                f'joints): {model.not_covered}')
        if not model.pair_list:
            self.get_logger().warning('link keep-out: the description has no pair of arm '
                                      'and fixed collision geometry; nothing is kept out')
        if self.gripper_params.enabled:
            joint = self.gripper_params.joint
            indices = model.pairs_moved_by(joint)
            self.gripper_moved = frozenset(
                {joint} | {name for name, (source, _, _) in model.extra_joints.items()
                           if source == joint})
            self.gripper_keepout = model.subset(indices) if indices else None
            self.get_logger().info(
                f'link keep-out: {len(indices)} pairs move with {joint}; a gripper step '
                'toward the margin is not taken')

    def _check_gripper_joint(self) -> None:
        """Refuse the start when the gripper's range is not within its joint's limits."""
        if not self.gripper_params.enabled:
            return
        try:
            lower, upper = grip.joint_limits(self.urdf, self.gripper_params.joint)
            grip.check_joint_range(self.gripper_params, lower, upper)
        except ValueError as err:
            raise StartupRefused(f'gripper: {err}') from err

    def _compare_description(self, text: str, source: str) -> None:
        """Disable teleop if ``text`` has other geometry than the keep-out model's."""
        try:
            other = ko.fingerprint(text, self._resolver(), self.base_frame, self.tip_frame,
                                   self.keepout_margin)
        except (ko.ModelError, ChainError, ValueError) as err:
            other = f'unusable: {err}'
        if other == self.keepout.fingerprint:
            self.get_logger().info(f'{source} has the same geometry as the link keep-out model')
            return
        fault = (f'{source} does not match the link keep-out model built at start-up '
                 '(other collision geometry or kinematics): teleop is disabled until it is '
                 'restarted')
        if fault != self.keepout_fault:
            self.get_logger().error(fault)
        self.keepout_fault = fault

    def _build_arm(self) -> None:
        """Build the solver's arm once the robot description and the tool offset are known."""
        if self.urdf is not None and self.keepout is None and not self._keepout_tried:
            self._build_keepout()
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
        """Log the cycle summary since the last one; stop the watcher thread and its node."""
        try:
            summary = self._cycles.report(self.mono(), final=True)
            if summary:
                self.get_logger().info(f'at shutdown: {summary}')
        except Exception:  # noqa: B902 - a log line must not stop the shutdown
            pass
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
        if self.simulation_only and not self.robot_description_checked:
            try:
                plugins = require_simulator(msg.data)
            except NotASimulator as err:
                raise StartupRefused(str(err)) from err
            self.robot_description_checked = True
            self.get_logger().info(
                f'simulation-only limits accepted: hardware {sorted(set(plugins))}')
        if self.urdf is None:
            self.urdf = msg.data
            self._check_gripper_joint()
            self._build_arm()
        elif msg.data != self.urdf and self.keepout is not None:
            self._compare_description(msg.data, 'a later /robot_description')

    def _on_owner(self, msg: ArmOwner) -> None:
        self.owner = msg.owner
        self.owner_gripper = bool(msg.gripper)
        self.owner_arrival = self.mono()

    def _on_gripper_held(self, msg: JointState) -> None:
        try:
            value = float(msg.position[list(msg.name).index(self.gripper_params.joint)])
        except (ValueError, IndexError):
            return
        self.gripper_held = (self.mono(), value)

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
        ready = (self.arm is not None and self.keepout is not None and not self.solver_fault
                 and not self.keepout_fault and not self.servo_description_pending)
        if not self.state.engaged or self.first_command_at is None:
            return ready, True
        reference = self.reference(now)
        if reference is not None and not np.all(np.isfinite(reference)):
            self.get_logger().error('MoveIt Servo output not finite; clutch released',
                                    throttle_duration_sec=1.0)
            return False, True
        if reference is None:
            if self.servo_status == ServoStatus.HALT_FOR_COLLISION:
                # Servo publishes nothing while it is halted for a collision: re-engaging
                # cannot move the arm out of the contact it stopped in.
                self.get_logger().error(
                    f'MoveIt Servo is halted for a collision (status {self.servo_status}) and '
                    'publishes nothing for the joint commands: teleop cannot move the arm out '
                    'of the contact; it has to be moved out directly; clutch released',
                    throttle_duration_sec=1.0)
            else:
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

    def arming_conditioning(self, joints) -> float:
        """Return the solver's orientation conditioning at ``joints`` (NaN if unknown)."""
        if joints is None or self.arm is None:
            return math.nan
        if self.keepout_fault:
            self.get_logger().error(f'arm refused: {self.keepout_fault}')
            return math.nan
        try:
            self.keepout_extra()
        except ko.ModelError as err:
            self.get_logger().error(f'arm refused: link keep-out: {err}')
            return math.nan
        try:
            return float(dls.levels(self.arm, joints, self.solver_params).orientation)
        except (np.linalg.LinAlgError, ValueError) as err:
            self.get_logger().error(f'orientation conditioning for the arm request failed: {err}')
            return math.nan

    def stuck_check(self, now: float, tool: Optional[Pose]) -> bool:
        """
        Return True once the arm is stuck near a collision (see STUCK_WINDOW_S).

        Judged on the measured tool pose against the joint commands sent: while MoveIt
        Servo reports collision deceleration for this engagement's commands, a window
        starts; any motion of the tool beyond STUCK_MOVED_* restarts it; after
        STUCK_WINDOW_S with at least STUCK_COMMANDED_* commanded and less than
        STUCK_FRACTION of it done, the arm is stuck. Reset on release.
        """
        if self.stuck_near_collision:
            return True
        decelerating = (self.state.engaged and self.servo_status_current()
                        and self.servo_status == ServoStatus.DECELERATE_FOR_COLLISION)
        if not decelerating or tool is None:
            self._stuck_window = None
            return False
        window = self._stuck_window
        if window is None or (v_norm(v_sub(tool.p, window[1].p)) > STUCK_MOVED_M
                              or q_between(tool.q, window[1].q) > STUCK_MOVED_RAD):
            self._stuck_window = (now, tool, 0.0, 0.0)
            return False
        start, pose, linear, angular = window
        if self.solution is not None:
            linear += self.solution.linear_speed * self.params.target_period
            angular += self.solution.angular_speed * self.params.target_period
        self._stuck_window = (start, pose, linear, angular)
        moved = v_norm(v_sub(tool.p, pose.p))
        turned = q_between(tool.q, pose.q)
        trapped = ((linear >= STUCK_COMMANDED_M and moved < STUCK_FRACTION * linear)
                   or (angular >= STUCK_COMMANDED_RAD and turned < STUCK_FRACTION * angular))
        if now - start >= STUCK_WINDOW_S and trapped:
            self.stuck_near_collision = True
            self.get_logger().error(
                f'stuck near a collision: MoveIt Servo slows every direction by its collision '
                f'scale; commanded {linear * 1000.0:.0f} mm and {math.degrees(angular):.0f} deg '
                f'in {now - start:.1f} s, the tool moved {moved * 1000.0:.1f} mm and '
                f'{math.degrees(turned):.1f} deg; clutch released. MoveIt cannot plan from here '
                'either; the arm has to be moved out directly')
            return True
        return False

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
        if self._cycle_started is not None:
            self._cycles.started(now, now - self._cycle_started)
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
        stuck = self.stuck_check(now, tool)
        recording = (joints is not None and self.state.want_armed
                     and owner == ArmOwner.OWNER_TELEOP and self.joint_state is not None)
        if recording:
            if not self._recording and self.path.resume(joints):
                self.get_logger().info('joint path: the arm moved while teleop was disarmed; '
                                       'the earlier path was dropped')
            self.path.add(self.joint_state[1] * 1e-9, joints)
        self._recording = recording
        blocked = (tuple(float(v) for v in self.solution.blocked)
                   if self.solution is not None and self.solution.blocked is not None
                   else (0.0, 0.0, 0.0))
        env = Env(now=now, owner_is_teleop=owner == ArmOwner.OWNER_TELEOP,
                  tool_pose=tool if ready else None, tool_pose_age=tool_age,
                  servo_active=(self.servo_state == SERVO_ACTIVE and self.servo_verified
                                and not self.servo_mode_wrong and following),
                  servo_halted=self.servo_halted() or stuck, solver_ready=solver_ready,
                  arming_conditioning=(self.arming_conditioning(joints)
                                       if self.state.arm_pending else math.inf),
                  blocked=blocked)
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
        marks.append(('log', self.mono()))

        stamp = self.get_clock().now().to_msg()
        self._command_steps = []
        if output.target is not None:
            if self.command(output, joints, tool, stamp, now, engaged_before):
                published = self.mono()
                self.publish_target(output.target, stamp)
                self._command_steps.append(('publish target', self.mono() - published))
        else:
            self.stop_command(stamp)
            self.solver_fault = ''
        marks.append(('target and joint command', self.mono()))
        if self.gripper_params.enabled:
            self.gripper_cycle(now, owner, joints, stamp, tool_age)
            marks.append(('gripper', self.mono()))
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
        parts = ', '.join(f'{name} {seconds * 1000.0:.1f} ms'
                          for name, seconds in self._command_steps)
        if self._cycle_took > SLOW_CYCLE_S:
            # Input of a headset frame waits at most PAIR_WAIT_S for its parts; a cycle this
            # long can make a frame count as having no pose.
            self._cycles.slow(self._cycle_took, self._cycle_slowest, parts)
            self.get_logger().warning(
                f'slow cycle: {self._cycle_took * 1000.0:.0f} ms (longest step: '
                f'{self._cycle_slowest}' + (f'; joint command: {parts}' if parts else '') + ')',
                throttle_duration_sec=1.0)
        summary = self._cycles.report(now)
        if summary:
            self.get_logger().info(summary)

    def gripper_measured(self):
        """Return ``(position, stamp ns)`` of the gripper joint's newest state, or None."""
        latest = self.joint_state
        if latest is None:
            return None
        value = latest[2].get(self.gripper_params.joint)
        if value is None or not math.isfinite(value):
            return None
        return float(value), latest[1]

    def gripper_clearance(self, joints):
        """Return the gripper's keep-out clearance as a function of its commanded position."""
        model = self.gripper_keepout
        if model is None:
            return None             # the fingers have no pair: nothing to keep apart
        latest = self.joint_state
        if joints is None or latest is None:
            return lambda position: math.nan
        others = {k: v for k, v in latest[2].items() if k not in self.gripper_moved}

        def clearance(position: float) -> float:
            values = dict(others)
            values[self.gripper_params.joint] = position
            try:
                return float(model.clearance(joints, model.extra_values(values))[0])
            except (ko.ModelError, ValueError):
                return math.nan

        return clearance

    def gripper_cycle(self, now: float, owner: int, joints, stamp,
                      joint_age: float = math.inf) -> None:
        """
        Run the gripper logic for this cycle and publish its command with ``stamp``.

        ``joint_age``: the age of the newest joint state, by local arrival and by its stamp,
        the larger (``measured``); beyond ``tool_pose_timeout`` the gripper freezes.
        """
        owned = owner == ArmOwner.OWNER_TELEOP and self.owner_gripper
        if not owned:
            self.gripper_owner_since = None
        elif self.gripper_owner_since is None:
            self.gripper_owner_since = now
        held = None
        latest = self.gripper_held
        if owned and latest is not None and latest[0] > self.gripper_owner_since:
            held = latest[1]        # produced after teleop's controller became active
        # Joint states recent enough to check the finger keep-out: the same age, by arrival
        # and by stamp, and the same timeout as the arm's measured tool pose.
        joints_fresh = joint_age <= self.params.tool_pose_timeout
        measured = None
        sample = self.gripper_measured()
        if sample is not None and joints_fresh:
            measured = sample[0]
        frame = self.state.frame
        inputs = grip.GripperInputs(
            now=now,
            armed=(self.params.motion_enabled and self.state.want_armed
                   and not self.state.arm_pending),
            owner=owned,
            input_ok=grip.input_ok(frame is not None,
                                   now - frame.arrival if frame is not None else math.inf,
                                   self.params.input_timeout, self.gripper_tripped),
            trigger=frame.trigger if frame is not None else None,
            held=held, measured=measured, clearance=self.gripper_clearance(joints),
            margin=self.keepout_margin, joints_fresh=joints_fresh)
        self.gripper_tripped = False
        self.gripper_state, out = grip.step(self.gripper_state, inputs, self.gripper_params)
        self.gripper_output = out
        for event in out.events:
            self.get_logger().info(event)
        if out.hold != self.gripper_last_hold:
            self.get_logger().info(f'gripper: {out.hold.name}')
            self.gripper_last_hold = out.hold
        if out.publish:
            msg = GripperCommand()
            msg.header.stamp = stamp
            msg.header.frame_id = self.base_frame
            msg.joint_name = self.gripper_params.joint
            msg.position = float(out.command)
            msg.request = int(out.request)
            msg.reference_source = int(out.source)
            self.gripper_pub.publish(msg)

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
                engaged_before: bool) -> bool:
        """
        Solve this cycle's target into joint velocities and send them, same stamp.

        Return False, after sending a zero command, when no valid command exists; the
        target is then not published and the gate releases the clutch (HOLD_SOLVER).
        """
        started = self.mono()
        if not engaged_before or self.first_command_at is None:
            # Engagement: the reference starts at the measured joint state (G-10).
            self.first_command_at = now
            self.output_confirmed = False
            self.confirmed_at = math.inf
            reference = joints
        else:
            reference = self.reference(now)
            newest = self.servo_output
            if reference is not None and self.output_confirmed and newest is not None:
                self._cycles.reference_age(now - newest[0])
        solving = self.mono()
        self._command_steps.append(('reference', solving - started))
        why = ''
        if reference is None or joints is None:
            why = 'no solver reference'
        elif not np.all(np.isfinite(reference)):
            why = 'solver reference not finite'
        else:
            try:
                extra = self.keepout_extra()
                solution = dls.solve(self.arm, reference, pose_to_matrix(output.target),
                                     self.solver_params, output.feedforward,
                                     keepout=self.keepout, extra=extra)
            except ko.ModelError as err:
                solution, why = None, f'link keep-out: {err}'
            except (np.linalg.LinAlgError, ValueError) as err:
                solution, why = None, f'solver failed: {err}'
            if solution is not None and not np.all(np.isfinite(solution.qdot)):
                why = 'joint command not finite'
        self._command_steps.append(('solve', self.mono() - solving))
        if not why:
            self.keepout_watch(solution, engaged_before)
        if why:
            # Never send a guess: stop now, and the gate releases the clutch next cycle.
            self.get_logger().error(f'{why}; sending a stop, clutch released')
            self.stop_command(stamp, force=True)
            self.solver_fault = why
            return False
        self.solution = solution
        self.own_reference = reference + solution.qdot * self.params.target_period
        self.lead = (v_norm(v_sub(output.target.p, tool.p)), q_between(output.target.q, tool.q))
        publishing = self.mono()
        self.publish_joint_command(solution.qdot, stamp)
        self._command_steps.append(('publish joint command', self.mono() - publishing))
        self.commanding = True
        return True

    def keepout_extra(self) -> Optional[dict]:
        """Return the measured values of the non-arm joints the keep-out model needs."""
        if self.keepout is None:
            return None
        latest = self.joint_state
        return self.keepout.extra_values(latest[2] if latest is not None else {})

    def keepout_watch(self, solution, engaged_before: bool) -> None:
        """Time the keep-out step, and report a clearance that fell below its floor."""
        if self.keepout is None:
            return
        took = solution.keepout_seconds
        self._cycles.keepout_step(took, solution.keepout_cpu_seconds)
        self._command_steps.append(('link keep-out', took))
        if took > KEEPOUT_SLOW_S:
            self.get_logger().warning(f'link keep-out step took {took * 1000.0:.1f} ms',
                                      throttle_duration_sec=1.0)
        clearance = solution.keepout_clearance
        if not engaged_before:
            self.keepout_floor = math.inf
        if (math.isfinite(self.keepout_floor)
                and clearance < self.keepout_floor - KEEPOUT_FLOOR_SLACK):
            self.get_logger().error(
                f'link keep-out: clearance {clearance * 1000.0:.2f} mm '
                f'({solution.keepout_pair[0]} - {solution.keepout_pair[1]}) is below the '
                f'floor of the cycle before, {self.keepout_floor * 1000.0:.2f} mm',
                throttle_duration_sec=1.0)
        self.keepout_floor = (min(clearance, self.keepout.margin) if math.isfinite(clearance)
                              else math.inf)

    def _on_joint_path(self, request, response):
        times, postures = self.path.path()
        response.path.joint_names = list(self.arm.joint_names) if self.arm is not None else []
        if len(times) == 0:
            response.message = ('no path recorded: teleop has not moved the arm since this '
                                'node started')
            return response
        first = int(round(times[0] * 1e9))
        response.path.header.stamp.sec, response.path.header.stamp.nanosec = divmod(
            first, 1_000_000_000)
        response.path.header.frame_id = self.base_frame
        for t, q in zip(times, postures):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in q]
            offset = int(round((t - times[0]) * 1e9))
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(
                offset, 1_000_000_000)
            response.path.points.append(point)
        response.message = (f'{len(times)} points over {times[-1] - times[0]:.1f} s'
                            + ('; older points were dropped (60 s kept)'
                               if self.path.dropped_for_age else ''))
        return response

    def _on_keepout_info(self, request, response):
        model = self.keepout
        if model is None:
            response.success = False
            response.message = 'no link keep-out model yet (no robot description)'
            return response
        lines = [f'margin {model.margin * 1000.0:.1f} mm; cell {model.cell * 1000.0:.0f} mm; '
                 f'{model.sphere_count} spheres; largest radius '
                 f'{model.largest_radius * 1000.0:.1f} mm (inflation below twice that per '
                 f'covered side); fingerprint {model.fingerprint}',
                 f'fixed links: {model.units.names}',
                 f'pairs: {len(model.pair_list)}']
        for item in model.excluded:
            lines.append(f'left out: {item.moving} - {item.other}: {item.reason}'
                         + ('' if item.reason == 'adjacent' else
                            f' (clearance {item.low * 1000.0:.1f} to '
                            f'{item.high * 1000.0:.1f} mm)'))
        if model.not_covered:
            lines.append(f'not covered: {model.not_covered}')
        if self.keepout_fault:
            lines.append(f'FAULT: {self.keepout_fault}')
        response.success = not self.keepout_fault
        response.message = '\n'.join(lines)
        return response

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
        self._stuck_window = None
        self.stuck_near_collision = False

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
        # MoveIt Servo slowing or halting the arm outranks this node's own solver limits:
        # it is the collision backstop, and the operator has to know it is acting.
        if (output.engaged and (limit == Limit.NONE or limit in SOLVER_LIMITS.values())
                and self.servo_status_current() and self.servo_status > 0):
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
        msg.translation_scale = float(self.params.translation_scale(self.state.coarse))
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
        msg.keepout_margin = self.keepout_margin
        clearance, pair = self.idle_clearance()
        msg.keepout_clearance = float(clearance)
        msg.keepout_link, msg.keepout_obstacle = pair
        msg.keepout_clearance_commanded = math.nan
        if solution is not None:
            msg.keepout_bound = bool(solution.keepout_bound)
            msg.keepout_clearance_commanded = float(solution.keepout_clearance)
        self.fill_gripper_state(msg)
        return msg

    def fill_gripper_state(self, msg: TeleopState) -> None:
        """
        Fill the gripper fields: command in force and measured opening, separately.

        The command in force is teleop's own command (or the reference it holds). When teleop
        holds none (not the owner yet or any more, or the held reference not yet arrived),
        what the gripper's controller holds is not known here; the measured opening stands
        in for it, with the reference source MEASURED. NaN only when gripper teleop is off
        or nothing is measured.
        """
        msg.gripper_enabled = self.gripper_params.enabled
        msg.gripper_command = math.nan
        if not self.gripper_params.enabled:
            msg.gripper_hold = int(grip.Hold.DISABLED)
            return
        out = self.gripper_output
        sample = self.gripper_measured()
        msg.gripper_hold = int(out.hold if out is not None else grip.Hold.NOT_ARMED)
        if out is not None:
            msg.gripper_active = bool(out.publish)
            msg.gripper_command = float(out.command)
            msg.gripper_request = int(out.request)
            msg.gripper_reference_source = int(out.source)
        if not math.isfinite(msg.gripper_command) and sample is not None:
            msg.gripper_command = sample[0]
            msg.gripper_reference_source = int(grip.Source.MEASURED)
        if sample is not None:
            msg.gripper_measured_valid = True
            msg.gripper_measured = sample[0]
            msg.gripper_measured_stamp.sec = int(sample[1] // 1_000_000_000)
            msg.gripper_measured_stamp.nanosec = int(sample[1] % 1_000_000_000)

    def idle_clearance(self):
        """
        Return the keep-out clearance and pair at the measured joints (NaN if unknown).

        Computed every IDLE_CLEARANCE_CYCLES cycles, every ENGAGED_CLEARANCE_CYCLES while
        engaged, and exact within the model's influence distance only, which keeps it cheap.
        """
        every = ENGAGED_CLEARANCE_CYCLES if self.state.engaged else IDLE_CLEARANCE_CYCLES
        self._idle_count = (self._idle_count + 1) % every
        if self._idle_count and self._idle_clearance is not None:
            return self._idle_clearance
        self._idle_clearance = (math.nan, ('', ''))
        if self.keepout is None or self.arm is None or self.joint_state is None:
            return self._idle_clearance
        positions = self.joint_state[2]
        try:
            q = [positions[name] for name in self.arm.joint_names]
            started = time.perf_counter()
            rows = self.keepout.evaluate(q, self.keepout.extra_values(positions), per_pair=1)
            self._cycles.measured_clearance_step(time.perf_counter() - started)
        except (KeyError, ko.ModelError, ValueError):
            return self._idle_clearance
        self._idle_clearance = (rows.clearance, rows.nearest)
        return self._idle_clearance

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
        self.compare_servo_description()
        self.check_servo_rigid_pairs()

    def check_servo_rigid_pairs(self) -> None:
        """
        Ask MoveIt Servo once for its description and SRDF; log the rigid pairs it checks.

        A pair with no joint of the servoed group between its links keeps its distance
        however the arm moves, yet MoveIt Servo slows every joint while it is within the
        self-collision threshold (a closed gripper). Such pairs are derived from the
        description (``rigid_pairs``) and listed at start-up as SRDF lines, so they can be
        disabled where MoveIt reads its collision matrix. Nothing is taken out of any check
        here: Servo follows move_group's collision matrix, not this node.
        """
        if self.servo_pairs_checked or not self.servo_description_client.service_is_ready():
            return
        self.servo_pairs_checked = True
        future = self.servo_description_client.call_async(GetParameters.Request(
            names=['robot_description', 'robot_description_semantic',
                   'moveit_servo.move_group_name']))

        def answered(done) -> None:
            log = self.get_logger()
            try:
                values = done.result().values
            except Exception as err:  # noqa: B902 - reported; nothing depends on it
                log.warning(f'MoveIt Servo description and SRDF not read: {err}')
                self.servo_pairs_checked = False
                return
            texts = [v.string_value for v in values] if len(values) == 3 else []
            if not texts or not all(texts):
                log.warning('MoveIt Servo reports no robot_description, '
                            'robot_description_semantic or move_group_name; its '
                            'self-collision pairs were not checked')
                return
            try:
                pairs = rigid_pairs.rigid_pairs(*texts)
            except rigid_pairs.DescriptionError as err:
                log.warning(f"MoveIt Servo's self-collision pairs were not checked: {err}")
                return
            self.servo_rigid_pairs = pairs
            if not pairs:
                log.info(f'MoveIt Servo: no enabled self-collision pair without a joint of '
                         f'{texts[2]!r} between its links')
                return
            log.warning(
                f'MoveIt Servo checks {len(pairs)} self-collision pairs with no joint of '
                f'{texts[2]!r} between the two links: arm motion cannot change their '
                'distance, yet Servo slows every joint while one is within its '
                'self-collision threshold (for example with the gripper closed). Disable '
                "them in the SRDF move_group reads (Servo follows move_group's collision "
                'matrix):\n' + '\n'.join(rigid_pairs.srdf_lines(pairs)))
        future.add_done_callback(answered)

    def compare_servo_description(self) -> None:
        """Ask MoveIt Servo once for its robot description and compare it with the model's."""
        if (self.servo_description_compared or self.keepout is None
                or not self.servo_description_client.service_is_ready()):
            return
        self.servo_description_compared = True
        self.servo_description_pending = True
        future = self.servo_description_client.call_async(
            GetParameters.Request(names=['robot_description']))

        def answered(done) -> None:
            self.servo_description_pending = False
            try:
                values = done.result().values
            except Exception as err:  # noqa: B902 - reported; Servo is verified otherwise
                self.get_logger().warning(f'MoveIt Servo robot description not read: {err}')
                self.servo_description_compared = False
                return
            text = values[0].string_value if len(values) == 1 else ''
            if not text:
                self.get_logger().warning('MoveIt Servo reports no robot_description; the '
                                          'link keep-out model was not compared with it')
            elif text != self.urdf:
                self._compare_description(text, "MoveIt Servo's robot description")
        future.add_done_callback(answered)

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
            self.state = replace(self.state, want_armed=False, arm_pending=False, coarse=False)


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
