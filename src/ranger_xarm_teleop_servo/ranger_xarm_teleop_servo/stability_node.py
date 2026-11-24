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
Stability monitor node: how far the robot is from starting to tip (contract, "Stability monitor").

Information only (INV-1): it publishes its own state and nothing else, has no service or
action client, and nothing in the stack reads its state to decide a command, a hold or a
refusal. Stopped, killed or never started, it changes nothing else (INV-2). Off by default
(started by the launch argument ``stability:=true``), simulation only: outside a simulator it
publishes "unavailable" and computes nothing. It starts with the stack, before arming. The
figure does not see a force on the tool.

Heard
  /robot_description      the masses, contacts and springs (``stability_model``); simulator check
  /joint_states           the arm's joints (and their accelerations)
  ``steering_topic``      control_msgs/DynamicJointState: the steering angles (empty: none)
  ``odom_topic`` (/odom)  speed, turning rate, at rest
  ``tilt_topic``          sensor_msgs/Imu: linear acceleration and angular rate ONLY (its
                          orientation is never read); empty: no tilt source, a figure only at rest
  /tf, /tf_static         the IMU's pose in the base frame
Published
  /teleop/stability/state  StabilityState, every cycle (50 Hz)

The mass file (``mass_file``) is read once at start; unreadable or refused, the state says so.
"""

import math
import sys
import time
from typing import List, Optional

from control_msgs.msg import DynamicJointState
from nav_msgs.msg import Odometry
import numpy as np
from ranger_xarm_teleop_interfaces.msg import StabilityState
from ranger_xarm_teleop_servo import stability_monitor as smon
from ranger_xarm_teleop_servo.simcheck import NotASimulator, require_simulator
from ranger_xarm_teleop_servo.teleop_node import LATCHED_QOS
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import String
import tf2_ros
import yaml

STATE_TOPIC = '/teleop/stability/state'
CYCLE_S = 0.02                  # 50 Hz
TIMING_REPORT_S = 30.0
IMU_FRAME_RETRY_S = 1.0
TILT_HINT = ' (no IMU messages: a simulator bridges its IMU only when started with ' \
            'sensors:=true)'


class StartupRefused(RuntimeError):
    """The node must not start with this configuration."""


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _stamp(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class StabilityNode(Node):
    """The monitor's ROS side; the figure is in ``stability_monitor.Monitor``."""

    def __init__(self, node_name: str = 'stability_node', parameter_overrides=None) -> None:
        """Read the configuration (every value required), subscribe, start the cycle."""
        super().__init__(node_name, parameter_overrides=parameter_overrides or [])
        p = self.declare_parameter
        number = {}
        for name in ('cutoff_hz', 'tilt_tau_s', 'joint_timeout_s', 'odometry_timeout_s',
                     'imu_timeout_s', 'stop_decel_m_s2', 'stop_turn_decel_rad_s2'):
            value = p(name, Parameter.Type.DOUBLE).value
            if value is None:
                raise StartupRefused(f'{name} is required and has no default; the simulation '
                                     'values are in the simulation-only stability file')
            number[name] = float(value)
        wheels = p('wheel_links', Parameter.Type.STRING_ARRAY).value
        if not wheels:
            raise StartupRefused('wheel_links is required (the links whose contacts carry the '
                                 'robot)')
        self.tilt_topic = str(p('tilt_topic', '').value)
        slope = p('assumed_slope_deg', Parameter.Type.DOUBLE).value
        self.mass_file = str(p('mass_file', '').value)
        if not self.mass_file:
            raise StartupRefused('mass_file is required (the simulation-only one corrects '
                                 'nothing)')
        self.base_frame = str(p('base_frame', 'base_link').value)
        self.imu_frame = str(p('imu_frame', '').value)     # empty: the IMU messages' frame
        odom_topic = str(p('odom_topic', '/odom').value)
        steering_topic = str(p('steering_topic', '/dynamic_joint_states').value)
        try:
            self.config = smon.Config(
                base_frame=self.base_frame, wheel_links=tuple(wheels),
                tilt_configured=bool(self.tilt_topic), assumed_slope_deg=(
                    None if slope is None else float(slope)),
                tilt_hint=TILT_HINT, **number)
            self.monitor = smon.Monitor(self.config, simulation=True)
        except ValueError as err:
            raise StartupRefused(str(err)) from err
        self.simulator: Optional[bool] = None
        self.not_simulator_text = ''
        self._load_mass_file()

        self.timing: List[float] = []
        self.logged_reason: Optional[tuple] = None      # (reason, text) last written to the log
        self.reason_log: List[str] = []                     # every such line (tests read it)
        self.timing_logged = time.monotonic()
        self.imu_frame_looked = -1e9
        self.imu_header_frame = ''
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(String, '/robot_description', self._on_description,
                                 LATCHED_QOS)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 50)
        if steering_topic:
            self.create_subscription(DynamicJointState, steering_topic, self._on_dynamic, 50)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 50)
        if self.tilt_topic:
            self.create_subscription(Imu, self.tilt_topic, self._on_imu, qos_profile_sensor_data)
        self.state_pub = self.create_publisher(StabilityState, STATE_TOPIC, 10)
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(CYCLE_S, self._cycle, clock=steady)
        tilt = (f'tilt from {self.tilt_topic} (acceleration and rate only)' if self.tilt_topic
                else f'no tilt source: a figure only at rest, slope bound {slope:g} deg')
        self.get_logger().info(
            f'stability monitor: information only; {tilt}; stop figure at '
            f'{number["stop_decel_m_s2"]:g} m/s^2; filters {number["cutoff_hz"]:g} Hz; '
            f'mass file {self.mass_file}; it does not see a force on the tool')

    def _load_mass_file(self) -> None:
        try:
            with open(self.mass_file, encoding='utf-8') as handle:
                data = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as err:
            self.monitor.set_mass_data(None, error=f'cannot read {self.mass_file}: {err}')
            self.get_logger().error(f'stability: cannot read the mass file: {err}')
            return
        self.monitor.set_mass_data(data)

    # ---- input ----

    def _on_description(self, msg: String) -> None:
        try:
            require_simulator(msg.data)
            self.simulator = True
        except NotASimulator as err:
            self.simulator = False
            self.not_simulator_text = f'simulation only until reviewed for hardware: {err}'
            self.get_logger().warning(f'stability: {self.not_simulator_text}')
            return
        self.monitor.set_description(msg.data)
        if self.monitor.model is None:
            self.get_logger().error(f'stability: {self.monitor.model_error}')
        elif self.monitor.mass is None:
            self.get_logger().error(f'stability: mass file refused: {self.monitor.mass_error}')
        else:
            self.get_logger().info(
                f'stability: model in force {self.monitor.mass.fingerprint[:12]} '
                f'({len(self.monitor.mass.bodies)} bodies)')

    def _on_joints(self, msg: JointState) -> None:
        if len(msg.position) != len(msg.name):
            return
        self.monitor.on_joint_states(time.monotonic(), _stamp(msg.header), list(msg.name),
                                     list(msg.position))

    def _on_dynamic(self, msg: DynamicJointState) -> None:
        names, values = [], []
        for name, iv in zip(msg.joint_names, msg.interface_values):
            for interface, value in zip(iv.interface_names, iv.values):
                if interface == 'position':
                    names.append(name)
                    values.append(value)
        self.monitor.on_joint_states(time.monotonic(), _stamp(msg.header), names, values,
                                     source_is_arm=False)

    def _on_odom(self, msg: Odometry) -> None:
        t = msg.twist.twist
        self.monitor.on_odometry(time.monotonic(), t.linear.x, t.linear.y, t.angular.z)

    def _on_imu(self, msg: Imu) -> None:
        # Linear acceleration and angular rate only: the orientation field is never read.
        self.imu_header_frame = msg.header.frame_id
        a, w = msg.linear_acceleration, msg.angular_velocity
        self.monitor.on_imu(time.monotonic(), _stamp(msg.header), (a.x, a.y, a.z),
                            (w.x, w.y, w.z))

    def _look_up_imu(self, now: float) -> None:
        if (self.monitor.imu_rotation is not None
                or now - self.imu_frame_looked < IMU_FRAME_RETRY_S):
            return
        self.imu_frame_looked = now
        frame = self.imu_frame or self.imu_header_frame
        if not frame:
            self.monitor.set_imu_frame(None, None, 'no IMU message yet to name its frame')
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame, rclpy.time.Time())
        except tf2_ros.TransformException as err:
            self.monitor.set_imu_frame(None, None, f'no transform {self.base_frame} -> {frame}: '
                                       f'{err}')
            return
        t, q = tf.transform.translation, tf.transform.rotation
        self.monitor.set_imu_frame(_quat_matrix(q), np.array([t.x, t.y, t.z]))
        self.get_logger().info(f'stability: IMU frame {frame} at ({t.x:.3f}, {t.y:.3f}, '
                               f'{t.z:.3f}) in {self.base_frame}')

    # ---- the cycle ----

    def _cycle(self) -> None:
        start = time.monotonic()
        if self.tilt_topic:
            self._look_up_imu(start)
        if self.simulator is False:
            result = smon.unavailable(smon.NOT_SIMULATION, self.not_simulator_text)
        else:
            result = self.monitor.cycle(start)
        self.publish(result)
        self._log_reason(result)
        self.timing.append(time.monotonic() - start)
        if start - self.timing_logged > TIMING_REPORT_S:
            self.timing_logged = start
            v = sorted(self.timing)
            self.timing = []
            self.get_logger().info(
                f'stability: cycle n {len(v)} p50 {v[len(v) // 2] * 1e3:.2f} p99 '
                f'{v[min(len(v) - 1, int(0.99 * len(v)))] * 1e3:.2f} max {v[-1] * 1e3:.2f} ms; '
                f'last: {"figure" if result.available else smon.reason_name(result.reason)}')

    def _log_reason(self, r: smon.Result) -> None:
        """Write the state's reason to the log once per change (it can name the fix)."""
        key = (bool(r.available), int(r.reason), '' if r.available else r.reason_text)
        if key == self.logged_reason:
            return
        self.logged_reason = key
        line = ('stability: figure available' if r.available else
                f'stability: unavailable ({smon.reason_name(r.reason)}): {r.reason_text}')
        self.reason_log.append(line)
        if r.available:
            self.get_logger().info(line)
        else:
            self.get_logger().warning(line)

    def publish(self, r: smon.Result) -> None:
        """Publish one state."""
        msg = StabilityState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.wall_time = time.time()
        msg.available = bool(r.available)
        msg.reason, msg.reason_text = int(r.reason), r.reason_text
        msg.tilt_source = int(r.tilt_source)
        msg.margin_m, msg.edge = float(r.margin), int(r.edge)
        msg.margin_pessimistic_m, msg.edge_pessimistic = (float(r.margin_pessimistic),
                                                          int(r.edge_pessimistic))
        msg.stop_margin_m = float(r.stop_margin)
        msg.stop_margin_pessimistic_m = float(r.stop_margin_pessimistic)
        msg.stop_edge = int(r.stop_edge)
        msg.stop_decel_m_s2 = float(self.config.stop_decel_m_s2)
        msg.stop_turn_decel_rad_s2 = float(self.config.stop_turn_decel_rad_s2)
        msg.energy_reserve_j, msg.energy_edge = float(r.energy_reserve), int(r.energy_edge)
        msg.cop_x_m, msg.cop_y_m = float(r.cop[0]), float(r.cop[1])
        msg.roll_deg, msg.pitch_deg = float(r.roll_deg), float(r.pitch_deg)
        msg.speed_m_s = float(r.speed) if math.isfinite(r.speed) else math.nan
        msg.turn_rate_rad_s = float(r.turn_rate) if math.isfinite(r.turn_rate) else math.nan
        msg.at_rest = bool(r.at_rest)
        msg.model_sha256 = r.fingerprint
        msg.note = r.note
        self.state_pub.publish(msg)


def main(args=None) -> None:
    """Run the stability monitor."""
    rclpy.init(args=args)
    logger = rclpy.logging.get_logger('stability_node')
    try:
        node = StabilityNode()
    except StartupRefused as err:
        logger.fatal(f'refusing to start: {err}')
        rclpy.try_shutdown()
        sys.exit(2)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
