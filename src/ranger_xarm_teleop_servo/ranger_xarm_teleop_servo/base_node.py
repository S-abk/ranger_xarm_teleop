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
Base node: hold-to-drive the mobile base from the headset's left controller (G-19).

A thin wrapper around ``base.py`` (the rule) in its own process, so that the arm's teleop node
gains no base path; this node has none to the arm (it only reads the arm's state). Off by
default (``enabled``), and simulation only: nothing is commanded unless the robot description's
hardware plugins are all simulators. It starts with the stack, before arming, and joins nothing
later.

Heard
  /teleop/input/{right/pose, head/pose, joy, status}  paired by stamp (``input_frames``)
  /teleop/state            the arm's clutch (engaged, stop ramp); fresh within 0.5 s
  /teleop/reset/state      a return to the ready posture under way
  /odom                    the base's rest (``base_rest``)
  /robot_description       simulator check; the arm's chain for the tool speed limit
  /joint_states            only with the tool speed limit set: the arm's posture
Published
  ``cmd_topic`` (/cmd_vel)  geometry_msgs/Twist, the command the base driver takes
  /teleop/base/cmd_vel      geometry_msgs/TwistStamped, the same command, the recorded action
  /teleop/base/state        BaseState, 10 Hz and on change

Both commands go out together, 50 Hz while the left grip is held, one zero on a stop, nothing
otherwise. The node checks the graph at 2 Hz and before every start: any other publisher of
either command refuses or stops the drive (G-9). On SIGINT or SIGTERM it sends the zero before
it leaves. Run it with a single-threaded executor (the input pairing has no lock).
"""

import math
import signal
import sys
import threading
import time
from typing import Dict, List, Optional

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from ranger_xarm_teleop_bridge.topics import (HEAD_POSE_TOPIC, JOY_TOPIC, POSE_TOPICS,
                                              STATUS_QOS, STATUS_TOPIC)
from ranger_xarm_teleop_interfaces.msg import BaseState, InputStatus, ResetState, TeleopState
from ranger_xarm_teleop_servo import base as bs
from ranger_xarm_teleop_servo.base_rest import AT_REST, RestParams, RestWatch
from ranger_xarm_teleop_servo.input_frames import FrameAssembler, PAIR_WAIT_S, stamp_key
from ranger_xarm_teleop_servo.kinematics import chain_from_urdf, ChainError, forward
from ranger_xarm_teleop_servo.simcheck import NotASimulator, require_simulator
from ranger_xarm_teleop_servo.teleop_node import (INPUT_JOY_QOS, INPUT_POSE_QOS, LATCHED_QOS,
                                                  pose_of, STATE_TOPIC)
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String

BASE_STATE_TOPIC = '/teleop/base/state'
BASE_CMD_TOPIC = '/teleop/base/cmd_vel'
RESET_STATE_TOPIC = '/teleop/reset/state'
# Commands: reliable, depth 10, as the simulated base subscribes; the state as the reset's.
CMD_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                     depth=10, durability=DurabilityPolicy.VOLATILE)
STATE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                       depth=10, durability=DurabilityPolicy.VOLATILE)
CYCLE_S = 0.02              # 50 Hz
STATE_PERIOD_S = 0.1        # BaseState at 10 Hz, and on change
GRAPH_LOOK_S = 0.5          # who else publishes a base command: 2 Hz, and before every start
TELEOP_FRESH_S = 0.5        # the arm's state, like the owner report's timeout
RESET_FRESH_S = 0.5         # the reset publishes at 10 Hz
JOINTS_FRESH_S = 0.5
TIMING_REPORT_S = 30.0      # how often the cycle and graph timings are logged


class StartupRefused(RuntimeError):
    """The node must not start with this configuration."""


class Timing:
    """Durations of one kind, for the log: count, p50, p99, max (ms)."""

    def __init__(self) -> None:
        """Start empty."""
        self.values: List[float] = []

    def add(self, seconds: float) -> None:
        """Take one duration."""
        self.values.append(seconds)

    def text(self) -> str:
        """Return the summary and start again."""
        if not self.values:
            return 'none'
        v = sorted(self.values)
        self.values = []

        def at(q):
            return v[min(len(v) - 1, int(q * len(v)))] * 1e3
        return f'n {len(v)} p50 {at(0.5):.2f} p99 {at(0.99):.2f} max {v[-1] * 1e3:.2f} ms'


class BaseNode(Node):
    """The base's ROS side; the decisions are in ``base.Base``."""

    def __init__(self, node_name: str = 'base_node', parameter_overrides=None) -> None:
        """Read the configuration, make every subscription and publisher, start the cycle."""
        super().__init__(node_name, parameter_overrides=parameter_overrides or [])
        p = self.declare_parameter
        self.enabled = bool(p('enabled', False).value)              # G-1
        required = {}
        for name in ('base_linear_cap', 'base_turn_cap', 'base_linear_accel',
                     'base_turn_accel'):
            value = p(name, Parameter.Type.DOUBLE).value
            if value is None:
                raise StartupRefused(f'{name} is required and has no default (G-19); the '
                                     'simulation value is in the simulation-only limits file')
            required[name] = float(value)
        tool_cap = p('base_tool_speed_cap', Parameter.Type.DOUBLE).value     # unset: no limit
        try:
            params = bs.Params(
                linear_cap=required['base_linear_cap'], turn_cap=required['base_turn_cap'],
                linear_accel=required['base_linear_accel'],
                turn_accel=required['base_turn_accel'],
                tool_speed_cap=None if tool_cap is None else float(tool_cap),
                dead_zone=float(p('base_stick_dead_zone', 0.15).value),
                snap_deg=float(p('base_stick_snap_deg', 10.0).value))
            self.base = bs.Base(params)
        except ValueError as err:
            raise StartupRefused(str(err)) from err
        self.params = params
        self.cmd_topic = str(p('cmd_topic', '/cmd_vel').value)
        self.base_frame = str(p('base_frame', 'base_link').value)
        self.tool_frame = str(p('tool_frame', 'xarm_link_tcp').value)
        odom_topic = str(p('odom_topic', '/odom').value)
        self.rest = RestWatch(RestParams())

        # What is heard.
        self.frames = FrameAssembler(PAIR_WAIT_S)
        self.seen: Dict[str, float] = {}
        self.teleop: Optional[bs.TeleopView] = None
        self.reset_state: Optional[int] = None
        self.reset_publisher = False
        self.simulator = False
        self.chain = None
        self.joints: Optional[Dict[str, float]] = None
        self.others: tuple = ()
        self.graph_looked = -1e9
        self.timing = {'cycle': Timing(), 'graph': Timing(), 'period': Timing()}
        self.last_cycle: Optional[float] = None
        self.timing_logged = time.monotonic()

        self.create_subscription(PoseStamped, POSE_TOPICS['right'], self._on_pose,
                                 INPUT_POSE_QOS)
        self.create_subscription(PoseStamped, HEAD_POSE_TOPIC, self._on_head, INPUT_POSE_QOS)
        self.create_subscription(Joy, JOY_TOPIC, self._on_joy, INPUT_JOY_QOS)
        self.create_subscription(InputStatus, STATUS_TOPIC, self._on_status, STATUS_QOS)
        self.create_subscription(TeleopState, STATE_TOPIC, self._on_teleop, 1)
        self.create_subscription(ResetState, RESET_STATE_TOPIC, self._on_reset, STATE_QOS)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 50)
        self.create_subscription(String, '/robot_description', self._on_description,
                                 LATCHED_QOS)
        if params.tool_speed_cap is not None:
            self.create_subscription(JointState, '/joint_states', self._on_joints, 50)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, CMD_QOS)
        self.stamped_pub = self.create_publisher(TwistStamped, BASE_CMD_TOPIC, CMD_QOS)
        self.state_pub = self.create_publisher(BaseState, BASE_STATE_TOPIC, STATE_QOS)
        self.published = None
        self.state_at = -1e9
        self.last_out: Optional[bs.Output] = None
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(CYCLE_S, self._cycle, clock=steady)
        cap = ('none' if params.tool_speed_cap is None
               else f'{params.tool_speed_cap:g} m/s at {self.tool_frame}')
        self.get_logger().info(
            f'base: {"ENABLED" if self.enabled else "off (enabled: false)"}; commands '
            f'{self.cmd_topic} and {BASE_CMD_TOPIC}; caps {params.linear_cap:g} m/s, '
            f'{params.turn_cap:g} rad/s; acceleration {params.linear_accel:g} m/s^2, '
            f'{params.turn_accel:g} rad/s^2; tool speed cap {cap}; hold the left grip')

    # ---- input ----

    def _on_pose(self, msg: PoseStamped) -> None:
        self.frames.add_pose(stamp_key(msg.header.stamp), pose_of(msg.pose))
        self._assemble()

    def _on_head(self, msg: PoseStamped) -> None:
        self.frames.add_head(stamp_key(msg.header.stamp), pose_of(msg.pose))
        self._assemble()

    def _on_joy(self, msg: Joy) -> None:
        self.frames.add_joy(stamp_key(msg.header.stamp), msg)
        self._assemble()

    def _on_status(self, msg: InputStatus) -> None:
        self.frames.add_status(time.monotonic(), msg)
        self._assemble()

    def _assemble(self) -> None:
        now = time.monotonic()
        while True:
            paired = self.frames.take(now)
            if paired is None:
                return
            joy, _, _ = self.frames.parts(paired.key)    # before done(): the joy of this frame
            self.base.on_sample(bs.sample_of(paired.status, joy, paired.arrival))
            self.frames.done(paired.key)

    def _on_teleop(self, msg: TeleopState) -> None:
        self.teleop = bs.TeleopView(clutch_engaged=bool(msg.clutch_engaged),
                                    stopping=bool(msg.stopping))
        self.seen[STATE_TOPIC] = time.monotonic()

    def _on_reset(self, msg: ResetState) -> None:
        self.reset_state = int(msg.state)
        self.seen[RESET_STATE_TOPIC] = time.monotonic()

    def _on_odom(self, msg: Odometry) -> None:
        t = msg.twist.twist
        self.rest.add(time.monotonic(), t.linear.x, t.linear.y, t.angular.z)

    def _on_joints(self, msg: JointState) -> None:
        self.joints = dict(zip(msg.name, msg.position))
        self.seen['/joint_states'] = time.monotonic()

    def _on_description(self, msg: String) -> None:
        try:
            require_simulator(msg.data)
            self.simulator = True
        except NotASimulator as err:
            self.simulator = False
            self.get_logger().warning(f'base driving refused for this robot: {err}')
        if self.params.tool_speed_cap is not None:
            try:
                self.chain = chain_from_urdf(msg.data, base=self.base_frame, tip=self.tool_frame)
            except ChainError as err:
                self.chain = None
                self.get_logger().error(f'no chain {self.base_frame} -> {self.tool_frame} for '
                                        f'the tool speed limit: {err}')

    # ---- the world ----

    def _fresh(self, topic: str, limit: float, now: float) -> bool:
        return now - self.seen.get(topic, -1e9) <= limit

    def look_at_graph(self, now: float) -> None:
        """Find every other publisher of a base command (G-9); time the query."""
        start = time.monotonic()
        mine = self.get_fully_qualified_name()
        others = []
        for topic in (self.cmd_topic, BASE_CMD_TOPIC):
            for info in self.get_publishers_info_by_topic(topic):
                name = info.node_namespace.rstrip('/') + '/' + info.node_name
                if name != mine and name not in others:
                    others.append(name)
        self.reset_publisher = self.count_publishers(RESET_STATE_TOPIC) > 0
        self.timing['graph'].add(time.monotonic() - start)
        self.others = tuple(others)
        self.graph_looked = now

    def tool_xy(self, now: float) -> Optional[tuple]:
        """Return the tool's horizontal position in the base frame from the measured joints."""
        if self.chain is None or self.joints is None:
            return None
        if not self._fresh('/joint_states', JOINTS_FRESH_S, now):
            return None
        names = self.chain.joint_names
        if not all(name in self.joints for name in names):
            return None
        tip = forward(self.chain, [self.joints[name] for name in names])[0]
        return float(tip[0, 3]), float(tip[1, 3])

    def world(self, now: float) -> bs.World:
        """Return what the node knows now, each part only while fresh."""
        if now - self.graph_looked >= GRAPH_LOOK_S:
            self.look_at_graph(now)
        return bs.World(
            now=now, enabled=self.enabled, simulator=self.simulator,
            teleop=self.teleop if self._fresh(STATE_TOPIC, TELEOP_FRESH_S, now) else None,
            reset_state=(self.reset_state if self._fresh(RESET_STATE_TOPIC, RESET_FRESH_S, now)
                         else None),
            reset_running=RESET_STATE_TOPIC in self.seen or self.reset_publisher,
            rest=self.rest.state(now), other_publishers=self.others,
            tool_xy=self.tool_xy(now) if self.params.tool_speed_cap is not None else None)

    # ---- the cycle ----

    def _cycle(self) -> None:
        now = time.monotonic()
        if self.last_cycle is not None:
            self.timing['period'].add(now - self.last_cycle)
        self.last_cycle = now
        self._assemble()                                # frames whose parts never came
        if not self.base.driving and self._press_pending():
            self.look_at_graph(now)                     # before every start: fresh
        out = self.base.tick(now, self.world(now), time.time())
        self._send(out)
        for line in self.base.take_log():
            self.get_logger().info('base ' + line)
        self._publish_state(now, out)
        self.timing['cycle'].add(time.monotonic() - now)
        if now - self.timing_logged >= TIMING_REPORT_S:
            self.timing_logged = now
            self.get_logger().info(
                'base timing: cycle ' + self.timing['cycle'].text() + '; period '
                + self.timing['period'].text() + '; graph query ' + self.timing['graph'].text())

    def _press_pending(self) -> bool:
        s = self.base.last
        return s is not None and s.left_grip and not self.base.need_release

    def _send(self, out: bs.Output) -> None:
        self.last_out = out
        if out.publish is None:
            return
        vx, vy, wz = out.publish
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.angular.z = float(vx), float(vy), float(wz)
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self.base_frame
        stamped.twist = twist
        self.cmd_pub.publish(twist)
        self.stamped_pub.publish(stamped)

    def stop_now(self, reason: int = bs.SHUTDOWN) -> bool:
        """Send the zero of a stop at once if a drive runs; return True if one was sent."""
        out = self.base.stop(reason, time.time())
        self._send(out)
        self._publish_state(time.monotonic(), out, force=True)
        for line in self.base.take_log():
            self.get_logger().info('base ' + line)
        return out.publish is not None

    def _publish_state(self, now: float, out: bs.Output, force: bool = False) -> None:
        b = self.base
        key = (out.state, out.reason, out.mode, out.driving, out.turn_ignored, out.active_limit,
               b.event_count, out.publish)
        if not force and key == self.published and now - self.state_at < STATE_PERIOD_S:
            return
        self.published, self.state_at = key, now
        msg = BaseState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.wall_time = time.time()
        msg.enabled = bool(self.enabled and self.simulator)
        msg.state, msg.mode, msg.reason = out.state, out.mode, out.reason
        msg.reason_text = b.reason_text(out.reason)
        msg.driving = bool(out.driving)
        if out.publish is not None:
            msg.command.linear.x, msg.command.linear.y, msg.command.angular.z = out.publish
        msg.turn_ignored = bool(out.turn_ignored)
        msg.at_rest = self.rest.state(now) == AT_REST
        age = self.rest.age(now)
        msg.odom_age = age if math.isfinite(age) else math.nan
        msg.linear_cap, msg.turn_cap = self.params.linear_cap, self.params.turn_cap
        msg.active_limit = out.active_limit
        cap = self.params.tool_speed_cap
        msg.tool_speed_cap = math.nan if cap is None else cap
        msg.tool_speed_scale = float(out.tool_scale)
        msg.event_count = b.event_count
        msg.last_event, msg.last_event_reason = b.last_event, b.last_event_reason
        msg.last_event_wall_time = float(b.last_event_wall)
        self.state_pub.publish(msg)


def main(args=None) -> None:
    """Run the base node; on SIGINT or SIGTERM send the stop's zero, then leave."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    logger = rclpy.logging.get_logger('base_node')
    try:
        node = BaseNode()
    except StartupRefused as err:
        logger.fatal(f'refusing to start: {err}')
        rclpy.try_shutdown()
        sys.exit(2)
    leave = threading.Event()

    def on_signal(signum, _frame):
        leave.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while not leave.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.05)
        if node.stop_now(bs.SHUTDOWN):
            node.get_logger().info('base: stopping, zero sent')
        time.sleep(0.1)                     # let the zero leave before the publisher goes
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)    # a second one must not cut this short
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
