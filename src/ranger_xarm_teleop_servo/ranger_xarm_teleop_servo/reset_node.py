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
Reset node: a hold-to-run return to a named posture from the headset (G-16).

A thin wrapper around ``reset.py`` (the rule) in its own process, so that the teleop node
gains no motion path. The operator holds the right thumbstick pulled back; after a dwell the
node sends move_group one goal to the posture (planned with collision checking, executed
through MoveIt's controller and so through the command mux, at the configured reduced speed)
and cancels it as soon as the hold breaks. Off by default (``enabled``), and simulation only:
nothing is sent unless the robot description's hardware plugins are all simulators.

Heard
  /teleop/input/{right/pose, head/pose, joy, status}  paired by stamp (``input_frames``)
  /teleop/state, /teleop/mux/owner, /teleop/episode/state, /joint_states, /robot_description
  /teleop/reset/watchdog          the watchdog's sign of life (no goal without a fresh one)
  /move_action                    status and feedback of the goal
Published
  /teleop/reset/state             ResetState, 10 Hz and on change
  /teleop/reset/heartbeat         ResetHeartbeat, 50 Hz, naming the goal while one is out

Every client is made at start-up, and the cancel path does one harmless round trip before
the node counts move_group as ready: nothing is created when the control is pressed. A goal
is sent once and is announced in the heartbeat before it goes out, so the watchdog can cancel
it by its id even if this node dies before the answer comes. Run it with a single-threaded
executor (the input pairing has no lock).
"""

import sys
import time
from typing import Dict, Optional
import uuid

from action_msgs.msg import GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from ranger_xarm_teleop_bridge.bridge_node import (HEAD_POSE_TOPIC, JOY_TOPIC, POSE_TOPICS,
                                                   STATUS_QOS, STATUS_TOPIC)
from ranger_xarm_teleop_interfaces.msg import (ArmOwner, EpisodeState, InputStatus,
                                               ResetHeartbeat, ResetState, ResetWatchdog,
                                               TeleopState)
from ranger_xarm_teleop_servo import reset as rs
from ranger_xarm_teleop_servo.input_frames import FrameAssembler, PAIR_WAIT_S, stamp_key
from ranger_xarm_teleop_servo.session import ARM_JOINTS
from ranger_xarm_teleop_servo.simcheck import NotASimulator, require_simulator
from ranger_xarm_teleop_servo.teleop_node import (INPUT_JOY_QOS, INPUT_POSE_QOS, LATCHED_QOS,
                                                  OWNER_TOPIC, pose_of, STATE_TOPIC)
import rclpy
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, qos_profile_action_status_default,
                       QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String
from unique_identifier_msgs.msg import UUID

RESET_STATE_TOPIC = '/teleop/reset/state'
HEARTBEAT_TOPIC = '/teleop/reset/heartbeat'
WATCHDOG_TOPIC = '/teleop/reset/watchdog'
EPISODE_TOPIC = '/teleop/episode/state'
MOVE_ACTION = '/move_action'

RESET_STATE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=10,
                             durability=DurabilityPolicy.VOLATILE)
# The two liveness topics (contract): reliable, depth 1.
LIVENESS_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST, depth=1,
                          durability=DurabilityPolicy.VOLATILE)
EPISODE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
CYCLE_S = 0.02              # the cycle and the heartbeat, 50 Hz
STATE_PERIOD_S = 0.1        # ResetState at 10 Hz, and on change
JOINTS_FRESH_S = 0.5        # a joint state older than this says nothing about the arm
GRAPH_LOOK_S = 1.0          # how often the graph is asked who publishes what
READY_ASK_S = 1.0           # the harmless round trip is asked again after this
# The watchdog counts as alive while its sign of life is fresh and echoes a heartbeat of this
# node no more than this many beats old (0.5 s at 50 Hz): it is there and hears this node.
# Fixed values, not parameters: they are never widened.
WATCHDOG_TIMEOUT_S = 0.5
ECHO_LAG_BEATS = 25


class StartupRefused(RuntimeError):
    """The node must not start with this configuration."""


class ResetNode(Node):
    """The reset's ROS side; the decisions are in ``reset.Reset``."""

    def __init__(self, node_name: str = 'reset_node', parameter_overrides=None) -> None:
        """Read the configuration, make every client, and start the cycle."""
        super().__init__(node_name, parameter_overrides=parameter_overrides or [])
        p = self.declare_parameter
        self.enabled = bool(p('enabled', False).value)              # G-1
        speed = p('reset_speed_scaling', Parameter.Type.DOUBLE).value
        if speed is None:
            raise StartupRefused('reset_speed_scaling is required and has no default (G-16); '
                                 'the simulation value is in the simulation-only limits file')
        names = p('reset_posture_names', Parameter.Type.STRING_ARRAY).value
        if not names:
            raise StartupRefused('reset_posture_names is required: the named postures a reset '
                                 'may go to come from the configuration')
        self.joint_names = list(p('arm_joints', list(ARM_JOINTS)).value)
        postures = {}
        for name in names:
            values = p(f'reset_postures.{name}', Parameter.Type.DOUBLE_ARRAY).value
            if values is None or len(values) != len(self.joint_names):
                raise StartupRefused(f'reset_postures.{name} needs {len(self.joint_names)} '
                                     'joint values')
            postures[name] = [float(v) for v in values]
        self.posture = str(p('reset_posture', names[0]).value)
        if self.posture not in postures:
            raise StartupRefused(f'reset_posture {self.posture!r} is not in reset_posture_names '
                                 f'{list(names)}')
        self.group = str(p('reset_group', 'xarm6').value)
        self.planning_time = float(p('reset_planning_time_s', 5.0).value)
        defaults = rs.Params(speed_scaling=float(speed))
        params = rs.Params(
            speed_scaling=float(speed),
            press_threshold=float(p('reset_press_threshold', defaults.press_threshold).value),
            hold_threshold=float(p('reset_hold_threshold', defaults.hold_threshold).value),
            dwell_s=float(p('reset_dwell_s', defaults.dwell_s).value),
            watchdog_timeout_s=WATCHDOG_TIMEOUT_S)
        try:
            self.reset = rs.Reset(params, self.posture, postures[self.posture],
                                  self.joint_names, lambda: uuid.uuid4().bytes)
        except ValueError as err:
            raise StartupRefused(str(err)) from err
        self.params = params
        self.started = time.monotonic()

        # What is heard.
        self.frames = FrameAssembler(PAIR_WAIT_S)
        self.seen: Dict[str, float] = {}                # topic -> monotonic time last heard
        self.owner: Optional[int] = None
        self.armed: Optional[bool] = None
        self.episode: Optional[int] = None
        self.watchdog = (False, bytes(16), 0)            # (stopping, goal id, heartbeat echoed)
        self.joints: Optional[tuple] = None
        self.simulator = False
        self.publishers_seen = {STATE_TOPIC: False, EPISODE_TOPIC: False}
        self.graph_looked = -1e9

        self.create_subscription(PoseStamped, POSE_TOPICS['right'], self._on_pose,
                                 INPUT_POSE_QOS)
        self.create_subscription(PoseStamped, HEAD_POSE_TOPIC, self._on_head, INPUT_POSE_QOS)
        self.create_subscription(Joy, JOY_TOPIC, self._on_joy, INPUT_JOY_QOS)
        self.create_subscription(InputStatus, STATUS_TOPIC, self._on_status, STATUS_QOS)
        self.create_subscription(TeleopState, STATE_TOPIC, self._on_teleop, 1)
        self.create_subscription(ArmOwner, OWNER_TOPIC, self._on_owner, 10)
        self.create_subscription(EpisodeState, EPISODE_TOPIC, self._on_episode, EPISODE_QOS)
        self.create_subscription(ResetWatchdog, WATCHDOG_TOPIC, self._on_watchdog, LIVENESS_QOS)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 50)
        self.create_subscription(String, '/robot_description', self._on_description,
                                 LATCHED_QOS)
        self.create_subscription(GoalStatusArray, f'{MOVE_ACTION}/_action/status',
                                 self._on_goal_status, qos_profile_action_status_default)
        self.state_pub = self.create_publisher(ResetState, RESET_STATE_TOPIC, RESET_STATE_QOS)
        self.beat_pub = self.create_publisher(ResetHeartbeat, HEARTBEAT_TOPIC, LIVENESS_QOS)
        self.beat_seq = 0

        # move_group: the action and its cancel path, made now, never on a press.
        self.action = ActionClient(self, MoveGroup, MOVE_ACTION)
        self.cancel_client = self.create_client(CancelGoal, f'{MOVE_ACTION}/_action/cancel_goal')
        self.moveit_ready = False
        self.ready_asked: Optional[float] = None
        self.cancel_future = None

        self.published = None                           # what the last ResetState said
        self.state_at = -1e9
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(CYCLE_S, self._cycle, clock=steady)
        self.get_logger().info(
            f'reset: {"ENABLED" if self.enabled else "off (enabled: false)"}; posture '
            f'{self.posture} {postures[self.posture]}, group {self.group}, speed scaling '
            f'{params.speed_scaling:g}; control: right thumbstick pulled back past '
            f'{params.press_threshold:g}, held above {params.hold_threshold:g}, dwell '
            f'{params.dwell_s:g} s')

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
            self.reset.on_sample(rs.sample_of(paired.status, joy, paired.arrival))
            self.frames.done(paired.key)

    def _on_teleop(self, msg: TeleopState) -> None:
        self.armed = bool(msg.armed)
        self.seen[STATE_TOPIC] = time.monotonic()

    def _on_owner(self, msg: ArmOwner) -> None:
        self.owner = int(msg.owner)
        self.seen[OWNER_TOPIC] = time.monotonic()

    def _on_episode(self, msg: EpisodeState) -> None:
        self.episode = int(msg.state)
        self.seen[EPISODE_TOPIC] = time.monotonic()

    def _on_watchdog(self, msg: ResetWatchdog) -> None:
        self.watchdog = (bool(msg.stopping), bytes(msg.goal_id), int(msg.heartbeat_seq))
        self.seen[WATCHDOG_TOPIC] = time.monotonic()

    def _on_joints(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position))
        if not all(name in positions for name in self.joint_names):
            return
        now = time.monotonic()
        self.joints = tuple(float(positions[name]) for name in self.joint_names)
        self.seen['/joint_states'] = now
        self.reset.on_joints(now, positions)

    def _on_description(self, msg: String) -> None:
        try:
            require_simulator(msg.data)
            self.simulator = True
        except NotASimulator as err:
            self.simulator = False
            self.get_logger().warning(f'reset refused for this robot: {err}')

    def _on_goal_status(self, msg: GoalStatusArray) -> None:
        now = time.monotonic()
        mine = False
        for entry in msg.status_list:
            goal_id = bytes(entry.goal_info.goal_id.uuid)
            mine = mine or goal_id == self.reset.goal_id
            self.reset.on_status(now, goal_id, int(entry.status))   # it keeps what is its own
        if not mine:
            self.reset.on_other_status()

    # ---- move_group ----

    def _check_ready(self, now: float) -> None:
        """One harmless round trip on the cancel path (an id no goal has), until answered."""
        if self.moveit_ready:
            return
        if not (self.action.server_is_ready() and self.cancel_client.service_is_ready()):
            return
        if self.ready_asked is not None and now - self.ready_asked < READY_ASK_S:
            return
        request = CancelGoal.Request()
        request.goal_info.goal_id = UUID(uuid=list(uuid.uuid4().bytes))
        self.ready_asked = now
        self.get_logger().info("asking move_group's cancel path once (a goal id nobody has)")
        self.cancel_client.call_async(request).add_done_callback(self._ready_answered)

    def _ready_answered(self, future) -> None:
        if future.result() is not None and not self.moveit_ready:
            self.moveit_ready = True
            self.get_logger().info('move_group is ready: its action and cancel path answer')

    def _send(self, goal_id: bytes) -> None:
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = self.group
        request.num_planning_attempts = 5
        request.allowed_planning_time = self.planning_time
        request.max_velocity_scaling_factor = self.params.speed_scaling
        request.max_acceleration_scaling_factor = self.params.speed_scaling
        request.start_state.is_diff = True
        request.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=name, position=float(value), tolerance_above=0.001,
                            tolerance_below=0.001, weight=1.0)
            for name, value in zip(self.joint_names, self.reset.target)])]
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        self.get_logger().info(f'goal {goal_id.hex()} to {self.posture} sent')
        future = self.action.send_goal_async(
            goal, feedback_callback=lambda fb, g=goal_id: self.reset.on_feedback(
                g, fb.feedback.state),
            goal_uuid=UUID(uuid=list(goal_id)))
        future.add_done_callback(lambda f, g=goal_id: self._answered(f, g))

    def _answered(self, future, goal_id: bytes) -> None:
        handle = future.result()
        accepted = handle is not None and handle.accepted
        self.reset.on_goal_answer(time.monotonic(), goal_id, accepted)
        if accepted:
            handle.get_result_async().add_done_callback(
                lambda f, g=goal_id: self._result(f, g))

    def _result(self, future, goal_id: bytes) -> None:
        answer = future.result()
        if answer is not None:
            self.reset.on_result(time.monotonic(), goal_id, int(answer.result.error_code.val))

    def _cancel(self, now: float) -> None:
        goal_id = self.reset.cancel_due(now)
        if goal_id is None:
            return
        request = CancelGoal.Request()
        request.goal_info.goal_id = UUID(uuid=list(goal_id))     # its own id, never zero
        self.reset.cancel_sent(now, goal_id)
        self.cancel_client.call_async(request).add_done_callback(
            lambda f, g=goal_id: self._cancel_answered(f, g))

    def _cancel_answered(self, future, goal_id: bytes) -> None:
        answer = future.result()
        if answer is not None:
            self.reset.cancel_answered(time.monotonic(), goal_id, int(answer.return_code),
                                       len(answer.goals_canceling))

    # ---- the cycle ----

    def _fresh(self, topic: str, limit: float, now: float) -> bool:
        return now - self.seen.get(topic, -1e9) <= limit

    def world(self, now: float) -> rs.World:
        """Return what the node knows now, each part only while fresh."""
        if now - self.graph_looked >= GRAPH_LOOK_S:
            self.graph_looked = now
            for topic in self.publishers_seen:
                self.publishers_seen[topic] = self.count_publishers(topic) > 0
        fresh_s = self.params.state_fresh_s
        stopping, goal, _ = self.watchdog
        alive = self.watchdog_alive(now)
        return rs.World(
            now=now, since_start=now - self.started, enabled=self.enabled,
            simulator=self.simulator,
            owner=self.owner if self._fresh(OWNER_TOPIC, fresh_s, now) else None,
            armed=self.armed if self._fresh(STATE_TOPIC, fresh_s, now) else None,
            teleop_running=STATE_TOPIC in self.seen or self.publishers_seen[STATE_TOPIC],
            episode=self.episode if self._fresh(EPISODE_TOPIC, fresh_s, now) else None,
            episode_running=EPISODE_TOPIC in self.seen or self.publishers_seen[EPISODE_TOPIC],
            watchdog_alive=alive, watchdog_stopping=alive and stopping, watchdog_goal=goal,
            moveit_ready=self.moveit_ready,
            joints=self.joints if self._fresh('/joint_states', JOINTS_FRESH_S, now) else None)

    def watchdog_alive(self, now: float) -> bool:
        """Return True while the watchdog's sign of life is fresh and echoes this node's beat."""
        echoed = self.watchdog[2]
        return (self._fresh(WATCHDOG_TOPIC, WATCHDOG_TIMEOUT_S, now)
                and echoed > 0 and self.beat_seq - echoed <= ECHO_LAG_BEATS)

    def _cycle(self) -> None:
        now = time.monotonic()
        self._assemble()                                # frames whose parts never came
        self._check_ready(now)
        goal_id = self.reset.tick(now, self.world(now), time.time())
        if goal_id is not None:
            self._beat()                                # announced before it goes out
            self._publish_state(now, force=True)
            self._send(goal_id)
        else:
            self._beat()
        self._cancel(now)
        for line in self.reset.take_log():
            self.get_logger().info('reset ' + line)
        self._publish_state(now)

    def _beat(self) -> None:
        self.beat_seq += 1
        msg = ResetHeartbeat()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.seq = self.beat_seq
        out, goal_id = self.reset.heartbeat_goal()      # the current goal, or an orphan's
        msg.goal_out = bool(out)
        msg.goal_id = list(goal_id)
        self.beat_pub.publish(msg)

    def _publish_state(self, now: float, force: bool = False) -> None:
        r = self.reset
        key = (r.state, r.reason, r.event_count, r.heartbeat_goal())
        if not force and key == self.published and now - self.state_at < STATE_PERIOD_S:
            return
        self.published, self.state_at = key, now
        msg = ResetState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.wall_time = time.time()
        msg.state, msg.reason = r.state, r.reason
        msg.reason_text = r.reason_text()
        msg.posture = self.posture
        out, goal_id = r.heartbeat_goal()
        msg.goal_out = bool(out)
        msg.goal_id = list(goal_id)
        msg.moveit_error_code = int(r.moveit_error_code)
        msg.event_count = r.event_count
        msg.last_event, msg.last_event_reason = r.last_event, r.last_event_reason
        msg.last_event_wall_time = float(r.last_event_wall)
        msg.stop_distance, msg.stop_time = float(r.stop_distance), float(r.stop_time)
        msg.enabled = bool(self.enabled and self.simulator)
        msg.watchdog_alive = self.watchdog_alive(now)
        self.state_pub.publish(msg)


def main(args=None) -> None:
    """Run the reset node; exit non-zero when it must not start."""
    rclpy.init(args=args)
    logger = rclpy.logging.get_logger('reset_node')
    try:
        node = ResetNode()
    except StartupRefused as err:
        logger.fatal(f'refusing to start: {err}')
        rclpy.try_shutdown()
        sys.exit(2)
    try:
        rclpy.spin(node)        # single-threaded: the input pairing has no lock
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
