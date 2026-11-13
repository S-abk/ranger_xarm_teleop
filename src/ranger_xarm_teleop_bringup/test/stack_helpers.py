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
"""Helpers for the tests that start real processes: own ROS domain, localhost discovery."""

import os
import signal
import subprocess
import threading
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from ranger_xarm_teleop_interfaces.msg import ArmOwner, InputStatus, TeleopState
from ranger_xarm_teleop_interfaces.srv import SetArmOwner
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = [f'xarm_joint{i}' for i in range(1, 7)]
READY = [0.0, -0.6, -0.7, 0.0, 1.3, 0.0]
HERE = os.path.dirname(os.path.abspath(__file__))
SERVO_FIXTURES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'ranger_xarm_teleop_servo',
                              'test', 'fixtures')


def child_environment(offset: int) -> dict:
    """
    Return an environment for child processes and this test's own ROS context.

    The domain is the session's domain plus ``offset`` so parallel test packages do not
    see each other; discovery stays on this machine; RViz, which the mock launch always
    starts, draws off screen.
    """
    env = dict(os.environ)
    base = int(env.get('ROS_DOMAIN_ID', '0'))
    env['ROS_DOMAIN_ID'] = str((base + offset) % 100)
    env['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    env['QT_QPA_PLATFORM'] = 'offscreen'
    return env


class Process:
    """A child process in its own process group, with its output in a log file."""

    def __init__(self, command, env, log_path):
        self.log_path = log_path
        self.log = open(log_path, 'w', encoding='utf-8')
        self.proc = subprocess.Popen(command, env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT, start_new_session=True)

    def wait(self, timeout):
        """Return the exit code, or None when still running after ``timeout``."""
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            return None

    def output(self):
        """Return everything the process has printed so far."""
        if not self.log.closed:
            self.log.flush()
        with open(self.log_path, encoding='utf-8', errors='replace') as handle:
            return handle.read()

    def stop(self):
        """Interrupt the whole process group, then kill what is left."""
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            if self.wait(15) is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.wait(5)
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)   # children of a dead launch
            except ProcessLookupError:
                pass
        self.log.close()


def kill_gz_partition(partition: str) -> int:
    """
    Kill every process whose environment carries this gz partition; return how many.

    A gz server can outlive the launch that started it (it leaves the process group),
    and a leftover server slows or breaks the next simulation.
    """
    killed = 0
    needle = f'GZ_PARTITION={partition}'.encode()
    for entry in os.listdir('/proc'):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f'/proc/{entry}/environ', 'rb') as handle:
                if needle not in handle.read().split(b'\0'):
                    continue
            os.kill(int(entry), signal.SIGKILL)
            killed += 1
        except (OSError, PermissionError):
            continue
    return killed


class Probe:
    """A node that watches the teleop stack and acts on it, spun in a background thread."""

    def __init__(self, env, moveit_controller='xarm6_traj_controller',
                 teleop_controller='teleop_arm_controller'):
        self.context = rclpy.Context()
        rclpy.init(context=self.context, domain_id=int(env['ROS_DOMAIN_ID']))
        self.node = rclpy.create_node('teleop_test_probe', context=self.context)
        self.lock = threading.Lock()
        self.states = []            # (monotonic, TeleopState)
        self.joints = []            # (monotonic, [q1..q6])
        self.owner = None
        self.inputs = []            # monotonic arrival of every input status
        self.stamped_joints = []    # (monotonic, header stamp, [q1..q6])
        self.description = None     # /robot_description text
        self.spin_errors = []
        self.moveit_controller = moveit_controller
        self.teleop_controller = teleop_controller
        self.node.create_subscription(TeleopState, '/teleop/state', self._on_state, 50)
        self.node.create_subscription(JointState, '/joint_states', self._on_joints, 50)
        self.node.create_subscription(ArmOwner, '/teleop/mux/owner', self._on_owner, 10)
        self.node.create_subscription(InputStatus, '/teleop/input/status', self._on_input, 50)
        self.node.create_subscription(
            String, '/robot_description', self._on_description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.trajectory_pub = self.node.create_publisher(
            JointTrajectory, f'/{teleop_controller}/joint_trajectory', 10)
        self.action = ActionClient(self.node, FollowJointTrajectory,
                                   f'/{moveit_controller}/follow_joint_trajectory')
        self.set_owner_client = self.node.create_client(SetArmOwner, '/teleop/mux/set_owner')
        self.disarm_client = self.node.create_client(Trigger, '/teleop/disarm')
        self.list_client = self.node.create_client(ListControllers,
                                                   '/controller_manager/list_controllers')
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        # Clients are created from the test thread while this thread spins; an
        # exception from that race must not end the spinning, or the probe would go
        # deaf and later checks would read stale values.
        while self.context.ok():
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as err:  # noqa: B902 - keep spinning, remember what happened
                if not self.context.ok():
                    break
                self.spin_errors.append(repr(err))

    def close(self):
        self.executor.shutdown(timeout_sec=2.0)
        self.node.destroy_node()
        rclpy.shutdown(context=self.context)
        self.thread.join(3.0)

    def _on_state(self, msg):
        with self.lock:
            self.states.append((time.monotonic(), msg))

    def _on_joints(self, msg):
        index = {name: i for i, name in enumerate(msg.name)}
        if all(j in index for j in ARM_JOINTS):
            q = [msg.position[index[j]] for j in ARM_JOINTS]
            now = time.monotonic()
            with self.lock:
                self.joints.append((now, q))
                self.stamped_joints.append((now, msg.header.stamp, q))

    def _on_description(self, msg):
        with self.lock:
            self.description = msg.data

    def _on_input(self, msg):
        with self.lock:
            self.inputs.append(time.monotonic())

    def _on_owner(self, msg):
        with self.lock:
            self.owner = msg

    # ---- waiting ----

    def wait_for(self, condition, timeout, what):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = condition()
            if value:
                return value
            time.sleep(0.05)
        raise AssertionError(f'timed out after {timeout} s waiting for {what}')

    def call(self, client, request, timeout=15.0):
        assert client.wait_for_service(timeout_sec=timeout), f'no service {client.srv_name}'
        future = client.call_async(request)
        self.wait_for(future.done, timeout, f'answer from {client.srv_name}')
        return future.result()

    # ---- observations ----

    def state(self):
        with self.lock:
            return self.states[-1][1] if self.states else None

    def assert_listening(self):
        """Fail unless a TeleopState arrived in the last half second."""
        with self.lock:
            newest = self.states[-1][0] if self.states else None
        assert newest is not None and time.monotonic() - newest < 0.5, (
            f'the probe is not receiving /teleop/state (spin errors: {self.spin_errors})')

    def states_since(self, t):
        with self.lock:
            return [s for when, s in self.states if when >= t]

    def timed_states_since(self, t):
        with self.lock:
            return [(when, s) for when, s in self.states if when >= t]

    def inputs_since(self, t):
        with self.lock:
            return [when for when in self.inputs if when >= t]

    def joint_positions(self):
        with self.lock:
            return list(self.joints[-1][1]) if self.joints else None

    def joints_since(self, t):
        with self.lock:
            return [q for when, q in self.joints if when >= t]

    def stamped_joints_since(self, t):
        with self.lock:
            return [(stamp, q) for when, stamp, q in self.stamped_joints if when >= t]

    def joint_speeds_since(self, t):
        """Return the largest joint speed between consecutive joint states (rad/s, stamps)."""
        samples = self.stamped_joints_since(t)
        best = 0.0
        for (s0, q0), (s1, q1) in zip(samples, samples[1:]):
            dt = (s1.sec - s0.sec) + (s1.nanosec - s0.nanosec) * 1e-9
            if dt > 1e-4:
                best = max(best, max(abs(a - b) for a, b in zip(q1, q0)) / dt)
        return best

    def max_joint_change(self, since, reference):
        return max((max(abs(a - b) for a, b in zip(q, reference))
                    for q in self.joints_since(since)), default=0.0)

    def controllers(self):
        result = self.call(self.list_client, ListControllers.Request())
        return {c.name: c.state for c in result.controller}

    def owner_value(self):
        with self.lock:
            return None if self.owner is None else self.owner.owner

    def tool_position(self):
        state = self.state()
        assert state is not None and state.measured_tool_pose_valid
        p = state.measured_tool_pose.position
        return (p.x, p.y, p.z)

    # ---- actions ----

    def set_owner(self, owner):
        return self.call(self.set_owner_client, SetArmOwner.Request(owner=owner))

    def disarm(self):
        self.assert_listening()
        self.call(self.disarm_client, Trigger.Request())
        self.wait_for(lambda: self.owner_value() == ArmOwner.OWNER_MOVEIT, 10.0,
                      'the arm to return to MoveIt after disarming')

    def send_goal(self, positions, seconds):
        """Send a joint goal to the MoveIt controller; return the goal handle (or None)."""
        assert self.action.wait_for_server(timeout_sec=15.0)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS
        whole = int(seconds)
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=[float(p) for p in positions], velocities=[0.0] * 6,
            time_from_start=Duration(sec=whole, nanosec=int((seconds - whole) * 1e9)))]
        future = self.action.send_goal_async(goal)
        self.wait_for(future.done, 15.0, 'the goal response')
        return future.result()

    def move_to(self, positions, seconds=3.0):
        """Execute a joint goal through MoveIt's controller and wait for success."""
        handle = self.send_goal(positions, seconds)
        assert handle is not None and handle.accepted, 'goal rejected'
        result = handle.get_result_async()
        self.wait_for(result.done, seconds + 20.0, 'the goal result')
        assert result.result().status == GoalStatus.STATUS_SUCCEEDED
        time.sleep(0.6)             # let "stopped" be established for the mux

    def publish_trajectory(self, positions, seconds=0.5):
        """Publish a trajectory on the teleop controller's command topic (as Servo does)."""
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        whole = int(seconds)
        msg.points = [JointTrajectoryPoint(
            positions=[float(p) for p in positions],
            time_from_start=Duration(sec=whole, nanosec=int((seconds - whole) * 1e9)))]
        self.trajectory_pub.publish(msg)


def replay(env, fixture, log_path, start_delay=3.0):
    """Start the replay node on a servo fixture; return the process."""
    return Process(
        ['ros2', 'run', 'ranger_xarm_teleop_bridge', 'replay_node', '--ros-args',
         '-p', f'file:={os.path.join(SERVO_FIXTURES, fixture + ".jsonl")}',
         '-p', 'wait_for_subscriber_s:=15.0', '-p', f'start_delay_s:={start_delay}'],
        env, log_path)


# Releases that mean "data arrived late", not "the operator or a limit stopped it".
STALE_DATA = (TeleopState.HOLD_INPUT_STALE, TeleopState.HOLD_POSE_ABSENT,
              TeleopState.HOLD_HEAD_ABSENT, TeleopState.HOLD_TOOL_POSE_STALE)


def replay_with_clean_input(probe, env, log_path, fixture, expected_engaged_s, attempts=3,
                            before=None):
    """
    Replay a fixture from a disarmed stack; return ``(start, discarded)``.

    The tests that use this need the recording to reach the gate as a steady stream.
    On a busy machine data is sometimes more than 100 ms late (replay, transport or
    this node's own queue); the gate then releases on stale data, as it must, and the
    run says nothing about what the test is after. A run is discarded and repeated
    only when the gate itself reported such a release before ``expected_engaged_s`` of
    engagement. Every discarded run is printed and returned, so it stays visible.
    """
    discarded = []
    start = time.monotonic()
    for attempt in range(attempts):
        probe.disarm()
        if before is not None:
            before()
        start = time.monotonic()
        process = replay(env, fixture, f'{log_path}.{attempt}')
        try:
            assert process.wait(90.0) == 0, process.output()
        finally:
            process.stop()
        time.sleep(1.5)
        states = probe.timed_states_since(start)
        engaged = [when for when, s in states if s.clutch_engaged]
        if not engaged:
            # Nothing was armed at all: the opening button presses of the recording did
            # not reach the teleop node (they are sent in the first second, while the
            # new publisher is still being discovered).
            if not any(s.armed for _, s in states) and attempt + 1 < attempts:
                discarded.append(f'attempt {attempt + 1}: the arm button press never arrived')
                print(f'discarded replay of {fixture}: {discarded[-1]}')
                continue
            return start, discarded
        first = engaged[0]
        release = next(((when, s) for when, s in states
                        if when > first and not s.clutch_engaged), None)
        early = release is not None and release[0] - first < expected_engaged_s
        if early and release[1].hold_reason in STALE_DATA and attempt + 1 < attempts:
            discarded.append(
                f'attempt {attempt + 1}: released after {release[0] - first:.2f} s with '
                f'hold_reason {release[1].hold_reason} (data late by '
                f'{release[1].input_age * 1000.0:.0f} ms input age)')
            print(f'discarded replay of {fixture}: {discarded[-1]}')
            continue
        return start, discarded
    return start, discarded
