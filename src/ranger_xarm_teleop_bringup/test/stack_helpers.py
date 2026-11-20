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

import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import PoseStamped
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
# The into-the-body tests (MoveIt Servo's collision halt, the backstop): the arm leans
# forward so that pulling the tool 0.28 m back does not bring it near the robot's fixed
# structure, where the link keep-out would stop it first; a box in the planning scene,
# which the keep-out does not know, is in the way instead (Servo follows move_group's
# scene). The tool starts at x 0.53 m and meets the box after about 0.12 m.
INTO_BODY_START = [0.0, 0.2, -1.4, 0.0, 1.2, 0.0]
BODY_OBSTACLE = ('into_body_obstacle', (0.37, 0.0, 0.26), (0.02, 0.30, 0.20))
BODY_OBSTACLE_LINK = 'xarm_link_base'     # attached there: checked as the robot's own body
# The shoulder tests: pulled back over joint 1's axis. From the ready posture the link
# keep-out stops link 2 before the structure behind the arm, short of the shoulder region;
# from here (the tool 0.1 m higher) the same hand motion crosses it with link 2 clear.
SHOULDER_START = [0.0, -0.5, -1.1, 0.0, 1.6, 0.0]
HERE = os.path.dirname(os.path.abspath(__file__))
SERVO_FIXTURES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'ranger_xarm_teleop_servo',
                              'test', 'fixtures')


def band_domain(base: int, offset: int) -> int:
    """Return the domain ``offset`` places after ``base`` in the band 10-99 (never 0-9)."""
    return 10 + (base - 10 + offset) % 90


# ---- logs kept when a test fails ----
#
# Every child process's output file (``Process``) and every ROS log directory of a child
# environment (``child_environment`` sets ``ROS_LOG_DIR``, so the launched nodes' own log
# files land there) is registered with the test module that created it. When a test fails and
# ``TELEOP_TEST_KEEP_LOGS`` names a directory, conftest.py copies what its module registered
# into a new subdirectory there, so an intermittent failure keeps its evidence.

KEEP_LOGS_ENV = 'TELEOP_TEST_KEEP_LOGS'
_LOGS = []      # (test module path, file or directory)
# The ROS_LOG_DIR and ROS_HOME of each child environment; removed when the session ends
# (conftest.py).
TEMP_LOG_DIRS = []


def _current_module() -> str:
    """Return the module of the running test ('' outside a test)."""
    return os.environ.get('PYTEST_CURRENT_TEST', '').split('::')[0]


def register_log(path: str) -> None:
    """Keep ``path`` (a file or a directory) for the running test module."""
    _LOGS.append((_current_module(), path))


def registered_logs(module: str):
    """Return the paths registered by ``module`` (a test module path, as pytest names it)."""
    return [path for owner, path in _LOGS if owner == module]


def keep_logs(module: str, test: str, destination: str) -> str:
    """Copy what ``module`` registered into a new directory under ``destination``."""
    stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{os.path.basename(module)}__{test}')[:150]
    target = os.path.join(destination, f'{stamp}_{name}')
    os.makedirs(target, exist_ok=True)
    for index, path in enumerate(registered_logs(module)):
        dest = os.path.join(target, f'{index:02d}_{os.path.basename(path.rstrip(os.sep))}')
        try:
            if os.path.isdir(path):
                shutil.copytree(path, dest, dirs_exist_ok=True)
            elif os.path.isfile(path):
                shutil.copy2(path, dest)
        except OSError as err:          # a log being written as it is copied: say so, go on
            with open(dest + '.copy_error.txt', 'w', encoding='utf-8') as handle:
                handle.write(f'{path}: {err}\n')
    return target


def child_environment(offset: int) -> dict:
    """
    Return an environment for child processes and this test's own ROS context.

    The domain is ``offset`` places after the session's domain in the band 10-99, so that
    test modules, also run in parallel, do not see each other and never reach domains 0-9;
    discovery stays on this machine; an RViz that a launch starts anyway draws off screen,
    where it aborts (and so fails the test) rather than open a window; the controller
    spawners' lock is this environment's own (``ROS_HOME``). A parallel worker
    (pytest-xdist) needs the session's domain to be set and in the band: it fails rather than
    derive its domain from a default.
    """
    env = dict(os.environ)
    raw = env.get('ROS_DOMAIN_ID', '')
    if 'PYTEST_XDIST_WORKER' in env and not (raw.isdigit() and 10 <= int(raw) <= 99):
        raise AssertionError('parallel stack tests need ROS_DOMAIN_ID set to the domain of this '
                             f'checkout, 10-99 (got {raw!r})')
    base = int(raw) if raw.isdigit() else 0
    env['ROS_DOMAIN_ID'] = str(band_domain(base, offset))
    env['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    env.pop('ROS_STATIC_PEERS', None)
    env['QT_QPA_PLATFORM'] = 'offscreen'
    # The launched nodes' own log files (rclcpp/rclpy file logging) for this environment only,
    # kept with the test's other logs when it fails.
    env['ROS_LOG_DIR'] = tempfile.mkdtemp(prefix=f'teleop_test_ros_log_{offset}_')
    register_log(env['ROS_LOG_DIR'])
    TEMP_LOG_DIRS.append(env['ROS_LOG_DIR'])
    # The controller spawners take one lock file under ROS_HOME (~/.ros unless set), whatever
    # the domain. With a home of its own, a spawner of this stack neither waits for nor holds
    # up the spawners of another stack on this machine (a parallel worker's, or a stack
    # started by hand), and a spawner that hangs blocks this stack only.
    env['ROS_HOME'] = tempfile.mkdtemp(prefix=f'teleop_test_ros_home_{offset}_')
    TEMP_LOG_DIRS.append(env['ROS_HOME'])
    return env


class Process:
    """A child process in its own process group, with its output in a log file."""

    def __init__(self, command, env, log_path):
        self.log_path = log_path
        register_log(str(log_path))
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


def prime(client, request, timeout=30.0, each=3.0):
    """
    Ask ``request`` until one answer arrives; return how often it was asked.

    For a client or a server that is new. The middleware drops a service's answer when the
    server has not yet matched the client's reply subscription: rmw_fastrtps waits 100 ms
    for that match, then logs "failed to send response ... client will not receive response"
    and the call never returns. The client cannot see from its side whether the server has
    matched it (``wait_for_service`` shows the client's view only). One answered round trip
    shows that it has, and a match stays. ``request`` must change nothing (a read, or a
    request for what already holds); its answer is thrown away. Only it is ever asked again:
    the call whose answer a test asserts on is made once, after it. Every lost answer is
    printed; no answer within ``timeout`` fails.
    """
    assert client.wait_for_service(timeout_sec=timeout), f'no service {client.srv_name}'
    end = time.monotonic() + timeout
    asked = 0
    while time.monotonic() < end:
        asked += 1
        future = client.call_async(request)
        limit = min(end, time.monotonic() + each)
        while not future.done() and time.monotonic() < limit:
            time.sleep(0.01)
        if future.done():
            return asked
        client.remove_pending_request(future)
        print(f'{client.srv_name}: no answer to harmless request {asked} within {each} s '
              '(the server has not matched this client yet?); asked again')
    raise AssertionError(f'{client.srv_name} answered none of {asked} harmless requests within '
                         f'{timeout} s')


class Probe:
    """A node that watches the teleop stack and acts on it, spun in a background thread."""

    def __init__(self, env, moveit_controller='xarm6_traj_controller',
                 teleop_controller='teleop_arm_controller'):
        self.domain = env['ROS_DOMAIN_ID']
        self.context = rclpy.Context()
        rclpy.init(context=self.context, domain_id=int(env['ROS_DOMAIN_ID']))
        self.node = rclpy.create_node('teleop_test_probe', context=self.context)
        self.lock = threading.Lock()
        self.states = []            # (monotonic, TeleopState)
        self.joints = []            # (monotonic, [q1..q6])
        self.owner = None
        self.inputs = []            # monotonic arrival of every input status
        self.input_stamps = []      # (monotonic arrival, header stamp s) of every input status
        self.stamped_joints = []    # (monotonic, header stamp, [q1..q6])
        self.targets = []           # (monotonic, (x, y, z)) of /teleop/target_pose
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
        self.node.create_subscription(PoseStamped, '/teleop/target_pose', self._on_target, 50)
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
        try:
            player = _PLAYERS.pop(self.domain, None)
            if player is not None:
                player.close()
        finally:
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

    def _on_target(self, msg):
        p = msg.pose.position
        with self.lock:
            self.targets.append((time.monotonic(), (p.x, p.y, p.z)))

    def _on_description(self, msg):
        with self.lock:
            self.description = msg.data

    def _on_input(self, msg):
        now = time.monotonic()
        with self.lock:
            self.inputs.append(now)
            self.input_stamps.append(
                (now, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9))

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

    def wait_for_streams(self, timeout=30.0):
        """Wait until this probe receives the teleop state, joint states and the owner."""
        def missing():
            with self.lock:
                return [name for name, got in (('/teleop/state', self.states),
                                               ('/joint_states', self.joints),
                                               ('/teleop/mux/owner', self.owner)) if not got]
        end = time.monotonic() + timeout
        while missing() and time.monotonic() < end:
            time.sleep(0.05)
        assert not missing(), (f'the probe received nothing on {missing()} within {timeout} s '
                               f'(spin errors: {self.spin_errors})')

    def prime_clients(self):
        """
        Make sure the stack answers this probe's own clients (``prime``).

        With requests that change nothing: the controller list, a disarm of a disarmed teleop
        node, and the owner the mux already reports.
        """
        prime(self.list_client, ListControllers.Request())
        prime(self.disarm_client, Trigger.Request())
        owner = self.owner_value()
        if owner in (ArmOwner.OWNER_MOVEIT, ArmOwner.OWNER_TELEOP):
            prime(self.set_owner_client, SetArmOwner.Request(owner=owner), each=6.0)

    def call(self, client, request, timeout=15.0):
        assert client.wait_for_service(timeout_sec=timeout), f'no service {client.srv_name}'
        future = client.call_async(request)
        self.wait_for(future.done, timeout, f'answer from {client.srv_name}')
        return future.result()

    # ---- observations ----

    def state(self):
        with self.lock:
            return self.states[-1][1] if self.states else None

    def state_age(self):
        """Return how long ago the newest TeleopState arrived (s), None before the first."""
        with self.lock:
            newest = self.states[-1][0] if self.states else None
        return None if newest is None else time.monotonic() - newest

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

    def input_stamps_since(self, t):
        """Return ``(arrival, stamp)`` of the input statuses that arrived since ``t``."""
        with self.lock:
            return [pair for pair in self.input_stamps if pair[0] >= t]

    def joint_positions(self):
        with self.lock:
            return list(self.joints[-1][1]) if self.joints else None

    def joints_since(self, t):
        with self.lock:
            return [q for when, q in self.joints if when >= t]

    def targets_since(self, t):
        with self.lock:
            return [(when, p) for when, p in self.targets if when >= t]

    def timed_joints_since(self, t):
        with self.lock:
            return [(when, q) for when, q in self.joints if when >= t]

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

    def joint_speeds_in_mock_execution_since(self, t):
        """Return the largest joint speed since ``t`` as mock execution moves (rad/s)."""
        with self.lock:
            rows = [(when, stamp.sec + stamp.nanosec * 1e-9, q)
                    for when, stamp, q in self.stamped_joints]
        first = next((i for i, row in enumerate(rows) if row[0] >= t), len(rows))
        # One state before ``t`` as well: the first step's period ends at the first state.
        return commanded_period_speed([(stamp, q) for _, stamp, q in rows[max(0, first - 1):]])

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

    def scene_box(self, name, centre, size, frame='xarm_link_base', attach_to=None):
        """
        Add a box to move_group's planning scene, which MoveIt Servo follows.

        An obstacle the link keep-out does not know (it covers the robot description
        only): Servo's collision check is then the only layer that sees it. With
        ``attach_to`` (a link fixed to the arm base) the box is part of the robot, checked
        as self-collision like the robot's own structure (Servo's self-collision threshold,
        not the larger scene one).
        """
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose as PoseMsg
        box = CollisionObject()
        box.header.frame_id = frame
        box.id = name
        box.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(size))]
        pose = PoseMsg()
        pose.position.x, pose.position.y, pose.position.z = centre
        pose.orientation.w = 1.0
        box.primitive_poses = [pose]
        box.operation = CollisionObject.ADD
        scene = PlanningScene(is_diff=True)
        if attach_to is None:
            self._apply_scene(scene, box, ApplyPlanningScene)
            return
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [
            AttachedCollisionObject(link_name=attach_to, object=box)]
        self._apply_scene(scene, None, ApplyPlanningScene)

    def remove_scene_box(self, name, frame='xarm_link_base', attach_to=None):
        """Remove a box added with ``scene_box`` (detached first if it was attached)."""
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        box = CollisionObject()
        box.header.frame_id = frame
        box.id = name
        box.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        if attach_to is not None:
            # Detaching puts the object back into the world; the world removal follows.
            scene.robot_state.is_diff = True
            scene.robot_state.attached_collision_objects = [
                AttachedCollisionObject(link_name=attach_to, object=box)]
        self._apply_scene(scene, box, ApplyPlanningScene)

    def _apply_scene(self, scene, box, service):
        if box is not None:
            scene.world.collision_objects = [box]
        client = getattr(self, '_scene_client', None)
        if client is None:
            from moveit_msgs.msg import PlanningScene
            client = self._scene_client = self.node.create_client(service,
                                                                  '/apply_planning_scene')
            # A new client: an empty change first, until move_group answers it.
            prime(client, service.Request(scene=PlanningScene(is_diff=True)), timeout=10.0)
        future = client.call_async(service.Request(scene=scene))
        self.wait_for(future.done, 10.0, 'the planning scene change')
        assert future.result().success, 'the planning scene change was refused'
        time.sleep(1.0)             # Servo's scene follows move_group's monitored scene

    def publish_path(self, positions, times, lead=0.5):
        """Publish a timed joint path on the teleop controller's command topic (as Servo does)."""
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        for q, t in zip(positions, times):
            t = float(t) + lead
            whole = int(t)
            msg.points.append(JointTrajectoryPoint(
                positions=[float(p) for p in q],
                time_from_start=Duration(sec=whole, nanosec=int((t - whole) * 1e9))))
        self.trajectory_pub.publish(msg)

    def publish_trajectory(self, positions, seconds=0.5):
        """Publish a trajectory on the teleop controller's command topic (as Servo does)."""
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        whole = int(seconds)
        msg.points = [JointTrajectoryPoint(
            positions=[float(p) for p in positions],
            time_from_start=Duration(sec=whole, nanosec=int((seconds - whole) * 1e9)))]
        self.trajectory_pub.publish(msg)


# ---- one input source per stack, matched and acknowledged before any test input ----
#
# A replay used to be a new process, with new publishers, every time. Its first frames were
# then sometimes lost while the teleop node and the new publishers discovered each other:
# ``replay_node`` waits for a subscriber on the status topic, which another node can be, and
# start delays only make the loss rarer. The calibrate and arm presses are in a recording's
# first second, so a test then saw no arming. Now a stack has one input source for its whole
# life (``input_player.py``: the bridge's own code path, as ``replay_node``). Before any test
# sends input, ``start_input`` waits for the two things that rule out loss to discovery:
#
# 1. matched: each publisher a frame is built from (status, joy, right pose, head pose) is
#    matched by every subscription known on its topic, the teleop node's among them;
# 2. acknowledged: the teleop node calibrated from the lead-in's calibrate press, which it
#    can only do from a frame with all four parts (``TeleopState.calibrated`` under the
#    lead-in's own XR session id).
#
# The lead-in is input no test asserts on: the frames of ``arm_only`` up to its arm press,
# under a session id of their own, so a fixture's first frame changes the session and the
# stack is not calibrated for it, as on a fresh stack. It is sent again and again until it is
# acknowledged, and how many passes that took is printed.

INPUT_PLAYER = os.path.join(HERE, 'input_player.py')
INPUT_NODE = '/teleop_test_input'       # the player's node (input_player.NODE_NAME)
TELEOP_NODE = 'teleop_servo'            # the node that takes the input
LEAD_IN_SESSION = 1                     # the lead-in's XR session id: no fixture uses it
QUIET_TIMEOUT_S = 2.0                   # a finished recording's input must have stopped by then
_PLAYERS = {}                           # ROS domain -> InputPlayer of the stack on it


class Replay:
    """One recording being played by an ``InputPlayer``; stands in for a replay process."""

    def __init__(self, player, ident, path, log_path):
        self.player, self.ident, self.path, self.log_path = player, ident, path, log_path
        self.started = time.monotonic()
        self.proc = self                # ``replay.proc.returncode``, ``.poll()``: as a process
        self._quiet = False             # the teleop node has seen this recording's input stop
        if log_path is not None:
            register_log(str(log_path))

    def _events(self):
        return self.player.events_of(self.ident)

    def events(self):
        """Return the events the player has reported about this recording so far."""
        return self._events()

    def _done(self):
        return next((e for e in self._events() if e.get('event') == 'done'), None)

    @property
    def returncode(self):
        """
        Return None while playing, then 0, 1 or 2.

        0: every record was sent (a looped recording: it was stopped, which is how it ends).
        1: the recording could not be read or played, or the player has gone.
        2: stopped before its last record, by ``stop`` or by the next recording.
        """
        done = self._done()
        if done is None:
            return None if self.player.running() else 1
        if done.get('failed'):
            return 1
        return 2 if done.get('stopped') and not done.get('loop') else 0

    @property
    def passes(self):
        """Return how many complete passes have been sent so far."""
        return max((int(e.get('passes', 0)) for e in self._events()
                    if e.get('event') in ('pass', 'done')), default=0)

    def poll(self):
        return self.returncode

    def _ended(self):
        """
        Return the exit code once the recording has ended for the stack too, else None.

        A recording sent to its last record has ended, for a test, when the teleop node sees
        its input stop (``TeleopState.input_age`` beyond the gate's timeout). A replay process
        took longer than that to exit, and tests read the stack's state right after waiting
        for it. Fails if the node still reports fresh input ``QUIET_TIMEOUT_S`` later.
        """
        code = self.returncode
        if code != 0 or self._quiet:
            return code
        done = self._done()
        probe = self.player.probe
        state = probe.state() if probe is not None else None
        if (done.get('loop') or state is None
                or not 0.0 <= state.input_age <= INPUT_TIMEOUT_S):
            self._quiet = True
            return code
        if time.time() - done.get('t', time.time()) >= QUIET_TIMEOUT_S:
            # The newest state the probe has: fresh input in it means something else sends
            # input only if that state is itself new.
            age = probe.state_age()
            assert age is not None and age < 0.5, (
                f'{self.path} was sent to its end over {QUIET_TIMEOUT_S} s ago and no teleop '
                f'state has reached the probe for {0.0 if age is None else age:.1f} s (the last '
                'one still showed fresh input): the teleop node or the probe has stopped')
            raise AssertionError(
                f'{self.path} was sent to its end over {QUIET_TIMEOUT_S} s ago and the teleop '
                f'node still reports fresh input (age {state.input_age * 1000.0:.0f} ms, seq '
                f'{state.input_seq}): something else is sending input')
        return None

    def wait(self, timeout):
        """Return the exit code, or None when not ended (``_ended``) after ``timeout``."""
        end = time.monotonic() + timeout
        code = self._ended()
        while code is None and time.monotonic() < end:
            time.sleep(0.02)
            code = self._ended()
        if code is not None:
            self._write_log()
        return code

    def output(self):
        """Return what the player reported about this recording."""
        lines = [json.dumps(e) for e in self._events()]
        return f'{self.path}\n' + '\n'.join(lines) + '\n' + self.player.output()[-3000:]

    def stop(self):
        """
        Stop the recording if it is still playing.

        Waits for the player to report it stopped, not for the stack's input to go quiet
        (``wait``): this runs in ``finally`` blocks, where a check that fails would replace
        the failure of the test itself.
        """
        if self.returncode is None:
            self.player.send(cmd='stop', id=self.ident)
            end = time.monotonic() + 5.0
            while self.returncode is None and time.monotonic() < end:
                time.sleep(0.02)
        self._write_log()

    def _write_log(self):
        if self.log_path is not None:
            with open(self.log_path, 'w', encoding='utf-8') as handle:
                handle.write(self.output())


class InputPlayer:
    """The input source of one stack: ``input_player.py`` as a child process."""

    def __init__(self, env, log_path):
        self.log_path = str(log_path)
        register_log(self.log_path)
        self.log = open(self.log_path, 'w', encoding='utf-8')
        self.proc = subprocess.Popen(
            [sys.executable, '-u', INPUT_PLAYER], env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, start_new_session=True)
        self.lock = threading.Lock()
        self.events = []
        self.probe = None               # the stack's probe (start_input): what the node sees
        self.acknowledged = False
        self.matched = {}               # the "matched" report (topic -> counts)
        self.lead_in_passes = None
        self._next = 0
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                event = {'event': 'text', 'text': line.rstrip()}
            with self.lock:
                self.events.append(event)

    def running(self):
        return self.proc.poll() is None

    def events_of(self, ident):
        with self.lock:
            return [e for e in self.events if e.get('id') == ident]

    def output(self):
        """Return the player's log so far."""
        self.log.flush()
        with open(self.log_path, encoding='utf-8', errors='replace') as handle:
            return handle.read()

    def send(self, **command):
        assert self.running(), f'the input player has exited:\n{self.output()[-3000:]}'
        self.proc.stdin.write(json.dumps(command) + '\n')
        self.proc.stdin.flush()

    def wait_event(self, name, timeout):
        """Return the first event called ``name``; fail when none comes in ``timeout``."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                found = next((e for e in self.events if e.get('event') == name), None)
            if found is not None:
                return found
            assert self.running(), f'the input player has exited:\n{self.output()[-3000:]}'
            time.sleep(0.02)
        raise AssertionError(f'the input player reported no "{name}" within {timeout} s:\n'
                             f'{self.output()[-3000:]}')

    def play(self, path, start_delay=0.0, loop=False, source='', log_path=None):
        """Play a recording; return its ``Replay``."""
        self._next += 1
        replay = Replay(self, self._next, path, log_path)
        self.send(cmd='play', id=self._next, path=str(path), start_delay=float(start_delay),
                  loop=bool(loop), source=source)
        return replay

    def close(self):
        """Tell the player to leave; kill what is left. Never raises."""
        try:
            if self.running():
                try:
                    self.proc.stdin.write(json.dumps({'cmd': 'quit'}) + '\n')
                    self.proc.stdin.flush()
                    self.proc.stdin.close()
                except (OSError, ValueError):
                    pass
                try:
                    self.proc.wait(10.0)
                except subprocess.TimeoutExpired:
                    pass
            if self.running():          # never a process group whose leader was reaped
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    self.proc.wait(5.0)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            self.log.close()


def lead_in_file(directory):
    """
    Write the lead-in recording; return ``(path, records)``.

    The frames of ``arm_only`` before its arm press (so with its calibrate press and nothing
    else), under ``LEAD_IN_SESSION``.
    """
    lines, calibrate = [], 0
    with open(os.path.join(SERVO_FIXTURES, 'arm_only.jsonl'), encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            msg = record.get('msg', {})
            buttons = msg.get('controllers', {}).get('right', {}).get('buttons', [])
            if len(buttons) > 5 and buttons[5].get('p'):
                break                               # the arm press: not part of the lead-in
            calibrate += bool(len(buttons) > 4 and buttons[4].get('p'))
            msg['session_id'] = LEAD_IN_SESSION
            lines.append(json.dumps(record))
    assert calibrate and len(lines) > calibrate, (
        'arm_only no longer starts with a calibrate press that ends before its arm press')
    path = os.path.join(str(directory), 'lead_in.jsonl')
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(lines) + '\n')
    return path, len(lines)


def lead_in_acknowledged(states) -> bool:
    """Return True when a teleop state reports the calibration made from the lead-in."""
    return any(s.calibrated and s.calibration_session_id == LEAD_IN_SESSION for s in states)


def start_input(probe, env, logs, match_timeout=60.0, acknowledge_timeout=30.0):
    """
    Start the stack's input source; return once the teleop node provably takes its input.

    Call it in the module's fixture, when the stack is up and before anything else is sent:
    ``replay`` refuses to play without it. It first waits until the probe itself receives the
    stack's streams (teleop state, joint states, owner), which the tests read, and until the
    stack answers the probe's service clients (``Probe.prime_clients``). Fails, with
    what is known, when the publishers are not matched within ``match_timeout`` or the
    lead-in is not acknowledged within ``acknowledge_timeout``.
    """
    assert env['ROS_DOMAIN_ID'] not in _PLAYERS, 'this stack already has an input player'
    probe.wait_for_streams()
    probe.prime_clients()
    player = InputPlayer(env, os.path.join(str(logs), 'input_player.log'))
    player.probe = probe
    _PLAYERS[env['ROS_DOMAIN_ID']] = player
    player.wait_event('up', 60.0)
    player.send(cmd='match', consumer=TELEOP_NODE, timeout=match_timeout)
    report = player.wait_event('matched', match_timeout + 10.0)
    player.matched = report['topics']
    assert report['ok'], (
        f'the input publishers are not matched after {match_timeout} s; per topic [matched by '
        f'the publisher, subscriptions known, /{TELEOP_NODE} among them]: {report["topics"]}')
    path, records = lead_in_file(logs)
    since = time.monotonic()
    lead_in = player.play(path, loop=True, source='replay:lead_in',
                          log_path=os.path.join(str(logs), 'lead_in.log'))
    try:
        probe.wait_for(lambda: (lead_in_acknowledged(probe.states_since(since))
                                or not player.running()),
                       acknowledge_timeout, 'the lead-in to be acknowledged')
        assert lead_in_acknowledged(probe.states_since(since)), (
            f'the input player exited during the lead-in:\n{player.output()[-3000:]}')
        # Complete passes before the acknowledging state arrived, by the player's own times
        # (not by when this thread came to look).
        arrived = next(when for when, s in probe.timed_states_since(since)
                       if lead_in_acknowledged([s]))
        seen = time.time() - (time.monotonic() - arrived)
        passes = sum(1 for e in lead_in.events() if e.get('event') == 'pass'
                     and e.get('t', 0.0) <= seen)
    except AssertionError as err:
        if 'the input player exited' in str(err):
            raise
        state = probe.state()
        seen = ('no teleop state' if state is None else
                f'input_age {state.input_age:.3f} s, input_seq {state.input_seq}, calibrated '
                f'{state.calibrated} (session {state.calibration_session_id})')
        raise AssertionError(
            f'the teleop node did not calibrate from the lead-in within {acknowledge_timeout} '
            f's although the publishers were matched ({report["topics"]}). Its last state: '
            f'{seen}. A fresh input_age with no calibration means the status arrives and the '
            'joy, pose or head messages do not (see "calibration refused" and "incomplete" in '
            'the teleop log).') from None
    finally:
        lead_in.stop()
    player.lead_in_passes = passes
    print(f'input player: matched {report["topics"]}; lead-in ({records} records) '
          f'acknowledged in pass {passes + 1}, {time.monotonic() - since:.2f} s after it began'
          + (f': {passes} pass(es) before it were not taken' if passes else ''))
    # Quiet again before the first recording, as between two recordings.
    probe.wait_for(lambda: not 0.0 <= probe.state().input_age <= 0.3, 10.0,
                   'the input to go quiet after the lead-in')
    player.acknowledged = True
    return player


def input_player(env):
    """Return the stack's acknowledged input player; fail when the fixture started none."""
    player = _PLAYERS.get(env['ROS_DOMAIN_ID'])
    assert player is not None and player.acknowledged, (
        'this stack has no acknowledged input player: call start_input(probe, env, logs) in '
        'the fixture, before any replay')
    return player


def replay_file(env, path, log_path, start_delay=3.0):
    """Play a JSONL recording on the stack's input player; return the ``Replay``."""
    return input_player(env).play(path, start_delay=start_delay, log_path=log_path)


def replay(env, fixture, log_path, start_delay=3.0):
    """Play a servo fixture on the stack's input player; return the ``Replay``."""
    return replay_file(env, os.path.join(SERVO_FIXTURES, fixture + '.jsonl'), log_path,
                       start_delay)


# Frames a teleop cycle (50 Hz) can be behind the newest frame it reports, with a 90 Hz
# input that may arrive in bursts: ``TeleopState.input_seq`` is the newest frame of the cycle,
# so the first one a state reports can be later than the first one the node received.
PRESS_SEQ_MARGIN = 10


def press_frames(path: str):
    """
    Return ``(first, last)``: the seqs of the first and last frame with A or B pressed.

    The right controller's A (calibrate) and B (arm); in ``arm_only`` A is 27-44, B 63-80.
    """
    first = last = None
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            msg = json.loads(line).get('msg', {})
            if msg.get('type') != 'xr_frame':
                continue
            buttons = msg.get('controllers', {}).get('right', {}).get('buttons', [])
            if any(len(buttons) > i and buttons[i].get('p') for i in (4, 5)):
                first = int(msg['seq']) if first is None else first
                last = int(msg['seq'])
    if first is None:
        raise ValueError(f'{path}: no A or B press')
    return first, last


def press_never_arrived(seqs, press_last: int, margin: int = PRESS_SEQ_MARGIN) -> bool:
    """
    Return True when none of a replay's presses can have reached the teleop node.

    ``seqs``: ``TeleopState.input_seq`` in order since the replay started; the leading entries
    can still be the previous stream's newest frame. The replay's own frames start at the
    first change. True only when none arrived, or when the first one reported is more than
    ``margin`` frames after the **last** pressed frame (``press_frames``). The gate reads
    buttons as levels and keeps the previous stream's, so any frame of a press registers it;
    it also keeps the calibration of the same XR session, so the arm press alone can arm; and
    a state can report a frame or a burst later than the first one received. Anything nearer
    counts as arrived: a node that may have got a press and did not arm is a real failure,
    never given another try.
    """
    seqs = list(seqs)
    if not seqs:
        return True
    start = next((i for i, seq in enumerate(seqs) if seq != seqs[0]), None)
    if start is None:
        return True                                     # no frame of the replay arrived
    return seqs[start] > press_last + margin


def arm_by_replay(probe, env, log_prefix, fixture='arm_only', start_delay=1.0, disarm=True):
    """
    Calibrate and arm with ``fixture``, once; return ``(start, discarded attempts)``.

    Nothing is repeated, so the second value is always empty (kept for the callers): the
    stack's input player was matched and acknowledged before any test input
    (``start_input``), so the presses in the recording's first second cannot be lost to
    discovery. A press that arrived without arming is returned as it is, for the caller's
    assertions to fail on: a refusal, a mux failure, a lost ownership answer and a bug in
    arming all look like that. Should the teleop node have armed nothing **and** its own
    ``TeleopState.input_seq`` show that no press can have reached it
    (``press_never_arrived``), this fails at once and says so: the input failed, not the
    stack, and the caller's assertion would name the wrong thing. ``disarm=False`` leaves
    out the disarm request before the recording (the caller's stack is disarmed already).
    """
    first, last = press_frames(os.path.join(SERVO_FIXTURES, fixture + '.jsonl'))
    if disarm:
        probe.disarm()
    start = time.monotonic()
    process = replay(env, fixture, f'{log_prefix}.0', start_delay=start_delay)
    try:
        assert process.wait(40.0) == 0, process.output()
    finally:
        process.stop()
    states = probe.states_since(start)
    seqs = [s.input_seq for s in states]
    if not any(s.armed for s in states) and press_never_arrived(seqs, last):
        changed = next((seq for seq in seqs if seq != seqs[0]), None) if seqs else None
        raise AssertionError(
            f'{fixture}: nothing was armed, and the first frame the teleop node reported was '
            f'seq {changed} (presses at seq {first}-{last}): the presses never arrived, '
            'although the input player was matched and acknowledged')
    return start, []


# Releases on data that came too late: of the headset input, and of the joint states.
STALE_INPUT = (TeleopState.HOLD_INPUT_STALE, TeleopState.HOLD_POSE_ABSENT,
               TeleopState.HOLD_HEAD_ABSENT)
STALE_DATA = STALE_INPUT + (TeleopState.HOLD_TOOL_POSE_STALE,)
LATE_WINDOW_S = 0.5         # how far before a release the cause is looked for
LATE_SLACK_S = 0.02         # one target cycle
INPUT_TIMEOUT_S = 0.1       # the gate's input timeout
TOOL_POSE_TIMEOUT_S = 0.1   # the gate's tool_pose_timeout_s
STEADY_GAP_S = 0.05         # teleop states (50 Hz) arriving closer than this: the probe is awake


def late_on_the_way(probe, release, when):
    """
    Return ``(late, text)``: whether a stale release's data was late outside the teleop node.

    ``release`` is the first released TeleopState, ``when`` its arrival at the probe. Only
    what the probe can show counts as late on the way:

    * ``INPUT_STALE``: by the stamps the input source gave its status messages (its time of
      sending), not by when they reached the probe: a probe that paused changes arrivals, not
      stamps. Late when two frames in a row were sent at least as far apart as the gate saw
      no input (``release.input_age``, less one cycle): the source itself paused.
    * ``TOOL_POSE_STALE``: by the arrival of the joint states at the probe (their stamps are
      the robot side's clock, which a slow simulator keeps even). Late when they left a gap of
      at least the gate's timeout, less one cycle, while teleop states kept arriving at the
      probe through that gap: neither the probe nor the teleop node was what paused.
    * ``POSE_ABSENT``, ``HEAD_ABSENT``: never. A pose, head or joy message lost on the way and
      one the teleop node took too late look the same from the probe, which takes none of
      those topics.
    """
    reason = release.hold_reason
    begin = when - LATE_WINDOW_S
    if reason == TeleopState.HOLD_INPUT_STALE:
        # Messages that arrived around the window (an arrival can be late; a stamp cannot).
        stamps = [stamp for arrival, stamp in probe.input_stamps_since(begin - 0.3)
                  if arrival <= when + 0.3]
        gap = max((b - a for a, b in zip(stamps, stamps[1:])), default=0.0)
        age = release.input_age if release.input_age > 0.0 else INPUT_TIMEOUT_S
        needed = max(STEADY_GAP_S, age - LATE_SLACK_S)
        return gap >= needed, (f'the input source sent two frames {gap * 1000.0:.0f} ms apart '
                               f'(the gate saw no input for {age * 1000.0:.0f} ms)')
    if reason == TeleopState.HOLD_TOOL_POSE_STALE:
        joints = [begin] + [t for t, _ in probe.timed_joints_since(begin) if t <= when] + [when]
        gap, a, b = max((y - x, x, y) for x, y in zip(joints, joints[1:]))
        states = [a] + [t for t, _ in probe.timed_states_since(a) if t < b] + [b]
        awake = max(y - x for x, y in zip(states, states[1:])) < STEADY_GAP_S
        return gap >= TOOL_POSE_TIMEOUT_S - LATE_SLACK_S and awake, (
            f'the joint states left a gap of {gap * 1000.0:.0f} ms at the probe, teleop states '
            f'{"kept arriving" if awake else "did not arrive either"} meanwhile')
    return False, ('the probe takes the status topic only: it cannot tell a pose, head or joy '
                   'message lost on the way from one the teleop node took too late')


def replay_with_clean_input(probe, env, log_path, fixture, expected_engaged_s, attempts=3,
                            before=None):
    """
    Replay a fixture from a disarmed stack; return ``(start, discarded)``.

    The tests that use this need the recording to reach the gate as a steady stream. On a
    busy machine data is sometimes more than 100 ms late; the gate then releases on stale
    data, as it must, and the run says nothing about what the test is after. A run is
    discarded and repeated only when

    * the gate itself reported such a release before ``expected_engaged_s`` of engagement,
    * **and** the data was late outside the teleop node, as far as the probe can show
      (``late_on_the_way``): the input source paused, or the robot side's joint states did.
      No test here asserts on either.

    Any other stale release is returned as it is, with a printed line, for the test to fail
    on: the teleop node being late itself is the stack's behaviour. So is a release for any
    other reason, and a run in which nothing engaged. Every discarded run is printed and
    returned, so it stays visible.
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
            # Nothing engaged: for the caller's assertions. (A press lost to discovery cannot
            # be the reason: the input player was acknowledged before any test input.)
            return start, discarded
        first = engaged[0]
        release = next(((when, s) for when, s in states
                        if when > first and not s.clutch_engaged), None)
        early = release is not None and release[0] - first < expected_engaged_s
        if not (early and release[1].hold_reason in STALE_DATA):
            return start, discarded
        late, how = late_on_the_way(probe, release[1], release[0])
        what = (f'released after {release[0] - first:.2f} s with hold_reason '
                f'{release[1].hold_reason}; {how}')
        if not late:
            print(f'replay of {fixture} not repeated: {what}: not shown late on its way, so '
                  "it counts as the stack's own")
            return start, discarded
        if attempt + 1 == attempts:
            print(f'replay of {fixture} not repeated again after {attempts} attempts: {what}')
            return start, discarded
        discarded.append(f'attempt {attempt + 1}: {what}: late on its way')
        print(f'discarded replay of {fixture}: {discarded[-1]}')
    return start, discarded


# ---- a process that joins the graph during an engagement ----
#
# A process that joins can stall an existing node for about a third of a second: the
# middleware sleeps 333 ms, up to three times, while a node opens the newcomer's shared-memory
# port (interface contract, "Service and action calls": every process of a session starts
# before arming). The gate then releases the clutch on stale data, as it must.

JOIN_STALL_REASONS = (TeleopState.HOLD_INPUT_STALE, TeleopState.HOLD_TOOL_POSE_STALE,
                      TeleopState.HOLD_SERVO_NOT_READY)
JOIN_STALL_MIN_S = 0.25     # one sleep is 0.333 s
JOIN_STALL_MAX_S = 1.2      # three in a row are 1.0 s
JOIN_WINDOW_S = 5.0         # a newcomer has been discovered up to 3 s after its start
JOIN_LOOK_BACK_S = 1.5
JOIN_LOOK_AHEAD_S = 1.0     # an input pause outlasts the release it causes


def pauses_around(probe, when, servo_output=()):
    """
    Return the longest pause of each stream the probe takes around the time ``when``.

    Every pause is measured by the stamps the sender gave its messages, between two messages
    in a row that reached the probe around ``when``. So a pause is the sender's: a probe that
    paused itself changes when messages arrive, not their stamps, and a stream that simply
    ended is no pause. ``servo_output`` is ``(arrival, stamp)`` of MoveIt Servo's output.
    Call this at least JOIN_LOOK_AHEAD_S after ``when``.
    """
    begin, end = when - JOIN_LOOK_BACK_S, when + JOIN_LOOK_AHEAD_S

    def secs(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def longest(stamps):
        return max((b - a for a, b in zip(stamps, stamps[1:])), default=0.0)

    with probe.lock:
        joints = [secs(stamp) for arrival, stamp, _ in probe.stamped_joints
                  if begin <= arrival <= end]
    return {
        'the input source sent no frame': longest(
            [stamp for arrival, stamp in probe.input_stamps_since(begin - 0.3)
             if arrival <= end + 0.3]),
        'the teleop node published no state': longest(
            [secs(s.header.stamp) for t, s in probe.timed_states_since(begin) if t <= end]),
        'the controller published no joint state': longest(joints),
        'MoveIt Servo published no output': longest(
            [stamp for arrival, stamp in servo_output if begin <= arrival <= end]),
    }


def join_stall(joined_at, when, reason, pauses):
    """
    Return ``(True, text)`` when a release is the stall that a joining process causes.

    ``joined_at`` is when the process was started, ``when`` and ``reason`` the release and
    its ``hold_reason``, ``pauses`` the result of ``pauses_around``. All three are needed:

    * the release came after the process was started, within JOIN_WINDOW_S (in a test that
      starts the process a second into a six second engagement that is all the rest of it:
      the two conditions below are what tells a stall from anything else);
    * its reason is one a stalled node causes: stale input, a stale tool pose, or MoveIt
      Servo's output not following. Never a missing pose, a halt or anything else;
    * a sender paused for the length of a stall around the release, by its own stamps.

    Anything else is ``(False, why not)``.
    """
    after = when - joined_at
    if not 0.0 <= after <= JOIN_WINDOW_S:
        return False, (f'it came {abs(after):.2f} s {"after" if after > 0.0 else "before"} the '
                       f'process was started, not within {JOIN_WINDOW_S:g} s after it')
    if reason not in JOIN_STALL_REASONS:
        return False, (f'hold_reason {reason} is not what a stalled node causes (stale input '
                       f'{TeleopState.HOLD_INPUT_STALE}, stale tool pose '
                       f'{TeleopState.HOLD_TOOL_POSE_STALE}, Servo output stopped '
                       f'{TeleopState.HOLD_SERVO_NOT_READY})')
    listed = ', '.join(f'{name} for {gap * 1000.0:.0f} ms' for name, gap in pauses.items())
    stalled = {name: gap for name, gap in pauses.items()
               if JOIN_STALL_MIN_S <= gap <= JOIN_STALL_MAX_S}
    if not stalled:
        return False, (f'no stream shows a pause of stall length ({JOIN_STALL_MIN_S:g} to '
                       f'{JOIN_STALL_MAX_S:g} s) around it; the longest: {listed}')
    name = max(stalled, key=stalled.get)
    return True, (f'{after:.2f} s after the process was started {name} for '
                  f'{stalled[name] * 1000.0:.0f} ms')


def commanded_period_speed(samples):
    """
    Return the largest joint speed in ``samples`` ((stamp in s, positions)) of mock execution.

    Mock execution has no dynamics: at each controller update the hardware reports the
    position commanded at the update before, stamped now. The step between two joint states
    was therefore made during the period that ended at the first of them, and its speed is
    the step over that period: the previous stamp difference, not the step's own.

    The two differ here because the controller's period is uneven without real-time
    scheduling (6.2 to 7.1 ms around 6.67, at times a doubled one). Dividing a step by its
    own stamp difference reads the true speed times the ratio of two neighbouring periods,
    up to a tenth too high or too low. Under Gazebo the positions are measured at their
    stamps, and ``Probe.joint_speeds_since`` is the right measure there.

    No state is left out. A joint state that never reached the probe makes one step two
    periods long over a divisor of one period: that step reads twice the speed (the step
    after it reads half), so the largest reading is too high and the bound fails.
    """
    best = 0.0
    for (s0, _), (s1, q1), (_, q2) in zip(samples, samples[1:], samples[2:]):
        period = s1 - s0
        if period > 1e-4:
            best = max(best, max(abs(a - b) for a, b in zip(q2, q1)) / period)
    return best


# ---- tool speed with the gripper open and closed (gripper_fix) ----

SERVO_GROUP = 'xarm6'           # the move_group_name of config/servo.yaml


def enabled_rigid_pairs(probe):
    """
    Return the self-collision pairs MoveIt Servo checks that no arm motion can change.

    Derived from the running robot description and the SRDF move_group reads (the file of
    ``ranger_xarm_moveit_config``; MoveIt Servo follows move_group's collision matrix).
    While any is left enabled, a closed gripper can slow the arm (README, Gripper).
    """
    from ament_index_python.packages import get_package_share_directory
    from ranger_xarm_teleop_servo.rigid_pairs import rigid_pairs
    probe.wait_for(lambda: probe.description is not None, 30.0, 'the robot description')
    path = os.path.join(get_package_share_directory('ranger_xarm_moveit_config'), 'config',
                        'ranger_xarm.srdf')
    with open(path, encoding='utf-8') as handle:
        return rigid_pairs(probe.description, handle.read(), SERVO_GROUP)


def tool_speed_profile(states, window=5):
    """
    Return (peak tool speed m/s, largest displacement m, servo codes) of the engaged cycles.

    The speed is the measured tool position's change over ``window`` cycles, by the state
    stamps; the displacement is from the first engaged cycle.
    """
    engaged = [s for s in states if s.armed and s.clutch_engaged]
    stamps = [s.header.stamp.sec + s.header.stamp.nanosec * 1e-9 for s in engaged]
    points = [(s.measured_tool_pose.position.x, s.measured_tool_pose.position.y,
               s.measured_tool_pose.position.z) for s in engaged]
    speeds = [math.dist(points[i], points[i - window]) / (stamps[i] - stamps[i - window])
              for i in range(window, len(engaged)) if stamps[i] > stamps[i - window]]
    displacement = max((math.dist(p, points[0]) for p in points), default=0.0)
    return max(speeds, default=0.0), displacement, [int(s.servo_status) for s in engaged]


# ---- isolation of the stack tests ----

#: Nodes of launched processes that exit while the stack keeps running, by the executable
#: name the launch log gives the process (``[spawner-3]: process has finished``): a spawner
#: leaves once it has loaded its controller. Any other exit fails. The mock stacks start
#: without RViz (``rviz:=false``); one that started anyway would fail, by its exit (it
#: aborts when it draws off screen) or by its node.
EXITING_NODES = {'spawner': (r'/spawner_\w+',)}

#: Every node of a mock stack (``teleop_mock.launch.py``) while it runs, as patterns, each
#: matched by exactly this many nodes; names with a number change from run to run.
MOCK_STACK_NODES = {
    **{re.escape(name): 1 for name in (
        '/arm_mux', '/arm_mux_client', '/arm_mux_observer', '/controller_manager',
        '/gripper_relay', '/joint_state_broadcaster', '/move_group', '/move_group/moveit',
        '/moveit_simple_controller_manager', '/operator_feedback', '/robot_state_publisher',
        '/servo_node',
        '/servo_node/moveit', '/teleop_arm_controller', '/teleop_gripper_controller',
        '/teleop_servo', '/teleop_servo_watcher', '/xarm6_traj_controller',
        '/xarm_gripper_traj_controller', '/xarm_mock_components_genericsystem',
        '/xarm_xarmgrippersystem')},
    r'/move_group_private_\d+': 1, r'/servo_node_private_\d+': 1, r'/moveit_\d+': 2,
    r'/transform_listener_impl_[0-9a-f]+': 2,
}
#: The same with ``record:=true``: the recorder and the episode node as well.
MOCK_RECORD_NODES = {**MOCK_STACK_NODES, '/episode_node': 1, '/teleop_episode_recorder': 1}

_PROCESS_EVENT = re.compile(r'^\[(?:INFO|ERROR)\] \[(.+)-\d+\]: process (started|has died|has '
                            r'finished)', re.MULTILINE)


def full_name(name, namespace):
    """Return a node's fully qualified name."""
    return ('' if namespace == '/' else namespace) + '/' + name


def launched_processes(launch):
    """Return the executables the launch has started and those of them that have exited."""
    started, exited = set(), set()
    for executable, event in _PROCESS_EVENT.findall(launch.output()):
        (started if event == 'started' else exited).add(executable)
    return started, exited


def exited_node_patterns(launch):
    """
    Return the compiled patterns of the nodes whose launched process has exited.

    Fails when a process has exited whose nodes are not known (``EXITING_NODES``): a stack
    process that dies is not something to ignore.
    """
    _, exited = launched_processes(launch)
    unknown = sorted(exited - set(EXITING_NODES))
    assert not unknown, f'launched processes have exited: {unknown}'
    return tuple(re.compile(p + r'\Z') for e in sorted(exited) for p in EXITING_NODES[e])


def ignoring(names, patterns):
    """Return the names that match none of the patterns."""
    return [n for n in names if not any(p.match(n) for p in patterns)]


def graph_nodes(probe, settle=2.0):
    """
    Return the fully qualified names of every node on the probe's ROS domain, sorted.

    Waits ``settle`` seconds first so that discovery has caught up with every participant.
    """
    time.sleep(settle)
    return sorted(full_name(name, ns) for name, ns in probe.node.get_node_names_and_namespaces())


def assert_stack_nodes(probe, launch, expected, settle=2.0):
    """
    Assert that the probe's ROS domain holds this stack, this probe and nothing else.

    Each pattern of ``expected`` must be matched by exactly its count of nodes, the probe by
    one and the stack's input player (when the fixture started one) by one, and no node may be
    left over: a second stack, another test or any unrelated node
    fails. Nodes of launched processes that have exited are left out (``EXITING_NODES``);
    those of such a process still running are allowed.
    """
    seen = graph_nodes(probe, settle)
    started, exited = launched_processes(launch)
    left = ignoring(seen, exited_node_patterns(launch))
    running = tuple(re.compile(p + r'\Z') for e in sorted(started - exited)
                    for p in EXITING_NODES.get(e, ()))
    wrong = []
    own = {re.escape('/' + probe.node.get_name()): 1}
    if getattr(probe, 'domain', None) in _PLAYERS:
        own[re.escape(INPUT_NODE)] = 1          # this stack's input player, and only one
    for pattern, count in {**expected, **own}.items():
        match = re.compile(pattern + r'\Z')
        hits = [n for n in left if match.match(n)]
        if len(hits) != count:
            wrong.append((pattern, count, hits))
        left = [n for n in left if n not in hits]
    left = ignoring(left, running)
    assert not wrong and not left, f'nodes not of this stack: {left}; counts wrong: {wrong}'


def node_pids(name, pgid=None, environ=None):
    """
    Return the pids of this test's processes running ``name`` (a node executable).

    Only processes of the process group ``pgid``, or with ``environ`` (a 'KEY=value' string)
    in their environment, count: never a process this test did not start.
    """
    out = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            if pgid is not None and os.getpgid(int(entry)) != pgid:
                continue
            if environ is not None:
                with open(f'/proc/{entry}/environ', 'rb') as handle:
                    if environ.encode() not in handle.read().split(b'\0'):
                        continue
            with open(f'/proc/{entry}/cmdline', 'rb') as handle:
                cmd = handle.read().replace(b'\0', b' ').decode(errors='replace')
        except (OSError, ProcessLookupError):
            continue
        if name in cmd and 'ros2 ' not in cmd:
            out.append(int(entry))
    return out


def cpu_seconds(pid):
    """Return the CPU time (user + system, s) the process has used so far."""
    with open(f'/proc/{pid}/stat', encoding='utf-8') as handle:
        fields = handle.read().rsplit(')', 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
