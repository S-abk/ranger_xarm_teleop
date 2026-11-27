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
MoveIt for a driven base in headless Gazebo (opt-in: ``TELEOP_TEST_GAZEBO=1``).

The simulator runs as in ``test_gazebo_base.py`` (``drive_base:=true``, cameras on), the stack
with ``base:=true reset:=true record:=true``. After a drive and a turn: every frame has one
parent and move_group plans in ``odom``, which is in the tree, with the base where the robot is;
move_group, MoveIt Servo and the session sidecar hold one semantic description; a reset plans
and executes. The box test is the regression test for Servo following move_group's planning
scene on a base path (before, Servo rejected it and saw no scene object): the same box, given in
the arm's base frame and then in ``odom`` after the drive, makes Servo slow the arm before
contact, and the teleop node releases it.
"""

import hashlib
import json
import math
import os
import threading
import time

from base_helpers import BaseWatcher, phase, play
from moveit_msgs.msg import ServoStatus
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, ResetState, TeleopState
from ranger_xarm_teleop_record import metadata as md
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import rclpy.time
from reset_helpers import off_ready, ResetWatcher
from stack_helpers import (BODY_OBSTACLE, child_environment, INTO_BODY_START,
                           kill_gz_partition, Process, READY, replay, start_input)
from test_mock_record import RecordProbe, wait_for_settled_graph
from tf2_msgs.msg import TFMessage

pytestmark = pytest.mark.skipif(os.environ.get('TELEOP_TEST_GAZEBO') != '1',
                                reason='set TELEOP_TEST_GAZEBO=1 to run the Gazebo tests')

MOVEIT, MOVEIT_G = 'xarm_xarm6_traj_controller', 'xarm_xarm_gripper_traj_controller'
FORWARD = (0.0, -1.0)
TURN = (1.0, 0.0)                # right stick x with the left grip held: turn in place
OFF = off_ready([0.6, 0.2, 0.2, 0.3, -0.3, 0.4])


class TfEdges:
    """Every parent -> child seen on /tf (in a window) and /tf_static (latched, kept)."""

    def __init__(self, node):
        self.lock = threading.Lock()
        self.edges = set()
        static = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                            reliability=ReliabilityPolicy.RELIABLE)
        node.create_subscription(TFMessage, '/tf', self._add('tf'), 100)
        node.create_subscription(TFMessage, '/tf_static', self._add('static'), static)

    def _add(self, kind):
        def callback(msg):
            with self.lock:
                for t in msg.transforms:
                    self.edges.add((t.header.frame_id.lstrip('/'), t.child_frame_id.lstrip('/'),
                                    kind))
        return callback

    def window(self, seconds):
        """Return {child: {parents}} over ``seconds`` of /tf plus everything latched."""
        with self.lock:
            self.edges = {e for e in self.edges if e[2] == 'static'}
        time.sleep(seconds)
        with self.lock:
            edges = set(self.edges)
        parents = {}
        for parent, child, _ in edges:
            parents.setdefault(child, set()).add(parent)
        return parents


def roots(parents):
    frames = set(parents) | {p for ps in parents.values() for p in ps}
    return sorted(f for f in frames if f not in parents)


def planning_scene(probe):
    """Return move_group's scene (robot state, fixed-frame transforms, object names)."""
    from moveit_msgs.msg import PlanningSceneComponents
    from moveit_msgs.srv import GetPlanningScene
    client = probe.node.create_client(GetPlanningScene, '/get_planning_scene')
    try:
        assert client.wait_for_service(timeout_sec=10.0), 'no /get_planning_scene'
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE | PlanningSceneComponents.TRANSFORMS
            | PlanningSceneComponents.WORLD_OBJECT_NAMES)
        future = client.call_async(request)
        probe.wait_for(future.done, 10.0, 'the planning scene')
        return future.result().scene
    finally:
        probe.node.destroy_client(client)


def parameter(probe, node, name):
    from rcl_interfaces.srv import GetParameters
    client = probe.node.create_client(GetParameters, f'{node}/get_parameters')
    try:
        assert client.wait_for_service(timeout_sec=10.0), f'no {node}/get_parameters'
        future = client.call_async(GetParameters.Request(names=[name]))
        probe.wait_for(future.done, 10.0, f'{node} {name}')
        return future.result().values[0].string_value
    finally:
        probe.node.destroy_client(client)


def transform(probe, parent, child):
    return probe.tf_buffer.lookup_transform(parent, child, rclpy.time.Time()).transform


def yaw(q):
    return 2.0 * math.atan2(q.z, q.w)


def box_in_odom(probe, name, centre, size, heading):
    """Add a box to move_group's scene given in odom, turned by ``heading`` about z."""
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject, PlanningScene
    from moveit_msgs.srv import ApplyPlanningScene
    from shape_msgs.msg import SolidPrimitive
    box = CollisionObject()
    box.header.frame_id = 'odom'
    box.id = name
    box.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(size))]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = centre
    pose.orientation.z, pose.orientation.w = math.sin(heading / 2), math.cos(heading / 2)
    box.primitive_poses = [pose]
    box.operation = CollisionObject.ADD
    probe._apply_scene(PlanningScene(is_diff=True), box, ApplyPlanningScene)


def rest(watch, extra=2.5, timeout=15.0):
    """Wait for ground truth at rest for 0.4 s, then ``extra`` seconds (the 2 s rule)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rows = watch.since('gt', time.monotonic() - 0.4)
        if rows and all(math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y) < 0.005
                        and abs(m.twist.twist.angular.z) < 0.01 for _, m in rows):
            time.sleep(extra)
            return
        time.sleep(0.1)
    raise AssertionError('the base did not come to rest by ground truth')


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('gazebo_basemoveit')
    data = str(tmp_path_factory.mktemp('data'))
    session = 'gazebo_basemoveit_' + time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    md.write_labels(data, 'test-task', 'test-operator')
    env = child_environment(offset=31)
    env['GZ_PARTITION'] = f'teleop_basemoveit_test_{os.getpid()}'
    env['GZ_IP'] = '127.0.0.1'
    sim = Process(['ros2', 'launch', 'ranger_xarm_gazebo', 'sim.launch.py', 'headless:=true',
                   'sensors:=true', 'drive_base:=true'], env, str(logs / 'sim.log'))
    probe = RecordProbe(env, moveit_controller=MOVEIT, moveit_gripper=MOVEIT_G)
    teleop = None
    try:
        watch = BaseWatcher(probe)
        resets = ResetWatcher(probe)
        tf = TfEdges(probe.node)
        probe.wait_for(lambda: probe.joint_positions() is not None, 180.0, 'joint states')
        probe.wait_for(lambda: probe.controllers().get(MOVEIT) == 'active', 120.0,
                       'the arm controller')
        probe.wait_for(lambda: watch.since('odom', 0.0) and watch.since('gt', 0.0), 120.0,
                       'odometry and ground truth of the drivable base')
        teleop = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_sim.launch.py',
                          'base:=true', 'reset:=true', 'record:=true', f'record_data_dir:={data}',
                          f'record_session:={session}'], env, str(logs / 'teleop.log'))
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 120.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: resets.last() is not None and resets.last().watchdog_alive
                       and resets.last().enabled, 60.0, 'the reset node and its watchdog')
        probe.wait_for(lambda: watch.last() is not None and watch.last().enabled
                       and watch.last().at_rest, 60.0, 'the base node, enabled, base at rest')
        time.sleep(5.0)             # MoveIt Servo start-up and verification
        probe.move_to(READY, seconds=4.0)
        start_input(probe, env, logs)
        wait_for_settled_graph(probe)
        yield {'probe': probe, 'env': env, 'logs': logs, 'watch': watch, 'resets': resets,
               'tf': tf, 'session': os.path.join(data, session), 'results': {}}
    finally:
        probe.close()
        if teleop is not None:
            teleop.stop()
        sim.stop()
        kill_gz_partition(env['GZ_PARTITION'])


def test_after_a_drive_every_frame_has_one_parent_and_move_group_plans_in_odom(stack):
    probe, watch, tf = stack['probe'], stack['watch'], stack['tf']
    before = tf.window(5.0)
    assert not {c: p for c, p in before.items() if len(p) > 1}
    play(stack, 'drive', [phase(0.4), phase(5.5, grip=True, left=FORWARD), phase(1.0),
                          phase(2.2, grip=True, right=TURN), phase(1.0)])
    rest(watch)
    parents = tf.window(5.0)
    assert not {c: p for c, p in parents.items() if len(p) > 1}, 'a frame with two parents'
    assert roots(parents) == ['odom'], roots(parents)
    base = transform(probe, 'odom', 'base_footprint')
    assert math.hypot(base.translation.x, base.translation.y) > 0.5, 'the base did not drive'
    heading = yaw(base.rotation)
    heading = math.atan2(math.sin(heading), math.cos(heading))
    assert abs(math.degrees(heading)) > 20.0, 'the base did not turn'
    scene = planning_scene(probe)
    frames = {t.header.frame_id for t in scene.fixed_frame_transforms}
    assert frames == {'odom'}, frames                       # the planning frame
    joints = scene.robot_state.multi_dof_joint_state
    assert list(joints.joint_names) == ['virtual_joint'] and joints.header.frame_id == 'odom'
    planar = joints.transforms[0]
    assert math.hypot(planar.translation.x - base.translation.x,
                      planar.translation.y - base.translation.y) < 0.005
    turn = yaw(planar.rotation) - yaw(base.rotation)
    assert abs(math.atan2(math.sin(turn), math.cos(turn))) < math.radians(0.5)
    stack['results']['moved'] = True


def test_move_group_servo_and_the_sidecar_hold_one_semantic_description(stack):
    probe = stack['probe']
    move_group = parameter(probe, '/move_group', 'robot_description_semantic')
    servo = parameter(probe, '/servo_node', 'robot_description_semantic')
    assert move_group == servo
    assert '<virtual_joint name="virtual_joint" type="planar" parent_frame="odom"' in servo
    with open(os.path.join(stack['session'], 'session.json'), encoding='utf-8') as handle:
        recorded = json.load(handle)['robot']['semantic_sha256']
    assert recorded == hashlib.sha256(servo.encode()).hexdigest()


def test_a_reset_after_driving_plans_and_executes(stack):
    probe, resets = stack['probe'], stack['resets']
    assert stack['results'].get('moved'), 'needs the drive of the first test'
    probe.move_to(OFF, seconds=4.0)
    rest(stack['watch'])
    before = resets.count()
    start, _, _ = play(stack, 'reset', [phase(0.5), phase(14.0, right=(0.0, 1.0)), phase(0.5)])
    arrived = resets.first(start, ResetState.STATE_ARRIVED, ResetState.REASON_REACHED,
                           after=before)
    assert arrived is not None, [(s.state, s.reason) for _, s in resets.since(start, before)]
    q = probe.joint_positions()
    assert max(abs(a - b) for a, b in zip(q, READY)) < 0.02


def approach(stack, name):
    """Pull slowly toward the box (clutch_into_body_slow); return the teleop states."""
    probe = stack['probe']
    probe.disarm()
    probe.move_to(INTO_BODY_START)
    start = time.monotonic()
    process = replay(stack['env'], 'clutch_into_body_slow', str(stack['logs'] / f'{name}.log'))
    assert process.wait(90.0) == 0, process.output()
    time.sleep(1.0)
    states = probe.states_since(start)
    probe.disarm()
    return states


def slowed_then_released(states):
    """Assert Servo slowed the arm before contact and the teleop node released it."""
    engaged = [s for s in states if s.clutch_engaged]
    assert len(engaged) > 60, 'the clutch did not stay engaged'
    # servo_status is the latest code received: the first engaged states can carry the code
    # of the previous engagement, until Servo reports again.
    since = next((i for i, s in enumerate(states)
                  if s.clutch_engaged and s.servo_status == ServoStatus.NO_WARNING), None)
    assert since is not None, 'MoveIt Servo never reported while engaged'
    fresh = states[since:]
    slowed = next((i for i, s in enumerate(fresh) if s.clutch_engaged
                   and s.servo_status == ServoStatus.DECELERATE_FOR_COLLISION), None)
    assert slowed is not None, 'MoveIt Servo never slowed the arm for the box'
    assert not any(s.servo_status == ServoStatus.HALT_FOR_COLLISION for s in fresh)
    released = next((i for i, s in enumerate(fresh)
                     if s.hold_reason == TeleopState.HOLD_SERVO_HALTED), None)
    assert released is not None and released > slowed, 'the teleop node did not release'
    return fresh[released].measured_tool_pose.position.x


def test_a_box_given_in_the_arms_frame_slows_the_arm_and_it_is_released(stack):
    probe = stack['probe']
    name, centre, size = BODY_OBSTACLE
    probe.scene_box(name, centre, size)               # in xarm_link_base
    time.sleep(1.5)
    try:
        stack['results']['arm_frame_x'] = slowed_then_released(approach(stack, 'box_arm'))
    finally:
        probe.remove_scene_box(name)


def test_the_same_box_given_in_odom_after_the_drive_does_the_same(stack):
    """Servo and move_group place the base where the robot is, away from the odom origin."""
    probe = stack['probe']
    assert stack['results'].get('moved'), 'needs the drive of the first test'
    name, centre, size = BODY_OBSTACLE
    probe.move_to(INTO_BODY_START)
    arm = transform(probe, 'odom', 'xarm_link_base')
    heading = yaw(arm.rotation)
    c, s = math.cos(heading), math.sin(heading)
    in_odom = (arm.translation.x + c * centre[0] - s * centre[1],
               arm.translation.y + s * centre[0] + c * centre[1],
               arm.translation.z + centre[2])
    box_in_odom(probe, name + '_odom', in_odom, size, heading)
    time.sleep(1.5)
    try:
        x = slowed_then_released(approach(stack, 'box_odom'))
    finally:
        probe.remove_scene_box(name + '_odom', frame='odom')
    if 'arm_frame_x' in stack['results']:
        assert abs(x - stack['results']['arm_frame_x']) < 0.003, (x, stack['results'])
