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
Gripper teleop in mock execution: replay, the held reference, ownership (INV-4 both ways).

One mock stack (``teleop_mock.launch.py``, gripper teleop on) for the module, on its own
ROS domain. In mock execution "MoveIt's gripper controller" is this repo's stand-in
(``xarm_gripper_traj_controller``); move_group cannot command it, so goals are sent to it
directly. The measured opening in mock execution is an echo of the command.
"""

import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from moveit_msgs.msg import ServoStatus
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, GripperCommand, TeleopState
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from stack_helpers import (assert_stack_nodes, BODY_OBSTACLE, BODY_OBSTACLE_LINK,
                           child_environment, enabled_rigid_pairs, INTO_BODY_START,
                           MOCK_STACK_NODES, Probe, Process, READY, replay,
                           replay_with_clean_input, tool_speed_profile)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

MOVEIT, TELEOP = 'xarm6_traj_controller', 'teleop_arm_controller'
MOVEIT_G, TELEOP_G = 'xarm_gripper_traj_controller', 'teleop_gripper_controller'
JOINT = 'xarm_drive_joint'
CLOSED = 0.849


class GripperProbe(Probe):
    """The stack probe plus the gripper command, the gripper joint and gripper clients."""

    def __init__(self, env, moveit_controller=MOVEIT, moveit_gripper=MOVEIT_G):
        super().__init__(env, moveit_controller=moveit_controller)
        self.moveit_gripper = moveit_gripper
        self.grips = []             # (monotonic, GripperCommand)
        self.gripper_joint = []     # (monotonic, position)
        self.node.create_subscription(GripperCommand, '/teleop/gripper/command',
                                      self._on_grip, 50)
        self.node.create_subscription(JointState, '/joint_states', self._on_gripper_joint, 50)
        self.gripper_action = ActionClient(self.node, FollowJointTrajectory,
                                           f'/{moveit_gripper}/follow_joint_trajectory')
        self.moveit_gripper_topic = self.node.create_publisher(
            JointTrajectory, f'/{moveit_gripper}/joint_trajectory', 10)
        self.teleop_gripper_topic = self.node.create_publisher(
            JointTrajectory, f'/{TELEOP_G}/joint_trajectory', 10)
        self.stray_command = self.node.create_publisher(
            GripperCommand, '/teleop/gripper/command', 10)
        self.switch_client = self.node.create_client(
            SwitchController, '/controller_manager/switch_controller')

    def _on_grip(self, msg):
        with self.lock:
            self.grips.append((time.monotonic(), msg))

    def _on_gripper_joint(self, msg):
        if JOINT in msg.name:
            with self.lock:
                self.gripper_joint.append((time.monotonic(),
                                           msg.position[list(msg.name).index(JOINT)]))

    def grips_since(self, t):
        with self.lock:
            return [m for when, m in self.grips if when >= t]

    def gripper_since(self, t):
        with self.lock:
            return [p for when, p in self.gripper_joint if when >= t]

    def gripper_position(self):
        with self.lock:
            return self.gripper_joint[-1][1] if self.gripper_joint else None

    def gripper_goal(self, position, seconds):
        assert self.gripper_action.wait_for_server(timeout_sec=15.0)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [JOINT]
        whole = int(seconds)
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=[float(position)], velocities=[0.0],
            time_from_start=Duration(sec=whole, nanosec=int((seconds - whole) * 1e9)))]
        future = self.gripper_action.send_goal_async(goal)
        self.wait_for(future.done, 15.0, 'the gripper goal response')
        return future.result()

    def move_gripper(self, position, seconds=1.0):
        handle = self.gripper_goal(position, seconds)
        assert handle is not None and handle.accepted, 'gripper goal rejected'
        result = handle.get_result_async()
        self.wait_for(result.done, seconds + 20.0, 'the gripper goal result')
        assert result.result().status == GoalStatus.STATUS_SUCCEEDED
        time.sleep(0.6)

    def switch(self, activate, deactivate):
        request = SwitchController.Request(activate_controllers=activate,
                                           deactivate_controllers=deactivate,
                                           strictness=SwitchController.Request.STRICT,
                                           activate_asap=True)
        return self.call(self.switch_client, request)


def point(position, seconds=0.1):
    msg = JointTrajectory()
    msg.joint_names = [JOINT]
    msg.points = [JointTrajectoryPoint(positions=[float(position)],
                                       time_from_start=Duration(nanosec=int(seconds * 1e9)))]
    return msg


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_gripper')
    env = child_environment(offset=9)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                      'rviz:=false'],
                     env, str(logs / 'launch.log'))
    probe = GripperProbe(env)
    try:
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: probe.controllers().get(TELEOP_G) == 'inactive', 60.0,
                       'the teleop gripper controller to be loaded inactive')
        probe.move_to(READY)
        probe.move_gripper(0.0)
        yield probe, env, logs, launch
    finally:
        probe.close()
        launch.stop()
        print(launch.output()[-6000:])


def arm(probe, env, logs, name, attempts=3):
    """
    Calibrate and arm with the arm_only recording; wait until teleop owns both.

    The arm button is pressed in the recording's first second, while the replay's new
    publisher is still being discovered; a run in which the press never arrived (no state
    says armed) is repeated, as replay_with_clean_input does.
    """
    for attempt in range(attempts):
        probe.disarm()
        start = time.monotonic()
        process = replay(env, 'arm_only', str(logs / f'replay_{name}.{attempt}.log'),
                         start_delay=1.0)
        try:
            assert process.wait(40.0) == 0, process.output()
        finally:
            process.stop()
        if any(s.armed for s in probe.states_since(start)) or attempt + 1 == attempts:
            break
        print(f'discarded replay {name}.{attempt}: the arm button press never arrived')
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_TELEOP, 10.0, 'teleop owner')
    return start


def test_both_pairs_start_on_moveits_side_and_the_mux_reports_the_gripper(stack):
    probe, _, _, _ = stack
    controllers = probe.controllers()
    assert controllers[MOVEIT] == 'active' and controllers[TELEOP] == 'inactive'
    assert controllers[MOVEIT_G] == 'active' and controllers[TELEOP_G] == 'inactive'
    assert probe.owner.gripper is True
    state = probe.state()
    assert state.gripper_enabled and not state.gripper_active
    assert state.gripper_hold == TeleopState.GRIPPER_HOLD_NOT_ARMED
    assert state.gripper_measured_valid


def test_replay_closes_and_opens_the_gripper_command_and_measured_on_topics(stack):
    probe, env, logs, _ = stack
    probe.disarm()
    start = time.monotonic()
    process = replay(env, 'gripper_clutch_strokes', str(logs / 'replay_strokes.log'))
    try:
        assert process.wait(60.0) == 0, process.output()
    finally:
        process.stop()
    time.sleep(1.0)
    grips = probe.grips_since(start)
    positions = [g.position for g in grips]
    assert positions and max(positions) == pytest.approx(CLOSED)
    assert positions[-1] == pytest.approx(0.0)
    closed_at = positions.index(max(positions))
    assert all(g.request == GripperCommand.REQUEST_CLOSE for g in grips[closed_at:closed_at + 5])
    measured = probe.gripper_since(start)
    assert max(measured) == pytest.approx(CLOSED, abs=0.01)
    assert measured[-1] == pytest.approx(0.0, abs=0.01)
    states = probe.states_since(start)
    assert any(s.gripper_active and s.clutch_engaged for s in states)
    assert any(s.gripper_active and not s.clutch_engaged for s in states)
    stamps = {(s.header.stamp.sec, s.header.stamp.nanosec) for s in states}
    assert all((g.header.stamp.sec, g.header.stamp.nanosec) in stamps for g in grips)
    probe.disarm()
    assert probe.controllers()[MOVEIT_G] == 'active'


def test_while_teleop_owns_moveits_gripper_controller_cannot_move_it(stack):
    probe, env, logs, _ = stack
    arm(probe, env, logs, 'own_teleop')
    controllers = probe.controllers()
    assert controllers[TELEOP_G] == 'active' and controllers[MOVEIT_G] == 'inactive'
    before = probe.gripper_position()
    start = time.monotonic()
    handle = probe.gripper_goal(0.6, 0.5)
    assert handle is None or not handle.accepted
    for _ in range(25):
        probe.moveit_gripper_topic.publish(point(0.6))
        time.sleep(0.02)
    time.sleep(0.5)
    assert max(abs(p - before) for p in probe.gripper_since(start)) < 1e-6
    probe.disarm()


def test_while_moveit_owns_teleops_gripper_side_cannot_move_it(stack):
    probe, _, _, _ = stack
    probe.disarm()
    before = probe.gripper_position()
    start = time.monotonic()
    for _ in range(25):
        probe.teleop_gripper_topic.publish(point(0.6))
        stray = GripperCommand(joint_name=JOINT, position=0.6,
                               request=GripperCommand.REQUEST_CLOSE)
        probe.stray_command.publish(stray)      # the relay forwards it; the controller ignores
        time.sleep(0.02)
    time.sleep(0.5)
    assert max(abs(p - before) for p in probe.gripper_since(start)) < 1e-6
    assert probe.controllers()[TELEOP_G] == 'inactive'


def test_arming_is_refused_while_moveits_gripper_goal_executes(stack):
    probe, env, logs, _ = stack
    probe.disarm()
    handle = probe.gripper_goal(0.5, 5.0)
    assert handle.accepted
    start = time.monotonic()
    process = replay(env, 'arm_only', str(logs / 'replay_refused.log'), start_delay=1.0)
    try:
        assert process.wait(40.0) == 0
    finally:
        process.stop()
    states = probe.states_since(start)
    assert any(s.mux_status in (ArmOwner.STATUS_REFUSED_GOAL_ACTIVE,
                                ArmOwner.STATUS_REFUSED_MOVING) for s in states)
    assert not any(s.gripper_active for s in states)
    assert probe.grips_since(start) == []
    result = handle.get_result_async()
    probe.wait_for(result.done, 20.0, 'the gripper goal result')
    assert result.result().status == GoalStatus.STATUS_SUCCEEDED
    assert probe.gripper_position() == pytest.approx(0.5, abs=0.01)
    time.sleep(0.6)


def test_arming_starts_from_the_opening_moveit_left_and_disarming_keeps_teleops(stack):
    probe, env, logs, _ = stack
    probe.disarm()
    probe.move_gripper(0.37)
    start = arm(probe, env, logs, 'held')
    grips = probe.wait_for(lambda: probe.grips_since(start), 5.0, 'a gripper command')
    assert grips[0].position == pytest.approx(0.37, abs=1e-6)
    assert grips[0].reference_source == GripperCommand.SOURCE_HELD
    assert all(g.position == grips[0].position for g in probe.grips_since(start))
    assert max(abs(p - 0.37) for p in probe.gripper_since(start)) < 1e-3
    active = [s for s in probe.states_since(start) if s.gripper_active]
    assert active and all(s.gripper_hold == TeleopState.GRIPPER_HOLD_AWAIT_TRIGGER
                          and s.gripper_reference_source == GripperCommand.SOURCE_HELD
                          for s in active)
    # The replay has ended: input stale, the gripper frozen at the same command.
    assert probe.state().gripper_hold == TeleopState.GRIPPER_HOLD_INPUT
    assert probe.state().gripper_command == pytest.approx(0.37, abs=1e-6)
    # Disarm: MoveIt's controller takes over holding what teleop last commanded.
    before = time.monotonic()
    probe.disarm()
    time.sleep(1.0)
    assert probe.controllers()[MOVEIT_G] == 'active'
    assert max(abs(p - 0.37) for p in probe.gripper_since(before)) < 1e-3


def test_a_moving_gripper_refuses_the_switch(stack):
    """The contract: only with arm and gripper stopped. A move without a goal (topic)."""
    probe, _, _, _ = stack
    probe.disarm()
    start = probe.gripper_position()
    target = 0.1 if start > 0.45 else 0.8
    probe.moveit_gripper_topic.publish(point(target, seconds=3.0))
    time.sleep(0.8)
    response = probe.set_owner(ArmOwner.OWNER_TELEOP)
    assert not response.success
    assert response.status == ArmOwner.STATUS_REFUSED_MOVING, response.detail
    assert 'gripper moved' in response.detail and 'xarm_drive_joint' in response.detail
    assert probe.controllers()[MOVEIT_G] == 'active'
    probe.wait_for(lambda: abs(probe.gripper_position() - target) < 1e-3, 10.0,
                   'the gripper to arrive')
    time.sleep(0.6)


def test_a_split_is_reported_never_switched_to_teleop_and_resolved_back_to_moveit(stack):
    probe, _, _, _ = stack
    probe.disarm()
    result = probe.switch([TELEOP_G], [MOVEIT_G])
    assert result.ok
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_NONE, 5.0, 'no owner')
    response = probe.set_owner(ArmOwner.OWNER_TELEOP)
    assert not response.success and 'split' in response.detail
    controllers = probe.controllers()
    assert controllers[TELEOP] == 'inactive' and controllers[TELEOP_G] == 'active'
    response = probe.set_owner(ArmOwner.OWNER_MOVEIT)
    assert response.success, response.detail
    controllers = probe.controllers()
    assert controllers[MOVEIT_G] == 'active' and controllers[TELEOP_G] == 'inactive'
    assert controllers[MOVEIT] == 'active'


# ---- gripper_fix: the arm is not slowed by a closed gripper; Servo still stops it ----

SPEED_TOLERANCE = 0.10          # peak tool speed, closed vs open, relative
TRAVEL_TOLERANCE = 0.005        # largest tool displacement, closed vs open, m


def speed_run(probe, env, logs, case):
    """Replay ``gripper_speed_<case>`` from READY with the gripper open; return the states."""
    probe.disarm()
    probe.move_to(READY)
    probe.move_gripper(0.0)
    start, _ = replay_with_clean_input(probe, env, str(logs / f'replay_speed_{case}.log'),
                                       f'gripper_speed_{case}', expected_engaged_s=2.5)
    time.sleep(1.0)
    return probe.states_since(start)


def assert_same_tool_speed(states_open, states_closed):
    engaged_open = [s for s in states_open if s.armed and s.clutch_engaged]
    engaged_closed = [s for s in states_closed if s.armed and s.clutch_engaged]
    assert len(engaged_open) > 100 and len(engaged_closed) > 100
    assert all(s.gripper_measured < 0.01 for s in engaged_open)
    assert all(s.gripper_measured > CLOSED - 0.01 for s in engaged_closed)
    peak_open, travel_open, codes_open = tool_speed_profile(states_open)
    peak_closed, travel_closed, codes_closed = tool_speed_profile(states_closed)
    print(f'peak tool speed open {peak_open:.4f} closed {peak_closed:.4f} m/s; '
          f'travel open {travel_open:.4f} closed {travel_closed:.4f} m')
    assert peak_open > 0.15                     # the hand's peak is 0.19 m/s
    assert abs(peak_closed / peak_open - 1.0) <= SPEED_TOLERANCE
    assert abs(travel_closed - travel_open) <= TRAVEL_TOLERANCE
    # MoveIt Servo never slowed either run (its first engaged report can still be the
    # previous engagement's last).
    for codes in (codes_open, codes_closed):
        assert ServoStatus.DECELERATE_FOR_COLLISION not in codes[1:], codes


def test_tool_speed_is_the_same_with_the_gripper_open_and_closed(stack):
    probe, env, logs, _ = stack
    pairs = enabled_rigid_pairs(probe)
    if pairs:
        pytest.skip(f'the SRDF leaves {len(pairs)} self-collision pairs enabled that no arm '
                    f'motion can change (MoveIt Servo slows the arm while the gripper is '
                    f'closed): {pairs}')
    assert_same_tool_speed(speed_run(probe, env, logs, 'open'),
                           speed_run(probe, env, logs, 'closed'))


def test_servo_still_stops_the_arm_before_a_collision_with_the_gripper_closed(stack):
    """
    The into-the-body test of test_mock_stack.py, gripper closed (INV-1).

    Meaningful once the SRDF disables the rigid pairs (the change this test guards): while
    they are enabled, MoveIt Servo already slows the arm for the gripper's own pairs and the
    node's release of an arm stuck near a collision can come before Servo's halt.
    """
    probe, env, logs, _ = stack
    pairs = enabled_rigid_pairs(probe)
    if pairs:
        pytest.skip(f'the SRDF leaves {len(pairs)} self-collision pairs enabled that no arm '
                    f'motion can change; run with them disabled: {pairs}')
    probe.disarm()
    probe.move_to(INTO_BODY_START)
    probe.move_gripper(CLOSED)
    probe.scene_box(*BODY_OBSTACLE, attach_to=BODY_OBSTACLE_LINK)
    start_tool = probe.tool_position()
    try:
        start = time.monotonic()
        process = replay(env, 'clutch_into_body', str(logs / 'replay_into_body_closed.log'))
        try:
            assert process.wait(40.0) == 0, process.output()
        finally:
            process.stop()
        time.sleep(1.0)
        states = probe.states_since(start)
    finally:
        probe.remove_scene_box(BODY_OBSTACLE[0], attach_to=BODY_OBSTACLE_LINK)
    engaged = [s for s in states if s.clutch_engaged]
    assert len(engaged) > 60
    assert all(s.gripper_measured > CLOSED - 0.01 for s in engaged)
    halted = [s for s in states if s.hold_reason == TeleopState.HOLD_SERVO_HALTED]
    assert halted, 'the halt for collision was never reported'
    assert all(s.servo_status == ServoStatus.HALT_FOR_COLLISION and not s.clutch_engaged
               for s in halted)
    first_halt = states.index(halted[0])
    assert not any(s.clutch_engaged for s in states[first_halt:])    # grip held: latched
    end_tool = probe.tool_position()
    travelled = start_tool[0] - end_tool[0]
    assert 0.05 < travelled < 0.23
    last = [s.measured_tool_pose.position.x for s in states[-40:]]
    assert max(last) - min(last) < 0.001
    probe.disarm()
    probe.move_gripper(0.0)
    probe.move_to(READY)


def test_no_other_stack_or_test_shares_this_domain(stack):
    """Isolation: the domain holds this stack's nodes and the probe, nothing else."""
    assert_stack_nodes(stack[0], stack[3], MOCK_STACK_NODES)
