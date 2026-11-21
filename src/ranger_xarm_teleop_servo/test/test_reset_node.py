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
The reset node and its watchdog node (G-16), wired to fakes of move_group, on a fake clock.

The nodes' callbacks are driven directly with real messages; move_group's action and cancel
path are fakes that can lose their answers. What the nodes publish is captured. The decisions
themselves are tested without ROS (test/unit/test_reset.py, test_reset_watchdog.py); here: the
messages in and out, the goal as sent, the order "announced, then sent", the cancel requests
by the goal's own id, and that nothing else is published or asked.
"""

import math

from action_msgs.msg import GoalInfo, GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped
import pytest
from ranger_xarm_teleop_interfaces.msg import (ArmOwner, EpisodeState, InputStatus, ResetState,
                                               ResetWatchdog, TeleopState)
from ranger_xarm_teleop_servo import reset_node, reset_watchdog_node
from ranger_xarm_teleop_servo.reset_node import ResetNode, StartupRefused
from ranger_xarm_teleop_servo.reset_watchdog_node import ResetWatchdogNode
from ranger_xarm_teleop_servo.session import ARM_JOINTS, OWNER_MOVEIT, OWNER_TELEOP
import rclpy
from rclpy.parameter import Parameter
import rclpy.time
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String

READY = [0.0, -0.6, -0.7, 0.0, 1.3, 0.0]
START = [0.4, -0.2, -0.3, 0.1, 1.0, 0.2]
SIM_DESCRIPTION = ('<robot name="r"><ros2_control name="a" type="system"><hardware><plugin>'
                   'mock_components/GenericSystem</plugin></hardware></ros2_control></robot>')
REAL_DESCRIPTION = ('<robot name="r"><ros2_control name="a" type="system"><hardware><plugin>'
                    'some_vendor/RealArm</plugin></hardware></ros2_control></robot>')
PARAMS = {'enabled': True, 'reset_speed_scaling': 0.2, 'reset_posture_names': ['ready'],
          'reset_postures.ready': READY}


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


class FakeTime:

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def time(self):
        return 1.7e9 + self.t


class Future:

    def __init__(self):
        self.callbacks, self.value, self.finished = [], None, False

    def add_done_callback(self, cb):
        self.callbacks.append(cb)
        if self.finished:
            cb(self)

    def answer(self, value):
        self.value, self.finished = value, True
        for cb in self.callbacks:
            cb(self)

    def result(self):
        return self.value

    def done(self):
        return self.finished


class FakeCancel:
    """move_group's cancel path: requests are kept; ``answering``: are they answered."""

    srv_name = '/move_action/_action/cancel_goal'

    def __init__(self):
        self.requests, self.answering, self.ready = [], True, True

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        future = Future()
        self.requests.append((bytes(request.goal_info.goal_id.uuid), future))
        if self.answering:
            answer = CancelGoal.Response()
            known = any(bytes(request.goal_info.goal_id.uuid) == g for g in self.goals)
            answer.return_code = 0 if known else CancelGoal.Response.ERROR_UNKNOWN_GOAL_ID
            if known:
                answer.goals_canceling = [GoalInfo()]
            future.answer(answer)
        return future

    goals = ()


class FakeAction:
    """move_group's action: goals are kept; their answers are given by the test."""

    def __init__(self):
        self.goals, self.ready = [], True

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal, feedback_callback=None, goal_uuid=None):
        future = Future()
        self.goals.append((bytes(goal_uuid.uuid), goal, future, feedback_callback))
        return future


class Handle:

    def __init__(self, accepted):
        self.accepted = accepted
        self.result_future = Future()

    def get_result_async(self):
        return self.result_future


class Rig:
    """A ResetNode on a fake clock, with fakes of move_group and captured publishers."""

    def __init__(self, monkeypatch, **params):
        self.clock = FakeTime()
        monkeypatch.setattr(reset_node, 'time', self.clock)
        values = {**PARAMS, **params}
        self.node = ResetNode(node_name='reset_under_test', parameter_overrides=[
            Parameter(k, value=v) for k, v in values.items()])
        self.action, self.cancel = FakeAction(), FakeCancel()
        self.node.action, self.node.cancel_client = self.action, self.cancel
        self.beats, self.states, self.log = [], [], []
        self.node.beat_pub.publish = lambda m: (self.beats.append(m),
                                                self.log.append(('beat', bytes(m.goal_id))))
        self.node.state_pub.publish = self.states.append
        self.node.count_publishers = lambda topic: 1
        original = self.action.send_goal_async

        def send(goal, feedback_callback=None, goal_uuid=None):
            self.log.append(('send', bytes(goal_uuid.uuid)))
            return original(goal, feedback_callback, goal_uuid)
        self.action.send_goal_async = send
        self.seq = 0
        self.session = 7
        self.q = list(START)
        self.armed, self.owner, self.episode = False, OWNER_MOVEIT, EpisodeState.STATE_IDLE
        self.watchdog = True
        self.echo = True                # the watchdog echoes the newest heartbeat it heard
        self.node._on_description(String(data=SIM_DESCRIPTION))

    def stamp(self):
        return rclpy.time.Time(nanoseconds=int(1e9 + self.seq * 1e7)).to_msg()

    def frame(self, stick=0.0, trigger=0.0, pose=True, head=True, emulated=False,
              send_pose=True, send_joy=True, reset_count=0, axes=None):
        self.seq += 1
        stamp = self.stamp()
        if send_pose and pose:
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.pose.orientation.w = 1.0
            self.node._on_pose(msg)
        if head:
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.pose.orientation.w = 1.0
            self.node._on_head(msg)
        if send_joy:
            joy = Joy()
            joy.header.stamp = stamp
            joy.axes = axes if axes is not None else [0.0, 0.0, 0.0, 0.0, trigger, 0.0, 0.0, stick]
            joy.buttons = [0] * 10
            self.node._on_joy(joy)
        status = InputStatus()
        status.header.stamp = stamp
        status.session_id = self.session
        status.seq = self.seq
        status.xr_time_ms = 5000.0 + self.seq * 11.0
        status.reset_count = reset_count
        status.right.pose_present = pose
        status.right.emulated_position = emulated
        status.viewer_present = head
        self.node._on_status(status)

    def world(self):
        self.node._on_teleop(TeleopState(armed=self.armed, owner=self.owner))
        self.node._on_owner(ArmOwner(owner=self.owner))
        self.node._on_episode(EpisodeState(state=self.episode))
        if self.watchdog:
            echo = self.node.beat_seq if self.echo else 0
            self.node._on_watchdog(ResetWatchdog(seq=1, heartbeat_seq=echo))
        self.node._on_joints(JointState(name=list(ARM_JOINTS), position=list(self.q)))

    def step(self, stick=0.0, frame=True, **frame_kw):
        self.clock.t += 0.02
        if frame:
            self.frame(stick, **frame_kw)
        self.world()
        self.node._cycle()

    def hold(self, seconds, stick=1.0, **kw):
        for _ in range(int(round(seconds / 0.02))):
            self.step(stick, **kw)

    def state(self):
        return self.states[-1]

    def goal(self):
        self.hold(1.2, 0.0)             # move_group's cancel path answers; the stick released
        assert self.node.moveit_ready
        self.hold(0.6)
        assert len(self.action.goals) == 1, self.state().reason_text
        return self.action.goals[0]

    def moving(self):
        goal_id, goal, future, feedback = self.goal()
        handle = Handle(True)
        future.answer(handle)
        self.node._on_goal_status(GoalStatusArray(status_list=[GoalStatus(
            goal_info=GoalInfo(goal_id=_uuid(goal_id)), status=2)]))
        self.q[0] -= 0.05
        self.step(1.0)
        return goal_id, goal, handle

    def close(self):
        self.node.destroy_node()


def _uuid(goal_id):
    from unique_identifier_msgs.msg import UUID
    return UUID(uuid=list(goal_id))


@pytest.fixture
def rig(ros, monkeypatch):
    made = []

    def make(**params):
        made.append(Rig(monkeypatch, **params))
        return made[-1]
    yield make
    for r in made:
        r.close()


@pytest.mark.parametrize('missing, text', [
    ('reset_speed_scaling', 'reset_speed_scaling is required'),
    ('reset_posture_names', 'reset_posture_names is required'),
    ('reset_postures.ready', 'reset_postures.ready needs 6'),
])
def test_the_node_refuses_to_start_without_a_required_value(ros, missing, text):
    values = {k: v for k, v in PARAMS.items() if k != missing}
    with pytest.raises(StartupRefused, match=text):
        ResetNode(node_name='reset_refused', parameter_overrides=[
            Parameter(k, value=v) for k, v in values.items()])


def test_a_posture_not_in_the_list_refuses_to_start(ros):
    with pytest.raises(StartupRefused, match='not in reset_posture_names'):
        ResetNode(node_name='reset_refused', parameter_overrides=[
            Parameter(k, value=v) for k, v in {**PARAMS, 'reset_posture': 'home'}.items()])


def test_enabled_defaults_to_false_and_then_every_request_is_refused(rig):
    r = rig(enabled=False)
    r.hold(1.2, 0.0)
    r.hold(0.6)
    assert r.state().state == ResetState.STATE_REFUSED
    assert r.state().reason == ResetState.REASON_OFF and not r.action.goals
    assert not r.state().enabled


def test_a_robot_that_is_not_a_simulator_gets_no_goal(rig):
    r = rig()
    r.node._on_description(String(data=REAL_DESCRIPTION))
    r.hold(1.2, 0.0)
    r.hold(0.6)
    assert r.state().reason == ResetState.REASON_NOT_SIMULATION and not r.action.goals


def test_the_goal_is_announced_before_it_is_sent_and_is_a_reduced_speed_plan(rig):
    r = rig()
    goal_id, goal, _, _ = r.goal()
    assert any(goal_id)
    sent = r.log.index(('send', goal_id))
    assert ('beat', goal_id) in r.log[:sent]        # the watchdog knew it first
    assert r.states[-1].goal_out and bytes(r.states[-1].goal_id) == goal_id
    request = goal.request
    assert request.group_name == 'xarm6' and not goal.planning_options.plan_only
    assert request.max_velocity_scaling_factor == pytest.approx(0.2)
    assert request.max_acceleration_scaling_factor == pytest.approx(0.2)
    constraints = request.goal_constraints[0].joint_constraints
    assert [c.joint_name for c in constraints] == list(ARM_JOINTS)    # no gripper joint
    assert [c.position for c in constraints] == pytest.approx(READY)


def test_the_node_publishes_and_asks_nothing_else(rig):
    """It never arms, never asks the mux, never commands a controller (INV-3, INV-4)."""
    r = rig()
    topics = {pub.topic_name for pub in r.node.publishers}
    assert topics - {'/rosout'} == {'/teleop/reset/state', '/teleop/reset/heartbeat',
                                    '/parameter_events'}
    services = {client.srv_name for client in r.node.clients}
    assert services == {'/move_action/_action/cancel_goal'}       # the action's are its own


def test_accepted_then_succeeded_is_start_then_arrived(rig):
    r = rig()
    goal_id, _, handle = r.moving()
    assert r.state().last_event == ResetState.EVENT_START
    result = type('R', (), {})()
    result.result = type('X', (), {})()
    result.result.error_code = type('E', (), {'val': 1})()
    handle.result_future.answer(result)
    r.step(1.0)
    assert r.state().state == ResetState.STATE_ARRIVED
    assert r.state().last_event == ResetState.EVENT_ARRIVED and not r.state().goal_out


def cancels(r, goal_id):
    ids = [g for g, _ in r.cancel.requests]
    assert ids and all(g == goal_id for g in ids[1:]), ids      # the first is the harmless one
    return ids[1:]


def test_release_cancels_by_the_goals_own_id(rig):
    r = rig()
    goal_id, _, _ = r.moving()
    r.step(0.0)
    assert r.state().state == ResetState.STATE_STOPPING
    assert r.state().reason == ResetState.REASON_RELEASED
    assert cancels(r, goal_id) and all(any(g) for g in cancels(r, goal_id))


def test_a_lost_hand_stops_and_a_nan_stick_stops(rig):
    for kw in ({'trigger': math.nan, 'stick': 0.0}, {'stick': math.nan}):
        r = rig()
        goal_id, _, _ = r.moving()
        r.step(**kw)
        assert r.state().reason == ResetState.REASON_CONTROLLER_MISSING, kw
        assert cancels(r, goal_id)


@pytest.mark.parametrize('frame_kw', [
    {'pose': False},                    # the status says no controller pose
    {'send_pose': False},               # announced, never arrived (the frame waits, then goes)
    {'emulated': True},
    {'head': False},
    {'reset_count': 4},
])
def test_pose_only_conditions_let_the_motion_continue(rig, frame_kw):
    r = rig()
    r.moving()
    r.hold(1.0, **frame_kw)
    assert r.state().state == ResetState.STATE_MOVING, r.state().reason_text
    assert len(r.cancel.requests) == 1                  # only the harmless one


def test_a_session_change_stops_the_motion(rig):
    r = rig()
    goal_id, _, _ = r.moving()
    r.session = 8
    r.step(1.0)
    assert r.state().reason == ResetState.REASON_SESSION_CHANGED and cancels(r, goal_id)


def test_a_sequence_going_backwards_stops_the_motion(rig):
    r = rig()
    goal_id, _, _ = r.moving()
    r.seq -= 5
    r.step(1.0)
    assert r.state().reason == ResetState.REASON_SEQ_BACKWARDS and cancels(r, goal_id)


def test_a_watchdog_that_does_not_hear_the_heartbeat_counts_as_absent(rig):
    r = rig()
    r.echo = False
    r.hold(1.2, 0.0)
    r.hold(0.6)
    assert r.state().reason == ResetState.REASON_WATCHDOG_ABSENT and not r.action.goals
    assert not r.state().watchdog_alive


def test_an_echo_of_a_later_heartbeat_than_this_node_sent_is_not_alive(rig):
    """After a restart of the reset node, an echo of the old node's seq proves nothing."""
    r = rig()
    r.watchdog = False
    r.hold(1.2, 0.0)
    r.node._on_watchdog(ResetWatchdog(seq=1, heartbeat_seq=r.node.beat_seq + 1000))
    assert not r.node.watchdog_alive(r.clock.t)
    r.node._on_watchdog(ResetWatchdog(seq=2, heartbeat_seq=r.node.beat_seq))
    assert r.node.watchdog_alive(r.clock.t)


def test_a_goal_accepted_after_it_was_given_up_is_cancelled_and_announced(rig):
    """The late answer comes before its status: the heartbeat names it, cancels follow."""
    r = rig()
    goal_id, _, future, _ = r.goal()
    r.cancel.answering = True                       # answers "unknown goal" (not listed)
    r.hold(2.1)                                     # no answer: stopped as unknown
    r.hold(4.0, 0.0)                                # the server never knew it: given up
    assert r.state().state == ResetState.STATE_STOPPED and not r.state().goal_out
    before = len(r.cancel.requests)
    future.answer(Handle(True))                     # it was accepted after all
    r.step(0.0)
    assert r.beats[-1].goal_out and bytes(r.beats[-1].goal_id) == goal_id
    assert r.state().goal_out and r.state().last_event == ResetState.EVENT_STOPPED
    assert [g for g, _ in r.cancel.requests[before:]] == [goal_id]


def test_stale_input_stops_the_motion(rig):
    r = rig()
    goal_id, _, _ = r.moving()
    r.hold(0.14, frame=False)
    assert r.state().reason == ResetState.REASON_INPUT_STALE and cancels(r, goal_id)


def test_teleop_armed_during_the_motion_stops_it(rig):
    r = rig()
    goal_id, _, _ = r.moving()
    r.armed, r.owner = True, OWNER_TELEOP
    r.step(1.0)
    assert r.state().reason == ResetState.REASON_TELEOP_ARMED and cancels(r, goal_id)


def test_no_watchdog_no_goal_and_a_watchdog_lost_mid_motion_stops_it(rig):
    r = rig()
    r.watchdog = False
    r.hold(1.2, 0.0)
    r.hold(0.6)
    assert r.state().reason == ResetState.REASON_WATCHDOG_ABSENT and not r.action.goals
    r = rig()
    goal_id, _, _ = r.moving()
    r.watchdog = False
    r.hold(0.6)
    assert r.state().reason == ResetState.REASON_WATCHDOG_LOST and cancels(r, goal_id)


def test_a_lost_goal_answer_is_unknown_and_lost_cancel_answers_are_asked_again(rig):
    r = rig()
    goal_id, _, _, _ = r.goal()
    r.cancel.answering = False                      # every cancel answer is lost from now
    r.hold(2.1)
    assert r.state().state == ResetState.STATE_STOPPING
    assert r.state().reason == ResetState.REASON_NO_ANSWER
    r.hold(2.0, 0.0)
    asked = cancels(r, goal_id)
    assert 4 <= len(asked) <= 6, len(asked)         # every 0.5 s while unanswered
    assert r.state().goal_out                       # not given up: not seen ended
    r.cancel.answering = True
    r.node._on_goal_status(GoalStatusArray(status_list=[GoalStatus(
        goal_info=GoalInfo(goal_id=_uuid(goal_id)), status=GoalStatus.STATUS_CANCELED)]))
    r.hold(1.5, 0.0)
    assert r.state().state == ResetState.STATE_STOPPED and not r.state().goal_out


def test_while_move_group_is_not_ready_requests_are_refused(rig):
    r = rig()
    r.cancel.ready = False
    r.hold(1.2, 0.0)
    r.hold(0.6)
    assert r.state().reason == ResetState.REASON_MOVEIT_NOT_READY and not r.action.goals


# ---- the watchdog node ----

class Dog:

    def __init__(self, monkeypatch):
        self.clock = FakeTime()
        monkeypatch.setattr(reset_watchdog_node, 'time', self.clock)
        self.node = ResetWatchdogNode(node_name='watchdog_under_test')
        self.cancel = FakeCancel()
        self.node.cancel_client = self.cancel
        self.alive = []
        self.node.alive_pub.publish = self.alive.append
        self.seq = 0

    def step(self, beat=True, goal_id=bytes(range(1, 17)), goal_out=True):
        from ranger_xarm_teleop_interfaces.msg import ResetHeartbeat
        self.clock.t += 0.01
        if beat:
            self.seq += 1
            self.node._on_beat(ResetHeartbeat(seq=self.seq, goal_out=goal_out,
                                              goal_id=list(goal_id)))
        self.node._on_joints(JointState(name=list(ARM_JOINTS), position=[0.1] * 6))
        self.node._cycle()

    def run(self, seconds, **kw):
        for _ in range(int(round(seconds / 0.01))):
            self.step(**kw)


@pytest.fixture
def dog(ros, monkeypatch):
    made = []

    def make():
        made.append(Dog(monkeypatch))
        return made[-1]
    yield make
    for d in made:
        d.node.destroy_node()


def test_the_watchdog_says_it_is_alive_only_after_its_cancel_path_answered(dog):
    d = dog()
    d.cancel.answering = False
    d.run(0.5, goal_out=False)
    d.node._publish_alive()
    assert not d.alive and not d.node.alive
    d.cancel.answering = True
    d.run(1.1, goal_out=False)
    d.node._publish_alive()
    assert d.node.alive and d.alive and not d.alive[-1].stopping


def test_the_watchdog_cancels_the_announced_goal_when_the_heartbeat_stops(dog):
    d = dog()
    goal = bytes(range(1, 17))
    d.run(1.2)
    harmless = len(d.cancel.requests)
    d.run(0.15, beat=False)
    assert len(d.cancel.requests) == harmless       # 150 ms: not yet
    d.run(0.1, beat=False)
    asked = [g for g, _ in d.cancel.requests[harmless:]]
    assert asked and all(g == goal for g in asked)
    d.node._publish_alive()
    assert d.alive[-1].stopping and bytes(d.alive[-1].goal_id) == goal


def test_the_watchdog_leaves_a_silent_node_without_a_goal_alone(dog):
    d = dog()
    d.run(1.2, goal_out=False)
    harmless = len(d.cancel.requests)
    d.run(1.0, beat=False)
    assert len(d.cancel.requests) == harmless
