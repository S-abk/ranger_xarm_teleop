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
"""G-16 hold-to-run reset (reset.py): the control, the refusals, the stops, the events."""

from dataclasses import replace
import math
import os
import re
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from ranger_xarm_teleop_servo import reset as rs
from ranger_xarm_teleop_servo.session import ARM_JOINTS, OWNER_MOVEIT, OWNER_TELEOP

TARGET = (0.0, -0.6, -0.7, 0.0, 1.3, 0.0)
START = (0.4, -0.2, -0.3, 0.1, 1.0, 0.2)
GOAL_IDS = [bytes([k] * 16) for k in range(1, 50)]
MSG = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ranger_xarm_teleop_interfaces',
                   'msg', 'ResetState.msg')
IDLE_EPISODE = 1


def joy(stick=0.0, trigger=0.0, axes=8):
    a = [0.0] * axes
    if axes > rs.TRIGGER_AXIS:
        a[rs.TRIGGER_AXIS] = trigger
    if axes > rs.STICK_AXIS:
        a[rs.STICK_AXIS] = stick
    return NS(axes=a, buttons=[0] * 10)


def status(seq, session=1, reset_count=0, pose=True, emulated=False, head=True,
           head_emulated=False):
    return NS(session_id=session, seq=seq, xr_time_ms=1000.0 + seq * 11.0,
              reset_count=reset_count, right=NS(pose_present=pose, emulated_position=emulated),
              viewer_present=head, viewer_emulated_position=head_emulated)


class Rig:
    """A Reset with a fake clock, frames at about 90 Hz and cycles at 50 Hz."""

    def __init__(self, **params):
        ids = iter(GOAL_IDS)
        self.r = rs.Reset(rs.Params(speed_scaling=0.2, **params), 'ready', TARGET, ARM_JOINTS,
                          lambda: next(ids))
        self.now = 100.0
        self.seq = 0
        self.session = 1
        self.q = list(START)
        self.world = rs.World(now=self.now, since_start=60.0, enabled=True, simulator=True,
                              owner=OWNER_MOVEIT, armed=False, teleop_running=True,
                              episode=IDLE_EPISODE, episode_running=True, watchdog_alive=True,
                              moveit_ready=True, joints=tuple(self.q))
        self.sent = []

    def joints(self):
        self.r.on_joints(self.now, dict(zip(ARM_JOINTS, self.q)))

    def step(self, stick=0.0, frame=True, trigger=0.0, st=None, joy_msg='default', **world):
        """One 20 ms cycle: maybe a frame, a joint state, then the cycle."""
        self.now += 0.02
        if frame:
            self.seq += 1
            message = joy(stick, trigger) if joy_msg == 'default' else joy_msg
            self.r.on_sample(rs.sample_of(st or status(self.seq, self.session), message,
                                          self.now))
        self.joints()
        if world:
            self.world = replace(self.world, **world)
        self.world = replace(self.world, now=self.now, joints=tuple(self.q))
        goal = self.r.tick(self.now, self.world, 1.7e9 + self.now)
        if goal is not None:
            self.sent.append(goal)
        return goal

    def hold(self, seconds, stick=1.0, **kw):
        goal = None
        for _ in range(int(round(seconds / 0.02))):
            goal = self.step(stick, **kw) or goal
        return goal

    def press_to_goal(self):
        self.hold(0.1, 0.0)                         # released first
        goal = self.hold(0.6)
        assert goal is not None and self.r.state == rs.PLANNING, self.r.reason_text()
        return goal

    def moving(self):
        goal = self.press_to_goal()
        self.r.on_goal_answer(self.now, goal, True)
        self.r.on_status(self.now, goal, 2)
        self.r.on_feedback(goal, 'MONITOR')
        self.q[0] -= 0.05                            # the arm moves
        self.step(1.0)
        assert self.r.state == rs.MOVING
        return goal


def test_constants_match_the_message():
    text = open(MSG, encoding='utf-8').read()
    for prefix, names in (('STATE', rs.STATE_NAMES),):
        for value, name in enumerate(names):
            assert re.search(rf'uint8 {prefix}_{name} = {value}\b', text), name
    for name, value in vars(rs).items():
        if isinstance(value, int) and name in re.findall(r'uint8 REASON_(\w+) = ', text):
            assert re.search(rf'uint8 REASON_{name} = {value}\b', text), name
    reasons = dict(re.findall(r'uint8 REASON_(\w+) = (\d+)', text))
    assert set(reasons) - {'NONE'} <= set(vars(rs)), set(reasons) - set(vars(rs))
    assert all(getattr(rs, name) == int(value) for name, value in reasons.items())
    assert {int(v) for v in reasons.values()} == set(rs.REASON_TEXT)
    events = dict(re.findall(r'uint8 EVENT_(\w+) = (\d+)', text))
    assert all(getattr(rs, 'EVENT_' + name) == int(value) for name, value in events.items())


@pytest.mark.parametrize('message, missing', [
    (None, True),
    (joy(axes=rs.STICK_AXIS), True),                        # too few axes
    (joy(stick=0.0, trigger=math.nan), True),               # the bridge's "no controller"
    (joy(stick=math.nan, trigger=0.0), True),
    (joy(stick=math.inf, trigger=0.0), True),
    (joy(stick=0.9, trigger=0.0), False),
    (joy(stick=0.0, trigger=1.0), False),
])
def test_the_no_controller_mark_is_read_in_one_place(message, missing):
    assert rs.controller_missing(message) is missing
    sample = rs.sample_of(status(3), message, 1.0)
    assert (sample.stick is None) is missing


def test_params_need_a_speed_scaling_and_sane_thresholds():
    with pytest.raises(TypeError):
        rs.Params()                                 # pylint: disable=no-value-for-parameter
    for bad in ({'speed_scaling': 0.0}, {'speed_scaling': 1.5},
                {'speed_scaling': math.nan}, {'speed_scaling': 0.2, 'hold_threshold': 0.9},
                {'speed_scaling': 0.2, 'dwell_s': -1.0}):
        with pytest.raises(ValueError):
            rs.Params(**bad).check()


def test_a_press_held_through_the_dwell_sends_one_goal_and_announces_it():
    rig = Rig()
    rig.hold(0.1, 0.0)
    assert rig.hold(0.46) is None and rig.r.state == rs.WAITING     # dwell 0.5 s
    goal = rig.hold(0.08)
    assert goal == GOAL_IDS[0] and rig.r.goal_out and rig.r.goal_id == goal
    assert rig.r.event_count == 0                   # the start is felt when the goal is accepted
    rig.r.on_goal_answer(rig.now, goal, True)
    assert rig.r.last_event == rs.EVENT_START and rig.r.event_count == 1
    assert rig.hold(1.0) is None and rig.sent == [goal]


def test_a_stick_already_held_at_start_must_be_released_first():
    rig = Rig()
    assert rig.hold(2.0) is None and rig.r.state == rs.IDLE
    rig.hold(0.1, 0.3)                              # below the hold threshold: released
    assert rig.hold(0.6) is not None


def test_a_brush_shorter_than_the_dwell_does_nothing_even_while_armed():
    rig = Rig()
    rig.hold(0.1, 0.0)
    rig.hold(0.3, armed=True, owner=OWNER_TELEOP)
    rig.hold(0.3, 0.0)
    assert rig.r.state == rs.IDLE and rig.r.event_count == 0 and not rig.sent


def test_a_half_pull_neither_starts_nor_holds():
    rig = Rig()
    rig.hold(0.1, 0.0)
    assert rig.hold(2.0, 0.7) is None and rig.r.state == rs.IDLE   # below the press threshold


REFUSALS = [
    (rs.OFF, {'enabled': False}),
    (rs.NOT_SIMULATION, {'simulator': False}),
    (rs.ARMED, {'armed': True, 'owner': OWNER_TELEOP}),
    (rs.NOT_MOVEIT_OWNER, {'owner': OWNER_TELEOP}),
    (rs.NOT_MOVEIT_OWNER, {'owner': None}),
    (rs.ARMED_UNKNOWN, {'armed': None}),
    (rs.ARMED_UNKNOWN, {'armed': None, 'teleop_running': False, 'since_start': 2.0}),
    (rs.RECORDING, {'episode': rs.EPISODE_RECORDING}),
    (rs.REVIEW, {'episode': rs.EPISODE_REVIEW}),
    (rs.EPISODE_UNKNOWN, {'episode': None}),
    (rs.WATCHDOG_ABSENT, {'watchdog_alive': False}),
    (rs.BUSY, {'watchdog_stopping': True}),
    (rs.MOVEIT_NOT_READY, {'moveit_ready': False}),
]


@pytest.mark.parametrize('reason, world', REFUSALS)
def test_every_refusal_has_its_reason_and_sends_nothing(reason, world):
    rig = Rig()
    rig.hold(0.1, 0.0)
    rig.hold(0.6, **world)
    assert rig.r.state == rs.REFUSED and rig.r.reason == reason, rig.r.reason_text()
    assert rig.r.last_event == rs.EVENT_REFUSED and rig.r.event_count == 1
    assert not rig.sent and not rig.r.goal_out and rig.r.cancel_due(rig.now) is None
    assert rs.REASON_TEXT[reason]


def test_unknown_joints_refuse():
    rig = Rig()
    rig.hold(0.1, 0.0)
    for _ in range(30):
        rig.now += 0.02
        rig.seq += 1
        rig.r.on_sample(rs.sample_of(status(rig.seq), joy(1.0), rig.now))
        rig.r.tick(rig.now, replace(rig.world, now=rig.now, joints=None), 0.0)
    assert rig.r.reason == rs.JOINTS_UNKNOWN and not rig.sent


def test_without_a_teleop_or_episode_node_after_the_discovery_time_a_reset_may_run():
    rig = Rig()
    rig.hold(0.1, 0.0)
    goal = rig.hold(0.6, armed=None, teleop_running=False, episode=None, episode_running=False)
    assert goal is not None


def test_a_stick_held_through_a_disarm_does_not_start_the_motion():
    rig = Rig()
    rig.hold(0.1, 0.0)
    rig.hold(0.6, armed=True, owner=OWNER_TELEOP)
    assert rig.r.reason == rs.ARMED
    assert rig.hold(3.0, armed=False, owner=OWNER_MOVEIT) is None      # disarmed, still held
    assert rig.r.state == rs.REFUSED and rig.r.event_count == 1        # no new event either
    rig.hold(0.1, 0.0)
    assert rig.hold(0.6) is not None                                   # a fresh press does


def test_a_condition_false_at_any_time_during_the_dwell_refuses():
    rig = Rig()
    rig.hold(0.1, 0.0)
    rig.hold(0.2)
    rig.hold(0.1, episode=rs.EPISODE_RECORDING)
    rig.hold(0.3, episode=IDLE_EPISODE)
    assert rig.r.state == rs.REFUSED and rig.r.reason == rs.RECORDING and not rig.sent


def test_at_the_posture_already_nothing_is_sent():
    rig = Rig()
    rig.q = list(TARGET)
    rig.hold(0.1, 0.0)
    rig.hold(0.6)
    assert rig.r.state == rs.ARRIVED and rig.r.reason == rs.ALREADY_THERE and not rig.sent
    assert rig.r.last_event == rs.EVENT_ARRIVED


STOPS = [
    (rs.TELEOP_ARMED, {'armed': True, 'owner': OWNER_TELEOP}),
    (rs.OWNER_CHANGED, {'owner': OWNER_TELEOP}),
    (rs.STATE_UNKNOWN, {'owner': None}),
    (rs.STATE_UNKNOWN, {'armed': None}),
    (rs.EPISODE_STARTED, {'episode': rs.EPISODE_RECORDING}),
    (rs.EPISODE_STARTED, {'episode': rs.EPISODE_REVIEW}),
    (rs.STATE_UNKNOWN, {'episode': None}),
    (rs.WATCHDOG_LOST, {'watchdog_alive': False}),
]


def check_stopping(rig, goal, reason):
    assert rig.r.state == rs.STOPPING and rig.r.reason == reason, rig.r.reason_text()
    assert rig.r.cancel_due(rig.now) == goal and any(goal)
    event = rs.EVENT_RELEASED if reason == rs.RELEASED else rs.EVENT_STOPPED
    assert rig.r.last_event == event and rig.r.last_event_reason == reason


@pytest.mark.parametrize('reason, world', STOPS)
def test_every_stop_in_the_world_cancels_the_goal(reason, world):
    rig = Rig()
    goal = rig.moving()
    rig.step(1.0, **world)
    check_stopping(rig, goal, reason)


def test_release_stops_the_motion():
    rig = Rig()
    goal = rig.moving()
    rig.step(0.4)                                   # below the hold threshold
    check_stopping(rig, goal, rs.RELEASED)


def test_stale_input_stops_the_motion_within_the_input_timeout():
    rig = Rig()
    goal = rig.moving()
    for _ in range(4):
        rig.step(1.0, frame=False)
    assert rig.r.state == rs.MOVING                 # 80 ms: within the 100 ms
    rig.step(1.0, frame=False)                      # 100 ms: the boundary
    rig.step(1.0, frame=False)                      # 120 ms: stale
    check_stopping(rig, goal, rs.INPUT_STALE)


def test_a_missing_controller_stops_the_motion():
    rig = Rig()
    goal = rig.moving()
    rig.step(0.0, trigger=math.nan)                 # what the bridge sends for a lost hand
    check_stopping(rig, goal, rs.CONTROLLER_MISSING)


def test_a_nan_stick_stops_the_motion():
    rig = Rig()
    goal = rig.moving()
    rig.step(math.nan)
    check_stopping(rig, goal, rs.CONTROLLER_MISSING)


def test_a_frame_without_joy_stops_the_motion():
    rig = Rig()
    goal = rig.moving()
    rig.step(joy_msg=None)
    check_stopping(rig, goal, rs.CONTROLLER_MISSING)


def test_a_session_change_stops_the_motion():
    rig = Rig()
    goal = rig.moving()
    rig.session = 2
    rig.step(1.0)
    check_stopping(rig, goal, rs.SESSION_CHANGED)


def test_a_sequence_going_backwards_stops_the_motion():
    rig = Rig()
    goal = rig.moving()
    rig.step(1.0, st=status(rig.seq - 5))
    check_stopping(rig, goal, rs.SEQ_BACKWARDS)


@pytest.mark.parametrize('pose_only', [
    {'pose': False}, {'emulated': True}, {'head': False}, {'head_emulated': True},
    {'reset_count': 3},
])
def test_pose_only_conditions_do_not_stop_the_motion(pose_only):
    """G-16: the controller out of view, its pose or the headset's, a frame reset."""
    rig = Rig()
    rig.moving()
    for _ in range(50):                             # one second of them
        rig.step(1.0, st=status(rig.seq + 1, **pose_only))
    assert rig.r.state == rs.MOVING and rig.r.event_count == 1     # only the start


def test_the_watchdog_cancelling_this_goal_is_a_late_heartbeat():
    rig = Rig()
    goal = rig.moving()
    rig.step(1.0, watchdog_stopping=True, watchdog_goal=goal)
    check_stopping(rig, goal, rs.HEARTBEAT_LATE)


@pytest.mark.parametrize('first', ['preempted', 'stale'])
def test_a_stop_the_watchdog_made_is_named_heartbeat_late(first):
    """Seen first as move_group's "preempted", or as stale input after a stall of this node."""
    rig = Rig()
    goal = rig.moving()
    if first == 'preempted':
        rig.r.on_result(rig.now, goal, rs.MOVEIT_PREEMPTED)
        assert rig.r.reason == rs.CANCELLED_ELSEWHERE
    else:
        rig.hold(0.14, frame=False)
        assert rig.r.reason == rs.INPUT_STALE
    count = rig.r.event_count
    rig.step(0.0, watchdog_goal=goal)               # the watchdog names the goal it cancelled
    assert rig.r.state == rs.STOPPING and rig.r.reason == rs.HEARTBEAT_LATE
    assert rig.r.last_event_reason == rs.HEARTBEAT_LATE and rig.r.event_count == count
    finish_stop(rig, goal)
    rig.hold(1.3, 0.0)
    assert rig.r.state == rs.STOPPED and rig.r.reason == rs.HEARTBEAT_LATE


def test_another_goal_the_watchdog_cancelled_does_not_rename_this_stop():
    rig = Rig()
    rig.moving()
    rig.step(0.0, watchdog_goal=bytes([9] * 16))
    assert rig.r.reason == rs.RELEASED


def test_a_goal_without_an_answer_is_stopped_as_unknown():
    rig = Rig()
    goal = rig.press_to_goal()
    rig.hold(1.9)
    assert rig.r.state == rs.PLANNING
    rig.hold(0.12)
    check_stopping(rig, goal, rs.NO_ANSWER)


def test_release_during_planning_cancels_the_goal():
    rig = Rig()
    goal = rig.press_to_goal()
    rig.r.on_goal_answer(rig.now, goal, True)
    rig.step(0.0)
    check_stopping(rig, goal, rs.RELEASED)


def finish_stop(rig, goal, statuses=(5,)):
    """Answer the cancel, list the goal ended, let the joints rest."""
    if rig.r.cancel_due(rig.now) is not None:
        rig.r.cancel_sent(rig.now, goal)
        rig.r.cancel_answered(rig.now, goal, 0, 1)
    for st in statuses:
        rig.r.on_status(rig.now, goal, st)
    assert rig.r.cancel_due(rig.now + 1.0) is None      # ended: no more requests


def test_a_stop_ends_once_the_goal_has_ended_and_the_joints_rest_and_is_measured():
    rig = Rig()
    goal = rig.moving()
    rig.step(0.0)
    rig.r.cancel_sent(rig.now, goal)
    rig.r.cancel_timed_out(goal)                    # the first cancel answer is lost
    assert rig.r.cancel_due(rig.now + 0.6) == goal  # asked again
    for k in range(5):                              # the arm still coasts a little
        rig.q[0] -= 0.002
        rig.step(0.0)
    finish_stop(rig, goal)
    rig.hold(0.5, 0.0)
    assert rig.r.state == rs.STOPPING               # not yet seen at rest for 1 s
    rig.hold(0.8, 0.0)
    assert rig.r.state == rs.STOPPED and rig.r.reason == rs.RELEASED and not rig.r.goal_out
    assert rig.r.stop_distance == pytest.approx(0.010, abs=1e-6)
    assert rig.r.stop_time == pytest.approx(0.1,
                                            abs=0.03)
    assert any('stopped (the control was released)' in line for line in rig.r.take_log())


def test_a_stick_held_through_a_fault_never_resumes():
    rig = Rig()
    goal = rig.moving()
    rig.step(1.0, frame=False)
    rig.hold(0.1, frame=False)
    assert rig.r.reason == rs.INPUT_STALE
    finish_stop(rig, goal)
    assert rig.hold(3.0) is None                    # input back, stick still held: nothing
    assert rig.r.state == rs.STOPPED
    rig.hold(0.1, 0.0)
    assert rig.hold(0.6) == GOAL_IDS[1]             # a fresh press: a new goal, a new id


def test_a_press_while_stopping_is_ignored():
    rig = Rig()
    goal = rig.moving()
    rig.step(0.0)
    rig.hold(0.6)
    assert rig.r.state == rs.STOPPING and rig.sent == [goal]


def test_success_arrives():
    rig = Rig()
    goal = rig.moving()
    rig.r.on_result(rig.now, goal, rs.MOVEIT_SUCCESS)
    assert rig.r.state == rs.ARRIVED and rig.r.reason == rs.REACHED and not rig.r.goal_out
    assert rig.r.last_event == rs.EVENT_ARRIVED
    assert rig.hold(1.0) is None                    # held on: nothing more


@pytest.mark.parametrize('code, reason', [(-1, rs.PLAN_FAILED), (-31, rs.PLAN_FAILED),
                                          (-10, rs.START_IN_COLLISION)])
def test_a_plan_that_fails_moves_nothing_and_says_so(code, reason):
    rig = Rig()
    goal = rig.press_to_goal()
    rig.r.on_goal_answer(rig.now, goal, True)
    rig.hold(1.2)                                   # joints watched from before the goal
    rig.r.on_result(rig.now, goal, code)
    assert rig.r.state == rs.REFUSED and rig.r.reason == reason, rig.r.reason_text()
    assert 'the joints did not move' in rig.r.detail
    assert rig.r.last_event == rs.EVENT_REFUSED
    if reason == rs.START_IN_COLLISION:
        assert 'retreat_arm.py' in rs.REASON_TEXT[reason]


def test_a_planning_error_after_motion_is_a_stop_not_a_claim_that_nothing_moved():
    rig = Rig()
    goal = rig.moving()
    rig.r.on_result(rig.now, goal, -1)
    check_stopping(rig, goal, rs.MOVEIT_FAILED)


def test_a_goal_preempted_by_someone_else_is_stopped_and_named():
    rig = Rig()
    goal = rig.moving()
    rig.r.on_result(rig.now, goal, rs.MOVEIT_PREEMPTED)
    check_stopping(rig, goal, rs.CANCELLED_ELSEWHERE)


def test_a_rejected_goal_is_refused():
    rig = Rig()
    goal = rig.press_to_goal()
    rig.r.on_goal_answer(rig.now, goal, False)
    assert rig.r.state == rs.REFUSED and rig.r.reason == rs.MOVEIT_NOT_READY
    assert not rig.r.goal_out


def give_up(rig):
    """Send a goal that is never answered and that the server never knows: it is given up."""
    goal = rig.press_to_goal()
    rig.hold(2.1)                                   # no answer: stopped as unknown
    for _ in range(200):                            # at most 4 s
        due = rig.r.cancel_due(rig.now)
        if due is None and rig.r.state != rs.STOPPING:
            break
        if due is not None:
            rig.r.cancel_sent(rig.now, due)
            rig.r.cancel_answered(rig.now, due, 2, 0)   # "unknown goal"
        rig.step(0.0)
    rig.hold(1.5, 0.0)
    assert rig.r.state == rs.STOPPED and not rig.r.goal_out
    assert rig.r.heartbeat_goal() == (False, bytes(16))
    return goal


def check_orphan(rig, goal):
    assert rig.r.cancel_due(rig.now) == goal
    assert rig.r.heartbeat_goal() == (True, goal)   # the watchdog covers it too
    assert rig.r.last_event == rs.EVENT_STOPPED and rig.r.last_event_reason == rs.NO_ANSWER


def test_a_given_up_goal_listed_running_later_is_cancelled():
    rig = Rig()
    goal = give_up(rig)
    rig.r.on_status(rig.now, goal, 2)               # it arrived after all, and runs
    check_orphan(rig, goal)


def test_a_given_up_goal_accepted_later_then_listed_is_cancelled():
    """The late answer first, then its status (the usual order): review finding."""
    rig = Rig()
    goal = give_up(rig)
    rig.r.on_goal_answer(rig.now, goal, True)
    check_orphan(rig, goal)
    rig.r.on_status(rig.now, goal, 2)
    rig.step(0.0)
    assert rig.r.cancel_due(rig.now) == goal or rig.r.orphans[goal].outstanding is not None


def test_a_given_up_goal_turning_up_during_a_newer_motion_stops_both():
    rig = Rig()
    old = give_up(rig)
    new = rig.moving()
    assert new != old
    rig.r.on_status(rig.now, old, 2)
    assert rig.r.state == rs.STOPPING and rig.r.reason == rs.NO_ANSWER
    first = rig.r.cancel_due(rig.now)
    rig.r.cancel_sent(rig.now, first)
    second = rig.r.cancel_due(rig.now)
    assert {first, second} == {old, new}            # each by its own id


def test_no_new_goal_while_a_given_up_goal_is_being_cancelled():
    rig = Rig()
    goal = give_up(rig)
    rig.r.on_status(rig.now, goal, 2)
    rig.hold(0.1, 0.0)
    rig.hold(0.6)
    assert rig.r.state == rs.REFUSED and rig.r.reason == rs.BUSY and len(rig.sent) == 1
    rig.r.cancel_sent(rig.now, goal)
    rig.r.cancel_answered(rig.now, goal, 0, 1)
    rig.r.on_status(rig.now, goal, 5)               # cancelled
    rig.hold(1.3, 0.0)                              # the joints seen at rest
    assert not rig.r.orphan_active() and goal not in rig.r.orphans
    assert rig.hold(0.6) is not None                # a press works again


def test_a_given_up_goal_turning_up_during_the_dwell_needs_a_fresh_press():
    """Review finding: the stick held through it must not start a goal afterwards."""
    rig = Rig()
    goal = give_up(rig)
    rig.hold(0.1, 0.0)
    rig.hold(0.2)                                   # in the dwell
    assert rig.r.state == rs.WAITING
    rig.r.on_goal_answer(rig.now, goal, True)       # the given-up goal turns up
    rig.r.cancel_sent(rig.now, goal)
    rig.r.cancel_answered(rig.now, goal, 0, 1)
    rig.r.on_status(rig.now, goal, 5)
    assert rig.hold(3.0) is None and len(rig.sent) == 1    # held on: nothing sent
    rig.hold(0.1, 0.0)
    assert rig.hold(0.6) is not None                # a fresh press does


def test_a_given_up_goal_seen_twice_is_one_stop_and_one_event():
    rig = Rig()
    goal = give_up(rig)
    count = rig.r.event_count
    rig.r.on_status(rig.now, goal, 2)               # listed first ...
    rig.r.on_goal_answer(rig.now, goal, True)       # ... then the late answer
    assert rig.r.event_count == count + 1
    assert rig.r.orphans[goal].statuses == [2]      # what was seen is kept


def test_a_given_up_goal_listed_ended_is_forgotten():
    rig = Rig()
    goal = give_up(rig)
    rig.r.on_status(rig.now, goal, 6)
    assert goal not in rig.r.orphans and rig.r.cancel_due(rig.now) is None


def test_a_rejected_goal_is_not_watched_afterwards():
    rig = Rig()
    goal = rig.press_to_goal()
    rig.r.on_goal_answer(rig.now, goal, False)
    assert goal not in rig.r.orphans


def test_events_are_counted_with_their_wall_time():
    rig = Rig()
    rig.hold(0.1, 0.0)
    rig.hold(0.6, enabled=False)
    assert rig.r.event_count == 1
    assert rig.r.last_event_wall == pytest.approx(1.7e9 + rig.now, abs=0.6)


def test_reset_has_no_ros_imports():
    pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    bridge = os.path.join(os.path.dirname(pkg), 'ranger_xarm_teleop_bridge')
    code = ('import sys\n'
            f'sys.path[:0] = [{pkg!r}, {bridge!r}]\n'
            'import ranger_xarm_teleop_servo.reset, ranger_xarm_teleop_servo.goal_stop\n'
            "bad = [m for m in sys.modules if m.split('.')[0] in ('rclpy', 'sensor_msgs', "
            "'std_msgs', 'action_msgs', 'moveit_msgs', 'ranger_xarm_teleop_interfaces')]\n"
            'assert not bad, bad\n')
    result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
