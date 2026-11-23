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
The node's side of the hardware rules: motion disabled has no command publisher, and G-17.

* With ``motion_enabled`` false the node creates no publisher on the target, the joint
  command or the gripper command; the preview topic exists then, and only then.
* G-17: armed teleop that has owned the arm and finds it under another owner, without having
  asked, disarms, forgets its own gripper command and never asks for the arm by itself.
"""

from dataclasses import replace
import os
import sys
import time

import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner
from ranger_xarm_teleop_servo import gripper as grip
from ranger_xarm_teleop_servo import params
from ranger_xarm_teleop_servo import teleop_node
from ranger_xarm_teleop_servo.teleop_node import (NoPublisher, PREVIEW_TOPIC, StartupRefused,
                                                  TARGET_TOPIC, TeleopNode)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_teleop_node import LIMITS, overrides, ros  # noqa: E402, I100

__all__ = ['ros']        # the module-scoped ROS fixture of the node tests

GRIPPER_LIMITS = {'gripper_open_position': 0.0, 'gripper_closed_position': 0.8,
                  'gripper_max_speed': 1.0}


def publishers(node, expect=None, timeout=3.0):
    """Return the node's publisher topics, waiting until ``expect`` (if given) shows."""
    end = time.monotonic() + timeout
    while True:
        topics = {name for name, _ in node.get_publisher_names_and_types_by_node(
            node.get_name(), node.get_namespace())}
        if expect is None or expect in topics or time.monotonic() > end:
            return topics
        time.sleep(0.05)


def unique(name):
    """Return a node name no other test process on the domain uses."""
    return f'{name}_{os.getpid()}_{time.monotonic_ns() % 1_000_000}'


@pytest.mark.parametrize('gripper', [False, True])
def test_motion_disabled_has_no_command_publisher(ros, gripper):
    values = {**LIMITS, 'motion_enabled': False}
    if gripper:
        values.update({'gripper_enabled': True, **GRIPPER_LIMITS})
    node = TeleopNode(node_name=unique('teleop_disabled_endpoints'),
                      parameter_overrides=overrides(values))
    try:
        topics = publishers(node, expect=PREVIEW_TOPIC)
        assert TARGET_TOPIC not in topics
        assert not any(t.endswith('/delta_joint_cmds') for t in topics), topics
        assert '/teleop/gripper/command' not in topics
        assert PREVIEW_TOPIC in topics
        assert isinstance(node.target_pub, NoPublisher)
        assert isinstance(node.servo_pub, NoPublisher)
        with pytest.raises(RuntimeError, match='motion is disabled'):
            node.servo_pub.publish(None)
        if gripper:
            assert isinstance(node.gripper_pub, NoPublisher)
    finally:
        node.close()


def test_motion_enabled_has_no_preview_topic(ros):
    node = TeleopNode(node_name=unique('teleop_enabled_endpoints'),
                      parameter_overrides=overrides({**LIMITS, 'motion_enabled': True}))
    try:
        topics = publishers(node, expect=TARGET_TOPIC)
        assert TARGET_TOPIC in topics and PREVIEW_TOPIC not in topics
        assert node.preview_pub is None
    finally:
        node.close()


class _Client:
    """Stands in for the mux client: records requests, never answers."""

    def __init__(self):
        self.requests = []

    def service_is_ready(self):
        return True

    def call_async(self, request):
        self.requests.append(request.owner)

        return _Future()


class _Future:
    """A request that is never answered."""

    def add_done_callback(self, callback):
        pass


def armed_node(ros):
    node = TeleopNode(node_name='teleop_g17',
                      parameter_overrides=overrides({**LIMITS, 'motion_enabled': True}))
    node.owner_client = _Client()
    node.state = replace(node.state, want_armed=True)
    return node


def report(node, owner, now):
    node.owner = owner
    node.owner_arrival = now
    node.manage_ownership(now, node.current_owner(now))


def test_g17_an_unrequested_loss_of_ownership_disarms_and_does_not_ask_again(ros):
    node = armed_node(ros)
    try:
        node.mono = lambda: 10.0
        report(node, ArmOwner.OWNER_MOVEIT, 10.0)       # armed: teleop asks for the arm
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP]
        node.owner_request_since = None
        report(node, ArmOwner.OWNER_TELEOP, 10.1)       # granted
        assert node.state.want_armed and node.owned_while_armed
        node.gripper_state = replace(grip.GripperState(), command=0.4)
        report(node, ArmOwner.OWNER_NONE, 10.2)          # the driver switched everything off
        assert not node.state.want_armed
        assert node.gripper_state.command is None
        for t in (10.3, 12.0, 20.0):                     # the arm comes back: no request
            report(node, ArmOwner.OWNER_MOVEIT, t)
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP]
        assert not node.state.want_armed
    finally:
        node.close()


@pytest.mark.parametrize('owner', [ArmOwner.OWNER_MOVEIT, ArmOwner.OWNER_NONE])
def test_g17_any_other_owner_counts(ros, owner):
    node = armed_node(ros)
    try:
        node.owner_request_since = None
        report(node, ArmOwner.OWNER_TELEOP, 5.0)
        report(node, owner, 5.1)
        assert not node.state.want_armed
    finally:
        node.close()


def test_g17_a_silent_mux_holds_but_does_not_disarm(ros):
    node = armed_node(ros)
    try:
        report(node, ArmOwner.OWNER_TELEOP, 5.0)
        node.manage_ownership(6.0, node.current_owner(6.0))   # report 1 s old: NONE, not fresh
        assert node.state.want_armed
        report(node, ArmOwner.OWNER_TELEOP, 6.1)
        assert node.state.want_armed
    finally:
        node.close()


def test_an_outside_activation_reported_as_no_owner_is_not_answered(ros):
    """The mux reports an outside activation as no owner: a disarmed teleop asks for nothing."""
    node = armed_node(ros)
    try:
        node.state = replace(node.state, want_armed=False)
        for t in (5.0, 6.0, 12.0):
            report(node, ArmOwner.OWNER_NONE, t)
        assert node.owner_client.requests == []
    finally:
        node.close()


def test_g17_a_loss_before_teleop_ever_owned_the_arm_is_not_a_loss(ros):
    node = armed_node(ros)
    try:
        report(node, ArmOwner.OWNER_MOVEIT, 5.0)         # still waiting for the switch
        assert node.state.want_armed
    finally:
        node.close()


def test_g17_disarming_on_purpose_is_not_a_loss(ros):
    node = armed_node(ros)
    try:
        report(node, ArmOwner.OWNER_MOVEIT, 4.9)         # teleop asks for the arm
        node.owner_request_since = None
        report(node, ArmOwner.OWNER_TELEOP, 5.0)
        node._disarm('test')
        node.owner_request_since = None
        report(node, ArmOwner.OWNER_TELEOP, 5.1)        # teleop gives the arm back itself
        assert node.owner_client.requests[-1] == ArmOwner.OWNER_MOVEIT
        node.owner_request_since = None
        node.state = replace(node.state, want_armed=True)   # armed again: asks normally
        report(node, ArmOwner.OWNER_MOVEIT, 7.0)
        assert node.state.want_armed
        assert node.owner_client.requests[-1] == ArmOwner.OWNER_TELEOP
    finally:
        node.close()


HARDWARE = {**LIMITS, 'tool_pose_timeout_s': 0.05, 'hardware_limits': True}


def test_hardware_limits_complete_start_the_node(ros):
    node = TeleopNode(node_name='teleop_hardware_complete',
                      parameter_overrides=overrides(HARDWARE))
    node.close()


@pytest.mark.parametrize('missing', params.HARDWARE_REQUIRED)
def test_hardware_limits_refuse_the_node_without_each_value(ros, missing):
    values = {k: v for k, v in HARDWARE.items() if k != missing}
    with pytest.raises(StartupRefused, match=missing):
        TeleopNode(node_name='teleop_hardware_missing', parameter_overrides=overrides(values))


def test_hardware_limits_refuse_simulation_only_values(ros):
    values = {**HARDWARE, 'limits_are_simulation_only': True}
    with pytest.raises(StartupRefused, match='never loads simulation-only'):
        TeleopNode(node_name='teleop_hardware_sim', parameter_overrides=overrides(values))


class _Done:
    """An answered ownership request."""

    def __init__(self, owner, success=True):
        from ranger_xarm_teleop_interfaces.srv import SetArmOwner
        self.response = SetArmOwner.Response(success=success, owner=owner,
                                             status=ArmOwner.STATUS_OK, detail='')

    def result(self):
        return self.response


def test_a_slow_switch_still_hands_the_arm_back_after_disarming(ros):
    """The mux's report from before the switch must not cost teleop its hand-back."""
    node = armed_node(ros)
    try:
        node.mono = lambda: 10.0
        report(node, ArmOwner.OWNER_MOVEIT, 10.0)          # armed: the request goes out
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP]
        for t in (10.02, 10.04, 10.1):                      # the switch takes a while
            report(node, ArmOwner.OWNER_MOVEIT, t)
        node._owner_answered(_Done(ArmOwner.OWNER_TELEOP), ArmOwner.OWNER_TELEOP,
                             node.owner_request_id)
        report(node, ArmOwner.OWNER_TELEOP, 10.2)
        node._disarm('test')
        node.owner_retry_after = 0.0
        report(node, ArmOwner.OWNER_TELEOP, 11.0)
        assert node.owner_client.requests[-1] == ArmOwner.OWNER_MOVEIT
    finally:
        node.close()


def test_with_no_owner_reported_an_arming_teleop_asks_the_mux(ros):
    """During an outside activation the mux reports no owner: teleop must ask, not command."""
    node = armed_node(ros)
    try:
        report(node, ArmOwner.OWNER_NONE, 5.0)
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP]
    finally:
        node.close()


# ---- an ownership request without an answer (contract, "Service and action calls") ----

class _Pending:
    """A request whose answer the test gives, or never gives (a lost reply)."""

    def __init__(self, owner):
        self.owner, self.callbacks, self.response = owner, [], None

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def answer(self, owner, success=True, detail=''):
        from ranger_xarm_teleop_interfaces.srv import SetArmOwner
        self.response = SetArmOwner.Response(
            success=success, owner=owner, detail=detail,
            status=ArmOwner.STATUS_OK if success else ArmOwner.STATUS_REFUSED_MOVING)
        for callback in self.callbacks:
            callback(self)

    def result(self):
        return self.response


class _AnsweredLater(_Client):
    """The mux client whose answers the test gives one by one."""

    def __init__(self):
        super().__init__()
        self.pending = []

    def call_async(self, request):
        self.requests.append(request.owner)
        self.pending.append(_Pending(request.owner))
        return self.pending[-1]


class _Log:
    """Collects the node's log lines."""

    def __init__(self):
        self.lines = []

    def _add(self, level):
        return lambda text, **kw: self.lines.append((level, text))

    def __getattr__(self, level):
        return self._add(level)


def asking_node(ros):
    node = armed_node(ros)
    node.owner_client = _AnsweredLater()
    node.log = _Log()
    node.get_logger = lambda: node.log
    return node


def at(node, t, owner=None):
    """Run the ownership step at ``t``; with ``owner``, a fresh mux report arrives first."""
    node.mono = lambda: t
    if owner is not None:
        node.owner, node.owner_arrival = owner, t
    node.manage_ownership(t, node.current_owner(t))


def no_answer_lines(node):
    return [text for _, text in node.log.lines
            if text.startswith('no answer from the command mux')]


def test_a_lost_first_reply_is_read_back_and_the_request_asked_again(ros):
    """No answer in time: the owner is read back, one line logged, and teleop asks again."""
    node = asking_node(ros)
    try:
        at(node, 10.0, ArmOwner.OWNER_MOVEIT)                    # armed: the request goes out
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP]
        for t in (10.5, 12.0, 14.9):                             # the reply is lost
            at(node, t, ArmOwner.OWNER_MOVEIT)
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP]
        assert node.mux_status == ArmOwner.STATUS_OK
        at(node, 10.0 + teleop_node.OWNER_REQUEST_TIMEOUT_S + 0.1, ArmOwner.OWNER_MOVEIT)
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP, ArmOwner.OWNER_TELEOP]
        lines = no_answer_lines(node)
        assert len(lines) == 1 and 'it reports MoveIt' in lines[0] and 'asked again' in lines[0]
        assert node.mux_status == ArmOwner.STATUS_NO_ANSWER     # the panel says it is repeated
        assert node.state.want_armed
        node.owner_client.pending[1].answer(ArmOwner.OWNER_TELEOP)    # the second is answered
        assert node.owner == ArmOwner.OWNER_TELEOP and node.mux_status == ArmOwner.STATUS_OK
        assert node.owner_request_since is None and node.state.want_armed
    finally:
        node.close()


def test_without_a_fresh_owner_report_an_unanswered_request_is_not_asked_again(ros):
    """Never repeated blindly: a silent mux (or one busy serving) gives no state to read back."""
    node = asking_node(ros)
    try:
        at(node, 10.0, ArmOwner.OWNER_MOVEIT)
        for t in (15.5, 16.0, 20.0):                             # no report since 10.0
            at(node, t)
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP]
        lines = no_answer_lines(node)
        assert len(lines) == 1 and 'nothing is asked again' in lines[0]
        assert node.mux_status == ArmOwner.STATUS_OK             # nothing is being repeated
        assert node.state.want_armed                             # held while the mux may serve
        at(node, 20.5, ArmOwner.OWNER_SWITCHING)                 # a switch in progress: wait
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP]
        at(node, 21.0, ArmOwner.OWNER_MOVEIT)                    # it reports: asked again
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP, ArmOwner.OWNER_TELEOP]
        assert node.mux_status == ArmOwner.STATUS_NO_ANSWER     # now it is being repeated
    finally:
        node.close()


def test_a_lost_reply_after_the_switch_is_settled_by_the_owner_report(ros):
    node = asking_node(ros)
    try:
        at(node, 10.0, ArmOwner.OWNER_MOVEIT)
        at(node, 12.0, ArmOwner.OWNER_TELEOP)                    # switched; the reply is lost
        at(node, 15.2, ArmOwner.OWNER_TELEOP)
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP]
        lines = no_answer_lines(node)
        assert len(lines) == 1 and 'answer lost' in lines[0]
        assert node.mux_status == ArmOwner.STATUS_OK and node.owner_request_since is None
        assert node.state.want_armed and node.owned_while_armed
    finally:
        node.close()


@pytest.mark.parametrize('granted', [True, False])
def test_a_late_answer_never_closes_the_newer_request(ros, granted):
    """A late answer is still an answer (a refusal disarms) but leaves the newer request open."""
    node = asking_node(ros)
    try:
        at(node, 10.0, ArmOwner.OWNER_MOVEIT)
        at(node, 15.1, ArmOwner.OWNER_MOVEIT)                    # asked again
        assert len(node.owner_client.pending) == 2
        node.owner_client.pending[0].answer(ArmOwner.OWNER_TELEOP if granted
                                            else ArmOwner.OWNER_MOVEIT, success=granted,
                                            detail='the arm is moving')
        assert node.owner_request_since == 15.1                  # the newer request stays open
        assert node.owner == ArmOwner.OWNER_MOVEIT               # the report is newer
        assert any(text.startswith('late answer') for _, text in node.log.lines)
        assert node.state.want_armed == granted
    finally:
        node.close()


def test_an_arm_request_unknown_too_long_disarms(ros):
    """The arm press does not wait on a silent mux, nor take the arm by itself much later."""
    node = asking_node(ros)
    try:
        at(node, 10.0, ArmOwner.OWNER_MOVEIT)
        at(node, 10.0 + teleop_node.OWNER_UNKNOWN_LIMIT_S - 0.1)
        assert node.state.want_armed
        at(node, 10.0 + teleop_node.OWNER_UNKNOWN_LIMIT_S + 0.1)
        assert not node.state.want_armed and node.owner_request_since is None
        at(node, 30.0, ArmOwner.OWNER_MOVEIT)                    # the mux is back: nothing asked
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP]
    finally:
        node.close()


def test_an_unknown_arm_request_disarms_when_the_mux_service_is_gone(ros):
    node = asking_node(ros)
    try:
        at(node, 10.0, ArmOwner.OWNER_MOVEIT)
        node.owner_client.service_is_ready = lambda: False
        at(node, 15.5)
        assert not node.state.want_armed and node.owner_request_since is None
    finally:
        node.close()


def test_a_new_arm_request_after_an_unrepeated_one_is_not_called_a_repeat(ros):
    """Status 6 only while a request is really being asked again; a new press starts clean."""
    node = asking_node(ros)
    try:
        at(node, 10.0, ArmOwner.OWNER_MOVEIT)
        at(node, 15.1, ArmOwner.OWNER_MOVEIT)                    # asked again: status 6
        assert node.mux_status == ArmOwner.STATUS_NO_ANSWER
        node._disarm('test')                                     # the operator gives up
        at(node, 20.2, ArmOwner.OWNER_MOVEIT)                    # settled, nothing asked
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP] * 2
        node.state = replace(node.state, want_armed=True)       # a fresh press
        at(node, 21.0, ArmOwner.OWNER_MOVEIT)
        assert node.owner_client.requests == [ArmOwner.OWNER_TELEOP] * 3
        assert node.mux_status == ArmOwner.STATUS_OK
        assert sum(text.startswith('asking the command mux again')
                   for _, text in node.log.lines) == 1                # only the real repeat
    finally:
        node.close()


def test_a_late_answer_after_g17_changes_nothing(ros):
    """G-17 forgets the open request; its late grant sets no owner and does not re-arm."""
    node = asking_node(ros)
    try:
        at(node, 10.0, ArmOwner.OWNER_MOVEIT)                    # the request goes out
        at(node, 10.2, ArmOwner.OWNER_TELEOP)                    # switched; no answer yet
        at(node, 12.0, ArmOwner.OWNER_NONE)                      # G-17: teleop disarms
        assert not node.state.want_armed and node.owner_request_since is None
        node.owner_client.pending[0].answer(ArmOwner.OWNER_TELEOP)   # the answer, late
        assert not node.state.want_armed and node.owner == ArmOwner.OWNER_NONE
        assert node.owner_request_since is None
        assert any(text.startswith('late answer') for _, text in node.log.lines)
    finally:
        node.close()
