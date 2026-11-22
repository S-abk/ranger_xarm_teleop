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
G-17 on the mux's side, in the node: an outside activation of the teleop controller.

The mux reports it (and no owner, so teleop cannot arm on it), waits until the arm is still
and no goal executes, then asks for a strict deactivation of that controller only; it never
asks for anything to be activated. A switch to teleop is refused meanwhile.
"""

import os
import sys

from controller_manager_msgs.srv import SwitchController
from ranger_xarm_teleop_interfaces.msg import ArmOwner
from ranger_xarm_teleop_servo import arm_mux_node
from ranger_xarm_teleop_servo.ownership import MuxStatus, Owner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_teleop_node import ros  # noqa: E402, I100

__all__ = ['ros']

OUTSIDE = {'xarm6_traj_controller': 'inactive', 'teleop_arm_controller': 'active',
           'joint_state_broadcaster': 'active'}


class _Answer:
    ok = True
    message = ''


def make_mux(monkeypatch, statuses):
    node = arm_mux_node.ArmMuxNode(node_name='arm_mux_outside_test')
    calls = []
    node._call = lambda client, request: calls.append(request) or _Answer()
    node.read_owner = lambda: Owner.NONE
    node.read_states = lambda: dict(OUTSIDE)
    seq = iter(statuses)
    monkeypatch.setattr(arm_mux_node, 'check_switch', lambda *args: next(seq))
    published = []
    node.owner_pub.publish = published.append
    return node, calls, published


def test_the_outside_teleop_controller_is_deactivated_only_once_still(ros, monkeypatch):
    node, calls, published = make_mux(monkeypatch, [MuxStatus.REFUSED_MOVING,
                                                    MuxStatus.REFUSED_GOAL_ACTIVE,
                                                    MuxStatus.OK])
    try:
        node.owner = Owner.TELEOP
        node.restore_order(dict(OUTSIDE))           # moving: reported, nothing asked
        node.restore_order(dict(OUTSIDE))           # a goal executing: nothing asked
        assert calls == []
        assert 'without a switch by the mux' in node.outside
        node.restore_order(dict(OUTSIDE))           # still, no goal: one strict deactivation
        assert len(calls) == 1
        request = calls[0]
        assert list(request.deactivate_controllers) == ['teleop_arm_controller']
        assert list(request.activate_controllers) == []
        assert request.strictness == SwitchController.Request.STRICT
    finally:
        node.destroy_node()


def test_while_outside_the_owner_is_reported_as_none_and_teleop_is_refused(ros, monkeypatch):
    node, calls, published = make_mux(monkeypatch, [MuxStatus.REFUSED_MOVING] * 10)
    try:
        node.read_owner = lambda: Owner.TELEOP
        node.owner_from = lambda states: Owner.TELEOP
        node._on_timer()
        assert published[-1].owner == ArmOwner.OWNER_NONE
        assert 'without a switch by the mux' in published[-1].detail
        status, detail = node.set_owner(Owner.TELEOP)
        assert status == MuxStatus.SWITCH_FAILED and 'deactivated before teleop' in detail
        assert all(not list(c.activate_controllers) for c in calls)
    finally:
        node.destroy_node()


def test_a_teleop_controller_the_mux_switched_is_not_outside(ros, monkeypatch):
    node, calls, published = make_mux(monkeypatch, [MuxStatus.OK] * 10)
    try:
        node.teleop_by_mux = True
        node.owner_from = lambda states: Owner.TELEOP
        node._on_timer()
        assert published[-1].owner == ArmOwner.OWNER_TELEOP and node.outside == ''
        assert calls == []
    finally:
        node.destroy_node()


def test_refuse_teleop_refuses_every_switch_to_teleop(ros):
    node = arm_mux_node.ArmMuxNode(node_name='arm_mux_refuse_test')
    try:
        assert node.refuse_teleop is False          # the default: no extra refusal
        node.refuse_teleop = True                   # as the launch sets it with motion disabled
        status, detail = node.set_owner(Owner.TELEOP)
        assert status == MuxStatus.SWITCH_FAILED
        assert 'teleop motion is disabled' in detail and 'motion_enabled:=true' in detail
    finally:
        node.destroy_node()


def test_a_request_before_the_timer_saw_the_outside_activation_is_refused(ros, monkeypatch):
    """The refusal reads the controller states of the request, not the timer's last look."""
    node, calls, published = make_mux(monkeypatch, [MuxStatus.OK] * 10)
    try:
        assert node.outside == ''                   # no timer tick since the reactivation
        status, detail = node.set_owner(Owner.TELEOP)
        assert status == MuxStatus.SWITCH_FAILED
        assert 'active without a switch by the mux' in detail
        assert calls == []
    finally:
        node.destroy_node()


def activity(states):
    """Return a ControllerManagerActivity with the given controller states."""
    from controller_manager_msgs.msg import ControllerManagerActivity, NamedLifecycleState
    from lifecycle_msgs.msg import State
    msg = ControllerManagerActivity()
    msg.controllers = [NamedLifecycleState(name=name, state=State(label=label))
                       for name, label in states.items()]
    return msg


def test_a_deactivation_between_two_reads_is_seen_from_the_activity_topic(ros, monkeypatch):
    """The driver switches everything off and on between two timer ticks: still outside."""
    node, calls, published = make_mux(monkeypatch, [MuxStatus.REFUSED_MOVING] * 10)
    try:
        node.teleop_by_mux = True                   # the mux had switched the arm to teleop
        node._on_activity(activity({'teleop_arm_controller': 'inactive',
                                    'xarm6_traj_controller': 'inactive'}))
        assert node.teleop_by_mux is False
        node._on_activity(activity(OUTSIDE))        # the driver's reactivation
        node.owner_from = lambda states: Owner.TELEOP
        node._on_timer()
        assert published[-1].owner == ArmOwner.OWNER_NONE
        status, detail = node.set_owner(Owner.TELEOP)
        assert status == MuxStatus.SWITCH_FAILED
    finally:
        node.destroy_node()


def test_the_mux_own_switch_is_not_forgotten_while_teleop_stays_active(ros, monkeypatch):
    node, calls, published = make_mux(monkeypatch, [MuxStatus.OK] * 10)
    try:
        node.teleop_by_mux = True
        node._on_activity(activity(OUTSIDE))        # teleop active: still the mux's
        assert node.teleop_by_mux is True
    finally:
        node.destroy_node()


def test_only_the_arm_teleop_controller_counts_as_an_outside_activation(ros, monkeypatch):
    """A gripper-only split stays the split rule's (reported, resolved to MoveIt on request)."""
    node, calls, published = make_mux(monkeypatch, [MuxStatus.OK] * 10)
    try:
        split = {'xarm6_traj_controller': 'active', 'teleop_arm_controller': 'inactive',
                 'teleop_gripper_controller': 'active'}
        node.read_states = lambda: dict(split)
        node.restore_order(dict(split))
        assert node.outside == '' and calls == []
    finally:
        node.destroy_node()


def test_moveit_is_not_activated_in_answer_to_an_outside_activation(ros, monkeypatch):
    node, calls, published = make_mux(monkeypatch, [MuxStatus.OK] * 10)
    try:
        status, detail = node.set_owner(Owner.MOVEIT)
        assert status == MuxStatus.SWITCH_FAILED and 'deactivated first' in detail
        assert calls == []
    finally:
        node.destroy_node()


def test_activity_from_before_the_mux_own_switch_is_ignored(ros, monkeypatch):
    node, calls, published = make_mux(monkeypatch, [MuxStatus.OK] * 10)
    try:
        node.teleop_by_mux = True
        node.switch_requested_ns = 2_000_000_000
        old = activity({'teleop_arm_controller': 'inactive'})
        old.header.stamp.sec = 1                    # stamped before the switch request
        node._on_activity(old)
        assert node.teleop_by_mux is True
        new = activity({'teleop_arm_controller': 'inactive'})
        new.header.stamp.sec = 3
        node._on_activity(new)
        assert node.teleop_by_mux is False
    finally:
        node.destroy_node()
