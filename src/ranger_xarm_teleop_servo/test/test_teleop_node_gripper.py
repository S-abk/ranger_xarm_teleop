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
Teleop node with gripper teleop: INV-1, -2, -3, -5, -6 of the gripper task, in process.

Built on the ideal-arm harness of test_teleop_node.py. The gripper side is an ideal
trajectory controller: while teleop owns the gripper it holds the last command it was sent
(or, before any, the opening it held at the switch), and the relay reports that as the held
reference one cycle later. The measured opening is the command a cycle late.
"""

from dataclasses import replace
import math
import os
import sys

import numpy as np
import pytest
from ranger_xarm_teleop_bridge.bridge_node import BridgeNode
from ranger_xarm_teleop_bridge.frame_parse import parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.jsonl_io import read_records
from ranger_xarm_teleop_interfaces.msg import ArmOwner, GripperCommand, TeleopState
from ranger_xarm_teleop_servo import teleop_node
from ranger_xarm_teleop_servo.teleop_node import StartupRefused, TeleopNode
import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unit'))
from keepout_robot import JOINTS as KO_JOINTS, resolve, robot_urdf  # noqa: E402, I100
from test_teleop_node import FIXTURES, Harness, JOINTS, LIMITS  # noqa: E402, I100

GRIPPER = {'gripper_enabled': True, 'gripper_open_position': 0.0,
           'gripper_closed_position': 0.849, 'gripper_max_speed': 1.0}
OPEN, CLOSED, STEP = 0.0, 0.849, 0.02
JOINT = 'xarm_drive_joint'


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    bridge = BridgeNode(node_name='bridge_for_gripper_test', source_label='test')
    yield bridge
    bridge.destroy_node()
    rclpy.shutdown()


class GripperHarness(Harness):
    """The ideal-arm harness plus an ideal gripper controller and per-cycle records."""

    def __init__(self, bridge, held=OPEN, **params):
        super().__init__(bridge, **params)
        node = self.node
        self.grips = []
        if node.gripper_pub is not None:
            node.gripper_pub.publish = self.grips.append
        self.controller = held          # what the gripper's controller holds
        self.measured_gripper = held
        self.owner_gripper = True       # the mux swaps the gripper with the arm
        self.force_owner = None         # ArmOwner value to report regardless of arming
        self.short_axes = False         # Joy without the trigger axis
        self.relay = True               # the relay reports the held reference
        self.cycles = []                # per cycle: (t, targets, grips, state)
        self.actions = {}               # time -> callable, run before that cycle

    def _ownership(self, now, owner):
        super()._ownership(now, owner)
        if self.force_owner is not None:
            self.node.owner = self.force_owner
        self.node.owner_gripper = self.owner_gripper

    def feed(self, parsed, arrival, drop=()):
        if not self.short_axes:
            return super().feed(parsed, arrival, drop)
        self.now = arrival
        stamp = rclpy.time.Time(nanoseconds=int(1e9 + parsed.seq * 1e7)).to_msg()
        messages = self.bridge.build_messages(parsed, stamp)
        messages.joy.axes = list(messages.joy.axes)[:4]
        if 'right' in messages.poses:
            self.node._on_pose(messages.poses['right'])
        if messages.head is not None:
            self.node._on_head(messages.head)
        self.node._on_joy(messages.joy)
        self.node._on_status(messages.status)

    def tick(self, now):
        node = self.node
        owned = node.owner == ArmOwner.OWNER_TELEOP and node.owner_gripper
        if owned and self.relay:
            node.gripper_held = (now, self.controller)      # the relay's report
        stamp_ns = node.get_clock().now().nanoseconds
        node.joint_state = (now, stamp_ns, {**dict(zip(JOINTS, self.measured_q)),
                                            JOINT: self.measured_gripper})
        node.joint_state_arrival = now
        before = (len(self.targets), len(self.grips), len(self.states))
        super().tick(now)
        grips = self.grips[before[1]:]
        if grips and owned:
            self.controller = grips[-1].position
        self.measured_gripper = self.controller     # seen in the next cycle: a cycle late
        self.cycles.append((now, self.targets[before[0]:], grips, self.states[-1]))

    def play(self, name, drop=lambda parsed: (), edit=lambda t, parsed: parsed, until=None):
        records = read_records(os.path.join(FIXTURES, name + '.jsonl'))
        frames = []
        for record in records:
            parsed = parse_text(record.text())
            if isinstance(parsed, ParsedFrame):
                frames.append((record.t_recv - records[0].t_recv, parsed))
        index, k = 0, 0
        end = frames[-1][0] + 0.3 if until is None else until
        while k * 0.02 <= end:
            now = k * 0.02
            for at in [a for a in self.actions if a <= now]:
                self.actions.pop(at)(self)
            while index < len(frames) and frames[index][0] <= now:
                t, parsed = frames[index]
                self.feed(edit(t, parsed), t, drop(parsed))
                index += 1
            self.tick(now)
            k += 1

    def state_at(self, t):
        return min(self.cycles, key=lambda c: abs(c[0] - t))[3]

    def commands_between(self, start, end):
        return [g.position for c in self.cycles if start <= c[0] < end for g in c[2]]


@pytest.fixture
def harness(ros):
    made = []

    def make(gripper=True, **kwargs):
        params = dict(GRIPPER) if gripper else {}
        params.update(kwargs.pop('params', {}))
        made.append(GripperHarness(ros, **kwargs, **params))
        return made[-1]
    yield make
    for h in made:
        h.close()


def published_topics(node):
    return {p.topic_name for p in node.publishers}


# ---- parameters ----

def test_enabled_without_the_range_and_speed_refuses_to_start(ros):
    for missing in ('gripper_open_position', 'gripper_closed_position', 'gripper_max_speed'):
        values = {**LIMITS, **{k: v for k, v in GRIPPER.items() if k != missing}}
        with pytest.raises(StartupRefused, match=missing):
            TeleopNode(node_name='teleop_gripper_missing',
                       parameter_overrides=[Parameter(k, value=v) for k, v in values.items()])


def test_a_range_outside_the_joint_limits_or_a_missing_joint_refuses_to_start(ros):
    for values, match in (({'gripper_joint': 'finger_left_joint',
                            'gripper_closed_position': 0.849}, 'outside the limits'),
                          ({'gripper_joint': 'no_such_joint'}, 'no joint no_such_joint')):
        node = TeleopNode(node_name='teleop_gripper_range', parameter_overrides=[
            Parameter(k, value=v) for k, v in {**LIMITS, **GRIPPER, **values}.items()])
        node._resolver = lambda: resolve
        try:
            with pytest.raises(StartupRefused, match=match):
                node._on_robot_description(String(data=robot_urdf()))
        finally:
            node.close()
            node.destroy_node()


# ---- INV-5 and INV-7 ----

def test_disabled_publishes_no_gripper_command_and_the_trigger_has_no_effect(harness):
    h = harness(gripper=False)
    assert not h.node.gripper_params.enabled
    assert h.node.gripper_pub is None
    assert '/teleop/gripper/command' not in published_topics(h.node)
    h.play('gripper_clutch_strokes')
    assert h.grips == []
    assert all(not s.gripper_enabled and s.gripper_hold == TeleopState.GRIPPER_HOLD_DISABLED
               and math.isnan(s.gripper_command) and not s.gripper_active for s in h.states)
    assert h.controller == OPEN


def test_enabled_by_default_is_false(ros):
    node = TeleopNode(node_name='teleop_gripper_default',
                      parameter_overrides=[Parameter(k, value=v) for k, v in LIMITS.items()])
    try:
        assert node.gripper_params.enabled is False and node.gripper_pub is None
    finally:
        node.close()
        node.destroy_node()


@pytest.mark.parametrize('fixture', ['clutch_line', 'clutch_reengage', 'gate_absent'])
def test_arm_output_is_bit_identical_with_gripper_teleop_on_and_the_trigger_untouched(
        harness, fixture):
    off, on = harness(gripper=False), harness()
    off.play(fixture)
    on.play(fixture)
    assert len(off.targets) == len(on.targets) > 0
    for a, b in zip(off.targets, on.targets):
        assert a.pose == b.pose                     # (stamps are this machine's clock)
    assert [list(c.velocities) for c in off.commands] == [list(c.velocities)
                                                          for c in on.commands]
    assert [s.hold_reason for s in off.states] == [s.hold_reason for s in on.states]
    assert all(g.position == OPEN for g in on.grips)


# ---- mapping, INV-2 ----

def test_clutch_strokes_close_between_strokes_hold_across_regrips_and_open_mid_stroke(harness):
    h = harness()
    h.play('gripper_clutch_strokes')
    engaged = [(c[0], c[3].clutch_engaged) for c in h.cycles]
    # The trigger works while the clutch is released: closing starts between strokes.
    closing = [c for c in h.cycles if 2.7 <= c[0] < 3.2]
    assert not any(c[3].clutch_engaged for c in closing)
    assert any(g.position > OPEN for c in closing for g in c[2])
    # Fully closed by 3.8 s, and held there across the re-grips and clutch releases.
    held = h.commands_between(3.7, 5.8)
    assert held and all(p == CLOSED for p in held)
    assert any(not e for t, e in engaged if 4.4 < t < 5.2)
    assert any(e for t, e in engaged if 5.2 < t < 5.8)
    # Released mid-stroke (clutch engaged): opens at the speed limit, fully by 6.9 s.
    opening = [c for c in h.cycles if 5.9 <= c[0] < 6.4]
    assert all(c[3].clutch_engaged for c in opening)
    assert h.commands_between(6.9, 7.9) and all(p == OPEN for p in h.commands_between(6.9, 7.9))
    positions = [g.position for g in h.grips]
    assert max(abs(b - a) for a, b in zip(positions, positions[1:])) <= STEP + 1e-9
    requests = {g.request for g in h.grips}
    assert requests == {GripperCommand.REQUEST_NONE, GripperCommand.REQUEST_CLOSE,
                        GripperCommand.REQUEST_OPEN}


def test_a_clutch_release_does_not_change_the_gripper_command(harness):
    h = harness()
    h.play('gripper_clutch_strokes')
    changes = 0
    for before, after in zip(h.cycles, h.cycles[1:]):
        if before[3].clutch_engaged != after[3].clutch_engaged:
            changes += 1
            a = before[3].gripper_command
            b = after[3].gripper_command
            # The command moves only by the ramp: never more than one step at a transition,
            # and not at all when the gripper is not ramping.
            assert abs(b - a) <= STEP + 1e-9
            if a in (OPEN, CLOSED) and after[0] < 5.8:
                assert b == a
    assert changes >= 5


def test_the_gripper_command_is_the_same_whether_the_clutch_is_used_or_not(harness):
    """INV-2: the clutch is not an input of the gripper; the same trigger, the same command."""
    def never_clutch(t, parsed):
        right = parsed.right
        if right is None or len(right.buttons) < 2:
            return parsed
        buttons = list(right.buttons)
        buttons[1] = replace(buttons[1], pressed=False, value=0.0)
        return replace(parsed, right=replace(right, buttons=tuple(buttons)))
    used, unused = harness(), harness()
    used.play('gripper_clutch_strokes')
    unused.play('gripper_clutch_strokes', edit=never_clutch)
    assert any(c[3].clutch_engaged for c in used.cycles)
    assert not any(c[3].clutch_engaged for c in unused.cycles)
    assert [(g.position, g.request) for g in used.grips] == [
        (g.position, g.request) for g in unused.grips]
    # From the clutch's first use on (before it, the runs are the same by construction).
    assert [c[3].gripper_command for c in used.cycles if c[0] > 1.2] == [
        c[3].gripper_command for c in unused.cycles if c[0] > 1.2]


def test_a_held_reference_from_before_the_owner_switch_is_not_used(harness):
    h = harness(held=0.3)
    h.relay = False
    h.node.gripper_held = (0.0, 0.7)            # an old report, from before teleop owned it
    h.play('arm_only')
    assert h.grips and all(g.reference_source == GripperCommand.SOURCE_MEASURED
                           for g in h.grips)
    assert h.grips[0].position == pytest.approx(0.3)


# ---- INV-6 and approval addition 1 ----

def test_command_and_state_share_the_cycle_stamp_and_the_target_stamp(harness):
    h = harness()
    h.play('gripper_clutch_strokes')
    both = 0
    for t, targets, grips, state in h.cycles:
        assert len(grips) <= 1
        for grip in grips:
            assert grip.header.stamp == state.header.stamp
            assert grip.joint_name == JOINT
            assert state.gripper_active
            for target in targets:
                assert target.header.stamp == grip.header.stamp
                both += 1
        if not grips:
            assert not state.gripper_active
    assert both > 50


def test_measured_and_commanded_are_separate_fields(harness):
    h = harness()
    h.play('gripper_clutch_strokes')
    ramp = [c[3] for c in h.cycles if 2.9 <= c[0] < 3.5]
    assert all(s.gripper_measured_valid for s in ramp)
    # The ideal controller's measurement trails the command by a cycle while it ramps.
    assert all(s.gripper_command - s.gripper_measured == pytest.approx(STEP, abs=1e-9)
               for s in ramp)
    stamp_ns = ramp[0].gripper_measured_stamp.sec * 10**9 + ramp[0].gripper_measured_stamp.nanosec
    assert stamp_ns > 0


def test_state_carries_the_command_in_force_every_cycle_also_while_frozen(harness):
    h = harness()
    h.play('gripper_dropout')
    frozen = [c for c in h.cycles if 2.33 <= c[0] < 2.48]
    assert frozen and all(c[2] == [] for c in frozen)
    part = frozen[0][3].gripper_command
    assert OPEN < part < CLOSED
    assert all(c[3].gripper_command == part for c in frozen)
    assert all(c[3].gripper_hold == TeleopState.GRIPPER_HOLD_INPUT for c in frozen)
    armed = [c for c in h.cycles if c[0] > 1.2]
    assert all(not math.isnan(c[3].gripper_command) for c in armed)


# ---- INV-1: one test per trigger ----

def ramping(h, start=2.0, end=2.3):
    """Return the last command before ``end``; the gripper was closing part-way."""
    commands = h.commands_between(start, end)
    assert commands and OPEN < commands[-1] < CLOSED
    return commands[-1]


def freeze(h, after, until):
    """Return (first frozen cycle time, part-closed command, last frozen cycle time)."""
    frozen = [c for c in h.cycles if after <= c[0] < until and not c[2]]
    assert frozen
    first, last = frozen[0][0], frozen[-1][0]
    window = [c for c in h.cycles if first <= c[0] <= last]
    assert all(c[2] == [] for c in window)          # one unbroken freeze
    part = [g.position for c in h.cycles if c[0] < first for g in c[2]][-1]
    assert OPEN < part < CLOSED
    # The command in force stays in the state while nothing is sent (approval addition 1).
    assert all(c[3].gripper_command == part for c in window)
    assert all(c[3].gripper_hold == TeleopState.GRIPPER_HOLD_INPUT for c in window)
    return first, part, last


def assert_held_awaiting(h, start, end, part):
    """From ``start`` to ``end`` the part-closed command is published, awaiting a crossing."""
    after = [c for c in h.cycles if start <= c[0] < end]
    assert after and all(c[2] for c in after)
    assert all(g.position == part for c in after for g in c[2])
    assert all(g.request == GripperCommand.REQUEST_NONE for c in after for g in c[2])
    assert all(c[3].gripper_hold == TeleopState.GRIPPER_HOLD_AWAIT_TRIGGER for c in after)


def test_freeze_on_an_absent_controller_mid_ramp(harness):
    h = harness()
    h.play('gripper_dropout')
    first, part, last = freeze(h, 2.25, 2.6)
    assert 2.28 <= first <= 2.34 and 2.46 <= last <= 2.54
    assert_held_awaiting(h, last + 0.03, 3.5, part)
    assert h.state_at(3.0).gripper_reference_source == GripperCommand.SOURCE_OWN
    # Released at 3.5 s: opens. Pressed again at 4.5 s: closes.
    assert h.commands_between(3.6, 3.7)[-1] < part
    assert h.commands_between(4.6, 4.7)[-1] > h.commands_between(4.5, 4.55)[0]


def test_freeze_on_stale_input_mid_ramp(harness):
    h = harness()
    h.play('gripper_stall')
    first, part, last = freeze(h, 2.25, 2.6)
    assert 2.38 <= first <= 2.44                    # 100 ms after the last frame (G-3)
    assert h.state_at(first).hold_reason == TeleopState.HOLD_INPUT_STALE
    assert_held_awaiting(h, last + 0.03, 3.5, part)


def test_freeze_on_a_missing_trigger_value(harness):
    h = harness()
    # Closing between clutch strokes (trigger pressed at 2.7 s, clutch released 2.4-3.2 s).
    h.actions[2.9] = lambda hh: setattr(hh, 'short_axes', True)
    h.actions[3.1] = lambda hh: setattr(hh, 'short_axes', False)
    h.play('gripper_clutch_strokes')
    first, part, last = freeze(h, 2.85, 3.15)
    assert 2.88 <= first <= 2.94 and 3.06 <= last <= 3.14
    window = [c for c in h.cycles if first <= c[0] <= last]
    # The arm side never saw a difference: the buttons were all there.
    assert all(c[3].hold_reason == TeleopState.HOLD_CLUTCH_RELEASED for c in window)
    # Held until the trigger crosses again (the re-grip wobble rises through 0.6 at 3.27 s).
    assert_held_awaiting(h, last + 0.03, 3.25, part)
    assert h.commands_between(3.3, 3.4)[-1] > part


def test_freeze_on_disarm_mid_ramp_and_nothing_after(harness):
    h = harness()
    h.actions[2.2] = lambda hh: hh.node._disarm('test')
    h.play('gripper_dropout')
    part = ramping(h, 2.0, 2.2)
    after = [c for c in h.cycles if c[0] >= 2.22]
    assert all(c[2] == [] for c in after)
    assert all(c[3].gripper_hold in (TeleopState.GRIPPER_HOLD_NOT_ARMED,
                                     TeleopState.GRIPPER_HOLD_NOT_OWNER) for c in after)
    assert h.controller == part


def test_freeze_on_owner_loss_mid_ramp(harness):
    h = harness()
    h.actions[2.2] = lambda hh: setattr(hh, 'force_owner', ArmOwner.OWNER_MOVEIT)
    h.play('gripper_dropout')
    ramping(h, 2.0, 2.2)
    after = [c for c in h.cycles if c[0] >= 2.26]
    assert all(c[2] == [] for c in after)
    assert all(c[3].gripper_hold == TeleopState.GRIPPER_HOLD_NOT_OWNER for c in after)
    assert all(math.isnan(c[3].gripper_command) for c in after)


def test_freeze_when_the_mux_does_not_swap_the_gripper(harness):
    h = harness()
    h.owner_gripper = False
    h.play('gripper_clutch_strokes')
    assert h.grips == []
    armed = [c[3] for c in h.cycles if c[0] > 1.5]
    assert all(s.gripper_hold == TeleopState.GRIPPER_HOLD_NOT_OWNER for s in armed)


def test_freeze_on_a_gate_trip_between_frames_with_the_clutch_released(harness):
    h = harness()

    def edit(t, parsed):
        # One frame whose sequence number steps back (G-4) at 2.2 s, clutch released.
        return replace(parsed, seq=parsed.seq - 50) if 2.19 <= t < 2.205 else parsed
    h.play('gripper_dropout', edit=edit)
    first, part, last = freeze(h, 2.15, 2.3)
    assert 2.18 <= first <= 2.24
    # The trip is between frames: the arm's gate (clutch released) does not report it.
    assert h.state_at(first).hold_reason == TeleopState.HOLD_CLUTCH_RELEASED
    # The gripper holds, awaiting a crossing (the gap 2.3-2.5 s freezes it once more).
    assert_held_awaiting(h, 2.6, 3.5, part)


def test_a_release_during_an_absence_does_not_open_the_gripper(harness):
    h = harness()
    h.play('gripper_absent_release')
    assert h.commands_between(2.6, 3.0)[-1] == CLOSED
    gap = [c for c in h.cycles if 3.02 <= c[0] < 3.5]
    assert all(c[2] == [] for c in gap)
    after = h.commands_between(3.6, 4.5)
    assert after and all(p == CLOSED for p in after)       # released, but no crossing seen
    assert h.state_at(4.0).gripper_hold == TeleopState.GRIPPER_HOLD_AWAIT_TRIGGER
    assert h.commands_between(5.6, 6.2)[-1] == OPEN        # pressed and released again


# ---- INV-3 ----

def test_arm_with_trigger_released_while_the_held_reference_is_closed_stays_closed(harness):
    h = harness(held=CLOSED)
    h.play('arm_only')
    assert h.grips and all(g.position == CLOSED for g in h.grips)
    assert all(g.reference_source == GripperCommand.SOURCE_HELD for g in h.grips)
    assert h.state_at(2.5).gripper_hold == TeleopState.GRIPPER_HOLD_AWAIT_TRIGGER
    assert h.controller == CLOSED


def test_arming_takes_an_intermediate_held_opening_as_it_is(harness):
    h = harness(held=0.37)
    h.play('arm_only')
    assert h.grips and all(g.position == 0.37 for g in h.grips)


def test_without_a_held_reference_the_measured_opening_is_used_after_the_timeout(harness):
    h = harness(held=0.3)
    h.relay = False                     # no held reference ever arrives
    h.play('arm_only')
    first = [c for c in h.cycles if c[2]]
    assert first
    start = next(c[0] for c in h.cycles
                 if c[3].gripper_hold == TeleopState.GRIPPER_HOLD_NO_REFERENCE)
    assert first[0][0] - start >= 0.5 - 1e-9
    assert first[0][2][0].reference_source == GripperCommand.SOURCE_MEASURED
    assert first[0][2][0].position == pytest.approx(0.3)


# ---- keep-out follows the fingers ----

def test_the_keepout_places_the_fingers_at_their_values(ros):
    values = {**LIMITS, **GRIPPER, 'gripper_joint': 'finger_left_joint',
              'gripper_closed_position': 0.04, 'gripper_max_speed': 0.05, 'motion_enabled': True}
    node = TeleopNode(node_name='teleop_gripper_keepout',
                      parameter_overrides=[Parameter(k, value=v) for k, v in values.items()])
    node._resolver = lambda: resolve
    try:
        node._on_robot_description(String(data=robot_urdf()))
        assert node.gripper_keepout is not None and node.gripper_keepout.pair_list
        assert {pair[0] for pair in node.gripper_keepout.pairs} == {'finger_left',
                                                                    'finger_right'}
        assert node.gripper_moved == {'finger_left_joint', 'finger_right_joint'}
        q = [0.0, -1.35, -0.3, 0.0, 1.6, 0.0]          # the tool near the screen
        node.joint_state = (node.mono(), node.get_clock().now().nanoseconds,
                            {**dict(zip(KO_JOINTS, q)), 'finger_left_joint': 0.0,
                             'finger_right_joint': 0.0})
        clearance = node.gripper_clearance(np.array(q))
        values = [clearance(p) for p in (0.0, 0.02, 0.04)]
        assert all(math.isfinite(v) for v in values)
        assert len({round(v, 6) for v in values}) > 1    # the fingers move in the model
        # The measured follower value is ignored: the candidate drives both fingers.
        node.joint_state[2]['finger_right_joint'] = 0.04
        assert clearance(0.0) == pytest.approx(values[0])
        assert math.isnan(node.gripper_clearance(None)(0.0))
    finally:
        node.close()
        node.destroy_node()


def test_without_finger_geometry_there_is_no_gripper_keepout_check(harness):
    h = harness()
    assert h.node.gripper_keepout is None
    assert h.node.gripper_clearance(np.zeros(6)) is None


# ---- the finger keep-out in the node (wiring of margin, model and measured joints) ----

NEAR_MAST = [2.4, -0.769, -0.81, 2.124, 1.204, 1.359]   # opening the fingers nears the mast
AWAY = [0.0, -0.6, -0.7, 0.0, 1.3, 0.0]


class _Output:
    condition = teleop_node.Hold.CLUTCH_RELEASED


def keepout_gripper_node(ros):
    values = {**LIMITS, **GRIPPER, 'gripper_joint': 'finger_left_joint',
              'gripper_open_position': 0.04, 'gripper_closed_position': 0.0,
              'gripper_max_speed': 0.05, 'motion_enabled': True}
    node = TeleopNode(node_name='teleop_gripper_keepout_cycle',
                      parameter_overrides=[Parameter(k, value=v) for k, v in values.items()])
    node._resolver = lambda: resolve
    node._on_robot_description(String(data=robot_urdf()))
    return node


def run_cycles(node, q, finger, triggers, now, published):
    """Run the gripper step of the node with the arm at ``q``; the finger follows the command."""
    for trigger in triggers:
        now += 0.02
        node.mono = lambda t=now: t
        node.joint_state = (now, node.get_clock().now().nanoseconds,
                            {**dict(zip(KO_JOINTS, q)), 'finger_left_joint': finger[0]})
        node.joint_state_arrival = now
        node.state = replace(node.state, frame=replace(node.state.frame, trigger=trigger))
        node.gripper_cycle(now, ArmOwner.OWNER_TELEOP, _Output(), np.array(q),
                           node.get_clock().now().to_msg())
        if node.gripper_output.publish:
            finger[0] = node.gripper_output.command
            published.append((node.gripper_output.command, node.gripper_output.hold))
    return now


def test_opening_into_the_margin_stops_there_and_resumes_when_the_arm_moves_away(ros):
    from ranger_xarm_teleop_servo.gate import Frame, GateState
    node = keepout_gripper_node(ros)
    try:
        published = []
        node.gripper_pub.publish = lambda msg: None
        node.owner_gripper = True
        node.state = GateState(want_armed=True, frame=Frame(
            arrival=0.0, xr_time=0.0, session_id=1, seq=1, reset_count=0, controller=None,
            controller_present=True, controller_emulated=False, head=None, head_present=True,
            head_emulated=False, grip=False, calibrate=False, arm=False, trigger=1.0))
        node.gripper_held = (0.0, 0.0)
        finger = [0.0]
        now = run_cycles(node, NEAR_MAST, finger, [1.0] * 3, 0.0, published)
        node.gripper_held = (now, 0.0)          # the relay's report after the switch
        now = run_cycles(node, NEAR_MAST, finger, [1.0] * 2 + [0.0] * 40, now, published)
        clearance = node.gripper_clearance(np.array(NEAR_MAST))
        stopped = published[-1][0]
        assert published[-1][1] == teleop_node.grip.Hold.KEEP_OUT
        assert 0.0 < stopped < 0.02
        assert clearance(stopped) >= 0.025 - 1e-6
        assert clearance(stopped + 0.001) < 0.025
        assert clearance(0.04) < 0.0                # where it would have gone without the hold
        # The arm moves away; the trigger stays released: the opening continues by itself.
        run_cycles(node, AWAY, finger, [0.0] * 40, now, published)
        assert published[-1][0] == pytest.approx(0.04)
        assert published[-1][1] == teleop_node.grip.Hold.NONE
    finally:
        node.close()
        node.destroy_node()


def test_the_arm_keepout_places_the_fingers_at_their_measured_values(ros):
    node = keepout_gripper_node(ros)
    try:
        node._on_tf_static(teleop_node.TFMessage(transforms=[]))     # builds the arm chain
        clearances = []
        for finger in (0.0, 0.04):
            node.joint_state = (node.mono(), node.get_clock().now().nanoseconds,
                                {**dict(zip(KO_JOINTS, NEAR_MAST)), 'finger_left_joint': finger})
            assert node.keepout_extra()['finger_right_joint'] == finger     # mimic resolved
            node._idle_clearance = None
            clearances.append(node.idle_clearance()[0])
        # Another pair is nearest with the fingers closed; opened, a finger is nearer still
        # (inside the mast), and the measured clearance the arm's keep-out uses shows it.
        assert clearances[1] < 0.0 < clearances[0]
    finally:
        node.close()
        node.destroy_node()
