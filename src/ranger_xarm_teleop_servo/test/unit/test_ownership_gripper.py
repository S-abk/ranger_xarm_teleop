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
"""Command mux with a gripper pair: one owner for arm and gripper (gripper task INV-4)."""

import pytest
from ranger_xarm_teleop_servo.ownership import Owner, owner_of, split_of, switch_lists

MOVEIT, TELEOP = 'xarm6_traj_controller', 'teleop_arm_controller'
MOVEIT_G, TELEOP_G = 'xarm_gripper_traj_controller', 'teleop_gripper_controller'
PAIRS = [(MOVEIT, TELEOP), (MOVEIT_G, TELEOP_G)]
ARM_ONLY = [(MOVEIT, TELEOP)]
A, I = 'active', 'inactive'  # noqa: E741


def states(m, t, mg, tg):
    out = {MOVEIT: m, TELEOP: t, MOVEIT_G: mg, TELEOP_G: tg}
    return {k: v for k, v in out.items() if v is not None}


@pytest.mark.parametrize('given, owner', [
    (states(A, I, A, I), Owner.MOVEIT),
    (states(I, A, I, A), Owner.TELEOP),
    (states(A, I, I, A), Owner.NONE),            # split: arm MoveIt, gripper teleop
    (states(I, A, A, I), Owner.NONE),            # split the other way
    (states(I, A, A, A), Owner.NONE),            # both gripper controllers active
    (states(I, A, I, I), Owner.NONE),            # teleop's arm, nobody's gripper
    (states(A, I, A, None), Owner.MOVEIT),       # teleop gripper not loaded yet
    (states(A, I, None, None), Owner.NONE),      # MoveIt's gripper controller missing
])
def test_owner_needs_arm_and_gripper_to_agree(given, owner):
    assert owner_of(given, MOVEIT, TELEOP, MOVEIT_G, TELEOP_G) == owner


def test_without_gripper_names_the_owner_is_the_arms_alone():
    assert owner_of(states(I, A, A, I), MOVEIT, TELEOP) == Owner.TELEOP
    assert owner_of(states(I, A, A, I), MOVEIT, TELEOP, '', '') == Owner.TELEOP


@pytest.mark.parametrize('given, split', [
    (states(A, I, A, I), False), (states(I, A, I, A), False), (states(I, I, I, I), False),
    (states(A, I, I, A), True), (states(I, A, A, I), True), (states(I, A, A, A), True),
    (states(A, A, A, I), True),
    (states(I, A, I, I), False),                 # incomplete: nobody's gripper
    (states(A, I, None, I), False),              # MoveIt's gripper controller still loading
])
def test_split_is_reported(given, split):
    text = split_of(given, PAIRS)
    assert bool(text) == split
    if split:
        assert MOVEIT_G in text and TELEOP_G in text


def test_switch_to_teleop_swaps_both_pairs_in_one_request():
    (activate, deactivate), why = switch_lists(states(A, I, A, I), PAIRS, Owner.TELEOP)
    assert why == ''
    assert activate == [TELEOP, TELEOP_G] and deactivate == [MOVEIT, MOVEIT_G]


def test_switch_to_moveit_swaps_both_pairs_in_one_request():
    (activate, deactivate), _ = switch_lists(states(I, A, I, A), PAIRS, Owner.MOVEIT)
    assert activate == [MOVEIT, MOVEIT_G] and deactivate == [TELEOP, TELEOP_G]


def test_an_incomplete_pair_is_completed_by_the_switch():
    (activate, deactivate), why = switch_lists(states(A, I, None, I), PAIRS, Owner.TELEOP)
    assert why == '' and activate == [TELEOP, TELEOP_G] and deactivate == [MOVEIT]


def test_a_split_is_never_switched_to_teleop_only_back_to_moveit():
    lists, why = switch_lists(states(A, I, I, A), PAIRS, Owner.TELEOP)
    assert lists is None and 'split' in why
    (activate, deactivate), why = switch_lists(states(A, I, I, A), PAIRS, Owner.MOVEIT)
    assert why == '' and activate == [MOVEIT_G] and deactivate == [TELEOP_G]


@pytest.mark.parametrize('given, wanted, expected', [
    # The arm-only switch requests exactly what it did before the gripper pair existed.
    ({MOVEIT: A, TELEOP: I}, Owner.TELEOP, ([TELEOP], [MOVEIT])),
    ({MOVEIT: I, TELEOP: A}, Owner.MOVEIT, ([MOVEIT], [TELEOP])),
    ({MOVEIT: I, TELEOP: I}, Owner.TELEOP, ([TELEOP], [])),
    ({TELEOP: I}, Owner.TELEOP, ([TELEOP], [])),
])
def test_arm_only_switch_is_unchanged(given, wanted, expected):
    lists, why = switch_lists(given, ARM_ONLY, wanted)
    assert why == '' and lists == expected
