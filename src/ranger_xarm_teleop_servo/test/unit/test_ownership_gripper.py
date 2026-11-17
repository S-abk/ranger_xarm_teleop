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

import math

import pytest
from ranger_xarm_teleop_servo.ownership import (BandTracker, check_switch, MuxParams, MuxStatus,
                                                Owner, owner_of, split_of, StillnessTracker,
                                                switch_lists)

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


def test_the_motion_report_names_the_joint_that_moved():
    tracker = StillnessTracker(MuxParams(), names=('a', 'b'))
    tracker.update(0.0, 10.0, (0.0, 0.0), (0.0, 0.0))
    tracker.update(0.01, 10.01, (0.0, 0.01), (0.0, 0.0))
    assert tracker.last_motion == 'joint speed 1.000 rad/s (b)'
    tracker.update(0.02, 10.02, (0.0, 0.01), (0.5, 0.0))
    assert tracker.last_motion == 'joint speed 0.500 rad/s (a)'


# ---- the gripper's "stopped": net travel within a band over the stillness window ----

def still_arm(seconds=1.0, rate=1000.0):
    tracker = StillnessTracker(MuxParams())
    for k in range(int(seconds * rate) + 1):
        tracker.update(k / rate, 100.0 + k / rate, (0.0,) * 6, (0.0,) * 6)
    return tracker


def gripper_samples(values, rate=1000.0, band=0.01):
    tracker = BandTracker(MuxParams(), band, names=('xarm_drive_joint',))
    for k, value in enumerate(values):
        tracker.update(k / rate, 100.0 + k / rate, (value,))
    return tracker


def test_jitter_that_goes_nowhere_is_still():
    # 0.156 rad/s sample-to-sample at 1 kHz (0.000156 rad steps back and forth).
    values = [0.4 + (0.000156 if k % 7 == 0 else 0.0) for k in range(1001)]
    tracker = gripper_samples(values)
    assert tracker.still()
    assert check_switch(1.0, still_arm(), False, False, tracker) == MuxStatus.OK


def test_a_ramp_at_the_speed_limit_is_motion():
    values = [min(0.849, 0.0005 * k) for k in range(1001)]          # 0.5 rad/s, still going
    tracker = gripper_samples(values)
    assert not tracker.still()
    assert 'xarm_drive_joint' in tracker.last_motion
    assert check_switch(1.0, still_arm(), False, False, tracker) == MuxStatus.REFUSED_MOVING


def test_a_ramp_that_ended_is_still_once_the_window_has_passed():
    values = [min(0.849, 0.001 * k) for k in range(2001)]           # ends at 0.849 s
    assert gripper_samples(values[:1000]).still() is False
    assert gripper_samples(values).still() is True


def test_a_short_history_or_a_hole_is_not_still():
    assert not gripper_samples([0.4] * 200).still()                 # 0.2 s covered
    tracker = gripper_samples([0.4] * 1000)
    tracker.update(2.0, 100.0 + 2.0, (0.4,))                         # 1 s hole
    assert not tracker.still() and 'gap' in tracker.last_motion


def test_stamps_going_back_restart_the_window():
    tracker = gripper_samples([0.4] * 1000)
    tracker.update(1.0, 99.0, (0.4,))
    assert not tracker.still() and 'back' in tracker.last_motion


def test_a_stale_gripper_joint_state_refuses_the_switch():
    tracker = gripper_samples([0.4] * 1000)
    assert check_switch(5.0, still_arm(), False, False, tracker) == MuxStatus.STALE_JOINT_STATE


def test_a_non_finite_gripper_position_is_not_still():
    tracker = gripper_samples([0.4] * 999 + [math.nan])
    assert not tracker.still()


def test_without_a_gripper_tracker_the_check_is_the_arms_alone():
    assert check_switch(1.0, still_arm(), False, False) == MuxStatus.OK


@pytest.mark.parametrize('band', [0.0, -0.01, math.nan, math.inf])
def test_the_band_must_be_finite_and_positive(band):
    with pytest.raises(ValueError, match='gripper_still_band'):
        BandTracker(MuxParams(), band)


def test_a_repeated_stamp_adds_nothing():
    tracker = gripper_samples([0.4] * 1000)
    before = len(tracker.samples)
    for _ in range(5000):
        tracker.update(1.0, 100.999, (0.4,))
    assert len(tracker.samples) == before and tracker.still()


def test_an_uncovered_window_says_so():
    tracker = gripper_samples([0.4] * 100)
    assert not tracker.still() and 'cover only' in tracker.last_motion
