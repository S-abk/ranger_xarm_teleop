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
"""Ownership logic of the command mux (G-9, INV-7 at the logic level)."""

import math

import pytest
from ranger_xarm_teleop_servo.ownership import (any_goal_active, check_switch, MuxParams,
                                                MuxStatus, Owner, owner_of, StillnessTracker)

MOVEIT, TELEOP = 'xarm6_traj_controller', 'teleop_arm_controller'
STILL = (0.1, -0.5, 0.3, 0.0, 1.0, 0.0)


def tracker_still_for(seconds, rate=150.0, start=0.0):
    tracker = StillnessTracker(MuxParams())
    t = start
    for _ in range(int(seconds * rate) + 1):
        tracker.update(t, t + 500.0, STILL, (0.0,) * 6)
        t += 1.0 / rate
    return tracker, t - 1.0 / rate


@pytest.mark.parametrize('states, owner', [
    ({MOVEIT: 'active', TELEOP: 'inactive'}, Owner.MOVEIT),
    ({MOVEIT: 'inactive', TELEOP: 'active'}, Owner.TELEOP),
    ({MOVEIT: 'inactive', TELEOP: 'inactive'}, Owner.NONE),
    ({MOVEIT: 'active', TELEOP: 'active'}, Owner.NONE),
    ({MOVEIT: 'active'}, Owner.MOVEIT),
    ({}, Owner.NONE),
    ({MOVEIT: 'unconfigured', TELEOP: 'finalized'}, Owner.NONE),
])
def test_owner_follows_the_active_controller(states, owner):
    assert owner_of(states, MOVEIT, TELEOP) == owner


def test_goal_states():
    assert not any_goal_active([])
    assert not any_goal_active([4, 5, 6, 0])        # succeeded, canceled, aborted, unknown
    for status in (1, 2, 3):                        # accepted, executing, canceling
        assert any_goal_active([4, status])


def test_switch_allowed_only_when_stopped_idle_and_fresh():
    tracker, now = tracker_still_for(0.5)
    assert check_switch(now, tracker, goal_active=False, teleop_active=False) == MuxStatus.OK
    assert check_switch(now, tracker, True, False) == MuxStatus.REFUSED_GOAL_ACTIVE
    assert check_switch(now, tracker, False, True) == MuxStatus.REFUSED_TELEOP_ACTIVE
    assert check_switch(now + 0.5, tracker, False, False) == MuxStatus.STALE_JOINT_STATE
    assert check_switch(0.0, StillnessTracker(MuxParams()), False, False) == \
        MuxStatus.STALE_JOINT_STATE


def test_not_still_long_enough_is_refused():
    tracker, now = tracker_still_for(0.2)
    assert check_switch(now, tracker, False, False) == MuxStatus.REFUSED_MOVING
    tracker, now = tracker_still_for(0.31)
    assert check_switch(now, tracker, False, False) == MuxStatus.OK


def test_reported_velocity_counts_as_moving():
    tracker, now = tracker_still_for(0.5)
    tracker.update(now + 0.005, now + 500.005, STILL, (0.0, 0.05, 0.0, 0.0, 0.0, 0.0))
    assert check_switch(now + 0.005, tracker, False, False) == MuxStatus.REFUSED_MOVING


def test_position_change_counts_as_moving_even_with_zero_reported_velocity():
    tracker, now = tracker_still_for(0.5)
    moved = list(STILL)
    moved[2] += 0.001                                # 0.15 rad/s at 150 Hz
    tracker.update(now + 1.0 / 150.0, now + 500.0 + 1.0 / 150.0, moved, (0.0,) * 6)
    assert check_switch(now + 1.0 / 150.0, tracker, False, False) == MuxStatus.REFUSED_MOVING
    # Stillness has to be shown again for the full duration.
    t = now + 1.0 / 150.0
    for _ in range(30):
        t += 1.0 / 150.0
        tracker.update(t, t + 500.0, moved, None)
    assert check_switch(t, tracker, False, False) == MuxStatus.REFUSED_MOVING
    for _ in range(20):
        t += 1.0 / 150.0
        tracker.update(t, t + 500.0, moved, None)
    assert check_switch(t, tracker, False, False) == MuxStatus.OK


def test_gap_in_joint_states_restarts_the_stillness_clock():
    tracker, now = tracker_still_for(0.5)
    tracker.update(now + 1.0, now + 501.0, STILL, (0.0,) * 6)
    assert check_switch(now + 1.0, tracker, False, False) == MuxStatus.REFUSED_MOVING


def test_non_finite_velocity_is_not_still():
    tracker, now = tracker_still_for(0.5)
    tracker.update(now + 0.005, now + 500.005, STILL, (math.nan,) * 6)
    assert check_switch(now + 0.005, tracker, False, False) == MuxStatus.REFUSED_MOVING


def test_late_delivery_of_an_unbroken_sample_sequence_is_still():
    """A stall of this process delays arrivals; the samples' own stamps show no hole."""
    tracker = StillnessTracker(MuxParams())
    for k in range(60):                              # 0.4 s of samples at 150 Hz ...
        tracker.update(10.0 + 0.0001 * k, 500.0 + k / 150.0, STILL, (0.0,) * 6)
    # ... all delivered in a burst after a 0.4 s stall: stillness is 0.39 s by stamps.
    assert tracker.still_for() == pytest.approx(59 / 150.0)
    assert check_switch(10.01, tracker, False, False) == MuxStatus.OK
    # But a newest sample that arrived long ago is stale, whatever its stamp says.
    assert check_switch(10.5, tracker, False, False) == MuxStatus.STALE_JOINT_STATE


def test_stamps_going_back_restart_stillness():
    tracker, now = tracker_still_for(0.5)
    tracker.update(now + 0.005, 100.0, STILL, (0.0,) * 6)
    assert check_switch(now + 0.005, tracker, False, False) == MuxStatus.REFUSED_MOVING
    assert 'went back' in tracker.last_motion
