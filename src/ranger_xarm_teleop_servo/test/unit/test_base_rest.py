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
"""The base's rest from odometry and its turn since the calibration (G-19, INV-2)."""

import math

import pytest

from ranger_xarm_teleop_servo.base_rest import (AT_REST, HeadingWatch, MOVING, RestParams,
                                                RestWatch, UNKNOWN, yaw_difference)

DT = 0.007                                  # about the simulated odometry's period


def feed(w, t0, seconds, v=0.0, wz=0.0, yaw=0.0, dt=DT):
    t = t0
    while t < t0 + seconds - 1e-9:
        w.add(t, v, 0.0, wz, yaw)
        t += dt
    return t


def test_no_odometry_is_unknown_not_rest():
    w = RestWatch()
    assert w.state(0.0) == UNKNOWN


def test_still_for_a_whole_window_is_rest():
    w = RestWatch()
    t = feed(w, 0.0, 0.29)
    assert w.state(t) == MOVING                 # not yet a whole window
    t = feed(w, t, 0.03)
    assert w.state(t) == AT_REST


def test_any_motion_in_the_window_is_moving():
    w = RestWatch()
    t = feed(w, 0.0, 1.0)
    t = feed(w, t, 0.01, v=0.02)
    t = feed(w, t, 0.2)
    assert w.state(t) == MOVING
    t = feed(w, t, 0.12)
    assert w.state(t) == AT_REST


def test_turning_counts_as_motion():
    w = RestWatch()
    t = feed(w, 0.0, 1.0, wz=0.03)
    assert w.state(t) == MOVING
    t = feed(w, t, 1.0, wz=0.019)
    assert w.state(t) == AT_REST


def test_stale_odometry_is_unknown():
    w = RestWatch()
    t = feed(w, 0.0, 1.0)
    assert w.state(t + 0.19) == AT_REST
    assert w.state(t + 0.21) == UNKNOWN


def test_after_a_gap_stillness_must_last_a_window_again():
    """Three still samples right after odometry resumes do not make a rest."""
    w = RestWatch()
    t = feed(w, 0.0, 1.0)
    t += 0.5                                    # odometry gone half a second
    t = feed(w, t, 0.03)
    assert w.state(t) == MOVING
    t = feed(w, t, 0.3)
    assert w.state(t) == AT_REST


def test_a_non_finite_twist_is_motion():
    w = RestWatch()
    t = feed(w, 0.0, 1.0)
    w.add(t, math.nan, 0.0, 0.0)
    assert w.state(t) == MOVING


def test_too_few_samples_is_not_rest():
    w = RestWatch(RestParams(min_samples=5))
    w.add(0.0, 0.0, 0.0, 0.0)
    w.add(0.15, 0.0, 0.0, 0.0)
    w.add(0.31, 0.0, 0.0, 0.0)
    assert w.state(0.31) == MOVING


def test_rest_params_are_checked():
    with pytest.raises(ValueError):
        RestWatch(RestParams(window_s=0.0))
    with pytest.raises(ValueError):
        RestWatch(RestParams(min_samples=0))


def test_yaw_difference_wraps():
    assert yaw_difference(math.pi - 0.01, -math.pi + 0.01) == pytest.approx(0.02)


def test_a_turn_past_the_limit_invalidates_the_calibration():
    h = HeadingWatch(math.radians(5.0))
    assert not h.invalid()                      # no calibration: nothing to invalidate here
    h.on_calibration(0.3)
    h.on_odometry(0.3 + math.radians(4.9))
    assert not h.invalid()
    h.on_odometry(0.3 + math.radians(5.1))
    assert h.invalid() and '5.1 deg' in h.describe()
    h.on_odometry(0.3)                          # turning back does not restore it
    assert h.invalid()
    h.on_calibration(0.3)                       # calibrating again does
    assert not h.invalid()


def test_a_commanded_turn_without_odometry_invalidates_the_calibration():
    h = HeadingWatch(math.radians(5.0))
    h.on_calibration(0.0)
    h.on_turn_command(0.2, odometry_fresh=True)
    assert not h.invalid()
    h.on_turn_command(0.2, odometry_fresh=False)
    assert h.invalid() and 'no odometry' in h.describe()


def test_a_calibration_made_without_odometry_is_invalid_after_any_commanded_turn():
    h = HeadingWatch(math.radians(5.0))
    h.on_calibration(None)
    h.on_turn_command(0.0, odometry_fresh=True)
    assert not h.invalid()
    h.on_turn_command(-0.1, odometry_fresh=True)
    assert h.invalid()


def test_a_calibration_before_any_odometry_takes_the_first_yaw_after_it():
    h = HeadingWatch(math.radians(5.0))
    h.on_calibration(None)
    h.on_odometry(1.0)
    h.on_odometry(1.0 + math.radians(4.0))
    assert not h.invalid()
    h.on_odometry(1.0 + math.radians(6.0))
    assert h.invalid()


def test_translation_keeps_the_calibration():
    h = HeadingWatch(math.radians(5.0))
    h.on_calibration(1.0)
    for _ in range(100):
        h.on_odometry(1.0 + math.radians(0.05))     # a crab's heading drift
    assert not h.invalid()
