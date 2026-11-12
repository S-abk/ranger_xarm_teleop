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
"""Calibration (contract "Calibration", G-13) and the required parameters (INV-10)."""

import math

import pytest
from ranger_xarm_teleop_servo.calibration import calibrate, Calibration, heading_from_pointing
from ranger_xarm_teleop_servo.mapping import clutch_relative
from ranger_xarm_teleop_servo.params import build_params, MissingValue, parse_planes, REQUIRED
from ranger_xarm_teleop_servo.rotations import (Pose, q_from_axis_angle, q_from_yaw, q_mul,
                                                q_rotate, v_sub)

from helpers import VALUES

X = (1.0, 0.0, 0.0)


@pytest.mark.parametrize('yaw_deg', [-170.0, -90.0, -20.0, 0.0, 35.0, 90.0, 179.0])
def test_pointing_direction_becomes_base_forward(yaw_deg):
    yaw = math.radians(yaw_deg)
    # Pitched 20 deg down as well: only the horizontal projection counts.
    orientation = q_mul(q_from_yaw(yaw), q_from_axis_angle((0.0, 1.0, 0.0), math.radians(20.0)))
    heading = heading_from_pointing(orientation, X, 0.5)
    pointing = q_rotate(orientation, X)
    horizontal = (pointing[0], pointing[1], 0.0)
    mapped = q_rotate(q_from_yaw(heading), horizontal)
    assert mapped[0] > 0.0 and abs(mapped[1]) < 1e-12
    # A hand motion along the pointing direction moves the tool along base +x.
    clutch = Pose((0.0, 0.0, 1.0), orientation)
    now = Pose((0.1 * math.cos(yaw), 0.1 * math.sin(yaw), 1.0), orientation)
    tool0 = Pose((0.3, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0))
    delta = v_sub(clutch_relative(now, clutch, heading, tool0).p, tool0.p)
    assert delta == pytest.approx((0.1, 0.0, 0.0), abs=1e-12)


def test_pointing_too_close_to_vertical_is_refused():
    def pitched(deg):
        return q_from_axis_angle((0.0, 1.0, 0.0), math.radians(deg))
    assert heading_from_pointing(pitched(-59.0), X, 0.5) is not None
    assert heading_from_pointing(pitched(-61.0), X, 0.5) is None
    assert heading_from_pointing(pitched(90.0), X, 0.5) is None
    assert calibrate(pitched(90.0), X, 0.5, 1, 2) is None


def test_other_pointing_axis():
    assert heading_from_pointing((0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 0.5) == pytest.approx(
        -math.pi / 2.0)


def test_calibration_is_bound_to_session_and_reset_count():
    cal = calibrate(q_from_yaw(0.3), X, 0.5, 42, 3)
    assert cal == Calibration(pytest.approx(-0.3), 42, 3)
    assert cal.valid_for(42, 3)
    assert not cal.valid_for(42, 4)
    assert not cal.valid_for(43, 3)


@pytest.mark.parametrize('name', REQUIRED)
def test_each_pending_value_is_required_without_default(name):
    values = dict(VALUES)
    del values[name]
    with pytest.raises(MissingValue, match=name):
        build_params(values)
    with pytest.raises(MissingValue, match=name):
        build_params({**VALUES, name: None})


def test_all_missing_values_are_named():
    with pytest.raises(MissingValue) as info:
        build_params({})
    for name in REQUIRED:
        assert name in str(info.value)


@pytest.mark.parametrize('name', REQUIRED)
@pytest.mark.parametrize('bad', [0.0, -1.0, float('nan'), float('inf'), 'fast', True])
def test_required_values_must_be_positive_numbers(name, bad):
    with pytest.raises(ValueError):
        build_params({**VALUES, name: bad})


def test_defaults_and_units():
    without = {k: v for k, v in VALUES.items() if k != 'motion_enabled'}
    p = build_params(without)
    assert p.motion_enabled is False            # G-1: opt-in
    assert p.input_timeout == 0.1 and p.target_period == pytest.approx(0.02)
    assert p.max_offset_rotation == pytest.approx(math.radians(60.0))
    assert p.view_cone_angle == pytest.approx(math.radians(45.0))
    assert p.raw_speed_limit == pytest.approx(1.5)
    assert p.raw_rate_limit == pytest.approx(math.radians(500.0))
    assert p.keep_out_planes == () and p.pivot_offset == (0.0, 0.0, 0.0)
    with pytest.raises(Exception):
        p.max_offset_translation = 1.0          # frozen: no runtime widening (G-7)


def test_keep_out_planes_are_normalised_and_validated():
    (plane,) = parse_planes([0.0, 0.0, 2.0, 0.4])
    assert plane.n == (0.0, 0.0, 1.0) and plane.d == pytest.approx(0.2)
    for bad in ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, float('nan')]):
        with pytest.raises(ValueError):
            parse_planes(bad)
