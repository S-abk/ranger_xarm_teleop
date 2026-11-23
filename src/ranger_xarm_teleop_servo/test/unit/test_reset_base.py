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
"""The reset with base driving on (G-19): refused while the left grip is held; no turn pulls."""

from types import SimpleNamespace as NS

import pytest
from ranger_xarm_teleop_servo import reset as rs
from test_reset import Rig

STICK_X = rs.STICK_X_AXIS


def joy(stick=0.0, x=0.0, left_grip=False):
    axes = [0.0] * 8
    axes[rs.STICK_AXIS], axes[STICK_X] = stick, x
    buttons = [0] * 10
    buttons[rs.LEFT_GRIP_BUTTON] = int(left_grip)
    return NS(axes=axes, buttons=buttons)


def press(rig, stick=1.0, x=0.0, left_grip=False, seconds=0.6):
    rig.hold(0.1, 0.0)                              # released first
    goal = None
    for _ in range(int(round(seconds / 0.02))):
        goal = rig.step(stick, joy_msg=joy(stick, x, left_grip)) or goal
    return goal


def test_the_sample_carries_the_left_grip_and_the_sticks_x():
    status = NS(session_id=1, seq=1, xr_time_ms=1000.0)
    s = rs.sample_of(status, joy(0.9, -0.4, True), 1.0)
    assert s.left_grip and s.stick_x == pytest.approx(-0.4) and s.stick == pytest.approx(0.9)


def test_with_base_driving_on_the_left_grip_refuses_the_reset():
    rig = Rig(base_driving=True)
    goal = press(rig, left_grip=True)
    assert goal is None and not rig.sent
    assert rig.r.state == rs.REFUSED and rig.r.reason == rs.BASE_DRIVE, rig.r.reason_text()


def test_with_base_driving_on_a_turn_is_not_a_pull():
    rig = Rig(base_driving=True)
    assert press(rig, stick=0.85, x=0.9) is None and not rig.sent
    assert rig.r.state == rs.IDLE
    assert press(rig, stick=1.0, x=0.3) is not None       # the pull dominates: a reset


def test_with_base_driving_off_the_reset_is_as_it_was():
    """INV-7: the left grip and the stick's x change nothing."""
    rig = Rig()
    assert press(rig, stick=0.85, x=0.9, left_grip=True) is not None
