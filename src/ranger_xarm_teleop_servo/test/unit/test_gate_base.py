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
"""The arm's clutch and the base (G-19, INV-2): refused, waited for, released; off, nothing."""

from dataclasses import replace

from helpers import frame, params, ready_state, TOOL_START
import pytest
from ranger_xarm_teleop_servo.base_rest import AT_REST, MOVING, UNKNOWN
from ranger_xarm_teleop_servo.gate import Env, Hold, on_frame, tick


def base_params(**kw):
    """Return the test's gate parameters with base driving on."""
    return replace(params(), base_driving=True, **kw)


class Rig:
    """The gate with frames at 50 Hz, one tick per frame, an ideal arm and a base."""

    def __init__(self, gate_params):
        self.p = gate_params
        self.state = ready_state()
        self.k = 0
        self.motion = AT_REST
        self.turned = False
        self.out = None

    def step(self, grip=True, left_grip=False, **frame_kw):
        self.k += 1
        f = frame(self.k, rate=50.0, grip=grip, left_grip=left_grip, **frame_kw)
        self.state = on_frame(self.state, f, self.p)
        env = Env(now=f.arrival, owner_is_teleop=True, tool_pose=TOOL_START, tool_pose_age=0.0,
                  servo_active=True, base_motion=self.motion, base_turned=self.turned)
        self.state, self.out = tick(self.state, env, self.p)
        return self.out

    def run(self, n, **kw):
        return [self.step(**kw) for _ in range(n)]


def test_with_base_driving_off_nothing_of_the_base_counts():
    """INV-7: the left grip, odometry unknown, moving or a turn change nothing."""
    plain = Rig(params())
    other = Rig(params())
    other.motion, other.turned = UNKNOWN, True
    a = plain.run(30) + plain.run(5, grip=False) + plain.run(20)
    b = other.run(30, left_grip=True) + other.run(5, grip=False, left_grip=True) + other.run(20)
    assert [(o.hold, o.engaged, o.target) for o in a] == [(o.hold, o.engaged, o.target)
                                                          for o in b]
    assert any(o.engaged for o in b)


def test_the_left_grip_refuses_the_clutch_and_needs_a_new_right_press():
    r = Rig(base_params())
    r.step(grip=False)
    out = r.step(grip=True, left_grip=True)
    assert out.hold == Hold.BASE_DRIVE and not out.engaged and out.target is None
    out = r.run(5, grip=True, left_grip=False)[-1]
    assert out.hold == Hold.BASE_DRIVE and not out.engaged, 'latched: a held grip waits'
    r.step(grip=False)
    assert r.step(grip=True).engaged


def test_both_grips_in_one_frame_the_clutch_is_refused():
    r = Rig(base_params())
    r.step(grip=False)
    out = r.step(grip=True, left_grip=True)
    assert out.hold == Hold.BASE_DRIVE and not out.engaged


def test_first_holder_wins_the_left_grip_does_not_release_an_engaged_clutch():
    r = Rig(base_params())
    r.step(grip=False)
    assert r.step(grip=True).engaged
    outs = r.run(10, grip=True, left_grip=True)
    assert all(o.engaged for o in outs)


def test_a_base_still_moving_is_waited_for_then_the_clutch_engages():
    r = Rig(base_params())
    r.motion = MOVING
    r.step(grip=False)
    outs = r.run(20, grip=True)
    assert all(o.hold == Hold.BASE_MOVING and not o.engaged for o in outs)
    r.motion = AT_REST
    assert r.step(grip=True).engaged


def test_a_base_that_does_not_come_to_rest_refuses_the_clutch():
    r = Rig(base_params(base_rest_wait_s=0.5))
    r.motion = MOVING
    r.step(grip=False)
    r.run(30, grip=True)                                    # 0.6 s
    r.motion = AT_REST
    out = r.step(grip=True)
    assert out.hold == Hold.BASE_MOVING and not out.engaged, 'latched after the wait'
    r.step(grip=False)
    assert r.step(grip=True).engaged


def test_base_motion_while_engaged_releases_the_clutch():
    r = Rig(base_params())
    r.step(grip=False)
    assert r.step(grip=True).engaged
    r.motion = MOVING
    out = r.step(grip=True)
    assert out.hold == Hold.BASE_MOVING and not out.engaged and out.target is None


def test_no_odometry_refuses_and_releases():
    r = Rig(base_params())
    r.motion = UNKNOWN
    r.step(grip=False)
    assert r.step(grip=True).hold == Hold.BASE_ODOM
    r.motion = AT_REST
    r.step(grip=False)
    assert r.step(grip=True).engaged
    r.motion = UNKNOWN
    out = r.step(grip=True)
    assert out.hold == Hold.BASE_ODOM and not out.engaged


def test_a_turn_past_the_limit_refuses_until_calibrated_again():
    r = Rig(base_params())
    r.turned = True
    r.step(grip=False)
    out = r.step(grip=True)
    assert out.hold == Hold.BASE_TURNED and not out.engaged


@pytest.mark.parametrize('hold', [Hold.BASE_DRIVE, Hold.BASE_MOVING, Hold.BASE_ODOM,
                                  Hold.BASE_TURNED])
def test_the_hold_values_follow_the_message(hold):
    import os
    import re
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'ranger_xarm_teleop_interfaces', 'msg', 'TeleopState.msg')
    text = open(path, encoding='utf-8').read()
    assert re.search(rf'uint8 HOLD_{hold.name} = {int(hold)}\b', text)


def test_pose_conditions_still_hold_the_arm_with_the_base_on():
    """The base adds conditions; it removes none of the arm's own."""
    r = Rig(base_params())
    r.step(grip=False)
    out = r.step(grip=True, controller=None, controller_present=False)
    assert out.hold == Hold.POSE_ABSENT
