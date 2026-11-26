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
"""Gate: the hold while the keep-out tables are built (G-15), and the feedforward's rotation."""

from dataclasses import replace
import math

from helpers import frame, HAND, moved, params, ready_state, TOOL_START
import pytest
from ranger_xarm_teleop_servo.gate import Env, Hold, rotation_low_pass, tick
from ranger_xarm_teleop_servo.rotations import Pose, q_from_axis_angle, q_mul


def _env(now, **kw):
    base = {'now': now, 'owner_is_teleop': True, 'tool_pose': TOOL_START, 'tool_pose_age': 0.0,
            'servo_active': True}
    base.update(kw)
    return Env(**base)


def _hold(state, gate_params, **env):
    from ranger_xarm_teleop_servo.gate import on_frame
    state = on_frame(state, frame(0), gate_params)
    _, out = tick(state, _env(0.0, **env), gate_params)
    return out.hold, out.target


def test_while_the_tables_are_built_nothing_is_published_and_the_hold_says_why():
    hold, target = _hold(ready_state(), params(), keepout_building=True)
    assert hold == Hold.KEEPOUT_BUILDING and target is None
    # Before NOT_ARMED: the panel says why arming cannot work yet.
    unarmed = replace(ready_state(), want_armed=False)
    assert _hold(unarmed, params(), keepout_building=True)[0] == Hold.KEEPOUT_BUILDING
    # Motion disabled stays first (the preview stack).
    assert _hold(ready_state(), params(motion_enabled=False),
                 keepout_building=True)[0] == Hold.MOTION_DISABLED
    assert _hold(ready_state(), params(), keepout_building=False)[0] != Hold.KEEPOUT_BUILDING


def test_the_rotation_low_pass_is_first_order_and_off_at_zero():
    off = replace(params(), feedforward_rotation_cutoff_hz=0.0)
    assert rotation_low_pass((0.0, 0.0, 0.0), (1.0, -2.0, 0.5), off) == (1.0, -2.0, 0.5)
    on = replace(params(), feedforward_rotation_cutoff_hz=2.0)
    alpha = 1.0 - math.exp(-2.0 * math.pi * 2.0 * on.target_period)
    value = (0.0, 0.0, 0.0)
    for _ in range(3):
        value = rotation_low_pass(value, (1.0, 0.0, 0.0), on)
    assert value[0] == pytest.approx(1.0 - (1.0 - alpha) ** 3)
    for _ in range(200):
        value = rotation_low_pass(value, (1.0, 0.0, 0.0), on)
    assert value[0] == pytest.approx(1.0, abs=1e-9)        # no steady loss of the rate


def _feedforwards(gate_params, motion):
    from ranger_xarm_teleop_servo.gate import on_frame
    state = ready_state()
    out_ff = []
    for k in range(40):
        t = k * gate_params.target_period
        state = on_frame(state, frame(int(round(t * 90.0)), controller=motion(t)), gate_params)
        state, out = tick(state, _env(t), gate_params)
        if out.feedforward is not None:
            out_ff.append(out.feedforward)
    return out_ff


def _turning(t):
    turn = q_from_axis_angle((0.0, 0.0, 1.0), 0.3 * t)          # 0.3 rad/s about up
    hand = moved(HAND, dx=0.05 * t)                              # and 50 mm/s along x
    return Pose(hand.p, q_mul(turn, hand.q))


def test_the_feedforward_rotation_is_smoothed_and_its_translation_untouched():
    raw = _feedforwards(replace(params(filter_cutoff_hz=0.0), feedforward_rotation_cutoff_hz=0.0),
                        _turning)
    smooth = _feedforwards(replace(params(filter_cutoff_hz=0.0),
                                   feedforward_rotation_cutoff_hz=2.0), _turning)
    assert len(raw) == len(smooth) > 10
    for a, b in zip(raw, smooth):
        assert a[:3] == pytest.approx(b[:3], abs=1e-12)        # translation as before
    # The step at engagement is followed with a lag, then the same rate.
    assert abs(smooth[1][5]) < abs(raw[1][5])
    assert smooth[-1][5] == pytest.approx(raw[-1][5], rel=0.05)
