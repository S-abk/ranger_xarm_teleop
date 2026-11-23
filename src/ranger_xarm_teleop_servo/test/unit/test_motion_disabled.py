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
Motion disabled (hardware profiles start so): arming is refused, and the preview.

INV-2 of the hardware task: arming is refused at the press with an actionable reason. The
preview (contract, "Arm ownership") computes the target an engagement would command while the
grip is held, only with motion disabled, only with a valid calibration and fresh joint states.
"""

from dataclasses import replace

from helpers import frame, HAND, moved, params, TOOL_START
from ranger_xarm_teleop_servo import preview
from ranger_xarm_teleop_servo.calibration import Calibration
from ranger_xarm_teleop_servo.gate import (_plane_depth, ARM_REFUSED_MOTION_DISABLED, Env,
                                           GateState, Hold, on_frame, tick)
from ranger_xarm_teleop_servo.params import Plane
from ranger_xarm_teleop_servo.rotations import v_norm, v_sub

CAL = Calibration(0.0, 7, 0)


def drive(frames, gate_params, state=None, tool=TOOL_START, tool_age=0.0, rate=50.0):
    """Feed frames in arrival order and tick at ``rate``; return the per-tick results."""
    state = state if state is not None else GateState()
    pv = preview.PreviewState()
    results = []
    i = 0
    t = 0.0
    end = frames[-1].arrival
    while t <= end + 1e-9:
        while i < len(frames) and frames[i].arrival <= t + 1e-12:
            state = on_frame(state, frames[i], gate_params)
            i += 1
        env = Env(now=t, owner_is_teleop=True, tool_pose=tool, tool_pose_age=tool_age,
                  servo_active=True)
        state, out = tick(state, env, gate_params)
        pv, target, why = preview.step(pv, state, env, gate_params)
        results.append((t, state, out, target, why))
        t += 1.0 / rate
    return results


def press_arm(n=90):
    """Return a clean stream with the arm button pressed on frames 10-20, grip released."""
    return [frame(k, grip=False, arm=10 <= k <= 20) for k in range(n)]


def test_arming_is_refused_at_the_press_while_motion_is_disabled():
    results = drive(press_arm(), params(motion_enabled=False))
    events = [e for _, _, out, _, _ in results for e in out.events]
    assert ARM_REFUSED_MOTION_DISABLED in events
    assert 'arm requested' not in events
    assert all(not s.want_armed for _, s, _, _, _ in results)
    assert all(out.hold == Hold.MOTION_DISABLED for _, _, out, _, _ in results)


def test_the_refusal_says_what_to_do():
    text = ARM_REFUSED_MOTION_DISABLED
    assert text.startswith('arm refused: ') and 'motion_enabled:=true' in text
    assert '\n' not in text


def test_with_motion_enabled_the_press_still_arms():
    results = drive(press_arm(), params(motion_enabled=True))
    events = [e for _, _, out, _, _ in results for e in out.events]
    assert 'arm requested' in events and ARM_REFUSED_MOTION_DISABLED not in events


def stroke(n=180, dx=0.10, grip=lambda k: True, **fields):
    """Hand moves dx metres along room +x over the stream, grip held."""
    return [frame(k, controller=moved(HAND, dx=dx * k / n), grip=grip(k), **fields)
            for k in range(n)]


def test_the_preview_follows_the_hand_with_motion_disabled():
    results = drive(stroke(), params(motion_enabled=False), state=GateState(calibration=CAL))
    shown = [(t, target) for t, _, _, target, why in results if target is not None]
    assert len(shown) > 80
    first, last = shown[0][1], shown[-1][1]
    assert v_norm(v_sub(first.p, TOOL_START.p)) < 1e-6        # starts at the measured pose
    assert last.p[0] - first.p[0] > 0.05                        # follows room +x (heading 0)
    assert all(out.target is None for _, _, out, _, _ in results)   # never a real target


def test_no_preview_with_motion_enabled():
    results = drive(stroke(), params(motion_enabled=True),
                    state=GateState(calibration=CAL, want_armed=True))
    assert all(target is None for *_, target, _ in results)
    assert {why for *_, why in results} == {preview.NOT_DISABLED}


def test_no_preview_without_a_calibration_and_it_says_so():
    results = drive(stroke(), params(motion_enabled=False))
    assert all(target is None for *_, target, _ in results)
    assert results[-1][4] == preview.NO_CALIBRATION


def test_no_preview_with_a_calibration_from_another_session():
    results = drive(stroke(), params(motion_enabled=False),
                    state=GateState(calibration=Calibration(0.0, 8, 0)))
    assert all(target is None for *_, target, _ in results)
    assert results[-1][4] == preview.NO_CALIBRATION


def test_no_preview_without_fresh_joint_states_and_it_says_so():
    stale = drive(stroke(), params(motion_enabled=False), state=GateState(calibration=CAL),
                  tool_age=1.0)
    assert all(target is None for *_, target, _ in stale)
    assert stale[-1][4] == preview.NO_JOINT_STATE
    unknown = drive(stroke(), params(motion_enabled=False), state=GateState(calibration=CAL),
                    tool=None)
    assert all(target is None for *_, target, _ in unknown)
    assert unknown[-1][4] == preview.NO_JOINT_STATE


def test_no_preview_with_the_grip_released():
    results = drive(stroke(grip=lambda k: False), params(motion_enabled=False),
                    state=GateState(calibration=CAL))
    assert all(target is None for *_, target, _ in results)
    assert results[-1][4] == preview.GRIP_RELEASED


def test_the_preview_is_bounded_by_the_offset_limit():
    p = params(motion_enabled=False)
    results = drive(stroke(n=270, dx=1.0), replace(p, max_offset_translation=0.10),
                    state=GateState(calibration=CAL))
    shown = [target for *_, target, _ in results if target is not None]
    assert shown
    assert max(v_norm(v_sub(t.p, TOOL_START.p)) for t in shown) <= 0.10 + 1e-9


def test_the_preview_stops_at_a_keep_out_plane():
    """The preview is bounded by keep_out_planes as an armed target is: tool and wrist point."""
    p = params(motion_enabled=False)
    wall = Plane((-1.0, 0.0, 0.0), -(TOOL_START.p[0] + 0.05))     # allowed: x <= start + 5 cm
    floor = Plane((0.0, 0.0, 1.0), TOOL_START.p[2] - 0.30)       # far below: never touched
    bounded = replace(p, keep_out_planes=(wall, floor))
    results = drive(stroke(n=270, dx=0.30), bounded, state=GateState(calibration=CAL))
    shown = [target for *_, target, _ in results if target is not None]
    free = [target for *_, target, _ in drive(stroke(n=270, dx=0.30), p,
                                              state=GateState(calibration=CAL))
            if target is not None]
    assert shown and free
    assert max(t.p[0] for t in free) > TOOL_START.p[0] + 0.10     # the hand goes beyond it
    for plane in bounded.keep_out_planes:
        assert max(_plane_depth(t, plane, bounded) for t in shown) <= 1e-9
    # It reaches the plane and stays on it while the hand goes on.
    assert abs(_plane_depth(shown[-1], wall, bounded)) < 1e-9
    assert all(abs(t.p[1] - shown[0].p[1]) < 1e-9 and abs(t.p[2] - shown[0].p[2]) < 1e-9
               for t in shown)


def test_the_wrist_point_stops_the_preview_first_when_it_reaches_a_plane_first():
    """With a wrist point 4 cm ahead of the tool point, the tool stops 4 cm short of the plane."""
    p = replace(params(motion_enabled=False), wrist_offset=(0.04, 0.0, 0.0))
    wall = Plane((-1.0, 0.0, 0.0), -(TOOL_START.p[0] + 0.05))
    bounded = replace(p, keep_out_planes=(wall,))
    results = drive(stroke(n=270, dx=0.30), bounded, state=GateState(calibration=CAL))
    shown = [target for *_, target, _ in results if target is not None]
    assert shown
    assert max(t.p[0] for t in shown) <= TOOL_START.p[0] + 0.01 + 1e-9
    assert abs(shown[-1].p[0] - (TOOL_START.p[0] + 0.01)) < 1e-9


def test_a_jump_ends_the_preview_until_the_grip_is_released():
    p = params(motion_enabled=False)
    frames = stroke(n=90, dx=0.0)
    jumped = [replace(f, controller=moved(HAND, dx=0.5)) if f.arrival > 0.5 else f
              for f in frames]
    results = drive(jumped, p, state=GateState(calibration=CAL))
    before = [target for t, _, _, target, _ in results if t < 0.5]
    after = [(target, why) for t, _, _, target, why in results if t > 0.55]
    assert before and all(target is not None for target in before)
    assert after and all(target is None for target, _ in after)
    assert after[-1][1] == preview.STEP
