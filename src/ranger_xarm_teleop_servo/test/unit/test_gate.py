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
"""Safety gate on hand-built frame streams: INV-2, INV-3, INV-5, INV-6, INV-8, INV-9."""

from dataclasses import replace
import math

from helpers import (engaged_spans, frame, HAND, HEAD, hold_at, max_step, moved, params,
                     PERIOD, ready_state, run, same_pose, stream, targets, TOOL_START)
import numpy as np
import pytest
from ranger_xarm_teleop_servo.calibration import Calibration
from ranger_xarm_teleop_servo.gate import (Env, GateState, Hold, Limit, on_frame, tick,
                                           view_angle, within_bounds)
from ranger_xarm_teleop_servo.rotations import (Pose, q_between, q_from_axis_angle, q_from_yaw,
                                                q_mul, q_rotate, v_add, v_dot, v_norm, v_sub)


def still(k):
    return HAND


def forward(speed):
    """Return a motion along room +x at ``speed`` m/s (90 frames per second)."""
    return lambda k: moved(HAND, dx=speed * k / 90.0)


# ---- G-1, G-2 (INV-2) ----

def test_no_target_unless_motion_enabled_armed_and_owner():
    frames = stream(90, forward(0.1))
    assert targets(run(frames, state=ready_state()))
    disabled = run(frames, params(motion_enabled=False), state=ready_state())
    assert not targets(disabled) and hold_at(disabled, 0.5) == Hold.MOTION_DISABLED
    unarmed = run(frames, state=replace(ready_state(), want_armed=False))
    assert not targets(unarmed) and hold_at(unarmed, 0.5) == Hold.NOT_ARMED
    not_owner = run(frames, state=ready_state(), owner_is_teleop=False)
    assert not targets(not_owner) and hold_at(not_owner, 0.5) == Hold.NOT_OWNER


def test_motion_enabled_defaults_to_false():
    from ranger_xarm_teleop_servo.params import build_params
    from helpers import VALUES
    values = {k: v for k, v in VALUES.items() if k != 'motion_enabled'}
    steps = run(stream(90, forward(0.1)), build_params(values), state=ready_state())
    assert not targets(steps)


def test_no_target_without_the_clutch_and_release_holds_at_once():
    frames = stream(180, forward(0.1), grip=lambda k: 45 <= k < 120)
    steps = run(frames, state=ready_state())
    (span,) = engaged_spans(steps)
    press, release = 45 / 90.0, 120 / 90.0
    assert press <= span[0] <= press + 3 * PERIOD      # one tick for Servo to confirm
    # The first tick that has seen the released grip publishes nothing.
    assert release - PERIOD <= span[1] < release
    for step in steps:
        if release <= step.t <= frames[-1].arrival:
            assert step.output.target is None and not step.output.want_servo_active
            assert step.output.hold == Hold.CLUTCH_RELEASED
    # The first target is the measured tool pose: engagement itself moves nothing.
    assert targets(steps)[0][1] == TOOL_START


def test_target_follows_the_hand_while_engaged():
    steps = run(stream(180, forward(0.1)), params(filter_cutoff_hz=0.0), state=ready_state())
    first_t = targets(steps)[0][0]
    t, last = [(t, pose) for t, pose in targets(steps) if t <= 1.5][-1]
    travelled = v_sub(last.p, TOOL_START.p)
    assert travelled[0] == pytest.approx(0.1 * (t - first_t), abs=0.004)
    assert abs(travelled[1]) < 1e-9 and abs(travelled[2]) < 1e-9
    assert q_between(last.q, TOOL_START.q) < 1e-9


# ---- G-3, G-11 (INV-3) ----

def test_stalled_input_releases_after_the_timeout_measured_by_arrival():
    frames = stream(90, forward(0.1))          # stream stops at 1.0 s with the grip held
    steps = run(frames, state=ready_state(), tail=1.0)
    last_arrival = frames[-1].arrival
    last_target_t = targets(steps)[-1][0]
    assert last_arrival <= last_target_t <= last_arrival + 0.1
    for step in steps:
        assert step.output.input_age == pytest.approx(
            step.t - max(f.arrival for f in frames if f.arrival <= step.t), abs=1e-9)
        if step.t > last_arrival + 0.1:
            assert step.output.target is None
            assert step.output.hold == Hold.INPUT_STALE
            assert step.output.input_age > 0.1


def test_age_uses_arrival_time_not_stamps_or_headset_time():
    """Frames whose headset time claims to be recent but which arrived long ago are stale."""
    frames = [replace(f, xr_time=f.xr_time + 1000.0) for f in stream(45, still)]
    steps = run(frames, state=ready_state(), tail=0.5)
    assert hold_at(steps, 0.9) == Hold.INPUT_STALE


def test_stream_resuming_with_the_grip_still_held_does_not_re_engage():
    frames = [f for f in stream(270, forward(0.05)) if not 90 <= f.seq - 1000 < 135]
    steps = run(frames, state=ready_state())
    spans = engaged_spans(steps)
    assert len(spans) == 1 and spans[0][1] < 1.12
    assert hold_at(steps, 1.3) == Hold.INPUT_STALE
    assert hold_at(steps, 2.5) == Hold.INPUT_STALE      # latched: the grip was never released


def test_no_frames_at_all():
    state, output = tick(GateState(want_armed=True), Env(1.0, True, TOOL_START, 0.0, True),
                         params())
    assert output.target is None and output.hold == Hold.INPUT_STALE
    assert math.isinf(output.input_age)


# ---- G-5 (INV-5) ----

@pytest.mark.parametrize('cutoff', [0.0, 2.0, 8.0])
def test_290_mm_input_jump_releases_even_with_the_filter_on(cutoff):
    def motion(k):
        return moved(HAND, dx=0.05 * k / 90.0, dy=0.29 if k >= 100 else 0.0)
    steps = run(stream(200, motion), params(filter_cutoff_hz=cutoff), state=ready_state())
    jump_t = 100 / 90.0
    (span,) = engaged_spans(steps)
    assert span[1] < jump_t
    assert hold_at(steps, jump_t + PERIOD) == Hold.INPUT_JUMP
    assert hold_at(steps, 2.0) == Hold.INPUT_JUMP           # latched while the grip is held
    # No published target moved toward the jump, not even a filtered fraction of it.
    assert max(abs(pose.p[1] - TOOL_START.p[1]) for _, pose in targets(steps)) < 1e-9
    assert max_step(steps)[0] < 0.002


def test_raw_check_uses_headset_time_between_consecutive_frames():
    """The same 20 mm step is fine over 100 ms of headset time and a jump over 11 ms."""
    def frames(gap):
        out = stream(60, still)
        late = [replace(f, xr_time=f.xr_time + gap, controller=moved(HAND, dy=0.02))
                for f in stream(120, still)[60:]]
        return out + late
    slow = run(frames(0.1), params(filter_cutoff_hz=0.0), state=ready_state())
    assert len(engaged_spans(slow)) == 1 and targets(slow)[-1][0] > 1.2
    fast = run(frames(0.0), params(filter_cutoff_hz=0.0), state=ready_state())
    assert hold_at(fast, 1.0) == Hold.INPUT_JUMP


def test_fast_rotation_of_the_raw_input_releases():
    def motion(k):
        return Pose(HAND.p, q_from_yaw(math.radians(40.0) if k >= 60 else 0.0))
    steps = run(stream(120, motion), state=ready_state())
    assert hold_at(steps, 1.0) == Hold.INPUT_JUMP
    assert max_step(steps)[1] < math.radians(0.1)


def test_final_pose_step_check_is_a_second_layer():
    """A step that passes the raw speed check is still rejected on the final target."""
    # 1.4 m/s is under the raw limit (1.5 m/s); after an 80 ms gap in arrivals (under
    # the timeout) the target would step 112 mm in one cycle.
    def late(k):
        return 0.08 if k >= 60 else 0.0
    frames = [replace(f, arrival=f.arrival + late(k), xr_time=f.xr_time + late(k),
                      controller=moved(HAND, dx=1.4 * late(k)))
              for k, f in enumerate(stream(120, still))]
    steps = run(frames, params(filter_cutoff_hz=0.0), state=ready_state())
    (span,) = engaged_spans(steps)
    assert hold_at(steps, 1.0) == Hold.STEP_REJECTED
    assert max_step(steps)[0] <= 0.03
    assert max(abs(pose.p[0] - TOOL_START.p[0]) for _, pose in targets(steps)) < 1e-9


def test_steps_at_the_limit_pass_and_are_never_clipped():
    """Motion just under the limit is followed exactly; nothing is slowed down."""
    steps = run(stream(90, forward(1.2)), params(filter_cutoff_hz=0.0,
                                                 max_offset_translation_m=5.0),
                state=ready_state())
    (span,) = engaged_spans(steps)
    assert span[1] > 0.95
    moving = [pose for t, pose in targets(steps) if t < 0.95]      # before the stream ends
    moves = [v_norm(v_sub(b.p, a.p)) for a, b in zip(moving, moving[1:])]
    # A tick sees one or two new 90 Hz frames: 13.3 mm or 26.7 mm, both passed unchanged.
    assert max(moves) == pytest.approx(2 * 1.2 / 90.0, abs=1e-9)
    assert min(moves[1:]) == pytest.approx(1.2 / 90.0, abs=1e-9)


# ---- G-6, G-7 (INV-6) ----

@pytest.mark.parametrize('cutoff', [0.0, 8.0])
def test_offset_limit_holds_after_filtering(cutoff):
    p = params(filter_cutoff_hz=cutoff)
    steps = run(stream(450, forward(0.2)), p, state=ready_state())       # 1.0 m of hand travel
    poses = [pose for _, pose in targets(steps)]
    assert len(poses) > 200
    distances = [v_norm(v_sub(pose.p, TOOL_START.p)) for pose in poses]
    assert max(distances) <= 0.30 + 1e-9
    assert distances[-1] == pytest.approx(0.30, abs=1e-6)
    assert steps[-20].output.limit == Limit.OFFSET_TRANSLATION
    assert steps[40].output.limit == Limit.NONE


@pytest.mark.parametrize('cutoff', [0.0, 8.0])
def test_rotation_offset_limit_holds_after_filtering(cutoff):
    def motion(k):
        return Pose(HAND.p, q_from_yaw(math.radians(min(120.0, 0.5 * k))))
    steps = run(stream(400, motion), params(filter_cutoff_hz=cutoff), state=ready_state())
    angles = [q_between(TOOL_START.q, pose.q) for _, pose in targets(steps)]
    assert max(angles) <= math.radians(60.0) + 1e-6
    assert angles[-1] == pytest.approx(math.radians(60.0), abs=1e-6)
    assert steps[-20].output.limit == Limit.OFFSET_ROTATION


@pytest.mark.parametrize('cutoff', [0.0, 8.0])
def test_keep_out_plane_is_not_crossed_by_the_tool_or_the_wrist(cutoff):
    # Allowed: x <= 0.40 (n = -x, d = -0.40). The wrist point is 50 mm ahead of the tool.
    p = params(filter_cutoff_hz=cutoff, keep_out_planes=[-1.0, 0.0, 0.0, -0.40],
               wrist_offset=[0.05, 0.0, 0.0])
    steps = run(stream(270, forward(0.1)), p, state=ready_state())
    poses = [pose for _, pose in targets(steps)]
    assert len(poses) > 100
    for pose in poses:
        wrist = v_add(pose.p, q_rotate(pose.q, p.wrist_offset))
        assert pose.p[0] <= 0.40 + 1e-9 and wrist[0] <= 0.40 + 1e-9
    assert poses[-1].p[0] == pytest.approx(0.35, abs=1e-6)       # wrist on the plane
    assert steps[-20].output.limit == Limit.KEEP_OUT


def test_wrist_point_rotating_into_a_plane_pushes_the_tool_back():
    # Allowed: y <= 0.03. The wrist is 50 mm ahead of the tool; a yaw swings it into +y.
    p = params(filter_cutoff_hz=0.0, keep_out_planes=[0.0, -1.0, 0.0, -0.03],
               wrist_offset=[0.05, 0.0, 0.0])

    def motion(k):
        return Pose(HAND.p, q_from_yaw(math.radians(min(60.0, 0.5 * k))))
    steps = run(stream(200, motion), p, state=ready_state())
    for _, pose in targets(steps):
        wrist = v_add(pose.p, q_rotate(pose.q, p.wrist_offset))
        assert wrist[1] <= 0.03 + 1e-9
    assert targets(steps)[-1][1].p[1] < -0.01
    assert steps[-20].output.limit == Limit.KEEP_OUT


def test_bounds_hold_for_random_motion_through_every_layer():
    rng = np.random.default_rng(7)
    p = params(filter_cutoff_hz=5.0, keep_out_planes=[0.0, 0.0, 1.0, 0.36, -1.0, 0.0, 0.0, -0.34],
               wrist_offset=[0.0, 0.0, 0.06], max_offset_translation_m=0.2,
               max_offset_rotation_deg=30.0)
    position = np.array(HAND.p)
    velocity = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    frames = []
    for k in range(1800):
        velocity = 0.98 * velocity + rng.normal(scale=0.03, size=3)
        velocity *= min(1.0, 1.0 / (np.linalg.norm(velocity) + 1e-9))     # at most 1 m/s
        position = position + velocity / 90.0
        axis = axis + rng.normal(scale=0.02, size=3)
        angle = 1.2 * math.sin(k / 150.0)
        frames.append(frame(k, controller=Pose(tuple(position), q_from_axis_angle(axis, angle)),
                            head=Pose(HEAD.p, HEAD.q)))
    p_wide = replace(p, view_cone_angle=math.pi)
    steps = run(frames, p_wide, state=ready_state())
    poses = [pose for _, pose in targets(steps)]
    assert len(poses) > 500
    limits = {s.output.limit for s in steps}
    assert {Limit.OFFSET_TRANSLATION, Limit.KEEP_OUT} <= limits
    for pose in poses:
        assert within_bounds(pose, TOOL_START, p_wide)
        assert v_norm(v_sub(pose.p, TOOL_START.p)) <= 0.2 + 1e-9
        assert q_between(TOOL_START.q, pose.q) <= math.radians(30.0) + 1e-6
        for plane in p_wide.keep_out_planes:
            wrist = v_add(pose.p, q_rotate(pose.q, p_wide.wrist_offset))
            assert v_dot(plane.n, pose.p) >= plane.d - 1e-9
            assert v_dot(plane.n, wrist) >= plane.d - 1e-9
    assert max_step(steps)[0] <= 0.03 + 1e-12


def test_engaging_inside_a_keep_out_region_never_moves_further_in():
    # Allowed: x <= 0.25, but the tool starts at x = 0.30.
    p = params(filter_cutoff_hz=0.0, keep_out_planes=[-1.0, 0.0, 0.0, -0.25])
    steps = run(stream(180, forward(0.05)), p, state=ready_state())
    assert all(pose.p[0] <= TOOL_START.p[0] + 1e-12 for _, pose in targets(steps))


# ---- G-10 (INV-8) ----

@pytest.mark.parametrize('cutoff', [0.0, 2.0, 8.0])
def test_re_engaging_after_a_move_does_not_jump(cutoff):
    def motion(k):
        # Forward during the first hold; the hand goes back and turns while released.
        if k < 180:
            return moved(HAND, dx=0.1 * k / 90.0)
        if k < 270:
            return Pose(moved(HAND, dx=0.2 - 0.25 * (k - 180) / 90.0).p, q_from_yaw(0.6))
        return Pose(moved(HAND, dx=-0.05 + 0.1 * (k - 270) / 90.0).p, q_from_yaw(0.6))
    frames = stream(450, motion, grip=lambda k: k < 180 or k >= 270)
    steps = run(frames, params(filter_cutoff_hz=cutoff), state=ready_state())
    first, second = engaged_spans(steps)
    by_time = dict(targets(steps))
    end_of_first = by_time[first[1]]
    start_of_second = by_time[second[0]]
    # The second engagement starts exactly at the measured tool pose (the ideal arm is
    # where the first one left it): no filter or limiter state carried over.
    assert start_of_second == end_of_first
    assert max_step(steps)[0] < 0.004
    assert max_step(steps)[1] < math.radians(0.1)
    # And it continues from there, relative to the new grip.
    final = targets(steps)[-1][1]
    assert final.p[0] - end_of_first.p[0] == pytest.approx(0.2 - 0.02, abs=0.02)
    assert q_between(final.q, TOOL_START.q) < 1e-9


def test_engagement_captures_the_measured_tool_pose_not_the_last_target():
    p = params()
    state = ready_state()
    for k in range(3):
        state = on_frame(state, frame(k), p)
    elsewhere = Pose((0.1, 0.2, 0.5), q_from_yaw(1.0))
    state, output = tick(state, Env(0.03, True, elsewhere, 0.0, True), p)
    assert output.engaged and output.target == elsewhere


# ---- G-13 (INV-9) ----

def test_no_target_without_a_calibration_for_this_session_and_reset_count():
    frames = stream(90, forward(0.1))
    for state in (GateState(want_armed=True),
                  ready_state(session_id=8),
                  ready_state(reset_count=1)):
        steps = run(frames, state=state)
        assert not targets(steps)
        assert hold_at(steps, 0.5) == Hold.NOT_CALIBRATED
    assert targets(run(frames, state=ready_state()))


def test_calibration_button_calibrates_and_reset_invalidates_until_redone():
    def fields(k):
        return {'calibrate': 20 <= k < 30 or 200 <= k < 210,
                'reset_count': 1 if k >= 150 else 0,
                'grip': 60 <= k < 120 or 240 <= k < 300}
    pointing = Pose(HAND.p, q_from_yaw(math.radians(25.0)))
    frames = [frame(k, controller=pointing, **fields(k)) for k in range(330)]
    steps = run(frames, state=GateState(want_armed=True))
    first, second = engaged_spans(steps)
    assert 60 / 90.0 <= first[0] and first[1] < 120 / 90.0
    assert 240 / 90.0 <= second[0]
    events = [e for s in steps for e in s.output.events]
    assert sum('calibrated: heading -25.0 deg' in e for e in events) == 2
    assert any('reset count changed 0 -> 1' in e for e in events)
    assert hold_at(steps, 1.9) == Hold.NOT_CALIBRATED       # after the reset, before the redo


def test_grip_held_without_calibration_then_calibrated_does_not_engage_until_regripped():
    frames = [frame(k, grip=30 <= k < 200, calibrate=100 <= k < 110) for k in range(200)]
    steps = run(frames, state=GateState(want_armed=True))
    assert not targets(steps)
    events = [e for s in steps for e in s.output.events]
    assert any('calibration ignored' in e for e in events)


@pytest.mark.parametrize('fields, text', [
    ({'controller_emulated': True}, 'POSE_EMULATED'),
    ({'controller': None}, None),
    ({'head_emulated': True}, 'HEAD_EMULATED'),
    ({'head': None, 'head_present': False}, 'HEAD_ABSENT'),
    ({'controller': Pose(HAND.p, q_from_axis_angle((0.0, 1.0, 0.0), 1.5))}, 'vertical'),
    ({'controller': Pose((-0.5, 0.0, 1.4), HAND.q)}, 'OUT_OF_VIEW'),
])
def test_calibration_is_refused_on_untrustworthy_input(fields, text):
    p = params()
    state = on_frame(GateState(), frame(0, grip=False), p)
    state = on_frame(state, frame(1, grip=False, calibrate=True, **fields), p)
    assert state.calibration is None
    if text is not None:
        assert any(text in e for e in state.events)


# ---- G-4 details not covered by the fixtures ----

def test_view_angle():
    assert view_angle(HEAD, (1.0, 0.0, 1.6)) == pytest.approx(0.0)
    assert view_angle(HEAD, (0.0, 1.0, 1.6)) == pytest.approx(math.pi / 2.0)
    assert view_angle(HEAD, (-1.0, 0.0, 1.6)) == pytest.approx(math.pi)
    turned = Pose(HEAD.p, q_from_yaw(math.radians(90.0)))
    assert view_angle(turned, (0.0, 1.0, 1.6)) == pytest.approx(0.0, abs=1e-7)
    assert view_angle(HEAD, HAND.p) == pytest.approx(math.radians(30.0), abs=math.radians(4.0))


def test_view_cone_boundary():
    def at(deg):
        a = math.radians(deg)
        return Pose((0.5 * math.cos(a), 0.5 * math.sin(a), 1.6), HAND.q)
    inside = run(stream(60, lambda k: at(44.0)), state=ready_state())
    outside = run(stream(60, lambda k: at(46.0)), state=ready_state())
    assert targets(inside) and not targets(outside)
    assert hold_at(outside, 0.5) == Hold.OUT_OF_VIEW


def test_missing_head_pose_counts_as_not_in_view():
    steps = run(stream(60, still, head=None), state=ready_state())
    assert not targets(steps) and hold_at(steps, 0.5) == Hold.HEAD_ABSENT


def test_duplicate_or_backwards_headset_time_releases():
    frames = stream(120, still)
    frames[60] = replace(frames[60], xr_time=frames[59].xr_time)
    steps = run(frames, state=ready_state())
    assert hold_at(steps, 1.0) == Hold.SEQ_BACKWARDS


# ---- arming, Servo readiness, tool pose ----

def test_arm_button_toggles_on_the_press_edge_only():
    p = params()
    state = GateState()
    pattern = [True, True, False, False, True, True, True, False, True]
    seen = []
    for k, pressed in enumerate(pattern):
        state = on_frame(state, frame(k, grip=False, arm=pressed), p)
        seen.append(state.want_armed)
    # Held on the first frame is not a press; then edges at index 4 and 8.
    assert seen == [False, False, False, False, True, True, True, True, False]


def test_disarming_while_engaged_releases_and_latches():
    frames = [frame(k, arm=90 <= k < 100 or 150 <= k < 160) for k in range(240)]
    steps = run(frames, state=ready_state())
    (span,) = engaged_spans(steps)
    assert span[1] < 1.0 + PERIOD
    assert hold_at(steps, 1.2) == Hold.NOT_ARMED
    assert hold_at(steps, 2.2) == Hold.NOT_ARMED      # re-armed, but the grip was never released


def test_buttons_of_an_absent_controller_are_not_presses_or_releases():
    """A dropout reads as all buttons up; coming back held is neither a press nor a re-grip."""
    def one(k):
        gone = 60 <= k < 90
        return frame(k, controller=None if gone else HAND, grip=not gone, arm=not gone)
    steps = run([one(k) for k in range(180)], state=ready_state())
    assert len(engaged_spans(steps)) == 1
    assert hold_at(steps, 1.5) == Hold.POSE_ABSENT
    assert not any('arm' in e for s in steps for e in s.output.events)


def test_servo_not_ready_never_engages_and_latches_after_the_timeout():
    p = params()
    steps = run(stream(120, still), p, state=ready_state(), servo_delay_ticks=10 ** 6)
    assert not targets(steps)
    assert all(s.output.want_servo_active for s in steps if 0.05 < s.t < 0.5)
    assert not any(s.output.want_servo_active for s in steps if s.t > 0.6)
    assert hold_at(steps, 1.0) == Hold.SERVO_NOT_READY
    assert any('did not become ready' in e for s in steps for e in s.output.events)


@pytest.mark.parametrize('tool, age', [(None, 0.0), (TOOL_START, 0.2), (TOOL_START, math.inf)])
def test_stale_or_missing_tool_pose_forbids_engagement(tool, age):
    p = params()
    state = ready_state()
    for k in range(3):
        state = on_frame(state, frame(k), p)
    state, output = tick(state, Env(0.03, True, tool, age, True), p)
    assert output.target is None and output.hold == Hold.TOOL_POSE_STALE


def test_calibration_changed_while_engaged_does_not_change_the_mapping():
    """The heading is captured at engagement (a new calibration needs the clutch released)."""
    p = params(filter_cutoff_hz=0.0)
    state = ready_state()
    state = on_frame(state, frame(0), p)
    state, _ = tick(state, Env(0.0, True, TOOL_START, 0.0, True), p)
    state = replace(state, calibration=Calibration(1.0, 7, 0))
    state = on_frame(state, frame(1, controller=moved(HAND, dx=0.01)), p)
    state, output = tick(state, Env(0.02, True, TOOL_START, 0.0, True), p)
    assert v_sub(output.target.p, TOOL_START.p) == pytest.approx((0.01, 0.0, 0.0), abs=1e-12)
    assert q_mul(output.target.q, (0.0, 0.0, 0.0, 1.0)) == pytest.approx(TOOL_START.q)


# ---- conditions that show on a frame between two cycles ----

def between_cycles():
    """Return k such that frames k and k + 1 both arrive between the same two cycles."""
    for k in range(40, 200):
        if math.floor(k / 90.0 / PERIOD) == math.floor((k + 1) / 90.0 / PERIOD) \
                and (k / 90.0) % PERIOD > 1e-6:
            return k
    raise AssertionError('no such frame')


@pytest.mark.parametrize('fields, reason', [
    ({'controller_emulated': True}, Hold.POSE_EMULATED),
    ({'controller': None}, Hold.POSE_ABSENT),
    ({'head_emulated': True}, Hold.HEAD_EMULATED),
    ({'head': None, 'head_present': False}, Hold.HEAD_ABSENT),
    ({'head': Pose(HEAD.p, q_from_yaw(math.radians(120.0)))}, Hold.OUT_OF_VIEW),
])
def test_condition_on_one_frame_between_two_cycles_still_releases(fields, reason):
    """Valid, bad, valid within one 20 ms cycle: the newest frame alone would hide it."""
    k = between_cycles()
    frames = [frame(i, **(fields if i == k else {})) for i in range(k + 60)]
    steps = run(frames, state=ready_state())
    (span,) = engaged_spans(steps)
    assert span[1] < k / 90.0
    assert hold_at(steps, (k + 1) / 90.0 + PERIOD) == reason
    assert hold_at(steps, (k + 50) / 90.0) == reason        # latched while the grip is held


def test_grip_opened_and_closed_within_one_cycle_restarts_the_engagement():
    """G-2, G-10: a release seen on a single frame is a release; state starts afresh."""
    k = between_cycles()

    def motion(i):
        return moved(HAND, dx=0.1 * i / 90.0)
    frames = stream(k + 90, motion, grip=lambda i: i != k)
    steps = run(frames, state=ready_state())
    first, second = engaged_spans(steps)
    assert first[1] < k / 90.0 < second[0] <= k / 90.0 + 4 * PERIOD
    events = [e for s in steps for e in s.output.events]
    assert sum('clutch engaged' in e for e in events) == 2
    by_time = dict(targets(steps))
    assert same_pose(by_time[second[0]], by_time[first[1]])      # restarted from the tool pose
    assert max_step(steps)[0] < 0.004


def engaged_state(p):
    state = ready_state()
    for i in range(3):
        state = on_frame(state, frame(i), p)
    state, output = tick(state, Env(0.03, True, TOOL_START, 0.0, True), p)
    assert output.engaged
    return on_frame(state, frame(3), p)


def test_servo_no_longer_running_under_the_clutch_releases():
    p = params()
    state, output = tick(engaged_state(p), Env(0.05, True, TOOL_START, 0.0, False), p)
    assert output.target is None and not output.engaged
    assert output.hold == Hold.SERVO_NOT_READY and not output.want_servo_active
    state = on_frame(state, frame(4), p)
    state, output = tick(state, Env(0.07, True, TOOL_START, 0.0, True), p)
    assert output.target is None and output.hold == Hold.SERVO_NOT_READY      # latched


def test_servo_halt_under_the_clutch_releases_so_the_target_cannot_run_ahead():
    p = params()
    env = Env(0.05, True, TOOL_START, 0.0, True, servo_halted=True)
    state, output = tick(engaged_state(p), env, p)
    assert output.target is None and not output.engaged
    assert output.hold == Hold.SERVO_HALTED
    state = on_frame(state, frame(4), p)
    state, output = tick(state, Env(0.07, True, TOOL_START, 0.0, True), p)
    assert output.target is None and output.hold == Hold.SERVO_HALTED         # latched


def test_view_cone_axis_can_be_tilted_down():
    """A controller held low is outside a level cone and inside one tilted toward it."""
    low = Pose((0.4, 0.0, 1.6 - 0.4), HAND.q)               # 45 deg below the forward axis
    assert view_angle(HEAD, low.p) == pytest.approx(math.radians(45.0))
    assert view_angle(HEAD, low.p, math.radians(30.0)) == pytest.approx(math.radians(15.0))
    assert view_angle(HEAD, low.p, math.radians(45.0)) == pytest.approx(0.0, abs=1e-7)
    # The tilt is about the headset's own left-right axis: it follows a head turn.
    turned = Pose(HEAD.p, q_from_yaw(math.radians(90.0)))
    low_left = (0.0, 0.4, 1.2)
    assert view_angle(turned, low_left, math.radians(45.0)) == pytest.approx(0.0, abs=1e-7)
    # And it narrows the view upward by the same amount.
    high = (0.4, 0.0, 2.0)
    assert view_angle(HEAD, high, math.radians(30.0)) == pytest.approx(math.radians(75.0))

    level = run(stream(60, lambda k: low), params(view_cone_half_angle_deg=40.0),
                state=ready_state())
    tilted = run(stream(60, lambda k: low),
                 params(view_cone_half_angle_deg=40.0, view_cone_tilt_deg=30.0),
                 state=ready_state())
    assert not targets(level) and hold_at(level, 0.5) == Hold.OUT_OF_VIEW
    assert targets(tilted)


def test_view_cone_tilt_defaults_to_zero_and_is_validated():
    assert params().view_cone_tilt == 0.0
    assert params(view_cone_tilt_deg=15.0).view_cone_tilt == pytest.approx(math.radians(15.0))
    for bad in (91.0, -91.0, float('nan')):
        with pytest.raises(ValueError):
            params(view_cone_tilt_deg=bad)


@pytest.mark.parametrize('bad, reason', [
    ({'controller_emulated': True}, Hold.POSE_EMULATED),
    ({'seq': 5}, Hold.SEQ_BACKWARDS),
    ({'head_emulated': True}, Hold.HEAD_EMULATED),
])
def test_fault_after_a_grip_release_in_the_same_cycle_is_not_lost(bad, reason):
    """Grip opens on one frame, a fault shows on the next, both inside one cycle."""
    k = between_cycles()

    def one(i):
        fields = dict(bad) if i == k + 1 else {}
        return frame(i, grip=i != k, **fields)
    steps = run([one(i) for i in range(k + 60)], state=ready_state())
    # The fault outranks the plain release: one engagement only, latched on the fault
    # until the grip is released again (it is held to the end here).
    (span,) = engaged_spans(steps)
    assert span[1] < k / 90.0
    assert hold_at(steps, (k + 40) / 90.0) == reason
