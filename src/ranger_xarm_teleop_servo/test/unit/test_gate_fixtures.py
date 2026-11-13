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
Safety gate driven by JSONL recordings: synthetic clutch fixtures and real headset clips.

INV-4 (one test per G-4 trigger), plus INV-2, INV-3, INV-5, INV-8 and INV-9 on whole
recordings. The synthetic fixtures calibrate (A), arm (B) and close the grip through
the buttons, so nothing is preset. The real clips were recorded without a clutch, so
the test holds the grip on them and starts armed and calibrated.
"""

import math
import os
import sys

from helpers import (bridge_fixture, engaged_spans, fixture, FIXTURES, hold_at, max_step,
                     params, PERIOD, ready_for, run, same_pose, targets, TOOL_START)
import pytest
from ranger_xarm_teleop_servo.frames import hold_grip
from ranger_xarm_teleop_servo.gate import Hold
from ranger_xarm_teleop_servo.rotations import q_between, q_from_yaw, q_mul, v_norm, v_sub

GRIP_ON = 1.2
TRIGGER_AT = 2.5


def events(steps):
    return [e for s in steps for e in s.output.events]


def test_fixtures_are_what_the_generator_writes():
    sys.path.insert(0, FIXTURES)
    try:
        import make_fixtures
    finally:
        sys.path.remove(FIXTURES)
    for name, lines in make_fixtures.generate().items():
        with open(os.path.join(FIXTURES, name + '.jsonl'), encoding='utf-8') as handle:
            assert handle.read().splitlines() == lines, name


# ---- whole motions ----

def test_line_calibrate_arm_clutch_then_tool_moves_along_base_x():
    steps = run(fixture('clutch_line'))
    (span,) = engaged_spans(steps)
    assert GRIP_ON <= span[0] <= GRIP_ON + 3 * PERIOD
    assert 4.2 - PERIOD <= span[1] < 4.2 + PERIOD            # INV-2: release holds at once
    assert same_pose(targets(steps)[0][1], TOOL_START)
    final = targets(steps)[-1][1]
    # The hand moved 0.20 m along the direction it pointed at when calibrating
    # (20 deg left of the room's forward): the tool moves 0.20 m along base +x.
    assert v_sub(final.p, TOOL_START.p) == pytest.approx((0.20, 0.0, 0.0), abs=1e-4)
    assert q_between(final.q, TOOL_START.q) < 1e-6
    assert any('calibrated: heading -20.0 deg' in e for e in events(steps))
    assert max_step(steps)[0] < 0.005       # peak hand speed 0.19 m/s, two frames per tick
    for step in steps:
        if step.t < GRIP_ON:
            assert step.output.target is None


def test_line_without_arming_or_calibrating_publishes_nothing():
    from dataclasses import replace
    unarmed = run(fixture('clutch_line'), edit=lambda f: replace(f, arm=False))
    assert not targets(unarmed) and hold_at(unarmed, 2.0) == Hold.NOT_ARMED
    uncalibrated = run(fixture('clutch_line'), edit=lambda f: replace(f, calibrate=False))
    assert not targets(uncalibrated) and hold_at(uncalibrated, 2.0) == Hold.NOT_CALIBRATED
    disabled = run(fixture('clutch_line'), params(motion_enabled=False))
    assert not targets(disabled) and hold_at(disabled, 2.0) == Hold.MOTION_DISABLED


def test_circle_returns_to_the_far_point_of_the_path():
    steps = run(fixture('clutch_circle'), params(filter_cutoff_hz=0.0))
    poses = [pose for _, pose in targets(steps)]
    # One and a half circles of radius 0.06 m in the horizontal plane end 0.12 m from
    # the start; every target stays on the circle through the start.
    assert v_norm(v_sub(poses[-1].p, TOOL_START.p)) == pytest.approx(0.12, abs=1e-4)
    assert max(abs(pose.p[2] - TOOL_START.p[2]) for pose in poses) < 1e-9
    centre_distance = [abs(v_norm(v_sub(pose.p, poses[-1].p)) ** 2
                           + v_norm(v_sub(pose.p, TOOL_START.p)) ** 2 - 0.12 ** 2)
                       for pose in poses]
    assert max(centre_distance) < 1e-6          # Thales: right angle over the diameter


def test_rotation_about_the_vertical_stays_about_the_vertical_and_returns():
    steps = run(fixture('clutch_rotation'), params(filter_cutoff_hz=0.0))
    poses = [pose for _, pose in targets(steps)]
    assert max(v_norm(v_sub(pose.p, TOOL_START.p)) for pose in poses) < 1e-9
    widest = max(poses, key=lambda pose: q_between(TOOL_START.q, pose.q))
    assert q_between(widest.q, q_mul(q_from_yaw(math.radians(40.0)), TOOL_START.q)) < 1e-4
    assert q_between(poses[-1].q, TOOL_START.q) < 1e-4


@pytest.mark.parametrize('cutoff', [0.0, 8.0])
def test_reengage_fixture_adds_both_moves_without_a_jump(cutoff):
    steps = run(fixture('clutch_reengage'), params(filter_cutoff_hz=cutoff))
    first, second = engaged_spans(steps)
    by_time = dict(targets(steps))
    assert same_pose(by_time[second[0]], by_time[first[1]])    # INV-8
    assert max_step(steps)[0] < 0.005 and max_step(steps)[1] < math.radians(0.2)
    final = targets(steps)[-1][1]
    assert v_sub(final.p, TOOL_START.p) == pytest.approx((0.20, 0.0, 0.0), abs=2e-3)
    assert q_between(final.q, TOOL_START.q) < 1e-6


@pytest.mark.parametrize('cutoff', [0.0, 8.0])
def test_jump_fixture_releases_before_any_target_moves(cutoff):
    steps = run(fixture('clutch_jump'), params(filter_cutoff_hz=cutoff))
    (span,) = engaged_spans(steps)
    assert span[1] < TRIGGER_AT
    assert hold_at(steps, TRIGGER_AT + PERIOD) == Hold.INPUT_JUMP
    assert hold_at(steps, 4.0) == Hold.INPUT_JUMP
    assert max(abs(pose.p[1]) for _, pose in targets(steps)) < 1e-6
    assert max_step(steps)[0] < 0.005       # peak hand speed 0.19 m/s, two frames per tick
    assert any('raw controller jump' in e and '26.' in e for e in events(steps))


def test_stall_fixture_releases_within_the_timeout_and_stays_released():
    frames = fixture('clutch_stall')
    steps = run(frames)
    last_before = max(f.arrival for f in frames if f.arrival < TRIGGER_AT)
    (span,) = engaged_spans(steps)
    assert last_before <= span[1] <= last_before + 0.1
    assert hold_at(steps, last_before + 0.1 + PERIOD) == Hold.INPUT_STALE
    assert hold_at(steps, 3.5) == Hold.INPUT_STALE            # stream is back, grip still held
    stale = [s for s in steps if last_before + 0.1 < s.t < 3.0]
    assert stale and all(s.output.input_age > 0.1 for s in stale)


def test_release_while_moving_publishes_nothing_after_the_release_frame():
    """INV-2: the grip opens at the peak hand speed; the target stops with it."""
    frames = fixture('clutch_release_moving')
    steps = run(frames)
    (span,) = engaged_spans(steps)
    assert TRIGGER_AT - PERIOD <= span[1] < TRIGGER_AT + PERIOD
    last = dict(targets(steps))[span[1]]
    # The hand went on to 0.20 m; the target stayed where the release found it.
    assert 0.08 < last.p[0] - TOOL_START.p[0] < 0.11
    assert all(s.output.target is None for s in steps if s.t > span[1])
    assert hold_at(steps, 3.0) == Hold.CLUTCH_RELEASED


def test_into_body_fixture_stays_engaged_so_the_stop_is_the_servo_layer():
    """The gate itself has no reason to stop this motion (0.28 m is inside the offset limit)."""
    steps = run(fixture('clutch_into_body'))
    (span,) = engaged_spans(steps)
    assert span[1] > 5.9
    final = targets(steps)[-1][1]
    assert v_sub(final.p, TOOL_START.p) == pytest.approx((-0.28, 0.0, 0.0), abs=1e-3)


# ---- INV-4: one test per trigger ----

@pytest.mark.parametrize('name, reason, at', [
    ('gate_absent', Hold.POSE_ABSENT, TRIGGER_AT),
    ('gate_emulated', Hold.POSE_EMULATED, TRIGGER_AT),
    ('gate_head_emulated', Hold.HEAD_EMULATED, TRIGGER_AT),
    ('gate_reset', Hold.RESET_CHANGED, TRIGGER_AT),
    ('gate_session', Hold.SESSION_CHANGED, TRIGGER_AT),
    ('gate_seq_backwards', Hold.SEQ_BACKWARDS, TRIGGER_AT),
    ('gate_out_of_view', Hold.OUT_OF_VIEW, TRIGGER_AT + 0.15),   # 45 deg reached mid-turn
])
def test_trigger_releases_the_clutch_and_freezes_the_target(name, reason, at):
    steps = run(fixture(name))
    spans = engaged_spans(steps)
    assert spans[0][0] <= GRIP_ON + 3 * PERIOD
    # Released within one cycle of the frame that shows the trigger.
    assert at - PERIOD <= spans[0][1] < at + 2 * PERIOD
    release = next(s for s in steps if s.t > spans[0][1])
    assert release.output.hold == reason and release.output.target is None
    assert not release.output.want_servo_active
    # The target was moving before; the frozen target is the last one published.
    frozen = dict(targets(steps))[spans[0][1]]
    assert v_norm(v_sub(frozen.p, TOOL_START.p)) > 0.01
    if name != 'gate_emulated':
        # The grip stays held to the end: nothing more is published.
        assert len(spans) == 1
        assert all(s.output.target is None for s in steps if s.t > spans[0][1])


def test_trigger_latches_until_the_grip_is_released_and_pressed_again():
    steps = run(fixture('gate_emulated'))
    first, second = engaged_spans(steps)
    # Emulated 2.5-3.0 s; tracked again from 3.0 s with the grip held: still released.
    assert all(s.output.target is None for s in steps if first[1] < s.t < 3.8)
    assert hold_at(steps, 3.3) == Hold.POSE_EMULATED
    assert hold_at(steps, 3.7) == Hold.CLUTCH_RELEASED
    assert 3.8 <= second[0] <= 3.8 + 3 * PERIOD
    by_time = dict(targets(steps))
    assert same_pose(by_time[second[0]], by_time[first[1]])   # re-engaged without a jump


@pytest.mark.parametrize('name', ['gate_reset', 'gate_session'])
def test_frame_change_also_invalidates_the_calibration(name):
    steps = run(fixture(name))
    assert hold_at(steps, 3.2) == Hold.NOT_CALIBRATED          # INV-9, G-13


# ---- real headset clips ----

def test_real_occlusion_clip_releases_when_the_pose_turns_emulated():
    frames = bridge_fixture('clip_controller_occlusion')
    emulated_at = next(f.arrival for f in frames if f.controller_emulated)
    # The controller was behind the operator (143-152 deg off the view axis): with the
    # view cone open it is followed until the headset flags the position as emulated.
    steps = run(frames, params(view_cone_half_angle_deg=180.0), state=ready_for(frames),
                edit=hold_grip, tail=0.0)
    (span,) = engaged_spans(steps)
    assert span[1] < emulated_at
    after = next(s for s in steps if s.t >= emulated_at)
    assert after.output.hold == Hold.POSE_EMULATED and after.output.target is None
    # With the simulation view cone (45 deg) it is never followed at all.
    steps = run(frames, state=ready_for(frames), edit=hold_grip, tail=0.0)
    assert not targets(steps)
    assert {s.output.hold for s in steps if s.t > 0.05} <= {Hold.OUT_OF_VIEW,
                                                            Hold.POSE_EMULATED}


def test_real_look_away_clip_releases_at_the_view_cone_while_still_reported_tracked():
    """Recording B: the head turns away and the pose stays 'tracked'; the cone catches it."""
    frames = fixture('clip_controller_look_away')
    assert not any(f.controller_emulated for f in frames)
    assert all(f.controller_present and f.head_present for f in frames)
    steps = run(frames, state=ready_for(frames), edit=hold_grip, tail=0.0)
    (span,) = engaged_spans(steps)
    assert 0.2 < span[1] < 0.4                  # 45 deg is crossed 0.30 s into the clip
    assert steps[-1].output.hold == Hold.OUT_OF_VIEW
    assert all(s.output.target is None for s in steps if s.t > span[1])


def test_real_hands_clip_is_never_followed():
    frames = bridge_fixture('clip_hands_head_turn')
    steps = run(frames, state=ready_for(frames), edit=hold_grip, tail=0.0)
    assert not targets(steps)
    assert all(s.output.hold == Hold.POSE_ABSENT for s in steps if s.t > 0.05)


@pytest.mark.parametrize('name, reason', [
    ('reset_event', Hold.RESET_CHANGED),
    ('new_session', Hold.SESSION_CHANGED),
    ('dropped_controller', Hold.POSE_ABSENT),
])
def test_bridge_fixtures_release_on_their_event(name, reason):
    frames = bridge_fixture(name)
    steps = run(frames, params(view_cone_half_angle_deg=180.0), state=ready_for(frames),
                edit=hold_grip)
    spans = engaged_spans(steps)
    assert len(spans) == 1
    release = next(s for s in steps if s.t > spans[0][1])
    assert release.output.hold == reason
    assert 1.9 < spans[0][1] < 2.6
