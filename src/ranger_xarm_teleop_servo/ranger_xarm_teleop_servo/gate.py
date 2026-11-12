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
The teleop safety gate as a pure state machine (SAFETY_GATE.md). No ROS imports.

Two functions drive it. ``on_frame`` is called for every headset frame, in arrival
order: it watches what only consecutive frames can show (session id, reset count,
sequence, raw controller speed) and handles the button edges. ``tick`` is called at
the target rate: it decides whether a target may be published this cycle and computes
it. Both take the state and return a new one; nothing is hidden in globals.

Time is a monotonic clock read by the caller at the local arrival of a frame
(``Frame.arrival``) and at each tick (``Env.now``); header stamps from other clocks
are never compared. Headset time (``Frame.xr_time``) is used only for differences
between frames.

Rules, by ID:

* G-1  no target unless ``motion_enabled`` and armed and teleop owns the arm.
* G-2  the grip is the clutch; released means no target from that cycle on.
* G-3  newest frame older than the timeout releases the clutch.
* G-4  absent or emulated controller pose, controller outside the view cone, absent or
  emulated headset pose, reset count or session id change, sequence or headset time
  going backwards: each releases the clutch.
* G-5  raw controller speed between consecutive frames above the limit releases the
  clutch, before any filtering; the step of the final target is checked again.
* G-6  offset limit and keep-out planes are applied after the filter and re-checked on
  the final pose; a pose that fails the re-check is replaced by the last accepted one.
* G-10 filter and limiter state restart from the measured tool pose at every engagement.
* While engaged, MoveIt Servo no longer running, or reporting a halt, releases the
  clutch: a target must not move on while the arm stands still.
* G-13 a calibration made under another (session id, reset count) is not used.

A release caused by anything but the grip latches: the clutch engages again only
after the grip was seen released.
"""

from dataclasses import dataclass, replace
from enum import IntEnum
import math
from typing import Optional, Tuple

from .calibration import calibrate, Calibration
from .mapping import apply_pivot, clutch_relative
from .params import GateParams
from .rotations import (Pose, q_between, q_rotate, q_slerp, v_add, v_dot, v_norm, v_scale,
                        v_sub)


class Hold(IntEnum):
    """Why no target is published; values match ``TeleopState.HOLD_*``."""

    NONE = 0
    MOTION_DISABLED = 1
    NOT_ARMED = 2
    NOT_OWNER = 3
    NOT_CALIBRATED = 4
    CLUTCH_RELEASED = 5
    INPUT_STALE = 6
    POSE_ABSENT = 7
    POSE_EMULATED = 8
    OUT_OF_VIEW = 9
    HEAD_ABSENT = 10
    HEAD_EMULATED = 11
    RESET_CHANGED = 12
    SESSION_CHANGED = 13
    SEQ_BACKWARDS = 14
    INPUT_JUMP = 15
    STEP_REJECTED = 16
    TOOL_POSE_STALE = 17
    SERVO_NOT_READY = 18
    SERVO_HALTED = 19


class Limit(IntEnum):
    """Bound that shaped the target; values match ``TeleopState.LIMIT_*``."""

    NONE = 0
    OFFSET_TRANSLATION = 1
    OFFSET_ROTATION = 2
    KEEP_OUT = 3
    SERVO = 4


# Slack for the re-check of a pose that was just clamped onto a bound.
BOUND_EPS = 1e-9


@dataclass(frozen=True)
class Frame:
    """One headset frame, paired by stamp: status, joy and the poses that came with it."""

    arrival: float            # monotonic seconds at local arrival
    xr_time: float            # headset frame time, seconds
    session_id: int
    seq: int
    reset_count: int
    controller: Optional[Pose]        # right controller in quest_world, None if no pose came
    controller_present: bool
    controller_emulated: bool
    head: Optional[Pose]
    head_present: bool
    head_emulated: bool
    grip: bool
    calibrate: bool
    arm: bool


@dataclass(frozen=True)
class Env:
    """What the node knows at a tick besides the input."""

    now: float                        # monotonic seconds
    owner_is_teleop: bool
    tool_pose: Optional[Pose]         # measured tool pose in the arm base frame
    tool_pose_age: float              # seconds; inf when unknown
    servo_active: bool                # MoveIt Servo confirmed running in pose mode
    servo_halted: bool = False        # MoveIt Servo reports a halt (collision, singularity)


@dataclass(frozen=True)
class GateState:
    """Everything the gate remembers between calls."""

    want_armed: bool = False
    calibration: Optional[Calibration] = None
    frame: Optional[Frame] = None
    # Buttons of the newest frame that had a controller: (grip, calibrate, arm). An
    # absent controller reports no buttons, which is not a release.
    buttons: Optional[Tuple[bool, bool, bool]] = None
    trip: Hold = Hold.NONE            # set by on_frame, consumed by the next tick
    latched: Hold = Hold.NONE         # cleared when the grip is seen released
    engaged: bool = False
    waiting_since: Optional[float] = None
    clutch: Optional[Pose] = None     # controller pose (pivot applied) at engagement
    tool0: Optional[Pose] = None      # measured tool pose at engagement
    heading: float = 0.0              # calibration heading captured at engagement
    filtered: Optional[Pose] = None
    last_target: Optional[Pose] = None
    events: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Output:
    """Result of one tick."""

    target: Optional[Pose]            # tool-frame target in the arm base frame, or None
    engaged: bool
    hold: Hold
    limit: Limit
    want_servo_active: bool           # the node keeps MoveIt Servo unpaused only when True
    input_age: float
    input_seq: int
    events: Tuple[str, ...]


def controller_valid(frame: Frame) -> bool:
    """Return True when the frame carries a tracked, measured controller pose."""
    return (frame.controller is not None and frame.controller_present
            and not frame.controller_emulated)


def head_valid(frame: Frame) -> bool:
    """Return True when the frame carries a tracked, measured headset pose."""
    return frame.head is not None and frame.head_present and not frame.head_emulated


def view_angle(head: Pose, controller_position) -> float:
    """
    Return the angle between the headset's forward axis and the direction to a point.

    After the bridge's axis conversion the headset looks along its local +x.
    """
    direction = v_sub(controller_position, head.p)
    distance = v_norm(direction)
    if distance < 1e-9:
        return 0.0
    forward = q_rotate(head.q, (1.0, 0.0, 0.0))
    cosine = v_dot(forward, direction) / distance
    return math.acos(max(-1.0, min(1.0, cosine)))


def _pose_reason(frame: Frame, params: GateParams) -> Hold:
    """Return the G-4 level condition of one frame, or NONE."""
    if not frame.controller_present or frame.controller is None:
        return Hold.POSE_ABSENT
    if frame.controller_emulated:
        return Hold.POSE_EMULATED
    if not frame.head_present or frame.head is None:
        return Hold.HEAD_ABSENT
    if frame.head_emulated:
        return Hold.HEAD_EMULATED
    if view_angle(frame.head, frame.controller.p) > params.view_cone_angle:
        return Hold.OUT_OF_VIEW
    return Hold.NONE


def _between_frames(prev: Frame, frame: Frame, params: GateParams) -> Tuple[Hold, str]:
    """Return what the step from ``prev`` to ``frame`` trips, with a log line."""
    if frame.session_id != prev.session_id:
        return Hold.SESSION_CHANGED, (
            f'XR session id changed {prev.session_id} -> {frame.session_id} at seq {frame.seq}')
    if frame.reset_count != prev.reset_count:
        return Hold.RESET_CHANGED, (
            f'reference-space reset count changed {prev.reset_count} -> {frame.reset_count} '
            f'at seq {frame.seq}')
    dt = frame.xr_time - prev.xr_time
    if frame.seq <= prev.seq or dt <= 0.0:
        return Hold.SEQ_BACKWARDS, (
            f'input went backwards: seq {prev.seq} -> {frame.seq}, headset time {dt:+.4f} s')
    if controller_valid(prev) and controller_valid(frame):
        speed = v_norm(v_sub(frame.controller.p, prev.controller.p)) / dt
        rate = q_between(prev.controller.q, frame.controller.q) / dt
        if speed > params.raw_speed_limit or rate > params.raw_rate_limit:
            return Hold.INPUT_JUMP, (
                f'raw controller jump at seq {frame.seq}: {speed:.2f} m/s, '
                f'{math.degrees(rate):.0f} deg/s over {dt * 1000.0:.1f} ms '
                f'(limits {params.raw_speed_limit:.2f} m/s, '
                f'{math.degrees(params.raw_rate_limit):.0f} deg/s)')
    return Hold.NONE, ''


def _try_calibrate(state: GateState, frame: Frame,
                   params: GateParams) -> Tuple[Optional[Calibration], str]:
    if state.engaged or frame.grip:
        return None, 'calibration ignored: release the clutch first'
    reason = _pose_reason(frame, params)
    if reason != Hold.NONE:
        return None, f'calibration refused: {reason.name}'
    result = calibrate(frame.controller.q, params.pointing_axis,
                       params.calibration_min_horizontal, frame.session_id, frame.reset_count)
    if result is None:
        return None, 'calibration refused: the controller points too close to vertical'
    return result, (f'calibrated: heading {math.degrees(result.heading):+.1f} deg, session '
                    f'{result.session_id}, reset count {result.reset_count}')


def on_frame(state: GateState, frame: Frame, params: GateParams) -> GateState:
    """Take in one headset frame; return the new state."""
    events = list(state.events)
    prev = state.frame
    trip = state.trip
    want_armed = state.want_armed
    calibration = state.calibration
    if prev is not None:
        tripped, text = _between_frames(prev, frame, params)
        # Only a held or engaged clutch can be released; otherwise the levels
        # (calibration validity, pose state) carry the condition.
        held = state.engaged or frame.grip or bool(state.buttons and state.buttons[0])
        if tripped != Hold.NONE:
            if held or tripped != Hold.INPUT_JUMP:
                events.append(text)
            if trip == Hold.NONE and held:
                trip = tripped
        # Cycles run at 50 Hz and frames arrive at 90 Hz: a condition that shows on
        # one frame and is gone on the next must still release the clutch, so every
        # frame is judged here, not only the newest one at the next cycle.
        if trip == Hold.NONE and held:
            trip = _pose_reason(frame, params)
        if (trip == Hold.NONE and state.engaged and frame.controller_present
                and not frame.grip):
            trip = Hold.CLUTCH_RELEASED     # a release, even if the grip closes again at once
    buttons = state.buttons
    if frame.controller_present:
        # Button edges, against the last frame that had a controller. The first such
        # frame only sets the reference: a button already held is not a press.
        if buttons is not None:
            if frame.arm and not buttons[2]:
                want_armed = not want_armed
                events.append('arm requested' if want_armed else 'disarm requested')
            if frame.calibrate and not buttons[1]:
                result, text = _try_calibrate(state, frame, params)
                events.append(text)
                if result is not None:
                    calibration = result
        buttons = (frame.grip, frame.calibrate, frame.arm)
    return replace(state, frame=frame, buttons=buttons, trip=trip, want_armed=want_armed,
                   calibration=calibration, events=tuple(events))


def _condition(state: GateState, env: Env, params: GateParams, age: float) -> Hold:
    """Return the first condition that forbids a target, ignoring the grip itself."""
    frame = state.frame
    if not params.motion_enabled:
        return Hold.MOTION_DISABLED
    if not state.want_armed:
        return Hold.NOT_ARMED
    if not env.owner_is_teleop:
        return Hold.NOT_OWNER
    if frame is None or not age <= params.input_timeout:
        return Hold.INPUT_STALE
    if state.trip != Hold.NONE:
        return state.trip
    reason = _pose_reason(frame, params)
    if reason != Hold.NONE:
        return reason
    if state.calibration is None or not state.calibration.valid_for(frame.session_id,
                                                                    frame.reset_count):
        return Hold.NOT_CALIBRATED
    if env.tool_pose is None or not env.tool_pose_age <= params.tool_pose_timeout:
        return Hold.TOOL_POSE_STALE
    if state.engaged and not env.servo_active:
        return Hold.SERVO_NOT_READY         # MoveIt Servo stopped running under the clutch
    if state.engaged and env.servo_halted:
        return Hold.SERVO_HALTED            # the arm stands still; the target must not run on
    return Hold.NONE


def low_pass(previous: Pose, raw: Pose, params: GateParams) -> Pose:
    """One step of a first-order low-pass at the target rate; cutoff 0 passes ``raw``."""
    if params.filter_cutoff_hz <= 0.0:
        return raw
    alpha = 1.0 - math.exp(-2.0 * math.pi * params.filter_cutoff_hz * params.target_period)
    position = v_add(previous.p, v_scale(v_sub(raw.p, previous.p), alpha))
    return Pose(position, q_slerp(previous.q, raw.q, alpha))


def clamp_offset(pose: Pose, tool0: Pose, params: GateParams) -> Tuple[Pose, Limit]:
    """Clamp ``pose`` onto the per-clutch offset limit around ``tool0`` (G-6, layer 2)."""
    limit = Limit.NONE
    position, orientation = pose.p, pose.q
    offset = v_sub(position, tool0.p)
    distance = v_norm(offset)
    if distance > params.max_offset_translation:
        position = v_add(tool0.p, v_scale(offset, params.max_offset_translation / distance))
        limit = Limit.OFFSET_TRANSLATION
    angle = q_between(tool0.q, orientation)
    if angle > params.max_offset_rotation:
        orientation = q_slerp(tool0.q, orientation, params.max_offset_rotation / angle)
        if limit == Limit.NONE:
            limit = Limit.OFFSET_ROTATION
    return Pose(position, orientation), limit


def _plane_depth(pose: Pose, plane, params: GateParams) -> float:
    """Return how far the tool point or the wrist point is inside a keep-out half-space."""
    wrist = v_add(pose.p, q_rotate(pose.q, params.wrist_offset))
    return max(plane.d - v_dot(plane.n, pose.p), plane.d - v_dot(plane.n, wrist))


def push_out_of_planes(pose: Pose, params: GateParams) -> Tuple[Pose, bool]:
    """Shift ``pose`` along the plane normals until no keep-out plane is crossed."""
    moved = False
    position = pose.p
    for _ in range(4):
        again = False
        for plane in params.keep_out_planes:
            depth = _plane_depth(Pose(position, pose.q), plane, params)
            if depth > 0.0:
                position = v_add(position, v_scale(plane.n, depth))
                moved = again = True
        if not again:
            break
    return Pose(position, pose.q), moved


def within_bounds(pose: Pose, tool0: Pose, params: GateParams) -> bool:
    """Return True when ``pose`` satisfies the offset limit and every keep-out plane."""
    if v_norm(v_sub(pose.p, tool0.p)) > params.max_offset_translation + BOUND_EPS:
        return False
    if q_between(tool0.q, pose.q) > params.max_offset_rotation + 1e-6:
        return False
    return all(_plane_depth(pose, plane, params) <= BOUND_EPS
               for plane in params.keep_out_planes)


def bound(pose: Pose, tool0: Pose, fallback: Pose,
          params: GateParams) -> Tuple[Pose, Limit]:
    """
    Apply the offset limit and the keep-out planes, then re-check both (G-6).

    A pose that still violates a bound after the adjustments is replaced by
    ``fallback`` (the last accepted target), never passed on.
    """
    clamped, limit = clamp_offset(pose, tool0, params)
    pushed, moved = push_out_of_planes(clamped, params)
    if moved:
        limit = Limit.KEEP_OUT
    if not within_bounds(pushed, tool0, params):
        return fallback, Limit.KEEP_OUT
    return pushed, limit


def _released(state: GateState, latched: Hold) -> GateState:
    """Return the state with the clutch disengaged and its per-engagement data cleared."""
    return replace(state, engaged=False, waiting_since=None, clutch=None, tool0=None,
                   filtered=None, last_target=None, trip=Hold.NONE, latched=latched,
                   events=())


def tick(state: GateState, env: Env, params: GateParams) -> Tuple[GateState, Output]:
    """Run one target cycle; return the new state and what to publish."""
    frame = state.frame
    events = list(state.events)
    age = env.now - frame.arrival if frame is not None else math.inf
    seq = frame.seq if frame is not None else 0
    grip = frame is not None and frame.grip
    latched = state.latched
    # The grip as last seen on a present controller; an absent one shows nothing.
    grip_seen = bool(state.buttons and state.buttons[0])
    if state.buttons is not None and not grip_seen:
        latched = Hold.NONE

    def out(new_state: GateState, target: Optional[Pose], hold: Hold, limit: Limit,
            want_servo: bool) -> Tuple[GateState, Output]:
        return new_state, Output(target, new_state.engaged, hold, limit, want_servo, age, seq,
                                 tuple(events))

    reason = _condition(state, env, params, age)
    if reason == Hold.CLUTCH_RELEASED:
        # The grip was seen open on a frame since the last cycle. Not latched: if it is
        # closed again, the next cycle engages afresh, with new per-engagement state.
        events.append('clutch released')
        return out(_released(state, latched), None, reason, Limit.NONE, False)
    if reason != Hold.NONE:
        if (state.engaged or grip_seen) and latched == Hold.NONE:
            latched = reason
        if state.engaged:
            events.append(f'clutch released: {reason.name}')
        return out(_released(state, latched), None, reason, Limit.NONE, False)
    if not grip:
        if state.engaged:
            events.append('clutch released')
        return out(_released(state, latched), None, Hold.CLUTCH_RELEASED, Limit.NONE, False)
    if latched != Hold.NONE:
        return out(_released(state, latched), None, latched, Limit.NONE, False)

    if not state.engaged:
        if not env.servo_active:
            since = state.waiting_since if state.waiting_since is not None else env.now
            if env.now - since > params.servo_ready_timeout:
                events.append('clutch not engaged: MoveIt Servo did not become ready')
                return out(_released(state, Hold.SERVO_NOT_READY), None,
                           Hold.SERVO_NOT_READY, Limit.NONE, False)
            waiting = replace(state, waiting_since=since, latched=latched, events=())
            return out(waiting, None, Hold.SERVO_NOT_READY, Limit.NONE, True)
        # Engagement: every per-engagement quantity restarts here (G-10).
        state = replace(state, engaged=True, waiting_since=None,
                        clutch=apply_pivot(frame.controller, params.pivot_offset),
                        tool0=env.tool_pose, heading=state.calibration.heading,
                        filtered=env.tool_pose, last_target=env.tool_pose)
        events.append(f'clutch engaged at seq {frame.seq}')

    raw = clutch_relative(apply_pivot(frame.controller, params.pivot_offset), state.clutch,
                          state.heading, state.tool0)
    filtered = low_pass(state.filtered, raw, params)
    target, limit = bound(filtered, state.tool0, state.last_target, params)
    step = v_norm(v_sub(target.p, state.last_target.p))
    turn = q_between(state.last_target.q, target.q)
    if step > params.step_translation_limit or turn > params.step_rotation_limit:
        events.append(f'clutch released: target step {step * 1000.0:.1f} mm, '
                      f'{math.degrees(turn):.1f} deg beyond the per-cycle limit')
        return out(_released(state, Hold.STEP_REJECTED), None, Hold.STEP_REJECTED,
                   Limit.NONE, False)
    state = replace(state, filtered=filtered, last_target=target, latched=Hold.NONE, events=())
    return out(state, target, Hold.NONE, limit, True)
