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
* G-14 the target may not lead the measured tool pose by more than the lead limits. The
  excess is clamped away and absorbed: the mapping's anchor and the filter state move by
  the same amount, so the hand's lead is not stored and the arm never catches up later.
  Where the solver reports that a position-type constraint (link keep-out, G-15, or a
  joint position bound) took tool motion out of its last command, the target's
  translation lead along that direction is absorbed as well: a target held inside the
  blocked direction would make a reversal of the hand wait until it has come back.
  Order: map, filter, G-6 bounds, G-5 step check (a jump is rejected, never clipped),
  G-14 lead clamp, G-6 re-check. The speed caps (G-8) act on the joint command (solver).
* While engaged, MoveIt Servo no longer running (or no longer following), or reporting a
  halt, releases the clutch: a target must not move on while the arm stands still. So
  does a solver that cannot run (no chain or joint state, commanded and measured apart).
* G-13 a calibration made under another (session id, reset count) is not used.

Translation scale: the right thumbstick click switches between the fine and the coarse
translation scale while armed; every arming starts at fine. Hand translation is
multiplied by the scale, rotation never. A switch while engaged re-anchors the mapping
at the controller pose of that cycle, so the target does not move at the switch. The
raw-input check (G-5) stays on the unscaled hand; every later check sees the scaled
target.

A release caused by anything but the grip latches: the clutch engages again only
after the grip was seen released.
"""

from dataclasses import dataclass, replace
from enum import IntEnum
import math
from typing import Optional, Sequence, Tuple

from .calibration import calibrate, Calibration
from .mapping import apply_pivot, clutch_relative, rescale
from .params import GateParams
from .rotations import (IDENTITY_Q, Pose, q_between, q_conj, q_mul, q_normalize, q_rotate,
                        q_slerp, q_to_rotvec, Quat, v_add, v_dot, v_norm, v_scale, v_sub)


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
    SOLVER = 20


class Limit(IntEnum):
    """Bound that shaped the target; values match ``TeleopState.LIMIT_*``."""

    NONE = 0
    OFFSET_TRANSLATION = 1
    OFFSET_ROTATION = 2
    KEEP_OUT = 3
    SERVO = 4
    LEAD = 5
    TOOL_SPEED = 6
    JOINT_VELOCITY = 7
    JOINT_POSITION = 8
    LINK_KEEPOUT = 9


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
    scale_toggle: bool = False        # right thumbstick click: fine <-> coarse translation scale
    # Right trigger, analog 0..1; None when the frame's Joy has no such value. Read only by
    # the gripper logic (gripper.py); the arm's mapping never uses it.
    trigger: Optional[float] = None


@dataclass(frozen=True)
class Env:
    """What the node knows at a tick besides the input."""

    now: float                        # monotonic seconds
    owner_is_teleop: bool
    tool_pose: Optional[Pose]         # measured tool pose in the arm base frame
    tool_pose_age: float              # seconds; inf when unknown
    servo_active: bool                # MoveIt Servo verified, running in joint-jog mode
    servo_halted: bool = False        # MoveIt Servo reports a halt (collision, joint bound)
    # The solver can run: kinematic chain and joint state known, and the commanded state
    # agrees with the measured one.
    solver_ready: bool = True
    # Orientation conditioning of the solver at the measured joints (NaN: unknown). Read
    # only when an arm request waits for it; the default leaves arming unchecked.
    arming_conditioning: float = math.inf
    # Tool velocity (m/s, base frame) that the solver's position-type constraints took out
    # of its last command; zero when none bound. The lead along it is absorbed (G-14).
    blocked: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GateState:
    """Everything the gate remembers between calls."""

    want_armed: bool = False
    arm_pending: bool = False         # armed by the button, the posture not yet checked
    calibration: Optional[Calibration] = None
    frame: Optional[Frame] = None
    # Buttons of the newest frame that had a controller: (grip, calibrate, arm, scale
    # toggle). An absent controller reports no buttons, which is not a release.
    buttons: Optional[Tuple[bool, bool, bool, bool]] = None
    trip: Hold = Hold.NONE            # set by on_frame, consumed by the next tick
    latched: Hold = Hold.NONE         # cleared when the grip is seen released
    engaged: bool = False
    waiting_since: Optional[float] = None
    clutch: Optional[Pose] = None     # controller pose (pivot applied) at engagement
    tool0: Optional[Pose] = None      # measured tool pose at engagement (G-6 offset limit)
    anchor: Optional[Pose] = None     # origin of the mapping: tool0 plus the absorbed lead
    heading: float = 0.0              # calibration heading captured at engagement
    coarse: bool = False              # the operator selected the coarse translation scale
    scale: float = 1.0                # translation scale the engagement's anchor is built for
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
    # Twist of the target before the lead clamp, (vx, vy, vz, wx, wy, wz) in m/s and rad/s
    # in the arm base frame: the hand's motion after filter and bounds, without any
    # measured motion in it. None when no target is published.
    feedforward: Optional[Tuple[float, ...]] = None
    # The first condition that forbids motion this cycle with the grip set aside (NONE when
    # only the grip, or a latched release, holds the arm). The gripper may be commanded
    # only while this is NONE or CLUTCH_RELEASED; the arm's outputs do not depend on it.
    condition: Hold = Hold.NONE


def controller_valid(frame: Frame) -> bool:
    """Return True when the frame carries a tracked, measured controller pose."""
    return (frame.controller is not None and frame.controller_present
            and not frame.controller_emulated)


def head_valid(frame: Frame) -> bool:
    """Return True when the frame carries a tracked, measured headset pose."""
    return frame.head is not None and frame.head_present and not frame.head_emulated


def view_angle(head: Pose, controller_position, tilt: float = 0.0) -> float:
    """
    Return the angle between the view cone's axis and the direction to a point.

    After the bridge's axis conversion the headset looks along its local +x, with
    local +z up. The cone axis is that forward axis tilted down by ``tilt`` (about the
    headset's own left-right axis), for operators who look ahead and hold the
    controller low.
    """
    direction = v_sub(controller_position, head.p)
    distance = v_norm(direction)
    if distance < 1e-9:
        return 0.0
    forward = q_rotate(head.q, (math.cos(tilt), 0.0, -math.sin(tilt)))
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
    if view_angle(frame.head, frame.controller.p,
                  params.view_cone_tilt) > params.view_cone_angle:
        return Hold.OUT_OF_VIEW
    return Hold.NONE


def frame_fault(prev: Optional[Frame], frame: Frame) -> Hold:
    """
    Return the session or sequence fault one frame shows against the previous one, or NONE.

    Used by the gripper, for every frame, clutch held or not: the XR session id changed
    (SESSION_CHANGED), or the sequence number or the headset time did not move forward
    (SEQ_BACKWARDS). Pose-only conditions (G-4 pose reasons, a raw jump, a reference-space
    reset) are not faults of the gripper's input: the trigger is a button.
    """
    if prev is None:
        return Hold.NONE
    if frame.session_id != prev.session_id:
        return Hold.SESSION_CHANGED
    if frame.seq <= prev.seq or frame.xr_time - prev.xr_time <= 0.0:
        return Hold.SEQ_BACKWARDS
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
    arm_pending = state.arm_pending
    coarse = state.coarse
    calibration = state.calibration
    if prev is not None:
        tripped, text = _between_frames(prev, frame, params)
        # Only a held or engaged clutch can be released; otherwise the levels
        # (calibration validity, pose state) carry the condition.
        held = state.engaged or frame.grip or bool(state.buttons and state.buttons[0])
        if tripped != Hold.NONE:
            if held or tripped != Hold.INPUT_JUMP:
                events.append(text)
            if trip in (Hold.NONE, Hold.CLUTCH_RELEASED) and held:
                trip = tripped              # a fault outranks a plain release seen earlier
        # Cycles run at 50 Hz and frames arrive at 90 Hz: a condition that shows on
        # one frame and is gone on the next must still release the clutch, so every
        # frame is judged here, not only the newest one at the next cycle.
        if trip in (Hold.NONE, Hold.CLUTCH_RELEASED) and held:
            reason = _pose_reason(frame, params)
            if reason != Hold.NONE:
                trip = reason
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
                arm_pending = want_armed
                coarse = False              # every arming starts at the fine scale
                events.append('arm requested' if want_armed else 'disarm requested')
            # Only once the arming has been accepted: a click on the frame that arms, or while
            # the arm request still waits for its posture check, must not start it at coarse.
            if (frame.scale_toggle and not buttons[3] and want_armed and state.want_armed
                    and not state.arm_pending):
                coarse = not coarse
                scale = params.translation_scale(coarse)
                events.append(f'translation scale: {"coarse" if coarse else "fine"} '
                              f'({scale:.2f})')
            if frame.calibrate and not buttons[1]:
                result, text = _try_calibrate(state, frame, params)
                events.append(text)
                if result is not None:
                    calibration = result
        buttons = (frame.grip, frame.calibrate, frame.arm, frame.scale_toggle)
    return replace(state, frame=frame, buttons=buttons, trip=trip, want_armed=want_armed,
                   arm_pending=arm_pending, coarse=coarse, calibration=calibration,
                   events=tuple(events))


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
    if not env.solver_ready:
        return Hold.SOLVER
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


def clamp_around(pose: Pose, centre: Pose, max_translation: float,
                 max_rotation: float) -> Tuple[Pose, bool, bool]:
    """
    Clamp ``pose`` to within a distance and an angle of ``centre``.

    Return the pose and whether the translation and the rotation were clamped. Position
    and orientation are clamped independently, each along its own straight path.
    """
    position, orientation = pose.p, pose.q
    offset = v_sub(position, centre.p)
    distance = v_norm(offset)
    moved = distance > max_translation
    if moved:
        position = v_add(centre.p, v_scale(offset, max_translation / distance))
    angle = q_between(centre.q, orientation)
    turned = angle > max_rotation
    if turned:
        orientation = q_slerp(centre.q, orientation, max_rotation / angle)
    return Pose(position, orientation), moved, turned


def clamp_offset(pose: Pose, tool0: Pose, params: GateParams) -> Tuple[Pose, Limit]:
    """Clamp ``pose`` onto the per-clutch offset limit around ``tool0`` (G-6, layer 2)."""
    clamped, moved, turned = clamp_around(pose, tool0, params.max_offset_translation,
                                          params.max_offset_rotation)
    limit = (Limit.OFFSET_TRANSLATION if moved else
             Limit.OFFSET_ROTATION if turned else Limit.NONE)
    return clamped, limit


def clamp_lead(pose: Pose, measured: Pose, params: GateParams) -> Tuple[Pose, bool]:
    """Clamp ``pose`` to within the lead limits of the measured tool pose (G-14)."""
    clamped, moved, turned = clamp_around(pose, measured, params.max_lead_translation,
                                          params.max_lead_rotation)
    return clamped, moved or turned


BLOCKED_FLOOR = 1e-4      # m/s: a smaller blocked tool velocity is not acted on


def absorb_blocked(pose: Pose, measured: Pose,
                   blocked: Sequence[float]) -> Tuple[Pose, bool]:
    """
    Remove the translation lead of ``pose`` over ``measured`` along ``blocked`` (G-14).

    Only a lead into the blocked direction is removed; a lead out of it, and every lead
    across it, is kept.
    """
    size = v_norm(tuple(blocked))
    if size <= BLOCKED_FLOOR:
        return pose, False
    unit = v_scale(tuple(blocked), 1.0 / size)
    lead = v_sub(pose.p, measured.p)
    along = v_dot(lead, unit)
    if along <= 0.0:
        return pose, False
    return Pose(v_sub(pose.p, v_scale(unit, along)), pose.q), True


def absorb(anchor: Pose, filtered: Pose, mapped_turn: Quat, before: Pose,
           after: Pose) -> Tuple[Pose, Pose]:
    """
    Move the mapping anchor and the filter state by the clamped-away lead (G-14).

    Afterwards the mapping yields ``after`` for the controller pose that yielded
    ``before``: the lead is absorbed, not stored.

    The mapping gives ``(anchor.p + dp, dq * anchor.q)``, where ``dq`` (``mapped_turn``) is
    the controller's rotation since engagement in the base frame. For the world-frame
    correction ``turn = after.q * before.q^-1`` the anchor rotation becomes
    ``dq^-1 * turn * dq * anchor.q``; the filter state is a world pose and takes ``turn``
    directly. Positions shift by ``after.p - before.p``.
    """
    shift = v_sub(after.p, before.p)
    turn = q_mul(after.q, q_conj(before.q))
    anchor_q = q_normalize(q_mul(q_mul(q_mul(q_conj(mapped_turn), turn), mapped_turn),
                                 anchor.q))
    return (Pose(v_add(anchor.p, shift), anchor_q),
            Pose(v_add(filtered.p, shift), q_normalize(q_mul(turn, filtered.q))))


def twist(previous: Pose, current: Pose, period: float) -> Tuple[float, ...]:
    """Return the twist taking ``previous`` to ``current`` in ``period`` (base frame)."""
    v = v_scale(v_sub(current.p, previous.p), 1.0 / period)
    w = v_scale(q_to_rotvec(q_mul(current.q, q_conj(previous.q))), 1.0 / period)
    return v + w


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
                   anchor=None, filtered=None, last_target=None, trip=Hold.NONE, latched=latched,
                   events=())


def _check_arming(state: GateState, env: Env, params: GateParams,
                  events: list) -> GateState:
    """
    Keep or refuse a pending arm request by the posture the arm is in.

    Near the wrist singularity the solver damps orientation and the wrist joints can take
    a branch the operator did not mean; teleop starts from a posture with room to move.
    Only the request is checked: once armed, the arm may go anywhere.
    """
    if not state.arm_pending:
        return state
    measure = env.arming_conditioning
    if params.min_arming_conditioning <= 0.0 or measure >= params.min_arming_conditioning:
        # The check is off, or the posture has room (NaN, unknown, fails the comparison).
        return replace(state, arm_pending=False)
    if math.isnan(measure):
        why = 'the posture is unknown (no joint state or kinematic chain yet)'
    else:
        why = (f'orientation conditioning {measure:.2f} is below '
               f'{params.min_arming_conditioning:.2f} (the wrist is near alignment)')
    events.append(f'arm refused: {why}; move the arm to the ready posture first')
    return replace(state, want_armed=False, arm_pending=False)


def tick(state: GateState, env: Env, params: GateParams) -> Tuple[GateState, Output]:
    """Run one target cycle; return the new state and what to publish."""
    events = list(state.events)
    state = _check_arming(state, env, params, events)
    if state.coarse and not state.want_armed:
        state = replace(state, coarse=False)    # disarmed by any path: the next arming is fine
    frame = state.frame
    age = env.now - frame.arrival if frame is not None else math.inf
    seq = frame.seq if frame is not None else 0
    grip = frame is not None and frame.grip
    latched = state.latched
    # The grip as last seen on a present controller; an absent one shows nothing.
    grip_seen = bool(state.buttons and state.buttons[0])
    if state.buttons is not None and not grip_seen:
        latched = Hold.NONE

    reason = _condition(state, env, params, age)

    def out(new_state: GateState, target: Optional[Pose], hold: Hold, limit: Limit,
            want_servo: bool, feedforward=None) -> Tuple[GateState, Output]:
        condition = Hold.STEP_REJECTED if hold == Hold.STEP_REJECTED else reason
        return new_state, Output(target, new_state.engaged, hold, limit, want_servo, age, seq,
                                 tuple(events), feedforward, condition)

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
                        tool0=env.tool_pose, anchor=env.tool_pose,
                        heading=state.calibration.heading,
                        scale=params.translation_scale(state.coarse),
                        filtered=env.tool_pose, last_target=env.tool_pose)
        events.append(f'clutch engaged at seq {frame.seq}')

    controller = apply_pivot(frame.controller, params.pivot_offset)
    scale = params.translation_scale(state.coarse)
    if scale != state.scale:
        # Switched while engaged: move the anchor so this cycle's controller pose maps to
        # the same target at the new scale; only motion from here on is scaled anew.
        state = replace(state, scale=scale,
                        anchor=rescale(state.anchor, controller, state.clutch, state.heading,
                                       state.scale, scale))
    raw = clutch_relative(controller, state.clutch, state.heading, state.anchor, state.scale)
    filtered = low_pass(state.filtered, raw, params)
    target, limit = bound(filtered, state.tool0, state.last_target, params)
    step = v_norm(v_sub(target.p, state.last_target.p))
    turn = q_between(state.last_target.q, target.q)
    if step > params.step_translation_limit or turn > params.step_rotation_limit:
        events.append(f'clutch released: target step {step * 1000.0:.1f} mm, '
                      f'{math.degrees(turn):.1f} deg beyond the per-cycle limit')
        return out(_released(state, Hold.STEP_REJECTED), None, Hold.STEP_REJECTED,
                   Limit.NONE, False)
    feedforward = twist(state.last_target, target, params.target_period)
    final, leading = clamp_lead(target, env.tool_pose, params)
    final, absorbed = absorb_blocked(final, env.tool_pose, env.blocked)
    leading = leading or absorbed
    if leading:
        limit = Limit.LEAD
        if not within_bounds(final, state.tool0, params):
            # G-6 outranks: the last accepted target is kept, and so is the arm.
            final, limit = state.last_target, Limit.KEEP_OUT
            feedforward = (0.0,) * 6
    anchor, filtered = state.anchor, filtered
    if final != target:
        mapped_turn = clutch_relative(controller, state.clutch, state.heading,
                                      Pose(state.anchor.p, IDENTITY_Q)).q
        anchor, filtered = absorb(state.anchor, filtered, mapped_turn, target, final)
    state = replace(state, anchor=anchor, filtered=filtered, last_target=final,
                    latched=Hold.NONE, events=())
    return out(state, final, Hold.NONE, limit, True, feedforward)
