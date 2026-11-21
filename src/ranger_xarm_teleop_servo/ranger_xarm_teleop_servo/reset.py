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
The reset: a hold-to-run return to a named posture from the headset (G-16), as pure logic.

The operator holds the right thumbstick pulled back; after a short dwell a MoveIt goal to the
posture goes out, and the arm moves only while the stick stays held and the input is fresh.

* **Hold-to-run (G-16).** Releasing the stick, stale input (G-3, which is also how a relay
  disconnect shows), a missing controller (no value for the stick) and a session or sequence
  fault each cancel the goal. Pose-only conditions do not: the controller out of view, an
  absent or emulated pose, the headset's pose, a reference-space reset. The stick is not a
  pose, and the operator must be free to watch the arm.
* **Only when it may.** A request is accepted only with teleop disarmed and MoveIt owning the
  arm (``session.may_recover``, the rule of the recovery tools), no episode recording or in
  review, the feature on, a simulator's robot description, a live watchdog and move_group
  ready. Otherwise it is refused with its reason, after the full dwell, so that a brush of
  the stick never vibrates.
* **Never by itself.** A start needs a press: the stick seen released, then pulled, with the
  conditions true at the press and all through the dwell. A stick held through a disarm, or
  through any stop, needs to be released and pulled again; a held stick never resumes after
  a fault.
* **The stop.** A stop cancels the goal by its own id (``goal_stop.GoalStop``, asked again
  until the goal has ended) and watches the joints until they rest. A goal whose answer did
  not come is unknown, not refused: it is stopped the same way. "Nothing moved" is said only
  when the joints were watched from before the goal went out and did not move.
* **A goal given up is still watched.** A goal whose stop ended without it ever being seen
  ended (the server only ever answered that it knew no such goal) is kept by its id. If it
  turns up after all, accepted or listed as running, in whatever order and even after a newer
  goal went out, it is cancelled by its id until it has ended and the joints rest, a motion in
  progress is stopped too, the heartbeat names it (so the watchdog covers it), and no new goal
  goes out meanwhile.

The node (``reset_node.py``) feeds this class with the input samples, what it knows of the
world at every cycle, the joint states and move_group's answers, and carries out what it asks:
send a goal, send a cancel request. No ROS imports.
"""

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ranger_xarm_teleop_bridge.joy_layout import AXIS_NAMES

from .calls import JointWatch, STILL_RAD
from .gate import frame_fault, Hold
from .goal_stop import ACTIVE, GoalStop, TERMINAL
from .params import INPUT_TIMEOUT_S
from .session import may_recover, NO_TELEOP_AFTER_S, OWNER_MOVEIT, STATE_FRESH_S

STICK_AXIS = AXIS_NAMES.index('right_thumbstick_y')      # positive when pulled back
TRIGGER_AXIS = AXIS_NAMES.index('right_trigger')          # NaN when there is no controller

# ResetState.STATE_* (checked against the message file by the unit tests).
IDLE, WAITING, REFUSED, PLANNING, MOVING, STOPPING, STOPPED, ARRIVED = range(8)
STATE_NAMES = ('IDLE', 'WAITING', 'REFUSED', 'PLANNING', 'MOVING', 'STOPPING', 'STOPPED',
               'ARRIVED')

# ResetState.REASON_*.
NONE = 0
OFF, NOT_SIMULATION, ARMED, NOT_MOVEIT_OWNER, ARMED_UNKNOWN = 1, 2, 3, 4, 5
RECORDING, REVIEW, EPISODE_UNKNOWN, WATCHDOG_ABSENT, MOVEIT_NOT_READY = 6, 7, 8, 9, 10
PLAN_FAILED, START_IN_COLLISION, BUSY, JOINTS_UNKNOWN = 11, 12, 13, 14
REACHED, ALREADY_THERE = 20, 21
RELEASED, INPUT_STALE, CONTROLLER_MISSING, SESSION_CHANGED, SEQ_BACKWARDS = 30, 31, 32, 33, 34
TELEOP_ARMED, OWNER_CHANGED, EPISODE_STARTED, WATCHDOG_LOST, HEARTBEAT_LATE = 35, 36, 37, 38, 39
CANCELLED_ELSEWHERE, MOVEIT_FAILED, NO_ANSWER, STATE_UNKNOWN = 40, 41, 42, 43

REASON_TEXT = {
    NONE: '',
    OFF: 'the reset is off by configuration',
    NOT_SIMULATION: "the robot description is not a simulator's (simulation only)",
    ARMED: 'teleop is armed: disarm first',
    NOT_MOVEIT_OWNER: 'the command mux does not report MoveIt as the owner of the arm',
    ARMED_UNKNOWN: 'a teleop node is running and no fresh state of it is at hand',
    RECORDING: 'an episode is being recorded',
    REVIEW: 'an episode waits to be kept or discarded',
    EPISODE_UNKNOWN: 'an episode node is running and no fresh state of it is at hand',
    WATCHDOG_ABSENT: 'no fresh sign of life from the reset watchdog',
    MOVEIT_NOT_READY: "move_group's action and its cancel path are not ready",
    PLAN_FAILED: 'MoveIt found no plan to the posture; the joints did not move',
    START_IN_COLLISION: ("MoveIt considers the arm's posture in collision; the joints did not "
                         'move: disarm and free the arm with tools/retreat_arm.py'),
    BUSY: 'the previous goal is still being stopped',
    JOINTS_UNKNOWN: 'no fresh joint state of the arm',
    REACHED: 'the arm reached the posture',
    ALREADY_THERE: 'the arm was already at the posture; no goal was sent',
    RELEASED: 'the control was released',
    INPUT_STALE: 'no fresh headset input',
    CONTROLLER_MISSING: 'no value for the control: the right controller is missing',
    SESSION_CHANGED: 'the XR session changed',
    SEQ_BACKWARDS: "the input's sequence or time went backwards",
    TELEOP_ARMED: 'teleop was armed during the motion',
    OWNER_CHANGED: 'the command mux stopped reporting MoveIt as the owner',
    EPISODE_STARTED: 'the episode state left idle during the motion',
    WATCHDOG_LOST: "the watchdog's sign of life stopped during the motion",
    HEARTBEAT_LATE: "the watchdog cancelled the goal: this node's heartbeat was late",
    CANCELLED_ELSEWHERE: 'the goal was cancelled by someone else',
    MOVEIT_FAILED: 'move_group ended the goal with an error',
    NO_ANSWER: 'move_group did not answer in time',
    STATE_UNKNOWN: 'a state the motion depends on went stale',
}

# What a stall of this node can make it see first when it runs again (``_heartbeat_late``).
STALL_REASONS = (INPUT_STALE, CANCELLED_ELSEWHERE, NO_ANSWER, WATCHDOG_LOST)

# ResetState.EVENT_*.
EVENT_NONE, EVENT_START, EVENT_ARRIVED, EVENT_REFUSED, EVENT_STOPPED, EVENT_RELEASED = range(6)

# EpisodeState.STATE_*: a reset may run only while no episode is recording or in review.
EPISODE_RECORDING, EPISODE_REVIEW = 2, 3

# MoveItErrorCodes.
MOVEIT_SUCCESS = 1
MOVEIT_PREEMPTED = -7
MOVEIT_START_IN_COLLISION = -10
# Errors of planning, before anything is executed.
MOVEIT_PLANNING_ERRORS = (-1, -2, -6, -11, -12, -13, -14, -15, -16, -17, -26, -27, -28, -31)


def controller_missing(joy) -> bool:
    """
    Return True when the frame carries no value for the reset's control.

    The one place that reads the bridge's "no controller" mark: no Joy, or too few axes, or
    the right trigger axis is NaN (the bridge sends NaN there for a hand without a controller;
    it sends 0.0 on the stick axes then), or the stick value itself is not finite. If the
    bridge later marks the stick axes the same way as the trigger, only this changes.
    """
    if joy is None or len(joy.axes) <= max(STICK_AXIS, TRIGGER_AXIS):
        return True
    return not (math.isfinite(joy.axes[TRIGGER_AXIS]) and math.isfinite(joy.axes[STICK_AXIS]))


@dataclass(frozen=True)
class Sample:
    """One headset frame as the reset sees it: when, which frame, and the stick (None: missing)."""

    arrival: float
    session_id: int
    seq: int
    xr_time: float
    stick: Optional[float]


def sample_of(status, joy, arrival: float) -> Sample:
    """Return the reset's sample of one paired frame (status and its joy, or None)."""
    stick = None if controller_missing(joy) else float(joy.axes[STICK_AXIS])
    return Sample(arrival, int(status.session_id), int(status.seq),
                  float(status.xr_time_ms) / 1000.0, stick)


@dataclass(frozen=True)
class Params:
    """The reset's settings. ``speed_scaling`` is required (G-16 value, PENDING: no default)."""

    speed_scaling: float
    press_threshold: float = 0.8        # the stick counts as pulled from here
    hold_threshold: float = 0.5         # and as held down to here
    dwell_s: float = 0.5                # held this long before anything goes out
    input_timeout_s: float = INPUT_TIMEOUT_S          # G-3, the gate's own value
    watchdog_timeout_s: float = 0.5     # the watchdog's sign of life is fresh this long
    state_fresh_s: float = STATE_FRESH_S
    absent_after_s: float = NO_TELEOP_AFTER_S   # a node is believed absent only after this
    at_posture_rad: float = 0.01        # every joint this close: at the posture
    goal_answer_s: float = 2.0          # move_group answers a goal (and a result) within this
    cancel_answer_s: float = 0.5        # a cancel request without an answer is asked again

    def check(self) -> None:
        """Raise ValueError for settings that cannot work."""
        if not (math.isfinite(self.speed_scaling) and 0.0 < self.speed_scaling <= 1.0):
            raise ValueError('reset_speed_scaling must be within (0, 1]')
        if not 0.0 < self.hold_threshold < self.press_threshold <= 1.0:
            raise ValueError('0 < reset_hold_threshold < reset_press_threshold <= 1 is needed')
        if not (math.isfinite(self.dwell_s) and self.dwell_s >= 0.0):
            raise ValueError('reset_dwell_s must be finite and not negative')


@dataclass(frozen=True)
class World:
    """What the node knows at a cycle besides the input. Freshness is judged by the node."""

    now: float
    since_start: float                  # s since the node started
    enabled: bool
    simulator: bool                     # the robot description was seen and is a simulator's
    owner: Optional[int]                # the mux's fresh report (None: none fresh)
    armed: Optional[bool]               # the teleop node's fresh state (None: none fresh)
    teleop_running: bool
    episode: Optional[int]              # a fresh EpisodeState.state (None: none fresh)
    episode_running: bool
    watchdog_alive: bool                # a fresh sign of life from the watchdog
    watchdog_stopping: bool = False     # the watchdog is cancelling a goal
    watchdog_goal: bytes = bytes(16)    # the goal it is cancelling or last cancelled
    moveit_ready: bool = False          # move_group's action and cancel path did a round trip
    joints: Optional[Tuple[float, ...]] = None      # fresh arm joint positions


def refusal(world: World, absent_after_s: float = NO_TELEOP_AFTER_S) -> int:
    """Return why a reset may not start now (a REASON_*), or NONE."""
    if not world.enabled:
        return OFF
    if not world.simulator:
        return NOT_SIMULATION
    running = world.teleop_running or world.since_start < absent_after_s
    ok, _ = may_recover(world.owner, world.armed, running)
    if not ok:
        if world.armed:
            return ARMED
        if world.owner != OWNER_MOVEIT:
            return NOT_MOVEIT_OWNER
        return ARMED_UNKNOWN
    episode = _episode_reason(world, EPISODE_UNKNOWN, absent_after_s)
    if episode != NONE:
        return episode
    if not world.watchdog_alive:
        return WATCHDOG_ABSENT
    if world.watchdog_stopping:
        return BUSY
    if not world.moveit_ready:
        return MOVEIT_NOT_READY
    if world.joints is None:
        return JOINTS_UNKNOWN
    return NONE


def _episode_reason(world: World, unknown: int,
                    absent_after_s: float = NO_TELEOP_AFTER_S) -> int:
    if world.episode is None:
        running = world.episode_running or world.since_start < absent_after_s
        return unknown if running else NONE
    if world.episode == EPISODE_RECORDING:
        return RECORDING
    if world.episode == EPISODE_REVIEW:
        return REVIEW
    return NONE


def motion_fault(world: World) -> int:
    """Return what in the world stops a motion in progress (a REASON_*), or NONE."""
    ok, _ = may_recover(world.owner, world.armed, world.teleop_running)
    if not ok:
        if world.armed:
            return TELEOP_ARMED
        if world.owner is not None and world.owner != OWNER_MOVEIT:
            return OWNER_CHANGED
        return STATE_UNKNOWN
    episode = _episode_reason(world, STATE_UNKNOWN)
    if episode in (RECORDING, REVIEW):
        return EPISODE_STARTED
    if episode != NONE:
        return episode
    if not world.watchdog_alive:
        return WATCHDOG_LOST
    return NONE


class StopMeter:
    """Joint travel and time after a stop request, until rest."""

    def __init__(self, names: Sequence[str], at: float) -> None:
        """Start measuring at ``at`` (the stop request)."""
        self.names, self.at = list(names), at
        self.samples: List[Tuple[float, List[float]]] = []

    def sample(self, when: float, positions) -> None:
        """Take one joint state (a mapping of joint name to position) after the request."""
        if when >= self.at and all(n in positions for n in self.names):
            self.samples.append((when, [float(positions[n]) for n in self.names]))

    def result(self) -> Tuple[float, float]:
        """Return (largest joint travel since the request in rad, time to rest in s); NaN, NaN."""
        if len(self.samples) < 2:
            return math.nan, math.nan
        first, final = self.samples[0][1], self.samples[-1][1]
        travel = max(max(abs(a - b) for a, b in zip(q, first)) for _, q in self.samples)
        rest = self.samples[0][0]
        for when, q in self.samples:
            if max(abs(a - b) for a, b in zip(q, final)) > STILL_RAD:
                rest = when
        return travel, max(0.0, rest - self.at)


class Reset:
    """The reset's state machine; see the module text."""

    def __init__(self, params: Params, posture: str, target: Sequence[float],
                 joint_names: Sequence[str], make_id: Callable[[], bytes]) -> None:
        """Go to ``target`` (named ``posture``) on ``joint_names``; ``make_id`` gives goal ids."""
        params.check()
        if len(target) != len(joint_names):
            raise ValueError('the posture needs one value per joint')
        self.params, self.posture = params, posture
        self.target = tuple(float(v) for v in target)
        self.names = list(joint_names)
        self.make_id = make_id
        self.state, self.reason = IDLE, NONE
        self.detail = ''                    # more words for the log (codes, cancel report)
        self.moveit_error_code = 0
        # Input.
        self.last: Optional[Sample] = None
        self.fault = NONE                   # a session or sequence fault since the last cycle
        self.need_release = True            # a start needs the stick seen released first
        self.dwell_since: Optional[float] = None
        self.dwell_refusal = NONE           # the first refusal seen during the dwell
        # The goal.
        self.goal_id: Optional[bytes] = None
        self.goal_out = False
        self.sent_at: Optional[float] = None
        self.answer: Optional[str] = None   # 'accepted', 'rejected', None while unknown
        self.status: Optional[int] = None   # the newest status of the goal
        self.terminal_at: Optional[float] = None
        self.stop: Optional[GoalStop] = None
        self.stop_reason = NONE
        self.stop_at: Optional[float] = None
        self.sent_watch = JointWatch(self.names)
        self.stop_watch = JointWatch(self.names)
        self.meter: Optional[StopMeter] = None
        self.stop_distance, self.stop_time = math.nan, math.nan
        # Goals given up without being seen ended: id -> their stop once they turned up.
        self.orphans: Dict[bytes, Optional[GoalStop]] = {}
        self.orphan_watch = JointWatch(self.names)
        # Events.
        self.event_count, self.last_event, self.last_event_reason = 0, EVENT_NONE, NONE
        self.last_event_wall = math.nan
        self.log: List[str] = []            # lines for the node to log, taken by the node
        self._wall_offset = math.nan        # host wall clock minus the node's clock
        self._watchdog_noted: Optional[bytes] = None   # the goal whose watchdog cancel was seen

    # ---- input ----

    def on_sample(self, sample: Sample) -> None:
        """Take one headset frame (every frame, in order)."""
        prev = self.last
        if prev is not None:
            fault = frame_fault(prev, sample)
            if fault != Hold.NONE and self.fault == NONE:
                self.fault = (SESSION_CHANGED if fault == Hold.SESSION_CHANGED
                              else SEQ_BACKWARDS)
        self.last = sample

    def _input(self, now: float) -> Tuple[int, Optional[float]]:
        """Return (what breaks the hold now or NONE, the stick value)."""
        if self.last is None or not now - self.last.arrival <= self.params.input_timeout_s:
            return INPUT_STALE, None
        if self.fault != NONE:
            return self.fault, None
        if self.last.stick is None:
            return CONTROLLER_MISSING, None
        return NONE, self.last.stick

    def on_joints(self, when: float, positions) -> None:
        """Take one joint state (a mapping of joint name to position)."""
        self.sent_watch.sample(when, positions)
        self.stop_watch.sample(when, positions)
        self.orphan_watch.sample(when, positions)
        if self.meter is not None:
            self.meter.sample(when, positions)

    # ---- move_group ----

    def on_goal_answer(self, now: float, goal_id: bytes, accepted: bool) -> None:
        """Take move_group's answer to a goal."""
        if goal_id in self.orphans:
            if accepted:
                self._turned_up(now, goal_id, 'accepted')
            return
        if goal_id != self.goal_id or self.answer is not None:
            return
        self.answer = 'accepted' if accepted else 'rejected'
        if self.state not in (PLANNING, MOVING):
            return
        if accepted:
            self._event(EVENT_START, NONE, now)
        else:
            self.goal_out = False
            self._end(REFUSED, MOVEIT_NOT_READY, now, 'move_group rejected the goal')

    def on_status(self, now: float, goal_id: bytes, status: int) -> None:
        """Take a status of a goal from move_group's status topic (any goal of the server)."""
        if goal_id in self.orphans:
            stop = self.orphans[goal_id]
            if stop is not None:
                stop.status(now, status)
            elif status in ACTIVE:
                self._turned_up(now, goal_id, 'listed as running')
                self.orphans[goal_id].status(now, status)       # what showed it is kept
            elif status in TERMINAL:
                del self.orphans[goal_id]           # it has ended: nothing to watch
            return
        if goal_id != self.goal_id:
            return
        self.status = int(status)
        if self.stop is not None:
            self.stop.status(now, status)
        if status in TERMINAL and self.terminal_at is None:
            self.terminal_at = now
        if status in ACTIVE and self.answer is None and self.state in (PLANNING, MOVING):
            # Listed by the server, the answer to the goal lost on its way: still unknown.
            self.log.append('move_group lists the goal although its answer has not arrived')

    def _turned_up(self, now: float, goal_id: bytes, how: str) -> None:
        """Cancel a goal given up as never arrived that turns up, and stop what moves now."""
        if self.orphans.get(goal_id) is not None:
            return                                  # already being cancelled
        self.need_release = True                    # whatever was held: a new press next
        if not any(stop is not None for stop in self.orphans.values()):
            self.orphan_watch.mark(now)
        stop = GoalStop(goal_id, self.params.cancel_answer_s)
        self.orphans[goal_id] = stop
        self.log.append(f'goal {goal_id.hex()}, given up as never arrived, was {how}: '
                        'cancelling it')
        if self.state in (PLANNING, MOVING):
            self._begin_stop(now, NO_ANSWER, f'an earlier goal was {how} after it was given up')
        elif self.state != STOPPING:
            self.state, self.reason = STOPPED, NO_ANSWER
            self.detail = f'an earlier goal was {how} after it was given up; cancelling it'
            self._event(EVENT_STOPPED, NO_ANSWER, now)

    def orphan_active(self) -> bool:
        """Return True while a goal given up earlier is being cancelled."""
        return any(stop is not None for stop in self.orphans.values())

    def heartbeat_goal(self) -> Tuple[bool, bytes]:
        """Return (a goal is out, its id) for the heartbeat: the current goal, or an orphan's."""
        if self.goal_out and self.goal_id is not None:
            return True, self.goal_id
        for goal_id, stop in self.orphans.items():
            if stop is not None:
                return True, goal_id
        return False, bytes(16)

    def on_other_status(self) -> None:
        """Note a status message that did not list the current goal."""
        if self.stop is not None:
            self.stop.other_status()
        for stop in self.orphans.values():
            if stop is not None:
                stop.other_status()

    def on_feedback(self, goal_id: bytes, state: str) -> None:
        """Take move_group's feedback (its state: PLANNING, MONITOR while executing, IDLE)."""
        if goal_id == self.goal_id and self.state == PLANNING and state == 'MONITOR':
            self.state = MOVING

    def on_result(self, now: float, goal_id: bytes, error_code: int) -> None:
        """Take the goal's result (move_group's error code)."""
        if goal_id != self.goal_id or not self.goal_out:
            return
        self.moveit_error_code = int(error_code)
        if self.state == STOPPING:
            self.goal_out = self.stop is not None and not self.stop.ended_by
            return
        if self.state not in (PLANNING, MOVING):
            return
        motion = self.sent_watch.motion(now)
        unmoved = motion.known and motion.from_before and motion.moved < STILL_RAD
        if error_code == MOVEIT_SUCCESS:
            self.goal_out = False
            self._end(ARRIVED, REACHED, now)
        elif error_code in (MOVEIT_START_IN_COLLISION,) + MOVEIT_PLANNING_ERRORS and unmoved:
            self.goal_out = False
            reason = (START_IN_COLLISION if error_code == MOVEIT_START_IN_COLLISION
                      else PLAN_FAILED)
            self._end(REFUSED, reason, now, f'move_group error code {error_code}; '
                      + motion.text())
        elif error_code == MOVEIT_PREEMPTED:
            self._begin_stop(now, CANCELLED_ELSEWHERE, f'move_group error code {error_code}')
        else:
            self._begin_stop(now, MOVEIT_FAILED, f'move_group error code {error_code}')

    # ---- cancel requests (the node sends them) ----

    def _stop_of(self, goal_id: bytes) -> Optional[GoalStop]:
        if self.stop is not None and goal_id == self.goal_id:
            return self.stop
        return self.orphans.get(goal_id)

    def cancel_due(self, now: float) -> Optional[bytes]:
        """Return the goal id to send a cancel request for now, or None (one per call)."""
        if self.stop is not None and self.stop.due(now):
            return self.goal_id
        for goal_id, stop in self.orphans.items():
            if stop is not None and stop.due(now):
                return goal_id
        return None

    def cancel_sent(self, now: float, goal_id: bytes) -> None:
        """Note that a cancel request for ``goal_id`` went out."""
        stop = self._stop_of(goal_id)
        if stop is not None:
            stop.sent(now)

    def cancel_answered(self, now: float, goal_id: bytes, code: int, cancelling: int) -> None:
        """Take the answer to a cancel request for ``goal_id``."""
        stop = self._stop_of(goal_id)
        if stop is not None:
            stop.answered(now, code, cancelling)

    def cancel_timed_out(self, goal_id: bytes) -> None:
        """Note that the open cancel request for ``goal_id`` got no answer in time."""
        stop = self._stop_of(goal_id)
        if stop is not None:
            stop.timed_out()

    # ---- the cycle ----

    def tick(self, now: float, world: World, wall: float) -> Optional[bytes]:
        """
        Run one cycle; return the id of a goal to send now, or None.

        The node announces the returned id in its heartbeat and state before it sends the goal.
        """
        self._wall_offset = wall - now
        broken, stick = self._input(now)
        self.fault = NONE
        if broken != NONE:
            self.need_release = True
        elif stick < self.params.hold_threshold:
            self.need_release = False
        held = broken == NONE and stick >= self.params.hold_threshold
        self._watch_orphans(now)
        self._heartbeat_late(world)
        if self.state in (PLANNING, MOVING):
            self._watch_goal(now, world, broken, held)
            return None
        if self.state == STOPPING:
            self._watch_stop(now)
            return None
        if self.state == WAITING:
            return self._dwell(now, world, held)
        # IDLE, REFUSED, STOPPED, ARRIVED: a press starts the dwell.
        if broken == NONE and not self.need_release and stick >= self.params.press_threshold:
            self.state, self.reason, self.detail = WAITING, NONE, ''
            self.dwell_since = now
            self.dwell_refusal = refusal(world, self.params.absent_after_s)
            return self._dwell(now, world, True)
        return None

    def _dwell(self, now: float, world: World, held: bool) -> Optional[bytes]:
        if not held:
            self.state, self.reason = IDLE, NONE        # let go early: no request, no event
            self.need_release = True
            return None
        if self.dwell_refusal == NONE:
            self.dwell_refusal = refusal(world, self.params.absent_after_s)
        if now - self.dwell_since < self.params.dwell_s:
            return None
        self.need_release = True                        # whatever happens, a new press next
        why = self.dwell_refusal
        if why == NONE and self.orphan_active():
            why = BUSY
        if why != NONE:
            self._end(REFUSED, why, now)
            return None
        if all(abs(a - b) <= self.params.at_posture_rad
               for a, b in zip(world.joints, self.target)):
            self._end(ARRIVED, ALREADY_THERE, now)
            return None
        goal_id = bytes(self.make_id())
        if len(goal_id) != 16 or not any(goal_id):
            raise ValueError('a goal id is 16 bytes and never zero')
        self.goal_id, self.goal_out, self.sent_at = goal_id, True, now
        self.answer, self.status, self.terminal_at = None, None, None
        self.stop, self.stop_reason, self.moveit_error_code = None, NONE, 0
        self.sent_watch.mark(now)                       # motion is counted from before the goal
        self.state, self.reason, self.detail = PLANNING, NONE, ''
        return goal_id

    def _watch_goal(self, now: float, world: World, broken: int, held: bool) -> None:
        if world.watchdog_goal == self.goal_id:          # the watchdog cancelled this goal
            self._begin_stop(now, HEARTBEAT_LATE)
            return
        if broken != NONE:
            self._begin_stop(now, broken)
            return
        if not held:
            self._begin_stop(now, RELEASED)
            return
        fault = motion_fault(world)
        if fault != NONE:
            self._begin_stop(now, fault)
            return
        if self.answer is None and now - self.sent_at >= self.params.goal_answer_s:
            self._begin_stop(now, NO_ANSWER, 'no answer to the goal within '
                             f'{self.params.goal_answer_s:g} s')
            return
        if self.status in TERMINAL and now - self.terminal_at >= self.params.goal_answer_s:
            # Listed as ended, and its result never came.
            at = world.joints is not None and all(
                abs(a - b) <= self.params.at_posture_rad
                for a, b in zip(world.joints, self.target))
            if self.status == 4 and at:
                self.goal_out = False
                self._end(ARRIVED, REACHED, now, 'listed as succeeded; its result was lost')
            elif self.status == 5:
                self._begin_stop(now, CANCELLED_ELSEWHERE, 'listed as cancelled')
            else:
                self._begin_stop(now, NO_ANSWER, 'listed as ended; its result was lost')
            return
        if self.state == PLANNING and self.sent_watch.motion(now).moved >= STILL_RAD:
            self.state = MOVING

    def _begin_stop(self, now: float, reason: int, detail: str = '') -> None:
        self.stop = GoalStop(self.goal_id, self.params.cancel_answer_s)
        if self.status is not None:
            self.stop.status(now, self.status)
        self.stop_reason, self.stop_at = reason, now
        self.stop_watch.mark(now)
        self.meter = StopMeter(self.names, now)
        if self.stop_watch.reference is not None:
            self.meter.sample(now, dict(zip(self.names, self.stop_watch.reference)))
        self.state, self.reason, self.detail = STOPPING, reason, detail
        self.need_release = True
        self._event(EVENT_RELEASED if reason == RELEASED else EVENT_STOPPED, reason, now)
        self.log.append(f'stopping: {REASON_TEXT[reason]}' + (f' ({detail})' if detail else ''))

    def _heartbeat_late(self, world: World) -> None:
        """
        Name the stop "heartbeat late" once the watchdog says it cancelled this goal.

        The watchdog cancels only when this node's heartbeat was late (a stall, as when a
        process joins the graph). What this node then sees first when it runs again is one of
        ``STALL_REASONS`` (stale input, move_group's "preempted", no answer in time, the
        watchdog's sign of life late); such a stop is renamed. A stop that began for another
        cause keeps it. The event was already counted (and felt); only its reason changes.
        """
        if (self.goal_id is None or world.watchdog_goal != self.goal_id
                or self.state not in (STOPPING, STOPPED) or self.reason == HEARTBEAT_LATE
                or self._watchdog_noted == self.goal_id):
            return
        self._watchdog_noted = self.goal_id
        if self.reason not in STALL_REASONS:
            # The stop began for its own cause (a release, teleop armed, ...): that stays what
            # the operator is told; the watchdog's cancel during the stop is only logged.
            self.log.append(f'the watchdog also cancelled goal {self.goal_id.hex()} during the '
                            f'stop ({REASON_TEXT[self.reason]})')
            return
        self.log.append(f'the watchdog cancelled goal {self.goal_id.hex()}: the stop '
                        f'({REASON_TEXT[self.reason]}) is a late heartbeat')
        self.reason = self.stop_reason = HEARTBEAT_LATE
        self.last_event_reason = HEARTBEAT_LATE

    def _watch_orphans(self, now: float) -> None:
        motion = self.orphan_watch.motion(now)
        for goal_id, stop in list(self.orphans.items()):
            if stop is None:
                continue
            stop.tick(now)
            if stop.finished(motion):
                self.log.append(f'earlier goal {goal_id.hex()} stopped: {stop.report(motion)}')
                if any(s in (4, 5, 6) for s in stop.statuses):
                    del self.orphans[goal_id]
                else:
                    self.orphans[goal_id] = None    # still never seen ended: keep watching
        while len(self.orphans) > 32:               # the oldest given-up goals are forgotten
            oldest = next(k for k, v in self.orphans.items() if v is None) \
                if any(v is None for v in self.orphans.values()) else None
            if oldest is None:
                break
            del self.orphans[oldest]

    def _watch_stop(self, now: float) -> None:
        self.stop.tick(now)
        if self.stop.ended_by:
            self.goal_out = False
        motion = self.stop_watch.motion(now)
        if not self.stop.finished(motion):
            return
        if (self.terminal_at is None and self.moveit_error_code == 0
                and self.answer != 'rejected'):
            # Never seen ended: if it turns up after all, it is cancelled then.
            self.orphans[self.goal_id] = None
        self.stop_distance, self.stop_time = self.meter.result()
        report = self.stop.report(motion)
        self.log.append(f'stopped ({REASON_TEXT[self.stop_reason]}): {report}; '
                        f'stop distance {self.stop_distance:.4f} rad, time {self.stop_time:.2f} s')
        self.state, self.reason = STOPPED, self.stop_reason
        self.detail = (self.detail + '; ' if self.detail else '') + report
        self.stop = None

    def _end(self, state: int, reason: int, now: float, detail: str = '') -> None:
        self.state, self.reason, self.detail = state, reason, detail
        self.need_release = True
        event = {REFUSED: EVENT_REFUSED, ARRIVED: EVENT_ARRIVED}[state]
        self._event(event, reason, now)
        self.log.append(f'{STATE_NAMES[state].lower()}: {REASON_TEXT[reason]}'
                        + (f' ({detail})' if detail else ''))

    def _event(self, event: int, reason: int, now: float) -> None:
        self.event_count += 1
        self.last_event, self.last_event_reason = event, reason
        self.last_event_wall = now + self._wall_offset

    def take_log(self) -> List[str]:
        """Return the lines to log since the last call."""
        lines, self.log = self.log, []
        return lines

    def reason_text(self) -> str:
        """Return the reason in words, with the detail."""
        text = REASON_TEXT.get(self.reason, f'reason {self.reason}')
        return text + (f' ({self.detail})' if self.detail else '')
