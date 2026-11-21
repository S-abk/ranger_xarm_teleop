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
Helpers of the reset's stack tests (G-16).

Headset recordings with the right thumbstick held, a watcher of the reset's state, and the
measurement of a stop.

A recording is a list of phases ``(seconds, how)``: the stick's value and what else the frames
of that phase show (the controller missing, the headset looking away, an emulated position, no
headset pose, a reference-space reset, another XR session). Frames are numbered on through the
phases, so a phase never shows a sequence fault unless it asks for one.
"""

import itertools
import json
import math
import os
import sys
import threading
import time

from ranger_xarm_teleop_interfaces.msg import EpisodeState, ResetState, ResetWatchdog
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from stack_helpers import ARM_JOINTS, READY, SERVO_FIXTURES

sys.path.insert(0, SERVO_FIXTURES)
import make_fixtures as mf  # noqa: E402, I100

RESET_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                       depth=10, durability=DurabilityPolicy.VOLATILE)
EPISODE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
LIVENESS_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST, depth=1,
                          durability=DurabilityPolicy.VOLATILE)
_SESSIONS = itertools.count(0x5E5E0001)
STATE = {v: k[len('STATE_'):] for k, v in vars(ResetState).items() if k.startswith('STATE_')}
REASON = {v: k[len('REASON_'):] for k, v in vars(ResetState).items() if k.startswith('REASON_')}


def phase(seconds, stick=0.0, **how):
    """Return one phase: ``stick`` held for ``seconds`` with ``how`` (see ``write_recording``)."""
    return (float(seconds), dict(stick=float(stick), **how))


def write_recording(path, phases):
    """
    Write a JSONL recording of ``phases``; return the times (s from its first frame) they begin.

    ``how`` keys: ``stick`` (right thumbstick y, positive pulled back), ``controller`` (False:
    no right controller in the frame), ``look`` ('away': the headset turned 180 degrees from
    the controller), ``emulated`` (the controller's position emulated), ``head`` (False: no
    headset pose), ``reset_count``, ``session`` ('new': another XR session from this phase on).
    """
    session = next(_SESSIONS)
    lines, begins, i = [], [], 0
    for seconds, how in phases:
        begins.append(i / mf.RATE_HZ)
        if how.get('session') == 'new':
            session = next(_SESSIONS)
        for _ in range(int(round(seconds * mf.RATE_HZ))):
            right = None
            if how.get('controller', True):
                right = mf.hand(mf.START, mf.yaw(mf.CAL_YAW_DEG), False,
                                emulated=bool(how.get('emulated', False)))
                right['axes'] = [0.0, 0.0, 0.0, float(how['stick'])]
            head = None
            if how.get('head', True):
                look = mf.yaw(180.0) if how.get('look') == 'away' else (0.0, 0.0, 0.0, 1.0)
                head = mf.viewer(look)
            lines.append(mf.line(i, right, head, reset_count=int(how.get('reset_count', 0)),
                                 session_id=session))
            i += 1
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(lines) + '\n')
    return begins


class ResetWatcher:
    """The reset's state and the watchdog's sign of life, as the probe receives them."""

    def __init__(self, probe):
        self.probe = probe
        self.lock = threading.Lock()
        self.states = []            # (monotonic, ResetState)
        self.alive = []             # (monotonic, ResetWatchdog)
        probe.node.create_subscription(ResetState, '/teleop/reset/state', self._on_state,
                                       RESET_QOS)
        probe.node.create_subscription(ResetWatchdog, '/teleop/reset/watchdog', self._on_alive,
                                       LIVENESS_QOS)

    def _on_state(self, msg):
        with self.lock:
            self.states.append((time.monotonic(), msg))

    def _on_alive(self, msg):
        with self.lock:
            self.alive.append((time.monotonic(), msg))

    def since(self, t, after=None):
        """Return the states since ``t`` (and with an event count above ``after``)."""
        with self.lock:
            return [(when, s) for when, s in self.states
                    if when >= t and (after is None or s.event_count > after)]

    def count(self):
        """Return the newest event count; 0 when no state arrived."""
        last = self.last()
        return 0 if last is None else last.event_count

    def last(self):
        with self.lock:
            return self.states[-1][1] if self.states else None

    def first(self, t, state=None, reason=None, after=None):
        """Return (when, msg) of the first state since ``t`` that matches, or None."""
        for when, s in self.since(t, after):
            if (state is None or s.state == state) and (reason is None or s.reason == reason):
                return when, s
        return None

    def wait(self, t, state, timeout, reason=None, after=None):
        """
        Wait for a matching state since ``t``; return (when, msg).

        ``after``: only states with an event count above it, so that a state still repeated
        from an earlier event (the state is published at 10 Hz) does not count.
        """
        self.probe.wait_for(lambda: self.first(t, state, reason, after) is not None, timeout,
                            f'reset state {STATE[state]}'
                            + (f' ({REASON[reason]})' if reason is not None else '')
                            + f'; last: {describe(self.last())}')
        return self.first(t, state, reason, after)

    def goals_since(self, t):
        """Return the distinct goal ids the reset node announced since ``t``."""
        return {bytes(s.goal_id) for _, s in self.since(t) if s.goal_out and any(s.goal_id)}

    def alive_since(self, t):
        with self.lock:
            return [(when, m) for when, m in self.alive if when >= t]


def describe(state):
    """Say a ResetState in words."""
    if state is None:
        return 'none'
    return (f'{STATE.get(state.state, state.state)} / {REASON.get(state.reason, state.reason)}: '
            f'{state.reason_text}')


class EpisodeStandIn:
    """Publishes an EpisodeState at 5 Hz, as the episode node does (IDLE unless set)."""

    def __init__(self, probe):
        self.state = EpisodeState.STATE_IDLE
        self.pub = probe.node.create_publisher(EpisodeState, '/teleop/episode/state',
                                               EPISODE_QOS)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            self.pub.publish(EpisodeState(state=self.state))
            time.sleep(0.2)

    def close(self):
        self.running = False
        self.thread.join(1.0)


def stop_measure(probe, request_at, settle=1.5):
    """
    Return the stop after ``request_at``, measured on the joint states.

    The result is (joint travel in rad, time to rest in s, joint speed at the request in
    rad/s, tool travel in m).

    Rest is the last joint state that differs from the final one by more than 0.001 rad.
    """
    time.sleep(settle)
    timed = probe.timed_joints_since(request_at - 0.2)
    before = [(w, q) for w, q in timed if w < request_at]
    after = [(w, q) for w, q in timed if w >= request_at]
    assert after, 'no joint state after the stop request'
    first, final = after[0][1], after[-1][1]
    travel = max(max(abs(a - b) for a, b in zip(q, first)) for _, q in after)
    rest = after[0][0]
    for when, q in after:
        if max(abs(a - b) for a, b in zip(q, final)) > 0.001:
            rest = when
    speed = 0.0
    if before:
        (w0, q0), (w1, q1) = before[-1], after[0]
        if w1 - w0 > 1e-3:
            speed = max(abs(a - b) for a, b in zip(q1, q0)) / (w1 - w0)
    tools = [s.measured_tool_pose.position for w, s in probe.timed_states_since(request_at)
             if s.measured_tool_pose_valid]
    tool = 0.0
    if tools:
        tool = max(math.dist((p.x, p.y, p.z), (tools[0].x, tools[0].y, tools[0].z))
                   for p in tools)
    return travel, max(0.0, rest - request_at), speed, tool


def at(probe, posture, tolerance=0.01):
    """Return True when the arm's joints are within ``tolerance`` of ``posture``."""
    q = probe.joint_positions()
    return q is not None and all(abs(a - b) <= tolerance for a, b in zip(q, posture))


def off_ready(delta):
    """Return the ready posture plus ``delta`` per joint."""
    return [r + d for r, d in zip(READY, delta)]


def contact_paths():
    """Return the recorded contact postures of the live session (joint paths)."""
    with open(os.path.join(SERVO_FIXTURES, 'session3_contact_paths.jsonl'),
              encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


__all__ = ['ARM_JOINTS', 'READY', 'phase', 'write_recording', 'ResetWatcher', 'describe',
           'EpisodeStandIn', 'stop_measure', 'at', 'off_ready', 'contact_paths', 'STATE',
           'REASON']
