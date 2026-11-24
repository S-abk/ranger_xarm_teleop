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
Helpers of the base's stack tests (G-19).

Headset recordings built from phases with both controllers, and a watcher of the base's
topics (its state, both commands, odometry, the operator's vibrations and panel).

A phase is ``(seconds, how)``. ``how`` keys: ``grip`` (left grip held), ``left`` (left stick
x, y), ``right`` (right stick x, y), ``rgrip`` (right grip held), ``move`` (the right
controller's velocity, m/s in WebXR axes, while the phase lasts), ``a``, ``b`` (right A, B
held), ``x``, ``y`` (left X, Y held: the episode's controls), ``left_missing`` (no left
controller), ``session`` ('new': another XR session from
this phase on), ``back`` (the phase's first frame goes this many frames back), ``gap`` (no
frames for the phase: the input goes stale).
"""

import itertools
import math
import os
import sys
import threading
import time

from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from ranger_xarm_teleop_interfaces.msg import BaseState, Haptic, OperatorStatus
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from stack_helpers import replay_file, SERVO_FIXTURES

sys.path.insert(0, SERVO_FIXTURES)
import make_fixtures as mf  # noqa: E402, I100

QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                 depth=10, durability=DurabilityPolicy.VOLATILE)
PANEL_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                       depth=1, durability=DurabilityPolicy.VOLATILE)
LEFT_POS = (-0.25, 1.30, -0.45)
_SESSIONS = itertools.count(0x5BA5E001)
B = BaseState
REASON = {v: k[len('REASON_'):] for k, v in vars(B).items() if k.startswith('REASON_')}
STATE = {v: k[len('STATE_'):] for k, v in vars(B).items() if k.startswith('STATE_')}


def phase(seconds, **how):
    """Return one phase: ``how`` held for ``seconds``."""
    return (float(seconds), how)


def write_recording(path, phases):
    """Write a JSONL recording of ``phases``; return the times (s from its start) they begin."""
    session = next(_SESSIONS)
    lines, begins = [], []
    i = 0                       # frame index: time and t_recv
    seq = 0
    right_pos = list(mf.START)
    for seconds, how in phases:
        begins.append(i / mf.RATE_HZ)
        if how.get('session') == 'new':
            session = next(_SESSIONS)
            seq = 0
        if how.get('back'):
            seq -= int(how['back'])
        count = int(round(seconds * mf.RATE_HZ))
        if how.get('gap'):
            i += count
            seq += count
            continue
        move = how.get('move', (0.0, 0.0, 0.0))
        for _ in range(count):
            right_pos = [right_pos[k] + move[k] / mf.RATE_HZ for k in range(3)]
            right = mf.hand(tuple(right_pos), mf.yaw(mf.CAL_YAW_DEG), bool(how.get('rgrip')),
                            a=bool(how.get('a')), b=bool(how.get('b')))
            rx, ry = how.get('right', (0.0, 0.0))
            right['axes'] = [0.0, 0.0, float(rx), float(ry)]
            left = None
            if not how.get('left_missing'):
                left = mf.hand(LEFT_POS, mf.yaw(0.0), bool(how.get('grip')),
                               a=bool(how.get('x')), b=bool(how.get('y')))
                lx, ly = how.get('left', (0.0, 0.0))
                left['axes'] = [0.0, 0.0, float(lx), float(ly)]
            lines.append(mf.line(i, right, mf.viewer(), seq=seq, session_id=session, left=left))
            i += 1
            seq += 1
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(lines) + '\n')
    return begins


def calibrate_and_arm():
    """Return the phases that calibrate (A) and arm (B), the controllers resting."""
    return [phase(0.3), phase(0.2, a=True), phase(0.3), phase(0.2, b=True), phase(1.0)]


def play(stack, name, phases, wait=True):
    """Write and play a recording; return (start, phase begins, replay)."""
    path = str(stack['logs'] / f'{name}.jsonl')
    begins = write_recording(path, phases)
    start = time.monotonic()
    stack['start'] = start
    process = replay_file(stack['env'], path, str(stack['logs'] / f'{name}.log'),
                          start_delay=0.0)
    if wait:
        assert process.wait(sum(p[0] for p in phases) + 20.0) == 0, process.output()
    return start, begins, process


class BaseWatcher:
    """The base's topics as the probe receives them, each with its monotonic arrival."""

    def __init__(self, probe):
        self.probe = probe
        self.lock = threading.Lock()
        self.states, self.cmds, self.stamped, self.odom, self.gt = [], [], [], [], []
        self.haptics, self.panel = [], []
        node = probe.node
        node.create_subscription(BaseState, '/teleop/base/state', self._add('states'), QOS)
        node.create_subscription(Twist, '/cmd_vel', self._add('cmds'), QOS)
        node.create_subscription(TwistStamped, '/teleop/base/cmd_vel', self._add('stamped'), QOS)
        node.create_subscription(Odometry, '/odom', self._add('odom'), 50)
        node.create_subscription(Odometry, '/ground_truth/odom', self._add('gt'), 50)
        node.create_subscription(Haptic, '/teleop/haptic', self._add('haptics'), QOS)
        node.create_subscription(OperatorStatus, '/teleop/operator/status', self._add('panel'),
                                 PANEL_QOS)

    def _add(self, name):
        def callback(msg):
            with self.lock:
                getattr(self, name).append((time.monotonic(), msg))
        return callback

    def since(self, name, t):
        with self.lock:
            return [(w, m) for w, m in getattr(self, name) if w >= t]

    def last(self):
        with self.lock:
            return self.states[-1][1] if self.states else None

    def wait(self, t, condition, timeout, what):
        """Wait for a BaseState since ``t`` for which ``condition`` holds; return it."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for w, m in self.since('states', t):
                if condition(m):
                    return w, m
            time.sleep(0.05)
        seen = sorted({(STATE.get(m.state), REASON.get(m.reason))
                       for _, m in self.since('states', t)})
        raise AssertionError(f'timed out after {timeout} s waiting for {what}; seen {seen}')

    def commands(self, t):
        """Return ``(arrival, vx, vy, wz)`` of every /cmd_vel since ``t``."""
        return [(w, m.linear.x, m.linear.y, m.angular.z) for w, m in self.since('cmds', t)]

    def speed(self):
        """Return the newest odometry: linear speed in m/s, turning rate in rad/s."""
        with self.lock:
            if not self.odom:
                return None
            t = self.odom[-1][1].twist.twist
        return math.hypot(t.linear.x, t.linear.y), abs(t.angular.z)

    def rest_at(self, t, timeout=3.0):
        """Wait until odometry reads zero twist; return when (monotonic)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for w, m in self.since('odom', t):
                tw = m.twist.twist
                if tw.linear.x == 0.0 and tw.linear.y == 0.0 and tw.angular.z == 0.0:
                    return w
            time.sleep(0.05)
        raise AssertionError(f'the base did not come to rest within {timeout} s')

    def felt(self, t, source='base'):
        return [m.event for _, m in self.since('haptics', t) if m.source == source]

    def hands(self, t, source='base'):
        return {(m.event, m.hand) for _, m in self.since('haptics', t) if m.source == source}

    def shown(self, t):
        return {m.base_text for _, m in self.since('panel', t) if m.base_valid}


def after_stop(cmds):
    """Return the commands from the first zero after a nonzero one on (the stop and after)."""
    moving = False
    for k, (_, vx, vy, wz) in enumerate(cmds):
        if vx or vy or wz:
            moving = True
        elif moving:
            return cmds[k:]
    return []


def describe(state):
    return f'{STATE.get(state.state)} {REASON.get(state.reason)}: {state.reason_text}'


__all__ = ['phase', 'write_recording', 'calibrate_and_arm', 'play', 'BaseWatcher',
           'after_stop', 'describe', 'B', 'REASON', 'STATE', 'os']
