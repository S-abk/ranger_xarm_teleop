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
"""Shared helpers for the gate tests: parameters, frames and fixture paths."""

from dataclasses import replace
import math
import os

from ranger_xarm_teleop_servo.calibration import Calibration
from ranger_xarm_teleop_servo.frames import frames_of_recording, simulate
from ranger_xarm_teleop_servo.gate import Frame, GateState
from ranger_xarm_teleop_servo.params import build_params
from ranger_xarm_teleop_servo.rotations import Pose, q_between, v_norm, v_sub

PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIXTURES = os.path.join(PKG_DIR, 'test', 'fixtures')
BRIDGE_FIXTURES = os.path.join(os.path.dirname(PKG_DIR), 'ranger_xarm_teleop_bridge', 'test',
                               'fixtures')

# The simulation-only values (the gate document leaves the real ones PENDING).
VALUES = {
    'max_offset_translation_m': 0.30,
    'max_offset_rotation_deg': 60.0,
    'view_cone_half_angle_deg': 45.0,
    'step_translation_limit_m': 0.030,
    'step_rotation_limit_deg': 10.0,
    'linear_speed_cap_m_s': 0.25,
    'angular_speed_cap_rad_s': 1.0,
    'motion_enabled': True,
}

TOOL_START = Pose((0.30, 0.0, 0.40), (0.0, 0.0, 0.0, 1.0))
HEAD = Pose((0.0, 0.0, 1.6), (0.0, 0.0, 0.0, 1.0))       # looks along +x
HAND = Pose((0.5, -0.15, 1.35), (0.0, 0.0, 0.0, 1.0))    # 30 deg off the view axis
PERIOD = 0.02


def params(**overrides):
    """Return gate parameters: the simulation values with ``overrides``."""
    return build_params({**VALUES, **overrides})


def fixture(name):
    """Return the gate frames of a fixture of this package."""
    return frames_of_recording(os.path.join(FIXTURES, name + '.jsonl'))


def bridge_fixture(name):
    """Return the gate frames of a fixture of the bridge package."""
    return frames_of_recording(os.path.join(BRIDGE_FIXTURES, name + '.jsonl'))


def run(frames, gate_params=None, **kwargs):
    """Simulate ``frames``; return the list of steps."""
    steps, _ = simulate(frames, gate_params or params(), kwargs.pop('tool', TOOL_START),
                        **kwargs)
    return steps


def frame(k, controller=HAND, grip=True, rate=90.0, **fields):
    """Return frame number ``k`` of a clean stream at ``rate`` frames per second."""
    base = dict(arrival=k / rate, xr_time=100.0 + k / rate, session_id=7, seq=1000 + k,
                reset_count=0, controller=controller, controller_present=controller is not None,
                controller_emulated=False, head=HEAD, head_present=True, head_emulated=False,
                grip=grip, calibrate=False, arm=False)
    base.update(fields)
    return Frame(**base)


def ready_state(session_id=7, reset_count=0, heading=0.0):
    """Return a state that is armed and calibrated for the given frame."""
    return GateState(want_armed=True, calibration=Calibration(heading, session_id, reset_count))


def ready_for(frames, heading=0.0):
    """Return a state armed and calibrated for the first of ``frames``."""
    return ready_state(frames[0].session_id, frames[0].reset_count, heading)


def targets(steps):
    """Return ``(t, pose)`` for every step that published a target."""
    return [(s.t, s.output.target) for s in steps if s.output.target is not None]


def engaged_spans(steps):
    """Return ``(t_first, t_last)`` of each run of consecutive published targets."""
    spans, start, last = [], None, None
    for s in steps:
        if s.output.target is not None:
            start = s.t if start is None else start
            last = s.t
        elif start is not None:
            spans.append((start, last))
            start = None
    if start is not None:
        spans.append((start, last))
    return spans


def hold_at(steps, t):
    """Return the hold reason of the step at time ``t``."""
    return min(steps, key=lambda s: abs(s.t - t)).output.hold


def max_step(steps):
    """Return the largest translation and rotation between consecutive targets."""
    poses = [pose for _, pose in targets(steps)]
    moves = [v_norm(v_sub(b.p, a.p)) for a, b in zip(poses, poses[1:])]
    turns = [q_between(a.q, b.q) for a, b in zip(poses, poses[1:])]
    return max(moves, default=0.0), max(turns, default=0.0)


def same_pose(a, b, tol=1e-12):
    """Return True when two poses agree to floating-point noise."""
    return v_norm(v_sub(a.p, b.p)) <= tol and q_between(a.q, b.q) <= 1e-7


def moved(pose, dx=0.0, dy=0.0, dz=0.0):
    """Return ``pose`` translated."""
    return Pose((pose.p[0] + dx, pose.p[1] + dy, pose.p[2] + dz), pose.q)


def stream(n, motion, grip=lambda k: True, **fields):
    """Return ``n`` frames whose controller pose is ``motion(k)``."""
    return [frame(k, controller=motion(k), grip=grip(k), **fields) for k in range(n)]


def degrees(angle):
    """Return ``angle`` in degrees."""
    return math.degrees(angle)


__all__ = ['replace']
