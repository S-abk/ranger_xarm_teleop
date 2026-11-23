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
The joint acceleration limit through the whole driving path (G-18), over the solver fixtures.

The path is the gate, the solver, the limiter and the delayed model arm (150 ms, Servo's
state as reference).
Three configurations: the simulation values (joint speed cap 1.9 rad/s, 10 rad/s^2, tool caps
0.25 m/s and 1 rad/s), and a hardware-like pair (0.4 rad/s, 2 rad/s^2: a 0.2 s stop) at the
simulation tool caps and at a fifth of them. These are test values only.
"""

from arm_urdf import arm_urdf, READY
from helpers import fixture, params
import numpy as np
import pytest
from ranger_xarm_teleop_servo import joint_accel as ja
from ranger_xarm_teleop_servo import kinematics as kin
from ranger_xarm_teleop_servo import solver as sv
from ranger_xarm_teleop_servo.rotations import pose_from_matrix
from ranger_xarm_teleop_servo.teleop_model import run

T = 0.02
CASES = {
    'simulation': {'cap': 1.9, 'a': 10.0, 'linear': 0.25, 'angular': 1.0},
    'hardware-like': {'cap': 0.4, 'a': 2.0, 'linear': 0.25, 'angular': 1.0},
    'hardware-like, fifth caps': {'cap': 0.4, 'a': 2.0, 'linear': 0.05, 'angular': 0.2},
}
JOINT_LIMIT_START = [0.0, -0.6, -0.7, 0.0, -1.2, 0.0]
NAMES = ('clutch_line', 'clutch_fast', 'clutch_step', 'clutch_release_moving',
         'clutch_joint_limit', 'clutch_reach', 'clutch_circle', 'clutch_scale_switch',
         'clutch_shoulder', 'clutch_rotation')
STARTS = {'clutch_joint_limit': JOINT_LIMIT_START,
          'clutch_shoulder': [0.0, -0.5, -1.1, 0.0, 1.6, 0.0]}
MARGIN = 0.15               # the solver's joint margin; MoveIt Servo halts at 0.1


@pytest.fixture(scope='module')
def arm():
    return sv.make_arm(kin.chain_from_urdf(arm_urdf()),
                       velocity_limits={f'xarm_joint{i}': 2.0 for i in range(1, 7)})


def _run(arm, name, case, limited=True):
    c = CASES[case]
    solver = sv.SolverParams(linear_speed_cap=c['linear'], angular_speed_cap=c['angular'],
                             joint_speed_cap=c['cap'],
                             joint_acceleration_limit=c['a'] if limited else float('inf'))
    return run(fixture(name), params(linear_speed_cap_m_s=c['linear'],
                                     angular_speed_cap_rad_s=c['angular']),
               arm, solver, STARTS.get(name, READY), delay=0.15, stale_ticks=1, tail=1.0,
               acceleration=c['a'] if limited else None)


@pytest.fixture(scope='module')
def runs(arm):
    return {(name, case): _run(arm, name, case) for name in NAMES for case in CASES}


def _tips(arm, steps):
    return np.array([pose_from_matrix(sv.tool_pose(arm, s.commanded)).p for s in steps])


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('name', NAMES)
def test_no_joint_command_changes_faster_than_the_limit(runs, name, case):
    """INV-1 in the model: engaged, released, every bound."""
    steps = runs[(name, case)]
    sent = np.array([s.qdot for s in steps])
    change = np.max(np.abs(np.diff(np.vstack([np.zeros(6), sent]), axis=0)), axis=1)
    assert change.max() <= CASES[case]['a'] * T * (1 + 1e-9)
    assert np.max(np.abs(sent)) <= CASES[case]['cap'] + 1e-9


@pytest.mark.parametrize('case', CASES)
def test_a_joint_bound_is_still_kept_under_the_limit(arm, runs, case):
    """INV-5: the solver's joint margin holds with the limited command."""
    steps = runs[('clutch_joint_limit', case)]
    q = np.array([s.commanded for s in steps])
    room = np.minimum(q - arm.lower, arm.upper - q)
    assert room.min() >= MARGIN - 1e-6


@pytest.mark.parametrize('case', CASES)
def test_every_release_ramps_to_zero_within_the_stop_time(runs, case):
    """G-18: the ramp after a release ends within cap / a, at most 0.2 s."""
    c = CASES[case]
    for name in NAMES:
        steps = runs[(name, case)]
        for i in range(1, len(steps)):
            if steps[i - 1].output.target is not None and steps[i].output.target is None:
                j = i
                while j < len(steps) and np.any(steps[j].qdot != 0.0):
                    j += 1
                assert (j - i) * T <= ja.stop_time(steps[i - 1].qdot, c['a']) + T + 1e-9
                assert (j - i) * T <= ja.STOP_TIME_LIMIT_S + T + 1e-9, (name, j - i)


@pytest.mark.parametrize('case', CASES)
def test_release_travel_is_within_the_formula(arm, runs, case):
    """INV-2: extra tool travel after a release <= v q / (2 a), from the ramp's start."""
    c = CASES[case]
    seen = 0
    for name in NAMES:
        steps = runs[(name, case)]
        tips = _tips(arm, steps)
        for i in range(1, len(steps)):
            if steps[i - 1].output.target is not None and steps[i].output.target is None:
                q_speed = float(np.max(np.abs(steps[i - 1].qdot)))
                if q_speed < 1e-6:
                    continue
                v = float(np.linalg.norm(tips[i - 1] - tips[i - 2])) / T
                j = i
                while j < len(steps) and np.any(steps[j].qdot != 0.0):
                    j += 1
                travel = float(np.sum(np.linalg.norm(np.diff(tips[i - 1:j + 1], axis=0),
                                                     axis=1)))
                bound = ja.stop_travel_bound(v, q_speed, c['a'])
                # 10 % and 0.2 mm for the curved path and the discrete ramp.
                assert travel <= bound * 1.1 + 0.0002, (name, travel, bound)
                seen += 1
    assert seen >= 1


def test_the_limiter_adds_no_catch_up_after_the_hand_stops(arm):
    """
    INV-3 / G-18: no catch-up after the hand stops beyond the stop ramp's own bound.

    With the limit the arm travels no further after the hand stops than without it, plus the
    stop bound; the mapping is never moved.
    """
    frames = fixture('clutch_fast')
    moving = [f.arrival for a, f in zip(frames, frames[1:])
              if a.controller is not None and f.controller is not None
              and np.linalg.norm(np.subtract(f.controller.p, a.controller.p)) > 1e-6]
    stopped = moving[-1]
    for case in CASES:
        c = CASES[case]
        travel = {}
        for limited in (False, True):
            steps = _run(arm, 'clutch_fast', case, limited)
            tips = _tips(arm, steps)
            k = next(i for i, s in enumerate(steps) if s.t >= stopped)
            travel[limited] = float(np.sum(np.linalg.norm(np.diff(tips[k:], axis=0), axis=1)))
            speed = float(np.max(np.abs(steps[k].qdot)))
        bound = ja.stop_travel_bound(c['linear'], speed, c['a'])
        assert travel[True] <= travel[False] + bound + 0.001, (case, travel, bound)
