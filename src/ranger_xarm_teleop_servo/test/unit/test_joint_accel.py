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
"""The joint acceleration limit of the teleop command (G-18, INV-1, INV-3, INV-4)."""

import math

import numpy as np
import pytest
from ranger_xarm_teleop_servo import joint_accel as ja

T = 0.02
A = 10.0


def _random_commands(rng, count, scale=2.0):
    return [rng.uniform(-scale, scale, 6) for _ in range(count)]


def test_no_joint_ever_changes_by_more_than_the_limit_in_one_cycle():
    rng = np.random.default_rng(3)
    sent = np.zeros(6)
    for desired in _random_commands(rng, 2000):
        step = ja.limit_step(sent, desired, A, T)
        assert np.max(np.abs(step.velocity - sent)) <= A * T * (1 + 1e-9)
        sent = step.velocity


def test_the_limited_change_is_parallel_to_the_requested_one():
    """INV-4: all joints scaled together, never one clipped on its own."""
    rng = np.random.default_rng(4)
    for _ in range(500):
        previous, desired = rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)
        step = ja.limit_step(previous, desired, A, T)
        change, asked = step.velocity - previous, desired - previous
        assert np.allclose(change, asked * step.scale, atol=1e-12)
        assert 0.0 < step.scale <= 1.0
        assert step.limited == (step.scale < 1.0)


def test_a_request_that_keeps_its_direction_keeps_the_commands_direction():
    """Speeding up or slowing down along one direction: the command stays on that direction."""
    direction = np.array([0.3, -0.8, 0.5, 0.0, 0.1, -0.2])
    sent = direction * 0.1
    for target in (1.5, 0.05, 1.0, 0.0):
        for _ in range(100):
            sent = ja.limit_step(sent, direction * target, A, T).velocity
            if np.linalg.norm(sent) > 1e-12:
                assert np.allclose(sent / np.linalg.norm(sent),
                                   direction / np.linalg.norm(direction), atol=1e-12)


def test_a_small_request_passes_unchanged():
    previous = np.array([0.1, 0.2, -0.1, 0.0, 0.05, 0.0])
    desired = previous + 0.5 * A * T
    step = ja.limit_step(previous, desired, A, T)
    assert not step.limited and step.scale == 1.0
    assert np.array_equal(step.velocity, desired)


def test_the_limiter_keeps_nothing_it_did_not_send():
    """INV-3: once the request is zero, the command goes to zero at the limit and stays."""
    sent = np.zeros(6)
    for _ in range(3):                        # a large request, partly delivered
        sent = ja.limit_step(sent, np.full(6, 1.5), A, T).velocity
    history = []
    for _ in range(50):                       # the hand has stopped: the request is zero
        sent = ja.limit_step(sent, np.zeros(6), A, T).velocity
        history.append(np.max(np.abs(sent)))
    assert history[-1] == 0.0
    assert all(b <= a for a, b in zip(history, history[1:]))   # never speeds up to catch up


def test_the_ramp_reaches_zero_in_the_stop_time_keeping_direction():
    start = np.array([1.9, -0.95, 0.4, 0.0, -1.2, 0.3])
    sent, cycles = start, 0
    while np.any(sent != 0.0):
        step = ja.ramp_step(sent, A, T)
        assert np.max(np.abs(step.velocity - sent)) <= A * T * (1 + 1e-9)
        if np.any(step.velocity != 0.0):
            assert np.allclose(step.velocity / np.linalg.norm(step.velocity),
                               start / np.linalg.norm(start), atol=1e-12)
        sent, cycles = step.velocity, cycles + 1
        assert cycles < 1000
    assert cycles == math.ceil(1.9 / (A * T) - 1e-9)
    assert cycles * T <= ja.stop_time(start, A) + T


def test_the_ramp_from_rest_is_zero_and_not_limited():
    step = ja.ramp_step(np.zeros(6), A, T)
    assert not step.limited and np.array_equal(step.velocity, np.zeros(6))


@pytest.mark.parametrize('cap, a', [(1.9, 10.0), (0.4, 2.0), (0.2, 1.0)])
def test_a_stop_within_the_stop_time_is_accepted(cap, a):
    assert ja.stop_time_refusal(a, cap) == ''


@pytest.mark.parametrize('cap, a', [(1.9, 9.0), (0.5, 2.0), (3.14, 2.0)])
def test_a_longer_stop_is_refused_with_one_line_naming_the_value(cap, a):
    line = ja.stop_time_refusal(a, cap)
    assert line and '\n' not in line
    assert ja.ACCELERATION_NAME in line and ja.SPEED_CAP_NAME in line
    assert f'at least {cap / 0.2:.4g}' in line


def test_the_stop_time_is_fixed_in_code():
    assert ja.STOP_TIME_LIMIT_S == 0.2


@pytest.mark.parametrize('value', [0.0, -1.0, math.inf, math.nan, True, 'x', None])
def test_values_must_be_finite_positive_numbers(value):
    with pytest.raises(ValueError):
        ja.check_values(value, 1.0)
    with pytest.raises(ValueError):
        ja.check_values(10.0, value)


def test_stop_travel_bound():
    assert ja.stop_travel_bound(0.25, 1.9, 10.0) == pytest.approx(0.02375)


@pytest.mark.parametrize('window, period, cycles', [(0.06, 0.02, 4), (0.05, 0.02, 4),
                                                    (0.04, 0.02, 3), (0.1, 0.02, 6)])
def test_the_flush_is_servos_window_in_cycles_plus_one(window, period, cycles):
    assert ja.flush_cycles(window, period) == cycles


@pytest.mark.parametrize('window, period', [(0.0, 0.02), (0.06, 0.0), (-0.06, 0.02)])
def test_the_flush_needs_a_window_and_a_period(window, period):
    with pytest.raises(ValueError):
        ja.flush_cycles(window, period)


def test_the_node_flushes_servos_window_from_servo_yaml_plus_one():
    """G-18: the node's flush is derived from servo.yaml's window and period, plus one."""
    import ast
    import os
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, '..', '..')
    yaml_text = open(os.path.join(src, '..', 'ranger_xarm_teleop_bringup', 'config',
                                  'servo.yaml')).read()
    latency = float(re.search(r'^max_expected_latency:\s*([0-9.]+)', yaml_text, re.M).group(1))
    period = float(re.search(r'^publish_period:\s*([0-9.]+)', yaml_text, re.M).group(1))
    tree = ast.parse(open(os.path.join(src, 'ranger_xarm_teleop_servo', 'teleop_node.py')).read())
    assigned = {node.targets[0].id: node.value for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert assigned['SERVO_WINDOW_S'].value == latency
    assert assigned['SERVO_PERIOD_S'].value == period
    # Derived, not a bare number: ja.flush_cycles(SERVO_WINDOW_S, SERVO_PERIOD_S).
    flush = assigned['FLUSH_CYCLES']
    assert isinstance(flush, ast.Call) and ast.unparse(flush) == \
        'ja.flush_cycles(SERVO_WINDOW_S, SERVO_PERIOD_S)'
    assert ja.flush_cycles(latency, period) == 4
