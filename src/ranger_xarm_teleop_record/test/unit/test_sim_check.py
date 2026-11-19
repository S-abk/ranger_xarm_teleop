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
"""The simulator machine check: joint-state arrival gaps against the gate's limit."""

import pytest
from ranger_xarm_teleop_record.sim_check import assess, parameter_from, real_time_factor


def steady(rate_hz, seconds, t0=0.0):
    n = int(rate_hz * seconds)
    return [t0 + i / rate_hz for i in range(n + 1)]


def clock(factor, seconds, t0=0.0, ros0=10.0, step=0.004):
    return [(ros0 + factor * step * i, t0 + step * i) for i in range(int(seconds / step) + 1)]


def test_a_fast_steady_simulator_passes():
    r = assess(steady(150.0, 30.0), clock(0.9, 30.0), 0.1, 30.0)
    assert r['ok'] and r['warnings'] == []
    assert r['gaps_over_limit'] == 0
    assert r['gap_ms']['p99'] == pytest.approx(6.7, abs=0.1)
    assert r['real_time_factor'] == pytest.approx(0.9, abs=1e-3)
    assert r['joint_state_rate_hz'] == pytest.approx(150.0, rel=1e-3)


def test_bursts_with_gaps_over_the_limit_fail_and_say_why():
    # Bursts as measured on the integrated GPU: 5 states within 30 ms, then 110 ms of nothing.
    arrivals = []
    t = 0.0
    while t < 30.0:
        arrivals += [t + 0.007 * k for k in range(5)]
        t += 0.14
    r = assess(arrivals, clock(0.36, 30.0), 0.1, 30.0)
    assert not r['ok']
    assert r['gaps_over_limit'] > 100
    text = ' '.join(r['warnings'])
    assert 'longer than tool_pose_timeout_s (100 ms)' in text
    assert 'runs at 0.36 of real time' in text
    assert 'discrete' in text                       # says what to do


def test_little_margin_warns_before_any_gap_reaches_the_limit():
    # Every gap 60 ms: none over 100 ms, but p99 above half of it.
    r = assess(steady(1 / 0.06, 30.0), clock(1.0, 30.0), 0.1, 30.0)
    assert r['gaps_over_limit'] == 0 and not r['ok']
    assert any('little margin' in w for w in r['warnings'])


def test_the_limit_comes_from_the_caller():
    arrivals = steady(1 / 0.06, 30.0)
    assert assess(arrivals, [], 0.2, 30.0)['ok']    # 60 ms gaps against a 200 ms limit
    assert not assess(arrivals, [], 0.1, 30.0)['ok']


def test_no_joint_states_is_a_failure_and_no_clock_is_only_noted():
    r = assess([], [], 0.1, 30.0)
    assert not r['ok'] and 'is the simulator running' in r['warnings'][0]
    hw = assess(steady(150.0, 30.0), [], 0.1, 30.0)
    assert hw['ok'] and hw['real_time_factor'] is None and hw['simulated_clock'] is False


def test_real_time_factor_edges():
    assert real_time_factor([]) is None
    assert real_time_factor([(10.0, 0.0), (10.5, 0.5)]) is None          # under 1 s of wall time
    assert real_time_factor([(0.0, 0.0), (5.0, 5.0)]) is None            # zero stamp: no time yet
    assert real_time_factor([(20.0, 0.0), (10.0, 5.0)]) is None          # clock went back
    assert real_time_factor([(10.0, 0.0), (12.0, 5.0)]) == pytest.approx(0.4)


def test_the_limit_is_read_from_nested_parameter_files_and_defaults_otherwise():
    flat = {'/**': {'ros__parameters': {'tool_pose_timeout_s': 0.15}}}
    nested = {'robot': {'teleop_servo': {'ros__parameters': {'tool_pose_timeout_s': 0.2}}}}
    unset = {'/**': {'ros__parameters': {'other': 1}}}
    assert parameter_from(flat, 'tool_pose_timeout_s', 0.1) == pytest.approx(0.15)
    assert parameter_from(nested, 'tool_pose_timeout_s', 0.1) == pytest.approx(0.2)
    for data in (unset, None, [], 'text', {'/**': None}, {'/**': {'ros__parameters': []}}):
        assert parameter_from(data, 'tool_pose_timeout_s', 0.1) == pytest.approx(0.1)
