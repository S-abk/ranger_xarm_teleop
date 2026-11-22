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
"""Measures: joint-state filtering, periods, stale repeats, acceleration, oscillation, lead."""

import math

import numpy
from ranger_xarm_teleop_check import stats

ARM = list(stats.ARM_JOINTS)


def test_arm_message_classified_in_joint_order():
    names = ['xarm_joint2', 'xarm_joint1'] + ARM[2:]
    pos = [2.0, 1.0, 3.0, 4.0, 5.0, 6.0]
    kind, p, v, e = stats.classify_joint_state(names, pos)
    assert kind == 'arm' and p == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert v is None and e is False


def test_gripper_only_message_is_not_arm():
    assert stats.classify_joint_state(['xarm_drive_joint'], [0.1])[0] == 'gripper'
    assert stats.classify_joint_state(ARM[:5], [0.0] * 5)[0] == 'other'
    assert stats.classify_joint_state(['other'], [0.0])[0] == 'other'


def test_arm_plus_gripper_message_is_arm():
    names = ARM + ['xarm_drive_joint']
    kind, p, _, _ = stats.classify_joint_state(names, [1, 2, 3, 4, 5, 6, 9])
    assert kind == 'arm' and len(p) == 6


def test_velocity_and_effort_need_finite_entries():
    kind, _, v, e = stats.classify_joint_state(ARM, [0.0] * 6, [0.1] * 6, [1.0] * 6)
    assert v == (0.1,) * 6 and e is True
    _, _, v, e = stats.classify_joint_state(ARM, [0.0] * 6, [math.nan] + [0.1] * 5,
                                            [math.inf] + [1.0] * 5)
    assert v is None and e is False
    _, _, v, e = stats.classify_joint_state(ARM, [0.0] * 6, [0.1] * 5, [1.0] * 5)
    assert v is None and e is False


def test_period_distribution_steady_with_a_gap():
    times = [i / 150 for i in range(300)]
    d = stats.period_distribution(times, 1 / 150)
    assert abs(d['rate_hz'] - 150) < 1e-6 and d['over_2x'] == 0
    assert abs(d['p50'] - 1 / 150) < 1e-9
    times = [i / 150 for i in range(100)] + [0.9 + i / 150 for i in range(100)]
    d = stats.period_distribution(times, 1 / 150)
    assert d['over_2x'] == 1 and abs(d['largest_gap'] - (0.9 - 99 / 150)) < 1e-9
    assert d['max'] == d['largest_gap'] and d['rate_hz'] < 150


def test_period_distribution_too_short():
    d = stats.period_distribution([1.0], 0.01)
    assert d['count'] == 1 and d['rate_hz'] is None and d['largest_gap'] is None
    assert stats.period_distribution([], None)['count'] == 0


def test_percentile_and_median():
    assert stats.percentile([], 50) is None
    assert stats.median([1, 2, 3]) == 2.0
    assert abs(stats.stamp_offset([10.0, 11.0], [9.99, 10.99]) - 0.01) < 1e-9


def _ramp(n=300, speed=0.5, hold=1, rate=150.0):
    """Joint positions at ``rate``; values refresh only every ``hold`` messages."""
    t = [i / rate for i in range(n)]
    pos = []
    for i in range(n):
        k = (i // hold) * hold
        pos.append(tuple(speed * k / rate * (j + 1) for j in range(6)))
    return t, pos


def test_stale_fresh_values_pass():
    t, pos = _ramp(hold=1)
    s = stats.stale_analysis(t, pos)
    assert s['judged'] and s['stale_count'] == 0 and s['longest_stale_messages'] == 0
    assert s['value_change_fraction'] > 0.99


def test_stale_every_30th_value_is_stale():
    t, pos = _ramp(hold=30)
    s = stats.stale_analysis(t, pos)
    assert s['judged'] and s['longest_stale_messages'] == 29
    assert s['value_change_fraction'] < 0.1
    assert s['longest_stale_seconds'] > 0.19


def test_stale_motionless_is_not_judged():
    t = [i / 150 for i in range(300)]
    s = stats.stale_analysis(t, [(0.0,) * 6] * 300)
    assert not s['judged'] and s['stale_count'] == 0


def test_stale_short_motion_is_not_judged():
    t, pos = _ramp(n=60, hold=1)          # 0.4 s: less than the 0.5 s span
    assert not stats.stale_analysis(t, pos)['judged']
    assert not stats.stale_analysis([0.0], [(0.0,) * 6])['judged']


def test_stale_motion_then_rest_counts_only_the_motion():
    t, pos = _ramp(n=300, hold=1)
    pos += [pos[-1]] * 200               # then still: identical repeats, not moving
    t += [300 / 150 + i / 150 for i in range(200)]
    s = stats.stale_analysis(t, pos)
    assert s['judged'] and s['stale_count'] == 0


def test_velocity_from_positions_and_reported():
    t, pos = _ramp(n=50, speed=1.0)
    v, source = stats.joint_velocities(t, pos, [None] * 50)
    assert source == 'position' and abs(v[10, 0] - 1.0) < 1e-6 and math.isnan(v[0, 0])
    v2, source = stats.joint_velocities(t, pos, [(2.0,) * 6] * 50)
    assert source == 'reported' and v2[3, 3] == 2.0


def test_acceleration_of_a_constant_acceleration():
    rate = 150.0
    t = [i / rate for i in range(200)]
    acc = 0.8
    pos = [tuple(0.5 * acc * (x ** 2) * (j + 1) for j in range(6)) for x in t]
    v, _ = stats.joint_velocities(t, pos, None)
    peaks = stats.peak_acceleration(t, v)
    for j in range(6):
        assert abs(peaks[j] - acc * (j + 1)) < 1e-3 * (j + 1)
    assert stats.peak_acceleration([0, 1], numpy.zeros((2, 6))) == [None] * 6


def test_oscillation_counts_in_one_second():
    t = [i / 150 for i in range(300)]
    quiet = [0.1] * 300
    assert stats.sign_changes_per_window(t, quiet) == 0
    # a 4 Hz sine: 8 sign changes per second
    fast = [0.5 * math.sin(2 * math.pi * 4 * x + 0.1) for x in t]
    assert stats.sign_changes_per_window(t, fast) >= 7
    slow = [0.5 * math.sin(2 * math.pi * 1 * x + 0.1) for x in t]
    assert stats.sign_changes_per_window(t, slow) <= 3
    tiny = [0.01 * math.sin(2 * math.pi * 4 * x + 0.1) for x in t]
    assert stats.sign_changes_per_window(t, tiny) == 0


def test_nearest_match_and_lead():
    idx = stats.nearest_match([1.0, 2.0, 3.0], [1.001, 2.5], 0.01)
    assert idx == [0, None, None]
    st = [1.0, 1.02, 1.04]
    meas = [(0.0, 0.0, 0.0)] * 3
    tt = [1.0, 1.02, 1.04]
    tgt = [(0.03, 0.04, 0.0), (0.0, 0.0, 0.1), (0.0, 0.0, 0.0)]
    lead = stats.lead_stats(st, meas, tt, tgt)
    assert lead['count'] == 3 and abs(lead['max'] - 0.1) < 1e-12
    assert abs(lead['p50'] - 0.05) < 1e-12
    assert stats.lead_stats(st, meas, [], [])['count'] == 0


def test_lag_estimate_finds_the_delay():
    rate, delay = 50.0, 0.2
    t = [i / rate for i in range(400)]

    def signal(x):
        return (math.sin(x * 1.3), math.cos(x * 0.9), math.sin(x * 2.1 + 1))

    target = [signal(x) for x in t]
    measured = [signal(x - delay) for x in t]
    lag = stats.lag_estimate(t, measured, t, target)
    assert abs(lag - delay) < 0.03
    assert stats.lag_estimate(t[:5], measured[:5], t[:5], target[:5]) is None
