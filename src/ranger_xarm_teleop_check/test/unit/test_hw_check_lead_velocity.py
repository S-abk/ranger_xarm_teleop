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
"""The lead rule against the G-14 clamp, and velocities from a backend that reports zero."""

from ranger_xarm_teleop_check import rules, stats
from ranger_xarm_teleop_check.rules import FAIL, INFO, Options, PASS

OPTS = Options(limits_given=True, max_lead_translation_m=0.05)


def lead(values):
    return {'lead': {'count': len(values), 'p50': sorted(values)[len(values) // 2],
                     'p99': max(values), 'max': max(values), 'values': list(values)},
            'lag_s': 0.08}


def test_a_lead_held_at_the_limit_by_the_clamp_passes_and_is_reported():
    out = {r.rule: r for r in rules.rules_lead(lead([0.01] * 50 + [0.05] * 50), OPTS)}
    assert out['target lead (largest, translation)'].verdict == PASS
    at_limit = out['target lead at the limit']
    assert at_limit.verdict == INFO and at_limit.measured.startswith('50 %')


def test_a_lead_beyond_the_limit_fails():
    out = {r.rule: r for r in rules.rules_lead(lead([0.01] * 99 + [0.06]), OPTS)}
    result = out['target lead (largest, translation)']
    assert result.verdict == FAIL and 'did not hold' in result.line()


def test_zero_reported_velocities_while_moving_are_not_used():
    t = [i * 0.01 for i in range(20)]
    pos = [[0.1 * x] * 6 for x in t]
    vel, source = stats.joint_velocities(t, pos, [[0.0] * 6] * 20)
    assert source == 'position'
    assert abs(vel[10][0] - 0.1) < 1e-9


def test_reported_velocities_are_used_when_they_report():
    t = [i * 0.01 for i in range(20)]
    pos = [[0.1 * x] * 6 for x in t]
    vel, source = stats.joint_velocities(t, pos, [[0.1] * 6] * 20)
    assert source == 'reported'


def test_zero_reported_velocities_at_rest_are_kept():
    t = [i * 0.01 for i in range(20)]
    vel, source = stats.joint_velocities(t, [[0.2] * 6] * 20, [[0.0] * 6] * 20)
    assert source == 'reported'
