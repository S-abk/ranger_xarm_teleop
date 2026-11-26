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
"""The wrist oscillation rule as amended: reversals counted past a +-0.1 rad/s dead band."""

import json
import math
import os

from ranger_xarm_teleop_check import rules

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def _data(name):
    """``data['arm']`` rows (arrival, stamp, time, positions, velocities, effort) of a fixture."""
    with open(os.path.join(FIXTURES, name), encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle][1:]
    return {'arm': [(1000.0 + r['t'], 1000.0 + r['t'], 1000.0 + r['t'], tuple(r['p']),
                     tuple(r['v']), False) for r in rows]}


def _wobble(speed, hz=5.0, seconds=3.0, rate=150.0):
    """Joint 4 swinging at ``hz`` with a peak speed ``speed`` (rad/s); the rest still."""
    arm = []
    for i in range(int(seconds * rate)):
        t = 1000.0 + i / rate
        v = speed * math.sin(2.0 * math.pi * hz * t)
        p = -speed / (2.0 * math.pi * hz) * math.cos(2.0 * math.pi * hz * t)
        arm.append((t, t, t, (0.0, 0.0, 0.0, p, 0.0, 0.0), (0.0, 0.0, 0.0, v, 0.0, 0.0), False))
    return {'arm': arm}


def test_today_s_unfiltered_wrist_at_hardware_limits_still_fails():
    # Gazebo, a recorded hand replay at hardware limits, the gate without the feedforward
    # rotation low-pass: its busiest second.
    mm = rules.motion_measures(_data('wrist_unfiltered_hw.jsonl'))
    result = rules.rule_oscillation(mm)
    assert result.verdict == rules.FAIL, result.measured
    assert mm['wrist_sign_changes_per_s']['xarm_joint4'] == 5
    assert mm['wrist_sign_changes_unbanded']['xarm_joint4'] == 12


def test_the_filtered_wrist_passes_and_the_unbanded_count_is_reported_beside_it():
    mm = rules.motion_measures(_data('wrist_filtered_hw.jsonl'))
    assert rules.rule_oscillation(mm).verdict == rules.PASS
    info = rules.info_wrist_reversals_unbanded(mm)
    assert info.verdict == rules.INFO
    assert info.measured == 'xarm_joint4: 8/s, xarm_joint6: 4/s'
    assert 'dead band' in info.threshold


def test_tremor_inside_the_band_is_not_counted_and_a_swing_through_it_is():
    small = rules.motion_measures(_wobble(0.08))
    assert small['wrist_sign_changes_per_s']['xarm_joint4'] == 0
    assert small['wrist_sign_changes_unbanded']['xarm_joint4'] >= 9
    assert rules.rule_oscillation(small).verdict == rules.PASS
    large = rules.motion_measures(_wobble(0.3))
    assert large['wrist_sign_changes_per_s']['xarm_joint4'] >= 9
    result = rules.rule_oscillation(large)
    assert result.verdict == rules.FAIL
    assert 'from above +0.1 rad/s to below -0.1 rad/s or the reverse' in result.threshold
