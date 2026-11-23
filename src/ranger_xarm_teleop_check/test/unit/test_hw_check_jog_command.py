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
"""The rule on the teleop node's own joint command (G-18 judged where it is enforced)."""

import numpy as np
import pytest
from ranger_xarm_teleop_check import rules, stats
from ranger_xarm_teleop_check.rules import FAIL, NOT_JUDGED, Options, PASS
from ranger_xarm_teleop_servo import joint_accel as ja

LIMIT = 2.0
PERIOD = rules.COMMAND_PERIOD_S


def options(limit=LIMIT, given=True):
    return Options(limits_given=given, joint_acceleration_limit=limit, servo_latency_s=0.0)


def message(velocities):
    return (stats.ARM_JOINTS, tuple(float(v) for v in velocities))


def limited_stroke(limit=LIMIT, cycles=60):
    """Return the node's own limiter and ramp: toward a desired command, then a release."""
    desired = np.array([0.4, -0.3, 0.2, 0.0, 0.1, -0.05])
    sent, out = np.zeros(6), [np.zeros(6)]
    for _ in range(cycles):
        sent = ja.limit_step(sent, desired, limit, PERIOD).velocity
        out.append(sent)
    while np.any(sent):
        sent = ja.ramp_step(sent, limit, PERIOD).velocity
        out.append(sent)
    return out


def judged(commands, announced=0, opts=None):
    data = {'jog': [message(v) for v in commands], 'rosout': {'immediate_stops': announced}}
    opts = opts or options()
    jm = rules.jog_measures(data, rules._command_limit(opts)[0])
    return jm, rules.rule_jog_acceleration(jm, opts)


def test_the_node_limiter_and_ramp_pass_at_the_limit():
    jm, r = judged(limited_stroke())
    assert r.verdict == PASS, r.line()
    assert r.rule == 'commanded joint acceleration (teleop joint command)'
    assert jm['peak'] == pytest.approx(LIMIT)
    assert jm['above'] == 0 and jm['above_to_zero'] == 0


def test_a_step_above_the_limit_fails():
    commands = limited_stroke()
    commands.insert(5, commands[4] + np.array([LIMIT * PERIOD * 1.01, 0, 0, 0, 0, 0]))
    jm, r = judged(commands)
    assert r.verdict == FAIL and jm['above'] >= 1
    assert 'faster than the limit it enforces' in r.advice


def test_an_announced_immediate_stop_passes_and_an_unannounced_one_fails():
    commands = limited_stroke(cycles=60)[:40] + [np.zeros(6)]       # one zero at once
    jm, r = judged(commands, announced=1)
    assert jm['above_to_zero'] == 1 and r.verdict == PASS, r.line()
    _, r = judged(commands, announced=0)
    assert r.verdict == FAIL and 'more often than the node announced' in r.advice


def test_not_judged_without_the_limit_or_without_motion():
    _, r = judged(limited_stroke(), opts=options(given=False))
    assert r.verdict == NOT_JUDGED and '--limits' in r.advice
    _, r = judged([np.zeros(6)] * 20)
    assert r.verdict == NOT_JUDGED and 'engage the clutch' in r.advice


def test_messages_without_the_six_arm_joints_are_left_out_and_said():
    data = {'jog': [message(v) for v in limited_stroke()] + [(('other',), (5.0,))],
            'rosout': {'immediate_stops': 0}}
    jm = rules.jog_measures(data, LIMIT)
    r = rules.rule_jog_acceleration(jm, options())
    assert r.verdict == PASS and '1 messages without all six arm joints' in r.measured


def test_the_teleop_stage_judges_it_first():
    from test_hw_check_rules import _teleop_data
    _, results = rules.evaluate('teleop', _teleop_data(), options())
    names = [x.rule for x in results]
    assert names.index(rules.JOG_RULE) < names.index(
        'commanded joint acceleration (MoveIt Servo output)')


def test_the_log_line_counted_is_the_one_the_node_writes():
    import os
    source = os.path.join(os.path.dirname(ja.__file__), 'teleop_node.py')
    with open(source, encoding='utf-8') as handle:
        assert "f'{ja.IMMEDIATE_STOP_LOG}: {immediate}'" in handle.read()


def test_the_step_period_is_the_node_command_period():
    from ranger_xarm_teleop_servo import params
    assert PERIOD == pytest.approx(1.0 / params.TARGET_RATE_HZ, abs=1e-12)
