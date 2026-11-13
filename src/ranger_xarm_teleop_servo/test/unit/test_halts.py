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
"""INV-4: every MoveIt Servo halt of a recorded session, explained from the rows alone."""

from arm_urdf import arm_urdf, READY
import numpy as np
import pytest
from ranger_xarm_teleop_servo import halts
from ranger_xarm_teleop_servo import kinematics as kin

CHAIN = kin.chain_from_urdf(arm_urdf())
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def tool_of(q):
    tip, _, _ = kin.forward(CHAIN, q)
    return [float(v) for v in tip[:3, 3]]


def wrist_session(start=0.0, lower=17.0, stop=30.0, description=True):
    """
    Joint 5 runs from 1.3 rad toward 0 while the clutch is held, at 150 Hz.

    Servo's status follows the condition number as MoveIt Servo would report it; at the
    halt the gate holds with SERVO_HALTED. The target leads the tool by 30 mm in +y.
    """
    rows = [{'t': start - 1.0, 'k': 'description', 'urdf': arm_urdf()}] if description else []
    halted = False
    q5 = 1.3
    for i in range(600):
        t = start + i / 150.0
        if not halted:
            q5 = max(0.05, 1.3 - 0.6 * (t - start))
        q = READY[:4] + [q5, 0.0]
        measure = kin.condition_number(CHAIN, q)
        rows.append({'t': t, 'k': 'joints', 'q': q, 'v': [0, 0, 0, 0, -0.6, 0]})
        if i % 3:
            continue
        code = 2 if measure >= stop else (1 if measure > lower else 0)
        tool = tool_of(q)
        rows.append({'t': t + 0.001, 'k': 'servo', 'code': code,
                     'msg': halts.SERVO_STATUS[code]})
        if not halted:
            rows.append({'t': t + 0.002, 'k': 'target', 'q': IDENTITY,
                         'p': [tool[0], tool[1] + 0.03, tool[2]]})
        rows.append({'t': t + 0.003, 'k': 'state', 'engaged': not halted,
                     'hold': 19 if halted else 0, 'servo': code, 'armed': True,
                     'tool': [tool, IDENTITY]})
        halted = halted or code == 2
    return rows


def test_one_wrist_halt_is_found_and_explained():
    result = halts.analyze(wrist_session())
    assert len(result['halts']) == 1
    halt = result['halts'][0]
    assert halt['servo_code'] == 2 and halt['servo_status'] == 'HALT_FOR_SINGULARITY'
    assert halt['hold_reason'] == 'SERVO_HALTED'
    assert halt['servo_message'] == 'HALT_FOR_SINGULARITY'
    assert halt['servo_codes_before'] == [0, 1]
    assert halt['joints'][:4] == READY[:4]
    assert 30.0 <= halt['condition_number'] < 31.5
    assert halt['resembles'] == 'wrist' and halt['wrist_deg'] < 7.0
    assert halt['nearest_limit']['joint'] == 'xarm_joint3'
    assert halt['nearest_limit']['margin_rad'] == pytest.approx(0.89198)


def test_trend_over_the_preceding_second_rises():
    trend = halts.analyze(wrist_session())['halts'][0]['trend']
    assert trend['at_minus_1.0s'] < trend['at_minus_0.5s'] < trend['at_minus_0.2s'] < 30.0
    assert trend['at_minus_1.0s'] == pytest.approx(
        kin.condition_number(CHAIN, READY[:4] + [0.1 + 0.6, 0.0]), rel=0.05)
    assert trend['max_in_window'] >= 30.0 and trend['rate_per_s'] > 5.0


def test_commanded_direction_and_its_part_along_the_weakest_direction():
    halt = halts.analyze(wrist_session())['halts'][0]
    command = halt['command']
    assert command['lead_mm'] == pytest.approx(30.0, abs=0.5)
    assert np.allclose(command['lead_direction'], [0, 1, 0], atol=1e-6)
    assert command['lead_rotation_deg'] == pytest.approx(0.0, abs=1e-6)
    # Near the wrist singularity the tool can hardly move sideways: that is what was asked.
    assert abs(halt['weak_direction']['linear'][1]) > 0.8
    assert command['along_weak_direction'] > 0.8
    assert command['speed_m_s'] > 0.0


def test_summary_counts_and_distribution_over_engaged_time():
    summary = halts.analyze(wrist_session())['summary']
    assert summary['halts'] == 1
    assert summary['by_reason'] == {'HALT_FOR_SINGULARITY / wrist': 1}
    # Engaged from 1.3 rad to the halt at about 0.1 rad, 0.6 rad/s.
    assert summary['engaged_s'] == pytest.approx(2.0, abs=0.06)
    measure = summary['condition_number']
    assert measure['min'] == pytest.approx(9.27, abs=0.01)
    assert measure['min'] < measure['p50'] < measure['p90'] < measure['p99'] <= measure['max']
    assert 29.0 < measure['max'] < 31.5
    # Above 17 from 0.4 rad on: 0.3 rad of the 1.2 rad.
    assert summary['share_above']['17.0'] == pytest.approx(0.25, abs=0.03)
    assert summary['share_above']['40.0'] == 0.0
    assert summary['share_decelerating'] == pytest.approx(0.25, abs=0.04)
    # From 29.1 (3 % below the hard stop) to the halt: joint 5 from about 0.115 to 0.1 rad.
    assert 0.01 < summary['stalled_s'] < 0.06


def test_status_halt_and_the_hold_it_causes_are_one_event_and_two_halts_are_two():
    rows = wrist_session() + [r for r in wrist_session(start=20.0) if r['k'] != 'description']
    result = halts.analyze(rows)
    assert [h['servo_code'] for h in result['halts']] == [2, 2]
    assert result['halts'][1]['t'] - result['halts'][0]['t'] == pytest.approx(20.0)
    assert result['summary']['by_reason'] == {'HALT_FOR_SINGULARITY / wrist': 2}


def test_hold_without_a_recorded_status_is_still_a_halt():
    rows = [r for r in wrist_session() if r['k'] != 'servo']
    for row in rows:
        if row['k'] == 'state':
            row['servo'] = -1
    halt = halts.analyze(rows)['halts'][0]
    assert halt['servo_code'] is None and halt['servo_status'] == 'not recorded'
    assert halt['hold_reason'] == 'SERVO_HALTED' and halt['resembles'] == 'wrist'


def test_collision_and_joint_bound_halts_are_listed_with_their_code():
    rows = [{'t': 0.0, 'k': 'description', 'urdf': arm_urdf()}]
    for i in range(100):
        t = i * 0.02
        rows.append({'t': t, 'k': 'joints', 'q': READY, 'v': None})
        rows.append({'t': t, 'k': 'servo', 'code': 5 if 40 <= i < 50 else (6 if i >= 90 else 0),
                     'msg': ''})
        rows.append({'t': t, 'k': 'state', 'engaged': i < 40, 'hold': 19 if i >= 40 else 0,
                     'servo': 0, 'tool': None})
    result = halts.analyze(rows)
    assert [h['servo_status'] for h in result['halts']] == ['HALT_FOR_COLLISION', 'JOINT_BOUND']
    assert result['halts'][0]['resembles'] == '' and result['halts'][0]['command'] is None
    assert result['summary']['by_reason'] == {'HALT_FOR_COLLISION': 1, 'JOINT_BOUND': 1}


def test_session_without_halts_or_engagement():
    rows = [{'t': 0.0, 'k': 'description', 'urdf': arm_urdf()},
            {'t': 0.1, 'k': 'joints', 'q': READY, 'v': None},
            {'t': 0.1, 'k': 'state', 'engaged': False, 'hold': 5, 'servo': -1, 'tool': None}]
    result = halts.analyze(rows)
    assert result['halts'] == [] and result['summary']['engaged_s'] == 0.0
    assert 'halts: 0' in halts.format_report(result)


def test_recording_without_a_description_needs_a_urdf():
    rows = wrist_session(description=False)
    with pytest.raises(halts.SessionError):
        halts.analyze(rows)
    assert len(halts.analyze(rows, urdf=arm_urdf())['halts']) == 1


def test_report_names_everything_the_table_must_show():
    text = halts.format_report(halts.analyze(wrist_session()))
    for wanted in ('halt 1', 'servo 2 HALT_FOR_SINGULARITY', 'hold SERVO_HALTED',
                   'joints (rad): +0.000 -0.600 -0.700 +0.000', 'condition number 30.',
                   '1.0 s before', 'nearest joint limit: xarm_joint3', 'resembles: wrist',
                   'target ahead of the tool: 30 mm (+0.00 +1.00 +0.00)',
                   'commanded motion:', 'halts: 1', '1  HALT_FOR_SINGULARITY / wrist',
                   'condition number over engaged time', 'share of engaged time at or above'):
        assert wanted in text, wanted
