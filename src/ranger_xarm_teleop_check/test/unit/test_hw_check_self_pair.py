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
Teleop stage INFO results: reversals, noise with the arm still, step times by self pair.

Reversals of the command and of Servo's output; the reported velocity with the arm still; the
teleop node's step times split by whether a self pair is active.
"""

import datetime
import json
import math

import numpy as np
import pytest

from ranger_xarm_teleop_check import report as rep
from ranger_xarm_teleop_check import rules, stats
from ranger_xarm_teleop_check.rules import Options

NAMES = stats.ARM_JOINTS
HZ = 100.0


def arm_rows(n, pos_fn, vel_fn, start=0.0):
    """``data['arm']`` rows: (arrival, system time, stamp, six positions, six velocities, ok)."""
    rows = []
    for k in range(n):
        t = start + k / HZ
        rows.append((t, t, t, tuple(pos_fn(t)), None if vel_fn is None else tuple(vel_fn(t)),
                     True))
    return rows


def flip(t, joint=3, speed=0.1, period=0.15):
    """Six velocities, one joint flipping sign every ``period`` s."""
    v = [0.0] * 6
    v[joint] = speed if int(t / period) % 2 == 0 else -speed
    return v


def test_the_still_windows_need_no_command_and_less_than_2_mm_over_each_half_second():
    t = np.arange(0, 5, 0.01)
    p = np.zeros(len(t))
    p[t >= 3.0] = 0.01                         # a step at 3 s
    windows = stats.still_windows(t, p, [])
    assert [(round(a, 2), round(b, 2)) for a, b in windows] == [(0.5, 2.99), (3.5, 4.99)]
    commanded = stats.still_windows(t, np.zeros(len(t)), [1.0])
    assert [(round(a, 2), round(b, 2)) for a, b in commanded] == [(1.51, 4.99)]
    drift = stats.still_windows(t, 0.01 * t, [])         # 5 mm in 0.5 s
    assert drift == []


def test_a_gap_in_the_samples_ends_a_still_stretch():
    t = np.concatenate([np.arange(0, 2, 0.01), np.arange(2.5, 4.5, 0.01)])
    windows = stats.still_windows(t, np.zeros(len(t)), [])
    assert len(windows) == 2 and windows[0][1] < 2.0 and windows[1][0] > 2.8


def test_noise_is_measured_inside_the_still_windows_only():
    t = np.arange(0, 6, 0.01)
    v = np.zeros(len(t))
    v[(t >= 4.0)] = 1.0                        # the arm moves from 4 s: not in a window
    windows = [(0.0, 3.0)]
    noise = stats.still_noise(t, v, windows, 0.02, 1.0)
    assert noise['rms'] == 0.0 and noise['peak'] == 0.0 and noise['sign_changes'] == 0
    assert noise['seconds'] == 3.0 and noise['windows'] == 1


def test_noise_rms_peak_and_sign_changes_of_a_chattering_velocity():
    t = np.arange(0, 4, 0.01)
    v = np.where(np.arange(len(t)) // 3 % 2 == 0, 0.05, -0.05)   # a flip every 0.03 s
    noise = stats.still_noise(t, v, [(0.0, 3.99)], 0.02, 1.0)
    assert noise['peak'] == pytest.approx(0.05)
    assert noise['rms'] == pytest.approx(0.05)
    assert noise['sign_changes'] == 34         # 0.03 s apart: 34 fit in under 1 s


def reversal_data(jog=True, command=None):
    data = {'arm': arm_rows(400, lambda t: [0.0] * 6, lambda t: flip(t, 3) if t < 2 else
                            [0.0] * 6)}
    if jog:
        # 200 messages at 50 Hz: joint 6 flips every 8 messages (0.16 s), joint 4 does not
        data['jog'] = [(NAMES, tuple(0.1 if k < 100 else 0.0 for _ in range(3))
                        + (0.1, 0.1, 0.1 if (k // 8) % 2 == 0 else -0.1)) for k in range(200)]
        data['jog_arrival'] = [k * 0.02 for k in range(200)]
    if command is not None:
        data['command'], data['command_velocities'] = command
    return data


def servo_messages(with_velocity, n=200):
    """Servo output: joint 4 flips every 8 messages (0.16 s); joint 6 holds."""
    messages, velocities = [], []
    angle = 0.0
    for k in range(n):
        t = k * 0.02
        v = [0.0] * 6
        v[3] = 0.1 if (k // 8) % 2 == 0 else -0.1
        angle += v[3] * 0.02
        pos = [0.0] * 6
        pos[3] = angle
        messages.append((t, int(t * 1e9), NAMES, [(0.02, tuple(pos))]))
        velocities.append((tuple(v),) if with_velocity else ((),))
    return messages, velocities


def test_reversals_count_the_command_the_servo_output_and_the_joint_states():
    messages, velocities = servo_messages(with_velocity=True)
    data = reversal_data(command=(messages, velocities))
    mm = rules.motion_measures(data)
    rm = rules.reversal_measures(data, mm)
    assert rm['command_source'] == 'velocities'
    assert rm['command']['xarm_joint6'] > 0 and rm['command']['xarm_joint4'] == 0
    assert rm['servo']['xarm_joint4'] > 0 and rm['servo']['xarm_joint6'] == 0
    # the same counting as the oscillation rule: joint 4 of the joint states, 0.15 s flips
    assert rm['joint_states']['xarm_joint4'] == mm['wrist_sign_changes_per_s']['xarm_joint4'] == 7
    assert rm['jog_messages'] == 200 and rm['joint_states']['xarm_joint6'] == 0
    # a flip every 0.16 s: 7 fit in under 1 s
    assert rm['servo']['xarm_joint4'] == 7 and rm['command']['xarm_joint6'] == 7
    result = rules.info_reversals_command_servo(rm, Options())
    assert result.verdict == rules.INFO
    assert result.measured.startswith(
        'joint 4: command 0, Servo output 7, joint states 7; '
        'joint 6: command 7, Servo output 0, joint states 0 (most sign changes in any 1 s '
        'window; command: joint-jog velocities of 200 messages on ')
    assert 'Servo output: velocities of the point of 200 messages' in result.measured
    assert 'joint states: velocity from reported' in result.measured


def test_servo_output_without_velocities_is_differenced_from_positions():
    messages, velocities = servo_messages(with_velocity=False)
    data = reversal_data(command=(messages, velocities))
    rm = rules.reversal_measures(data, rules.motion_measures(data))
    assert rm['command_source'] == 'positions'
    assert rm['servo']['xarm_joint4'] > 0
    text = rules.info_reversals_command_servo(rm, Options()).measured
    assert 'no velocities in the points, central difference of the positions' in text


def test_a_command_message_list_without_the_velocity_key_falls_back_to_positions():
    messages, _ = servo_messages(with_velocity=True)
    data = reversal_data(command=(messages, []))
    rm = rules.reversal_measures(data, rules.motion_measures(data))
    assert rm['command_source'] == 'positions'


def test_reversals_say_so_for_a_source_without_messages():
    data = {'arm': arm_rows(300, lambda t: [0.0] * 6, lambda t: flip(t, 3))}
    rm = rules.reversal_measures(data, rules.motion_measures(data))
    result = rules.info_reversals_command_servo(rm, Options(jog_topic='/j', command_topic='/c'))
    assert result.measured.startswith('joint 4: command n/a, Servo output n/a, joint states 7;')
    assert 'command: no message on /j' in result.measured
    assert 'Servo output: no message on /c' in result.measured


def test_reversals_with_no_joint_states_say_so():
    rm = rules.reversal_measures({}, rules.motion_measures({}))
    result = rules.info_reversals_command_servo(rm, Options())
    assert 'joint states n/a' in result.measured
    assert 'joint states: no six-joint message' in result.measured


def test_the_oscillation_rule_is_unchanged_by_the_new_data():
    data = reversal_data()
    mm = rules.motion_measures(data)
    assert rules.rule_oscillation(mm).verdict == rules.FAIL
    assert rules.OSCILLATION_LIMIT == 4 and rules.OSCILLATION_SPEED == 0.02


def still_data(noise=0.0, jog_until=None, seconds=4.0, velocities=True):
    n = int(seconds * HZ)
    rows = arm_rows(n, lambda t: [0.0] * 6,
                    (lambda t: [noise * (1 if round(t * HZ) // 3 % 2 else -1)] * 6) if velocities
                    else None)
    data = {'arm': rows, 'jog': [], 'jog_arrival': []}
    if jog_until:
        for k in range(int(jog_until / 0.02)):
            data['jog'].append((NAMES, (0.1,) * 6))
            data['jog_arrival'].append(k * 0.02)
    return data


def test_still_noise_is_reported_for_joints_4_and_6_with_the_rule_stated():
    sm = rules.still_noise_measures(still_data(noise=0.03))
    result = rules.info_still_velocity_noise(sm)
    assert result.verdict == rules.INFO
    assert result.measured.startswith('joint 4: still 3.5 s in 1 window, RMS 0.0300 rad/s, '
                                      'peak 0.0300 rad/s, most sign changes above 0.02 rad/s '
                                      'in any 1 s window ')
    assert 'joint 6: still' in result.measured and 'no joint-jog message' in result.measured
    assert 'no nonzero joint-jog command for 0.5 s before and during' in result.threshold
    assert 'within 0.002 rad over each 0.5 s span' in result.threshold
    # alternating every 30 ms: 34 flips fit in one 1 s window
    assert sm['joints']['xarm_joint4']['sign_changes'] == 34


def test_a_command_early_in_the_run_leaves_the_rest_as_still_time():
    sm = rules.still_noise_measures(still_data(jog_until=1.0))
    s = sm['joints']['xarm_joint4']
    assert s['windows'] == 1 and s['seconds'] == pytest.approx(2.5, abs=0.02)


def test_no_still_window_of_one_second_is_said():
    sm = rules.still_noise_measures(still_data(jog_until=3.0, seconds=4.0))
    text = rules.info_still_velocity_noise(sm).measured
    assert text.startswith('joint 4: no still window of at least 1 s; joint 6: no still '
                           'window of at least 1 s')


def test_a_moving_arm_has_no_still_window():
    rows = arm_rows(400, lambda t: [0.1 * t] * 6, lambda t: [0.1] * 6)
    sm = rules.still_noise_measures({'arm': rows})
    assert 'no still window' in rules.info_still_velocity_noise(sm).measured


def test_absent_reported_velocity_is_said_and_nothing_more():
    sm = rules.still_noise_measures(still_data(velocities=False))
    result = rules.info_still_velocity_noise(sm)
    assert result.measured.startswith('the driver reports no velocity in /joint_states')
    assert sm['joints'] == {}


def test_still_noise_without_joint_states_says_so():
    result = rules.info_still_velocity_noise(rules.still_noise_measures({}))
    assert result.measured == 'no six-joint message'


FULL = ('link keep-out by self pair: active 40 steps, median 1.10 ms, p99 1.90 ms, longest '
        '2.50 ms; not active 812 steps, median 0.91 ms, p99 1.40 ms, longest 2.10 ms.')
ZERO = ('link keep-out by self pair: active 0 steps; not active 812 steps, median 0.91 ms, '
        'p99 1.40 ms, longest 2.10 ms.')
CYCLE = ('teleop cycle computation: 3000 cycles, median 0.80 ms, p99 3.20 ms, longest '
         '12.00 ms, 2 over the 20 ms period.')


def test_the_self_pair_sentence_is_parsed():
    got = stats.parse_teleop_summary(FULL)
    assert got['cycles'] is None
    assert got['pair']['active'] == {'steps': 40, 'median_ms': 1.10, 'p99_ms': 1.90,
                                     'longest_ms': 2.50}
    assert got['pair']['inactive'] == {'steps': 812, 'median_ms': 0.91, 'p99_ms': 1.40,
                                       'longest_ms': 2.10}


def test_a_half_without_steps_has_no_numbers():
    got = stats.parse_teleop_summary(ZERO)
    assert got['pair']['active'] == {'steps': 0, 'median_ms': None, 'p99_ms': None,
                                     'longest_ms': None}
    assert got['pair']['inactive']['steps'] == 812
    both = stats.parse_teleop_summary(
        'link keep-out by self pair: active 3 steps, median 1.00 ms, p99 1.00 ms, longest '
        '1.00 ms; not active 0 steps.')
    assert both['pair']['inactive']['steps'] == 0 and both['pair']['active']['steps'] == 3


def test_both_sentences_are_found_among_other_sentences():
    text = ('Slow target cycles: 2 over 30 ms. ' + FULL + ' Late starts: 0. ' + CYCLE
            + ' Gripper idle.')
    got = stats.parse_teleop_summary(text)
    assert got['pair']['active']['steps'] == 40
    assert got['cycles'] == {'cycles': 3000, 'median_ms': 0.80, 'p99_ms': 3.20,
                             'longest_ms': 12.00, 'over': 2, 'period_ms': 20.0}
    assert stats.parse_teleop_summary('nothing of the kind here') == {'pair': None,
                                                                      'cycles': None}


def test_summaries_are_totalled_over_the_run():
    second = FULL.replace('active 40 steps, median 1.10', 'active 60 steps, median 1.30'
                          ).replace('p99 1.90', 'p99 2.20').replace('longest 2.50',
                                                                    'longest 2.40')
    data = {'rosout': {'teleop_summaries': [FULL + ' ' + CYCLE, second, ZERO]}}
    km = rules.keepout_step_measures(data)
    active, idle = km['pair']['active'], km['pair']['inactive']
    assert active['steps'] == 100 and active['summaries'] == 3
    assert active['max_p99_ms'] == 2.20 and active['longest_ms'] == 2.50
    assert active['median_of_medians_ms'] == pytest.approx(1.20)
    assert idle['steps'] == 812 * 3
    assert km['cycles']['cycles'] == 3000 and km['cycles']['over'] == 2
    result = rules.info_keepout_step_time(km)
    assert result.verdict == rules.INFO
    assert result.measured == (
        'link keep-out by self pair (3 summaries): active 100 steps, median 1.20 ms, '
        'largest p99 2.20 ms, longest 2.50 ms; not active 2436 steps, median 0.91 ms, '
        'largest p99 1.40 ms, longest 2.10 ms; teleop cycle computation (1 summary): '
        '3000 cycles, median 0.80 ms, largest p99 3.20 ms, longest 12.00 ms, 2 over the '
        'period')


def test_an_all_zero_half_is_written_without_numbers():
    km = rules.keepout_step_measures({'rosout': {'teleop_summaries': [ZERO]}})
    text = rules.info_keepout_step_time(km).measured
    assert 'active 0 steps; not active 812 steps, median 0.91 ms' in text
    assert 'no cycle-computation summary' in text


def test_without_a_summary_it_says_so():
    for data in ({}, {'rosout': {'teleop_summaries': []}},
                 {'rosout': {'teleop_summaries': ['unrelated text']}}):
        result = rules.info_keepout_step_time(rules.keepout_step_measures(data))
        assert result.measured == (
            'no keep-out summary from the teleop node in this run (it logs one every 60 s '
            'while commanding)')


def test_the_teleop_stage_reports_the_three_and_judges_nothing_by_them():
    data = reversal_data(command=servo_messages(True))
    data['rosout'] = {'teleop_summaries': [FULL]}
    measures, results = rules.evaluate('teleop', data, Options())
    info = {r.rule: r for r in results}
    for name in (rules.REVERSALS_INFO, rules.STILL_NOISE_INFO, rules.KEEPOUT_STEP_INFO):
        assert info[name].verdict == rules.INFO
    assert 'reversals' in measures and 'still_velocity_noise' in measures
    assert not math.isnan(measures['keepout_step_time']['pair']['active']['steps'])
    for stage in ('motion', 'idle'):
        _, other = rules.evaluate(stage, data, Options())
        assert rules.REVERSALS_INFO not in {r.rule for r in other}


def test_a_cycle_sentence_alone_is_shown_with_the_missing_keep_out_summary():
    km = rules.keepout_step_measures({'rosout': {'teleop_summaries': [CYCLE]}})
    text = rules.info_keepout_step_time(km).measured
    assert text.startswith('no keep-out summary from the teleop node in this run')
    assert '3000 cycles, median 0.80 ms' in text


def test_the_still_stretch_starts_at_its_first_qualified_sample():
    t = np.arange(0, 5, 0.01)
    windows = stats.still_windows(t, np.zeros(len(t)), [2.0])
    assert [(round(a, 2), round(b, 2)) for a, b in windows] == [(0.5, 1.99), (2.51, 4.99)]
    assert stats.still_windows(t, np.zeros(len(t)), [2.0], min_len=3.0) == []


def test_the_measures_are_written_into_the_report(tmp_path):
    data = reversal_data(command=servo_messages(True))
    data['rosout'] = {'teleop_summaries': [FULL + ' ' + CYCLE]}
    measures, results = rules.evaluate('teleop', data, Options())
    when = datetime.datetime(2026, 1, 1)
    document = rep.build_report('teleop', when, when, measures, results, {}, [], [], 'x', 0)
    path = rep.write_report(str(tmp_path / 'r.json'), document)
    written = json.load(open(path))
    assert written['measures']['reversals']['servo']['xarm_joint4'] == 7
    assert 'xarm_joint4' in written['measures']['still_velocity_noise']['joints']
