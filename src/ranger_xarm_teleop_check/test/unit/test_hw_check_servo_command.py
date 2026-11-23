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
"""The teleop stage's rules on MoveIt Servo's output (G-18): synthetic trajectory messages."""

import datetime
import json

import numpy as np

from ranger_xarm_teleop_check import report as rep, rules, stats, sysinfo
from ranger_xarm_teleop_check.rules import FAIL, INFO, NOT_JUDGED, Options, PASS
from test_hw_check_rules import _teleop_data, motion_data

LIMIT = 2.0
PERIOD = 0.02
STEP_NS = 20_000_000
EPOCH_NS = 1_700_000_000 * 1_000_000_000          # real stamps are far from zero
WEIGHTS = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
NAMES = tuple(reversed(stats.ARM_JOINTS))          # by name, not by position in the message
RAMP = LIMIT * PERIOD                              # velocity change per cycle at the limit


def commands(*parts):
    """Return the joint velocity (rad/s) per cycle from lists of scalar speeds."""
    return np.concatenate([np.asarray(p, dtype=float) for p in parts])[:, None] * WEIGHTS


def up(to, accel=LIMIT):
    return np.arange(1, int(round(to / (accel * PERIOD))) + 1) * accel * PERIOD


def down(frm, steps):
    """Return ``steps`` speeds falling linearly from ``frm`` to zero (the last is zero)."""
    return frm * (1.0 - np.arange(1, steps + 1) / steps)


def servo_messages(cmds, first_ns=EPOCH_NS, start=0.0, resend=False):
    """
    Return ``data['command']`` rows for Servo-like output: one point per 20 ms cycle.

    The point of cycle k is the sum of the first k commands times the period; a message is
    stamped at the cycle and holds one point at time_from_start one period. ``resend`` adds
    the next message's state again in each message, as Servo does in later windows.
    """
    pos = np.vstack([np.zeros(6), np.cumsum(cmds * PERIOD, axis=0)])
    rows = []
    for k in range(len(pos)):
        points = [(PERIOD, tuple(float(x) for x in pos[k][::-1]))]
        rows.append((start + k * PERIOD, first_ns + k * STEP_NS - STEP_NS, NAMES, points))
    if resend:
        rows = [(a, s, n, p + ([(PERIOD + PERIOD, rows[k + 1][3][0][1])]
                               if k + 1 < len(rows) else [])) for k, (a, s, n, p)
                in enumerate(rows)]
    return rows


def servo_data(cmds, release_at=None, capture=(0.0, 100.0), engage_at=None, **kw):
    """
    Return the data of a teleop run on the Servo output.

    ``release_at`` is the cycle whose point is the first after the release; ``engage_at`` the
    cycle of a later re-engage. ``capture`` is (began, ended) on the arrival clock; arrivals
    start at 10 s.
    """
    rows = servo_messages(cmds, start=10.0, **kw)
    clutch = [(EPOCH_NS + i * STEP_NS, True) for i in range(5)]
    if release_at is not None:
        # The release is stamped just before the point of cycle ``release_at``.
        stamp = EPOCH_NS + release_at * STEP_NS - 5_000_000
        clutch.append((stamp, False))
        if engage_at is not None:
            clutch.append((EPOCH_NS + engage_at * STEP_NS, True))
    return {'command': rows, 'command_capture': {'began': capture[0], 'ended': capture[1]},
            'clutch': clutch}


def options(limit=LIMIT, given=True):
    return Options(limits_given=given, joint_acceleration_limit=limit, servo_latency_s=0.0)


def judge(data, opts=None, latency=0.0):
    cm = rules.command_measures(data, rules._command_limit(opts or options())[0], latency)
    return (cm, rules.rule_command_acceleration(cm, opts or options()),
            rules.rule_stop_time(cm, latency))


# ---- stats: the position path ----------------------------------------------------------

def test_points_are_taken_by_joint_name_with_time_from_start():
    names = ('other',) + stats.ARM_JOINTS[::-1]
    pos = (9.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
    path = stats.command_points([(2_000_000_000, names, [(0.02, pos)])])
    assert path['origin_ns'] == 2_000_000_000 and path['skipped'] == 0
    assert np.allclose(path['t'], [0.02])
    assert path['p'][0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_messages_without_the_six_joints_or_with_odd_lengths_are_skipped():
    full = stats.ARM_JOINTS
    rows = [(1, full[:5], [(0.0, (0.0,) * 5)]), (2, full, [(0.0, (0.0,) * 5)]),
            (3, full, []), (4, full, [(0.0, (0.0,) * 6)])]
    path = stats.command_points(rows)
    assert path['skipped'] == 3 and len(path['t']) == 1
    empty = stats.command_points([])
    assert empty['origin_ns'] is None and len(empty['t']) == 0


def test_points_resent_in_later_windows_are_merged_and_sorted():
    names = stats.ARM_JOINTS
    a = (1.0,) * 6
    b = (2.0,) * 6
    c = (3.0,) * 6
    # Out of order on arrival; b and c are sent twice, the second time 0.5 ms later.
    rows = [(1_000_000_000 + 40_500_000, names, [(0.02, c)]),
            (1_000_000_000, names, [(0.02, a), (0.04, b)]),
            (1_000_000_000 + 20_000_000, names, [(0.02, b), (0.04, c)])]
    path = stats.command_points(rows)
    assert len(path['t']) == 3
    assert np.allclose(path['t'], [0.02, 0.04, 0.06], atol=1e-3)
    assert path['p'][:, 0].tolist() == [1.0, 2.0, 3.0]


def test_runs_split_at_gaps_over_a_tenth_of_a_second():
    t = [0.0, 0.02, 0.04, 0.2, 0.22, 0.5]
    assert stats.split_runs(t) == [(0, 3), (3, 5), (5, 6)]
    assert stats.split_runs([0.0, 0.1]) == [(0, 2)]
    assert stats.split_runs([]) == []


def test_a_ramp_at_the_limit_reads_the_limit_at_every_point_including_the_first():
    cmds = commands(up(0.8), [0.8] * 5, down(0.8, 20))
    path = stats.command_points([m[1:] for m in servo_messages(cmds)])
    an = stats.command_analysis(path['t'], path['p'])
    peaks = np.nanmax(np.abs(an['acc']), axis=0)
    assert np.allclose(peaks, LIMIT * WEIGHTS)
    first_moving = int(np.nonzero(np.abs(an['acc'][:, 0]) > 0)[0][0])
    assert abs(an['acc'][first_moving, 0] - LIMIT) < 1e-6


def test_a_stream_that_starts_moving_reads_a_spike_unless_the_start_is_open():
    cmds = commands([0.8] * 10, down(0.8, 20))
    path = stats.command_points([m[1:] for m in servo_messages(cmds)])
    closed = stats.command_analysis(path['t'], path['p'])
    assert abs(closed['acc'][0, 0]) > 10 * LIMIT
    opened = stats.command_analysis(path['t'], path['p'], open_start=True)
    assert np.isnan(opened['acc'][0, 0])
    assert np.nanmax(np.abs(opened['acc'])) <= LIMIT * (1 + 1e-6)


def test_capture_edges():
    assert stats.capture_edges(10.0, 20.0, 0.0, 30.0) == (False, False)
    assert stats.capture_edges(0.05, 20.0, 0.0, 30.0) == (True, False)
    assert stats.capture_edges(10.0, 29.95, 0.0, 30.0) == (False, True)
    assert stats.capture_edges(None, None, 0.0, 30.0) == (False, False)


def test_clutch_events():
    s = [(1, False), (2, True), (3, True), (4, False), (5, False), (6, True), (7, False)]
    assert stats.clutch_events(s) == ([4, 7], [2, 6])
    assert stats.clutch_events([(1, True)]) == ([], [])


def test_resampled_acceleration_of_a_ramp_at_two_reads_two():
    t = np.arange(0, 1.5, 1 / 150) + 1.7e9
    x = (t - 1.7e9) ** 2                    # acceleration 2.0
    pos = np.outer(x, np.arange(1, 7))
    peaks = stats.resampled_peak_acceleration(t, pos)
    assert np.allclose(peaks, 2.0 * np.arange(1, 7), rtol=1e-6)
    assert stats.resampled_peak_acceleration([0.0, 1.0], np.zeros((2, 6))) == [None] * 6


# ---- the rule: commanded joint acceleration --------------------------------------------

NAME = 'commanded joint acceleration (MoveIt Servo output)'


def test_a_compliant_ramp_at_exactly_the_limit_passes_with_no_skip():
    cm, accel, _ = judge(servo_data(commands(up(0.8), [0.8] * 5, down(0.8, 20))))
    assert accel.rule == NAME and accel.verdict == PASS, accel.line()
    assert cm['above_1x'] == 0 and cm['above_2x'] == 0
    assert '0 points above 1 x limit' in accel.measured
    assert 'p50 20.0 p99' in accel.measured and 'ms' in accel.measured
    assert 'x joint_acceleration_limit_rad_s2 = 4 rad/s^2' in accel.threshold
    assert 'a Servo skip can join two consecutive command steps into one' in accel.threshold
    assert 'up to 2 x the limit' in accel.threshold


def test_resent_points_do_not_change_the_reading():
    cmds = commands(up(0.8), [0.8] * 5, down(0.8, 20))
    plain = judge(servo_data(cmds))[0]
    resent = judge(servo_data(cmds, resend=True))[0]
    assert resent['points'] == plain['points']
    assert resent['peak_overall'] == plain['peak_overall']


def test_a_step_joined_by_a_skip_passes_with_one_point_counted():
    skip = np.concatenate([up(0.4), [0.4 + 2 * RAMP], [0.4 + 2 * RAMP] * 5])
    cmds = commands(skip, down(0.4 + 2 * RAMP, 20))
    cm, accel, _ = judge(servo_data(cmds))
    assert accel.verdict == PASS, accel.line()
    assert cm['above_1x'] == 1 and cm['above_2x'] == 0
    assert '1 points above 1 x limit (joined by a skip)' in accel.measured
    assert '2.00 x limit' in accel.measured


def test_two_and_a_half_times_the_limit_fails_as_a_stop():
    skip = np.concatenate([up(0.4), [0.4 + 2.5 * RAMP], [0.4 + 2.5 * RAMP] * 5])
    cmds = commands(skip, down(0.4 + 2.5 * RAMP, 20))
    cm, accel, _ = judge(servo_data(cmds))
    assert accel.verdict == FAIL and cm['above_2x'] == 1
    assert 'a stop: (procedure stage 4)' in accel.advice
    assert 'a stop: (procedure stage 4)' in accel.line()


def test_without_limits_or_without_the_value_it_is_not_judged():
    data = servo_data(commands(up(0.8), [0.8] * 5, down(0.8, 20)))
    cm = rules.command_measures(data, None)
    r = rules.rule_command_acceleration(cm, Options())
    assert r.verdict == NOT_JUDGED and '--limits' in r.advice
    assert cm['above_1x'] is None and 'peak 1: ' in r.measured
    for bad in (None, 'x', 0.0, -1.0, True):
        r = rules.rule_command_acceleration(cm, options(limit=bad))
        assert r.verdict == NOT_JUDGED and 'joint_acceleration_limit_rad_s2' in r.advice


def test_no_motion_or_no_messages_is_not_judged():
    still = commands([0.0] * 30)
    r = judge(servo_data(still))[1]
    assert r.verdict == NOT_JUDGED and 'no Servo output with motion' in r.advice
    r = judge({'command': []})[1]
    assert r.verdict == NOT_JUDGED and rules.DEFAULT_COMMAND_TOPIC in r.advice
    assert '0 points' in r.measured


def test_a_capture_that_cut_a_moving_stream_is_not_read_as_a_stop():
    cmds = commands([0.8] * 10, down(0.8, 20))
    abrupt = judge(servo_data(cmds, capture=(0.0, 100.0)))[1]
    assert abrupt.verdict == FAIL                    # a real stream that starts at speed
    cut = judge(servo_data(cmds, capture=(9.95, 100.0)))
    assert cut[1].verdict == PASS and cut[0]['open_start']
    assert 'cut the stream at its start' in cut[1].measured


# ---- the rule: stop time after a release ---------------------------------------------------

STOP = 'stop time after a release'


def release_run(steps, hold=10, accel=LIMIT, **kw):
    """Run up at the limit, hold, release, ramp down over ``steps`` cycles (last speed zero)."""
    peak = 0.36
    pre = commands(up(peak, accel), [peak] * hold)
    cmds = commands(up(peak, accel), [peak] * hold, down(peak, steps), [0.0] * 3)
    return servo_data(cmds, release_at=len(pre), **kw)


def test_a_release_with_a_ramp_of_0_18_s_passes():
    cm, accel, stop = judge(release_run(9))
    assert stop.rule == STOP and stop.verdict == PASS, stop.line()
    assert accel.verdict == PASS
    # Measured from the release (5 ms before the ramp's first point) to the end of motion.
    assert len(cm['stops']) == 1 and abs(cm['stops'][0]['duration_s'] - 0.165) < 1e-6
    assert '1 releases with motion of 1 seen' in stop.measured
    assert 'longest 0.165 s' in stop.measured
    assert '0.2 s + 0.02 s command period = 0.22 s' in stop.threshold
    assert 'up to one cycle after the release' in stop.threshold


def test_a_release_with_a_ramp_of_0_30_s_fails_as_a_stop():
    cm, accel, stop = judge(release_run(15))
    assert accel.verdict == PASS                     # slower than the limit, but too long
    assert stop.verdict == FAIL and 'a stop: (procedure stage 4)' in stop.line()
    assert abs(cm['stops'][0]['duration_s'] - 0.285) < 1e-6     # from the release, 5 ms early


def test_the_stop_time_limit_is_the_limit_plus_one_cycle():
    # Measured from the release, 5 ms before the ramp's first point.
    assert judge(release_run(11))[2].verdict == PASS      # 0.205 s
    assert judge(release_run(12))[2].verdict == FAIL      # 0.225 s, above 0.22 s


def test_each_release_is_listed_and_the_window_ends_at_the_next_engage():
    peak = 0.36
    first = commands(up(peak), [peak] * 5, down(peak, 9), [0.0] * 10)
    second = commands(up(peak), [peak] * 5, down(peak, 9), [0.0] * 3)
    cmds = np.vstack([first, second])
    k1 = len(up(peak)) + 5
    k2 = len(first) + len(up(peak)) + 5
    data = servo_data(cmds, release_at=k1)
    data['clutch'] += [(EPOCH_NS + (k1 + 14) * STEP_NS, True),
                       (EPOCH_NS + k2 * STEP_NS - 5_000_000, False)]
    cm, _, stop = judge(data)
    assert stop.verdict == PASS and len(cm['stops']) == 2 and cm['releases_seen'] == 2
    assert all(abs(s['duration_s'] - 0.165) < 1e-6 for s in cm['stops'])
    assert '2 releases with motion of 2 seen' in stop.measured


def test_a_release_with_no_motion_or_none_at_all_is_not_judged():
    still = commands([0.0] * 40)
    stop = judge(servo_data(still, release_at=20))[2]
    assert stop.verdict == NOT_JUDGED and 'no release with motion' in stop.advice
    moving = commands(up(0.8), [0.8] * 5, down(0.8, 20))
    stop = judge(servo_data(moving))[2]
    assert stop.verdict == NOT_JUDGED and '0 releases with motion of 0 seen' in stop.measured


def test_a_stream_that_ends_while_still_moving_is_reported_not_passed():
    peak = 0.36
    pre = commands(up(peak), [peak] * 10)
    cmds = commands(up(peak), [peak] * 10, down(peak, 15)[:6])      # cut after six cycles
    end = 10.0 + len(cmds) * PERIOD + 0.01
    data = servo_data(cmds, release_at=len(pre), capture=(0.0, end))
    cm, accel, stop = judge(data)
    assert cm['open_end'] and cm['stops'][0]['unfinished']
    assert stop.verdict == NOT_JUDGED
    assert '1 stops where Servo output had not come to rest when the stream ended' in (
        stop.measured)
    assert accel.verdict == PASS                      # no rest assumed at the open end


def test_a_stall_of_the_output_inside_a_stop_counts_into_it():
    peak = 0.36
    pre = commands(up(peak), [peak] * 10)
    cmds = commands(up(peak), [peak] * 10, down(peak, 9), [0.0] * 3)
    data = servo_data(cmds, release_at=len(pre))
    # Servo's output stalls for 0.15 s four cycles into the ramp, then goes on.
    cut = len(pre) + 4
    data['command'] = [(a + (0.15 if k > cut else 0.0), s + (150_000_000 if k > cut else 0), n, p)
                       for k, (a, s, n, p) in enumerate(data['command'])]
    cm, accel, stop = judge(data)
    assert cm['runs'] == 2
    assert abs(cm['stops'][0]['duration_s'] - 0.315) < 1e-6
    assert stop.verdict == FAIL and not cm['stops'][0]['unfinished']


def test_a_release_at_the_very_end_of_a_moving_stream_is_unfinished():
    peak = 0.36
    cmds = commands(up(peak), [peak] * 10)
    end = 10.0 + (len(cmds) + 1) * PERIOD + 0.01
    cm, _, stop = judge(servo_data(cmds, release_at=len(cmds), capture=(0.0, end)))
    assert cm['stops'] and cm['stops'][0]['unfinished']
    assert cm['stops'][0]['duration_s'] == 0.0
    assert stop.verdict == NOT_JUDGED and '1 stops where' in stop.measured


def test_an_unfinished_stop_already_too_long_fails():
    peak = 0.36
    pre = commands(up(peak), [peak] * 10)
    cmds = commands(up(peak), [peak] * 10, down(peak, 40)[:20])
    end = 10.0 + len(cmds) * PERIOD + 0.01
    cm, _, stop = judge(servo_data(cmds, release_at=len(pre), capture=(0.0, end)))
    assert cm['stops'][0]['unfinished'] and stop.verdict == FAIL


def test_a_finished_stop_passes_beside_an_unfinished_one():
    peak = 0.36
    first = commands(up(peak), [peak] * 5, down(peak, 9), [0.0] * 10)
    second = commands(up(peak), [peak] * 5, down(peak, 15)[:4])
    cmds = np.vstack([first, second])
    k1 = len(up(peak)) + 5
    k2 = len(first) + len(up(peak)) + 5
    data = servo_data(cmds, release_at=k1, capture=(0.0, 10.0 + len(cmds) * PERIOD + 0.01))
    data['clutch'] += [(EPOCH_NS + (k1 + 14) * STEP_NS, True),
                       (EPOCH_NS + k2 * STEP_NS - 5_000_000, False)]
    cm, _, stop = judge(data)
    assert [s['unfinished'] for s in cm['stops']] == [False, True]
    assert stop.verdict == PASS and '1 stops where' in stop.measured


# ---- the stages --------------------------------------------------------------------------------

def _results(stage, data, opts):
    return {r.rule: r for r in rules.evaluate(stage, data, opts)[1]}


def test_the_teleop_stage_does_not_judge_the_joint_state_acceleration():
    opts = Options(joint_limits_given=True,
                   joint_acceleration_limits={f'xarm_joint{i}': 0.001 for i in range(1, 7)})
    res = _results('teleop', motion_data(), opts)
    assert 'peak joint acceleration' not in res
    info = res['joint-state acceleration (20 ms, INFO)']
    assert info.verdict == INFO
    assert 'differentiated joint states, 20 ms window; on the real arm encoder noise adds an ' \
        "unknown amount; MoveIt's max_acceleration 0.001 for reference" in info.line()
    none = _results('teleop', motion_data(), Options())
    assert 'not given' in none['joint-state acceleration (20 ms, INFO)'].threshold
    assert res[NAME].verdict == NOT_JUDGED and res[STOP].verdict == NOT_JUDGED


def test_the_motion_stage_keeps_its_rule_and_has_no_servo_rules():
    opts = Options(joint_limits_given=True,
                   joint_acceleration_limits={f'xarm_joint{i}': 1e6 for i in range(1, 7)})
    res = _results('motion', motion_data(), opts)
    assert res['peak joint acceleration'].verdict == PASS
    assert NAME not in res and STOP not in res
    assert not any('20 ms' in r for r in res)
    tight = Options(joint_limits_given=True,
                    joint_acceleration_limits={f'xarm_joint{i}': 1e-9 for i in range(1, 7)})
    assert _results('motion', motion_data(), tight)['peak joint acceleration'].verdict in (
        PASS, FAIL)


def test_the_teleop_stage_judges_the_servo_output_end_to_end(tmp_path):
    data = _teleop_data()
    data.update(release_run(9))
    opts = Options(limits_given=True, tool_pose_timeout_s=1.0, max_lead_translation_m=0.05,
                   joint_acceleration_limit=LIMIT, servo_latency_s=0.0)
    measures, results = rules.evaluate('teleop', data, opts)
    by = {r.rule: r for r in results}
    assert by[NAME].verdict == PASS and by[STOP].verdict == PASS
    assert measures['servo_command']['stops'][0]['unfinished'] is False
    start = datetime.datetime(2026, 1, 2, 3, 4, 5).astimezone()
    doc = rep.build_report('teleop', start, start, measures, results, {}, ['x'], [], 'abc',
                           rules.exit_status(results))
    path = rep.write_report(rep.report_path(str(tmp_path), 'teleop', start), doc)
    back = json.loads(open(path).read(), parse_constant=_reject)
    assert back['measures']['servo_command']['stops'][0]['unfinished'] is False
    assert back['measures']['joint_state_acceleration_20ms']['window_s'] == 0.02


def _reject(constant):
    raise ValueError('not JSON: ' + constant)


# ---- the release on Servo's point times (max_expected_latency) -----------------------------

def _shifted(data, seconds):
    """Return ``data`` with every Servo point stamped ``seconds`` later, as Servo does."""
    shift = int(round(seconds * 1e9))
    rows = [(r[0], r[1] + shift) + tuple(r[2:]) for r in data['command']]
    return dict(data, command=rows)


def test_servos_latency_moves_the_release_onto_its_point_times():
    """Points stamped 0.06 s ahead read the same stop once the release is moved by 0.06 s."""
    plain = judge(release_run(9))[0]['stops'][0]['duration_s']
    data = _shifted(release_run(9), 0.06)
    late = judge(data, latency=0.06)
    assert abs(late[0]['stops'][0]['duration_s'] - plain) < 1e-6 and late[2].verdict == PASS
    # Not moved, the same points would read the lead as part of the stop.
    unmoved = judge(data, latency=0.0)[0]['stops'][0]['duration_s']
    assert abs(unmoved - (plain + 0.06)) < 1e-6


def test_without_servos_latency_the_stop_is_not_judged():
    cm = rules.command_measures(release_run(9), LIMIT)
    stop = rules.rule_stop_time(cm, None)
    assert stop.verdict == NOT_JUDGED and 'max_expected_latency' in stop.measured


def test_the_latency_is_read_from_the_teleop_servo_yaml():
    import os
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ranger_xarm_teleop_bringup',
                        'config', 'servo.yaml')
    assert sysinfo.servo_latency(path) == 0.06
    assert sysinfo.servo_latency(os.path.join(os.path.dirname(__file__), 'missing.yaml')) is None
