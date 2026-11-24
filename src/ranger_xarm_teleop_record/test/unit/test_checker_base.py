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
"""The base report of the checker: motion, drive time, and BASE_DURING_CLUTCH (no ROS)."""

import copy

import numpy as np
import pytest
from ranger_xarm_teleop_record import checker, profile
from ranger_xarm_teleop_record.streams import Stream
import synth

CHECKS = profile.load_checks()
T0 = synth.T0
# The clean episode's clutch is engaged 1-4, 5-8 and 9-11.5 s after T0; the base rests in between.
REST_DRIVE = (4.1, 4.9)          # a drive wholly inside the 4-5 s rest, ending with a zero command


def add(ep, name, stamp, fields, has_header=True):
    """Add the profile stream ``name`` to ``ep`` (a copy is not made)."""
    spec = ep.profile['streams'][name]
    stamp = np.asarray(stamp, dtype=float)
    ep.streams[name] = Stream(
        name=name, topic=spec['topic'], kind=spec['kind'], stamp=stamp,
        recv=stamp + synth.LATENCY, fields=fields, has_header=has_header,
        frame_id=[synth.BASE_FRAME] * len(stamp) if has_header else [])
    return ep


def pose_line(x, count):
    """Return ``count`` poses along x with the identity orientation."""
    pose = np.zeros((count, 7))
    pose[:, 0] = x
    pose[:, 6] = 1.0
    return pose


def cmd_rows(first, last, vx=0.3, wz=0.0, hz=50.0, zero_after=True):
    """Return (stamps, twist) of a command from ``first`` to ``last`` s, then one zero row."""
    n = int(round((last - first) * hz))
    stamps = T0 + first + np.arange(n + (1 if zero_after else 0)) / hz
    twist = np.zeros((len(stamps), 6))
    twist[:n, 0] = vx
    twist[:n, 5] = wz
    return stamps, twist


def odom_rows(first, last, vx, hz=20.0, travel=None, ground=False):
    """Return (stamps, fields) of odometry that moves at ``vx`` from ``first`` to ``last`` s."""
    stamps = T0 + np.arange(0.0, 12.0, 1.0 / hz)
    u = stamps - T0
    moving = (u >= first) & (u < last)
    x = np.cumsum(np.where(moving, vx / hz, 0.0)) * (1.0 if travel is None else travel)
    twist = np.zeros((len(stamps), 6))
    twist[moving, 0] = vx
    return stamps, {'pose': pose_line(x + (0.001 if ground else 0.0), len(stamps)),
                    'twist': twist}


def with_drive(first, last, with_odom=True, vx=0.3, odom_first=None, odom_last=None):
    """Return a clean episode with a base drive from ``first`` to ``last`` s after T0."""
    ep = synth.build()
    stamps, twist = cmd_rows(first, last, vx)
    add(ep, 'base_cmd', stamps, {'twist': twist})
    if with_odom:
        o_stamps, fields = odom_rows(first if odom_first is None else odom_first,
                                     last if odom_last is None else odom_last, vx)
        add(ep, 'odom', o_stamps, fields)
        g_stamps, g_fields = odom_rows(first if odom_first is None else odom_first,
                                       last if odom_last is None else odom_last, vx,
                                       travel=1.04, ground=True)
        add(ep, 'ground_truth_odom', g_stamps, g_fields)
    return ep


def with_base_state(ep):
    """Record the base node's state (base driving on), unless the episode has it already."""
    if 'base_state' not in ep.streams:
        stamps = T0 + np.arange(0.0, 12.0, 0.1)
        n = len(stamps)
        add(ep, 'base_state', stamps, {name: np.zeros(n) for name in (
            'state', 'mode', 'reason', 'active_limit')})
    return ep


def run(ep, checks=CHECKS):
    """Check an episode recorded with base driving on."""
    return checker.check_episode(with_base_state(ep), checks)


def flags(report, name):
    """Return the flags of one kind."""
    return [f for f in report['flags'] if f['flag'] == name]


def failing_checks():
    """Return the tolerances with BASE_DURING_CLUTCH among the flags that fail an episode."""
    checks = copy.deepcopy(CHECKS)
    checks['flags']['fail_flags'] = list(checks['flags']['fail_flags']) + ['BASE_DURING_CLUTCH']
    return checks


# ---- moved or not, travel, drive time, cycles ----

def test_base_streams_absent_report_not_recorded_and_raise_nothing():
    report = checker.check_episode(synth.build(), CHECKS)
    assert report['base'] == {'mode': 'parked: no base topics recorded', 'streams': {}}
    assert flags(report, 'BASE_DURING_CLUTCH') == []


def test_a_base_that_never_moved_is_parked_with_zero_cycles_driven():
    ep = synth.build()
    stamps = T0 + np.arange(0.0, 12.0, 0.1)
    add(ep, 'base_cmd', stamps[:1], {'twist': np.zeros((1, 6))})
    add(ep, 'odom', stamps, {'pose': pose_line(0.0, len(stamps)),
                             'twist': np.zeros((len(stamps), 6))})
    report = run(ep)
    base = report['base']
    assert base['mode'] == 'parked: base topics present, no motion'
    assert flags(report, 'BASE_MOVED') == [] and flags(report, 'BASE_DURING_CLUTCH') == []
    assert base['summary']['cmd_cycles'] == 1 and base['summary']['cmd_nonzero_cycles'] == 0
    assert base['summary']['drive_time_s'] == 0.0
    assert base['summary']['travel_m']['odom'] == 0.0


def test_a_drive_in_the_rest_moves_the_base_and_reports_travel_by_both_odometries():
    report = run(with_drive(*REST_DRIVE))
    base = report['base']
    assert base['mode'] == 'moved'
    assert len(flags(report, 'BASE_MOVED')) == 1
    assert 'base_cmd' in flags(report, 'BASE_MOVED')[0]['reason']
    travel = base['summary']['travel_m']
    assert travel['odom'] == pytest.approx(0.24, abs=0.02)
    assert travel['ground_truth_odom'] == pytest.approx(1.04 * 0.24, abs=0.03)
    assert travel['ground_truth_odom'] > travel['odom']
    assert base['streams']['odom']['travel_m'] == travel['odom']          # unchanged entries
    assert base['summary']['yaw_change_rad'] == {'odom': 0.0, 'ground_truth_odom': 0.0}
    assert report['verdict'] == 'PASS', report['reasons']


def test_yaw_change_by_odometry_is_reported():
    ep = with_drive(*REST_DRIVE)
    pose = ep.streams['odom'].fields['pose'].copy()
    pose[40:, 3:] = synth.qaxis((0, 0, 1), 0.3)
    ep.streams['odom'].fields['pose'] = pose
    summary = run(ep)['base']['summary']
    assert summary['yaw_change_rad']['odom'] == pytest.approx(0.3)
    assert summary['yaw_change_rad']['ground_truth_odom'] == 0.0


def test_drive_time_and_commanded_cycles_from_base_cmd():
    summary = run(with_drive(*REST_DRIVE))['base']['summary']
    assert summary['cmd_nonzero_cycles'] == 40
    assert summary['cmd_cycles'] == 41                 # the 40 commands and the one zero
    assert summary['drive_time_cmd_s'] == pytest.approx(0.8, abs=1e-6)
    assert summary['drive_time_s'] == summary['drive_time_cmd_s']


def test_drive_time_by_base_state_is_preferred_when_recorded():
    ep = with_drive(*REST_DRIVE)
    stamps = T0 + 4.0 + 0.1 * np.arange(11)           # 4.0 .. 5.0 s, 10 Hz
    driving = np.zeros(11, dtype=bool)
    driving[1:9] = True                                # rows 4.1 .. 4.8 s
    zeros = np.zeros((11, 6))
    add(ep, 'base_state', stamps, {'driving': driving, 'command': zeros,
                                   'state': np.where(driving, 2, 1)})
    base = run(ep)['base']
    assert base['summary']['drive_time_state_s'] == pytest.approx(0.8)
    assert base['summary']['drive_time_s'] == base['summary']['drive_time_state_s']
    assert base['streams']['base_state'] == {'kind': 'base_state', 'n': 11, 'driving_rows': 8}


def test_a_stopped_publisher_does_not_count_the_silence_as_drive_time():
    ep = synth.build()
    stamps, twist = cmd_rows(4.1, 4.5, zero_after=False)       # no zero row: the node died
    add(ep, 'base_cmd', stamps, {'twist': twist})
    summary = run(ep)['base']['summary']
    assert summary['cmd_nonzero_cycles'] == 20
    assert summary['drive_time_cmd_s'] == pytest.approx(19 * 0.02)   # the last row holds 0 s


def test_cmd_vel_alone_still_moves_the_base_as_before():
    ep = synth.build()
    stamps, twist = cmd_rows(4.1, 4.5)
    add(ep, 'cmd_vel', stamps, {'twist': twist}, has_header=False)
    report = run(ep)
    assert report['base']['streams']['cmd_vel']['nonzero_rows'] == 20
    assert len(flags(report, 'BASE_MOVED')) == 1


# ---- BASE_DURING_CLUTCH ----

def test_a_drive_wholly_in_a_rest_raises_no_clutch_flag():
    report = run(with_drive(*REST_DRIVE))
    assert flags(report, 'BASE_DURING_CLUTCH') == []
    assert report['base']['clutch_overlap']['by_command'] is None
    assert report['base']['clutch_overlap']['by_odometry'] is None


def test_clutch_while_commanded_raises_the_flag_with_the_time_and_the_cause():
    report = run(with_drive(2.0, 2.5, with_odom=False))
    (flag,) = flags(report, 'BASE_DURING_CLUTCH')
    assert flag['time'] == pytest.approx(T0 + 2.0)
    assert 'commanded' in flag['reason'] and 'base_cmd' in flag['reason']
    assert 'moving by odometry' not in flag['reason']
    assert report['base']['clutch_overlap']['by_command']['first_time'] == pytest.approx(T0 + 2.0)
    assert report['base']['clutch_overlap']['by_command']['streams'] == ['base_cmd']


def test_clutch_while_moving_by_odometry_alone_raises_the_flag():
    # Commanded in the rest, moving by odometry during a stroke.
    ep = with_drive(4.1, 4.9, odom_first=2.0, odom_last=2.5)
    report = run(ep)
    (flag,) = flags(report, 'BASE_DURING_CLUTCH')
    assert 'moving by odometry' in flag['reason'] and 'commanded' not in flag['reason']
    assert flag['time'] == pytest.approx(T0 + 2.0)
    assert report['base']['clutch_overlap']['by_command'] is None
    assert report['base']['clutch_overlap']['by_odometry']['first_time'] == pytest.approx(T0 + 2.0)


def test_a_slow_creep_below_the_thresholds_is_not_moving():
    ep = with_drive(4.1, 4.9, odom_first=2.0, odom_last=2.5)
    ep.streams['odom'].fields['twist'][:, 0] *= 0.02          # 6 mm/s, below 0.01 m/s
    assert flags(run(ep), 'BASE_DURING_CLUTCH') == []
    ep.streams['odom'].fields['twist'][:, 0] = 0.0
    ep.streams['odom'].fields['twist'][40:50, 5] = 0.03       # a turn above 0.02 rad/s
    assert len(flags(run(ep), 'BASE_DURING_CLUTCH')) == 1


def test_a_turn_command_counts_as_a_command():
    ep = synth.build()
    stamps, twist = cmd_rows(2.0, 2.4, vx=0.0, wz=0.4)
    add(ep, 'base_cmd', stamps, {'twist': twist})
    assert len(flags(run(ep), 'BASE_DURING_CLUTCH')) == 1


def test_cmd_vel_counts_as_a_command_too():
    ep = synth.build()
    stamps, twist = cmd_rows(2.0, 2.4)
    add(ep, 'cmd_vel', stamps, {'twist': twist}, has_header=False)
    (flag,) = flags(run(ep), 'BASE_DURING_CLUTCH')
    assert 'cmd_vel' in flag['reason']


def test_the_command_window_decides_a_command_just_before_the_clutch():
    # The clutch is engaged from 1.0 s. A command at 0.97 s is in force until 1.02 s.
    for stamp, raised in ((0.97, True), (0.90, False)):
        ep = synth.build()
        add(ep, 'base_cmd', [T0 + stamp, T0 + stamp + 0.01],
            {'twist': np.array([[0.3, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0]])})
        assert bool(flags(run(ep), 'BASE_DURING_CLUTCH')) is raised, stamp


def test_a_zero_command_during_the_clutch_raises_nothing():
    ep = synth.build()
    stamps = T0 + 2.0 + np.arange(50) / 50.0
    add(ep, 'base_cmd', stamps, {'twist': np.zeros((50, 6))})
    assert flags(run(ep), 'BASE_DURING_CLUTCH') == []


def test_both_causes_are_named_in_one_flag():
    ep = with_drive(2.0, 2.5)
    (flag,) = flags(run(ep), 'BASE_DURING_CLUTCH')
    assert 'commanded' in flag['reason'] and 'moving by odometry' in flag['reason']


def test_without_teleop_state_the_overlap_is_not_judged():
    ep = with_base_state(with_drive(2.0, 2.5))
    del ep.streams['state']
    overlap = checker._base_report(ep, CHECKS, [])['clutch_overlap']
    assert overlap == {'checked': False, 'why': 'no teleop state'}


def test_the_thresholds_are_in_the_tolerance_file_and_move_the_judgement():
    assert CHECKS['flags']['base_clutch_cmd_window_s'] == 0.05
    assert CHECKS['flags']['base_clutch_linear_mps'] == 0.01
    assert CHECKS['flags']['base_clutch_turn_rps'] == 0.02
    checks = copy.deepcopy(CHECKS)
    checks['flags']['base_clutch_cmd_window_s'] = 0.2    # a command at 0.90 s now reaches 1.1
    ep = synth.build()
    add(ep, 'base_cmd', [T0 + 0.90, T0 + 0.91],
        {'twist': np.array([[0.3, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0]])})
    assert flags(run(ep), 'BASE_DURING_CLUTCH') == []
    assert len(flags(run(ep, checks), 'BASE_DURING_CLUTCH')) == 1


def test_the_flag_fails_the_episode_when_the_tolerances_list_it():
    ep = with_drive(2.0, 2.5)
    report = run(ep, failing_checks())
    assert report['verdict'] == 'FAIL'
    assert any(r.startswith('BASE_DURING_CLUTCH') for r in report['reasons'])
    assert run(with_drive(*REST_DRIVE), failing_checks())['verdict'] == 'PASS'


def test_the_checker_does_not_change_an_episode_with_base_streams():
    ep = with_base_state(with_drive(2.0, 2.5))
    before = synth.snapshot(ep)
    run(ep)
    assert synth.same(before, synth.snapshot(ep))


def test_without_the_base_nodes_state_the_overlap_is_not_judged():
    """Base driving off (no base state recorded): another source's base command is not judged."""
    ep = synth.build()
    stamps, twist = cmd_rows(2.0, 2.4)
    add(ep, 'cmd_vel', stamps, {'twist': twist}, has_header=False)
    report = checker.check_episode(ep, CHECKS)
    assert flags(report, 'BASE_DURING_CLUTCH') == []
    overlap = report['base']['clutch_overlap']
    assert overlap == {'checked': False,
                       'why': 'no base state or base command recorded: base driving was off'}


def test_the_flag_fails_an_episode_with_the_shipped_tolerances():
    assert 'BASE_DURING_CLUTCH' in CHECKS['flags']['fail_flags']


def test_a_lost_base_state_stream_does_not_skip_the_check():
    """The base command recorded without the base state: still judged."""
    ep = with_drive(2.0, 2.5)                  # base_cmd during the clutch, no base_state
    report = checker.check_episode(ep, CHECKS)
    assert report['base']['clutch_overlap']['checked']
    assert flags(report, 'BASE_DURING_CLUTCH')
