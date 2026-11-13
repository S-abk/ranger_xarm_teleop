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
"""The figures of a recorded run: tracking, delay, joint and tool peaks, solver state."""

import math

import numpy as np
import pytest
from ranger_xarm_teleop_servo import tracking

NAMES = [f'xarm_joint{i}' for i in range(1, 7)]
STATE_HZ, JOINT_HZ = 50.0, 100.0


def _quat_z(angle):
    return [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)]


def recording(seconds=3.0, speed=0.1, turn_rate=0.5, solver=True, stamps=True, hold=None,
              joint_q=None, joint_v=None):
    """
    Return the rows of one engagement, 0.5 s to ``seconds - 0.5``, and the quiet around it.

    The tool moves along x at ``speed`` m/s and turns about z at ``turn_rate`` rad/s; the
    target is 20 mm ahead. ``solver`` adds the solver fields of the state rows; ``stamps``
    gives the joint rows header stamps (state and target rows always have them).
    """
    rows = []
    for i in range(int(seconds * STATE_HZ)):
        t = i / STATE_HZ
        engaged = 0.5 <= t < seconds - 0.5
        run = min(max(t - 0.5, 0.0), seconds - 1.0)
        # The measured pose updates every second row (joint states arrive at half the rate
        # the state is published), so poses repeat.
        held = (i // 2) * 2 / STATE_HZ - 0.5
        held = min(max(held, 0.0), seconds - 1.0)
        tool = [[0.3 + speed * held, 0.0, 0.4], _quat_z(turn_rate * held)]
        row = {'t': 100.0 + t, 'k': 'state', 'stamp': 5_000_000_000 + i * 20_000_000,
               'engaged': engaged, 'hold': 0 if engaged else 5, 'servo': 0, 'tool': tool}
        if solver:
            row.update(conditioning=1.0, damped=False, damping=0.0, lead_translation=0.0,
                       lead_rotation=0.0)
        rows.append(row)
        if engaged:
            rows.append({'t': 100.0 + t, 'k': 'target', 'stamp': row['stamp'],
                         'p': [0.32 + speed * run, 0.0, 0.4], 'q': _quat_z(turn_rate * run)})
    for i in range(int(seconds * JOINT_HZ)):
        t = i / JOINT_HZ
        q = joint_q(t) if joint_q else [0.0] * 6
        row = {'t': 100.0 + t, 'k': 'joints', 'q': list(q),
               'v': list(joint_v(t)) if joint_v else None}
        if stamps:
            row['stamp'] = 5_000_000_000 + i * 10_000_000
        rows.append(row)
    return rows


def only(records, **kwargs):
    (run,) = tracking.analyze(records, **kwargs)['engagements']
    return run


def set_state(rows, **fields):
    """Set ``fields`` on the engaged state rows (a callable gets the row's index)."""
    index = 0
    for row in rows:
        if row['k'] == 'state' and row['engaged']:
            for name, value in fields.items():
                row[name] = value(index) if callable(value) else value
            index += 1


def test_peak_joint_speed_from_the_positions_whatever_velocity_is_reported():
    # Mock hardware with a position interface reports zero velocity while it moves.
    reported = recording(joint_q=lambda t: [0.0, 0.4 * t, 0, 0, 0, 0],
                         joint_v=lambda t: [0.0] * 6)
    assert only(reported)['peak_joint_speed'] == pytest.approx(0.4, rel=1e-6)
    # No velocities: the positions' difference over the stamps; joint 2 at 0.4 rad/s.
    differenced = recording(joint_q=lambda t: [0.0, 0.4 * t, 0, 0, 0, 0])
    assert only(differenced)['peak_joint_speed'] == pytest.approx(0.4, rel=1e-6)
    # Without header stamps the recorder's clock is used.
    unstamped = recording(joint_q=lambda t: [0.0, 0.4 * t, 0, 0, 0, 0], stamps=False)
    assert only(unstamped)['peak_joint_speed'] == pytest.approx(0.4, rel=1e-6)


def test_peak_acceleration_raw_and_after_three_sample_average():
    # Constant acceleration of 2 rad/s^2: both agree.
    steady = only(recording(joint_q=lambda t: [0.0, 0.0, t * t, 0, 0, 0]))
    assert steady['peak_joint_acceleration'] == pytest.approx(2.0, rel=1e-6)
    assert steady['peak_joint_acceleration_smoothed'] == pytest.approx(2.0, rel=1e-6)
    # One position glitch of 1 mrad at 100 Hz is [e, -2e, e] / dt^2 in the second difference:
    # raw peak 2e/dt^2 = 20 rad/s^2; the average of any three of those is at most e/(3 dt^2).
    dt = 1.0 / JOINT_HZ
    glitch = only(recording(
        joint_q=lambda t: [0.0, 0.001 if abs(t - 1.5) < dt / 2 else 0.0, 0, 0, 0, 0]))
    assert glitch['peak_joint_acceleration'] == pytest.approx(2 * 0.001 / dt ** 2, rel=1e-6)
    assert glitch['peak_joint_acceleration_smoothed'] == pytest.approx(
        0.001 / (3 * dt ** 2), rel=1e-6)


def test_joint_acceleration_needs_three_samples_and_ignores_repeated_times():
    times = np.array([0.0, 0.01, 0.01, 0.02])
    positions = np.zeros((4, 1))
    assert all(math.isnan(v) for v in tracking.joint_accelerations(times[:2], positions[:2]))
    raw, smoothed = tracking.joint_accelerations(times, positions)    # the repeat is dropped
    assert raw == 0.0 and smoothed == 0.0


def test_peak_tool_speed_between_different_poses_despite_repeated_rows():
    run = only(recording(speed=0.1, turn_rate=0.5))
    assert run['peak_tool_speed'] == pytest.approx(0.1, rel=1e-6)
    assert run['peak_tool_angular_speed'] == pytest.approx(0.5, rel=1e-6)
    # Faster tool: the figure follows.
    assert only(recording(speed=0.4, turn_rate=1.5))['peak_tool_speed'] == pytest.approx(0.4)
    # A tool that never moves has no speed to report beyond zero.
    still = only(recording(speed=0.0, turn_rate=0.0))
    assert math.isnan(still['peak_tool_speed'])


def test_largest_step_of_the_commanded_joint_velocity():
    rows = recording()
    steps = [0.0, 0.0, 0.3, 0.35, 0.35, -0.1, 0.0]
    for k, value in enumerate(steps):
        rows.append({'t': 100.6 + k * 0.02, 'k': 'command', 'stamp': 0,
                     'names': NAMES, 'v': [0.0, value, 0.0, 0.0, 0.05 * value, 0.0]})
    # The release zero comes after the engagement and is not one of its steps.
    rows.append({'t': 102.6, 'k': 'command', 'stamp': 0, 'names': NAMES, 'v': [0.0] * 6})
    assert only(rows)['command_step_max'] == pytest.approx(0.45)    # 0.35 -> -0.1
    assert math.isnan(only(recording())['command_step_max'])


def test_command_names_in_another_order_still_compare_joint_by_joint():
    rows = recording()
    rows.append({'t': 100.6, 'k': 'command', 'stamp': 0, 'names': NAMES, 'v': [0.1] * 6})
    rows.append({'t': 100.62, 'k': 'command', 'stamp': 0, 'names': NAMES[::-1],
                 'v': [0.1] * 6})
    assert only(rows)['command_step_max'] == pytest.approx(0.0)


def test_solver_figures_cover_the_engaged_time_only():
    rows = recording()
    set_state(rows, lead_translation=lambda i: 0.001 * i, lead_rotation=lambda i: 0.002 * i,
              damped=lambda i: i % 4 == 0, conditioning=lambda i: 1.0 - 0.01 * i)
    # Values outside the engagement must not count.
    for row in rows:
        if row['k'] == 'state' and not row['engaged']:
            row.update(lead_translation=9.0, lead_rotation=9.0, damped=True, conditioning=-5.0,
                       servo=2)
    run = only(rows)
    count = sum(1 for r in rows if r['k'] == 'state' and r['engaged'])
    assert run['lead_translation_max'] == pytest.approx(0.001 * (count - 1))
    assert run['lead_rotation_max_deg'] == pytest.approx(math.degrees(0.002 * (count - 1)))
    assert run['damped_share'] == pytest.approx(math.ceil(count / 4) / count)
    assert run['conditioning_min'] == pytest.approx(1.0 - 0.01 * (count - 1))
    assert run['servo_halts'] == 0


def test_servo_halts_count_entries_into_a_halt_status():
    rows = recording()
    codes = [0, 1, 2, 2, 2, 3, 0, 6, 6, 0, 5]
    set_state(rows, servo=lambda i: codes[i] if i < len(codes) else 0)
    assert only(rows)['servo_halts'] == 3            # singularity, joint bound, collision
    set_state(rows, servo=0)
    assert only(rows)['servo_halts'] == 0


def test_a_recording_from_before_the_solver_still_works():
    run = only(recording(solver=False, stamps=False))
    for name in ('lead_translation_max', 'lead_rotation_max_deg', 'damped_share',
                 'conditioning_min', 'command_step_max'):
        assert math.isnan(run[name]), name
    assert run['servo_halts'] == 0                  # old rows carry the servo status
    assert run['peak_tool_speed'] == pytest.approx(0.1, rel=1e-2)
    text = tracking.format_report(tracking.analyze(recording(solver=False, stamps=False)))
    assert 'damped n/a % of the engaged time' in text
    assert 'minimum conditioning n/a' in text
    assert 'commanded joint velocity n/a rad/s per cycle' in text
    # A state row that has no servo field at all reports n/a, not zero.
    rows = recording(solver=False)
    for row in rows:
        row.pop('servo', None)
    assert only(rows)['servo_halts'] is None
    assert 'Servo halt statuses n/a' in tracking.format_report(tracking.analyze(rows))


def test_report_names_every_new_figure():
    rows = recording(joint_q=lambda t: [0.0, 0.0, t * t, 0, 0, 0])
    set_state(rows, lead_translation=0.012, lead_rotation=math.radians(3.0), damped=True,
              conditioning=0.25)
    rows.append({'t': 100.6, 'k': 'command', 'stamp': 0, 'names': NAMES, 'v': [0.0] * 6})
    rows.append({'t': 100.62, 'k': 'command', 'stamp': 0, 'names': NAMES, 'v': [0.1] * 6})
    text = tracking.format_report(tracking.analyze(rows))
    for wanted in ('joints: peak speed ', 'peak acceleration 2.0 rad/s^2 raw',
                   '2.0 rad/s^2 after a 3-sample moving average',
                   'largest step of the commanded joint velocity 0.10 rad/s per cycle',
                   'tool: peak speed 100 mm/s, 28.6 deg/s',
                   'max lead 12.0 mm / 3.0 deg', 'damped 100 % of the engaged time',
                   'minimum conditioning 0.250', 'Servo halt statuses 0'):
        assert wanted in text, wanted
