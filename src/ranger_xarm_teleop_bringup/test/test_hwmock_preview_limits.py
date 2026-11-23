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
The link keep-out at the stage 4 posture, seen before arming and measured while armed.

On the first hardware stage 4 run (link_keepout_margin_m 0.04, the arm at about the tests'
ready posture) every hw_check rule passed and the arm did not track: link 2 against the arm
base, inside the 40 mm margin at that posture, held joints 1 and 2 still and the tool pivoted
about the wrist. At 0.025 the arm tracked. This module shows the same on the hardware-shaped
mock (the bring-up's execution stage, the real description):

- with motion disabled, the preview reports the link keep-out at 0.04 (state and log, the
  pair named) and nothing at 0.025, from the same posture, commanding nothing;
- armed, hw_check's INFO lines read the hold at 0.04 (the keep-out shaping the command in most
  engaged cycles, the pair named, the target's rotation lead rising although the stroke asks
  for no rotation: the tool pivots) and none at 0.025, while the translation lead rule passes
  at both (the gate absorbs the lead the keep-out blocks).

The teleop launch is restarted for each case, at a fifth of the simulation speed caps (the
first-session values); the arm starts at the ready posture. Test values, not guidance.
"""

import os
import re
import time

from ament_index_python.packages import get_package_share_directory
from hw_helpers import (check_report, start_attach, start_check, start_hwmock,
                        wait_for_controllers, write_hardware_limits)
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from stack_helpers import (child_environment, input_player, lead_in_acknowledged, lead_in_file,
                           Probe, READY, replay, start_input)

EXECUTION = {'joint_state_broadcaster': 'active', 'xarm6_traj_controller': 'active',
             'teleop_arm_controller': 'inactive'}
JOINT_LIMITS = os.path.join(get_package_share_directory('ranger_xarm_moveit_config'), 'config',
                            'joint_limits.yaml')
FIFTH = {'linear_speed_cap_m_s': 0.05, 'angular_speed_cap_rad_s': 0.2,
         'joint_speed_cap_rad_s': 0.4, 'joint_acceleration_limit_rad_s2': 2.0}
KEEPOUT_HOLD = 'link keep-out shaping the command (INFO)'
ROTATION_LEAD = 'target lead (rotation, INFO)'
PAIR = 'xarm_link2 - xarm_link_base'


@pytest.fixture(scope='module')
def mock(tmp_path_factory):
    logs = tmp_path_factory.mktemp('hwmock_preview_limits')
    env = child_environment(offset=26)
    bringup = start_hwmock(env, 'execution', logs)
    probe = Probe(env)
    limits = {}
    for margin in (0.025, 0.04):
        folder = logs / f'margin_{margin * 1000.0:.0f}'
        folder.mkdir()
        limits[margin] = write_hardware_limits(folder, link_keepout_margin_m=margin, **FIFTH)
    holder = {'probe': probe, 'env': env, 'logs': logs, 'limits': limits, 'attach': None,
              'config': None, 'starts': 0}
    try:
        wait_for_controllers(probe, EXECUTION, 120.0)
        probe.move_to(READY, 4.0)
        attach(holder, 0.025, enabled=False, first=True)
        start_input(probe, env, logs)
        yield holder
    finally:
        probe.close()
        if holder['attach'] is not None:
            holder['attach'].stop()
            print(holder['attach'].output()[-6000:])
        bringup.stop()
        print(bringup.output()[-3000:])


def attach(holder, margin, enabled, first=False):
    """
    Have the teleop launch run with the margin's limits; wait until it knows the arm.

    Restarted only when it runs with other settings, so each test states what it needs and
    does not depend on the test before it.
    """
    if holder['config'] == (margin, enabled):
        return holder['attach']
    if holder['attach'] is not None:
        holder['attach'].stop()
        holder['attach'] = None
        time.sleep(1.0)
    holder['starts'] += 1
    label = f'{"enabled" if enabled else "disabled"}_{margin * 1000.0:.0f}_{holder["starts"]}'
    extra = ('motion_enabled:=true',) if enabled else ()
    holder['attach'] = start_attach(holder['env'], 'execution', holder['limits'][margin],
                                    holder['logs'], label, *extra)
    probe = holder['probe']
    since = time.monotonic()
    probe.wait_for(lambda: any(s.measured_tool_pose_valid and s.motion_enabled == enabled
                               and abs(s.keepout_margin - margin) < 1e-9
                               for s in probe.states_since(since)), 120.0,
                   f'the teleop node of {label} to know the arm')
    holder['config'] = (margin, enabled)
    if not first:
        reacquire(holder, label)
    return holder['attach']


def at_ready(holder):
    """Plan-free move to the ready posture (the stage 4 posture) through MoveIt's controller."""
    probe = holder['probe']
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 15.0,
                   'MoveIt to own the arm')
    probe.move_to(READY, 4.0)


def reacquire(holder, label):
    """Play the lead-in until the restarted node takes the input again (as start_input)."""
    probe, logs = holder['probe'], holder['logs']
    player = input_player(holder['env'])
    path, _ = lead_in_file(logs)
    since = time.monotonic()
    lead_in = player.play(path, loop=True, source='replay:lead_in',
                          log_path=os.path.join(str(logs), f'lead_in_{label}.log'))
    try:
        probe.wait_for(lambda: lead_in_acknowledged(probe.states_since(since)), 60.0,
                       f'the lead-in to be acknowledged by the node of {label}')
    finally:
        lead_in.stop()
    probe.wait_for(lambda: not 0.0 <= probe.state().input_age <= 0.3, 10.0,
                   'the input to go quiet after the lead-in')


def preview_stroke(holder, label):
    """Play one stroke with motion disabled; return the states that carried a preview."""
    probe, env, logs = holder['probe'], holder['env'], holder['logs']
    start = time.monotonic()
    stroke = replay(env, 'clutch_line', str(logs / f'stroke_{label}.log'), start_delay=1.0)
    try:
        assert stroke.wait(60.0) == 0, stroke.output()
    finally:
        stroke.stop()
    states = probe.states_since(start)
    assert probe.targets_since(start) == []
    assert not any(s.armed for s in states)
    shown = [s for s in states if s.preview_reason == '' and s.hold_reason
             == TeleopState.HOLD_MOTION_DISABLED]
    assert len(shown) > 50, f'{len(shown)} states with a preview'
    return shown


def test_at_25_mm_the_preview_at_the_ready_posture_reports_nothing(mock):
    attach(mock, 0.025, enabled=False)
    at_ready(mock)
    shown = preview_stroke(mock, 'preview_25')
    limits = {s.active_limit for s in shown}
    assert TeleopState.LIMIT_LINK_KEEPOUT not in limits, limits
    assert TeleopState.LIMIT_JOINT_POSITION not in limits, limits


def test_at_40_mm_the_preview_reports_the_link_keepout_and_names_the_pair(mock):
    attach(mock, 0.04, enabled=False)
    at_ready(mock)
    shown = preview_stroke(mock, 'preview_40')
    held = [s for s in shown if s.active_limit == TeleopState.LIMIT_LINK_KEEPOUT]
    # Every direction of joints 1 and 2 is closed at this posture: once the target has moved,
    # the solver's check reports the keep-out in each cycle of the stroke.
    assert len(held) > len(shown) // 2, f'{len(held)} of {len(shown)}'
    output = mock['attach'].output()
    assert f'preview: LINK_KEEPOUT would act (nearest pair {PAIR}, clearance' in output, \
        output[-4000:]
    assert 'margin 40.0 mm); nothing is commanded with motion disabled' in output


def armed_stroke(holder, margin, label):
    """Arm, play one stroke under hw_check teleop, disarm; return the check's verdicts."""
    probe, env, logs = holder['probe'], holder['env'], holder['logs']
    process, data_dir = start_check(env, 'teleop', logs, label, '--seconds', '25',
                                    '--limits', holder['limits'][margin], '--joint-limits',
                                    JOINT_LIMITS)
    try:
        time.sleep(3.0)                 # everything joins before arming
        start = time.monotonic()
        stroke = replay(env, 'clutch_line', str(logs / f'stroke_{label}.log'), start_delay=1.0)
        try:
            assert stroke.wait(60.0) == 0, stroke.output()
        finally:
            stroke.stop()
        assert any(s.armed and s.owner == ArmOwner.OWNER_TELEOP and s.clutch_engaged
                   for s in probe.states_since(start)), 'never engaged with the arm owned'
        probe.disarm()
    finally:
        code, verdicts, report, output = check_report(process, data_dir, 90.0)
    lines = {r['rule']: r for r in report['rules']}
    return verdicts, lines[KEEPOUT_HOLD], lines[ROTATION_LEAD], output


def _rotation_p99(line):
    found = re.search(r'p99 ([0-9.]+) max', line['measured'])
    assert found, line
    return float(found.group(1))


def _share(measured):
    found = re.match(r'(\d+) % of (\d+) engaged cycles', measured)
    assert found, measured
    return int(found.group(1)), int(found.group(2))


def test_at_25_mm_armed_the_check_reads_no_hold(mock):
    attach(mock, 0.025, enabled=True)
    at_ready(mock)
    verdicts, line, rotation, output = armed_stroke(mock, 0.025, 'armed_25')
    assert line['verdict'] == 'INFO' and rotation['verdict'] == 'INFO', output[-4000:]
    share, engaged = _share(line['measured'])
    assert engaged > 50 and share == 0, line
    assert _rotation_p99(rotation) < 1.0, rotation     # the stroke asks for no rotation
    assert verdicts.get('target lead (largest, translation)') == 'PASS', output[-4000:]


def test_at_40_mm_armed_the_check_reads_the_hold_and_names_the_pair(mock):
    """
    Reproduce the first stage 4 run on the arm, on the mock.

    The translation lead rule passes (the gate absorbs the blocked lead), and the INFO lines
    show the hold: the keep-out shaping most engaged cycles, the pair, and the tool turning.
    """
    attach(mock, 0.04, enabled=True)
    at_ready(mock)                  # the posture of the preview test and of the arm's run
    verdicts, line, rotation, output = armed_stroke(mock, 0.04, 'armed_40')
    assert line['verdict'] == 'INFO' and rotation['verdict'] == 'INFO', output[-4000:]
    share, engaged = _share(line['measured'])
    assert engaged > 50 and share > 50, line
    assert PAIR in line['measured'], line
    # The tool pivots about the wrist: the rotation lead runs up toward its clamp (12 deg in
    # these limits) although the stroke asks for no rotation.
    assert _rotation_p99(rotation) > 3.0, rotation
    assert verdicts.get('target lead (largest, translation)') == 'PASS', output[-4000:]
