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
base, inside the 40 mm margin at that posture in the sphere model (the meshes are about 70 mm
apart), held joints 1 and 2 still and the tool pivoted about the wrist. The model now covers
that pair finer. This module shows on the hardware-shaped mock (the bring-up's execution
stage, the real description):

- at the ready posture, at 0.025 and at 0.04: with motion disabled the preview reports no
  keep-out, and armed, hw_check's INFO lines read no hold and no rotation lead;
- where the structure behind the arm is really near (the ready posture with the tool 5 cm
  back, pulled further back), at 0.04: the preview reports the link keep-out (state and log,
  link 2 named against that structure, not the arm base), commanding nothing; armed, the
  check reads the hold and names the pair, while the translation lead rule passes (the gate
  absorbs the lead the keep-out blocks).

The teleop launch is restarted for each case, at a fifth of the simulation speed caps (the
first-session values). Test values, not guidance.
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
MOVING = 'xarm_link2'
BASE = 'xarm_link_base'
# The ready posture with the tool 5 cm back (base frame -x, orientation held): link 2 is 38 mm
# from the structure behind the arm in the model (about 51 mm in the meshes).
PULLED_BACK = [0.0, -0.8, -0.581, 0.0, 1.381, 0.0]


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


def at_ready(holder, posture=READY):
    """Plan-free move to the ready posture (the stage 4 posture) through MoveIt's controller."""
    probe = holder['probe']
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 15.0,
                   'MoveIt to own the arm')
    probe.move_to(posture, 4.0)


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


def preview_stroke(holder, label, fixture='clutch_line'):
    """Play one stroke with motion disabled; return the states that carried a preview."""
    probe, env, logs = holder['probe'], holder['env'], holder['logs']
    start = time.monotonic()
    stroke = replay(env, fixture, str(logs / f'stroke_{label}.log'), start_delay=1.0)
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


@pytest.mark.parametrize('margin', [0.025, 0.04])
def test_the_preview_at_the_ready_posture_reports_nothing(mock, margin):
    attach(mock, margin, enabled=False)
    at_ready(mock)
    shown = preview_stroke(mock, f'preview_{margin * 1000.0:.0f}')
    limits = {s.active_limit for s in shown}
    assert TeleopState.LIMIT_LINK_KEEPOUT not in limits, limits
    assert TeleopState.LIMIT_JOINT_POSITION not in limits, limits


def test_at_40_mm_pulled_back_the_preview_reports_the_keepout_and_names_the_pair(mock):
    attach(mock, 0.04, enabled=False)
    at_ready(mock, PULLED_BACK)
    since = len(mock['attach'].output())
    shown = preview_stroke(mock, 'preview_40_back', 'clutch_keepout_slide')
    held = [s for s in shown if s.active_limit == TeleopState.LIMIT_LINK_KEEPOUT]
    # The target runs back into the structure from a posture already inside the margin.
    assert len(held) > 25, f'{len(held)} of {len(shown)}'
    output = mock['attach'].output()[since:]
    found = re.findall(r'preview: LINK_KEEPOUT would act \(nearest pair (\S+) - (\S+), '
                       r'clearance', output)
    assert found, output[-4000:]
    assert all(moving == MOVING and other != BASE for moving, other in found), found
    assert 'margin 40.0 mm); nothing is commanded with motion disabled' in output


def armed_stroke(holder, margin, label, fixture='clutch_line'):
    """Arm, play one stroke under hw_check teleop, disarm; return the check's verdicts."""
    probe, env, logs = holder['probe'], holder['env'], holder['logs']
    process, data_dir = start_check(env, 'teleop', logs, label, '--seconds', '25',
                                    '--limits', holder['limits'][margin], '--joint-limits',
                                    JOINT_LIMITS)
    try:
        time.sleep(3.0)                 # everything joins before arming
        start = time.monotonic()
        stroke = replay(env, fixture, str(logs / f'stroke_{label}.log'), start_delay=1.0)
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


@pytest.mark.parametrize('margin', [0.025, 0.04])
def test_armed_at_the_ready_posture_the_check_reads_no_hold(mock, margin):
    attach(mock, margin, enabled=True)
    at_ready(mock)
    verdicts, line, rotation, output = armed_stroke(mock, margin,
                                                    f'armed_{margin * 1000.0:.0f}')
    assert line['verdict'] == 'INFO' and rotation['verdict'] == 'INFO', output[-4000:]
    share, engaged = _share(line['measured'])
    assert engaged > 50 and share == 0, line
    assert _rotation_p99(rotation) < 1.0, rotation     # the stroke asks for no rotation
    assert verdicts.get('target lead (largest, translation)') == 'PASS', output[-4000:]


def test_at_40_mm_armed_pulled_back_the_check_reads_the_hold_and_names_the_pair(mock):
    """
    The keep-out still holds where the structure is really near: link 2 against it.

    The translation lead rule passes (the gate absorbs the blocked lead), and the INFO line
    shows the hold and the pair.
    """
    attach(mock, 0.04, enabled=True)
    at_ready(mock, PULLED_BACK)
    verdicts, line, _, output = armed_stroke(mock, 0.04, 'armed_40_back', 'clutch_keepout_slide')
    assert line['verdict'] == 'INFO', output[-4000:]
    share, engaged = _share(line['measured'])
    assert engaged > 50 and share > 10, line
    found = re.search(r'(\S+) - (\S+)', line['measured'].split('nearest pair')[-1])
    assert found and found.group(1) == MOVING and found.group(2) != BASE, line
    assert verdicts.get('target lead (largest, translation)') == 'PASS', output[-4000:]
