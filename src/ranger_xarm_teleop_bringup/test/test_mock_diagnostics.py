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
Session recorder, halt table and recovery against the teleop stack in mock execution.

Covers: the recorder never starts or stops while the clutch is engaged; a recorded
singularity halt comes out of the halt table with the measure MoveIt Servo acted on;
re-engaging beyond the hard stop cannot move the arm; ``move_arm.py`` says why a goal
was rejected; ``recover_arm.py`` disarms, waits for MoveIt and moves, and moves nothing
while teleop keeps the arm. One mock stack for the module, on its own ROS domain.
"""

import json
import os
import signal
import time

from moveit_msgs.msg import ServoStatus
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState

from stack_helpers import child_environment, Probe, Process, READY, replay

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, '..', '..', '..', 'tools')
# Wrist 0.05 rad from aligned: MoveIt Servo's measure is 34.6 there, beyond its hard stop.
BEYOND_HARD_STOP = [0.0, -0.6, -0.7, 0.0, 0.05, 0.0]
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(TOOLS, 'record_session.py')),
                                reason='tools/ is not next to the source tree')


def tool(name, *arguments):
    return ['python3', os.path.join(TOOLS, name), *arguments]


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_diagnostics')
    env = child_environment(offset=4)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py'],
                     env, str(logs / 'launch.log'))
    probe = Probe(env)
    try:
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: probe.controllers().get('teleop_arm_controller') == 'inactive',
                       60.0, 'the teleop controller to be loaded inactive')
        probe.move_to(READY)
        yield probe, env, logs, launch
    finally:
        probe.close()
        launch.stop()
        print(launch.output()[-6000:])


def wait_for_text(process, text, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if text in process.output():
            return True
        time.sleep(0.1)
    return False


def start_recorder(env, logs, name, duration=0.0):
    """Start the recorder and wait until it records; return the process and the file."""
    process = Process(tool('record_session.py', 'record', name, '--dir', str(logs),
                           '--duration', str(duration)), env, str(logs / f'{name}.log'))
    assert wait_for_text(process, 'recording to', 30.0), process.output()
    return process, str(logs / f'{name}.jsonl')


def test_recorder_refuses_to_start_without_a_teleop_state(tmp_path):
    env = child_environment(offset=5)       # a domain with nothing on it
    process = Process(tool('record_session.py', 'record', 'nothing', '--dir', str(tmp_path)),
                      env, str(tmp_path / 'recorder.log'))
    try:
        assert process.wait(40.0) == 2, process.output()
        assert 'REFUSED, nothing recorded' in process.output()
        assert not os.path.exists(tmp_path / 'nothing.jsonl')
    finally:
        process.stop()


def test_recorder_neither_starts_nor_stops_while_the_clutch_is_engaged(stack):
    probe, env, logs, _ = stack
    probe.disarm()
    recorder, path = start_recorder(env, logs, 'circle')
    late = None
    player = replay(env, 'clutch_circle', str(logs / 'replay_circle.log'))    # grip held 6 s
    try:
        probe.wait_for(lambda: probe.state().clutch_engaged, 40.0, 'the clutch to engage')
        engaged_at = time.monotonic()
        time.sleep(1.0)
        # A second recorder started now must refuse, and the first asked to stop must go on.
        late = Process(tool('record_session.py', 'record', 'late', '--dir', str(logs)),
                       env, str(logs / 'late.log'))
        os.killpg(recorder.proc.pid, signal.SIGINT)
        assert late.wait(30.0) == 2, late.output()
        assert 'the clutch is engaged' in late.output()
        assert not os.path.exists(logs / 'late.jsonl')
        assert wait_for_text(recorder, 'still recording', 5.0), recorder.output()
        still_engaged = probe.state().clutch_engaged
        assert recorder.proc.poll() is None or not still_engaged
        probe.wait_for(lambda: not probe.state().clutch_engaged, 20.0, 'the release')
        released_at = time.monotonic()
        assert recorder.wait(10.0) == 0, recorder.output()
        assert player.wait(30.0) == 0, player.output()
    finally:
        player.stop()
        recorder.stop()
        if late is not None:
            late.stop()
    with open(path, encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    kinds = {row['k'] for row in rows}
    assert {'description', 'state', 'target', 'input', 'joints', 'servo', 'owner'} <= kinds
    engaged = [row['t'] for row in rows if row['k'] == 'state' and row['engaged']]
    # The file covers the engagement to its end, although the stop was asked for early.
    assert engaged and max(engaged) - min(engaged) > 0.8 * (released_at - engaged_at)
    probe.disarm()


def test_singularity_halt_is_recorded_and_re_engaging_cannot_move_away(stack):
    probe, env, logs, _ = stack
    probe.disarm()
    probe.move_to(BEYOND_HARD_STOP)
    before = probe.joint_positions()
    start = time.monotonic()
    recorder, path = start_recorder(env, logs, 'singularity')
    player = replay(env, 'clutch_singularity', str(logs / 'replay_singularity.log'))
    try:
        assert player.wait(60.0) == 0, player.output()
        time.sleep(1.0)
        os.killpg(recorder.proc.pid, signal.SIGINT)
        assert recorder.wait(15.0) == 0, recorder.output()
    finally:
        player.stop()
        recorder.stop()
    states = probe.states_since(start)
    halted = [s for s in states if s.hold_reason == TeleopState.HOLD_SERVO_HALTED]
    assert halted and all(s.servo_status == ServoStatus.HALT_FOR_SINGULARITY for s in halted)
    # Two engagements (tilt further in, then tilt back out): the arm moved in neither.
    assert probe.max_joint_change(start, before) < 2e-3
    report = Process(tool('record_session.py', 'halts', path, '--json'), env,
                     str(logs / 'halts.json'))
    assert report.wait(60.0) == 0, report.output()
    result = json.loads(report.output())
    report.stop()
    assert len(result['halts']) == 2
    for halt in result['halts']:
        assert halt['servo_code'] == 2 and halt['hold_reason'] == 'SERVO_HALTED'
        assert halt['servo_message'] == 'Very close to a singularity, emergency stop'
        assert 34.0 < halt['condition_number'] < 35.5
        assert halt['resembles'] == 'wrist'
        assert halt['joints'] == pytest.approx(BEYOND_HARD_STOP, abs=2e-3)
        assert halt['nearest_limit']['joint'] == 'xarm_joint3'
    assert result['summary']['by_reason'] == {'HALT_FOR_SINGULARITY / wrist': 2}


def test_move_arm_says_why_the_goal_was_rejected(stack):
    probe, env, logs, _ = stack
    # Still armed from the replay: teleop owns the arm.
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_TELEOP, 5.0, 'teleop owner')
    before = probe.joint_positions()
    start = time.monotonic()
    process = Process(tool('move_arm.py'), env, str(logs / 'move_arm.log'))
    try:
        assert process.wait(60.0) == 1, process.output()
    finally:
        process.stop()
    assert 'goal REJECTED: teleop owns the arm: disarm first' in process.output()
    assert probe.max_joint_change(start, before) < 1e-6


def test_recovery_moves_nothing_while_teleop_keeps_the_arm(stack):
    probe, env, logs, _ = stack
    assert probe.owner_value() == ArmOwner.OWNER_TELEOP and probe.state().armed
    before = probe.joint_positions()
    start = time.monotonic()
    # The disarm request goes nowhere, so teleop stays armed and the owner.
    process = Process(tool('recover_arm.py', '--disarm-service', '/no_such/disarm',
                           '--wait', '4'), env, str(logs / 'recover_refused.log'))
    try:
        assert process.wait(60.0) == 4, process.output()
    finally:
        process.stop()
    assert 'NOT RECOVERED, nothing moved' in process.output()
    assert 'owner: TELEOP' in process.output()
    time.sleep(0.5)
    assert probe.max_joint_change(start, before) < 1e-6
    assert probe.owner_value() == ArmOwner.OWNER_TELEOP and probe.state().armed
    controllers = probe.controllers()
    assert controllers['xarm6_traj_controller'] == 'inactive'
    assert controllers['teleop_arm_controller'] == 'active'


def test_one_command_recovers_from_the_halt_to_the_ready_posture(stack):
    probe, env, logs, _ = stack
    assert probe.owner_value() == ArmOwner.OWNER_TELEOP
    process = Process(tool('recover_arm.py'), env, str(logs / 'recover.log'))
    try:
        assert process.wait(120.0) == 0, process.output()
    finally:
        process.stop()
    output = process.output()
    assert 'disarm: disarmed' in output and 'MoveIt owns the arm' in output
    assert 'recovered: the arm is at' in output
    time.sleep(0.5)
    assert probe.joint_positions() == pytest.approx(READY, abs=5e-3)
    assert probe.owner_value() == ArmOwner.OWNER_MOVEIT and not probe.state().armed
    controllers = probe.controllers()
    assert controllers['xarm6_traj_controller'] == 'active'
    assert controllers['teleop_arm_controller'] == 'inactive'
