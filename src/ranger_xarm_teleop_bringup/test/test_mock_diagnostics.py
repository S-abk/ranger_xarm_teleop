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

Covers: the recorder never starts or stops while the clutch is engaged; a session that
used to halt for a singularity (pose commands) is recorded without any halt and its halt
table is empty; ``move_arm.py`` says why a goal was rejected; ``recover_arm.py`` disarms,
waits for MoveIt and moves, and moves nothing while teleop keeps the arm. One mock stack
for the module, on its own ROS domain.
"""

import json
import os
import signal
import time

from moveit_msgs.msg import ServoStatus
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from rclpy.qos import QoSProfile, ReliabilityPolicy

from stack_helpers import (assert_stack_nodes, child_environment, JOIN_LOOK_AHEAD_S,
                           join_stall, keep_record, MOCK_STACK_NODES, pauses_around, Probe,
                           Process, READY, replay, start_input)
from trajectory_msgs.msg import JointTrajectory

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, '..', '..', '..', 'tools')
# Wrist 0.05 rad from aligned: MoveIt Servo's measure is 34.6 there, beyond its hard stop;
# with pose commands every engagement from here halted at once.
BEYOND_HARD_STOP = [0.0, -0.6, -0.7, 0.0, 0.05, 0.0]
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(TOOLS, 'record_session.py')),
                                reason='tools/ is not next to the source tree')


def tool(name, *arguments):
    return ['python3', os.path.join(TOOLS, name), *arguments]


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_diagnostics')
    env = child_environment(offset=4)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                      'rviz:=false'],
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
        start_input(probe, env, logs)
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


def wait_for_any(process, texts, timeout):
    """Return the first of ``texts`` to appear in the process's output, or None."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        output = process.output()
        found = [text for text in texts if text in output]
        if found:
            return min(found, key=output.index)
        time.sleep(0.05)
    return None


REFUSED = 'REFUSED, nothing recorded: the clutch is engaged'
STARTED = 'recording to'


def test_recorder_neither_starts_nor_stops_while_the_clutch_is_engaged(stack):
    """
    The recorder never starts and never stops a recording while the clutch is engaged.

    One second into a six second engagement a second recorder is started and the first is
    asked to stop. Asserted, as the tool's own doing:

    * the second one refuses with its reason and writes nothing, as long as the clutch is
      engaged when it looks;
    * the first one goes on, and its file covers the engagement to its end;
    * neither leaves the graph before the clutch is released.

    Not asserted: that the engagement outlasts the second process joining the graph. A
    process that joins can stall an existing node for about a third of a second, and the gate
    then releases the clutch on stale data, as it must (interface contract: every process of
    a session starts before arming). The test prints which of the two happened, and a join
    stall is recorded with the test's logs where the logs of failed tests are kept
    (``stack_helpers.keep_record``); a run without such a record outlasted the join. A
    release with any other cause fails the test (``stack_helpers.join_stall``).
    """
    probe, env, logs, _ = stack
    probe.disarm()
    servo_output = []           # when MoveIt Servo's output arrived, a fourth stream
    # A second of it is kept, so that a pause of this process loses none of it.
    watch = probe.node.create_subscription(
        JointTrajectory, '/teleop_arm_controller/joint_trajectory',
        lambda msg: servo_output.append(time.monotonic()),
        QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))
    recorder, path = start_recorder(env, logs, 'circle')
    late = None
    player = replay(env, 'clutch_circle', str(logs / 'replay_circle.log'))    # grip held 6 s
    try:
        probe.wait_for(lambda: probe.state().clutch_engaged, 40.0, 'the clutch to engage')
        engaged_at = time.monotonic()
        time.sleep(1.0)
        # A second recorder is started now, and the first is asked to stop.
        joined_at = time.monotonic()
        late = Process(tool('record_session.py', 'record', 'late', '--dir', str(logs)),
                       env, str(logs / 'late.log'))
        os.killpg(recorder.proc.pid, signal.SIGINT)
        # While the clutch is engaged neither may leave. When the second one gives its
        # verdict is noted, to compare with the release.
        left, verdict_at = [], None
        end = time.monotonic() + 20.0
        while probe.state().clutch_engaged and time.monotonic() < end:
            if verdict_at is None and any(t in late.output() for t in (REFUSED, STARTED)):
                verdict_at = time.monotonic()
            for name, process in (('the first recorder', recorder), ('the second', late)):
                if process.proc.poll() is not None and name not in left:
                    time.sleep(0.2)             # the probe may be a state behind the tool
                    if probe.state().clutch_engaged:
                        left.append(name)
            time.sleep(0.02)
        assert not probe.state().clutch_engaged, 'the clutch was still engaged after 20 s'
        assert not left, f'{" and ".join(left)} left while the clutch was engaged'
        verdict = wait_for_any(late, (REFUSED, STARTED), 20.0)
        if verdict_at is None:
            verdict_at = time.monotonic()
        # The recording to its end and a while more: a pause that released the clutch can
        # outlast the release, and is looked for on both sides of it.
        assert player.wait(30.0) == 0, player.output()
        time.sleep(JOIN_LOOK_AHEAD_S)

        # How the engagement ended. The grip is held for 6.0 s in the fixture.
        during = probe.timed_states_since(engaged_at - 0.5)
        held = [when for when, s in during if s.clutch_engaged]
        released_at, release = next((when, s) for when, s in during if when > max(held))
        lasted = max(held) - min(held)
        outlasted = lasted > 5.7 and release.hold_reason == TeleopState.HOLD_CLUTCH_RELEASED
        if outlasted:
            print(f'the engagement outlasted the joining recorder ({lasted:.2f} s)')
        else:
            stalled, how = join_stall(joined_at, released_at, release.hold_reason,
                                      pauses_around(probe, released_at, servo_output))
            assert stalled, (
                f'the clutch was released after {lasted:.2f} s with hold_reason '
                f'{release.hold_reason}, and not by the stall of a joining process: {how}')
            record = (f'JOIN STALL: the clutch was released after {lasted:.2f} s with '
                      f'hold_reason {release.hold_reason}: {how}')
            print(record)
            print('recorded in ' + (keep_record(record, 'join_stall') or 'this output only'))

        # The second recorder.
        assert verdict is not None, late.output()
        if verdict == REFUSED:
            # It looked while the clutch was engaged: refused with its reason, stayed on (it
            # was not among those that left), wrote nothing.
            assert 'staying on until the grip is released' in late.output()
            assert late.wait(10.0) == 2, late.output()
            assert not os.path.exists(logs / 'late.jsonl')
            print('the second recorder was refused while the clutch was engaged')
        else:
            # It may record only if the clutch was released when it looked, which here only
            # a join stall can have done; never while the clutch was engaged.
            assert not outlasted, ('the second recorder started during the engagement: '
                                   + late.output())
            assert verdict_at >= released_at - 0.2, (
                f'the second recorder started {released_at - verdict_at:.2f} s before the '
                f'clutch was released: {late.output()}')
            os.killpg(late.proc.pid, signal.SIGINT)
            assert late.wait(15.0) == 0, late.output()
            # By its own file too: the first state it recorded shows a released clutch.
            with open(logs / 'late.jsonl', encoding='utf-8') as handle:
                recorded = [json.loads(line) for line in handle]
            first = next((row for row in recorded if row['k'] == 'state'), None)
            assert first is not None, 'the second recorder recorded no teleop state'
            assert not first['engaged'], 'the second recorder started while the clutch was engaged'
            print('the second recorder started after the join stall had released the clutch')

        # The first recorder: asked to stop a second into the engagement, it went on to the
        # release (it was not among those that left) and stopped then.
        assert 'still recording' in recorder.output(), recorder.output()
        assert recorder.wait(10.0) == 0, recorder.output()
    finally:
        probe.node.destroy_subscription(watch)
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
    assert engaged and max(engaged) - min(engaged) > 0.8 * lasted
    probe.disarm()


def test_session_that_used_to_halt_is_recorded_without_a_halt(stack):
    """Replaces the pose-mode test (two singularity halts in the table, the arm still)."""
    probe, env, logs, _ = stack
    probe.disarm()
    probe.move_to(BEYOND_HARD_STOP)
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
    assert any(s.clutch_engaged for s in states)
    assert not any(s.hold_reason == TeleopState.HOLD_SERVO_HALTED for s in states)
    assert not any(s.servo_status == ServoStatus.HALT_FOR_SINGULARITY for s in states)
    assert probe.max_joint_change(start, BEYOND_HARD_STOP) > 0.2      # it moved
    report = Process(tool('record_session.py', 'halts', path, '--json'), env,
                     str(logs / 'halts.json'))
    assert report.wait(60.0) == 0, report.output()
    result = json.loads(report.output())
    report.stop()
    assert result['halts'] == [] and result['summary']['by_reason'] == {}
    assert result['summary']['engaged_s'] > 2.0


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
    assert 'NOT RECOVERED, no motion goal sent' in process.output()
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


def test_no_other_stack_or_test_shares_this_domain(stack):
    """Isolation: the domain holds this stack's nodes and the probe, nothing else."""
    assert_stack_nodes(stack[0], stack[3], MOCK_STACK_NODES)
