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
Procedure stages 3 and 4 on the hardware-shaped mock: the bring-up's execution stage.

The mock (ranger_xarm_teleop_hardware hwmock.launch.py stage:=execution) has the real
execution stage's shape: MoveIt's trajectory controller active, the teleop controller loaded
inactive (as the workspace patch adds it), the joint state broadcaster at 150 Hz with the arm
joints, a gripper action with assumed state publishing its own /joint_states at 10 Hz,
move_group with the real MoveIt controller file, no simulation clock. The tests run in file
order, as the procedure does:

* the preflight refuses a missing and an active teleop controller (each with what to do);
* attached with motion disabled: no switch to teleop, nothing on the teleop controller;
* attached with motion enabled: a planned motion with the motion check, the switch to teleop
  and back with the arm standing still, one short armed stroke at low limits with the teleop
  check, and the gripper proxy's partial joint states never taking the arm's pose away;
* G-17: the arm driver's reaction to an arm error replayed (deactivate everything, then ask
  for everything, best effort), and each outcome the controller manager can produce;
* move_group killed during a planned motion, recorded for the procedure;
* the camera check against a stand-in camera.

Nothing here starts a robot.
"""

import math
import os
import threading
import time

from ament_index_python.packages import get_package_share_directory
from controller_manager_msgs.srv import ListControllers, SwitchController
from hw_helpers import (check_report, publishers_of, start_attach, start_check, start_hwmock,
                        wait_for_controllers, write_hardware_limits)
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from ranger_xarm_teleop_interfaces.srv import SetArmOwner
from sensor_msgs.msg import CameraInfo, Image
from stack_helpers import (arm_by_replay, child_environment, node_pids, prime, Probe, Process,
                           READY, replay, start_input)
import yaml

MOVEIT, TELEOP, JSB = 'xarm6_traj_controller', 'teleop_arm_controller', 'joint_state_broadcaster'
EXECUTION = {JSB: 'active', MOVEIT: 'active', TELEOP: 'inactive'}
TELEOP_COMMAND = f'/{TELEOP}/joint_trajectory'
ELSEWHERE = [0.3, -0.4, -0.9, 0.2, 1.1, -0.3]
JOINT_LIMITS = os.path.join(get_package_share_directory('ranger_xarm_moveit_config'), 'config',
                            'joint_limits.yaml')


def control(env, logs, label, *args, timeout=60.0):
    """Run a ``ros2 control`` command on the mock (a test's own graph); return its output."""
    process = Process(['ros2', 'control', *args], env, os.path.join(str(logs),
                                                                    f'control_{label}.log'))
    try:
        code = process.wait(timeout)
        output = process.output()
        assert code == 0, f'ros2 control {" ".join(args)}: exit {code}\n{output[-2000:]}'
        return output
    finally:
        process.stop()


def switch(probe, activate, deactivate, strictness):
    """Ask the controller manager for one switch, as the arm driver does; return the answer."""
    client = getattr(probe, '_switch_client', None)
    if client is None:
        client = probe.node.create_client(SwitchController,
                                          '/controller_manager/switch_controller')
        probe._switch_client = client
    request = SwitchController.Request()
    request.activate_controllers = list(activate)
    request.deactivate_controllers = list(deactivate)
    request.strictness = strictness
    request.timeout.sec = 2
    return probe.call(client, request, 20.0)


@pytest.fixture(scope='module')
def stage3(tmp_path_factory):
    logs = tmp_path_factory.mktemp('hwmock_execution')
    env = child_environment(offset=20)
    bringup = start_hwmock(env, 'execution', logs)
    probe = Probe(env)
    holder = {'probe': probe, 'env': env, 'logs': logs, 'bringup': bringup, 'attach': None,
              'limits': write_hardware_limits(logs)}
    try:
        wait_for_controllers(probe, EXECUTION, 120.0)
        # The mock starts with every joint at zero (the wrist aligned); a first session starts
        # with a planned motion to the ready posture, and so does this.
        probe.move_to(READY, 4.0)
        yield holder
    finally:
        probe.close()
        if holder['attach'] is not None:
            holder['attach'].stop()
            print(holder['attach'].output()[-6000:])
        bringup.stop()
        print(bringup.output()[-3000:])


def attach(holder, label, *extra):
    """(Re)start the hardware teleop launch on the mock; wait for its first state."""
    if holder['attach'] is not None:
        holder['attach'].stop()
        holder['attach'] = None
        time.sleep(1.0)
    holder['attach'] = start_attach(holder['env'], 'execution', holder['limits'], holder['logs'],
                                    label, *extra)
    probe = holder['probe']
    since = time.monotonic()
    probe.wait_for(lambda: any(True for _ in probe.states_since(since)), 120.0,
                   f'the teleop state ({label})')
    return holder['attach']


def refused(holder, label):
    """Start the attach once more and return its output; it must refuse and start nothing."""
    process = start_attach(holder['env'], 'execution', holder['limits'], holder['logs'], label,
                           'preflight_wait:=5.0')
    try:
        code = process.wait(90.0)
        output = process.output()
    finally:
        process.stop()
    assert code is not None, output[-3000:]
    assert 'ATTACH REFUSED' in output, output[-3000:]
    assert 'teleop_node' not in output.split('ATTACH REFUSED')[-1], output[-3000:]
    return output


# ---- the preflight (INV-4, INV-5) ----

def test_the_preflight_refuses_a_missing_teleop_controller(stage3):
    probe, env, logs = stage3['probe'], stage3['env'], stage3['logs']
    control(env, logs, 'unload', 'unload_controller', TELEOP)
    try:
        output = refused(stage3, 'missing')
        line = next(x for x in output.splitlines() if 'ATTACH REFUSED' in x and TELEOP in x)
        assert 'not loaded in the bring-up: apply the workspace patch' in line
    finally:
        control(env, logs, 'reload', 'load_controller', '--set-state', 'inactive', TELEOP)
    wait_for_controllers(probe, EXECUTION)


def test_the_preflight_refuses_an_active_teleop_controller(stage3):
    probe, env, logs = stage3['probe'], stage3['env'], stage3['logs']
    control(env, logs, 'activate', 'switch_controllers', '--activate', TELEOP, '--deactivate',
            MOVEIT, '--strict')
    try:
        output = refused(stage3, 'active')
        assert any(f'{TELEOP} is active: only the command mux may activate it' in x
                   for x in output.splitlines()), output[-3000:]
    finally:
        control(env, logs, 'restore', 'switch_controllers', '--activate', MOVEIT,
                '--deactivate', TELEOP, '--strict')
    wait_for_controllers(probe, EXECUTION)


# ---- stage 3 begins with motion disabled ----

def test_motion_disabled_never_switches_to_teleop(stage3):
    probe = stage3['probe']
    attach(stage3, 'disabled')
    assert 'ATTACH PREFLIGHT OK (execution stage)' in stage3['attach'].output()
    assert [n for n in publishers_of(probe, TELEOP_COMMAND) if n != 'teleop_test_probe'] == []
    prime(probe.set_owner_client, SetArmOwner.Request(owner=ArmOwner.OWNER_TELEOP), each=6.0)
    before = probe.joint_positions()
    start = time.monotonic()
    response = probe.set_owner(ArmOwner.OWNER_TELEOP)
    assert not response.success and 'teleop motion is disabled' in response.detail
    assert probe.controllers() == EXECUTION
    assert probe.max_joint_change(start, before) < 1e-6
    assert not probe.state().motion_enabled


# ---- stage 3 with motion enabled: a planned motion and the switch, nothing armed ----

def test_a_planned_motion_passes_the_motion_check(stage3):
    probe, env, logs = stage3['probe'], stage3['env'], stage3['logs']
    attach(stage3, 'enabled', 'motion_enabled:=true')
    start_input(probe, env, logs)
    process, data_dir = start_check(env, 'motion', logs, 'motion', '--seconds', '8',
                                    '--joint-limits', JOINT_LIMITS)
    try:
        time.sleep(2.0)
        probe.move_to(ELSEWHERE, 3.0)
        probe.move_to(READY, 3.0)
    finally:
        code, verdicts, report, output = check_report(process, data_dir, 60.0)
    assert verdicts.get('stale repeats while moving') == 'PASS', output[-4000:]
    assert verdicts.get('joint_states rate') == 'PASS', output[-4000:]
    assert verdicts.get('wrist oscillation') == 'PASS', output[-4000:]
    # Mock execution follows the plan exactly: the acceleration is judged, whatever it is.
    assert verdicts.get('peak joint acceleration') in ('PASS', 'FAIL'), output[-4000:]


def test_the_switch_to_teleop_and_back_moves_nothing(stage3):
    probe = stage3['probe']
    assert [n for n in publishers_of(probe, TELEOP_COMMAND)
            if n != 'teleop_test_probe'] == ['servo_node']
    prime(probe.set_owner_client, SetArmOwner.Request(owner=ArmOwner.OWNER_MOVEIT), each=6.0)
    before = probe.joint_positions()
    start = time.monotonic()
    to_teleop = probe.set_owner(ArmOwner.OWNER_TELEOP)
    assert to_teleop.success, to_teleop.detail
    time.sleep(1.0)
    back = probe.set_owner(ArmOwner.OWNER_MOVEIT)
    assert back.success, back.detail
    assert probe.controllers() == EXECUTION
    assert probe.max_joint_change(start, before) < 1e-6
    assert not any(s.armed for s in probe.states_since(start))


# ---- stage 4: armed at low limits, one short stroke ----

def test_one_short_armed_stroke_with_the_teleop_check(stage3):
    probe, env, logs = stage3['probe'], stage3['env'], stage3['logs']
    process, data_dir = start_check(env, 'teleop', logs, 'teleop', '--seconds', '25',
                                    '--limits', stage3['limits'], '--joint-limits',
                                    JOINT_LIMITS)
    try:
        time.sleep(3.0)                 # everything joins before arming
        assert not probe.state().armed
        before = probe.tool_position()
        start = time.monotonic()
        # The recording arms (B), engages and draws one short line, as stage 4's first stroke.
        stroke = replay(env, 'clutch_line', str(logs / 'stroke.log'), start_delay=1.0)
        try:
            assert stroke.wait(60.0) == 0, stroke.output()
        finally:
            stroke.stop()
        assert any(s.armed and s.owner == ArmOwner.OWNER_TELEOP
                   for s in probe.states_since(start)), 'never armed and owning the arm'
        assert len(probe.targets_since(start)) > 20
        after = probe.tool_position()
        assert math.dist(before, after) > 0.01
        probe.disarm()
    finally:
        code, verdicts, report, output = check_report(process, data_dir, 90.0)
    assert verdicts.get('stale repeats while moving') == 'PASS', output[-4000:]
    assert verdicts.get('target lead (largest, translation)') == 'PASS', output[-4000:]
    # G-18: the stage-4 stop rules are on MoveIt Servo's output against the limits file; the
    # joint-state acceleration is information only (a 20 ms window, MoveIt's value shown).
    assert verdicts.get('commanded joint acceleration (MoveIt Servo output)') == 'PASS', \
        output[-4000:]
    assert verdicts.get('stop time after a release') in ('PASS', 'NOT JUDGED'), output[-4000:]
    assert verdicts.get('joint-state acceleration (20 ms, INFO)') == 'INFO', output[-4000:]
    assert 'peak joint acceleration' not in verdicts
    # The gripper proxy's own joint states (10 Hz, gripper only) never took the arm's
    # measured pose away while engaged.
    engaged = [s for s in probe.states_since(start) if s.clutch_engaged]
    assert engaged and all(s.measured_tool_pose_valid for s in engaged)
    assert not any(s.hold_reason == TeleopState.HOLD_TOOL_POSE_STALE for s in engaged)


def test_a_stroke_at_a_fifth_of_the_caps_passes_the_acceleration_and_stop_rules(stage3):
    """
    G-18 at the first-session speed caps: hw_check's stage-4 stop rules pass.

    The limits are a fifth of the simulation speed caps and a hardware-like joint speed cap
    and acceleration limit (0.4 rad/s, 2 rad/s^2: a 0.2 s stop), test values only. The
    stroke releases while moving, so a stop is measured.
    """
    probe, env, logs = stage3['probe'], stage3['env'], stage3['logs']
    original = stage3['limits']
    fifth = logs / 'fifth'
    fifth.mkdir(exist_ok=True)
    stage3['limits'] = write_hardware_limits(
        fifth, linear_speed_cap_m_s=0.05, angular_speed_cap_rad_s=0.2,
        joint_speed_cap_rad_s=0.4, joint_acceleration_limit_rad_s2=2.0)
    try:
        attach(stage3, 'fifth', 'motion_enabled:=true')
        probe.disarm()
        probe.move_to(READY, 4.0)
        process, data_dir = start_check(env, 'teleop', logs, 'teleop_fifth', '--seconds', '25',
                                        '--limits', stage3['limits'], '--joint-limits',
                                        JOINT_LIMITS)
        try:
            time.sleep(3.0)                 # everything joins before arming
            start = time.monotonic()
            stroke = replay(env, 'clutch_release_moving', str(logs / 'stroke_fifth.log'),
                            start_delay=1.0)
            try:
                assert stroke.wait(60.0) == 0, stroke.output()
            finally:
                stroke.stop()
            assert any(s.armed and s.owner == ArmOwner.OWNER_TELEOP
                       for s in probe.states_since(start)), 'never armed and owning the arm'
            assert any(s.stopping for s in probe.states_since(start)), 'no stop ramp seen'
            probe.disarm()
        finally:
            code, verdicts, report, output = check_report(process, data_dir, 90.0)
        assert verdicts.get('commanded joint acceleration (MoveIt Servo output)') == 'PASS', \
            output[-4000:]
        assert verdicts.get('stop time after a release') == 'PASS', output[-4000:]
        assert verdicts.get('joint-state acceleration (20 ms, INFO)') == 'INFO', output[-4000:]
    finally:
        stage3['limits'] = original
        attach(stage3, 'enabled_again', 'motion_enabled:=true')


# ---- G-17: the arm driver after an arm error ----

def arm_and_own(probe, env, logs, label):
    arm_by_replay(probe, env, str(logs / label), disarm=False)
    probe.wait_for(lambda: probe.state().armed and probe.owner_value() == ArmOwner.OWNER_TELEOP,
                   15.0, 'armed and owning the arm')


def driver_deactivates_everything(probe):
    """As uf_robot_system_hardware does on an arm error: every listed controller, best effort."""
    names = list(probe.controllers())
    return switch(probe, [], names, SwitchController.Request.BEST_EFFORT), names


def assert_disarmed_still_and_frozen(probe, since, before, what):
    states = probe.states_since(since)
    assert states and not states[-1].armed, f'{what}: still armed'
    assert probe.targets_since(since + 0.5) == [], f'{what}: targets after the event'
    assert all(not s.gripper_active for s in states), f'{what}: gripper commanded'
    assert probe.max_joint_change(since, before) < 1e-4, f'{what}: the arm moved'


def test_g17_the_drivers_full_reactivation_is_refused_and_teleop_disarms(stage3):
    """Outcome "neither": the controller manager refuses the driver's request as a whole."""
    probe, env, logs = stage3['probe'], stage3['env'], stage3['logs']
    arm_and_own(probe, env, logs, 'g17_neither')
    before = probe.joint_positions()
    since = time.monotonic()
    answer, names = driver_deactivates_everything(probe)
    assert answer.ok, answer.message
    probe.wait_for(lambda: not probe.state().armed, 5.0, 'teleop to disarm (G-17)')
    # The joint state broadcaster went too: no joint states during the gap, and the gate holds.
    gap = time.monotonic()
    time.sleep(1.5)
    assert probe.joints_since(gap + 0.2) == []
    assert all(s.hold_reason != TeleopState.HOLD_NONE for s in probe.states_since(gap))
    # The driver then asks for every controller it listed, best effort.
    refusal = switch(probe, names, [], SwitchController.Request.BEST_EFFORT)
    assert not refusal.ok
    print(f"G-17 neither: the driver's reactivation of {names} was refused: "
          f'{refusal.message!r}')
    states = probe.controllers()
    assert all(v != 'active' for v in states.values()), states
    assert_disarmed_still_and_frozen(probe, since, before, 'neither')
    time.sleep(6.0)                 # longer than teleop's request window: it never asks again
    assert not probe.state().armed and probe.controllers()[TELEOP] != 'active'
    # Back to the stage's state for the next case (a person restarts; the test restores).
    answer = switch(probe, [JSB, MOVEIT], [], SwitchController.Request.STRICT)
    assert answer.ok, answer.message
    wait_for_controllers(probe, EXECUTION)


def test_g17_moveit_wins_and_teleop_stays_disarmed(stage3):
    probe, env, logs = stage3['probe'], stage3['env'], stage3['logs']
    arm_and_own(probe, env, logs, 'g17_moveit')
    before = probe.joint_positions()
    since = time.monotonic()
    answer, _ = driver_deactivates_everything(probe)
    assert answer.ok
    probe.wait_for(lambda: not probe.state().armed, 5.0, 'teleop to disarm (G-17)')
    assert switch(probe, [JSB, MOVEIT], [], SwitchController.Request.BEST_EFFORT).ok
    probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 10.0, 'MoveIt owner')
    time.sleep(6.0)
    assert probe.controllers() == EXECUTION
    assert_disarmed_still_and_frozen(probe, since, before, 'MoveIt wins')


def test_g17_teleop_wins_and_the_mux_deactivates_it(stage3):
    probe, env, logs = stage3['probe'], stage3['env'], stage3['logs']
    arm_and_own(probe, env, logs, 'g17_teleop')
    before = probe.joint_positions()
    since = time.monotonic()
    answer, _ = driver_deactivates_everything(probe)
    assert answer.ok
    probe.wait_for(lambda: not probe.state().armed, 5.0, 'teleop to disarm (G-17)')
    assert switch(probe, [JSB, TELEOP], [], SwitchController.Request.BEST_EFFORT).ok
    probe.wait_for(lambda: probe.controllers().get(TELEOP) == 'inactive', 15.0,
                   'the mux to deactivate the teleop controller')
    states = probe.controllers()
    assert states[MOVEIT] == 'inactive', 'the mux activated something in response'
    assert_disarmed_still_and_frozen(probe, since, before, 'teleop wins')
    assert 'without a switch by the mux' in stage3['attach'].output()
    # The person restores MoveIt through the mux (procedure).
    prime(probe.list_client, ListControllers.Request())
    restored = probe.set_owner(ArmOwner.OWNER_MOVEIT)
    assert restored.success, restored.detail
    wait_for_controllers(probe, EXECUTION)


# ---- a planned motion outlives a crashed move_group ----

def test_a_planned_motion_when_move_group_dies(stage3):
    """Recorded for the procedure: what the trajectory controller does without move_group."""
    probe, env = stage3['probe'], stage3['env']
    pids = node_pids('move_group', environ=f'ROS_HOME={env["ROS_HOME"]}')
    assert pids, 'no move_group process of this test'
    handle = probe.send_goal(ELSEWHERE, 4.0)
    assert handle is not None and handle.accepted
    time.sleep(1.0)
    for pid in pids:
        os.kill(pid, 9)
    start = time.monotonic()
    time.sleep(5.0)
    q = probe.joint_positions()
    reached = max(abs(a - b) for a, b in zip(q, ELSEWHERE)) < 0.01
    print(f'move_group killed 1.0 s into a 4.0 s goal: the arm '
          f'{"finished the trajectory" if reached else "stopped short"}; joints {q}')
    # The goal was sent to the controller directly here; through move_group the controller
    # holds the same trajectory. Either way the controller, not move_group, runs it.
    assert reached
    assert probe.joints_since(start)


# ---- stage 5: the camera check ----

def test_the_camera_check_judges_a_stand_in_camera(stage3):
    probe, env, logs = stage3['probe'], stage3['env'], stage3['logs']
    profile = os.path.join(get_package_share_directory('ranger_xarm_teleop_record'),
                           'profiles', 'hardware.yaml')
    with open(profile, encoding='utf-8') as handle:
        streams = yaml.safe_load(handle)['streams']
    images = {k: v for k, v in streams.items() if v.get('kind') == 'image'}
    pubs = {k: probe.node.create_publisher(Image, v['topic'], 10) for k, v in images.items()}
    infos = {k: probe.node.create_publisher(CameraInfo, v['topic'], 10)
             for k, v in streams.items() if v.get('kind') == 'camera_info'}
    stop = threading.Event()

    def run():
        period = 1.0 / 30.0
        next_at = time.monotonic()
        while not stop.is_set():
            stamp = probe.node.get_clock().now().to_msg()
            for key, pub in pubs.items():
                spec = images[key].get('image', {})
                msg = Image(width=spec.get('width', 640), height=spec.get('height', 480),
                            encoding=spec.get('encoding', 'rgb8'))
                msg.header.stamp = stamp
                msg.step = msg.width * 3
                msg.data = bytes(16)
                pub.publish(msg)
            for pub in infos.values():
                info = CameraInfo()
                info.header.stamp = stamp
                pub.publish(info)
            next_at += period
            time.sleep(max(0.0, next_at - time.monotonic()))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        process, data_dir = start_check(env, 'cameras', logs, 'cameras', '--seconds', '8',
                                        '--profile', profile)
        code, verdicts, report, output = check_report(process, data_dir, 60.0)
    finally:
        stop.set()
        thread.join(2.0)
    rates = [r for r in report['rules'] if 'rate' in r['rule'] and 'camera' in r['rule'].lower()
             or r['rule'].startswith('/camera')]
    assert rates, output[-3000:]
    assert all(r['verdict'] in ('PASS', 'FAIL') for r in rates), rates
    assert code in (0, 1), output[-3000:]
