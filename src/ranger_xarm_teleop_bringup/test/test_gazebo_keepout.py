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
Link keep-out (G-15) and the retreat in headless Gazebo (opt-in: ``TELEOP_TEST_GAZEBO=1``).

The same checks as ``test_mock_keepout.py``, with the simulated controller and physics.
"""

import os
import time

from moveit_msgs.msg import ServoStatus
import numpy as np
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from stack_helpers import (child_environment, kill_gz_partition, Probe, Process, READY,
                           replay_with_clean_input, start_input)
from test_mock_keepout import (_arm_teleop, _contact_paths, _drive_into_contact,  # noqa
                               _extra, _recover, _tool, _valid, dls, ko, MARGIN,
                               SERVO_HALT_CODES, TOOLS)

pytestmark = pytest.mark.skipif(os.environ.get('TELEOP_TEST_GAZEBO') != '1',
                                reason='set TELEOP_TEST_GAZEBO=1 to run the Gazebo tests')

MOVEIT, TELEOP = 'xarm_xarm6_traj_controller', 'teleop_arm_controller'


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('gazebo_keepout')
    env = child_environment(offset=8)
    env['GZ_PARTITION'] = f'teleop_keepout_{os.getpid()}'
    env['GZ_IP'] = '127.0.0.1'
    sim = Process(['ros2', 'launch', 'ranger_xarm_gazebo', 'sim.launch.py', 'headless:=true'],
                  env, str(logs / 'sim.log'))
    probe = Probe(env, moveit_controller=MOVEIT)
    teleop = None
    try:
        probe.wait_for(lambda: probe.joint_positions() is not None, 180.0, 'joint states')
        probe.wait_for(lambda: probe.controllers().get(MOVEIT) == 'active', 120.0,
                       'the arm controller')
        teleop = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup',
                          'teleop_sim.launch.py'], env, str(logs / 'teleop.log'))
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 120.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: probe.description is not None, 30.0, 'the robot description')
        time.sleep(5.0)             # MoveIt Servo start-up and verification
        probe.move_to(READY, seconds=4.0)
        from ament_index_python.packages import get_package_share_directory
        model = ko.build(probe.description, ko.make_resolver(get_package_share_directory),
                         'xarm_link_base', 'xarm_link_tcp', MARGIN)
        arm = dls.arm_from_urdf(probe.description, 'xarm_link_base', 'xarm_link_tcp')
        start_input(probe, env, logs)
        yield probe, env, logs, model, arm
    finally:
        probe.close()
        if teleop is not None:
            teleop.stop()
            print(teleop.output()[-4000:])
        sim.stop()
        kill_gz_partition(env['GZ_PARTITION'])


def test_pulled_into_the_structure_the_arm_keeps_the_margin_slides_and_leaves(stack):
    probe, env, logs, model, arm = stack

    def ready():
        probe.move_to(READY, seconds=4.0)
        time.sleep(1.0)
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_slide.log'),
                                       'clutch_keepout_slide', expected_engaged_s=7.0,
                                       before=ready)
    time.sleep(1.5)
    timed = probe.timed_states_since(start)
    engaged = [s for _, s in timed if s.clutch_engaged]
    assert len(engaged) > 300
    assert not any(s.servo_status in SERVO_HALT_CODES for _, s in timed)
    assert any(s.keepout_bound for s in engaged)
    assert min(s.keepout_clearance_commanded for s in engaged) >= MARGIN - 1e-3
    joints = probe.joints_since(start)
    extra = _extra(model)
    measured = [model.clearance(q, extra)[0] for q in joints[::3]]
    print(f'gazebo slide: smallest measured clearance {min(measured) * 1000.0:.1f} mm')
    assert min(measured) >= MARGIN - 3e-3, min(measured)
    tools = np.array([_tool(arm, q) for q in joints])
    assert tools[:, 1].max() - tools[0, 1] > 0.05
    low = int(np.argmin(tools[:, 0]))
    assert tools[-1, 0] - tools[low, 0] > 0.18
    probe.disarm()


def test_turning_back_from_the_structure_is_followed_at_once(stack):
    probe, env, logs, model, arm = stack

    def ready():
        probe.move_to(READY, seconds=4.0)
        time.sleep(1.0)
    start, _ = replay_with_clean_input(probe, env, str(logs / 'replay_reverse.log'),
                                       'clutch_keepout_reverse', expected_engaged_s=5.0,
                                       before=ready)
    time.sleep(1.5)
    timed = probe.timed_states_since(start)
    assert not any(s.servo_status in SERVO_HALT_CODES for _, s in timed)
    targets = [(t, p) for t, p in probe.targets if t >= start]
    t_target = np.array([t for t, _ in targets])
    x_target = np.array([p[0] for _, p in targets])
    v_target = np.gradient(x_target, t_target)
    back = int(np.argmin(x_target))
    turned = next(i for i in range(back, len(v_target)) if v_target[i] > 0.02)
    timed_tools = [(when, s.measured_tool_pose.position.x) for when, s in timed
                   if s.measured_tool_pose_valid]
    t_tool = np.array([t for t, _ in timed_tools])
    x_tool = np.array([x for _, x in timed_tools])
    window = 5
    v_tool = (x_tool[window:] - x_tool[:-window]) / (t_tool[window:] - t_tool[:-window])
    after = np.nonzero((t_tool[:-window] >= t_target[turned]) & (v_tool > 0.01))[0]
    assert len(after), 'the arm never followed the hand back'
    delay = t_tool[after[0]] - t_target[turned]
    print(f'gazebo reverse: the measured tool follows the turned target after {delay:.3f} s')
    # The lag of the Gazebo stack is about 0.13-0.17 s; at most a few cycles more.
    assert delay < 0.30, delay
    probe.disarm()


@pytest.mark.parametrize('index', [0, 1, 2])
def test_retreat_frees_the_arm_from_each_recorded_contact(stack, index):
    probe, env, logs, model, arm = stack
    path = _contact_paths()[index]
    extra = _extra(model)
    probe.disarm()
    probe.move_to(path['positions'][0], seconds=5.0)
    time.sleep(1.0)
    _arm_teleop(probe, env, logs, index)
    reached = _drive_into_contact(probe, path)
    halted_at = probe.joint_positions()
    print(f'gazebo {path["name"]}: model clearance at the contact '
          f'{model.clearance(halted_at, extra)[0] * 1000.0:.1f} mm, reached {reached}')
    assert model.clearance(halted_at, extra)[0] < MARGIN
    assert not _valid(probe, halted_at, extra)
    process = Process(['python3', os.path.join(TOOLS, 'retreat_arm.py')], env,
                      str(logs / f'retreat_{index}.log'))
    try:
        code = process.wait(180.0)
    finally:
        process.stop()
    output = process.output()
    print(output)
    assert code == 0 and 'retreated' in output, output
    freed = probe.joint_positions()
    assert model.clearance(freed, extra)[0] >= MARGIN - 2e-3
    assert _valid(probe, freed, extra)
    _recover(env, logs, index)
    before = _tool(arm, probe.joint_positions())
    start, _ = replay_with_clean_input(probe, env, str(logs / f'replay_after_{index}.log'),
                                       'clutch_line', expected_engaged_s=2.8)
    states = probe.states_since(start)
    assert not any(s.hold_reason in (TeleopState.HOLD_SERVO_HALTED,
                                     TeleopState.HOLD_SERVO_NOT_READY)
                   and s.servo_status == ServoStatus.HALT_FOR_COLLISION for s in states)
    moved = _tool(arm, probe.joint_positions()) - before
    assert moved[0] == pytest.approx(0.20, abs=0.015)
    probe.disarm()
