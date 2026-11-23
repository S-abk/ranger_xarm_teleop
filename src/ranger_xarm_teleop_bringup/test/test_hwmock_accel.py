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
Procedure stage 4 at a fifth of the speed caps on the hardware-shaped mock (G-18).

The mock is the bring-up's execution stage (ranger_xarm_teleop_hardware hwmock.launch.py
stage:=execution). The teleop launch attaches once, at the start of the module, with limits
written for this module only: a fifth of the simulation speed caps and a hardware-like joint
speed cap and acceleration limit (0.4 rad/s, 2 rad/s^2: a 0.2 s stop). These are test values,
not guidance. One stroke releases while moving, and hw_check's stage-4 stop rules must pass:
the commanded joint acceleration on MoveIt Servo's output and the stop time after a release.
Nothing is restarted within the module. Nothing here starts a robot.
"""

import os
import time

from ament_index_python.packages import get_package_share_directory
from hw_helpers import (check_report, start_attach, start_check, start_hwmock,
                        wait_for_controllers, write_hardware_limits)
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner
from stack_helpers import child_environment, Probe, READY, replay, start_input

EXECUTION = {'joint_state_broadcaster': 'active', 'xarm6_traj_controller': 'active',
             'teleop_arm_controller': 'inactive'}
JOINT_LIMITS = os.path.join(get_package_share_directory('ranger_xarm_moveit_config'), 'config',
                            'joint_limits.yaml')
FIFTH = {'linear_speed_cap_m_s': 0.05, 'angular_speed_cap_rad_s': 0.2,
         'joint_speed_cap_rad_s': 0.4, 'joint_acceleration_limit_rad_s2': 2.0}


@pytest.fixture(scope='module')
def fifth(tmp_path_factory):
    logs = tmp_path_factory.mktemp('hwmock_accel')
    env = child_environment(offset=22)
    bringup = start_hwmock(env, 'execution', logs)
    probe = Probe(env)
    limits = write_hardware_limits(logs, **FIFTH)
    attach = None
    try:
        wait_for_controllers(probe, EXECUTION, 120.0)
        probe.move_to(READY, 4.0)
        attach = start_attach(env, 'execution', limits, logs, 'fifth', 'motion_enabled:=true')
        since = time.monotonic()
        probe.wait_for(lambda: any(True for _ in probe.states_since(since)), 120.0,
                       'the teleop state')
        start_input(probe, env, logs)
        # The node builds its chain and keep-out model first; an arm press before it knows the
        # arm is refused (posture unknown).
        probe.wait_for(lambda: probe.state().measured_tool_pose_valid, 120.0,
                       'the teleop node to know the arm')
        yield {'probe': probe, 'env': env, 'logs': logs, 'limits': limits}
    finally:
        probe.close()
        if attach is not None:
            attach.stop()
            print(attach.output()[-6000:])
        bringup.stop()
        print(bringup.output()[-3000:])


def test_a_stroke_at_a_fifth_of_the_caps_passes_the_acceleration_and_stop_rules(fifth):
    """G-18 at the first-session speed caps: hw_check's stage-4 stop rules pass."""
    probe, env, logs = fifth['probe'], fifth['env'], fifth['logs']
    process, data_dir = start_check(env, 'teleop', logs, 'teleop_fifth', '--seconds', '25',
                                    '--limits', fifth['limits'], '--joint-limits', JOINT_LIMITS)
    try:
        time.sleep(3.0)                 # everything joins before arming
        assert not probe.state().armed
        start = time.monotonic()
        # The recording arms (B), engages, and releases the grip while the hand still moves.
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
