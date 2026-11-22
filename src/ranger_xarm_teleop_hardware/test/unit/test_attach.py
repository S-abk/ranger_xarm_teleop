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
"""INV-4, INV-5: what the hardware attach refuses, each in one line saying what to do."""

import pytest
from ranger_xarm_teleop_hardware.attach import (ARM_JOINTS, check_launch_options,
                                                description_args, judge, PATCH, tool_frame)

DESCRIPTION = '<robot name="r"/>'
CHECK = {'joint_state_broadcaster': 'active'}
EXECUTION = {'joint_state_broadcaster': 'active', 'xarm6_traj_controller': 'active',
             'teleop_arm_controller': 'inactive'}


def problems(stage, controllers, description=DESCRIPTION, clock=0, joints=ARM_JOINTS):
    return judge(stage, controllers, description, clock, joints)


def test_a_complete_bring_up_is_attached_at_each_stage():
    assert problems('check', CHECK) == []
    assert problems('execution', EXECUTION) == []


def test_without_a_controller_manager_nothing_is_attached():
    lines = problems('execution', None, None, 0, None)
    assert any('no controller manager' in line and 'start the bring-up' in line for line in lines)
    assert any('/robot_description' in line for line in lines)


def test_the_execution_stage_needs_the_teleop_controller_from_the_patch():
    lines = problems('execution', {k: v for k, v in EXECUTION.items()
                                   if k != 'teleop_arm_controller'})
    assert lines == [f'teleop_arm_controller is not loaded in the bring-up: apply {PATCH}, '
                     'then restart the bring-up']


@pytest.mark.parametrize('stage', ['check', 'execution'])
def test_an_active_teleop_controller_is_refused(stage):
    controllers = dict(EXECUTION if stage == 'execution' else CHECK)
    controllers['teleop_arm_controller'] = 'active'
    if stage == 'execution':
        controllers['xarm6_traj_controller'] = 'inactive'
    lines = problems(stage, controllers)
    assert any(line.startswith('teleop_arm_controller is active: only the command mux')
               for line in lines)


def test_the_execution_stage_needs_moveit_owning_the_arm():
    lines = problems('execution', dict(EXECUTION, xarm6_traj_controller='inactive'))
    assert any(line.startswith('xarm6_traj_controller is inactive') for line in lines)


def test_no_joint_states_or_a_simulator_clock_is_refused():
    assert any('joint state' in line for line in problems('check', CHECK, joints=None))
    assert any('lack xarm_joint6' in line for line in problems('check', CHECK,
                                                               joints=ARM_JOINTS[:5]))
    assert any('/clock' in line for line in problems('check', CHECK, clock=1))
    assert any('joint_state_broadcaster is inactive' in line
               for line in problems('check', {'joint_state_broadcaster': 'inactive'}))


def test_every_problem_is_one_line():
    lines = problems('execution', {'joint_state_broadcaster': 'inactive',
                                   'teleop_arm_controller': 'active'}, None, 2, None)
    assert len(lines) >= 5 and all('\n' not in line for line in lines)


@pytest.mark.parametrize('options, words', [
    (dict(stage='check', motion_enabled=True), 'refused in the check stage'),
    (dict(simulation=True), 'simulation:=true is refused'),
    (dict(reset=True), 'reset:=true is refused'),
    (dict(gripper_enabled=True), 'gripper_enabled:=true is refused'),
    (dict(stage='both'), "stage must be 'check' or 'execution'"),
])
def test_launch_options_the_hardware_path_refuses(options, words):
    values = dict(stage='execution', motion_enabled=False, simulation=False, reset=False,
                  gripper_enabled=False)
    values.update(options)
    with pytest.raises(ValueError, match=words):
        check_launch_options(**values)


def test_motion_may_be_enabled_in_the_execution_stage_only():
    check_launch_options('execution', True, False, False, False)
    check_launch_options('check', False, False, False, False)


def test_the_description_matches_the_stage_and_names_the_real_plugin():
    assert 'add_gripper:=false' in description_args('check')
    assert 'add_gripper:=true' in description_args('execution')
    for stage in ('check', 'execution'):
        args = description_args(stage)
        assert 'xarm_velocity_control:=false' in args
        assert 'xarm_ros2_control_plugin:=uf_robot_hardware/UFRobotSystemHardware' in args
        assert not any(a.split(':=')[0].endswith('_ip') for a in args)   # no address


def test_the_check_stage_has_no_gripper_so_the_tool_frame_is_the_flange():
    assert tool_frame('check') == 'xarm_link_eef'
    assert tool_frame('execution') == 'xarm_link_tcp'
    assert tool_frame('check', 'my_tool') == 'my_tool'
