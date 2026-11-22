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
The teleop stack on a real bring-up it did not start (the hardware path).

Start the robot workspace's bring-up first, at the stage you are in (``stage:=check`` for
real_hardware_check.launch.py, ``stage:=execution`` for real_execution.launch.py), then this.
It never starts, stops or reconfigures the bring-up: no controller manager, robot state
publisher, move_group or spawner of its own. The only change it can ever make there is the
command mux's switch between the two arm controllers (G-9), and with motion disabled not even
that.

In order:

1. The options and the limits file are checked (``limits_file`` required, every hardware
   value set by the human; the template and the simulation-only file are refused; reset and
   gripper teleop refused; ``motion_enabled:=true`` refused in the check stage).
2. The preflight reads the bring-up once (``attach.judge``): a controller manager answering,
   a robot description, no /clock, arm joint states, and in the execution stage MoveIt's
   controller active and the teleop controller loaded inactive (the workspace patch). Any
   problem stops the launch with one line saying what to do.
3. Only then the teleop core starts (ranger_xarm_teleop_bringup teleop_core.launch.py without
   ``simulation``), and the recorder with the ``hardware`` profile if ``record:=true``.

``motion_enabled`` defaults to false: the stack computes and publishes its state and the
preview of the target, refuses arming, starts no MoveIt Servo and never lets the arm switch to
teleop. Enabling it is a deliberate argument (procedure, stage 3 onwards).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ranger_xarm_teleop_bringup.limits import check_hardware_limits_file
from ranger_xarm_teleop_hardware.attach import check_launch_options, description_args, tool_frame


def _true(text: str) -> bool:
    return text.strip().lower() in ('true', '1', 'yes')


def launch_setup(context, *args, **kwargs):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    stage = arg('stage')
    motion_enabled = _true(arg('motion_enabled'))
    check_launch_options(stage, motion_enabled, _true(arg('simulation')), _true(arg('reset')),
                         _true(arg('gripper_enabled')))           # raises: nothing started
    limits_file = arg('limits_file')
    check_hardware_limits_file(limits_file)                       # raises: nothing started
    frame = tool_frame(stage, arg('tool_frame'))

    core = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ranger_xarm_teleop_bringup'), 'launch',
            'teleop_core.launch.py')),
        launch_arguments={
            'limits_file': limits_file,
            'simulation': 'false',
            'motion_enabled': 'true' if motion_enabled else 'false',
            'use_sim_time': 'false',
            'description_args': ' '.join(description_args(stage)),
            'moveit_controller': 'xarm6_traj_controller',
            'teleop_controller': 'teleop_arm_controller',
            'tool_frame': frame,
            'servo_tip_frame': frame,
            'teleop_params_file': arg('teleop_params_file'),
            'gripper_enabled': 'false',
            'reset': 'false',
            'feedback': arg('feedback'),
        }.items())
    started = [LogInfo(msg=f'teleop_hardware: attaching to the {stage} stage, motion '
                           f'{"ENABLED" if motion_enabled else "disabled"}'), core]
    if _true(arg('record')):
        started.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('ranger_xarm_teleop_record'), 'launch',
                'record.launch.py')),
            launch_arguments={
                'profile': 'hardware',
                'data_dir': arg('record_data_dir'),
                'session': arg('record_session'),
                'record_extra': arg('record_extra'),
                'limits_file': limits_file,
                'tool_frame': frame,
            }.items()))

    preflight = ExecuteProcess(
        cmd=['ros2', 'run', 'ranger_xarm_teleop_hardware', 'attach_preflight',
             '--stage', stage, '--wait', arg('preflight_wait')],
        name='attach_preflight', output='screen')

    def after_preflight(event, _context):
        if event.returncode == 0:
            return started
        return [LogInfo(msg='teleop_hardware: the preflight refused the attach (lines above); '
                            'nothing was started'),
                EmitEvent(event=Shutdown(reason='attach preflight refused'))]

    return [preflight, RegisterEventHandler(OnProcessExit(target_action=preflight,
                                                          on_exit=after_preflight))]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'stage', description="The bring-up stage that is running: 'check' "
                                 "(real_hardware_check) or 'execution' (real_execution)."),
        DeclareLaunchArgument(
            'limits_file',
            description='Your filled-in copy of hardware_limits_TEMPLATE.yaml, outside any '
                        'checkout (required, no default).'),
        DeclareLaunchArgument(
            'motion_enabled', default_value='false',
            description='G-1: false computes and shows, never commands; true only from the '
                        'execution stage on.'),
        DeclareLaunchArgument(
            'tool_frame', default_value='',
            description='Empty: xarm_link_eef in the check stage (no gripper in that '
                        'description), xarm_link_tcp in the execution stage.'),
        DeclareLaunchArgument('teleop_params_file', default_value=''),
        DeclareLaunchArgument('feedback', default_value='true'),
        DeclareLaunchArgument('record', default_value='false'),
        DeclareLaunchArgument('record_data_dir', default_value=''),
        DeclareLaunchArgument('record_session', default_value=''),
        DeclareLaunchArgument('record_extra', default_value=''),
        DeclareLaunchArgument('preflight_wait', default_value='15.0'),
        # Refused here (they exist so that a mistaken argument is refused, not ignored).
        DeclareLaunchArgument('simulation', default_value='false'),
        DeclareLaunchArgument('reset', default_value='false'),
        DeclareLaunchArgument('gripper_enabled', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
