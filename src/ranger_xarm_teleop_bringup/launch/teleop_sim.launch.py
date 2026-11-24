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
Teleop against a simulator (Gazebo or Isaac Sim) that is already running.

Start the simulator and its controllers first, for example
``ros2 launch ranger_xarm_gazebo sim.launch.py headless:=true``. This file then starts
move_group (``ranger_xarm_isaac moveit.launch.py``, unless ``start_moveit:=false``) and
the teleop core with the simulation-only limit values and ``motion_enabled`` true.
Gripper teleop is on (``gripper_enabled`` true) and swaps with the simulator's
``xarm_xarm_gripper_traj_controller``.

Clocks: MoveIt Servo, the mux, the teleop node and the teleop controller use simulation
time (their stamps must agree with the simulator's joint states). The input bridge or
replay stays on the system clock; input staleness is measured on the monotonic clock.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

SIM_DESCRIPTION_ARGS = (
    'add_gripper:=true xarm_load_gazebo_plugin:=false '
    'xarm_ros2_control_plugin:=gz_ros2_control/GazeboSimSystem')
# With base:=true the simulator runs the base on its wheels (sim.launch.py drive_base:=true):
# MoveIt Servo's description must then be that robot too, or the teleop node refuses to arm
# (its link keep-out model is built from the running robot's description).
DRIVABLE_BASE_ARGS = ' use_wheels:=true fix_base_to_world:=false'


def generate_launch_description():
    share = get_package_share_directory('ranger_xarm_teleop_bringup')
    isaac_share = get_package_share_directory('ranger_xarm_isaac')
    return LaunchDescription([
        DeclareLaunchArgument('motion_enabled', default_value='true'),
        DeclareLaunchArgument('start_moveit', default_value='true'),
        DeclareLaunchArgument('teleop_params_file', default_value=''),
        DeclareLaunchArgument('tool_frame', default_value='xarm_link_tcp'),
        DeclareLaunchArgument('gripper_enabled', default_value='true'),
        DeclareLaunchArgument(
            'record', default_value='false',
            description='Start the demonstration recorder and episode node (paused until X).'),
        DeclareLaunchArgument(
            'base', default_value='false',
            description='Base driving from the headset (G-19, simulation only).'),
        DeclareLaunchArgument(
            'reset', default_value='false',
            description='Hold the right thumbstick pulled back (disarmed) to return the arm to '
                        'the ready posture (G-16).'),
        DeclareLaunchArgument(
            'feedback', default_value='true',
            description='Start the operator feedback node (teleop_core.launch.py, feedback).'),
        DeclareLaunchArgument('record_profile', default_value='gazebo'),
        DeclareLaunchArgument('record_data_dir', default_value=''),
        DeclareLaunchArgument('record_session', default_value=''),
        DeclareLaunchArgument('record_extra', default_value=''),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(isaac_share, 'launch', 'moveit.launch.py')),
            condition=IfCondition(LaunchConfiguration('start_moveit'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, 'launch', 'teleop_core.launch.py')),
            launch_arguments={
                'limits_file': os.path.join(share, 'config', 'SIMULATION_ONLY_limits.yaml'),
                'simulation': 'true',
                'motion_enabled': LaunchConfiguration('motion_enabled'),
                'use_sim_time': 'true',
                'description_args': PythonExpression([
                    repr(SIM_DESCRIPTION_ARGS), ' + (', repr(DRIVABLE_BASE_ARGS), " if '",
                    LaunchConfiguration('base'), "'.lower() in ('true', '1', 'yes') else '')"]),
                'moveit_controller': 'xarm_xarm6_traj_controller',
                'tool_frame': LaunchConfiguration('tool_frame'),
                'teleop_params_file': LaunchConfiguration('teleop_params_file'),
                'gripper_enabled': LaunchConfiguration('gripper_enabled'),
                'reset': LaunchConfiguration('reset'),
                'base': LaunchConfiguration('base'),
                'feedback': LaunchConfiguration('feedback'),
                'moveit_gripper_controller': 'xarm_xarm_gripper_traj_controller',
            }.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('ranger_xarm_teleop_record'), 'launch',
                'record.launch.py')),
            launch_arguments={
                'profile': LaunchConfiguration('record_profile'),
                'data_dir': LaunchConfiguration('record_data_dir'),
                'session': LaunchConfiguration('record_session'),
                'record_extra': LaunchConfiguration('record_extra'),
                'limits_file': os.path.join(share, 'config', 'SIMULATION_ONLY_limits.yaml'),
                'tool_frame': LaunchConfiguration('tool_frame'),
            }.items(),
            condition=IfCondition(LaunchConfiguration('record'))),
    ])
