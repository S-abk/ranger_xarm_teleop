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
Teleop against mock execution: no simulator, no hardware.

Starts ``ranger_xarm_moveit_config fake_execution.launch.py`` (ros2_control with mock
components, move_group, and RViz unless ``rviz`` is false) and the teleop core with the
simulation-only limit values and ``motion_enabled`` true. Every node runs on the system
clock.

Gripper teleop is on (``gripper_enabled`` true). The robot workspace's mock execution has
no gripper controller, so this file also starts a stand-in for MoveIt's
(``xarm_gripper_traj_controller``, config/mock_moveit_gripper_controller.yaml): the mux
needs a second side to swap. move_group cannot command it in mock.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

MOCK_MOVEIT_GRIPPER_CONTROLLER = 'xarm_gripper_traj_controller'

# Must describe the robot fake_execution.launch.py starts.
MOCK_DESCRIPTION_ARGS = (
    'use_xarm:=true use_static_geometry:=true use_static_collision:=true add_gripper:=true '
    'xarm_ros2_control_plugin:=mock_components/GenericSystem xarm_mount_yaw:=0.0 '
    'use_laptop_collision:=true laptop_open_angle_deg:=90.0')


def generate_launch_description():
    share = get_package_share_directory('ranger_xarm_teleop_bringup')
    moveit_share = get_package_share_directory('ranger_xarm_moveit_config')
    return LaunchDescription([
        DeclareLaunchArgument('motion_enabled', default_value='true'),
        DeclareLaunchArgument('teleop_params_file', default_value=''),
        DeclareLaunchArgument('tool_frame', default_value='xarm_link_tcp'),
        DeclareLaunchArgument('gripper_enabled', default_value='true'),
        DeclareLaunchArgument(
            'record', default_value='false',
            description='Start the demonstration recorder and episode node (paused until X).'),
        DeclareLaunchArgument(
            'reset', default_value='false',
            description='Hold the right thumbstick pulled back (disarmed) to return the arm to '
                        'the ready posture (G-16).'),
        DeclareLaunchArgument('record_profile', default_value='mock'),
        DeclareLaunchArgument('record_data_dir', default_value=''),
        DeclareLaunchArgument('record_session', default_value=''),
        DeclareLaunchArgument('record_extra', default_value=''),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Start RViz with the planning view.'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(moveit_share, 'launch', 'fake_execution.launch.py')),
            launch_arguments={'use_rviz': LaunchConfiguration('rviz')}.items()),
        Node(package='controller_manager', executable='spawner', output='screen',
             arguments=[MOCK_MOVEIT_GRIPPER_CONTROLLER, '--param-file',
                        os.path.join(share, 'config', 'mock_moveit_gripper_controller.yaml'),
                        '--controller-manager', '/controller_manager',
                        '--controller-manager-timeout', '120'],
             condition=IfCondition(LaunchConfiguration('gripper_enabled'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, 'launch', 'teleop_core.launch.py')),
            launch_arguments={
                'limits_file': os.path.join(share, 'config', 'SIMULATION_ONLY_limits.yaml'),
                'simulation': 'true',
                'motion_enabled': LaunchConfiguration('motion_enabled'),
                'use_sim_time': 'false',
                'description_args': MOCK_DESCRIPTION_ARGS,
                'moveit_controller': 'xarm6_traj_controller',
                'tool_frame': LaunchConfiguration('tool_frame'),
                'teleop_params_file': LaunchConfiguration('teleop_params_file'),
                'gripper_enabled': LaunchConfiguration('gripper_enabled'),
                'reset': LaunchConfiguration('reset'),
                'moveit_gripper_controller': MOCK_MOVEIT_GRIPPER_CONTROLLER,
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
