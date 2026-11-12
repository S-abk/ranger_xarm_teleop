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
components, move_group, RViz) and the teleop core with the simulation-only limit values
and ``motion_enabled`` true. Every node runs on the system clock.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

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
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(moveit_share, 'launch', 'fake_execution.launch.py'))),
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
            }.items()),
    ])
