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
from launch.substitutions import LaunchConfiguration

SIM_DESCRIPTION_ARGS = (
    'add_gripper:=true xarm_load_gazebo_plugin:=false '
    'xarm_ros2_control_plugin:=gz_ros2_control/GazeboSimSystem')


def generate_launch_description():
    share = get_package_share_directory('ranger_xarm_teleop_bringup')
    isaac_share = get_package_share_directory('ranger_xarm_isaac')
    return LaunchDescription([
        DeclareLaunchArgument('motion_enabled', default_value='true'),
        DeclareLaunchArgument('start_moveit', default_value='true'),
        DeclareLaunchArgument('teleop_params_file', default_value=''),
        DeclareLaunchArgument('tool_frame', default_value='xarm_link_tcp'),
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
                'description_args': SIM_DESCRIPTION_ARGS,
                'moveit_controller': 'xarm_xarm6_traj_controller',
                'tool_frame': LaunchConfiguration('tool_frame'),
                'teleop_params_file': LaunchConfiguration('teleop_params_file'),
            }.items()),
    ])
