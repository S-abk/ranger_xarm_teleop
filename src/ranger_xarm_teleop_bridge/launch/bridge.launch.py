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
"""Launch the relay-to-ROS input bridge."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Start ``bridge_node`` with the relay URL and an optional JSONL recording path."""
    return LaunchDescription([
        DeclareLaunchArgument('relay_url', default_value='ws://127.0.0.1:8443/ws',
                              description='WebSocket URL of the vr-teleop-kit relay'),
        DeclareLaunchArgument('record_path', default_value='',
                              description='JSONL file for the raw relay text (empty: off)'),
        DeclareLaunchArgument('frame_id', default_value='quest_world',
                              description='frame_id of the published poses'),
        Node(
            package='ranger_xarm_teleop_bridge',
            executable='bridge_node',
            name='teleop_input_bridge',
            output='screen',
            parameters=[{
                'relay_url': LaunchConfiguration('relay_url'),
                'record_path': LaunchConfiguration('record_path'),
                'frame_id': LaunchConfiguration('frame_id'),
            }],
        ),
    ])
