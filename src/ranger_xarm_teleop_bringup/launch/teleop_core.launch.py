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
Teleop core: MoveIt Servo, the teleop trajectory controller, the command mux, the teleop node.

It expects a running ros2_control stack (controller manager, joint states, robot state
publisher) and move_group, and starts nothing that addresses hardware itself. The input
side (``ranger_xarm_teleop_bridge`` bridge or replay) is launched separately.

``limits_file`` is required: the safety limits have no defaults. A file marked
simulation-only is refused unless ``simulation:=true``, which only the mock and
simulator wrappers pass. ``motion_enabled`` defaults to false.
"""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ranger_xarm_teleop_bringup.limits import check_limits_file
import yaml


def _true(text: str) -> bool:
    return text.strip().lower() in ('true', '1', 'yes')


def _load(path: str):
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def launch_setup(context, *args, **kwargs):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    simulation = _true(arg('simulation'))
    use_sim_time = _true(arg('use_sim_time'))
    limits_file = arg('limits_file')
    limits = check_limits_file(limits_file, simulation)      # raises: the launch stops here

    share = get_package_share_directory('ranger_xarm_teleop_bringup')
    moveit_share = get_package_share_directory('ranger_xarm_moveit_config')
    description_share = get_package_share_directory('ranger_xarm_description')

    servo = _load(os.path.join(share, 'config', 'servo.yaml'))
    servo['scale'] = {'linear': float(limits['linear_speed_cap_m_s']),
                      'rotational': float(limits['angular_speed_cap_rad_s']),
                      'joint': 0.5}
    teleop_controller = arg('teleop_controller')
    servo['command_out_topic'] = f'/{teleop_controller}/joint_trajectory'

    robot_description = {'robot_description': ParameterValue(Command([
        'xacro ', os.path.join(description_share, 'urdf', 'ranger_xarm.urdf.xacro'),
        ' ', arg('description_args')]), value_type=str)}
    with open(os.path.join(moveit_share, 'config', 'ranger_xarm.srdf'), encoding='utf-8') as f:
        semantic = {'robot_description_semantic': f.read()}
    kinematics = {'robot_description_kinematics':
                  _load(os.path.join(moveit_share, 'config', 'kinematics.yaml'))}
    joint_limits = {'robot_description_planning':
                    _load(os.path.join(moveit_share, 'config', 'joint_limits.yaml')).get(
                        'robot_description_planning', {})}
    sim_time = {'use_sim_time': use_sim_time}

    # The controller's parameter file, with its clock setting for this backend.
    controller = _load(os.path.join(share, 'config', 'teleop_arm_controller.yaml'))
    controller_params = controller.pop('teleop_arm_controller')
    controller_params['ros__parameters']['use_sim_time'] = use_sim_time
    controller[teleop_controller] = controller_params
    handle = tempfile.NamedTemporaryFile('w', prefix='teleop_arm_controller_', suffix='.yaml',
                                         delete=False)
    yaml.safe_dump(controller, handle)
    handle.close()

    spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        arguments=[teleop_controller, '--inactive', '--param-file', handle.name,
                   '--controller-manager', arg('controller_manager'),
                   '--controller-manager-timeout', '120'])

    servo_node = Node(
        package='moveit_servo', executable='servo_node', name='servo_node', output='screen',
        parameters=[{'moveit_servo': servo}, {'update_period': servo['publish_period']},
                    {'planning_group_name': servo['move_group_name']},
                    robot_description, semantic, kinematics, joint_limits, sim_time])

    mux = Node(
        package='ranger_xarm_teleop_servo', executable='arm_mux_node', name='arm_mux',
        output='screen',
        parameters=[{'moveit_controller': arg('moveit_controller'),
                     'teleop_controller': teleop_controller,
                     'controller_manager': arg('controller_manager')}, sim_time])

    teleop_parameters = [limits_file]
    if arg('teleop_params_file'):
        teleop_parameters.append(arg('teleop_params_file'))
    teleop_parameters.append({'motion_enabled': _true(arg('motion_enabled')),
                              'tool_frame': arg('tool_frame'),
                              'servo_tip_frame': arg('servo_tip_frame'),
                              'servo_node': '/servo_node'})
    teleop_parameters.append(sim_time)
    teleop = Node(
        package='ranger_xarm_teleop_servo', executable='teleop_node', name='teleop_servo',
        output='screen', parameters=teleop_parameters)

    return [spawner, servo_node, mux, teleop]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'limits_file',
            description='Parameter file with the safety limit values (required, no default).'),
        DeclareLaunchArgument(
            'simulation', default_value='false',
            description='True only on a simulator path; a simulation-only limits file is '
                        'refused otherwise.'),
        DeclareLaunchArgument(
            'motion_enabled', default_value='false',
            description='G-1: no target is published unless this is true.'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'description_args', default_value='',
            description='xacro arguments of the robot description MoveIt Servo checks '
                        'collisions against; must describe the robot that is running.'),
        DeclareLaunchArgument('moveit_controller', default_value='xarm6_traj_controller'),
        DeclareLaunchArgument('teleop_controller', default_value='teleop_arm_controller'),
        DeclareLaunchArgument('controller_manager', default_value='/controller_manager'),
        DeclareLaunchArgument('tool_frame', default_value='xarm_link_tcp'),
        DeclareLaunchArgument('servo_tip_frame', default_value='xarm_link_tcp'),
        DeclareLaunchArgument(
            'teleop_params_file', default_value='',
            description='Optional extra parameters of the teleop node (keep-out planes, '
                        'wrist frame, wrist-pivot offset, filter).'),
        OpaqueFunction(function=launch_setup),
    ])
