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
A hardware-shaped MOCK bring-up, for rehearsing a hardware procedure on a laptop.

What it is: a stand-in with the shape of the robot workspace's real bring-up
(real_hardware_check.launch.py for stage check, real_execution.launch.py for stage
execution). The same robot description, ros2_control_node, controller files, controller
spawners and move_group, with the real xArm plugin replaced by mock_components/GenericSystem
and the real gripper proxy replaced by a stand-in (hwmock_gripper_proxy). The execution stage
also loads this package's teleop controller file and spawns teleop_arm_controller inactive,
as the real bring-up does once that fragment is added there. Everything runs on the system
clock; nothing publishes /clock.

What it is not: a robot. Nothing here talks to hardware, nothing is measured, and it proves
nothing about real timing, tracking, limits or the gripper. The arm starts wherever
mock_components starts it (every joint at 0.0: the description has no initial-value hook), the
gripper at the assumed initial_gripper_position. Omitted from the real execution launch: the
trusted-volume filter, the xarm_api parameter file (the mock plugin takes neither) and the
Ouster octomap sensor file (no sensor to feed it). The execution stage also restricts
joint_state_broadcaster to the arm joints (config/hwmock_joint_state_broadcaster.yaml),
because the mock description, unlike the real one, has a gripper system.

It refuses to start beside a running controller manager or a /robot_description publisher
on the same ROS domain: a mock must never be mistaken for, or share a graph with, a real
bring-up.

Arguments: stage (check or execution, required), rviz (default false) and
initial_gripper_position (execution; default 0.0).
"""

import os
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ranger_xarm_teleop_hardware import hwmock
import yaml

GUARD_WAIT_S = 2.0
GUARD_POLL_S = 0.1


def _load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def graph_refusal():
    """Return the refusal reason when a controller manager or description publisher is up."""
    import rclpy
    from rclpy.context import Context

    context = Context()
    rclpy.init(args=[], context=context)
    try:
        node = rclpy.create_node('hwmock_guard', context=context, enable_rosout=False,
                                 start_parameter_services=False)
        try:
            deadline = time.monotonic() + GUARD_WAIT_S
            while True:
                # Node names, not service names: a service name is listed for its clients
                # too, and a client of the controller manager is not a running bring-up.
                names = [('' if ns == '/' else ns) + '/' + name
                         for name, ns in node.get_node_names_and_namespaces()]
                reason = hwmock.refusal_reason(
                    names, node.count_publishers(hwmock.ROBOT_DESCRIPTION_TOPIC))
                if reason or time.monotonic() >= deadline:
                    return reason
                time.sleep(GUARD_POLL_S)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown(context=context)


def _description(stage, planning=False):
    xacro_file = PathJoinSubstitution([
        FindPackageShare('ranger_xarm_description'), 'urdf', 'ranger_xarm.urdf.xacro'])
    args = hwmock.planning_description_args(stage) if planning else hwmock.description_args(stage)
    command = ['xacro ', xacro_file]
    for arg in args:
        command += [' ', arg]
    return {'robot_description': ParameterValue(Command(command), value_type=str)}


def _move_group(pkg, stage, rviz):
    planning_description = _description(stage, planning=True)
    srdf = {'robot_description_semantic': (pkg / 'config' / 'ranger_xarm.srdf').read_text(
        encoding='utf-8')}
    kinematics = {'robot_description_kinematics': _load_yaml(pkg / 'config' / 'kinematics.yaml')}
    limits = _load_yaml(pkg / 'config' / 'joint_limits.yaml')
    planning = {'robot_description_planning': limits.get('robot_description_planning', {})}
    pipeline = {'planning_pipelines': ['ompl'], 'default_planning_pipeline': 'ompl',
                'ompl': _load_yaml(pkg / 'config' / 'ompl_planning.yaml')}
    scene_monitor = {'planning_scene_monitor': {
        'publish_planning_scene': True, 'publish_geometry_updates': True,
        'publish_state_updates': True, 'publish_transforms_updates': True}}
    controllers = _load_yaml(pkg / 'config' / 'moveit_controllers_real.yaml')
    execution = {'trajectory_execution': {
        'allowed_execution_duration_scaling': 1.5, 'allowed_goal_duration_margin': 1.0,
        'allowed_start_tolerance': 0.01}}
    move_group = Node(
        package='moveit_ros_move_group', executable='move_group', output='screen',
        parameters=[
            planning_description, srdf, kinematics, planning, pipeline,
            scene_monitor, controllers, execution,
            {'allow_trajectory_execution': True},
            {'moveit_manage_controllers': False},
            {'publish_robot_description': False},
            {'publish_robot_description_semantic': True},
            {'use_sim_time': False},
        ])
    rviz_config = PathJoinSubstitution([
        FindPackageShare('ranger_xarm_moveit_config'), 'rviz', 'planning.rviz'])
    rviz_node = Node(
        package='rviz2', executable='rviz2', output='screen', condition=IfCondition(rviz),
        arguments=['-d', rviz_config],
        parameters=[planning_description, srdf, kinematics, pipeline])
    return [move_group, rviz_node]


def _stage_actions(context):
    stage = LaunchConfiguration('stage').perform(context)
    try:
        hwmock.check_stage(stage)
    except ValueError as err:
        return [LogInfo(msg=str(err)), Shutdown(reason=str(err))]
    reason = graph_refusal()
    if reason:
        return [LogInfo(msg=reason), Shutdown(reason=reason)]

    pkg = Path(get_package_share_directory('ranger_xarm_moveit_config'))
    own = get_package_share_directory('ranger_xarm_teleop_hardware')
    execution = stage == 'execution'
    description = _description(stage)

    controller_params = [str(pkg / 'config' / 'real_ros2_controllers.yaml')]
    controller_params += [os.path.join(own, name) for name in hwmock.own_config_files(stage)]

    actions = [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen', parameters=[description]),
        Node(package='controller_manager', executable='ros2_control_node', output='screen',
             parameters=controller_params,
             remappings=[('/controller_manager/robot_description', '/robot_description')]),
    ]
    for name, extra in hwmock.spawner_specs(stage):
        actions.append(Node(package='controller_manager', executable='spawner',
                            output='screen', arguments=hwmock.spawner_arguments(name, extra)))
    if not execution:
        actions.append(Node(
            package='rviz2', executable='rviz2', output='screen',
            condition=IfCondition(LaunchConfiguration('rviz')),
            arguments=['-d', PathJoinSubstitution([
                FindPackageShare('ranger_xarm_moveit_config'), 'rviz', 'hardware_check.rviz'])],
            parameters=[description]))
    if execution:
        actions.append(Node(
            package='ranger_xarm_teleop_hardware', executable='hwmock_gripper_proxy',
            name='xarm_gripper_action_proxy', output='screen',
            parameters=[{
                'proxy_action_name': '/xarm_gripper/gripper_action',
                'joint_state_topic': '/joint_states',
                'joint_name': hwmock.GRIPPER_JOINT,
                'initial_position': ParameterValue(
                    LaunchConfiguration('initial_gripper_position'), value_type=float),
                'publish_rate_hz': 10.0,
                'speed_rad_s': 1.0,
            }]))
        actions += _move_group(pkg, stage, LaunchConfiguration('rviz'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('stage', description='check or execution (required).'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Start RViz.'),
        DeclareLaunchArgument(
            'initial_gripper_position', default_value='0.0',
            description='Assumed gripper joint position at start (rad); execution stage.'),
        LogInfo(msg=['hwmock: HARDWARE-SHAPED MOCK (mock_components, no robot) - stage ',
                     LaunchConfiguration('stage')]),
        OpaqueFunction(function=_stage_actions),
    ])
