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

Gripper teleop (``gripper_enabled``, default false): adds the teleop gripper controller
(inactive), names both gripper controllers to the mux so it swaps them with the arm's,
and starts the gripper relay. The limits file must then hold the gripper's range and speed.

Operator feedback (``feedback``, default true): the operator feedback node starts with the
stack, before arming. It observes /teleop/state and /teleop/episode/state and publishes
/teleop/haptic and /teleop/operator/status, which the bridge forwards to the headset. It
decides nothing for teleop; parameters from ``feedback_params_file`` (patterns, rates), and
the gripper's range from the limits file.

Reset (``reset``, default false, rule G-16): a hold-to-run return to a named posture from the
headset, with teleop disarmed. Starts the reset node and its watchdog with the stack, before
arming; with ``reset:=false`` neither starts. The postures and the control's thresholds come
from ``reset_params_file`` (default: this package's config/reset.yaml); the speed scaling,
which has no default, from the limits file (``reset_speed_scaling``). Simulation only: the
reset node refuses every request unless the robot description shows simulator hardware only.
"""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ranger_xarm_teleop_bringup.limits import (check_extra_params_file, check_gripper_limits,
                                               check_limits_file, load_limits)
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
    gripper = _true(arg('gripper_enabled'))
    if gripper:
        check_gripper_limits(limits_file, limits)
        if not arg('moveit_gripper_controller'):
            raise ValueError('gripper_enabled: moveit_gripper_controller (the controller '
                             'MoveIt commands the gripper with) is required')

    share = get_package_share_directory('ranger_xarm_teleop_bringup')
    moveit_share = get_package_share_directory('ranger_xarm_moveit_config')
    description_share = get_package_share_directory('ranger_xarm_description')

    servo = _load(os.path.join(share, 'config', 'servo.yaml'))
    # Joint-jog commands in speed units: Servo's scale.* do not apply. The tool speed caps
    # of the limits file are enforced by the teleop node's solver.
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
    # The same velocity limits for the teleop node's solver (it also reads the URDF's and
    # uses the smaller), so MoveIt Servo's own velocity scaling is never what limits.
    velocity_names, velocity_values = [], []
    for name, entry in sorted(joint_limits['robot_description_planning'].get(
            'joint_limits', {}).items()):
        if isinstance(entry, dict) and entry.get('has_velocity_limits'):
            velocity_names.append(name)
            velocity_values.append(float(entry['max_velocity']))
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
    gripper_nodes = []
    teleop_gripper = arg('teleop_gripper_controller')
    if gripper:
        controller = _load(os.path.join(share, 'config', 'teleop_gripper_controller.yaml'))
        controller_params = controller.pop('teleop_gripper_controller')
        params = controller_params['ros__parameters']
        params['use_sim_time'] = use_sim_time
        joint = arg('gripper_joint')
        tolerances = params['constraints'].pop('xarm_drive_joint')
        params['joints'] = [joint]
        params['constraints'][joint] = tolerances
        controller[teleop_gripper] = controller_params
        handle = tempfile.NamedTemporaryFile('w', prefix='teleop_gripper_controller_',
                                             suffix='.yaml', delete=False)
        yaml.safe_dump(controller, handle)
        handle.close()
        gripper_nodes.append(Node(
            package='controller_manager', executable='spawner', output='screen',
            arguments=[teleop_gripper, '--inactive', '--param-file', handle.name,
                       '--controller-manager', arg('controller_manager'),
                       '--controller-manager-timeout', '120']))
        gripper_nodes.append(Node(
            package='ranger_xarm_teleop_servo', executable='gripper_relay_node',
            output='screen',
            parameters=[{'mode': 'trajectory', 'controller': teleop_gripper, 'joint': joint},
                        sim_time]))

    # No ``name=`` here either (see below): MoveIt Servo's planning scene monitor makes a tf
    # TransformListener with a node of its own, which the remap would also rename to
    # servo_node, so that it answered /servo_node/get_parameters too (with no values).
    # Servo's node is called servo_node by its code, so its parameters and services keep
    # that name.
    servo_node = Node(
        package='moveit_servo', executable='servo_node', output='screen',
        parameters=[{'moveit_servo': servo}, {'update_period': servo['publish_period']},
                    {'planning_group_name': servo['move_group_name']},
                    robot_description, semantic, kinematics, joint_limits, sim_time])

    # No ``name=`` for the two nodes of this package: a launch name becomes a ``__node``
    # remap, which renames every node of the process, so the helper nodes inside would
    # share the main node's name (and its parameter services). Their own names are
    # arm_mux and teleop_servo.
    mux = Node(
        package='ranger_xarm_teleop_servo', executable='arm_mux_node',
        output='screen',
        parameters=[{'moveit_controller': arg('moveit_controller'),
                     'teleop_controller': teleop_controller,
                     'controller_manager': arg('controller_manager'),
                     'moveit_gripper_controller': (arg('moveit_gripper_controller')
                                                   if gripper else ''),
                     'teleop_gripper_controller': teleop_gripper if gripper else '',
                     'gripper_joints': [arg('gripper_joint')]}, sim_time])

    teleop_parameters = [limits_file]
    if arg('teleop_params_file'):
        check_extra_params_file(arg('teleop_params_file'))     # raises: no overriding
        teleop_parameters.append(arg('teleop_params_file'))
    teleop_parameters.append({'motion_enabled': _true(arg('motion_enabled')),
                              'tool_frame': arg('tool_frame'),
                              'servo_tip_frame': arg('servo_tip_frame'),
                              'servo_node': '/servo_node',
                              'servo_output_topic': servo['command_out_topic'],
                              'joint_velocity_limit_names': velocity_names,
                              'joint_velocity_limits': velocity_values,
                              'gripper_enabled': gripper,
                              'gripper_joint': arg('gripper_joint')})
    teleop_parameters.append(sim_time)
    # Single-threaded linear algebra: the solver works on 6 x 6 matrices every 20 ms, and
    # BLAS worker threads competing with a simulator stretch a cycle past the time a
    # headset frame waits for its parts.
    teleop = Node(
        package='ranger_xarm_teleop_servo', executable='teleop_node',
        output='screen', parameters=teleop_parameters,
        additional_env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})

    feedback_nodes = []
    if _true(arg('feedback')):
        params_file = arg('feedback_params_file') or os.path.join(
            get_package_share_directory('ranger_xarm_teleop_bridge'), 'config',
            'operator_feedback.yaml')
        # The limits file gives the gripper's open and closed positions (shown as % closed).
        feedback_nodes.append(Node(
            package='ranger_xarm_teleop_bridge', executable='feedback_node', output='screen',
            parameters=[params_file, limits_file, sim_time]))

    reset_nodes = []
    if _true(arg('reset')):
        speed = limits.get('reset_speed_scaling')
        if not isinstance(speed, (int, float)) or isinstance(speed, bool):
            raise ValueError(f'{limits_file}: reset_speed_scaling is required with reset:=true '
                             'and has no default (G-16)')
        reset_params = arg('reset_params_file') or os.path.join(share, 'config', 'reset.yaml')
        check_extra_params_file(reset_params)                  # raises: no overriding
        if 'enabled' in load_limits(reset_params):
            raise ValueError(f'{reset_params} sets enabled: the reset is switched on by the '
                             'launch argument reset:=true only')
        reset_nodes.append(Node(
            package='ranger_xarm_teleop_servo', executable='reset_watchdog_node',
            output='screen', parameters=[sim_time]))
        reset_nodes.append(Node(
            package='ranger_xarm_teleop_servo', executable='reset_node', output='screen',
            parameters=[reset_params, {'reset_speed_scaling': float(speed),
                                       'enabled': True}, sim_time]))

    return [spawner, servo_node, mux, teleop] + gripper_nodes + feedback_nodes + reset_nodes


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
        DeclareLaunchArgument(
            'gripper_enabled', default_value='false',
            description='G-1: the right trigger opens and closes the gripper only if true. '
                        'Off is safe: nothing is ever sent to the gripper.'),
        DeclareLaunchArgument('gripper_joint', default_value='xarm_drive_joint'),
        DeclareLaunchArgument(
            'moveit_gripper_controller', default_value='',
            description='The controller MoveIt commands the gripper with (required with '
                        'gripper_enabled); the mux swaps it with the teleop one.'),
        DeclareLaunchArgument('teleop_gripper_controller',
                              default_value='teleop_gripper_controller'),
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
        DeclareLaunchArgument(
            'reset', default_value='false',
            description='Start the reset node and its watchdog: hold the right thumbstick '
                        'pulled back, disarmed, to return the arm to the ready posture (G-16; '
                        'simulation only).'),
        DeclareLaunchArgument(
            'reset_params_file', default_value='',
            description="Postures and control thresholds of the reset (empty: this package's "
                        'config/reset.yaml).'),
        DeclareLaunchArgument(
            'feedback', default_value='true',
            description='Start the operator feedback node (vibration and panel status for the '
                        'headset; information only).'),
        DeclareLaunchArgument(
            'feedback_params_file', default_value='',
            description='Parameters of the operator feedback node (empty: the bridge '
                        "package's config/operator_feedback.yaml)."),
        OpaqueFunction(function=launch_setup),
    ])
