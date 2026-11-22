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
Pure helpers of the hardware-shaped mock bring-up (launch/hwmock.launch.py).

No launch, rclpy or ROS imports here, so every decision is testable without a ROS graph:
the xacro arguments and spawner lists of each stage, the refusal rules, and the gripper
stand-in's clamp and step.
"""

STAGES = ('check', 'execution')

MOCK_PLUGIN = 'mock_components/GenericSystem'

# The planning margin of the real execution launch (move_group's description only).
PLANNING_MARGIN = '0.005'

ARM_JOINTS = tuple('xarm_joint%d' % n for n in range(1, 7))
GRIPPER_JOINT = 'xarm_drive_joint'

# The gripper's model-state range the real proxy keeps (just inside the URDF limit).
GRIPPER_MIN = 0.0
GRIPPER_MAX = 0.849

CONTROLLER_MANAGER_NODE = '/controller_manager'
ROBOT_DESCRIPTION_TOPIC = '/robot_description'

REFUSAL = ('hwmock refused: a controller manager is already answering on this ROS domain '
           '(a real bring-up or another stack); hwmock never starts beside one')

# The arguments of the real bring-up's xacro call, in its order, for each stage: the real
# launch files' own (ranger_xarm_moveit_config real_hardware_check.launch.py and
# real_execution.launch.py) with the plugin replaced, and xarm_robot_ip left out. The robot
# address is only read inside the real plugin's branch of the description, which the mock
# plugin never takes, so the xacro default (empty) is used and no address appears here.
_COMMON_HEAD = (
    'use_xarm:=true',
    'use_static_geometry:=true',
    'use_static_collision:=true',
)
_COMMON_TAIL = (
    'xarm_report_type:=normal',
    'xarm_velocity_control:=false',
    'laptop_open_angle_deg:=90.0',
)
_ADD_GRIPPER = {'check': 'add_gripper:=false', 'execution': 'add_gripper:=true'}


def check_stage(stage):
    """Return the stage, or raise ValueError with the one-line refusal."""
    if stage not in STAGES:
        raise ValueError(
            "hwmock refused: stage must be 'check' or 'execution', got %r" % (stage,))
    return stage


def description_args(stage):
    """Return the xacro arguments (list of key:=value) of the stage's robot description."""
    check_stage(stage)
    return [*_COMMON_HEAD, _ADD_GRIPPER[stage], 'xarm_ros2_control_plugin:=' + MOCK_PLUGIN,
            *_COMMON_TAIL]


def planning_description_args(stage):
    """Return the arguments of move_group's description: the stage's plus the margin."""
    return description_args(stage) + ['planning_margin:=' + PLANNING_MARGIN]


def spawner_specs(stage):
    """
    Return the stage's controller spawners as (name, extra spawner arguments) tuples.

    Check: the broadcaster only. Execution: the broadcaster, the trajectory controller
    (active) and the teleop trajectory controller, inactive.
    """
    check_stage(stage)
    specs = [('joint_state_broadcaster', ())]
    if stage == 'execution':
        specs += [('xarm6_traj_controller', ()),
                  ('teleop_arm_controller', ('--inactive',))]
    return specs


def spawner_arguments(name, extra=()):
    """Return a spawner's command-line arguments, as the real launch files give them."""
    return [name, *extra, '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '30']


TELEOP_FRAGMENT = 'config/teleop_arm_controller_real.yaml'
BROADCASTER_FILE = 'config/hwmock_joint_state_broadcaster.yaml'


def own_config_files(stage):
    """
    Return this package's controller files for the stage (package-share relative paths).

    They are loaded by ros2_control_node after the workspace's real_ros2_controllers.yaml.

    Execution only: the teleop controller file, and a file restricting joint_state_broadcaster
    to the six arm joints. On hardware the broadcaster publishes the arm joints only (the
    gripper is not a ros2_control system there); the mock plugin's description has a gripper
    system, so without the restriction the broadcaster would publish the gripper joint beside
    the gripper stand-in.
    """
    check_stage(stage)
    return [TELEOP_FRAGMENT, BROADCASTER_FILE] if stage == 'execution' else []


def refusal_reason(node_names, robot_description_publishers):
    """
    Return the one-line refusal when the graph shows a running bring-up, else None.

    node_names: the fully qualified names of the nodes in the graph (a controller manager is
    a node named /controller_manager; a service name would not do, as the graph lists a
    service for its clients too). robot_description_publishers: how many publishers
    /robot_description has.
    """
    if CONTROLLER_MANAGER_NODE in node_names or robot_description_publishers > 0:
        return REFUSAL
    return None


def clamp_position(value, low=GRIPPER_MIN, high=GRIPPER_MAX):
    """Return value limited to [low, high]."""
    return min(max(float(value), low), high)


def step_toward(current, target, speed, dt):
    """Return the position after moving from current toward target at speed for dt seconds."""
    step = max(speed, 0.0) * max(dt, 0.0)
    if abs(target - current) <= step:
        return target
    return current + step if target > current else current - step
