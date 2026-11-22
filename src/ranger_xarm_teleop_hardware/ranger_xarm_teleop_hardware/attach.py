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
What the teleop stack needs from a real bring-up before it attaches. No ROS imports.

The hardware teleop launch (launch/teleop_hardware.launch.py) attaches to a bring-up it did
not start and never starts, stops or reconfigures it (only the command mux switches the two
arm controllers, G-9). Its preflight reads the bring-up once and refuses with one line per
problem, each saying what to do. ``judge`` decides; the preflight node only reads.

Stages, as the robot workspace's real bring-up has them:

* ``check`` (real_hardware_check.launch.py): joint states only; no trajectory controller, no
  gripper. Teleop runs with motion disabled: it computes its state and the preview.
* ``execution`` (real_execution.launch.py, with the workspace patch that adds the teleop
  controller): MoveIt's trajectory controller active and teleop's loaded inactive.
"""

from typing import Dict, List, Optional, Sequence

STAGES = ('check', 'execution')
ARM_JOINTS = tuple('xarm_joint%d' % n for n in range(1, 7))
MOVEIT_CONTROLLER = 'xarm6_traj_controller'
TELEOP_CONTROLLER = 'teleop_arm_controller'
BROADCASTER = 'joint_state_broadcaster'
REAL_PLUGIN = 'uf_robot_hardware/UFRobotSystemHardware'
PATCH = ('the workspace patch that adds teleop_arm_controller to real_ros2_controllers.yaml '
         'and spawns it inactive in real_execution.launch.py')


def check_stage(stage: str) -> str:
    """Return the stage, or raise ValueError with the one-line refusal."""
    if stage not in STAGES:
        raise ValueError(f"stage must be 'check' or 'execution' (the bring-up stage that is "
                         f'running), got {stage!r}')
    return stage


def check_launch_options(stage: str, motion_enabled: bool, simulation: bool, reset: bool,
                         gripper_enabled: bool) -> None:
    """Raise ValueError with one line for an option the hardware path refuses."""
    check_stage(stage)
    if simulation:
        raise ValueError('simulation:=true is refused on the hardware path: it attaches to a '
                         'real bring-up, never to a simulator')
    if stage == 'check' and motion_enabled:
        raise ValueError('motion_enabled:=true is refused in the check stage: that bring-up has '
                         'no trajectory controller; start the execution stage for motion')
    if reset:
        raise ValueError('reset:=true is refused on the hardware path: the reset (G-16) is '
                         'simulation only')
    if gripper_enabled:
        raise ValueError('gripper_enabled:=true is refused on the hardware path: the gripper '
                         'adapter for the real gripper action is not built yet')


def tool_frame(stage: str, requested: str = '') -> str:
    """
    Return the tool frame (and MoveIt Servo's tip) for the stage.

    The check stage's description has no gripper, and with it no ``xarm_link_tcp``: the
    flange (``xarm_link_eef``) is the last frame there. A frame the person names is used as
    it is.
    """
    check_stage(stage)
    if requested:
        return requested
    return 'xarm_link_eef' if stage == 'check' else 'xarm_link_tcp'


def description_args(stage: str) -> List[str]:
    """Return the xacro arguments of the stage's real robot description (MoveIt Servo's model)."""
    check_stage(stage)
    return ['use_xarm:=true', 'use_static_geometry:=true', 'use_static_collision:=true',
            'add_gripper:=' + ('true' if stage == 'execution' else 'false'),
            'xarm_ros2_control_plugin:=' + REAL_PLUGIN, 'xarm_report_type:=normal',
            'xarm_velocity_control:=false', 'laptop_open_angle_deg:=90.0']


def judge(stage: str, controllers: Optional[Dict[str, str]], description: Optional[str],
          clock_publishers: int, joint_names: Optional[Sequence[str]]) -> List[str]:
    """
    Return the problems that refuse the attach (empty: attach).

    controllers: name -> state from the controller manager (None: it did not answer).
    description: the robot description text (None: none arrived). clock_publishers: how
    many publishers /clock has. joint_names: the names of an arm joint state that arrived
    (None: none).
    """
    check_stage(stage)
    problems = []
    if controllers is None:
        problems.append('no controller manager answered on /controller_manager: start the '
                        f'bring-up ({stage} stage) first; this launch never starts it')
        controllers = {}
    if description is None:
        problems.append('no /robot_description arrived: start the bring-up first')
    if clock_publishers:
        problems.append('something publishes /clock: a simulator is running or a node uses '
                        'simulation time; the hardware path runs on the host clock only')
    if controllers and controllers.get(BROADCASTER) != 'active':
        problems.append(f'{BROADCASTER} is {controllers.get(BROADCASTER, "not loaded")}: the '
                        'bring-up publishes no joint states; restart the bring-up')
    if controllers or stage == 'execution':
        teleop = controllers.get(TELEOP_CONTROLLER)
        if teleop == 'active':
            problems.append(f'{TELEOP_CONTROLLER} is active: only the command mux may activate '
                            'it; stop everything and restart the bring-up')
        elif stage == 'execution' and teleop is None and controllers:
            problems.append(f'{TELEOP_CONTROLLER} is not loaded in the bring-up: apply {PATCH}, '
                            'then restart the bring-up')
        elif stage == 'execution' and teleop not in (None, 'inactive'):
            problems.append(f'{TELEOP_CONTROLLER} is {teleop}, not inactive: restart the bring-up')
    if stage == 'execution' and controllers:
        moveit = controllers.get(MOVEIT_CONTROLLER)
        if moveit != 'active':
            problems.append(f'{MOVEIT_CONTROLLER} is {moveit or "not loaded"}: the execution '
                            'stage needs MoveIt owning the arm; restart the bring-up')
    if joint_names is None:
        problems.append('no joint state with the six arm joints arrived on /joint_states: '
                        'check the bring-up and the arm connection')
    else:
        missing = [j for j in ARM_JOINTS if j not in joint_names]
        if missing:
            problems.append('the joint states lack ' + ', '.join(missing))
    return problems
