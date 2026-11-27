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
MoveIt for a driven base (simulation): what this stack asks of the robot workspace. No ROS imports.

With the base driven (``base:=true`` in simulation) the simulator publishes ``odom ->
base_footprint`` and the robot description has wheels and no weld to ``world``. move_group is
then started with ``drive_base:=true`` (``ranger_xarm_isaac`` ``moveit.launch.py``): it loads the
semantic description below, which adds a planar virtual joint ``odom -> base_footprint`` (so
MoveIt plans in ``odom`` with the base where the robot is) and disables the wheel/body pairs that
touch at rest. MoveIt Servo loads the same file, so it reads move_group's planning scene, and the
episode sidecar records it. With the base off nothing here is used: both keep the shipped file
and move_group gets no argument at all.
"""

from typing import Dict, Iterable, Optional

# The workspace's names this stack depends on (ranger_xarm_isaac moveit.launch.py and
# ranger_xarm_moveit_config/config).
DRIVE_BASE_ARG = 'drive_base'
SHIPPED_SRDF = 'ranger_xarm.srdf'
DRIVEN_BASE_SRDF = 'ranger_xarm_mobile_base_wheels.srdf'

# The launch files' truth rule for a launch argument's text: text.lower() in TRUE_WORDS.
TRUE_WORDS = ('true', '1', 'yes')

# moveit.launch.py's own arguments that this stack sets itself or does not offer. An included
# launch file is not scoped: one given on the command line would reach it unseen.
NOT_OFFERED = {
    'drive_base': 'it is set from base (base:=true drives the base and starts MoveIt for it)',
    'planning_frame': "this stack follows the robot workspace's default planning frame for a "
                      'driven base (odom) and does not offer the choice',
    'obstacle_map': "this stack does not start MoveIt's obstacle map",
}
MOCK_NOT_OFFERED = ('drive_base',)


def refused_arguments(given: Iterable[str], names: Iterable[str]) -> Optional[str]:
    """Return the refusal for the ``names`` among the ``given`` launch arguments (None: none)."""
    found = [name for name in names if name in set(given)]
    if not found:
        return None
    return ('refused: ' + '; '.join(f'{name}:= is not an argument here ({NOT_OFFERED[name]})'
                                    for name in found))


def moveit_arguments(base: bool) -> Dict[str, str]:
    """Return the arguments for the workspace's moveit.launch.py (none unless base driven)."""
    return {DRIVE_BASE_ARG: 'true'} if base else {}


def semantic_file(drive_base: bool) -> str:
    """Return the semantic description file (in ranger_xarm_moveit_config/config) Servo loads."""
    return DRIVEN_BASE_SRDF if drive_base else SHIPPED_SRDF


def check_drive_base(drive_base: bool, base: bool, simulation: bool) -> None:
    """
    Refuse ``drive_base`` where it does not belong (raises ``ValueError`` naming why).

    The driven-base semantic description is the simulated base's: its virtual joint needs the
    simulator's ``odom -> base_footprint`` and its wheel pairs the simulator's wheeled description.
    The hardware path keeps the base off and welded; the base node runs only with ``base:=true``.
    """
    if not drive_base:
        return
    if not simulation:
        raise ValueError('drive_base:=true is refused outside simulation: the driven-base '
                         'semantic description is for the simulated base (the hardware path '
                         'keeps the base off and welded to world)')
    if not base:
        raise ValueError('drive_base:=true needs base:=true: MoveIt Servo would load the '
                         'driven-base semantic description beside a robot whose base is not '
                         'driven (teleop_sim.launch.py sets drive_base from base)')
