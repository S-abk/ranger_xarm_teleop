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
Refuse simulation-only limit values on anything but a simulator. No ROS imports.

The gate document lets limit values be chosen "for simulation only" in a file that the
hardware path must refuse. The file sets ``limits_are_simulation_only``; the node then
reads the robot description and starts only if every ros2_control hardware plugin in
it is one of the simulator plugins below. The list is code, not a parameter, so no
launch file can extend it.
"""

from typing import List
import xml.etree.ElementTree as ElementTree

SIMULATOR_PLUGINS = (
    'mock_components/GenericSystem',
    'gz_ros2_control/GazeboSimSystem',
    'joint_state_topic_hardware_interface/JointStateTopicSystem',
)


class NotASimulator(RuntimeError):
    """The robot description does not show a simulator and nothing else."""


def hardware_plugins(robot_description: str) -> List[str]:
    """Return the hardware plugin of every ``ros2_control`` block of a URDF."""
    try:
        root = ElementTree.fromstring(robot_description)
    except ElementTree.ParseError as err:
        raise NotASimulator(f'robot description is not valid XML: {err}') from err
    plugins = []
    for block in root.iter('ros2_control'):
        found = [(node.text or '').strip() for node in block.findall('./hardware/plugin')]
        plugins.extend(found if found else [''])
    return plugins


def require_simulator(robot_description: str) -> List[str]:
    """Return the plugins when all are simulators; raise ``NotASimulator`` otherwise."""
    plugins = hardware_plugins(robot_description)
    if not plugins:
        raise NotASimulator('robot description has no ros2_control hardware: cannot tell '
                            'that this is a simulation')
    other = sorted({p or '<none>' for p in plugins if p not in SIMULATOR_PLUGINS})
    if other:
        raise NotASimulator('simulation-only limit values refused: hardware plugin '
                            + ', '.join(other) + ' is not a simulator')
    return plugins
