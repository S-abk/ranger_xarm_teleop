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
"""The stack tests' launch-log reading and expected-node check on recorded lines (no ROS graph)."""

import pytest
from stack_helpers import (assert_stack_nodes, exited_node_patterns, ignoring,
                           launched_processes, MOCK_RECORD_NODES, MOCK_STACK_NODES)

# Lines as a mock stack's launch prints them (abridged from a real log).
STARTED = """[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [robot_state_publisher-1]: process started with pid [3184664]
[INFO] [ros2_control_node-2]: process started with pid [3184665]
[INFO] [spawner-3]: process started with pid [3184666]
[INFO] [move_group-5]: process started with pid [3184672]
[INFO] [rviz2-6]: process started with pid [3184673]
[INFO] [servo_node-9]: process started with pid [3184680]
[INFO] [arm_mux_node-10]: process started with pid [3184682]
[INFO] [teleop_node-11]: process started with pid [3184683]
[INFO] [gripper_relay_node-13]: process started with pid [3184689]
[rviz2-6] [ERROR] [1795010055.962286643] [rviz2]: rviz::RenderSystem: error creating render window
"""
RVIZ_DIED = ('[ERROR] [rviz2-6]: process has died [pid 3184673, exit code -6, cmd '
             '/opt/ros/jazzy/lib/rviz2/rviz2 -d planning.rviz].\n')
SPAWNER_DONE = '[INFO] [spawner-3]: process has finished cleanly [pid 3184666]\n'
RELAY_DIED = ('[ERROR] [gripper_relay_node-13]: process has died [pid 3184689, exit code -2, '
              'cmd gripper_relay_node].\n')

# The nodes of a running mock stack (a live listing; the numbered names change per run).
LIVE = ['/arm_mux', '/arm_mux_client', '/arm_mux_observer', '/controller_manager',
        '/gripper_relay', '/joint_state_broadcaster', '/move_group', '/move_group/moveit',
        '/move_group_private_110062872965072', '/moveit_174894846', '/moveit_844807136',
        '/moveit_simple_controller_manager', '/robot_state_publisher', '/servo_node',
        '/servo_node/moveit', '/servo_node_private_100409403087936', '/teleop_arm_controller',
        '/teleop_gripper_controller', '/teleop_servo', '/teleop_servo_watcher',
        '/teleop_test_probe', '/transform_listener_impl_641a02aa8e70',
        '/transform_listener_impl_72cdf4009a40', '/xarm6_traj_controller',
        '/xarm_gripper_traj_controller', '/xarm_mock_components_genericsystem',
        '/xarm_xarmgrippersystem']


class _Launch:
    """A launch whose output is fixed text."""

    def __init__(self, text):
        self.text = text

    def output(self):
        return self.text


class _Node:
    """A probe's node that sees a fixed list of nodes."""

    def __init__(self, names):
        self.names = names

    def get_name(self):
        return 'teleop_test_probe'

    def get_node_names_and_namespaces(self):
        return [(n.rsplit('/', 1)[1], n.rsplit('/', 1)[0] or '/') for n in self.names]


class _Probe:
    """A probe with that node."""

    def __init__(self, names):
        self.node = _Node(names)


def check(names, log, expected=MOCK_STACK_NODES):
    assert_stack_nodes(_Probe(names), _Launch(log), expected, settle=0.0)


def test_started_and_exited_processes_are_read_from_the_launch_log():
    assert launched_processes(_Launch(STARTED)) == (
        {'robot_state_publisher', 'ros2_control_node', 'spawner', 'move_group', 'rviz2',
         'servo_node', 'arm_mux_node', 'teleop_node', 'gripper_relay_node'}, set())
    started, exited = launched_processes(_Launch(STARTED + RVIZ_DIED + SPAWNER_DONE))
    assert exited == {'rviz2', 'spawner'} and exited < started


def test_only_nodes_of_known_exiting_processes_are_left_out_and_any_other_exit_fails():
    patterns = exited_node_patterns(_Launch(STARTED + RVIZ_DIED + SPAWNER_DONE))
    assert ignoring(['/rviz', '/rviz2', '/spawner_xarm6_traj_controller', '/move_group'],
                    patterns) == ['/rviz2', '/move_group']
    assert exited_node_patterns(_Launch(STARTED)) == ()
    with pytest.raises(AssertionError, match='gripper_relay_node'):
        exited_node_patterns(_Launch(STARTED + RVIZ_DIED + RELAY_DIED))


def test_the_live_stack_matches_its_expected_nodes_with_or_without_the_dead_rviz():
    check(LIVE, STARTED + RVIZ_DIED)
    check(LIVE + ['/rviz'], STARTED + RVIZ_DIED)          # its entry lingering until the lease
    check(LIVE + ['/rviz'], STARTED)                      # RViz still running: allowed
    check(LIVE + ['/episode_node', '/teleop_episode_recorder'], STARTED, MOCK_RECORD_NODES)


@pytest.mark.parametrize('names, message', [
    (LIVE + ['/unrelated_tool_node'], "nodes not of this stack: ['/unrelated_tool_node']"),
    (LIVE + ['/teleop_test_probe'], 'teleop_test_probe'),               # a second probe
    (LIVE + ['/arm_mux'], 'arm_mux'),                                    # a second stack
    ([n for n in LIVE if n != '/teleop_servo'], 'teleop_servo'),        # a missing node
    (LIVE + ['/moveit_123'], 'moveit_'),                                 # one numbered too many
    (LIVE + ['/episode_node'], 'episode_node'),                          # not in this stack
])
def test_anything_but_this_stack_and_its_probe_fails(names, message):
    with pytest.raises(AssertionError, match=message.replace('[', r'\[').replace(']', r'\]')):
        check(names, STARTED + RVIZ_DIED)
