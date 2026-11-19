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

import os

import pytest
import stack_helpers
from stack_helpers import (assert_stack_nodes, exited_node_patterns, ignoring, keep_logs,
                           launched_processes, MOCK_RECORD_NODES, MOCK_STACK_NODES,
                           press_frames, press_never_arrived, SERVO_FIXTURES)

# Lines as a mock stack's launch prints them (abridged from a real log; ``rviz:=false``).
STARTED = """[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [robot_state_publisher-1]: process started with pid [3184664]
[INFO] [ros2_control_node-2]: process started with pid [3184665]
[INFO] [spawner-3]: process started with pid [3184666]
[INFO] [move_group-5]: process started with pid [3184672]
[INFO] [servo_node-9]: process started with pid [3184680]
[INFO] [arm_mux_node-10]: process started with pid [3184682]
[INFO] [teleop_node-11]: process started with pid [3184683]
[INFO] [gripper_relay_node-13]: process started with pid [3184689]
"""
SPAWNER_DONE = '[INFO] [spawner-3]: process has finished cleanly [pid 3184666]\n'
RELAY_DIED = ('[ERROR] [gripper_relay_node-13]: process has died [pid 3184689, exit code -2, '
              'cmd gripper_relay_node].\n')
# RViz started after all (the argument not reaching the MoveIt launch), then aborting off screen.
RVIZ_STARTED = ('[INFO] [rviz2-6]: process started with pid [3184673]\n'
                '[rviz2-6] [ERROR] [1795010055.962286643] [rviz2]: rviz::RenderSystem: error '
                'creating render window\n')
RVIZ_DIED = ('[ERROR] [rviz2-6]: process has died [pid 3184673, exit code -6, cmd '
             '/opt/ros/jazzy/lib/rviz2/rviz2 -d planning.rviz].\n')

# The nodes of a running mock stack (a live listing; the numbered names change per run).
LIVE = ['/arm_mux', '/arm_mux_client', '/arm_mux_observer', '/controller_manager',
        '/gripper_relay', '/joint_state_broadcaster', '/move_group', '/move_group/moveit',
        '/move_group_private_110062872965072', '/moveit_174894846', '/moveit_844807136',
        '/moveit_simple_controller_manager', '/operator_feedback', '/robot_state_publisher',
        '/servo_node',
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
        {'robot_state_publisher', 'ros2_control_node', 'spawner', 'move_group', 'servo_node',
         'arm_mux_node', 'teleop_node', 'gripper_relay_node'}, set())
    started, exited = launched_processes(_Launch(STARTED + SPAWNER_DONE + RVIZ_STARTED
                                                 + RVIZ_DIED))
    assert exited == {'rviz2', 'spawner'} and exited < started


def test_only_nodes_of_finished_spawners_are_left_out_and_any_other_exit_fails():
    patterns = exited_node_patterns(_Launch(STARTED + SPAWNER_DONE))
    assert ignoring(['/spawner_xarm6_traj_controller', '/spawner', '/move_group'],
                    patterns) == ['/spawner', '/move_group']
    assert exited_node_patterns(_Launch(STARTED)) == ()
    with pytest.raises(AssertionError, match='gripper_relay_node'):
        exited_node_patterns(_Launch(STARTED + SPAWNER_DONE + RELAY_DIED))
    with pytest.raises(AssertionError, match='rviz2'):
        exited_node_patterns(_Launch(STARTED + SPAWNER_DONE + RVIZ_STARTED + RVIZ_DIED))


def test_the_live_stack_matches_its_expected_nodes_with_or_without_a_spawner():
    check(LIVE, STARTED + SPAWNER_DONE)
    check(LIVE + ['/spawner_xarm_gripper_traj_controller'], STARTED)      # still loading
    check(LIVE + ['/spawner_xarm_gripper_traj_controller'], STARTED + SPAWNER_DONE)  # lingering
    check(LIVE + ['/episode_node', '/teleop_episode_recorder'], STARTED, MOCK_RECORD_NODES)


@pytest.mark.parametrize('names, log, message', [
    pytest.param(LIVE + ['/unrelated_tool_node'], '',
                 "nodes not of this stack: ['/unrelated_tool_node']", id='unrelated_node'),
    pytest.param(LIVE + ['/teleop_test_probe'], '', 'teleop_test_probe', id='second_probe'),
    pytest.param(LIVE + ['/arm_mux'], '', 'arm_mux', id='second_stack'),
    pytest.param([n for n in LIVE if n != '/teleop_servo'], '', 'teleop_servo',
                 id='missing_node'),
    pytest.param(LIVE + ['/moveit_123'], '', 'moveit_', id='numbered_node_too_many'),
    pytest.param(LIVE + ['/episode_node'], '', 'episode_node', id='record_node'),
    pytest.param(LIVE + ['/rviz'], RVIZ_STARTED, "nodes not of this stack: ['/rviz']",
                 id='rviz_running'),
    pytest.param(LIVE + ['/rviz'], RVIZ_STARTED + RVIZ_DIED,
                 "launched processes have exited: ['rviz2']", id='rviz_exited'),
])
def test_anything_but_this_stack_and_its_probe_fails(names, log, message):
    with pytest.raises(AssertionError, match=message.replace('[', r'\[').replace(']', r'\]')):
        check(names, STARTED + SPAWNER_DONE + log)


# ---- the arm retry: only a press that demonstrably never arrived is tried again ----

def test_the_presses_of_arm_only_are_seq_27_to_80():
    # A (calibrate) 27-44, B (arm) 63-80.
    assert press_frames(os.path.join(SERVO_FIXTURES, 'arm_only.jsonl')) == (27, 80)


def test_a_press_is_lost_only_when_the_first_frame_is_well_after_the_last_press():
    # The previous stream's newest frame (683) first, then the replay's own frames. The last
    # pressed frame is 80; with the margin (10) only a first frame after 90 means "never arrived".
    assert press_never_arrived([], 80)                          # nothing at all
    assert press_never_arrived([683, 683, 683], 80)             # no frame of the replay
    assert press_never_arrived([683, 144, 145], 80)             # the Gazebo failure: seq 144
    assert press_never_arrived([683, 91, 92], 80)               # just beyond the margin
    assert not press_never_arrived([683, 90, 91], 80)           # within it: may have arrived
    assert not press_never_arrived([683, 60, 61], 80)           # A lost, B can arm (same session)
    assert not press_never_arrived([683, 0, 1, 2], 80)          # from the start
    assert not press_never_arrived([683, 30, 31], 80)           # mid-press: the level registers
    assert not press_never_arrived([269, 28, 29], 80)           # 27 may have come between cycles
    assert not press_never_arrived([683, 20, 150], 80)          # a gap later: still arrived
    assert not press_never_arrived([5, 6, 7], 80)               # no previous stream


# ---- logs kept when a test fails ----

def test_keep_logs_copies_what_the_module_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(stack_helpers, '_LOGS', [])
    launch = tmp_path / 'launch.log'
    launch.write_text('launch output\n')
    ros_logs = tmp_path / 'ros_log'
    (ros_logs / 'run1').mkdir(parents=True)
    (ros_logs / 'run1' / 'teleop_node.log').write_text('node log\n')
    other = tmp_path / 'other.log'
    other.write_text('another module\n')
    monkeypatch.setenv('PYTEST_CURRENT_TEST', 'test/test_mock_x.py::test_a (setup)')
    stack_helpers.register_log(str(launch))
    stack_helpers.register_log(str(ros_logs))
    monkeypatch.setenv('PYTEST_CURRENT_TEST', 'test/test_mock_y.py::test_b (setup)')
    stack_helpers.register_log(str(other))
    target = keep_logs('test/test_mock_x.py', 'test_a', str(tmp_path / 'kept'))
    kept = sorted(os.path.relpath(os.path.join(d, f), target)
                  for d, _, files in os.walk(target) for f in files)
    assert kept == ['00_launch.log', os.path.join('01_ros_log', 'run1', 'teleop_node.log')]
    assert os.path.basename(target).endswith('test_mock_x.py__test_a')
