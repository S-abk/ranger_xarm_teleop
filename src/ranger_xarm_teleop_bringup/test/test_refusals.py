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
INV-10: a PENDING value has no default, and simulation-only values stay in simulation.

Real processes on their own ROS domain: the teleop node must exit on its own with a
non-zero status, and the core launch file must refuse the simulation-only limits file
unless it is told it runs a simulation. Nothing here starts a robot or a simulator.
"""

import glob
import os
import re
import time

from ament_index_python.packages import get_package_share_directory
import pytest
from ranger_xarm_teleop_bringup.limits import (check_extra_params_file, check_limits_file,
                                               load_limits, RESERVED, SimulationOnlyRefused)
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from stack_helpers import child_environment, Process
from std_msgs.msg import String
import yaml

SHARE = get_package_share_directory('ranger_xarm_teleop_bringup')
SIM_LIMITS = os.path.join(SHARE, 'config', 'SIMULATION_ONLY_limits.yaml')
REQUIRED = ['max_offset_translation_m', 'max_offset_rotation_deg', 'view_cone_half_angle_deg',
            'step_translation_limit_m', 'step_rotation_limit_deg', 'linear_speed_cap_m_s',
            'angular_speed_cap_rad_s', 'max_lead_translation_m', 'max_lead_rotation_deg']


def teleop_node(env, log, *ros_args):
    return Process(['ros2', 'run', 'ranger_xarm_teleop_servo', 'teleop_node', '--ros-args',
                    *ros_args], env, log)


def urdf(plugin):
    return ('<robot name="r"><link name="base"/><ros2_control name="arm" type="system">'
            f'<hardware><plugin>{plugin}</plugin></hardware></ros2_control></robot>')


@pytest.fixture
def env():
    return child_environment(offset=2)


def test_simulation_only_file_says_so_and_holds_every_pending_value():
    assert 'SIMULATION_ONLY' in os.path.basename(SIM_LIMITS)
    with open(SIM_LIMITS, encoding='utf-8') as handle:
        assert handle.readline().startswith('# SIMULATION ONLY')
    values = load_limits(SIM_LIMITS)
    assert values['limits_are_simulation_only'] is True
    for name in REQUIRED:
        assert isinstance(values[name], float) and values[name] > 0.0
    assert 'motion_enabled' not in values


def test_node_exits_without_the_pending_values(env, tmp_path):
    node = teleop_node(env, str(tmp_path / 'node.log'))
    try:
        assert node.wait(30.0) == 2
        output = node.output()
        assert 'refusing to start' in output
        for name in REQUIRED:
            assert name in output
    finally:
        node.stop()


@pytest.mark.parametrize('missing', ['view_cone_half_angle_deg', 'linear_speed_cap_m_s',
                                     'max_lead_translation_m', 'max_lead_rotation_deg'])
def test_node_exits_when_one_pending_value_is_missing(env, tmp_path, missing):
    values = {k: v for k, v in load_limits(SIM_LIMITS).items() if k != missing}
    path = tmp_path / 'limits.yaml'
    path.write_text(yaml.safe_dump({'/**': {'ros__parameters': values}}), encoding='utf-8')
    node = teleop_node(env, str(tmp_path / 'node.log'), '--params-file', str(path))
    try:
        assert node.wait(30.0) == 2
        assert missing in node.output()
    finally:
        node.stop()


def test_simulation_only_values_are_refused_on_a_description_that_is_not_a_simulator(
        env, tmp_path):
    """A hardware path that loaded the file anyway: the node reads the description and exits."""
    context = rclpy.Context()
    rclpy.init(context=context, domain_id=int(env['ROS_DOMAIN_ID']))
    publisher_node = rclpy.create_node('fake_robot_description', context=context)
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = publisher_node.create_publisher(String, '/robot_description', latched)
    publisher.publish(String(data=urdf('example_vendor/RealArmHardware')))
    node = teleop_node(env, str(tmp_path / 'node.log'), '--params-file', SIM_LIMITS)
    try:
        code = None
        for _ in range(300):
            time.sleep(0.1)         # the latched description needs no spinning
            code = node.wait(0.0)
            if code is not None:
                break
        assert code == 2, node.output()
        output = node.output()
        assert 'simulation-only limit values refused' in output
        assert 'example_vendor/RealArmHardware' in output
    finally:
        node.stop()
        publisher_node.destroy_node()
        rclpy.shutdown(context=context)


def test_simulation_only_values_without_any_robot_description_end_the_node(env, tmp_path):
    node = teleop_node(env, str(tmp_path / 'node.log'), '--params-file', SIM_LIMITS,
                       '-p', 'robot_description_timeout_s:=2.0')
    try:
        assert node.wait(40.0) == 2
        assert 'no robot description' in node.output()
    finally:
        node.stop()


def test_core_launch_refuses_the_simulation_only_file_outside_simulation(env, tmp_path):
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_core.launch.py',
                      f'limits_file:={SIM_LIMITS}'], env, str(tmp_path / 'launch.log'))
    try:
        code = launch.wait(60.0)
        output = launch.output()
        assert code is not None and code != 0, output
        assert 'simulation-only limit values' in output
        assert 'process started' not in output            # no node was started
    finally:
        launch.stop()


def test_core_launch_needs_a_limits_file(env, tmp_path):
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_core.launch.py'],
                     env, str(tmp_path / 'launch.log'))
    try:
        code = launch.wait(60.0)
        assert code is not None and code != 0
        assert 'limits_file' in launch.output()
        assert 'process started' not in launch.output()
    finally:
        launch.stop()


def test_limits_file_check():
    with pytest.raises(SimulationOnlyRefused):
        check_limits_file(SIM_LIMITS, simulation=False)
    assert check_limits_file(SIM_LIMITS, simulation=True)['linear_speed_cap_m_s'] == 0.25


def test_limits_file_without_speed_caps_is_refused(tmp_path):
    path = tmp_path / 'limits.yaml'
    path.write_text(yaml.safe_dump({'/**': {'ros__parameters': {
        'max_offset_translation_m': 0.1}}}), encoding='utf-8')
    with pytest.raises(ValueError, match='linear_speed_cap_m_s'):
        check_limits_file(str(path), simulation=False)


def test_only_the_mock_and_simulator_wrappers_enable_motion_and_simulation():
    """G-1: the core defaults are false; only the two wrappers pass true."""
    sources = {os.path.basename(p): open(p, encoding='utf-8').read()
               for p in glob.glob(os.path.join(SHARE, 'launch', '*.launch.py'))}
    assert set(sources) == {'teleop_core.launch.py', 'teleop_mock.launch.py',
                            'teleop_sim.launch.py'}
    core = sources['teleop_core.launch.py']
    assert re.search(r"'simulation', default_value='false'", core)
    assert re.search(r"'motion_enabled', default_value='false'", core)
    assert 'SIMULATION_ONLY' not in core
    for name in ('teleop_mock.launch.py', 'teleop_sim.launch.py'):
        assert "'simulation': 'true'" in sources[name]
        assert 'SIMULATION_ONLY_limits.yaml' in sources[name]
        assert re.search(r"'motion_enabled', default_value='true'", sources[name])


@pytest.mark.parametrize('name', RESERVED)
def test_extra_parameter_file_may_not_set_a_limit_or_a_guard(tmp_path, name):
    path = tmp_path / 'extra.yaml'
    path.write_text(yaml.safe_dump({'/**': {'ros__parameters': {name: False}}}),
                    encoding='utf-8')
    with pytest.raises(ValueError, match=name):
        check_extra_params_file(str(path))


def test_extra_parameter_file_with_other_settings_is_accepted(tmp_path):
    path = tmp_path / 'extra.yaml'
    path.write_text(yaml.safe_dump({'/**': {'ros__parameters': {
        'keep_out_planes': [0.0, 0.0, 1.0, 0.1], 'wrist_frame': 'xarm_link6'}}}),
        encoding='utf-8')
    check_extra_params_file(str(path))


def test_core_launch_refuses_an_extra_file_that_overrides_the_guard(env, tmp_path):
    """The simulation-only flag cannot be switched off by a second parameter file."""
    path = tmp_path / 'extra.yaml'
    path.write_text(yaml.safe_dump({'/**': {'ros__parameters': {
        'limits_are_simulation_only': False}}}), encoding='utf-8')
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_core.launch.py',
                      f'limits_file:={SIM_LIMITS}', 'simulation:=true',
                      f'teleop_params_file:={path}'], env, str(tmp_path / 'launch.log'))
    try:
        code = launch.wait(60.0)
        output = launch.output()
        assert code is not None and code != 0, output
        assert 'limits_are_simulation_only' in output
        assert 'process started' not in output
    finally:
        launch.stop()


def test_move_arm_tool_sends_nothing_on_a_description_that_is_not_a_simulator(env, tmp_path):
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tools',
                        'move_arm.py')
    if not os.path.exists(tool):
        pytest.skip('tools/ is not next to the source tree')
    context = rclpy.Context()
    rclpy.init(context=context, domain_id=int(env['ROS_DOMAIN_ID']))
    publisher_node = rclpy.create_node('fake_robot_description', context=context)
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = publisher_node.create_publisher(String, '/robot_description', latched)
    publisher.publish(String(data=urdf('example_vendor/RealArmHardware')))
    process = Process(['python3', tool], env, str(tmp_path / 'move_arm.log'))
    try:
        assert process.wait(40.0) == 3, process.output()
        assert 'REFUSED, nothing sent' in process.output()
    finally:
        process.stop()
        publisher_node.destroy_node()
        rclpy.shutdown(context=context)


@pytest.mark.parametrize('document', [
    {'teleop_servo': {'ros__parameters': {'limits_are_simulation_only': False}}},
    {'/**': {'teleop_servo': {'ros__parameters': {'motion_enabled': True}}}},
    {'/ns': {'node': {'ros__parameters': {'view_cone_half_angle_deg': 180.0}}}},
])
def test_extra_parameter_file_is_checked_under_any_node_or_namespace_key(tmp_path, document):
    path = tmp_path / 'extra.yaml'
    path.write_text(yaml.safe_dump(document), encoding='utf-8')
    with pytest.raises(ValueError):
        check_extra_params_file(str(path))
