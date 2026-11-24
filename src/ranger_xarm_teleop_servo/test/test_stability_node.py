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
The stability monitor node, its callbacks driven directly with real messages on a fake clock.

The robot description is the simulator's wheeled one, made by xacro from the robot workspace
(no simulator runs). The figure itself is tested without ROS (test/unit/test_stability_*.py);
here: what the node publishes and subscribes (INV-1), one unavailable state per missing input
(INV-4), the IMU's orientation never read, the simulator check, and the time of a cycle.
"""

import math
import os
import subprocess
import time

from ament_index_python.packages import get_package_share_directory
from control_msgs.msg import DynamicJointState, InterfaceValue
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import pytest
from ranger_xarm_teleop_interfaces.msg import StabilityState
from ranger_xarm_teleop_servo import stability_node
from ranger_xarm_teleop_servo.stability_node import StabilityNode, StartupRefused
import rclpy
from rclpy.parameter import Parameter
import rclpy.time
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import String
import yaml

WHEELS = ['front_left_wheel_link', 'front_right_wheel_link', 'rear_left_wheel_link',
          'rear_right_wheel_link']
ARM = [f'xarm_joint{i}' for i in range(1, 7)]
READY = [0.0, -0.6, -0.7, 0.0, 1.3, 0.0]
STEER = [w.replace('_wheel_link', '_steer_joint') for w in WHEELS]
REAL_DESCRIPTION = ('<robot name="r"><ros2_control name="a" type="system"><hardware><plugin>'
                    'some_vendor/RealArm</plugin></hardware></ros2_control></robot>')
MASS = {'stability_mass_model': 1, 'simulation_only': True,
        'default_uncertainty': {'mass_fraction': 0.1, 'com_m': 0.02}}
PARAMS = {'cutoff_hz': 15.0, 'tilt_tau_s': 3.0, 'joint_timeout_s': 0.5,
          'odometry_timeout_s': 0.5, 'imu_timeout_s': 0.5, 'stop_decel_m_s2': 1.0,
          'stop_turn_decel_rad_s2': 3.5, 'wheel_links': WHEELS, 'tilt_topic': '/imu',
          'imu_frame': 'imu'}
G = 9.80665


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope='module')
def wheeled_description():
    xacro = os.path.join(get_package_share_directory('ranger_xarm_description'), 'urdf',
                         'ranger_xarm.urdf.xacro')
    args = ['xarm_ros2_control_plugin:=gz_ros2_control/GazeboSimSystem',
            'xarm_load_gazebo_plugin:=true', 'xarm_ros2_control_params:=/dev/null',
            'add_gripper:=true', 'use_wheels:=true', 'fix_base_to_world:=false',
            'gz_sensors:=true', 'wheels_command_interface:=velocity']
    return subprocess.run(['xacro', xacro] + args, capture_output=True, text=True,
                          check=True).stdout


@pytest.fixture
def mass_file(tmp_path):
    path = tmp_path / 'mass.yaml'
    path.write_text(yaml.safe_dump(MASS))
    return str(path)


class FakeTime:

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def time(self):
        return 1.7e9 + self.t


class Rig:
    """A StabilityNode on a fake clock with its state captured."""

    def __init__(self, monkeypatch, description, mass_file, **params):
        self.clock = FakeTime()
        monkeypatch.setattr(stability_node, 'time', self.clock)
        values = {**PARAMS, 'mass_file': mass_file, **params}
        values = {k: v for k, v in values.items() if v is not None}
        self.node = StabilityNode(node_name='stability_under_test', parameter_overrides=[
            Parameter(k, value=v) for k, v in values.items()])
        self.states = []
        self.node.state_pub.publish = self.states.append
        self.stamp_s = 10.0
        if description is not None:
            self.node._on_description(String(data=description))
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'base_link', 'imu'
        tf.transform.translation.x, tf.transform.translation.z = -0.375, 1.146
        tf.transform.rotation.w = 1.0
        self.node.tf_buffer.set_transform_static(tf, 'test')

    def stamp(self):
        return rclpy.time.Time(nanoseconds=int(self.stamp_s * 1e9)).to_msg()

    def step(self, joints=True, steering=True, odom=True, imu=True, v=(0.0, 0.0, 0.0),
             f=(0.0, 0.0, G), orientation=(0.0, 0.0, 0.0, 1.0), q=READY):
        if joints:
            js = JointState()
            js.header.stamp = self.stamp()
            js.name = ARM + ['xarm_drive_joint']
            js.position = list(q) + [0.0]
            self.node._on_joints(js)
        if steering:
            d = DynamicJointState()
            d.header.stamp = self.stamp()
            d.joint_names = STEER
            d.interface_values = [InterfaceValue(interface_names=['position', 'velocity'],
                                                 values=[0.0, 0.0]) for _ in STEER]
            self.node._on_dynamic(d)
        if odom:
            o = Odometry()
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z = v
            self.node._on_odom(o)
        if imu:
            m = Imu()
            m.header.stamp = self.stamp()
            m.header.frame_id = 'imu'
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = f
            (m.orientation.x, m.orientation.y, m.orientation.z,
             m.orientation.w) = orientation
            self.node._on_imu(m)
        self.node._cycle()
        self.clock.t += 0.02
        self.stamp_s += 0.02

    def run(self, seconds, **kw):
        for _ in range(int(round(seconds / 0.02))):
            self.step(**kw)
        return self.states[-1]


@pytest.fixture
def rig(ros, monkeypatch, wheeled_description, mass_file):
    made = []

    def make(description='default', mass=None, **params):
        r = Rig(monkeypatch, wheeled_description if description == 'default' else description,
                mass if mass is not None else mass_file, **params)
        made.append(r)
        return r
    yield make
    for r in made:
        r.node.destroy_node()


@pytest.mark.parametrize('missing', ['cutoff_hz', 'tilt_tau_s', 'joint_timeout_s',
                                     'odometry_timeout_s', 'imu_timeout_s', 'stop_decel_m_s2',
                                     'stop_turn_decel_rad_s2', 'wheel_links', 'mass_file'])
def test_the_node_refuses_to_start_without_a_required_value(ros, mass_file, missing):
    values = {k: v for k, v in {**PARAMS, 'mass_file': mass_file}.items() if k != missing}
    with pytest.raises(StartupRefused, match=missing):
        StabilityNode(node_name='stability_refused', parameter_overrides=[
            Parameter(k, value=v) for k, v in values.items()])


def test_without_a_tilt_source_the_slope_bound_is_required(ros, mass_file):
    values = {**PARAMS, 'mass_file': mass_file, 'tilt_topic': ''}
    with pytest.raises(StartupRefused, match='assumed_slope_deg'):
        StabilityNode(node_name='stability_refused', parameter_overrides=[
            Parameter(k, value=v) for k, v in values.items()])


def test_the_node_publishes_its_state_only_and_asks_nothing(rig):
    """INV-1: one publisher, its own state; no service client, no action; the inputs listed."""
    r = rig()
    topics = {pub.topic_name for pub in r.node.publishers}
    assert topics - {'/rosout', '/parameter_events'} == {'/teleop/stability/state'}
    assert not list(r.node.clients)
    own = {s.srv_name for s in r.node.services}
    assert all(name.split('/')[-1] in ('describe_parameters', 'get_parameters',
                                       'get_parameter_types', 'list_parameters',
                                       'set_parameters', 'set_parameters_atomically',
                                       'get_type_description') for name in own), own
    assert not [w for w in r.node.waitables if 'Action' in type(w).__name__]
    subs = {s.topic_name for s in r.node.subscriptions}
    assert subs <= {'/robot_description', '/joint_states', '/dynamic_joint_states', '/odom',
                    '/imu', '/tf', '/tf_static', '/parameter_events'}


def test_a_figure_at_rest_from_the_wheeled_description(rig):
    r = rig()
    s = r.run(4.0)
    assert s.available and s.reason == StabilityState.REASON_OK, s.reason_text
    assert s.tilt_source == StabilityState.TILT_MEASURED
    assert s.margin_m == pytest.approx(0.1835, abs=0.003)          # stage 1: 0.1837 static
    assert s.edge == StabilityState.EDGE_LEFT
    assert s.margin_pessimistic_m < s.margin_m
    assert s.stop_decel_m_s2 == 1.0 and s.stop_turn_decel_rad_s2 == 3.5
    assert len(s.model_sha256) == 64 and 'force on the tool' in s.note
    assert s.header.frame_id == 'base_link' and s.at_rest


def test_the_imus_orientation_is_never_read(rig):
    a = rig().run(4.0)
    b = rig().run(4.0, orientation=(math.nan, 5.0, -3.0, math.nan))
    assert b.available and b.margin_m == a.margin_m and b.roll_deg == a.roll_deg


def test_a_robot_that_is_not_a_simulator_gets_no_figure(rig):
    s = rig(description=REAL_DESCRIPTION).run(1.0)
    assert not s.available and s.reason == StabilityState.REASON_NOT_SIMULATION
    assert math.isnan(s.margin_m)


@pytest.mark.parametrize('kw, reason', [
    ({'joints': False}, StabilityState.REASON_NO_JOINT_STATES),
    ({'steering': False}, StabilityState.REASON_NO_JOINT_STATES),
    ({'odom': False}, StabilityState.REASON_NO_ODOMETRY),
    ({'imu': False}, StabilityState.REASON_NO_TILT),
])
def test_one_missing_input_is_unavailable_with_its_reason(rig, kw, reason):
    s = rig().run(4.0, **kw)
    assert not s.available and s.reason == reason and math.isnan(s.margin_m)
    assert s.reason_text


def test_no_tilt_says_how_the_simulator_bridges_its_imu(rig):
    s = rig().run(4.0, imu=False)
    assert 'sensors:=true' in s.reason_text


@pytest.mark.parametrize('kw, reason', [
    ({'joints': False}, StabilityState.REASON_JOINT_STATES_STALE),
    ({'odom': False}, StabilityState.REASON_ODOMETRY_STALE),
    ({'imu': False}, StabilityState.REASON_NO_TILT),
])
def test_an_input_going_stale_is_unavailable(rig, kw, reason):
    r = rig()
    assert r.run(4.0).available
    s = r.run(0.6, **kw)
    assert not s.available and s.reason == reason


def test_no_description_and_an_unreadable_mass_file(rig, tmp_path):
    assert rig(description=None).run(0.5).reason == StabilityState.REASON_NO_DESCRIPTION
    s = rig(mass=str(tmp_path / 'nowhere.yaml')).run(0.5)
    assert s.reason == StabilityState.REASON_MASS_FILE_REFUSED and 'nowhere' in s.reason_text
    bad = tmp_path / 'bad.yaml'
    bad.write_text(yaml.safe_dump({**MASS, 'template': True}))
    s = rig(mass=str(bad)).run(0.5)
    assert s.reason == StabilityState.REASON_MASS_FILE_REFUSED and 'template' in s.reason_text


def test_without_a_tilt_source_a_figure_only_at_rest(rig):
    r = rig(tilt_topic='', assumed_slope_deg=5.0)
    s = r.run(1.0, imu=False, v=(0.2, 0.0, 0.0))
    assert s.reason == StabilityState.REASON_MOVING_WITHOUT_TILT and not s.available
    s = r.run(1.0, imu=False)
    assert s.available and s.tilt_source == StabilityState.TILT_LEVEL_ASSUMED
    assert s.margin_pessimistic_m < s.margin_m


def test_a_cycle_takes_little_time(rig):
    r = rig()
    r.run(1.0)
    times = []
    for _ in range(200):
        start = time.perf_counter()
        r.step()
        times.append(time.perf_counter() - start)
    times.sort()
    p99 = times[int(0.99 * len(times))] * 1e3
    print(f'stability node cycle with every input: p50 {times[100] * 1e3:.2f} ms, '
          f'p99 {p99:.2f} ms, max {times[-1] * 1e3:.2f} ms')
    assert p99 < 15.0            # the cycle is 20 ms
