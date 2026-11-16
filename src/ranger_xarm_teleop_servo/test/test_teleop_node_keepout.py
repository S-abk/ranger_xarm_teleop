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
"""Teleop node and the link keep-out (G-15): model, description checks, state, joint path."""

import math
import os
import sys

import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner
from ranger_xarm_teleop_interfaces.srv import GetJointPath
from ranger_xarm_teleop_servo import teleop_node
from ranger_xarm_teleop_servo.teleop_node import StartupRefused, TeleopNode
import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import Trigger

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unit'))
from keepout_robot import JOINTS, READY, resolve, robot_urdf  # noqa: E402, I100

LIMITS = {
    'max_offset_translation_m': 0.30,
    'max_offset_rotation_deg': 60.0,
    'view_cone_half_angle_deg': 45.0,
    'step_translation_limit_m': 0.030,
    'step_rotation_limit_deg': 10.0,
    'linear_speed_cap_m_s': 0.25,
    'angular_speed_cap_rad_s': 1.0,
    'max_lead_translation_m': 0.05,
    'max_lead_rotation_deg': 12.0,
    'max_translation_scale': 2.0,
    'link_keepout_margin_m': 0.025,
    'motion_enabled': True,
}


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros):
    made = []

    def make(urdf=None, **values):
        made.append(TeleopNode(node_name=f'teleop_keepout_{len(made)}',
                               parameter_overrides=[Parameter(k, value=v) for k, v in
                                                    {**LIMITS, **values}.items()]))
        made[-1]._resolver = lambda: resolve
        if urdf is not None:
            made[-1]._on_robot_description(String(data=urdf))
        return made[-1]
    yield make
    for n in made:
        n.close()
        n.destroy_node()


def _joints(node, q, gripper=0.02, stamp_ns=None):
    stamp = stamp_ns if stamp_ns is not None else node.get_clock().now().nanoseconds
    values = dict(zip(JOINTS, q))
    values['finger_left_joint'] = gripper
    node.joint_state = (node.mono(), stamp, values)


def test_margin_has_no_default(ros):
    values = {k: v for k, v in LIMITS.items() if k != 'link_keepout_margin_m'}
    with pytest.raises(StartupRefused, match='link_keepout_margin_m'):
        TeleopNode(node_name='teleop_keepout_nomargin',
                   parameter_overrides=[Parameter(k, value=v) for k, v in values.items()])
    with pytest.raises(StartupRefused, match='link_keepout_margin_m'):
        TeleopNode(node_name='teleop_keepout_badmargin',
                   parameter_overrides=[Parameter(k, value=v) for k, v in
                                        {**LIMITS, 'link_keepout_margin_m': -0.01}.items()])


def test_model_is_built_from_the_description_and_reported(node):
    n = node(robot_urdf())
    assert n.keepout is not None and n.keepout.margin == 0.025
    assert ('l2', 'screen') in n.keepout.pairs
    info = n._on_keepout_info(Trigger.Request(), Trigger.Response())
    assert info.success
    assert 'left out: l1 - xarm_link_base: adjacent' in info.message
    assert 'not covered' in info.message and 'wheel' in info.message
    assert n.keepout.fingerprint in info.message


def test_state_reports_the_clearance_when_idle(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    _joints(n, READY)
    clearance, pair = n.idle_clearance()
    assert 0.03 < clearance < 0.30 and pair[0] and pair[1]
    _joints(n, [0.0, -1.35, -0.3, 0.0, 1.6, 0.0])           # into the screen
    n._idle_clearance = None                                # (it is refreshed at 10 Hz)
    clearance, pair = n.idle_clearance()
    assert clearance < 0.0 and pair == ('l2', 'screen')
    n.joint_state = None
    n._idle_clearance = None
    assert math.isnan(n.idle_clearance()[0])


def test_a_missing_gripper_joint_state_makes_the_clearance_unknown(node):
    n = node(robot_urdf())
    n.joint_state = (n.mono(), n.get_clock().now().nanoseconds, dict(zip(JOINTS, READY)))
    n._idle_clearance = None
    assert math.isnan(n.idle_clearance()[0])
    with pytest.raises(teleop_node.ko.ModelError, match='finger'):
        n.keepout_extra()


def test_a_description_the_model_cannot_follow_refuses_the_start(node):
    broken = robot_urdf().replace('meshes/sensor.stl', 'meshes/sensor.dae')
    with pytest.raises(StartupRefused, match='only STL'):
        node(broken)


@pytest.mark.parametrize('later, fault', [
    (robot_urdf().replace('<robot name="test_robot">', '<robot name="test_robot"><!-- x -->'),
     False),                                                   # same geometry
    (robot_urdf(screen_x=-0.32), True),                        # the screen moved
    (robot_urdf(extra_tool=True), True),                       # a tool added
])
def test_a_later_description_with_other_geometry_disables_teleop(node, later, fault):
    n = node(robot_urdf())
    n._on_robot_description(String(data=later))
    assert bool(n.keepout_fault) is fault
    if fault:
        assert 'restarted' in n.keepout_fault
        assert math.isnan(n.arming_conditioning(READY))          # the arm request is refused
        ready, _ = n.solver_checks(0.0, None)
        assert not ready                                         # and the clutch releases


def test_servo_checking_another_description_disables_teleop(node):
    n = node(robot_urdf())
    n._compare_description(robot_urdf(), "MoveIt Servo's robot description")
    assert not n.keepout_fault
    n._compare_description(robot_urdf(screen_height=0.30), "MoveIt Servo's robot description")
    assert "MoveIt Servo's robot description does not match" in n.keepout_fault


def test_joint_path_is_timed_decimated_and_bounded(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    empty = n._on_joint_path(GetJointPath.Request(), GetJointPath.Response())
    assert not empty.path.points and 'no path' in empty.message
    q = list(READY)
    t0 = 1_000_000_000_000
    # What the cycle records while armed (fed directly here; the next test runs the cycle).
    for k in range(400):
        q[1] -= 0.001 if k % 2 else 0.003
        n.path.add((t0 + k * 20_000_000) * 1e-9, q)
    response = n._on_joint_path(GetJointPath.Request(), GetJointPath.Response())
    points = response.path.points
    assert response.path.joint_names == JOINTS
    assert 1 < len(points) <= 400
    steps = [max(abs(a - b) for a, b in zip(p.positions, r.positions))
             for p, r in zip(points[1:], points[:-1])]
    assert min(steps) >= 0.002 - 1e-12
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]
    assert times == sorted(times) and times[-1] == pytest.approx(399 * 0.02, abs=0.03)
    # 60 s kept at most.
    n.path.add((t0 * 1e-9) + 120.0, [0.0] * 6)
    assert len(n.path.points) == 1 and n.path.dropped_for_age


def test_the_cycle_records_the_path_while_teleop_is_armed(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    n.state = teleop_node.replace(n.state, want_armed=False)
    n.owner, n.owner_arrival = ArmOwner.OWNER_TELEOP, n.mono()
    q = list(READY)
    for k in range(5):
        q[0] += 0.01
        _joints(n, q)
        n._on_timer()
    assert not n.path.points                                # not armed
    n.state = teleop_node.replace(n.state, want_armed=True)
    for k in range(5):
        q[0] += 0.01
        _joints(n, q)
        n.owner_arrival = n.mono()
        n._on_timer()
    assert len(n.path.points) == 5


def test_arming_is_refused_without_the_joint_states_the_model_needs(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    n.joint_state = (n.mono(), n.get_clock().now().nanoseconds, dict(zip(JOINTS, READY)))
    assert math.isnan(n.arming_conditioning(READY))
    _joints(n, READY)
    assert not math.isnan(n.arming_conditioning(READY))


def test_teleop_waits_for_servos_robot_description_once_asked(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    n.servo_description_pending = True
    assert not n.solver_checks(0.0, None)[0]
    n.servo_description_pending = False
    assert n.solver_checks(0.0, None)[0] or n.arm is None
