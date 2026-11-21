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
    # Beyond the influence distance the value is a lower bound, without a pair.
    assert 0.03 < clearance < 0.30
    assert bool(pair[0]) == (clearance <= n.keepout.influence)
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


class _Request:
    """A request that is out: answered when the test says so, or never."""

    def __init__(self, names):
        self.names, self.callbacks, self.values = list(names), [], None

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def result(self):
        from types import SimpleNamespace as NS
        return NS(values=[NS(string_value=v) for v in self.values])

    def answer(self, *values):
        self.values = values
        for callback in self.callbacks:
            callback(self)


class _ParameterClient:
    """MoveIt Servo's parameter service as the node's client sees it; it answers nothing."""

    def __init__(self):
        self.out, self.removed, self.ready = [], [], True

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.out.append(_Request(request.names))
        return self.out[-1]

    def remove_pending_request(self, request):
        self.removed.append(request)


def _asking(node):
    """Give ``node`` a silent parameter client, a clock and a log; return (client, clock, log)."""
    from types import SimpleNamespace as NS
    client, clock, log = _ParameterClient(), {'t': 100.0}, []
    node.servo_description_client = client
    node.mono = lambda: clock['t']
    node.get_logger = lambda: NS(
        warning=lambda text, **_: log.append(text), info=lambda text, **_: log.append(text),
        error=lambda text, **_: log.append(text))
    return client, clock, log


def _later(node, clock, seconds):
    clock['t'] += seconds
    node.watch_servo_reads(clock['t'])


def test_teleop_waits_for_servos_robot_description_once_asked(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    client, clock, log = _asking(n)
    assert n.solver_checks(0.0, None)[0] or n.arm is None
    n.compare_servo_description()
    assert [r.names for r in client.out] == [['robot_description']]     # asked at once
    assert not n.solver_checks(0.0, None)[0]
    n.compare_servo_description()                                        # once only
    assert len(client.out) == 1
    client.out[0].answer(n.urdf)
    assert n.solver_checks(0.0, None)[0] or n.arm is None
    assert not n.keepout_fault and not client.removed


def test_an_unanswered_description_request_is_asked_again_and_says_so(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    client, clock, log = _asking(n)
    n.compare_servo_description()
    _later(n, clock, teleop_node.DESCRIPTION_TIMEOUT_S - 0.02)
    assert len(client.out) == 1 and not log and not client.removed       # still waiting
    _later(n, clock, 0.02)
    # Given up, dropped from the client (a late answer is then not delivered), said in the
    # log with its effect, and asked again in the same cycle.
    assert client.removed == [client.out[0]]
    assert len(client.out) == 2 and client.out[1].names == ['robot_description']
    assert len(log) == 1 and 'did not answer the request for its robot description' in log[0]
    assert 'asked 1 time)' in log[0] and 'asking again' in log[0]
    assert 'teleop stays disabled until it answers' in log[0]
    assert not n.solver_checks(0.0, None)[0]                             # still not ready
    client.out[1].answer(n.urdf)
    assert n.solver_checks(0.0, None)[0] or n.arm is None
    _later(n, clock, 10.0)
    assert len(client.out) == 2 and len(log) == 1                        # answered: asked no more


def test_a_description_request_is_asked_for_as_long_as_no_answer_comes(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    client, clock, log = _asking(n)
    n.compare_servo_description()
    for attempt in range(1, 31):
        _later(n, clock, teleop_node.DESCRIPTION_TIMEOUT_S)
        assert len(client.out) == attempt + 1 and len(log) == attempt
        assert f'asked {attempt} time' in log[-1]
        assert not n.solver_checks(0.0, None)[0]       # never ready on an unanswered request
    assert client.removed == client.out[:-1]


def test_a_late_answer_to_a_request_given_up_is_not_used(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    client, clock, log = _asking(n)
    n.compare_servo_description()
    _later(n, clock, teleop_node.DESCRIPTION_TIMEOUT_S)
    # rclpy does not deliver an answer to a request removed from the client. Even if it
    # did, an answer with another robot's geometry to the request given up changes nothing:
    client.out[0].answer(robot_urdf().replace('0.05', '0.06'))
    assert not n.keepout_fault and not n.solver_checks(0.0, None)[0]
    client.out[1].answer(n.urdf)
    assert not n.keepout_fault
    assert n.solver_checks(0.0, None)[0] or n.arm is None


def test_a_description_with_other_geometry_still_disables_teleop_after_a_retry(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    client, clock, log = _asking(n)
    n.compare_servo_description()
    _later(n, clock, teleop_node.DESCRIPTION_TIMEOUT_S)
    other = robot_urdf().replace('0.05', '0.06')
    assert other != n.urdf
    client.out[1].answer(other)
    assert 'does not match the link keep-out model' in n.keepout_fault
    assert not n.solver_checks(0.0, None)[0]


def test_the_description_request_waits_for_the_service_and_says_so(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    client, clock, log = _asking(n)
    client.ready = False
    n.compare_servo_description()
    assert not client.out and not n.solver_checks(0.0, None)[0]
    assert len(log) == 1 and "waiting for MoveIt Servo's parameter service" in log[0]
    client.ready = True
    _later(n, clock, 0.02)
    assert len(client.out) == 1


def test_the_rigid_pairs_request_is_asked_again_and_holds_nothing_up(node):
    n = node(robot_urdf())
    n._on_tf_static(teleop_node.TFMessage(transforms=[]))
    client, clock, log = _asking(n)
    ready_before = n.solver_checks(0.0, None)[0]
    n.check_servo_rigid_pairs()
    names = ['robot_description', 'robot_description_semantic', 'moveit_servo.move_group_name']
    assert [r.names for r in client.out] == [names]
    assert n.solver_checks(0.0, None)[0] == ready_before          # nothing waits on this one
    _later(n, clock, teleop_node.DESCRIPTION_TIMEOUT_S)
    assert [r.names for r in client.out] == [names, names] and client.removed == [client.out[0]]
    assert len(log) == 1 and 'its robot description and SRDF' in log[0]
    assert 'asking again' in log[0] and 'teleop stays disabled' not in log[0]
    client.out[1].answer('', '', '')                              # answered, though with nothing
    assert 'were not checked' in log[-1]
    _later(n, clock, 10.0)
    assert len(client.out) == 2


def test_the_clients_of_servos_parameter_service_keep_a_deep_reply_queue(node):
    from ranger_xarm_teleop_servo import calls
    n = node(robot_urdf())
    for client in (n.servo_params_client, n.servo_description_client):
        assert client.srv_name.endswith('/get_parameters')
        assert client.qos_profile.depth == calls.PARAMETER_REPLY_DEPTH >= 1000


def test_the_floor_monitor_reports_only_a_real_drop(node):
    from types import SimpleNamespace as NS
    n = node(robot_urdf())
    errors = []
    n.get_logger = lambda: NS(error=lambda text, **_: errors.append(text),
                              warning=lambda *a, **k: None, info=lambda *a, **k: None)

    def cycle(clearance, engaged_before=True):
        n.keepout_watch(NS(keepout_seconds=0.0005, keepout_cpu_seconds=0.0004,
                           keepout_clearance=clearance, keepout_pair=('l2', 'screen')),
                        engaged_before)
    cycle(0.052, engaged_before=False)          # first cycle of an engagement
    cycle(0.040)
    cycle(0.030)
    cycle(0.026)
    assert not errors
    cycle(0.020)                                # below the margin floor
    assert len(errors) == 1 and 'below the floor' in errors[0]
