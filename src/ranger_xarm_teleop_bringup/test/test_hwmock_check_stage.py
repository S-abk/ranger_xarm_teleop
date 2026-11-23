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
Procedure stage 2 on the hardware-shaped mock: the bring-up's check stage, nothing moving.

The mock (ranger_xarm_teleop_hardware hwmock.launch.py stage:=check) has the real check
stage's shape: joint states only, no trajectory controller, no gripper, no simulation clock.
The teleop stack attaches to it with motion disabled (teleop_hardware.launch.py stage:=check)
and must: start with motion disabled, leave the bring-up as it found it (INV-4), refuse arming
and any switch to teleop with an actionable reason (INV-2, INV-5), have no publisher on the
teleop controller's command topic, and show the preview of the target while the grip is held.
Nothing here starts a robot.
"""

import os
import threading
import time

from geometry_msgs.msg import PoseStamped
from hw_helpers import (check_report, publishers_of, start_attach, start_check, start_hwmock,
                        wait_for_controllers, write_hardware_limits)
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from ranger_xarm_teleop_interfaces.srv import SetArmOwner
from stack_helpers import (arm_by_replay, child_environment, graph_nodes, node_pids, prime,
                           Probe, replay, start_input)

PREVIEW = '/teleop/preview/target_pose'
TELEOP_COMMAND = '/teleop_arm_controller/joint_trajectory'


@pytest.fixture(scope='module')
def stage2(tmp_path_factory):
    logs = tmp_path_factory.mktemp('hwmock_check')
    env = child_environment(offset=23)
    bringup = start_hwmock(env, 'check', logs)
    probe = Probe(env)
    attach = None
    previews = []
    lock = threading.Lock()

    def on_preview(msg):
        with lock:
            previews.append((time.monotonic(), msg))

    probe.node.create_subscription(PoseStamped, PREVIEW, on_preview, 50)
    try:
        before = wait_for_controllers(probe, {'joint_state_broadcaster': 'active'})
        bringup_nodes = graph_nodes(probe)
        manager_pids = node_pids('ros2_control_node', environ=f'ROS_HOME={env["ROS_HOME"]}')
        limits = write_hardware_limits(logs)
        attach = start_attach(env, 'check', limits, logs, 'disabled')
        # The node builds its kinematic chain and keep-out model from the description first
        # (over a second): the stage is attached once the measured tool pose is valid.
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 90.0,
                       'the teleop state with a valid measured tool pose')
        start_input(probe, env, logs)
        yield {'probe': probe, 'env': env, 'logs': logs, 'attach': attach, 'bringup': bringup,
               'before': before, 'bringup_nodes': bringup_nodes, 'manager_pids': manager_pids,
               'limits': limits,
               'previews': previews, 'lock': lock}
    finally:
        probe.close()
        if attach is not None:
            attach.stop()
            print(attach.output()[-6000:])
        bringup.stop()
        print(bringup.output()[-3000:])


def test_the_attach_passed_its_preflight_and_starts_with_motion_disabled(stage2):
    probe, attach = stage2['probe'], stage2['attach']
    assert 'ATTACH PREFLIGHT OK (check stage)' in attach.output()
    state = probe.state()
    assert not state.motion_enabled and not state.armed
    assert state.hold_reason == TeleopState.HOLD_MOTION_DISABLED
    assert state.measured_tool_pose_valid            # real joint states from the bring-up


def test_the_attach_leaves_the_bring_up_as_it_found_it(stage2):
    """INV-4: no node of the bring-up stopped or restarted, controllers as before."""
    probe, env = stage2['probe'], stage2['env']
    assert probe.controllers() == stage2['before']
    now = set(graph_nodes(probe))
    assert set(stage2['bringup_nodes']) - {'/teleop_test_probe'} <= now
    assert node_pids('ros2_control_node',
                     environ=f'ROS_HOME={env["ROS_HOME"]}') == stage2['manager_pids']
    assert 'spawner' not in stage2['attach'].output().split('ATTACH PREFLIGHT OK')[-1]


def test_nothing_publishes_on_the_teleop_controller_or_the_joint_command(stage2):
    probe = stage2['probe']
    # The test's own probe has a publisher there (for other tests); nothing of the stack does.
    assert [n for n in publishers_of(probe, TELEOP_COMMAND) if n != 'teleop_test_probe'] == []
    assert publishers_of(probe, '/servo_node/delta_joint_cmds') == []
    assert publishers_of(probe, '/teleop/target_pose') == []
    assert publishers_of(probe, PREVIEW) == ['teleop_servo']
    names = {n for n, _ in probe.node.get_node_names_and_namespaces()}
    assert 'servo_node' not in names                 # MoveIt Servo is not started


def test_a_switch_to_teleop_is_refused_with_the_reason(stage2):
    probe = stage2['probe']
    # With motion disabled a request for teleop changes nothing (it is refused), so it may be
    # the harmless request that shows the mux has matched this client.
    prime(probe.set_owner_client, SetArmOwner.Request(owner=ArmOwner.OWNER_TELEOP), each=6.0)
    response = probe.set_owner(ArmOwner.OWNER_TELEOP)
    assert not response.success
    assert 'teleop motion is disabled' in response.detail
    assert 'motion_enabled:=true' in response.detail
    assert 'teleop_arm_controller' not in probe.controllers()


def test_arming_is_refused_at_the_press_with_the_reason(stage2):
    probe, env, logs = stage2['probe'], stage2['env'], stage2['logs']
    start, _ = arm_by_replay(probe, env, str(logs / 'arm_refused'), disarm=False)
    states = probe.states_since(start)
    assert states and not any(s.armed for s in states)
    probe.wait_for(lambda: 'arm refused: teleop motion is disabled' in stage2['attach'].output(),
                   10.0, 'the refusal in the teleop log')
    line = next(x for x in stage2['attach'].output().splitlines()
                if 'arm refused: teleop motion is disabled' in x)
    assert 'motion_enabled:=true' in line


def test_the_preview_follows_a_stroke_and_is_never_a_command(stage2):
    probe, env, logs = stage2['probe'], stage2['env'], stage2['logs']
    start = time.monotonic()
    process = replay(env, 'clutch_line', str(logs / 'preview_stroke.log'), start_delay=1.0)
    try:
        assert process.wait(60.0) == 0, process.output()
    finally:
        process.stop()
    with stage2['lock']:
        shown = [m for t, m in stage2['previews'] if t >= start]
    reasons = {s.preview_reason for s in probe.states_since(start)}
    assert len(shown) > 50, f'previews {len(shown)}; reasons {reasons}'
    xs = [m.pose.position.x for m in shown]
    ys = [m.pose.position.y for m in shown]
    zs = [m.pose.position.z for m in shown]
    travel = max(max(v) - min(v) for v in (xs, ys, zs))
    assert travel > 0.02, f'the preview did not move ({travel:.4f} m)'
    assert all(m.header.frame_id == 'xarm_link_base' for m in shown)
    assert probe.targets_since(start) == []
    assert all(not s.armed for s in probe.states_since(start))
    # Released at the end of the stroke: the reason says so.
    probe.wait_for(lambda: probe.state().preview_reason.startswith('grip released')
                   or probe.state().preview_reason.startswith('no fresh input'), 10.0,
                   'the preview to stop with the grip released')


def robot_workspace():
    """Return the path of the robot workspace that the version gate reads, or None."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.environ.get('RANGER_XARM_WS', ''),
                 os.path.join(os.environ.get('TELEOP_PROJECT', ''), '..', 'ranger_xarm_ws'),
                 os.path.join(here, '..', '..', '..', '..', '..', 'ranger_xarm_ws')):
        if path and os.path.isdir(os.path.join(path, 'src', 'xarm_ros2')):
            return os.path.abspath(path)
    return None


def test_the_check_tool_only_subscribes(stage2):
    """INV-3: while it runs, the check tool's node has no client, no service, no own publisher."""
    probe, env, logs = stage2['probe'], stage2['env'], stage2['logs']
    process, data_dir = start_check(env, 'idle', logs, 'endpoints', '--seconds', '8',
                                    '--limits', stage2['limits'])
    try:
        node = probe.node

        def subscriptions():
            try:
                return {n for n, _ in node.get_subscriber_names_and_types_by_node('hw_check', '/')}
            except Exception:  # noqa: B902 - the node is not on the graph yet
                return set()
        # Discovery shows a node before its endpoints: wait for the tool's own subscriptions.
        probe.wait_for(lambda: '/joint_states' in subscriptions(), 20.0,
                       "the check tool's subscriptions on the graph")
        time.sleep(1.0)
        pubs = {n for n, _ in node.get_publisher_names_and_types_by_node('hw_check', '/')}
        services = node.get_service_names_and_types_by_node('hw_check', '/')
        clients = node.get_client_names_and_types_by_node('hw_check', '/')
        subs = {n for n, _ in node.get_subscriber_names_and_types_by_node('hw_check', '/')}
    finally:
        code, verdicts, report, output = check_report(process, data_dir, 60.0)
    assert pubs <= {'/parameter_events'}, pubs          # rclpy's own; no /rosout either
    assert services == [] and clients == [], (services, clients)
    assert '/joint_states' in subs and '/controller_manager/activity' in subs
    assert output.splitlines()[0].startswith('read-only: this tool only subscribes')


def test_the_idle_check_passes_on_the_check_stage(stage2):
    """Procedure stage 2: the read-only checks that do not depend on the machine pass."""
    env, logs = stage2['env'], stage2['logs']
    extra = ['--seconds', '10', '--limits', stage2['limits'],
             '--expect-active', 'joint_state_broadcaster']
    ws = robot_workspace()
    if ws:
        extra += ['--workspace', ws]
    process, data_dir = start_check(env, 'idle', logs, 'idle', *extra)
    code, verdicts, report, output = check_report(process, data_dir, 90.0)
    for rule in ('joint_states rate', 'joint_states largest gap'):
        assert verdicts.get(rule) == 'PASS', (rule, output[-4000:])
    # The mock plugin's branch of the description writes no velocity_control parameter (the
    # real plugin's does): not judged here, said so, never passed silently.
    assert verdicts.get('velocity_control') == 'NOT JUDGED', output[-4000:]
    assert verdicts.get('joint_states effort') == 'INFO'
    # The stand-in for the vendor driver's /xarm/robot_states: err 0, mode 1, state 2.
    assert verdicts.get('arm state (/xarm/robot_states)') == 'PASS', output[-4000:]
    clock = [r for r in report['rules'] if 'clock' in r['rule']]
    assert clock and all(r['verdict'] == 'PASS' for r in clock), clock
    if ws:
        assert verdicts.get('xarm_ros2 joint-state change') == 'PASS', output[-3000:]
    # The machine's own rules (governor) may fail on a laptop; they are reported, not hidden.
    assert 'cpu governor' in verdicts
