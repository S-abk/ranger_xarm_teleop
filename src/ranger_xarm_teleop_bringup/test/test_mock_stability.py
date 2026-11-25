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
The stability monitor on the mock stack: information only, harmless when killed.

Started with ``stability:=true base:=true`` and the stand-in base. The mock stack's robot
description has no wheels, so the monitor says "unavailable" and why; it must still be
harmless (INV-1: it publishes its own state only, asks nothing, and nothing reads it; INV-2:
killed mid-session, driving and the arm's teleop state go on as before). The launch refuses it
on the hardware path and without its files. The last test kills the monitor, so it runs last.
"""

import os
import signal
import sys
import threading
import time

from ament_index_python.packages import get_package_share_directory
from base_helpers import after_stop, BaseWatcher, phase, play
from hw_helpers import write_hardware_limits
import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner, StabilityState, TeleopState
from rclpy.action import get_action_client_names_and_types_by_node
from reset_helpers import node_pid
from stack_helpers import (assert_stack_nodes, child_environment, graph_nodes,
                           MOCK_STACK_NODES, Probe, Process, start_input)

FORWARD = (0.0, -1.0)           # left stick pushed up: robot forward
HERE = os.path.dirname(os.path.abspath(__file__))
OFFSET = 29


class StabilityWatcher:
    """The monitor's state and the arm's teleop state, each with its monotonic arrival."""

    def __init__(self, probe):
        self.lock = threading.Lock()
        self.states, self.teleop = [], []
        probe.node.create_subscription(StabilityState, '/teleop/stability/state',
                                       self._add(self.states), 10)
        probe.node.create_subscription(TeleopState, '/teleop/state', self._add(self.teleop), 10)

    def _add(self, store):
        def callback(msg):
            with self.lock:
                store.append((time.monotonic(), msg))
        return callback

    def since(self, store, t):
        with self.lock:
            return [m for m in store if m[0] >= t]


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_stability')
    env = child_environment(offset=OFFSET)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                      'rviz:=false', 'base:=true', 'stability:=true'], env,
                     str(logs / 'launch.log'))
    base = Process([sys.executable, os.path.join(HERE, 'fake_base.py')], env,
                   str(logs / 'fake_base.log'))
    probe = Probe(env)
    results = []
    try:
        watch = BaseWatcher(probe)
        stab = StabilityWatcher(probe)
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the command mux with MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None, 60.0, 'the teleop state')
        probe.wait_for(lambda: watch.last() is not None and watch.last().enabled
                       and watch.last().at_rest, 60.0, 'the base node, enabled, base at rest')
        probe.wait_for(lambda: len(stab.states) > 10, 60.0, 'the stability state')
        start_input(probe, env, logs)
        yield {'probe': probe, 'env': env, 'logs': logs, 'launch': launch, 'watch': watch,
               'stab': stab, 'results': results}
    finally:
        probe.close()
        base.stop()
        launch.stop()
        for line in results:
            print(line)


def drive_and_release(stack, name):
    """Drive forward 1.5 s and release; assert one zero then silence; return the end time."""
    start, _, _ = play(stack, name, [phase(0.4), phase(1.5, grip=True, left=FORWARD),
                                     phase(2.0)])
    cmds = stack['watch'].commands(start)
    assert any(vx > 0.1 for _, vx, _, _ in cmds), 'the base was never commanded forward'
    tail = after_stop(cmds)
    assert len(tail) == 1 and tail[0][1:] == (0.0, 0.0, 0.0), tail[:5]
    return time.monotonic()


def rate(samples, t0, t1):
    n = sum(1 for t, _ in samples if t0 <= t <= t1)
    return n / max(1e-9, t1 - t0)


def test_the_stack_runs_the_stability_node(stack):
    expected = {**MOCK_STACK_NODES, '/base_node': 1, '/fake_base': 1, '/stability_node': 1}
    assert_stack_nodes(stack['probe'], stack['launch'], expected)
    stack['nodes'] = graph_nodes(stack['probe'])


def test_on_the_mock_the_monitor_is_unavailable_and_says_why(stack):
    t = time.monotonic()
    time.sleep(1.0)
    states = stack['stab'].since(stack['stab'].states, t)
    assert len(states) >= 30, f'{len(states)} states in 1 s'          # 50 Hz, with slack
    last = states[-1][1]
    assert not last.available
    assert last.reason == StabilityState.REASON_DESCRIPTION_UNUSABLE, last.reason_text
    assert 'wheel' in last.reason_text
    assert 'force on the tool' in last.note
    stack['results'].append(f'mock: {len(states)} states/s, "{last.reason_text}"')


def test_the_monitor_publishes_its_state_only_and_nothing_reads_it(stack):
    """INV-1 on the live graph: one publisher, its own topics only, no client, no action."""
    node = stack['probe'].node
    pubs = {i.node_name for i in node.get_publishers_info_by_topic('/teleop/stability/state')}
    assert pubs == {'stability_node'}
    readers = {i.node_name for i in
               node.get_subscriptions_info_by_topic('/teleop/stability/state')}
    assert readers <= {'teleop_test_probe', 'recorder'}, readers
    own = {name for name, _ in node.get_publisher_names_and_types_by_node('stability_node', '/')}
    assert own - {'/rosout', '/parameter_events'} == {'/teleop/stability/state'}
    assert node.get_client_names_and_types_by_node('stability_node', '/') == []
    assert get_action_client_names_and_types_by_node(node, 'stability_node', '/') == []
    # Nothing else in the stack publishes or reads anything under /teleop/stability.
    for topic, _ in node.get_topic_names_and_types():
        if topic.startswith('/teleop/stability'):
            assert topic == '/teleop/stability/state'


def test_the_launch_refuses_the_monitor_on_the_hardware_path(tmp_path):
    env = child_environment(offset=OFFSET)
    limits = write_hardware_limits(tmp_path)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_core.launch.py',
                      f'limits_file:={limits}', 'stability:=true'], env,
                     str(tmp_path / 'launch.log'))
    try:
        code = launch.wait(60.0)
        output = launch.output()
        assert code is not None and code != 0, output
        assert 'stability:=true is refused outside simulation' in output
        assert 'process started' not in output
    finally:
        launch.stop()


def test_the_core_launch_needs_the_monitors_files(tmp_path):
    env = child_environment(offset=OFFSET)
    sim_limits = os.path.join(get_package_share_directory('ranger_xarm_teleop_bringup'),
                              'config', 'SIMULATION_ONLY_limits.yaml')
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_core.launch.py',
                      f'limits_file:={sim_limits}', 'simulation:=true', 'stability:=true'],
                     env, str(tmp_path / 'launch.log'))
    try:
        code = launch.wait(60.0)
        output = launch.output()
        assert code is not None and code != 0, output
        assert 'stability_params_file and stability_mass_file' in output
        assert 'process started' not in output
    finally:
        launch.stop()


def test_killed_the_monitor_changes_nothing_else(stack):
    """INV-2: SIGKILL mid-session; the base drives and stops as before, the arm's state goes on."""
    probe, stab = stack['probe'], stack['stab']
    stack.setdefault('nodes', graph_nodes(probe))
    t0 = time.monotonic()
    drive_and_release(stack, 'before_kill')
    t1 = time.monotonic()
    before = rate(stab.teleop, t0, t1)
    os.kill(node_pid(stack, 'stability_node'), signal.SIGKILL)
    killed = time.monotonic()
    probe.wait_for(lambda: not stab.since(stab.states, killed + 0.5)
                   and time.monotonic() > killed + 1.0, 10.0, 'the monitor to be silent')
    t2 = time.monotonic()
    drive_and_release(stack, 'after_kill')
    t3 = time.monotonic()
    after = rate(stab.teleop, t2, t3)
    assert after == pytest.approx(before, rel=0.15), (before, after)
    last = probe.state()
    assert last is not None and time.monotonic() - stab.teleop[-1][0] < 0.5
    # Every node of the stack is still there; only the monitor is gone. (assert_stack_nodes
    # refuses any launched process that has exited, which this one has, on purpose.)
    # A killed process sends no goodbye: discovery drops it when its lease runs out.
    probe.wait_for(lambda: '/stability_node' not in graph_nodes(probe, settle=0.5), 40.0,
                   'discovery to drop the killed monitor')
    before_nodes = set(stack['nodes'])
    now_nodes = set(graph_nodes(probe))
    assert now_nodes == before_nodes - {'/stability_node'}, (before_nodes ^ now_nodes)
    stack['results'].append(f'kill: teleop state {before:.1f}/s before, {after:.1f}/s after; '
                            'the drive after the kill ended in one zero, then silence')
