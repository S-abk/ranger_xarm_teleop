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
The session launcher with the Gazebo profile (opt-in: ``TELEOP_TEST_GAZEBO=1``).

One command brings up the simulator (headless, both cameras on, on the NVIDIA card where
there is one), measures it, starts the teleop stack with recording, feedback and the reset,
and the bridge; input played into its relay brings it to "session ready"; one stop ends every
process, the simulator's included.
"""

import os
import time

import pytest
from ranger_xarm_teleop_interfaces.msg import ArmOwner
from ranger_xarm_teleop_record import metadata as md
from ranger_xarm_teleop_session import procs
from stack_helpers import graph_nodes, kill_gz_partition, Process, register_log
from test_mock_record import RecordProbe
from test_mock_session import environment, FAKE_QUEST, Launcher, tool

pytestmark = pytest.mark.skipif(os.environ.get('TELEOP_TEST_GAZEBO') != '1',
                                reason='set TELEOP_TEST_GAZEBO=1 to run the Gazebo tests')

PORT = 18452
OFFSET = 21
MOVEIT, MOVEIT_G = 'xarm_xarm6_traj_controller', 'xarm_xarm_gripper_traj_controller'
STILL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__)))), 'ranger_xarm_teleop_bridge', 'test', 'fixtures', 'still_controller.jsonl')


def test_one_command_brings_up_gazebo_with_recording_and_one_stop_ends_it(tmp_path):
    data = tmp_path / 'data'
    md.write_labels(str(data), 'test-task', 'test-operator')
    env = environment(tmp_path, offset=OFFSET)
    env['GZ_PARTITION'] = f'teleop_session_test_{os.getpid()}'
    env['GZ_IP'] = '127.0.0.1'
    probe = RecordProbe(env, moveit_controller=MOVEIT, moveit_gripper=MOVEIT_G)
    launcher = Launcher(env, tmp_path, 'launcher', '--profile', 'gazebo', '--headless', '--yes',
                        '--keep-labels', '--data-dir', str(data), '--port', str(PORT),
                        '--sim-check-seconds', '15')
    quest = None
    try:
        state = launcher.wait_phase(('stack_ready', 'session_ready'), 900.0)
        register_log(state['log_dir'])
        out = launcher.output()
        assert 'sim.launch.py headless:=true sensors:=true' in out
        assert '[UNKNOWN] Simulator speed' not in out
        assert 'Simulator speed and joint-state delivery' in out
        for item in ('Teleop disarmed, MoveIt owns the arm', 'Recorder',
                     'Operator feedback node', 'Reset node and watchdog'):
            assert f'[     ok] {item}' in out, item
        assert '/teleop_session_probe' not in graph_nodes(probe, settle=2.0)
        assert probe.owner_value() == ArmOwner.OWNER_MOVEIT and not probe.state().armed
        quest = Process(['python3', FAKE_QUEST, STILL, '--loop', '--url',
                         f'ws://127.0.0.1:{PORT}/ws'], env, str(tmp_path / 'fake_quest.log'))
        launcher.wait_phase(('session_ready',), 60.0)
    finally:
        if quest is not None:
            quest.stop()
        probe.close()
        if launcher.process.wait(0.0) is None:
            code, stopped = tool(env, 'stop')
        else:
            code, stopped = None, ''
        launcher.process.stop()
        left = kill_gz_partition(env['GZ_PARTITION'])
        print(launcher.output()[-6000:])
    assert code == 0, stopped
    assert 'PASS' in stopped, stopped
    order = [line.split('stopping ')[1].split(':')[0] for line in launcher.output().splitlines()
             if ' stopping ' in line and ':' in line.split('stopping ')[1]]
    assert order == ['bridge', 'stack', 'simulator', 'relay'], order
    assert left == 0, f'{left} gz processes of the session were left'
    assert procs.marked(state['mark']) == []
    time.sleep(0.5)
    assert not os.path.exists(procs.state_path(env))
