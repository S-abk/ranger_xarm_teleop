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
A second SIGINT while the episode node marks an episode still leaves it "interrupted".

The node runs its own main() in a child process. A recording episode is begun as an X press
begins one; the test sends the first SIGINT. The second comes from inside the node's
shutdown(), before the episode is marked, so it lands there every run rather than by timing.
"""

import glob
import json
import os
import signal
import sys
import time

from stack_helpers import child_environment, Process

CHILD = r"""
import os
import signal
import sys
import time

import rclpy
from ranger_xarm_teleop_interfaces.msg import TeleopState
from ranger_xarm_teleop_record import episode_node as en

real_spin, real_shutdown = rclpy.spin, en.EpisodeNode.shutdown


def spin(node, *args, **kwargs):
    node.state = TeleopState()
    node.begin()                          # an episode recording, as an X press leaves it
    print('episode recording', flush=True)
    real_spin(node, *args, **kwargs)


def shutdown(node):
    os.kill(os.getpid(), signal.SIGINT)   # the second SIGINT, before the episode is marked
    time.sleep(0.5)
    real_shutdown(node)
    print('episode marked', flush=True)


rclpy.spin = spin
en.EpisodeNode.shutdown = shutdown
en.main(['--ros-args', '-p', 'session_dir:=' + sys.argv[1]])
"""


def test_a_second_sigint_during_shutdown_still_marks_the_episode_interrupted(tmp_path):
    env = child_environment(offset=20)
    session_dir = tmp_path / 'data' / 'session'
    node = Process([sys.executable, '-c', CHILD, str(session_dir)], env,
                   str(tmp_path / 'episode_node.log'))
    try:
        end = time.monotonic() + 60.0
        while 'episode recording' not in node.output():
            assert time.monotonic() < end and node.proc.poll() is None, node.output()[-3000:]
            time.sleep(0.1)
        os.kill(node.proc.pid, signal.SIGINT)         # the first SIGINT
        end = time.monotonic() + 30.0
        while node.proc.poll() is None:
            assert time.monotonic() < end, 'still running: ' + node.output()[-3000:]
            time.sleep(0.1)
        code = node.proc.returncode
    finally:
        node.stop()
    out = node.output()
    sidecars = sorted(glob.glob(str(session_dir / 'episodes' / '*.json')))
    assert len(sidecars) == 1, out[-3000:]
    with open(sidecars[0], encoding='utf-8') as handle:
        episode = json.load(handle)['episode']
    assert episode['status'] == 'interrupted', out[-3000:]
    assert episode['stop_reason'] == 'shutdown'
    assert 'episode marked' in out and code == 0, out[-3000:]
