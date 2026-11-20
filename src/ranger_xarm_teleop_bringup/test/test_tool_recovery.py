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
``tools/recover_arm.py`` and what it has heard of teleop before it moves the arm.

With a teleop node running, the tool needs a fresh report that teleop is not armed before
any goal goes out; with no teleop node at all a recovery still works, and the tool says
which of the two cases it is in; a motion is stopped when teleop is armed during it. The
tool is run against a stand-in (``fake_recovery.py``) for move_group, the command mux's
report and the teleop node.
"""

import os
import re
import time

import pytest

from stack_helpers import child_environment, Process

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, '..', '..', '..', 'tools')
TARGET = ['0.3', '0', '0', '0', '0', '0']


@pytest.fixture(scope='module')
def env():
    return child_environment(offset=16)


def run(env, tmp_path, teleop, seconds, *options):
    """Run recover_arm.py against the stand-in; return (exit, tool, stand-in)."""
    fake = Process(['python3', os.path.join(HERE, 'fake_recovery.py'), teleop, str(seconds)],
                   env, str(tmp_path / 'fake.log'))
    try:
        end = time.monotonic() + 20.0
        while 'ready:' not in fake.output() and time.monotonic() < end:
            time.sleep(0.05)
        assert 'ready:' in fake.output(), fake.output()
        tool = Process(['python3', os.path.join(TOOLS, 'recover_arm.py'), *options, *TARGET],
                       env, str(tmp_path / 'tool.log'))
        try:
            code = tool.wait(90.0)
        finally:
            tool.stop()
        time.sleep(0.3)
        return code, tool.output(), fake.output()
    finally:
        fake.stop()


def test_with_no_teleop_node_at_all_a_recovery_works_and_says_so(env, tmp_path):
    code, tool, fake = run(env, tmp_path, 'none', 1.0)
    assert code == 0, tool
    assert 'disarm: no /teleop/disarm service' in tool
    assert 'MoveIt owns the arm, no teleop node is running' in tool
    assert 'still looking' not in tool
    assert 'not armed' not in tool and 'disarmed' not in tool      # nothing of it was heard
    assert 'recovered: the arm is at' in tool and 'goal reached' in fake


def test_with_a_teleop_node_the_tool_moves_after_hearing_that_it_is_not_armed(env, tmp_path):
    code, tool, fake = run(env, tmp_path, 'disarmed', 1.0)
    assert code == 0, tool
    assert 'disarm: was not armed' in tool
    assert 'MoveIt owns the arm, teleop reports that it is not armed' in tool
    assert 'recovered: the arm is at' in tool and 'goal reached' in fake


def test_a_teleop_node_that_reports_nothing_gets_no_goal(env, tmp_path):
    """The service and the state topic are there, no state comes: armed or not is unknown."""
    code, tool, fake = run(env, tmp_path, 'silent', 1.0, '--wait', '3')
    assert code == 4, tool
    assert 'NOT RECOVERED, no motion goal sent: a teleop node is running and no fresh' in tool
    assert 'whether teleop is armed is not known' in tool
    assert 'MoveIt owns the arm' not in tool
    assert 'goal accepted' not in fake


def test_a_state_topic_alone_shows_a_teleop_node(env, tmp_path):
    """The disarm service asked for does not exist; the state topic has a publisher."""
    code, tool, fake = run(env, tmp_path, 'silent', 1.0, '--wait', '3',
                           '--disarm-service', '/no_such/disarm')
    assert code == 4, tool
    assert 'disarm: no /no_such/disarm service' in tool
    assert 'whether teleop is armed is not known' in tool
    assert 'no teleop node is running' not in tool and 'goal accepted' not in fake


def test_teleop_armed_right_before_the_goal_gets_no_goal(env, tmp_path):
    """Not armed at the first look, armed once the tool has tried the cancel path."""
    code, tool, fake = run(env, tmp_path, 'arms_early', 1.0)
    assert code == 4, tool
    assert 'MoveIt owns the arm, teleop reports that it is not armed' in tool
    assert 'NOT RECOVERED, no motion goal sent: MoveIt does not own the arm (owner: TELEOP)' \
        in tool
    assert 'goal accepted' not in fake


def test_teleop_armed_during_the_motion_stops_it(env, tmp_path):
    code, tool, fake = run(env, tmp_path, 'arms', 10.0)
    assert code == 1, tool
    assert 'MoveIt owns the arm, teleop reports that it is not armed' in tool
    assert 'NOT RECOVERED, STOPPED: teleop was armed during the motion' in tool
    assert 'the server is cancelling the goal' in tool
    moved = re.search(r'the joints moved by up to (\d+\.\d+) rad and are still now', tool)
    assert moved, tool
    # Armed a second into a 10 s motion of 0.3 rad: stopped well short of the target.
    assert 0.02 < float(moved.group(1)) < 0.1
    assert re.search(r'cancel request: the goal, cancelled at 0\.0\d+ rad', fake), fake
    assert 'goal reached' not in fake
