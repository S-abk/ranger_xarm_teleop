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

``tools/move_arm.py`` follows the same rule where a command mux runs, and refuses a goal when
a teleop node runs without a mux (its section at the end).
"""

import os
import re
import time

from fake_teleop_side import ARM_AFTER_RAD
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


# ---- tools/move_arm.py: a goal straight to MoveIt's controller ----
#
# With a command mux, the rule of recover_arm.py (may_recover) and the same stop; with no mux
# and no teleop node, as before; with no mux but a teleop node, no goal. Run against
# fake_controller.py (the trajectory controller) and fake_teleop_side.py (the mux's report
# and the teleop node).

def run_move(env, tmp_path, mux, teleop, seconds):
    """Run move_arm.py against the two stand-ins; return (exit, tool, controller, side)."""
    controller = Process(['python3', os.path.join(HERE, 'fake_controller.py'), 'stand_in',
                          'answer'], env, str(tmp_path / 'controller.log'))
    side = Process(['python3', os.path.join(HERE, 'fake_teleop_side.py'), mux, teleop],
                   env, str(tmp_path / 'side.log'))
    try:
        end = time.monotonic() + 20.0
        while not ('ready:' in controller.output() and 'ready:' in side.output()) \
                and time.monotonic() < end:
            time.sleep(0.05)
        assert 'ready:' in controller.output(), controller.output()
        assert 'ready:' in side.output(), side.output()
        tool = Process(['python3', os.path.join(TOOLS, 'move_arm.py'), '--controller',
                        'stand_in', '--seconds', str(seconds), *TARGET],
                       env, str(tmp_path / 'tool.log'))
        try:
            code = tool.wait(90.0)
        finally:
            tool.stop()
        time.sleep(0.3)
        return code, tool.output(), controller.output(), side.output()
    finally:
        side.stop()
        controller.stop()


def test_move_arm_with_a_mux_moves_after_hearing_that_teleop_is_not_armed(env, tmp_path):
    code, tool, controller, _ = run_move(env, tmp_path, 'mux', 'disarmed', 1.0)
    assert code == 0, tool
    assert ('the command mux reports that MoveIt owns the arm, teleop reports that it is not '
            'armed') in tool
    assert 'goal succeeded' in tool and 'goal reached' in controller


def test_move_arm_with_a_mux_stops_when_teleop_is_armed_during_the_motion(env, tmp_path):
    code, tool, controller, side = run_move(env, tmp_path, 'mux', 'arms', 10.0)
    assert code == 1, tool
    assert 'armed' in side
    assert 'goal STOPPED: teleop was armed during the motion' in tool, tool
    assert 'the server is cancelling the goal' in tool
    moved = re.search(r'the joints moved by up to (\d+\.\d+) rad and are still now', tool)
    assert moved, tool
    # It moved until teleop was armed, at the stand-in's trigger, and stopped well short of
    # 0.3 rad. The stop can land exactly on the trigger; the figure is printed with three
    # decimals, so it may read up to 0.0005 rad below it.
    assert ARM_AFTER_RAD - 0.0005 <= float(moved.group(1)) < 0.15, tool
    assert 'goal reached' not in controller and 'every goal' not in controller


def test_move_arm_without_a_mux_or_a_teleop_node_moves_as_before(env, tmp_path):
    code, tool, controller, _ = run_move(env, tmp_path, 'none', 'none', 1.0)
    assert code == 0, tool
    assert 'no command mux and no teleop node are running' in tool
    assert 'goal succeeded' in tool and 'goal reached' in controller


def test_move_arm_without_a_mux_but_with_a_teleop_node_sends_no_goal(env, tmp_path):
    code, tool, controller, _ = run_move(env, tmp_path, 'none', 'disarmed', 1.0)
    assert code == 4, tool
    assert ('REFUSED, no motion goal sent: a teleop node is running but no command mux reports '
            'who owns the arm') in tool
    assert 'accepted' not in controller and 'goal reached' not in controller
