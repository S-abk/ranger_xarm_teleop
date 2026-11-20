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
The arm tools and a controller that loses its answers.

A reply can be lost without the caller being told, so a goal without an answer is not a
rejected goal: the controller may be moving. ``tools/move_arm.py`` is run against a stand-in
controller (``fake_controller.py``) that accepts, rejects, or loses one of its answers. The
other two tools send their goals through the same code (``tools/_calls.py``).
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
    return child_environment(offset=15)


def run(env, tmp_path, mode, seconds):
    """Run move_arm.py against the stand-in in ``mode``; return (exit, tool, controller)."""
    controller = Process(['python3', os.path.join(HERE, 'fake_controller.py'), 'stand_in', mode],
                         env, str(tmp_path / 'controller.log'))
    try:
        end = time.monotonic() + 20.0
        while 'ready:' not in controller.output() and time.monotonic() < end:
            time.sleep(0.05)
        assert 'ready:' in controller.output(), controller.output()
        tool = Process(['python3', os.path.join(TOOLS, 'move_arm.py'), '--controller',
                        'stand_in', '--seconds', str(seconds), *TARGET],
                       env, str(tmp_path / 'tool.log'))
        try:
            code = tool.wait(90.0)
        finally:
            tool.stop()
        time.sleep(0.3)
        return code, tool.output(), controller.output()
    finally:
        controller.stop()


def test_an_answered_goal_moves_the_arm_and_succeeds(env, tmp_path):
    code, tool, controller = run(env, tmp_path, 'answer', 1.0)
    assert code == 0, tool
    assert 'goal succeeded' in tool and 'UNKNOWN' not in tool
    # The cancel path was tried first, with a request that cancels nothing.
    assert controller.index('a goal the controller never had') < controller.index('accepted')
    assert 'goal reached' in controller and 'cancel request: the goal' not in controller


def test_a_rejected_goal_is_reported_with_what_the_joints_did(env, tmp_path):
    code, tool, controller = run(env, tmp_path, 'reject', 1.0)
    assert code == 1, tool
    assert 'goal REJECTED' in tool and 'the joints did not move' in tool
    assert 'UNKNOWN' not in tool and 'cancel request: the goal' not in controller


def test_a_goal_without_an_answer_is_unknown_and_the_motion_is_cancelled(env, tmp_path):
    """The controller executes the goal and its answer is lost: the arm is moving."""
    code, tool, controller = run(env, tmp_path, 'goal_moves', 30.0)
    assert code == 1, tool
    assert 'goal UNKNOWN: no answer to the goal within 10 s' in tool, tool
    assert 'REJECTED' not in tool and 'did not move' not in tool and 'nothing' not in tool
    # What it saw: the goal listed as running, the cancel taken, the joints moved and at rest.
    # (The status topic keeps the newest status only, so a step in between may be missing.)
    assert re.search(r'the server listed the goal as (accepted, then )?executing'
                     r'(, then cancelling)?, then cancelled;', tool), tool
    assert 'the server is cancelling the goal;' in tool
    moved = re.search(r'the joints moved by up to (\d+\.\d+) rad and are still now', tool)
    assert moved, tool
    # A third of the way at most: 0.3 rad in 30 s, cancelled after the 10 s wait.
    assert 0.05 < float(moved.group(1)) < 0.15
    stopped = re.search(r'cancel request: the goal, cancelled at (\d+\.\d+) rad', controller)
    assert stopped and float(stopped.group(1)) == pytest.approx(float(moved.group(1)), abs=0.01)
    assert 'goal reached' not in controller             # it never ran on to the target
    assert 'every goal' not in controller               # only the tool's own goal was named


def test_a_goal_request_that_was_lost_is_unknown_and_nothing_is_claimed_unseen(env, tmp_path):
    """The request never reached the controller: nothing moves, and that is observed."""
    code, tool, controller = run(env, tmp_path, 'goal_ignored', 1.0)
    assert code == 1, tool
    assert 'goal UNKNOWN: no answer to the goal within 10 s' in tool, tool
    assert 'REJECTED' not in tool
    assert 'the server did not list the goal' in tool
    assert 'the server knows no such goal (the goal never reached it)' in tool
    assert re.search(r'the joints did not move \(largest change 0\.0000 rad in \d+\.\d s\)', tool)
    # The tool's one real cancel request came after the one that tries the path.
    assert controller.count('cancel request: a goal the controller never had') == 2


def test_a_lost_result_and_an_unanswered_cancel_are_reported_as_such(env, tmp_path):
    """The motion ran to its end; neither its result nor the cancel's answer arrives."""
    code, tool, controller = run(env, tmp_path, 'result_lost', 0.5)
    assert code == 1, tool
    assert "goal UNKNOWN: no answer to the request for the goal's result within 30.5 s" in tool
    assert 'FAILED' not in tool
    assert re.search(r'the server listed the goal as (accepted, then )?executing, then '
                     r'succeeded;', tool), tool
    assert 'the cancel request was NOT ANSWERED' in tool
    assert re.search(r'the joints moved by up to 0\.300 rad and are still now', tool), tool
    # The cancel names the tool's own goal, so asking three times cancels nothing else.
    assert controller.count('cancel request: the goal, never answered') == 3
    assert 'every goal' not in controller
