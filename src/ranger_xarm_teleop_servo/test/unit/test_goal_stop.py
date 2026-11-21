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
"""Stopping a goal by its own id (goal_stop), and the rule for a direct goal (session)."""

import pytest

from ranger_xarm_teleop_servo import session
from ranger_xarm_teleop_servo.calls import Motion
from ranger_xarm_teleop_servo.goal_stop import GoalStop, SETTLE_S

GOAL = bytes(range(1, 17))
STILL = Motion(True, 0.05, True, 2.0)
MOVING = Motion(True, 0.05, False, 2.0)


def test_the_zero_id_or_a_short_one_is_refused():
    with pytest.raises(ValueError, match='every goal'):
        GoalStop(bytes(16), 1.0)
    with pytest.raises(ValueError):
        GoalStop(b'\x01' * 15, 1.0)


def test_the_tools_ask_at_most_their_attempts_and_stop_at_the_first_answer():
    stop = GoalStop(GOAL, 2.0, attempts=3)
    assert stop.due(0.0)
    stop.sent(0.0)
    assert not stop.due(1.9)                    # still waiting for the answer
    assert stop.due(2.0) and stop.unanswered_count == 1
    stop.sent(2.0)
    stop.answered(2.5, 0, 1)
    assert not stop.due(10.0) and stop.accepted()
    assert stop.cancel_answer() == (0, 1) and stop.sent_count == 2

    stop = GoalStop(GOAL, 2.0, attempts=3)
    for k in range(3):
        assert stop.due(2.0 * k)
        stop.sent(2.0 * k)
        stop.timed_out()
    assert not stop.due(100.0)
    assert stop.cancel_answer() is None and stop.unanswered_count == 3
    assert 'NOT ANSWERED' in stop.report(STILL)


def test_without_a_limit_it_asks_again_until_the_goal_has_ended():
    stop = GoalStop(GOAL, 0.5)
    for k in range(10):                         # every answer lost
        assert stop.due(0.5 * k)
        stop.sent(0.5 * k)
    assert stop.sent_count == 10 and not stop.ended_by
    stop.answered(5.1, 0, 1)                    # cancelling: ask again until it is listed ended
    assert stop.due(5.2)
    stop.status(5.3, 3)
    assert stop.due(5.4)
    stop.status(5.5, 5)
    assert stop.ended_by == 'the server listed it as cancelled'
    assert not stop.due(5.6)


def test_ended_is_not_finished_until_the_joints_are_still():
    stop = GoalStop(GOAL, 0.5)
    stop.status(1.0, 6)
    assert stop.ended_by and not stop.finished(MOVING) and stop.finished(STILL)
    assert not stop.finished(Motion(False))     # joints unknown: not shown to be still


def test_an_unknown_goal_ends_only_after_answers_for_the_settle_time():
    stop = GoalStop(GOAL, 0.5)
    stop.sent(0.0)
    stop.answered(0.1, 2, 0)
    assert not stop.ended_by and stop.due(0.6)   # it may still be on its way
    stop.tick(0.1 + SETTLE_S - 0.01)
    assert not stop.ended_by
    stop.tick(0.1 + SETTLE_S)
    assert 'knew no goal with this id' in stop.ended_by


def test_a_goal_listed_active_after_an_unknown_answer_restarts_the_wait():
    stop = GoalStop(GOAL, 0.5)
    stop.answered(0.0, 2, 0)
    stop.status(1.0, 2)                         # the goal arrived late and runs
    stop.tick(0.0 + SETTLE_S + 0.5)
    assert not stop.ended_by and stop.due(3.0)
    stop.answered(3.0, 0, 1)
    stop.status(3.2, 5)
    assert stop.ended_by == 'the server listed it as cancelled'


def test_cannot_be_cancelled_ends_after_the_settle_time_without_an_active_status():
    stop = GoalStop(GOAL, 0.5)
    stop.answered(0.0, 3, 0)
    stop.tick(SETTLE_S)
    assert 'could no longer be cancelled' in stop.ended_by
    stop = GoalStop(GOAL, 0.5)
    stop.answered(0.0, 3, 0)
    stop.status(0.5, 3)                         # being cancelled: wait for the end
    stop.tick(SETTLE_S + 1.0)
    assert not stop.ended_by


def test_the_report_says_what_was_asked_answered_listed_and_moved():
    stop = GoalStop(GOAL, 0.5)
    stop.other_status()
    stop.sent(0.0)
    stop.timed_out()
    stop.sent(0.5)
    stop.answered(0.6, 0, 1)
    stop.status(0.7, 2)
    stop.status(0.8, 5)
    text = stop.report(STILL)
    assert "2 cancel request(s) for this goal's own id, 1 unanswered" in text
    assert 'the server is cancelling the goal' in text
    assert 'the server listed the goal as executing, then cancelled' in text
    assert 'the joints moved by up to 0.050 rad and are still now' in text


@pytest.mark.parametrize('mux, owner, armed, running, ok, words', [
    # With a mux: the rule of may_recover.
    (True, session.OWNER_MOVEIT, False, True, True, ''),
    (True, session.OWNER_MOVEIT, None, False, True, ''),
    (True, session.OWNER_MOVEIT, None, True, False, 'not known'),
    (True, session.OWNER_TELEOP, True, True, False, 'owner: TELEOP'),
    (True, None, False, True, False, 'not reported'),
    # Without a mux.
    (False, None, None, False, True, ''),
    (False, None, False, True, False, 'no command mux reports who owns the arm'),
])
def test_a_direct_goal_needs_the_mux_rule_or_no_one_who_could_take_the_arm(
        mux, owner, armed, running, ok, words):
    allowed, why = session.may_move_directly(mux, owner, armed, running)
    assert allowed == ok and words in why


def test_the_direct_case_is_said_in_words():
    assert session.direct_case(False, None) == 'no command mux and no teleop node are running'
    assert session.direct_case(True, False) == (
        'the command mux reports that MoveIt owns the arm, teleop reports that it is not armed')
    assert session.direct_case(True, None).endswith('no teleop node is running')
