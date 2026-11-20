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
"""A read whose reply may be lost is asked again; nothing waits for ever."""

from ranger_xarm_teleop_servo import calls


def test_a_read_is_not_due_until_it_is_wanted():
    read = calls.Reread('the description', 2.0)
    assert not read.wanted and not read.due()
    assert read.lost(100.0) is None
    read.start()
    assert read.wanted and read.due()


def test_an_answered_read_is_no_longer_wanted():
    read = calls.Reread('the description', 2.0)
    read.start()
    first = object()
    read.sent(first, 10.0)
    assert read.wanted and not read.due()           # out: neither asked twice nor given up
    assert read.lost(11.9) is None
    assert read.answered(first)
    assert not read.wanted and not read.due() and read.attempts == 1
    assert read.lost(100.0) is None


def test_a_request_without_an_answer_is_given_up_and_the_read_is_due_again():
    read = calls.Reread('the description', 2.0)
    read.start()
    first = object()
    read.sent(first, 10.0)
    assert read.lost(11.99) is None
    assert read.lost(12.0) is first                 # handed back once, to be dropped
    assert read.lost(12.0) is None
    assert read.wanted and read.due()               # the answer is still wanted
    second = object()
    read.sent(second, 12.0)
    assert read.attempts == 2
    assert read.lost(13.9) is None and read.lost(14.0) is second
    assert read.wanted and read.due()


def test_a_read_never_stops_being_wanted_without_an_answer():
    read = calls.Reread('the description', 2.0)
    read.start()
    now = 0.0
    for attempt in range(1, 50):
        assert read.due()
        read.sent(object(), now)
        now += 2.0
        assert read.lost(now) is not None
        assert read.wanted and read.attempts == attempt


def test_a_late_answer_to_a_request_given_up_is_not_used():
    read = calls.Reread('the description', 2.0)
    read.start()
    first, second = object(), object()
    read.sent(first, 0.0)
    assert read.lost(2.0) is first
    read.sent(second, 2.0)
    assert not read.answered(first)                 # too late: the second request is out
    assert read.wanted and read.request is second
    assert read.answered(second)
    assert not read.wanted


def test_a_dropped_read_is_asked_from_scratch_when_started_again():
    read = calls.Reread('the description', 2.0)
    read.start()
    request = object()
    read.sent(request, 0.0)
    read.drop()
    assert not read.wanted and not read.due() and read.lost(10.0) is None
    assert not read.answered(request)               # its answer no longer counts
    read.start()
    assert read.due() and read.attempts == 0


def test_starting_twice_does_not_reset_a_read_that_is_out():
    read = calls.Reread('the description', 2.0)
    read.start()
    request = object()
    read.sent(request, 0.0)
    read.start()
    assert read.request is request and read.attempts == 1


def test_parameter_clients_keep_more_replies_than_the_default_of_ten():
    assert calls.PARAMETER_REPLY_DEPTH >= 1000


# ---- a motion goal without an answer ----

NAMES = ('j1', 'j2')


def rows(*points):
    return [(t, {'j1': a, 'j2': b, 'finger': 0.0}) for t, a, b in points]


def test_an_arm_that_did_not_move_is_reported_so_only_when_it_was_watched_at_rest():
    samples = rows(*[(k * 0.1, 0.5, -0.2) for k in range(31)])          # 3 s at rest
    motion = calls.motion_seen(samples, NAMES, since=1.0, now=3.0)
    assert motion.known and motion.still and motion.moved == 0.0
    assert 'did not move' in motion.text()
    assert calls.motion_seen([], NAMES, since=1.0, now=3.0).text().startswith(
        'no joint states arrived')
    assert 'not known' in calls.motion_seen(rows((0.0, 0.5, 0.0)), ('j1', 'j9'), 0.0, 1.0).text()


def test_joint_states_that_stopped_arriving_do_not_show_an_arm_at_rest():
    samples = rows(*[(k * 0.1, 0.5, -0.2) for k in range(11)])          # the last one at 1.0 s
    motion = calls.motion_seen(samples, NAMES, since=0.5, now=3.0)
    assert motion.known and not motion.still
    assert 'did not move' not in motion.text()


def test_motion_since_the_goal_is_reported_with_whether_it_has_ended():
    moving = rows(*[(k * 0.1, 0.5 + 0.01 * k, -0.2) for k in range(31)])
    motion = calls.motion_seen(moving, NAMES, since=1.0, now=3.0)
    assert not motion.still and abs(motion.moved - 0.20) < 1e-9          # from the sample at 1.0 s
    assert 'moved by up to 0.200 rad' in motion.text() and 'STILL MOVING' in motion.text()
    stopped = moving + rows(*[(3.0 + k * 0.1, 0.8, -0.2) for k in range(1, 16)])
    motion = calls.motion_seen(stopped, NAMES, since=1.0, now=4.5)
    assert motion.still and abs(motion.moved - 0.20) < 1e-9
    assert 'are still now' in motion.text() and 'did not move' not in motion.text()


def test_a_joint_that_went_and_came_back_counts_as_moved():
    there_and_back = rows((0.0, 0.5, 0.0), (0.5, 0.9, 0.0), (1.0, 0.5, 0.0), (1.5, 0.5, 0.0),
                          (2.0, 0.5, 0.0), (2.4, 0.5, 0.0))
    motion = calls.motion_seen(there_and_back, NAMES, since=0.0, now=2.5)
    assert abs(motion.moved - 0.4) < 1e-9 and motion.still


def test_the_report_of_an_unanswered_goal_says_what_was_seen_and_never_guesses():
    at_rest = calls.motion_seen(rows(*[(k * 0.1, 0.5, -0.2) for k in range(31)]), NAMES, 1.0, 3.0)
    text = calls.unanswered_goal('the goal', 10.0, [], (0, 0), at_rest)
    assert 'UNKNOWN' in text and 'never listed the goal' in text
    assert 'no goal to cancel' in text and 'did not move' in text
    moving = calls.motion_seen(rows(*[(k * 0.1, 0.5 + 0.01 * k, -0.2) for k in range(31)]),
                               NAMES, 1.0, 3.0)
    text = calls.unanswered_goal('the goal', 10.0, [1, 2], None, moving)
    assert 'listed the goal as accepted, then executing;' in text and 'NOT ANSWERED' in text
    assert 'STILL MOVING' in text and 'did not move' not in text and 'nothing moved' not in text
    blind = calls.unanswered_goal('the goal', 10.0, [2, 3, 5], (0, 1), calls.Motion(False))
    assert 'executing, then cancelling, then cancelled;' in blind
    assert 'cancelling 1 goal;' in blind and 'not known' in blind
    assert 'did not move' not in blind and 'nothing moved' not in blind


def test_a_refused_cancel_is_reported_with_its_reason():
    assert 'no such goal' in calls.cancel_text((2, 0))
    assert 'already ended' in calls.cancel_text((3, 0))
    assert 'return code 9' in calls.cancel_text((9, 0))
    assert calls.cancel_text((0, 2)).endswith('cancelling 2 goals')
