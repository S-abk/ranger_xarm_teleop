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


def watch(points, mark_at=None, names=NAMES):
    """Return a JointWatch fed with (time, j1, j2) points, marked at ``mark_at`` in between."""
    seen = calls.JointWatch(names)
    for t, a, b in points:
        if mark_at is not None and seen.marked_at is None and t > mark_at:
            seen.mark(mark_at)
        seen.sample(t, {'j1': a, 'j2': b, 'finger': 0.0})
    if mark_at is not None and seen.marked_at is None:
        seen.mark(mark_at)
    return seen


def test_an_arm_that_did_not_move_is_reported_so_only_when_it_was_watched_at_rest():
    motion = watch([(k * 0.1, 0.5, -0.2) for k in range(31)], mark_at=1.0).motion(3.0)
    assert motion.known and motion.still and motion.from_before and motion.moved == 0.0
    assert 'did not move' in motion.text() and '2.0 s' in motion.text()
    nothing = calls.JointWatch(NAMES).motion(3.0).text()
    assert nothing.startswith("no fresh joint state with all of the arm's joints was at hand")
    # Joint states came, but none with the joints asked about: the same is said, and not
    # that no joint state arrived.
    other = watch([(0.0, 0.5, 0.0), (0.5, 0.5, 0.0)], mark_at=0.2, names=('j1', 'j9'))
    assert other.motion(1.0).text() == nothing and 'not known' in nothing


def test_joint_states_that_stopped_arriving_do_not_show_an_arm_at_rest():
    motion = watch([(k * 0.1, 0.5, -0.2) for k in range(11)], mark_at=0.5).motion(3.0)
    assert motion.known and not motion.still and abs(motion.stale_s - 2.0) < 1e-9
    text = motion.text()
    # Neither at rest nor moving: nobody saw what the joints did after the last joint state.
    assert 'did not move' not in text and 'STILL MOVING' not in text and 'still now' not in text
    assert 'had moved by up to 0.000 rad when the joint states stopped arriving 2.0 s ago' in text
    assert 'what they do now is not known' in text
    # The same for an arm that was moving when they stopped.
    moving = watch([(k * 0.1, 0.5 + 0.01 * k, -0.2) for k in range(11)], mark_at=0.5).motion(3.0)
    assert 'had moved by up to 0.050 rad when the joint states stopped' in moving.text()
    assert 'STILL MOVING' not in moving.text()
    # Joint states up to the report are not stale.
    fresh = watch([(k * 0.1, 0.5, -0.2) for k in range(31)], mark_at=1.0).motion(3.2)
    assert fresh.stale_s is None and fresh.still


def test_motion_since_the_goal_is_reported_with_whether_it_has_ended():
    moving = [(k * 0.1, 0.5 + 0.01 * k, -0.2) for k in range(31)]
    motion = watch(moving, mark_at=1.0).motion(3.0)
    assert not motion.still and abs(motion.moved - 0.20) < 1e-9      # from the state at 1.0 s
    assert 'moved by up to 0.200 rad' in motion.text() and 'STILL MOVING' in motion.text()
    stopped = moving + [(3.0 + k * 0.1, 0.8, -0.2) for k in range(1, 16)]
    motion = watch(stopped, mark_at=1.0).motion(4.5)
    assert motion.still and abs(motion.moved - 0.20) < 1e-9
    assert 'are still now' in motion.text() and 'did not move' not in motion.text()


def test_a_joint_that_went_and_came_back_counts_as_moved():
    there_and_back = [(0.0, 0.5, 0.0), (0.5, 0.9, 0.0)] + [
        (1.0 + k * 0.1, 0.5, 0.0) for k in range(16)]
    motion = watch(there_and_back, mark_at=0.0).motion(2.5)
    assert abs(motion.moved - 0.4) < 1e-9 and motion.still
    assert 'did not move' not in motion.text()


def test_motion_is_counted_however_long_ago_the_goal_was_sent():
    # Ten minutes of joint states at 150 Hz after the mark; the arm moved in the first second.
    seen = calls.JointWatch(NAMES)
    seen.sample(0.0, {'j1': 0.0, 'j2': 0.0})
    seen.mark(0.0)
    for k in range(1, 90001):
        seen.sample(k / 150.0, {'j1': min(0.3, k / 150.0 * 0.3), 'j2': 0.0})
    assert len(seen.recent) < 400                   # only the last two seconds are kept
    motion = seen.motion(600.0)
    assert abs(motion.moved - 0.3) < 1e-9 and motion.still
    assert 'did not move' not in motion.text()      # the motion is not forgotten


def test_without_a_joint_state_from_before_the_goal_nothing_is_said_about_no_motion():
    late = calls.JointWatch(NAMES)
    late.mark(10.0)                                 # no joint state yet
    for k in range(30):
        late.sample(10.5 + k * 0.1, {'j1': 0.5, 'j2': 0.0})
    motion = late.motion(13.4)
    assert motion.known and not motion.from_before and motion.still
    assert 'did not move' not in motion.text()
    assert 'no joint state from before the request was at hand' in motion.text()
    stale = calls.JointWatch(NAMES)
    stale.sample(0.0, {'j1': 0.5, 'j2': 0.0})
    stale.mark(10.0)                                # the newest joint state is 10 s old
    assert not stale.from_before


def test_the_report_of_an_unanswered_goal_says_what_was_seen_and_never_guesses():
    at_rest = watch([(k * 0.1, 0.5, -0.2) for k in range(31)], mark_at=1.0).motion(3.0)
    text = calls.unanswered_goal('the goal', 10.0, [], (2, 0), at_rest)
    assert 'its state is not known' in text and 'the server did not list the goal' in text
    assert 'knew no goal with this id when asked to cancel it' in text and 'did not move' in text
    moving = watch([(k * 0.1, 0.5 + 0.01 * k, -0.2) for k in range(31)], mark_at=1.0).motion(3.0)
    text = calls.unanswered_goal('the goal', 10.0, [1, 2], None, moving)
    assert 'listed the goal as accepted, then executing;' in text and 'NOT ANSWERED' in text
    assert 'STILL MOVING' in text and 'did not move' not in text and 'nothing moved' not in text
    blind = calls.unanswered_goal('the goal', 10.0, [2, 3, 5], (0, 1), calls.Motion(False))
    assert 'executing, then cancelling, then cancelled;' in blind
    assert 'the server is cancelling the goal;' in blind and 'not known' in blind
    assert 'did not move' not in blind and 'nothing moved' not in blind


def test_no_status_message_at_all_is_not_a_goal_the_server_did_not_list():
    quiet = calls.status_text([], any_status=False)
    assert 'no status message' in quiet and 'not known' in quiet
    assert calls.status_text([], any_status=True) == 'the server did not list the goal'


def test_a_refused_cancel_is_reported_with_its_reason():
    assert 'return code 9' in calls.cancel_text((9, 0))
    assert 'rejected the cancel request' in calls.cancel_text((1, 0))
    assert 'named no goal that it cancels' in calls.cancel_text((0, 0))


def test_a_cancel_answer_is_reported_as_what_it_shows_and_no_more():
    # An unknown id: the goal never arrived, or it ended and the server forgot it (a server
    # keeps a finished goal for some seconds only), or it is still on its way. Not one of
    # them is claimed.
    assert calls.cancel_text((2, 0)) == ('the server knew no goal with this id when asked to '
                                         'cancel it')
    # A goal whose result was lost, cancelled after the server forgot it: no contradiction
    # between what was listed and what the cancel is said to show.
    still = watch([(k * 0.1, 0.5, -0.2) for k in range(31)], mark_at=1.0).motion(3.0)
    text = calls.unanswered_goal("the request for the goal's result", 30.0, [2, 4], (2, 0), still)
    assert 'listed the goal as executing, then succeeded; the server knew no goal' in text
    assert 'never reached' not in text
    # A goal already being cancelled (a second request after a late answer) is answered like
    # one that has ended: the text says both.
    past = calls.cancel_text((3, 0))
    assert 'can no longer be cancelled' in past and 'already being cancelled' in past
