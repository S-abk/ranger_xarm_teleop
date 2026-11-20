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
