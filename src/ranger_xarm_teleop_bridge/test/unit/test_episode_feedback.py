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
"""INV-3: one vibration request per episode event, never doubled or invented (no ROS)."""

import os
import re

import pytest
from ranger_xarm_teleop_bridge.episode_feedback import (DEFAULT_PATTERNS, EpisodeEvents,
                                                        EpisodeSample, EVENT_NAMES, Pattern)

MSG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                       'ranger_xarm_teleop_interfaces', 'msg'))
START, STOP, KEEP, DISCARD, MARK, REFUSED, AUTO_STOP = range(1, 8)
NOW = 1000.0


def constants(msg, prefix):
    with open(os.path.join(MSG_DIR, msg), encoding='utf-8') as handle:
        text = handle.read()
    return {int(v): n for n, v in re.findall(r'^uint8 %s(\w+) = (\d+)' % prefix, text, re.M)}


def sample(count, event, session='s1', detail='', wall=NOW, age=0.0):
    return EpisodeSample(session, count, event, detail, stamp=50.0 + age,
                         last_event_stamp=50.0, wall_time=wall)


def feed(events, samples, now=NOW):
    return [events.update(s, now) for s in samples]


def test_event_names_match_the_message():
    assert EVENT_NAMES == constants('EpisodeState.msg', 'EVENT_')


def test_every_event_has_a_distinct_default_pattern():
    assert set(DEFAULT_PATTERNS) == set(EVENT_NAMES.values()) - {'NONE'}
    shapes = {(len(p.pulses), p.duration_ms) for p in DEFAULT_PATTERNS.values()}
    assert len(shapes) == len(DEFAULT_PATTERNS)       # distinct by count and length
    must_not_miss = min(DEFAULT_PATTERNS[e].priority for e in ('REFUSED', 'AUTO_STOP'))
    others = max(p.priority for e, p in DEFAULT_PATTERNS.items()
                 if e not in ('REFUSED', 'AUTO_STOP'))
    assert must_not_miss > others
    for event in ('REFUSED', 'AUTO_STOP'):
        p = DEFAULT_PATTERNS[event]
        assert p.duration_ms >= 800 and max(i for i, _, _ in p.pulses) == 1.0


def test_first_message_is_a_baseline_and_repeats_give_nothing():
    e = EpisodeEvents()
    assert feed(e, [sample(5, STOP)] * 4) == [None] * 4


def test_each_new_event_gives_exactly_one_request():
    e = EpisodeEvents()
    out = feed(e, [sample(0, 0), sample(1, START), sample(1, START), sample(2, MARK),
                   sample(2, MARK), sample(3, STOP), sample(4, KEEP), sample(4, KEEP)])
    assert [r.event if r else None for r in out] == [
        None, 'START', None, 'MARK', None, 'STOP', 'KEEP', None]
    assert all(r.missed == 0 for r in out if r)
    assert out[1].pattern == DEFAULT_PATTERNS['START'] and out[1].source_count == 1


def test_a_lost_message_counts_the_missed_event_and_invents_none():
    e = EpisodeEvents()
    out = feed(e, [sample(1, START), sample(4, DISCARD)])
    assert out[0] is None
    assert out[1].event == 'DISCARD' and out[1].missed == 2
    assert e.missed_total == 2


def test_new_session_or_count_going_back_is_a_new_baseline():
    e = EpisodeEvents()
    out = feed(e, [sample(7, KEEP), sample(0, 0, session='s2'), sample(1, START, session='s2'),
                   sample(0, 0, session='s2')])
    assert [r.event if r else None for r in out] == [None, None, 'START', None]


def test_old_events_are_never_played_late():
    e = EpisodeEvents(max_age_s=0.75)
    out = feed(e, [sample(1, START), sample(2, STOP, wall=NOW - 0.76)])
    assert out == [None, None] and e.too_old == 1
    # Age taken on the event's own time: published 0.5 s after it in ROS time.
    assert e.update(sample(3, KEEP, wall=NOW, age=0.8), NOW) is None
    assert e.update(sample(4, KEEP, wall=NOW, age=0.7), NOW).event == 'KEEP'


def test_an_empty_pattern_vibrates_nothing():
    patterns = dict(DEFAULT_PATTERNS, MARK=Pattern((), 1))
    e = EpisodeEvents(patterns)
    out = feed(e, [sample(1, START), sample(2, MARK), sample(3, STOP)])
    assert out[1] is None and out[2].event == 'STOP'
    e = EpisodeEvents(dict(DEFAULT_PATTERNS, MARK=Pattern.from_flat([0.0, 40, 0], 1)))
    assert feed(e, [sample(1, START), sample(2, MARK)])[1] is None


@pytest.mark.parametrize('flat', [[1.0, 10], [1.5, 10, 0], [0.5, -1, 0], [0.5, 10.5, 0],
                                  [float('nan'), 10, 0], [1.0, 3000, 0, 1.0, 3000, 0]])
def test_bad_patterns_are_refused(flat):
    with pytest.raises(ValueError):
        Pattern.from_flat(flat, 1)


def test_pattern_round_trip():
    for p in DEFAULT_PATTERNS.values():
        assert Pattern.from_flat(p.flat(), p.priority) == p
    assert Pattern.from_flat([], 1).pulses == ()
