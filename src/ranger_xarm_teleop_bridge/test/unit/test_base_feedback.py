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
"""Base events to vibration, the base panel line and the base part of the status (no ROS)."""

import json
import math
import os
import re

import pytest
from ranger_xarm_teleop_bridge import base_feedback as bf
from ranger_xarm_teleop_bridge import relay_messages as rm
from ranger_xarm_teleop_bridge.base_feedback import BaseEvents, BaseSample
from ranger_xarm_teleop_bridge.episode_feedback import Pattern
from ranger_xarm_teleop_bridge.operator_status import BaseView, Snapshot, StatusTracker
import yaml

MSG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                       'ranger_xarm_teleop_interfaces', 'msg'))
CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config',
                                      'operator_feedback.yaml'))
START, REFUSED, STOPPED, RELEASED, AT_REST = range(1, 6)
OFF, IDLE, DRIVING, STOPPING, S_REFUSED, S_STOPPED = range(6)
NOW = 1000.0


def constants(msg, prefix):
    with open(os.path.join(MSG_DIR, msg), encoding='utf-8') as handle:
        text = handle.read()
    return {int(v): n for n, v in re.findall(r'^uint8 %s(\w+) = (\d+)' % prefix, text, re.M)}


def sample(count, event, wall=NOW):
    return BaseSample(count, event, wall)


def feed(events, samples, now=NOW):
    return [events.update(s, now) for s in samples]


def names(out):
    return [r.event if r else None for r in out]


# ---- code tables ----

def test_tables_match_the_message():
    assert bf.STATE_NAMES == constants('BaseState.msg', 'STATE_')
    assert bf.REASON_NAMES == constants('BaseState.msg', 'REASON_')
    assert bf.EVENT_NAMES == constants('BaseState.msg', 'EVENT_')
    assert bf.LIMIT_TOOL_SPEED == {v: k for k, v in constants('BaseState.msg', 'LIMIT_').items()
                                   }['TOOL_SPEED']


# ---- exactly once ----

def test_first_message_is_a_baseline_and_repeats_give_nothing():
    e = BaseEvents()
    assert feed(e, [sample(5, STOPPED)] * 4) == [None] * 4
    assert e.baselines == 1


def test_each_new_event_gives_exactly_one_request():
    e = BaseEvents()
    out = feed(e, [sample(0, 0), sample(1, START), sample(1, START), sample(2, RELEASED),
                   sample(3, AT_REST), sample(3, AT_REST), sample(4, REFUSED),
                   sample(5, STOPPED), sample(5, STOPPED)])
    assert names(out) == [None, 'START', None, None, 'AT_REST', None, 'REFUSED', 'STOPPED', None]
    assert all(r.missed == 0 for r in out if r)
    assert out[1].source_count == 1 and out[1].pattern == bf.DEFAULT_PATTERNS['START']


def test_count_up_by_k_gives_the_newest_event_and_counts_k_minus_one_missed():
    e = BaseEvents()
    out = feed(e, [sample(1, START), sample(4, REFUSED)])
    assert out[0] is None
    assert out[1].event == 'REFUSED' and out[1].missed == 2 and out[1].source_count == 4
    assert e.missed_total == 2


def test_count_going_back_is_a_new_baseline_not_an_event():
    e = BaseEvents()
    out = feed(e, [sample(7, AT_REST), sample(2, START), sample(2, START), sample(3, AT_REST)])
    assert names(out) == [None, None, None, 'AT_REST']
    assert e.baselines == 2 and e.missed_total == 0


def test_an_old_event_is_not_sent_and_is_counted():
    e = BaseEvents(max_age_s=0.75)
    out = feed(e, [sample(1, START), sample(2, START, wall=NOW - 0.8)])
    assert out == [None, None] and e.too_old == 1
    assert e.update(sample(3, START, wall=NOW - 0.7), NOW).source_count == 3
    assert e.update(sample(3, START, wall=NOW - 0.7), NOW) is None


def test_an_event_with_no_time_is_not_sent():
    e = BaseEvents()
    out = feed(e, [sample(1, START), sample(2, START, wall=math.nan)])
    assert out == [None, None] and e.too_old == 1


def test_no_event_and_unknown_events_give_nothing():
    e = BaseEvents()
    assert feed(e, [sample(1, 0), sample(2, 0), sample(3, 99)]) == [None] * 3


# ---- patterns and hands ----

@pytest.mark.parametrize('code,event,pulses,priority,hand', [
    (START, 'START', ((0.6, 100, 0),), 2, 'left'),
    (REFUSED, 'REFUSED', ((1.0, 350, 200), (1.0, 350, 0)), 3, 'left'),
    (STOPPED, 'STOPPED', ((1.0, 350, 200), (1.0, 350, 0)), 3, 'left'),
    (AT_REST, 'AT_REST', ((0.5, 60, 0),), 2, 'right'),
])
def test_event_patterns_and_hands(code, event, pulses, priority, hand):
    e = BaseEvents()
    e.update(sample(1, 0), NOW)
    r = e.update(sample(2, code), NOW)
    assert r.event == event and r.pattern.pulses == pulses and r.pattern.priority == priority
    assert r.hand == hand and r.wall_time == NOW and r.missed == 0


def test_the_at_rest_pulse_is_felt_on_the_other_hand_than_the_drive():
    assert bf.DEFAULT_HANDS['AT_REST'] == 'right'
    assert {bf.DEFAULT_HANDS[e] for e in ('START', 'REFUSED', 'STOPPED')} == {'left'}
    assert set(bf.DEFAULT_HANDS) == set(bf.DEFAULT_PATTERNS) == set(bf.EVENTS)


def test_hands_are_configurable_per_event_and_checked():
    e = BaseEvents(hands={'START': 'right', 'AT_REST': 'left'})
    e.update(sample(1, 0), NOW)
    assert e.update(sample(2, START), NOW).hand == 'right'
    assert e.update(sample(3, AT_REST), NOW).hand == 'left'
    assert e.update(sample(4, STOPPED), NOW).hand == 'left'       # the default of the others
    with pytest.raises(ValueError):
        BaseEvents(hands={'START': 'both'})


def test_a_release_has_no_vibration():
    e = BaseEvents()
    e.update(sample(1, 0), NOW)
    assert e.update(sample(2, RELEASED), NOW) is None
    assert e.too_old == 0 and e.missed_total == 0
    assert 'RELEASED' not in bf.DEFAULT_PATTERNS and 'RELEASED' not in bf.EVENTS


def test_a_silent_pattern_makes_no_request():
    e = BaseEvents({'START': Pattern.from_flat([0.0, 200.0, 0.0], 2)})
    e.update(sample(1, 0), NOW)
    assert e.update(sample(2, START), NOW) is None


def test_config_matches_the_defaults():
    params = yaml.safe_load(open(CONFIG, encoding='utf-8'))['operator_feedback']['ros__parameters']
    assert params['base_stale_s'] == 0.5
    assert set(params['base_haptic']) == set(bf.DEFAULT_PATTERNS) == set(bf.EVENTS)
    assert set(params['base_haptic_priority']) == set(bf.EVENTS)
    assert params['base_hand'] == bf.DEFAULT_HANDS
    for event, flat in params['base_haptic'].items():
        assert all(isinstance(v, float) for v in flat), event
        assert Pattern.from_flat(flat, params['base_haptic_priority'][event]) == \
            bf.DEFAULT_PATTERNS[event], event


# ---- wording ----

CODE = {name: code for code, name in bf.REASON_NAMES.items()}
DRIVING_WORDS = 'Base: driving, release left grip to stop (stick up = robot front)'
TURN_WORDS = 'Base: driving; turn ignored, centre the left stick to turn'
REASON_WORDS = [
    ('OFF', 'off (parameter)'),
    ('NOT_SIMULATION', 'simulation only'),
    ('RELEASED', 'released'),
    ('INPUT_STALE', 'headset input late or lost'),
    ('CONTROLLER_MISSING', 'left controller missing'),
    ('SESSION_CHANGED', 'headset session changed'),
    ('SEQ_BACKWARDS', 'input out of order'),
    ('CLUTCH_ENGAGED', 'release the arm clutch (right grip) first'),
    ('RIGHT_GRIP_HELD', 'release the right grip (arm clutch) first'),
    ('TELEOP_UNKNOWN', 'arm state unknown'),
    ('ODOM_MISSING', 'base odometry missing'),
    ('RESET_ACTIVE', 'reset running'),
    ('RESET_UNKNOWN', 'reset state unknown'),
    ('OTHER_PUBLISHER', 'another node publishes the base command'),
    ('TOOL_POSE_UNKNOWN', 'arm posture unknown (tool speed limit)'),
    ('SHUTDOWN', 'base node stopping'),
]


def text(state, reason=0, age=2.0, turn_ignored=False, limit=0, mode=1, other=''):
    return bf.base_text(state, reason, mode, turn_ignored, limit, age, other)


def test_driving_words():
    assert text(DRIVING) == DRIVING_WORDS
    assert text(DRIVING, age=math.nan) == DRIVING_WORDS       # no age, no suffix, no hiding
    assert text(DRIVING, turn_ignored=True) == TURN_WORDS
    assert text(DRIVING, limit=3) == DRIVING_WORDS + ' (slowed: tool speed)'
    assert text(DRIVING, turn_ignored=True, limit=3) == TURN_WORDS + ' (slowed: tool speed)'
    # Only the tool speed limit is worded; the caps and the acceleration limit are not.
    assert text(DRIVING, limit=1) == DRIVING_WORDS and text(DRIVING, limit=2) == DRIVING_WORDS
    assert text(DRIVING, mode=2) == DRIVING_WORDS


def test_stopping_words_only_while_stopping():
    assert text(STOPPING, CODE['RELEASED']) == 'Base: stopping…'
    assert text(STOPPING, CODE['RELEASED'], age=math.nan) == 'Base: stopping…'
    assert text(IDLE, CODE['RELEASED']) == '' and text(S_STOPPED, CODE['RELEASED'], 0.0) != \
        'Base: stopping…'


@pytest.mark.parametrize('reason,words', REASON_WORDS)
def test_wording_of_refusals_and_stops(reason, words):
    assert text(S_REFUSED, CODE[reason], age=12.0) == f'Base refused: {words} (12 s ago)'
    assert text(S_STOPPED, CODE[reason], age=0.0) == f'Base stopped: {words} (0 s ago)'


def test_every_reason_has_wording():
    assert set(bf.REASON_WORDS) == {r for r in bf.REASON_NAMES.values() if r != 'NONE'}
    assert {r for r, _ in REASON_WORDS} == set(bf.REASON_WORDS)
    assert all(words and words == words.strip() for words in bf.REASON_WORDS.values())


def test_off_and_idle_have_no_line():
    for state in (OFF, IDLE):
        assert text(state) == '' and text(state, CODE['OFF'], age=1.0) == ''
        assert text(state, CODE['RELEASED'], age=1.0) == ''


def test_lines_age_for_30_seconds_then_go():
    assert text(S_REFUSED, CODE['ODOM_MISSING'], age=29.9) == \
        'Base refused: base odometry missing (29 s ago)'
    assert text(S_REFUSED, CODE['ODOM_MISSING'], age=30.0) == ''
    assert text(S_STOPPED, CODE['INPUT_STALE'], age=45.0) == ''
    assert text(S_REFUSED, CODE['ODOM_MISSING'], age=math.nan) == ''    # unknown age: never shown
    assert text(S_STOPPED, CODE['INPUT_STALE'], age=-0.3) == \
        'Base stopped: headset input late or lost (0 s ago)'


def test_another_publisher_is_named_and_clipped():
    assert text(S_REFUSED, CODE['OTHER_PUBLISHER'], other='/joy_teleop') == \
        'Base refused: another node publishes the base command: /joy_teleop (2 s ago)'
    long = text(S_REFUSED, CODE['OTHER_PUBLISHER'], other='/' + 'x' * 100)
    assert len(long) < 110 and '…' in long
    assert text(S_REFUSED, CODE['INPUT_STALE'], other='/ignored') == \
        'Base refused: headset input late or lost (2 s ago)'


def test_unknown_codes_give_a_generic_line_that_still_ages_out():
    assert text(9, 3) == 'Base: state 9, reason 3'
    assert text(S_REFUSED, 99, age=1.0) == 'Base: state 4, reason 99 (1 s ago)'
    assert text(S_STOPPED, 0, age=1.0) == 'Base: state 5, reason 0 (1 s ago)'
    assert text(S_STOPPED, 0, age=31.0) == ''
    assert text(S_STOPPED, 0, age=math.nan) == ''


# ---- the tracker and the status ----

def view(state=DRIVING, reason=0, event=0, wall=math.nan, mode=1, ignored=False, limit=0,
         reason_text=''):
    return BaseView(state, reason, mode, ignored, limit, reason_text, event, wall)


def test_nothing_received_reports_defaults_and_no_age():
    s = StatusTracker().snapshot(10.0, 100.0)
    assert (s.base_valid, s.base_state, s.base_reason, s.base_text) == (False, 0, 0, '')
    assert math.isnan(s.base_age) and math.isnan(s.base_event_age)


def test_base_fields_are_valid_for_the_stale_time_then_defaults():
    t = StatusTracker(base_stale_s=0.5)
    t.on_base(view(event=START, wall=99.0), 10.0)
    s = t.snapshot(10.4, 100.0)
    assert s.base_valid and s.base_age == pytest.approx(0.4)
    assert s.base_state == DRIVING and s.base_reason == 0
    assert s.base_text == DRIVING_WORDS
    assert s.base_event_age == pytest.approx(1.0)
    late = t.snapshot(10.6, 100.2)
    assert not late.base_valid and late.base_age == pytest.approx(0.6)
    assert (late.base_state, late.base_reason, late.base_text) == (0, 0, '')
    assert math.isnan(late.base_event_age)
    t.on_base(view(state=S_STOPPED, reason=CODE['INPUT_STALE'], event=STOPPED, wall=100.0), 10.7)
    assert t.snapshot(10.8, 100.3).base_valid


def test_stale_time_is_a_parameter():
    t = StatusTracker(base_stale_s=2.0)
    t.on_base(view(), 10.0)
    assert t.snapshot(11.9, 100.0).base_valid and not t.snapshot(12.1, 100.0).base_valid


def test_the_tracker_passes_the_drive_details_to_the_words():
    t = StatusTracker()
    t.on_base(view(ignored=True, limit=3), 10.0)
    assert t.snapshot(10.1, 100.0).base_text == TURN_WORDS + ' (slowed: tool speed)'
    t.on_base(view(state=S_REFUSED, reason=CODE['OTHER_PUBLISHER'], event=REFUSED, wall=95.0,
                   reason_text='something else publishes the base command: /a, /b'), 10.2)
    assert t.snapshot(10.3, 100.0).base_text == \
        'Base refused: another node publishes the base command: /a, /b (5 s ago)'


def test_a_refusal_ages_and_goes_in_the_snapshot_text():
    t = StatusTracker()
    t.on_base(view(state=S_REFUSED, reason=CODE['ODOM_MISSING'], event=REFUSED, wall=95.0), 10.0)
    assert t.snapshot(10.1, 100.0).base_text == 'Base refused: base odometry missing (5 s ago)'
    assert t.snapshot(10.1, 125.0).base_text == ''            # 30 s: no line (still valid)
    assert t.snapshot(10.1, 125.0).base_valid


def test_a_stop_is_aged_from_the_newest_event_so_the_arrival_at_rest_restarts_it():
    # The base node's state stays STOPPED when the base comes to rest: the newest event is then
    # AT_REST, and the age of the line is the age of that event (as BaseState.last_event_wall_time
    # says). Pinned so a change to latching the stop's own time is a deliberate one.
    t = StatusTracker()
    reason = CODE['INPUT_STALE']
    t.on_base(view(state=S_STOPPED, reason=reason, event=STOPPED, wall=90.0), 10.0)
    assert t.snapshot(10.1, 100.0).base_text.endswith('(10 s ago)')
    t.on_base(view(state=S_STOPPED, reason=reason, event=AT_REST, wall=98.0), 10.2)
    assert t.snapshot(10.3, 100.0).base_text.endswith('(2 s ago)')


def test_other_sections_are_not_changed_by_a_base_state():
    t = StatusTracker()
    t.on_base(view(), 10.0)
    s = t.snapshot(10.1, 100.0)
    assert not s.teleop_valid and not s.episode_valid and not s.reset_valid
    assert s.events_missed == 0 and math.isnan(s.reset_age)


# ---- the relay body ----

PANEL = rm.panel_config('left_controller', [0.0, 0.1, 0.0], 0.16, True)


def body(snap):
    return rm.encode(rm.status_payload(snap, 1, 1, PANEL, 1000, 200), 0.0, None)


def test_the_body_is_unchanged_without_a_base_state():
    msg = json.loads(body(Snapshot()))
    assert 'base' not in msg
    assert list(msg) == ['type', 'v', 'run', 'seq', 'stale_ms', 'period_ms', 'teleop',
                         'teleop_age_s', 'episode', 'episode_age_s', 'sim_text', 'panel',
                         'age_ms', 'echo']
    assert 'base' not in json.loads(body(Snapshot(reset_valid=True, reset_age=0.1)))


def test_a_body_without_base_attributes_is_unchanged_too():
    class Old:
        pass
    snap = Snapshot()
    old = Old()
    for name in vars(snap):
        if not name.startswith('base_'):
            setattr(old, name, getattr(snap, name))
    assert body(old) == body(snap)


def test_the_body_carries_a_base_section_once_a_state_was_seen():
    snap = Snapshot(base_valid=True, base_age=0.1, base_state=S_STOPPED,
                    base_reason=CODE['INPUT_STALE'], base_event_age=2.5,
                    base_text='Base stopped: headset input late or lost (2 s ago)')
    msg = json.loads(body(snap))
    assert msg['base'] == {
        'valid': True, 'state': 'STOPPED', 'reason': 'INPUT_STALE',
        'text': 'Base stopped: headset input late or lost (2 s ago)', 'event_age_s': 2.5}
    assert 'reset' not in msg


def test_both_sections_are_carried_independently_reset_first():
    snap = Snapshot(base_valid=True, base_age=0.1, base_state=DRIVING, reset_valid=True,
                    reset_age=0.1, reset_state=4)
    msg = json.loads(body(snap))
    assert list(msg).index('reset') < list(msg).index('base')
    assert msg['reset']['state'] == 'MOVING' and msg['base']['state'] == 'DRIVING'


def test_a_stale_base_section_has_no_old_values_and_no_nan():
    snap = Snapshot(base_valid=False, base_age=3.0)
    text_ = body(snap)
    assert 'NaN' not in text_
    assert json.loads(text_)['base'] == {'valid': False, 'state': None, 'reason': None,
                                         'text': '', 'event_age_s': None}
    nan_age = Snapshot(base_valid=True, base_age=0.1, base_state=DRIVING)
    assert json.loads(body(nan_age))['base']['event_age_s'] is None


def test_unknown_codes_in_the_body_are_named_not_crashed():
    snap = Snapshot(base_valid=True, base_age=0.0, base_state=77, base_reason=88)
    r = json.loads(body(snap))['base']
    assert r['state'] == 'UNKNOWN_77' and r['reason'] == 'UNKNOWN_88'
