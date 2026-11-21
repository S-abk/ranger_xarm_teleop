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
"""Reset events to vibration, the reset panel line and the reset part of the status (no ROS)."""

import json
import math
import os
import re

import pytest
from ranger_xarm_teleop_bridge import relay_messages as rm
from ranger_xarm_teleop_bridge import reset_feedback as rf
from ranger_xarm_teleop_bridge.episode_feedback import Pattern
from ranger_xarm_teleop_bridge.operator_status import ResetView, Snapshot, StatusTracker
from ranger_xarm_teleop_bridge.reset_feedback import ResetEvents, ResetSample
import yaml

MSG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                       'ranger_xarm_teleop_interfaces', 'msg'))
CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config',
                                      'operator_feedback.yaml'))
START, ARRIVED, REFUSED, STOPPED, RELEASED = range(1, 6)
NOW = 1000.0


def constants(msg, prefix):
    with open(os.path.join(MSG_DIR, msg), encoding='utf-8') as handle:
        text = handle.read()
    return {int(v): n for n, v in re.findall(r'^uint8 %s(\w+) = (\d+)' % prefix, text, re.M)}


def sample(count, event, wall=NOW):
    return ResetSample(count, event, wall)


def feed(events, samples, now=NOW):
    return [events.update(s, now) for s in samples]


def names(out):
    return [r.event if r else None for r in out]


# ---- code tables ----

def test_tables_match_the_message():
    assert rf.STATE_NAMES == constants('ResetState.msg', 'STATE_')
    assert rf.REASON_NAMES == constants('ResetState.msg', 'REASON_')
    assert rf.EVENT_NAMES == constants('ResetState.msg', 'EVENT_')


# ---- exactly once ----

def test_first_message_is_a_baseline_and_repeats_give_nothing():
    e = ResetEvents()
    assert feed(e, [sample(5, STOPPED)] * 4) == [None] * 4
    assert e.baselines == 1


def test_each_new_event_gives_exactly_one_request():
    e = ResetEvents()
    out = feed(e, [sample(0, 0), sample(1, START), sample(1, START), sample(2, ARRIVED),
                   sample(2, ARRIVED), sample(3, REFUSED), sample(4, STOPPED),
                   sample(4, STOPPED)])
    assert names(out) == [None, 'START', None, 'ARRIVED', None, 'REFUSED', 'STOPPED', None]
    assert all(r.missed == 0 for r in out if r)
    assert out[1].source_count == 1 and out[1].pattern == rf.DEFAULT_PATTERNS['START']


def test_count_up_by_k_gives_the_newest_event_and_counts_k_minus_one_missed():
    e = ResetEvents()
    out = feed(e, [sample(1, START), sample(4, REFUSED)])
    assert out[0] is None
    assert out[1].event == 'REFUSED' and out[1].missed == 2 and out[1].source_count == 4
    assert e.missed_total == 2


def test_count_going_back_is_a_new_baseline_not_an_event():
    e = ResetEvents()
    out = feed(e, [sample(7, ARRIVED), sample(2, START), sample(2, START), sample(3, ARRIVED)])
    assert names(out) == [None, None, None, 'ARRIVED']
    assert e.baselines == 2 and e.missed_total == 0


def test_an_old_event_is_not_sent_and_is_counted():
    e = ResetEvents(max_age_s=0.75)
    out = feed(e, [sample(1, START), sample(2, START, wall=NOW - 0.8)])
    assert out == [None, None] and e.too_old == 1
    # Within the age it is sent; the count was taken, so a repeat of it is nothing.
    assert e.update(sample(3, START, wall=NOW - 0.7), NOW).source_count == 3
    assert e.update(sample(3, START, wall=NOW - 0.7), NOW) is None


def test_an_event_with_no_time_is_not_sent():
    e = ResetEvents()
    out = feed(e, [sample(1, START), sample(2, START, wall=math.nan)])
    assert out == [None, None] and e.too_old == 1


def test_no_event_and_unknown_events_give_nothing():
    e = ResetEvents()
    assert feed(e, [sample(1, 0), sample(2, 0), sample(3, 99)]) == [None] * 3


# ---- patterns ----

@pytest.mark.parametrize('code,event,pulses,priority', [
    (START, 'START', ((0.8, 200, 0),), 2),
    (ARRIVED, 'ARRIVED', ((0.7, 500, 0),), 2),
    (REFUSED, 'REFUSED', ((1.0, 350, 200), (1.0, 350, 0)), 3),
    (STOPPED, 'STOPPED', ((1.0, 350, 200), (1.0, 350, 0)), 3),
])
def test_event_patterns(code, event, pulses, priority):
    e = ResetEvents()
    e.update(sample(1, 0), NOW)
    r = e.update(sample(2, code), NOW)
    assert r.event == event and r.pattern.pulses == pulses and r.pattern.priority == priority
    assert r.wall_time == NOW and r.missed == 0


def test_a_release_has_no_vibration():
    e = ResetEvents()
    e.update(sample(1, 0), NOW)
    assert e.update(sample(2, RELEASED), NOW) is None
    assert e.too_old == 0 and e.missed_total == 0
    assert 'RELEASED' not in rf.DEFAULT_PATTERNS


def test_a_silent_pattern_makes_no_request():
    e = ResetEvents({'START': Pattern.from_flat([0.0, 200.0, 0.0], 2)})
    e.update(sample(1, 0), NOW)
    assert e.update(sample(2, START), NOW) is None


def test_config_matches_the_default_patterns():
    params = yaml.safe_load(open(CONFIG, encoding='utf-8'))['operator_feedback']['ros__parameters']
    assert params['reset_hand'] == 'right' and params['reset_stale_s'] == 0.5
    assert set(params['reset_haptic']) == set(rf.DEFAULT_PATTERNS) == set(rf.EVENTS)
    for event, flat in params['reset_haptic'].items():
        assert all(isinstance(v, float) for v in flat), event
        assert Pattern.from_flat(flat, params['reset_haptic_priority'][event]) == \
            rf.DEFAULT_PATTERNS[event], event


# ---- wording ----

POSTURE = 'ready'
MOVING_WORDS = [
    (1, 'Reset: keep holding…'),
    (3, 'Reset: planning to ready…'),
    (4, 'Reset: moving to ready; release to stop'),
    (5, 'Reset: stopping…'),
]
ARRIVED_WORDS = [
    ('REACHED', 'Reset: at ready. Arm with B'),
    ('ALREADY_THERE', 'Reset: already at ready'),
]
STOPPED_WORDS = [
    ('RELEASED', 'Reset stopped: released. Pull back again to go on'),
    ('INPUT_STALE', 'Reset stopped: headset input late or lost'),
    ('CONTROLLER_MISSING', 'Reset stopped: right controller lost'),
    ('SESSION_CHANGED', 'Reset stopped: new headset session'),
    ('SEQ_BACKWARDS', 'Reset stopped: input out of order'),
    ('TELEOP_ARMED', 'Reset stopped: teleop was armed'),
    ('OWNER_CHANGED', 'Reset stopped: arm handed over'),
    ('EPISODE_STARTED', 'Reset stopped: an episode started'),
    ('WATCHDOG_LOST', 'Reset stopped: watchdog lost'),
    ('HEARTBEAT_LATE', 'Reset stopped: heartbeat late'),
    ('CANCELLED_ELSEWHERE', 'Reset stopped: cancelled elsewhere'),
    ('MOVEIT_FAILED', 'Reset stopped: MoveIt stopped the motion (code -31)'),
    ('NO_ANSWER', 'Reset stopped: MoveIt did not answer'),
    ('STATE_UNKNOWN', 'Reset stopped: arm or teleop state lost'),
]
REFUSED_WORDS = [
    ('OFF', 'Reset refused: off (parameter)'),
    ('NOT_SIMULATION', 'Reset refused: simulation only'),
    ('ARMED', 'Reset refused: disarm first (B)'),
    ('NOT_MOVEIT_OWNER', 'Reset refused: arm not handed to MoveIt yet'),
    ('ARMED_UNKNOWN', 'Reset refused: teleop state unknown'),
    ('RECORDING', 'Reset refused: recording, stop the episode (X)'),
    ('REVIEW', 'Reset refused: keep (Y) or discard (hold Y) first'),
    ('EPISODE_UNKNOWN', 'Reset refused: recording state unknown'),
    ('WATCHDOG_ABSENT', 'Reset refused: watchdog not running'),
    ('MOVEIT_NOT_READY', 'Reset refused: MoveIt not ready'),
    ('PLAN_FAILED', 'Reset refused: no path to ready; nothing moved'),
    ('START_IN_COLLISION', 'Reset refused: arm in collision; nothing moved. Disarm, use the '
                           'retreat tool'),
    ('BUSY', 'Reset refused: still stopping, wait'),
    ('JOINTS_UNKNOWN', 'Reset refused: arm position unknown'),
]
CODE = {name: code for code, name in rf.REASON_NAMES.items()}


def text(state, reason=0, age=2.0, posture=POSTURE, code=-31):
    return rf.reset_text(state, reason, posture, code, age)


@pytest.mark.parametrize('state,words', MOVING_WORDS)
def test_wording_of_a_motion_in_progress(state, words):
    assert text(state) == words
    assert text(state, age=math.nan) == words        # no age, no suffix, no hiding


@pytest.mark.parametrize('reason,words', ARRIVED_WORDS)
def test_wording_of_arrivals(reason, words):
    assert text(7, CODE[reason], age=3.9) == words + ' (3 s ago)'


@pytest.mark.parametrize('reason,words', STOPPED_WORDS)
def test_wording_of_stops(reason, words):
    assert text(6, CODE[reason], age=0.0) == words + ' (0 s ago)'


@pytest.mark.parametrize('reason,words', REFUSED_WORDS)
def test_wording_of_refusals(reason, words):
    assert text(2, CODE[reason], age=12.0) == words + ' (12 s ago)'


def test_every_reason_has_a_wording():
    refusals = {r for r in rf.REASON_NAMES.values() if 1 <= CODE[r] <= 14}
    assert refusals == set(rf.REFUSED_TEXT) == {r for r, _ in REFUSED_WORDS}
    assert {r for r in rf.REASON_NAMES.values() if 30 <= CODE[r] <= 43} == set(rf.STOPPED_TEXT)
    assert {r for r in rf.REASON_NAMES.values() if 20 <= CODE[r] <= 21} == set(rf.ARRIVED_TEXT)


def test_idle_has_no_line():
    assert text(0) == '' and text(0, 30, age=1.0) == ''


def test_lines_age_for_30_seconds_then_go():
    assert text(2, CODE['ARMED'], age=29.9) == 'Reset refused: disarm first (B) (29 s ago)'
    assert text(2, CODE['ARMED'], age=30.0) == ''
    assert text(6, CODE['RELEASED'], age=45.0) == ''
    assert text(7, CODE['REACHED'], age=30.0) == ''
    assert text(2, CODE['ARMED'], age=math.nan) == ''        # an unknown age is never shown


def test_unknown_codes_give_a_generic_line():
    assert text(9, 3) == 'Reset: state 9, reason 3'
    # In a state that ages, an unknown reason still ages out like the known ones.
    assert text(2, 99, age=1.0) == 'Reset: state 2, reason 99 (1 s ago)'
    assert text(6, 0, age=1.0) == 'Reset: state 6, reason 0 (1 s ago)'
    assert text(6, 0, age=31.0) == ''


def test_posture_name_is_tidy():
    assert text(3, posture='') == 'Reset: planning to the posture…'
    assert len(text(3, posture='x' * 100)) < 70


# ---- the tracker and the status ----

def view(state=4, reason=0, event=0, wall=math.nan, posture=POSTURE, code=0):
    return ResetView(state, reason, posture, code, event, wall)


def test_nothing_received_reports_defaults_and_no_age():
    s = StatusTracker().snapshot(10.0, 100.0)
    assert (s.reset_valid, s.reset_state, s.reset_reason, s.reset_text) == (False, 0, 0, '')
    assert math.isnan(s.reset_age) and math.isnan(s.reset_event_age)


def test_reset_fields_are_valid_for_the_stale_time_then_defaults():
    t = StatusTracker(reset_stale_s=0.5)
    t.on_reset(view(state=4, event=START, wall=99.0), 10.0)
    s = t.snapshot(10.4, 100.0)
    assert s.reset_valid and s.reset_age == pytest.approx(0.4)
    assert s.reset_state == 4 and s.reset_reason == 0
    assert s.reset_text == 'Reset: moving to ready; release to stop'
    assert s.reset_event_age == pytest.approx(1.0)
    late = t.snapshot(10.6, 100.2)
    assert not late.reset_valid and late.reset_age == pytest.approx(0.6)
    assert (late.reset_state, late.reset_reason, late.reset_text) == (0, 0, '')
    assert math.isnan(late.reset_event_age)
    t.on_reset(view(state=7, reason=CODE['REACHED'], event=ARRIVED, wall=100.0), 10.7)
    assert t.snapshot(10.8, 100.3).reset_valid


def test_stale_time_is_a_parameter():
    t = StatusTracker(reset_stale_s=2.0)
    t.on_reset(view(), 10.0)
    assert t.snapshot(11.9, 100.0).reset_valid and not t.snapshot(12.1, 100.0).reset_valid


def test_a_refusal_ages_and_goes_in_the_snapshot_text():
    t = StatusTracker()
    t.on_reset(view(state=2, reason=CODE['ARMED'], event=REFUSED, wall=95.0), 10.0)
    assert t.snapshot(10.1, 100.0).reset_text == 'Reset refused: disarm first (B) (5 s ago)'
    assert t.snapshot(10.1, 125.0).reset_text == ''           # 30 s: no line (still valid)
    assert t.snapshot(10.1, 125.0).reset_valid


def test_other_sections_are_not_changed_by_a_reset_state():
    t = StatusTracker()
    t.on_reset(view(), 10.0)
    s = t.snapshot(10.1, 100.0)
    assert not s.teleop_valid and not s.episode_valid and s.events_missed == 0


# ---- the relay body ----

PANEL = rm.panel_config('left_controller', [0.0, 0.1, 0.0], 0.16, True)


def body(snap):
    return rm.encode(rm.status_payload(snap, 1, 1, PANEL, 1000, 200), 0.0, None)


def test_the_body_is_unchanged_without_a_reset_state():
    msg = json.loads(body(Snapshot()))
    assert 'reset' not in msg
    assert list(msg) == ['type', 'v', 'run', 'seq', 'stale_ms', 'period_ms', 'teleop',
                         'teleop_age_s', 'episode', 'episode_age_s', 'sim_text', 'panel',
                         'age_ms', 'echo']
    assert 'reset' not in body(Snapshot(teleop_valid=True, teleop_age=0.1, episode_age=0.2))


def test_a_body_without_reset_attributes_is_unchanged_too():
    class Old:
        pass
    snap = Snapshot()
    old = Old()
    for name in vars(snap):
        if not name.startswith('reset_'):
            setattr(old, name, getattr(snap, name))
    assert body(old) == body(snap)


def test_the_body_carries_a_reset_section_once_a_state_was_seen():
    snap = Snapshot(reset_valid=True, reset_age=0.1, reset_state=6,
                    reset_reason=CODE['INPUT_STALE'], reset_event_age=2.5,
                    reset_text='Reset stopped: headset input late or lost (2 s ago)')
    assert json.loads(body(snap))['reset'] == {
        'valid': True, 'state': 'STOPPED', 'reason': 'INPUT_STALE',
        'text': 'Reset stopped: headset input late or lost (2 s ago)', 'event_age_s': 2.5}


def test_a_stale_reset_section_has_no_old_values_and_no_nan():
    snap = Snapshot(reset_valid=False, reset_age=3.0)
    text_ = body(snap)
    assert 'NaN' not in text_
    assert json.loads(text_)['reset'] == {'valid': False, 'state': None, 'reason': None,
                                          'text': '', 'event_age_s': None}
    nan_age = Snapshot(reset_valid=True, reset_age=0.1, reset_state=4)
    assert json.loads(body(nan_age))['reset']['event_age_s'] is None


def test_unknown_codes_in_the_body_are_named_not_crashed():
    snap = Snapshot(reset_valid=True, reset_age=0.0, reset_state=77, reset_reason=88)
    r = json.loads(body(snap))['reset']
    assert r['state'] == 'UNKNOWN_77' and r['reason'] == 'UNKNOWN_88'
