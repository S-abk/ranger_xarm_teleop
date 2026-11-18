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
"""The panel's words and latching (no ROS): every code has words, faults and limits are kept."""

import math
import os
import re

import pytest
from ranger_xarm_teleop_bridge import operator_status as st
from ranger_xarm_teleop_bridge.operator_status import (EpisodeView, StatusTracker,
                                                       TeleopSample)

MSG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                       'ranger_xarm_teleop_interfaces', 'msg'))


def constants(msg, prefix):
    with open(os.path.join(MSG_DIR, msg), encoding='utf-8') as handle:
        text = handle.read()
    return {int(v): n for n, v in re.findall(r'^uint8 %s(\w+) = (\d+)' % prefix, text, re.M)}


def fields(msg):
    with open(os.path.join(MSG_DIR, msg), encoding='utf-8') as handle:
        lines = [ln.split('#')[0].split() for ln in handle]
    return [ln[1] for ln in lines if len(ln) == 2]


@pytest.mark.parametrize('names,prefix,msg', [
    (st.HOLD_NAMES, 'HOLD_', 'TeleopState.msg'),
    (st.LIMIT_NAMES, 'LIMIT_', 'TeleopState.msg'),
    (st.GRIPPER_HOLD_NAMES, 'GRIPPER_HOLD_', 'TeleopState.msg'),
    (st.EPISODE_STATE_NAMES, 'STATE_', 'EpisodeState.msg'),
])
def test_code_tables_match_the_messages(names, prefix, msg):
    assert names == constants(msg, prefix)


def test_every_code_has_plain_words():
    for name in st.HOLD_NAMES.values():
        assert name in st.HOLD_TEXT and (st.HOLD_TEXT[name] or name == 'NONE')
    for name in st.LIMIT_NAMES.values():
        assert name in st.LIMIT_TEXT and name in st.LIMIT_SEVERITY
        assert st.LIMIT_TEXT[name] or name == 'NONE'
    for name in st.GRIPPER_HOLD_NAMES.values():
        assert name in st.GRIPPER_HOLD_TEXT
    for text in list(st.HOLD_TEXT.values()) + list(st.LIMIT_TEXT.values()):
        assert '_' not in text                         # words, not codes


def test_snapshot_fields_are_the_message_fields():
    msg_fields = [f for f in fields('OperatorStatus.msg') if f not in ('header', 'wall_time')]
    assert list(st.Snapshot.__dataclass_fields__) == msg_fields


def test_servo_halted_says_what_to_do_first():
    assert st.hold_text(19) == ('Arm halted near a collision or joint limit: disarm (B), '
                                'then use the retreat tool')


@pytest.mark.parametrize('detail,words', [
    ('teleop is not armed', 'arm first (B)'),
    ('not ready: teleop is not armed', 'arm first (B)'),
    ('teleop does not own the arm yet', 'arm not handed over yet'),
    ('keep (Y) or discard (hold Y) the last episode first', 'keep (Y) or discard (hold Y) first'),
    ('the recorder is busy with the last press', 'recorder busy, press again'),
    ('camera_wrist: missing; teleop is not armed', 'camera_wrist: missing (+1 more)'),
    ('not ready: ', 'not ready'),
])
def test_refusal_words(detail, words):
    assert st.refusal_words(detail) == words


def test_long_details_are_clipped():
    assert len(st.refusal_words('x' * 500)) == st.MAX_TEXT


def teleop(**kw):
    base = {'motion_enabled': True, 'armed': True, 'clutch_engaged': False, 'calibrated': True,
            'translation_scale': 0.5, 'hold_reason': 5, 'active_limit': 0,
            'gripper_enabled': True, 'gripper_active': True, 'gripper_hold': 0,
            'gripper_request': 2, 'gripper_measured_valid': True, 'gripper_measured': 0.4245}
    base.update(kw)
    return TeleopSample(**base)


def episode(**kw):
    base = {'state': 2, 'episode_index': 3, 'elapsed': 12.5, 'kept_count': 2,
            'discarded_count': 1, 'discard_armed': False, 'last_event': 1, 'detail': '',
            'event_wall_time': 1000.0, 'event_count': 7, 'disk_minutes_left': float('nan')}
    base.update(kw)
    return EpisodeView(**base)


def test_one_cycle_fault_is_latched_for_the_latch_time():
    t = StatusTracker(fault_latch_s=10.0)
    t.on_teleop(teleop(hold_reason=15, clutch_engaged=True), 0.0)    # INPUT_JUMP, one cycle
    for i in range(1, 50):
        t.on_teleop(teleop(hold_reason=5), 0.02 * i)
    s = t.snapshot(0.98, 0.0)
    assert s.hold_reason == 5 and s.last_fault == 15 and 'too fast' in s.last_fault_text
    assert abs(s.last_fault_age - 0.98) < 1e-9
    for i in range(50, 600):
        t.on_teleop(teleop(hold_reason=5), 0.02 * i)
    assert t.snapshot(12.0, 0.0).last_fault == 0


def test_state_holds_are_not_faults():
    t = StatusTracker()
    for code in (0, 1, 2, 3, 4, 5):
        t.on_teleop(teleop(hold_reason=code), 0.0)
    assert t.snapshot(0.0, 0.0).last_fault == 0


def test_limit_is_held_and_the_most_severe_wins():
    t = StatusTracker(limit_hold_s=1.0)
    t.on_teleop(teleop(active_limit=3), 0.0)     # keep-out
    t.on_teleop(teleop(active_limit=6), 0.1)     # speed: less severe, does not replace
    t.on_teleop(teleop(active_limit=0), 0.2)
    s = t.snapshot(0.5, 0.0)
    assert s.limit == 3 and s.limit_text == 'keep-out zone'
    assert t.snapshot(1.01, 0.0).limit == 0
    t.on_teleop(teleop(active_limit=6), 1.1)     # held one expired: a lesser one shows
    assert t.snapshot(1.2, 0.0).limit == 6


def test_stale_sources_report_no_old_values():
    t = StatusTracker(teleop_stale_s=0.5, episode_stale_s=1.0)
    t.on_teleop(teleop(), 0.0)
    t.on_episode(episode(), 0.0)
    s = t.snapshot(0.4, 1000.4)
    assert s.teleop_valid and s.armed and s.episode_valid and s.episode_index == 3
    s = t.snapshot(0.6, 1000.6)
    assert not s.teleop_valid and not s.armed and s.hold_text == '' and s.teleop_age == 0.6
    assert s.episode_valid
    s = t.snapshot(1.1, 1001.1)
    assert not s.episode_valid and s.episode_index == 0 and s.last_event_text == ''


def test_nothing_received_is_not_valid():
    s = StatusTracker().snapshot(0.0, 0.0)
    assert not s.teleop_valid and not s.episode_valid and math.isnan(s.teleop_age)


def test_gripper_words():
    t = StatusTracker(gripper_open=0.0, gripper_closed=0.849)
    t.on_teleop(teleop(), 0.0)
    s = t.snapshot(0.0, 0.0)
    assert s.gripper_text == 'closing, 50 % closed' and abs(s.gripper_closed_fraction - 0.5) < 1e-3
    t.on_teleop(teleop(gripper_hold=6), 0.0)
    assert t.snapshot(0.0, 0.0).gripper_text.startswith('pull the trigger')
    t2 = StatusTracker()
    t2.on_teleop(teleop(gripper_request=0), 0.0)
    assert t2.snapshot(0.0, 0.0).gripper_text == 'still, at 0.42'
    t2.on_teleop(teleop(gripper_enabled=False), 0.0)
    assert t2.snapshot(0.0, 0.0).gripper_text == ''


def test_refusal_is_kept_after_later_events():
    t = StatusTracker()
    t.on_episode(episode(state=1, last_event=6, detail='teleop is not armed',
                         event_wall_time=1000.0, event_count=4), 0.0)
    t.on_episode(episode(state=2, last_event=1, event_wall_time=1003.0, event_count=5), 0.1)
    s = t.snapshot(0.2, 1005.0)
    assert s.last_refusal_text == 'arm first (B)' and s.last_refusal_age == 5.0
    assert s.last_event_text == 'Episode 3 started' and s.last_event_age == 2.0


@pytest.mark.parametrize('event,detail,text', [
    (2, '', 'Episode 3 stopped'), (3, '', 'Episode 3 kept'), (4, '', 'Episode 3 discarded'),
    (5, '', 'Mark set'), (6, 'teleop is not armed', 'Refused: arm first (B)'),
    (7, 'disarmed', 'Episode 3 stopped by itself: teleop was disarmed'),
    (7, 'disk low', 'Episode 3 stopped by itself: disk almost full'),
])
def test_event_texts(event, detail, text):
    t = StatusTracker()
    t.on_episode(episode(last_event=event, detail=detail), 0.0)
    assert t.snapshot(0.0, 1000.0).last_event_text == text
