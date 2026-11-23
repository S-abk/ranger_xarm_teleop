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


def test_the_no_answer_status_matches_the_message():
    assert constants('ArmOwner.msg', 'STATUS_')[st.MUX_NO_ANSWER] == 'NO_ANSWER'


def test_waiting_for_the_arm_says_when_the_request_is_repeated():
    """Contract, service calls: an unanswered request for the arm is asked again; say so."""
    assert st.hold_text(3) == 'Waiting for the arm (MoveIt owns it)'
    assert st.hold_text(3, st.MUX_NO_ANSWER) == st.NOT_OWNER_NO_ANSWER_TEXT
    assert 'asking for the arm again' in st.NOT_OWNER_NO_ANSWER_TEXT
    assert '_' not in st.NOT_OWNER_NO_ANSWER_TEXT
    assert st.hold_text(5, st.MUX_NO_ANSWER) == st.HOLD_TEXT['CLUTCH_RELEASED']
    tracker = StatusTracker()
    tracker.on_teleop(teleop(hold_reason=3, mux_status=st.MUX_NO_ANSWER), 1.0)
    assert tracker.snapshot(1.1, 100.0).hold_text == st.NOT_OWNER_NO_ANSWER_TEXT
    tracker.on_teleop(teleop(hold_reason=3, mux_status=0), 1.2)
    assert tracker.snapshot(1.3, 100.0).hold_text == 'Waiting for the arm (MoveIt owns it)'


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


def test_with_motion_disabled_the_limit_is_shown_as_the_previews():
    """Nothing moves then: the limit line says what would act on the previewed target."""
    t = StatusTracker()
    t.on_teleop(teleop(motion_enabled=False, armed=False, hold_reason=1, active_limit=9), 0.0)
    s = t.snapshot(0.1, 0.0)
    assert s.limit == 9 and s.limit_text == 'near robot body (preview)'
    t = StatusTracker()
    t.on_teleop(teleop(active_limit=9), 0.0)
    assert t.snapshot(0.1, 0.0).limit_text == 'near robot body'


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


# ---- SimSpeed: the ROS clock against the wall clock ----

def feed(speed, factor, seconds, ros0=100.0, wall0=1000.0, period=0.2):
    """Feed statuses every ``period`` s of wall time with ROS time at ``factor``; last result."""
    out = None
    for i in range(int(round(seconds / period)) + 1):
        out = speed.update(ros0 + factor * period * i, wall0 + period * i)
    return out


def test_sim_speed_real_time_is_never_slow():
    s = st.SimSpeed()
    assert feed(s, 1.0, 10.0) == pytest.approx(1.0)
    assert s.text() == ''


def test_sim_speed_slow_simulator_is_said_with_one_decimal():
    s = st.SimSpeed()
    assert feed(s, 0.36, 1.0) is None                 # not a full window yet
    assert s.text() == ''
    assert feed(s, 0.36, 10.0) == pytest.approx(0.36)
    assert s.text() == 'Simulator slow: 0.4x real time'


def test_sim_speed_hysteresis():
    s = st.SimSpeed(slow_below=0.6, clear_above=0.7)
    clock = {'ros': 100.0, 'wall': 1000.0}

    def run(factor, seconds):
        for _ in range(int(round(seconds / 0.2))):
            clock['ros'] += factor * 0.2
            clock['wall'] += 0.2
            s.update(clock['ros'], clock['wall'])

    run(0.5, 6.0)
    assert s.slow
    run(0.65, 6.0)                                    # between the two: stays slow
    assert s.slow and s.text().startswith('Simulator slow: 0.')
    run(0.8, 6.0)
    assert not s.slow and s.text() == ''
    run(0.65, 6.0)                                    # between the two from fast: stays clear
    assert not s.slow


def test_sim_speed_restart_of_the_clock_or_a_wall_gap_starts_again():
    s = st.SimSpeed()
    feed(s, 0.3, 6.0)
    assert s.slow
    assert s.update(5.0, 1006.4) is None              # ROS time went back (simulation restarted)
    assert s.text() == ''
    feed(s, 0.3, 6.0, ros0=5.0, wall0=1006.6)
    assert s.slow
    assert s.update(100.0, 1100.0) is None            # 90 s without a status
    assert not s.slow
    assert s.update(float('nan'), 1100.2) is None     # ignored, nothing changes
    z = st.SimSpeed()
    for i in range(30):                               # unset stamps: unknown, never slow
        z.update(0.0, 1000.0 + 0.2 * i)
    assert z.factor is None and z.text() == ''


def test_sim_speed_text_does_not_flicker_on_a_bursty_clock():
    # The measured factor jitters around 0.45 (bursts of /clock): 0.4/0.5 by plain rounding.
    s = st.SimSpeed()
    ros, wall, texts = 100.0, 1000.0, []
    for i in range(600):                              # 120 s of statuses at 5 Hz
        ros += 0.2 * (0.42 if i % 2 else 0.48)
        wall += 0.2
        s.update(ros, wall)
        texts.append(s.text())
    shown = [t for t in texts if t]
    assert shown and all(t.startswith('Simulator slow: ') for t in shown)
    changes = sum(1 for a, b in zip(shown, shown[1:]) if a != b)
    assert changes <= 1, changes


def test_sim_speed_a_paused_simulator_is_said_so():
    s = st.SimSpeed()
    for i in range(30):
        s.update(100.0 + 0.0001 * i, 1000.0 + 0.2 * i)
    assert s.text() == 'Simulator paused or stalled'


def test_sim_speed_parameters_checked():
    for bad in ({'slow_below': 0.7, 'clear_above': 0.6}, {'slow_below': 0.0},
                {'clear_above': 1.5}, {'window_s': 0.0}):
        with pytest.raises(ValueError):
            st.SimSpeed(**bad)
