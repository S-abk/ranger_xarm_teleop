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
"""Tests of the recording analysis on the synthetic fixtures and on hand-made records."""

import json
import os

import pytest

from ranger_xarm_teleop_bridge.analysis import analyze, format_report
from ranger_xarm_teleop_bridge.jsonl_io import read_records, Record

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fixtures')


def records(name):
    return read_records(os.path.join(FIXTURES, name + '.jsonl'))


def frame(seq, t, pos=(0.2, 0.9, -0.4), reset=0, client_offset=0.0, hand='right',
          emulated=False, session=7):
    """Return a Record for one frame received at ``t``, ``client_offset`` ms after it was sent."""
    msg = {'type': 'xr_frame', 't_client': t * 1000.0 - 100.0 - client_offset,
           't_xr': t * 1000.0 - 100.5, 'seq': seq, 'reset_count': reset, 'session_id': session,
           'controllers': {}, 'viewer': None}
    if hand:
        msg['controllers'][hand] = {'position': list(pos), 'orientation': [0.0, 0.0, 0.0, 1.0],
                                    'emulated_position': emulated, 'buttons': [], 'axes': []}
    return Record(t_recv=t, msg=msg)


def test_still_controller_is_still_while_the_head_turns():
    report = analyze(records('still_controller'), hand='right')
    sp = report['spread']
    assert sp['max_deviation_mm'] < 1.0 and sp['pass_position']
    assert sp['max_angle_deg'] < 0.1 and sp['pass_angle']
    assert report['absent_intervals'] == [] and report['emulated_intervals'] == []
    assert report['resets']['count'] == 0 and report['seq']['missing'] == 0
    assert report['viewer']['yaw_span_deg'] > 170
    assert report['viewer']['position_range_m'][0] > 0.2
    assert 60 < report['mean_rate_hz'] < 80
    assert report['longest_gap']['t_recv_s'] < 0.03


def test_dropped_controller_intervals_and_reacquisition():
    report = analyze(records('dropped_controller'), hand='right')
    absent = report['absent_intervals']
    emulated = report['emulated_intervals']
    assert len(absent) == 1 and absent[0]['start'] == pytest.approx(2.0, abs=0.03)
    assert absent[0]['end'] == pytest.approx(3.0, abs=0.03)
    assert len(emulated) == 1 and emulated[0]['start'] == pytest.approx(4.0, abs=0.03)
    assert emulated[0]['end'] == pytest.approx(5.0, abs=0.03)
    jumps = report['reacquisition_jumps']
    assert [j['reacquired_at'] for j in jumps] == [pytest.approx(3.0, abs=0.03),
                                                   pytest.approx(5.0, abs=0.03)]
    assert jumps[0]['position_jump_mm'] < 2.0 and jumps[0]['pass']
    assert jumps[1]['position_jump_mm'] < 2.0 and jumps[1]['pass']
    assert len(report['untracked_intervals']) == 2
    # Emulated frames are excluded from the spread, so the 2 cm drift does not show.
    assert report['spread']['max_deviation_mm'] < 1.0
    left = analyze(records('dropped_controller'), hand='left')
    assert left['absent_intervals'] == [] and left['emulated_intervals'] == []


def test_reset_event_is_listed():
    report = analyze(records('reset_event'), hand='left')
    assert report['resets']['count'] == 1
    assert report['resets']['events'][0]['at'] == pytest.approx(2.5, abs=0.03)
    assert report['resets']['final_reset_count'] == 1
    # The jump at the reset makes the whole file fail the still test; a window before it passes.
    assert not report['spread']['pass_position']
    before = analyze(records('reset_event'), hand='left', t_to=2.4)
    assert before['spread']['pass_position'] and before['resets']['count'] == 0


def test_new_session_is_listed_like_a_reset():
    report = analyze(records('new_session'), hand='right')
    assert report['resets']['count'] == 0
    assert report['sessions']['count'] == 1
    event = report['sessions']['events'][0]
    assert event['at'] == pytest.approx(2.5, abs=0.03)
    assert (event['from'], event['to']) == (0x12345678, 0xCAFE0042)
    assert event['seq'] == 0
    assert report['sessions']['first_session_id'] == 0x12345678
    assert report['sessions']['final_session_id'] == 0xCAFE0042
    # seq restarted: reported as a jump, not as missing frames.
    assert report['seq']['jumps'] == [{'from': 179, 'to': 0, 'at': pytest.approx(2.5, abs=0.03)}]
    assert report['seq']['missing'] == 0
    assert not report['spread']['pass_position']
    before = analyze(records('new_session'), hand='right', t_to=2.4)
    assert before['spread']['pass_position'] and before['sessions']['count'] == 0
    text = format_report(report)
    assert 'XR session changes: 1' in text and f'{0x12345678} -> {0xCAFE0042}' in text


def test_session_changes_on_hand_made_records():
    recs = [frame(0, 10.0), frame(1, 10.014), frame(2, 10.028, session=9),
            frame(3, 10.042, session=9), frame(4, 10.056, session=11)]
    report = analyze(recs, hand='right')
    assert report['sessions']['count'] == 2
    assert [(e['from'], e['to'], e['seq']) for e in report['sessions']['events']] == [
        (7, 9, 2), (9, 11, 4)]
    assert report['sessions']['final_session_id'] == 11
    assert report['resets']['count'] == 0


def test_moving_controller_is_not_still():
    report = analyze(records('moving_controller'), hand='right')
    assert report['spread']['max_deviation_mm'] > 100
    assert not report['spread']['pass_position']
    assert not report['spread']['pass_angle']


def test_malformed_fixture_counts():
    report = analyze(records('malformed'), hand='right')
    c = report['counts']
    assert c['malformed'] > 10 and c['raw_lines'] >= 2 and c['ignored'] == 2
    assert set(c['ignored_types']) == {'camera_list', 'echo'}
    assert c['frames'] >= 4


def test_gaps_seq_and_timing_on_hand_made_records():
    recs = [frame(0, 10.0), frame(1, 10.014, client_offset=2.0),
            frame(2, 10.028, client_offset=0.5), frame(5, 10.300),
            frame(6, 10.314, client_offset=9.0)]
    report = analyze(recs, hand='right')
    assert report['seq'] == {'first': 0, 'last': 6, 'missing': 2,
                             'jumps': [{'from': 2, 'to': 5, 'at': pytest.approx(0.3)}]}
    assert report['longest_gap']['t_recv_s'] == pytest.approx(0.272)
    assert report['longest_gap']['t_recv_at'] == pytest.approx(0.3)
    t = report['timing']
    assert t['clock_offset_ms'] == pytest.approx(100.0)
    assert t['jitter_ms']['max'] == pytest.approx(9.0)
    assert t['jitter_ms']['p50'] == pytest.approx(0.5)


def test_window_and_empty_cases():
    recs = [frame(i, 10.0 + i * 0.01) for i in range(100)]
    full = analyze(recs, hand='right')
    assert full['frames_in_window'] == 100
    part = analyze(recs, hand='right', t_from=0.5, t_to=0.6)
    assert 10 <= part['frames_in_window'] <= 12
    assert 'error' in analyze(recs, hand='right', t_from=5.0)
    assert 'error' in analyze([], hand='right')
    assert analyze(recs, hand='left')['spread'] is None
    with pytest.raises(ValueError):
        analyze(recs, hand='head')


def test_format_report_and_json_round_trip():
    report = analyze(records('dropped_controller'), hand='right')
    text = format_report(report)
    assert 'absent' in text and 'emulated' in text and 'PASS' in text and 'clock offset' in text
    json.dumps(report)
    assert 'error' in format_report(analyze([], hand='right'))
