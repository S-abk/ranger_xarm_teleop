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
"""The JSON the headset receives (no ROS): fields, no NaN, ages and echo."""

import json
from types import SimpleNamespace

import pytest
from ranger_xarm_teleop_bridge import relay_messages as rm
from ranger_xarm_teleop_bridge.episode_feedback import DEFAULT_PATTERNS
from ranger_xarm_teleop_bridge.operator_status import Snapshot

PANEL = rm.panel_config('left_controller', [0.0, 0.1, 0.0], 0.16, True)


def haptic(event='REFUSED'):
    p = DEFAULT_PATTERNS[event]
    return SimpleNamespace(hand='left', source='episode', event=event, priority=p.priority,
                           intensity=[i for i, _, _ in p.pulses],
                           on_ms=[on for _, on, _ in p.pulses],
                           off_ms=[off for _, _, off in p.pulses],
                           max_age=0.75, source_count=4, missed=1)


def test_haptic_message():
    payload = rm.haptic_payload(haptic(), run=7, ident=3)
    text = rm.encode(payload, 0.0123, rm.echo_of((11, 500, 1234.5, 10.0), 10.002))
    msg = json.loads(text)
    assert msg['type'] == 'teleop_haptic' and msg['v'] == 1 and msg['run'] == 7
    assert msg['id'] == 3 and msg['event'] == 'REFUSED' and msg['hand'] == 'left'
    assert msg['pulses'] == [[1.0, 350, 200], [1.0, 350, 0]]
    assert msg['priority'] == 3 and msg['max_age_ms'] == 750.0 and msg['missed'] == 1
    assert msg['age_ms'] == 12.3
    assert msg['echo'] == {'session_id': 11, 'seq': 500, 't_client': 1234.5,
                           'held_ms': pytest.approx(2.0)}


def test_mismatched_pulse_arrays_are_refused():
    h = haptic()
    h.on_ms = h.on_ms[:1]
    with pytest.raises(ValueError):
        rm.haptic_payload(h, 1, 1)


def test_status_without_sources_has_null_sections_and_no_nan():
    payload = rm.status_payload(Snapshot(), run=1, seq=1, panel=PANEL, stale_ms=1000,
                                period_ms=200)
    text = rm.encode(payload, 0.0, None)
    assert 'NaN' not in text
    msg = json.loads(text)
    assert msg['teleop'] is None and msg['episode'] is None and msg['echo'] is None
    assert msg['teleop_age_s'] is None and msg['panel'] == PANEL


def test_status_content():
    s = Snapshot(teleop_valid=True, teleop_age=0.01, armed=True, clutch_engaged=True,
                 calibrated=True, motion_enabled=True, translation_scale=0.5, hold_reason=0,
                 hold_text='', last_fault=15, last_fault_text='Hand moved too fast',
                 last_fault_age=3.2, limit=3, limit_text='keep-out zone',
                 gripper_text='closing, 50 % closed', episode_valid=True, episode_age=0.1,
                 episode_state=2, episode_index=4, elapsed=12.5, kept_count=3,
                 discarded_count=1, last_event=1, last_event_text='Episode 4 started',
                 last_event_age=12.5, last_refusal_text='arm first (B)', last_refusal_age=40.0,
                 events_missed=1)
    msg = json.loads(rm.encode(rm.status_payload(s, 2, 9, PANEL, 1000, 200), 0.05, None))
    assert msg['teleop'] == {'armed': True, 'clutch': True, 'calibrated': True,
                             'motion_enabled': True, 'scale': 0.5, 'hold': '',
                             'fault': 'Hand moved too fast', 'fault_age_s': 3.2,
                             'limit': 'keep-out zone', 'gripper': 'closing, 50 % closed'}
    e = msg['episode']
    assert (e['state'], e['index'], e['elapsed_s'], e['kept'], e['discarded']) == (
        'RECORDING', 4, 12.5, 3, 1)
    assert e['last_event'] == 'START' and e['refusal'] == 'arm first (B)' and e['missed'] == 1
    assert e['disk_minutes_left'] is None and msg['seq'] == 9 and msg['age_ms'] == 50.0


def test_a_fault_that_still_applies_is_shown_as_the_hold_only():
    s = Snapshot(teleop_valid=True, hold_reason=7, hold_text='Right controller not tracked',
                 last_fault=7, last_fault_text='Right controller not tracked', last_fault_age=0.0)
    t = rm.status_payload(s, 1, 1, PANEL, 1000, 200)['teleop']
    assert t['hold'] == 'Right controller not tracked' and t['fault'] == ''
    assert t['fault_age_s'] is None


@pytest.mark.parametrize('anchor,offset,width', [('centre', [0, 0, 0], 0.16),
                                                 ('head', [0, 0], 0.16),
                                                 ('head', [0, 0, 5.0], 0.16),
                                                 ('head', [0, 0, 0], 2.0)])
def test_bad_panel_parameters_are_refused(anchor, offset, width):
    with pytest.raises(ValueError):
        rm.panel_config(anchor, offset, width, True)
