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
"""Run the headset feedback tests (node) as part of the quick tier."""

import json
import os
import shutil
import subprocess

import pytest

JS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'js'))
TEST_JS = os.path.join(JS_DIR, 'feedback.test.js')
REPLAY_JS = os.path.join(JS_DIR, 'panel_replay.js')
SESSION = 4242


def node_or_skip():
    """Return the node executable, or skip the test when there is none."""
    node = shutil.which('node') or shutil.which('nodejs')
    if node is None:
        pytest.skip('NOT RUN: node is not installed (apt package: nodejs)')
    return node


def echo(t_ms, down_ms=20.0, session=SESSION):
    """Return an echo whose downlink is down_ms at time t_ms."""
    return {'session_id': session, 'seq': 1, 't_client': t_ms - down_ms - 5.0, 'held_ms': 5.0}


def status(t_ms, seq, state='RECORDING'):
    """Return a teleop_status message as received at t_ms."""
    return {
        'type': 'teleop_status', 'v': 1, 'run': 1, 'seq': seq, 'age_ms': 10.0, 'stale_ms': 1000,
        'period_ms': 200, 'echo': echo(t_ms),
        'teleop': {'armed': True, 'clutch': False, 'calibrated': True, 'motion_enabled': True,
                   'scale': 0.5, 'hold': '', 'fault': '', 'fault_age_s': None, 'limit': '',
                   'gripper': ''},
        'teleop_age_s': 0.1,
        'episode': {'state': state, 'index': 7, 'elapsed_s': 42.0, 'kept': 5, 'discarded': 1,
                    'discard_armed': False, 'last_event': '', 'last_event_text': '',
                    'last_event_age_s': None, 'refusal': '', 'refusal_age_s': None, 'missed': 0,
                    'disk_minutes_left': None},
        'episode_age_s': 0.1,
        'panel': {'anchor': 'left_controller', 'offset': [0, 0.1, 0], 'width_m': 0.16,
                  'visible_default': True},
    }


def replay(records, session=SESSION):
    """Feed records (dicts) to panel_replay.js and return its output lines, parsed."""
    node = node_or_skip()
    stdin = ''.join(json.dumps(r) + '\n' for r in records)
    proc = subprocess.run([node, REPLAY_JS, '--session', str(session)], input=stdin,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines()]


def test_feedback_js_unit_tests():
    """The node:test suite for feedback.js passes (works on node 18 and later)."""
    node = node_or_skip()
    proc = subprocess.run([node, '--test', TEST_JS], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_panel_replay_goes_stale_after_a_gap():
    """A status is shown, then with no more messages the panel says NO DATA and nothing else."""
    out = replay([
        {'t_ms': 1000.0, 'msg': status(1000.0, 1)},
        {'t_ms': 1200.0, 'msg': status(1200.0, 2)},
        {'query_ms': 1300.0},
        {'query_ms': 2300.0},
        {'t_ms': 2400.0, 'msg': status(2400.0, 3, state='IDLE')},
        {'query_ms': 2500.0},
    ])
    assert [o['stale'] for o in out] == [False, True, False]
    assert out[0]['lines'][0].endswith('ep 7  0:42')
    assert out[1]['lines'] == ['NO DATA']
    assert out[2]['lines'][0] == 'IDLE  X starts ep 8'


def test_panel_replay_ignores_other_session_and_old_seq():
    """A status echoing another XR session is not accepted; an old seq does not replace a newer."""
    msg = status(1000.0, 5)
    msg['echo']['session_id'] = SESSION + 1
    out = replay([
        {'t_ms': 1000.0, 'msg': msg},
        {'query_ms': 1010.0},
        {'t_ms': 1100.0, 'msg': status(1100.0, 6)},
        {'t_ms': 1150.0, 'msg': status(1150.0, 5, state='IDLE')},
        {'query_ms': 1200.0},
    ])
    assert out[0]['stale'] is True
    assert out[1]['stale'] is False
    assert out[1]['lines'][0].startswith('● REC')


def test_panel_replay_plays_a_haptic_once():
    """A haptic is played at the right frames and a resend of it is not played again."""
    haptic = {'type': 'teleop_haptic', 'v': 1, 'run': 1, 'id': 1, 'hand': 'left',
              'source': 'episode', 'event': 'START', 'priority': 1,
              'pulses': [[0.8, 100, 50]], 'age_ms': 10, 'max_age_ms': 300, 'missed': 0,
              'echo': echo(1000.0)}
    out = replay([
        {'t_ms': 1000.0, 'msg': haptic},
        {'t_ms': 1010.0, 'msg': haptic},
        {'tick_ms': 1020.0},
        {'tick_ms': 1030.0},
    ])
    assert out[0]['pulses'] == [{'hand': 'left', 'intensity': 0.8, 'duration': 80}]
    assert out[1]['pulses'] == []


def test_panel_replay_needs_a_session_id():
    """Without --session the CLI exits with a usage error."""
    node = node_or_skip()
    proc = subprocess.run([node, REPLAY_JS], input='', capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2
    assert 'usage' in proc.stderr
