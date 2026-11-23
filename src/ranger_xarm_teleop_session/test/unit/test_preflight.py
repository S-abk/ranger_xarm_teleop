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
"""The pre-flight with injected faults: each item ok, missing or unknown, never unknown as ok."""

import pytest
from ranger_xarm_teleop_session import preflight as pf

GOOD_PROBE = {
    'streams': {'state': {'problem': None}, 'joint_states': {'problem': None}},
    'teleop': {'seen': True, 'armed': False, 'owner': 1},
    'episode': {'seen': True, 'state': 0,
                'detail': 'joy: no message on /teleop/input/joy; status: no message on '
                          '/teleop/input/status'},
    'rates': {'/teleop/operator/status': 5.0, '/teleop/reset/state': 10.0},
    'reset': {'enabled': True, 'watchdog_alive': True},
}
HEADSET = ['joy', 'status']
SIM_OK = {'seconds': 30.0, 'joint_states': 4200, 'gap_ms': {'median': 7.0, 'p99': 20.0},
          'gaps_over_limit': 0, 'tool_pose_timeout_ms': 100.0, 'real_time_factor': 0.9}


def probe(**changes):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in GOOD_PROBE.items()}
    for key, value in changes.items():
        out[key] = value
    return out


def test_unknown_is_never_shown_as_ok():
    item = pf.Item('x', 'X', pf.UNKNOWN, 'not judged', 'look')
    assert '[UNKNOWN]' in item.line() and '[     ok]' not in item.line()
    assert pf.missing([item]) == [] and pf.unknown([item]) == [item]


def test_a_good_stack_is_ok_item_by_item():
    items = [pf.streams_item(GOOD_PROBE), pf.owner_item(GOOD_PROBE),
             pf.recorder_item(GOOD_PROBE, True, HEADSET), pf.feedback_item(GOOD_PROBE, True),
             pf.reset_item(GOOD_PROBE, True)]
    assert [i.status for i in items] == [pf.OK] * 5


def test_no_probe_report_is_unknown_not_ok():
    assert pf.streams_item(None).status == pf.UNKNOWN
    assert pf.owner_item(None).status == pf.UNKNOWN
    assert pf.recorder_item(None, True, HEADSET).status == pf.UNKNOWN
    assert pf.feedback_item(None, True).status == pf.UNKNOWN
    assert pf.reset_item(None, True).status == pf.UNKNOWN


def test_a_slow_or_silent_stream_is_missing_with_its_own_words():
    p = probe(streams={'state': {'problem': None},
                       'joint_states': {'problem': 'joint_states: 60.0 Hz, below its nominal '
                                                   '150 Hz'}})
    item = pf.streams_item(p)
    assert item.status == pf.MISSING and '60.0 Hz, below its nominal 150 Hz' in item.detail
    assert pf.streams_item(probe(streams={})).status == pf.UNKNOWN


@pytest.mark.parametrize('teleop,why', [({'seen': True, 'armed': True, 'owner': 1}, 'armed True'),
                                        ({'seen': True, 'armed': False, 'owner': 2},
                                         'owner teleop'),
                                        ({'seen': True, 'armed': False, 'owner': 0},
                                         'owner none')])
def test_armed_or_not_moveit_at_start_is_missing(teleop, why):
    item = pf.owner_item(probe(teleop=teleop))
    assert item.status == pf.MISSING and why in item.detail


def test_the_recorder_waiting_for_more_than_the_headset_is_missing():
    p = probe(episode={'seen': True, 'state': 0,
                       'detail': 'joy: no message on /teleop/input/joy; recorder '
                                 '/teleop_episode_recorder not reachable'})
    item = pf.recorder_item(p, True, HEADSET)
    assert item.status == pf.MISSING
    assert 'not reachable' in item.detail and 'joy:' not in item.detail
    assert pf.recorder_item(probe(episode={'seen': True, 'state': 1}), True, HEADSET).status \
        == pf.OK
    assert pf.recorder_item(probe(episode={'seen': False}), True, HEADSET).status == pf.MISSING
    assert pf.recorder_item(probe(episode={'seen': True, 'state': 2}), True, HEADSET).status \
        == pf.MISSING
    assert pf.recorder_item(None, False, HEADSET).status == pf.OFF


def test_feedback_and_reset_nodes_by_their_rates():
    p = probe(rates={'/teleop/operator/status': 0.0, '/teleop/reset/state': 1.0})
    assert pf.feedback_item(p, True).status == pf.MISSING
    assert pf.reset_item(p, True).status == pf.MISSING
    assert pf.feedback_item(p, False).status == pf.OFF
    assert pf.reset_item(p, False).status == pf.OFF


def test_a_reset_node_without_its_watchdog_or_switched_off_by_itself_is_missing():
    item = pf.reset_item(probe(reset={'enabled': True, 'watchdog_alive': False}), True)
    assert item.status == pf.MISSING and 'watchdog' in item.detail
    item = pf.reset_item(probe(reset={'enabled': False, 'watchdog_alive': True}), True)
    assert item.status == pf.MISSING and 'off' in item.detail


def test_simulator_check_mapping():
    assert pf.sim_check_item(SIM_OK, 0).status == pf.OK
    slow = dict(SIM_OK, real_time_factor=0.5)
    item = pf.sim_check_item(slow, 4)
    assert item.status == pf.MISSING and 'gpu:=nvidia' in item.todo
    bursty = dict(SIM_OK, gap_ms={'p99': 51.0})
    assert pf.sim_check_item(bursty, 4).status == pf.MISSING
    at_margin = dict(SIM_OK, gap_ms={'p99': 50.0})
    assert pf.sim_check_item(at_margin, 0).status == pf.OK
    # Gaps over the timeout itself are shown; the status follows the task's two criteria.
    over = dict(SIM_OK, gaps_over_limit=2)
    item = pf.sim_check_item(over, 4)
    assert item.status == pf.OK and '2 gaps over' in item.detail
    assert pf.sim_check_item(dict(SIM_OK, real_time_factor=None), 0).status == pf.UNKNOWN
    assert pf.sim_check_item(dict(SIM_OK, joint_states=1, gap_ms={}), 4).status == pf.MISSING
    item = pf.sim_check_item(None, 3, 'refused: a teleop node is running')
    assert item.status == pf.UNKNOWN and 'a teleop node is running' in item.detail
    assert pf.sim_check_item(None, None, enabled=False).status == pf.OFF


def test_gpu_item():
    assert pf.gpu_item(True, 'nvidia', 's', True, True).status == pf.OK
    assert pf.gpu_item(True, None, 's', True, True).status == pf.MISSING   # card idle
    assert pf.gpu_item(True, 'nvidia', 's', False, True).status == pf.MISSING
    assert pf.gpu_item(True, None, 'no NVIDIA card found', False, True).status == pf.INFO
    assert pf.gpu_item(False, None, '', False, False).status == pf.OFF


def test_disk_item():
    assert pf.disk_item(50.0, 20.0, '/d', True).status == pf.OK
    item = pf.disk_item(5.0, 20.0, '/d', True)
    assert item.status == pf.MISSING and '5.0 GB' in item.detail
    assert pf.disk_item(None, 20.0, '/d', True).status == pf.UNKNOWN
    assert pf.disk_item(5.0, 20.0, '/d', False).status == pf.OFF


def test_labels_item_never_shows_a_value():
    item = pf.labels_item(True, None, True)
    assert item.status == pf.OK and item.detail == 'both set (kept in the data directory only)'
    item = pf.labels_item(False, 'labels file /d/labels.json: no operator', True)
    assert item.status == pf.MISSING
    assert pf.labels_item(False, None, False).status == pf.OFF


def test_commits_item():
    good = {'commit': 'a' * 40, 'dirty': False, 'changed': []}
    dirty = {'commit': 'b' * 40, 'dirty': True, 'changed': ['x', 'y']}
    item = pf.commits_item(good, dirty)
    assert item.status == pf.OK and 'clean' in item.detail and '2 changed files' in item.detail
    item = pf.commits_item(good, {'commit': None})
    assert item.status == pf.UNKNOWN and 'robot workspace' in item.todo


def test_adb_item():
    assert pf.adb_item(None, False).status == pf.UNKNOWN
    assert pf.adb_item({'adb': False}, True).status == pf.UNKNOWN
    item = pf.adb_item({'adb': None}, True)
    assert item.status == pf.UNKNOWN and 'not consulted' in item.detail
    assert pf.adb_item({'adb': True, 'devices': []}, True).status == pf.MISSING
    ok = {'adb': True, 'devices': ['1WMHH'], 'reverse': ['UsbFfs tcp:8443 tcp:8443']}
    assert pf.adb_item(ok, True).status == pf.OK
    ran = {'adb': True, 'devices': ['1WMHH'], 'reverse': [],
           'ran': {'command': 'adb reverse tcp:8443 tcp:8443', 'code': 1,
                   'output': 'error: more than one device'}}
    item = pf.adb_item(ran, True)
    assert item.status == pf.MISSING and 'more than one device' in item.detail
    assert 'adb reverse tcp:8443 tcp:8443' in item.todo


def test_pairs_item():
    assert pf.pairs_item('none', 0, '').status == pf.OK
    assert pf.pairs_item('some', 3, '').status == pf.MISSING
    assert pf.pairs_item('not_checked', 0, 'why').status == pf.UNKNOWN
    assert pf.pairs_item(None, 0, '').status == pf.UNKNOWN


def test_headset_item_stack_role():
    assert pf.headset_item(None, None, 'stack').status == pf.UNKNOWN
    assert pf.headset_item({'lines': 1, 'connected': False}, None, 'stack').status == pf.MISSING
    assert pf.headset_item({'lines': 1, 'connected': True, 'rate': 0.0}, None,
                           'stack').status == pf.MISSING
    assert pf.headset_item({'lines': 2, 'connected': True, 'rate': 72.0}, None,
                           'stack').status == pf.OK


def test_headset_item_relay_role():
    assert pf.headset_item(None, {'peers': {}}, 'relay').status == pf.MISSING
    item = pf.headset_item(None, {'peers': {'127.0.0.1:5': '2026-11-21 10:00:00'}}, 'relay')
    assert item.status == pf.OK and item.title == 'Headset page connected to this relay'
    assert 'still flow shows on the robot computer' in item.detail


def test_version_item():
    peers = {'peers': {'127.0.0.1:5': 't'}}
    item = pf.version_item('feedback-7', False, peers)
    assert item.status == pf.UNKNOWN and 'client feedback-7' in item.todo
    assert pf.version_item('feedback-7', True, peers).status == pf.OK
    item = pf.version_item('feedback-7', True, {'peers': {}})
    assert item.status == pf.UNKNOWN and 'client feedback-7' in item.todo
    assert pf.version_item(None, True, peers).status == pf.UNKNOWN


def test_headset_streams_of_a_profile():
    profile = {'streams': {'joy': {'kind': 'joy'}, 'status': {'kind': 'input_status'},
                           'state': {'kind': 'teleop_state'}}}
    assert pf.headset_streams(profile) == ['joy', 'status']
