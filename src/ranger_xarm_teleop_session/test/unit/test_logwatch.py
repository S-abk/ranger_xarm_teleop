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
"""The log parsers on lines as the bridge, the relay and the teleop node print them."""

import os

from ranger_xarm_teleop_session import logwatch as lw

BRIDGE = ('[bridge_node-1] [INFO] [1795284131.1] [teleop_input_bridge]: relay: connected=True '
          '71.8 frames/s frames=359 malformed=0 ignored=2 last_seq=358 feedback: sent haptic=0 '
          'status=25 dropped old haptic=0 status=0 overflow=0 stalls=0 errors=0')
RELAY = [
    'INFO:     Uvicorn running on http://127.0.0.1:8443 (Press CTRL+C to quit)',
    '2026-11-21 13:01:46,472 INFO vr_teleop: ws connect 127.0.0.1:37190  (now 1 clients)',
    'INFO:     127.0.0.1:50000 - "GET /static/feedback.js?v=6 HTTP/1.1" 200 OK',
    '2026-11-21 13:02:04,911 INFO vr_teleop: ws connect 127.0.0.1:54128  (now 2 clients)',
    "2026-11-21 13:02:04,912 INFO vr_teleop: first 'xr_frame' msg from 127.0.0.1:54128 | "
    "keys=['controllers']",
    "2026-11-21 13:02:05,000 INFO vr_teleop: first 'teleop_status' msg from 127.0.0.1:37190 | "
    "keys=['type']",
]


def test_bridge_statistics_line():
    b = lw.BridgeState()
    assert not b.feed('[bridge_node-1] [INFO] [1.0] [teleop_input_bridge]: connected to relay')
    assert b.feed(BRIDGE)
    seen = (b.connected, b.rate, b.frames, b.status_sent, b.source)
    assert seen == (True, 71.8, 359, 25, 'relay')
    assert b.feed(BRIDGE.replace('connected=True 71.8', 'connected=False 0.0'))
    assert b.connected is False and b.rate == 0.0 and b.lines == 2


def test_relay_log_peers_and_page_loads():
    r = lw.RelayState()
    for line in RELAY:
        r.feed(line)
    assert r.up
    assert r.headset_peers() == {'127.0.0.1:54128': '2026-11-21 13:02:04'}   # not the bridge
    assert r.page_versions == {'6': ''}
    r.feed('2026-11-21 13:03:00,000 INFO vr_teleop: ws disconnect 127.0.0.1:54128 code=1005 '
           "totals={'xr_frame': 10}")
    assert r.headset_peers() == {}


def test_rigid_pairs_lines():
    p = lw.PairsState()
    p.feed('[teleop_node-10] [INFO] [1.0] [teleop_node]: MoveIt Servo: no enabled '
           "self-collision pair without a joint of 'xarm6' between its links")
    assert p.result == 'none'
    p.feed('[teleop_node-10] [WARN] [1.0] [teleop_node]: MoveIt Servo checks 3 self-collision '
           "pairs with no joint of 'xarm6' between the two links: arm motion cannot change")
    assert (p.result, p.count) == ('some', 3)
    p.feed("[teleop_node-10] [WARN] [1.0] [teleop_node]: MoveIt Servo's self-collision pairs "
           'were not checked: no group xarm6')
    assert p.result == 'not_checked' and 'no group xarm6' in p.why


def test_the_served_feedback_version_is_read_from_the_page_source():
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'third_party',
                        'vr-teleop-kit', 'src', 'vr_teleop_kit', 'relay', 'web', 'feedback.js')
    with open(path, encoding='utf-8') as handle:
        version = lw.feedback_version(handle.read())
    assert version and version.startswith('feedback-')
    assert lw.feedback_version('nothing here') is None
