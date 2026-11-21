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
"""
What the running stack already prints, read line by line. No ROS imports.

After the probe has left, nothing joins the ROS graph (INV-6), so the headset items are judged
from output the launcher owns anyway:

- the bridge's statistics line, every ``stats_period_s`` (5 s):
  ``relay: connected=True 71.8 frames/s frames=359 ... feedback: sent haptic=0 status=25 ...``;
- the relay's log: ``ws connect PEER``, ``first 'xr_frame' msg from PEER``, ``ws disconnect PEER``
  and the HTTP access lines of the page's files;
- the teleop node's start-up check of MoveIt Servo's self-collision pairs (rigid pairs).
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional

BRIDGE_STATS = re.compile(
    r'(?P<source>[\w:./-]+): connected=(?P<connected>True|False) (?P<rate>[0-9.]+) frames/s '
    r'frames=(?P<frames>\d+)')
BRIDGE_STATUS_SENT = re.compile(r' status=(?P<status>\d+)')
RELAY_UP = re.compile(r'Uvicorn running on (?P<url>\S+)')
RELAY_CONNECT = re.compile(r'ws connect (?P<peer>\S+)')
RELAY_DISCONNECT = re.compile(r'ws disconnect (?P<peer>\S+)')
RELAY_FIRST_FRAME = re.compile(r"first 'xr_frame' msg from (?P<peer>\S+)")
RELAY_PAGE_FILE = re.compile(r'"GET /static/feedback\.js\?v=(?P<v>\w+) HTTP')
# A log line's own clock time, as the relay's logging prints it.
RELAY_TIME = re.compile(r'^(?P<t>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
PAIRS_NONE = re.compile(r'MoveIt Servo: no enabled self-collision pair without a joint')
PAIRS_SOME = re.compile(r'MoveIt Servo checks (?P<n>\d+) self-collision pairs with no joint')
PAIRS_NOT_CHECKED = re.compile(r"(?P<why>MoveIt Servo(?:'s self-collision pairs were not "
                               r'checked: .*| reports no robot_description.*| description and '
                               r'SRDF not read: .*))')
FEEDBACK_VERSION = re.compile(r"FEEDBACK_VERSION\s*=\s*'(?P<v>[^']+)'")
# The launch's own lines about the processes it runs.
PROCESS_EVENT = re.compile(r'^\[(?:INFO|ERROR)\] \[(?P<proc>[\w.-]+)-(?P<n>\d+)\]: process '
                           r'(?P<event>started|has died|has finished cleanly)')
# Executables a launch starts that end by themselves while the stack keeps running: a
# controller spawner leaves once its controller is loaded; gz's ``create`` once the robot is
# spawned.
EXITING = ('spawner', 'create')


@dataclass
class BridgeState:
    """The newest statistics line of the bridge."""

    lines: int = 0
    connected: Optional[bool] = None
    rate: Optional[float] = None
    frames: Optional[int] = None
    status_sent: Optional[int] = None
    status_sent_before: Optional[int] = None
    source: str = ''

    def feed(self, line: str) -> bool:
        """Read one output line; return True when it was a statistics line."""
        m = BRIDGE_STATS.search(line)
        if not m:
            return False
        self.lines += 1
        self.source = m['source']
        self.connected = m['connected'] == 'True'
        self.rate = float(m['rate'])
        self.frames = int(m['frames'])
        s = BRIDGE_STATUS_SENT.search(line[m.end():])
        self.status_sent_before = self.status_sent
        self.status_sent = int(s['status']) if s else None
        return True


@dataclass
class RelayState:
    """Peers of the relay this launcher started, from its log."""

    up: bool = False
    connected: Dict[str, str] = field(default_factory=dict)     # peer -> time of connect
    sending: Dict[str, str] = field(default_factory=dict)       # peer -> time of first frame
    page_versions: Dict[str, str] = field(default_factory=dict)  # v= -> time of last load

    def feed(self, line: str) -> None:
        """Read one log line."""
        t = RELAY_TIME.search(line)
        when = t['t'] if t else ''
        if RELAY_UP.search(line):
            self.up = True
        m = RELAY_CONNECT.search(line)
        if m:
            self.connected[m['peer']] = when
            return
        m = RELAY_DISCONNECT.search(line)
        if m:
            self.connected.pop(m['peer'], None)
            self.sending.pop(m['peer'], None)
            return
        m = RELAY_FIRST_FRAME.search(line)
        if m:
            self.sending[m['peer']] = when
            return
        m = RELAY_PAGE_FILE.search(line)
        if m:
            self.page_versions[m['v']] = when

    def headset_peers(self) -> Dict[str, str]:
        """Return the connected peers that have sent headset frames (peer -> since)."""
        return {p: t for p, t in self.sending.items() if p in self.connected}


@dataclass
class PairsState:
    """The teleop node's start-up check of MoveIt Servo's rigid self-collision pairs."""

    result: Optional[str] = None        # 'none', 'some', 'not_checked'
    count: int = 0
    why: str = ''

    def feed(self, line: str) -> None:
        """Read one output line of the teleop stack."""
        if PAIRS_NONE.search(line):
            self.result, self.count = 'none', 0
        elif PAIRS_SOME.search(line):
            self.result, self.count = 'some', int(PAIRS_SOME.search(line)['n'])
        elif PAIRS_NOT_CHECKED.search(line):
            self.result, self.why = 'not_checked', PAIRS_NOT_CHECKED.search(line)['why']


@dataclass
class LaunchState:
    """The processes of one launch, from the launch's own lines."""

    started: Dict[str, str] = field(default_factory=dict)    # 'name-N' -> name
    finished: Dict[str, str] = field(default_factory=dict)
    died: Dict[str, str] = field(default_factory=dict)       # 'name-N' -> the line
    events: int = 0

    def feed(self, line: str) -> None:
        """Read one output line of the launch."""
        m = PROCESS_EVENT.search(line)
        if not m:
            return
        self.events += 1
        key = f'{m["proc"]}-{m["n"]}'
        if m['event'] == 'started':
            self.started[key] = m['proc']
        elif m['event'] == 'has died':
            self.died[key] = line.strip()
        else:
            self.finished[key] = m['proc']

    def pending(self) -> List[str]:
        """Return the processes that end by themselves and have not ended yet."""
        return sorted(k for k, name in self.started.items()
                      if name in EXITING and k not in self.finished and k not in self.died)

    def ended_wrongly(self) -> List[str]:
        """Return the processes that died, or ended though they should keep running."""
        out = [f'{k} died: {line}' for k, line in sorted(self.died.items())]
        out += [f'{k} ended' for k, name in sorted(self.finished.items()) if name not in EXITING]
        return out


def feedback_version(source: str) -> Optional[str]:
    """Return the FEEDBACK_VERSION a feedback.js source declares, or None."""
    m = FEEDBACK_VERSION.search(source or '')
    return m['v'] if m else None
