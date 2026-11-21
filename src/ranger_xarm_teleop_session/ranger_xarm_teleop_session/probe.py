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
The pre-flight's look at the running stack: a short-lived node that exits before "stack ready".

It subscribes to what the stack already publishes, never publishes, calls no service and
changes nothing. It watches until every stack item of the pre-flight is ok (at least
``--min-seconds``) or ``--timeout`` passes, writes what it saw as JSON and exits. The launcher
waits for this process to end before it says "stack ready": from then on nothing joins or
leaves the ROS graph (INV-6). The judgement is ``preflight``'s, the rate rule the recorder's
(``StreamWatch``).

It never subscribes to the reset node's two liveness topics; the reset node's own state
carries ``watchdog_alive``.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

from ranger_xarm_teleop_interfaces.msg import (EpisodeState, OperatorStatus, ResetState,
                                               TeleopState)
from ranger_xarm_teleop_record.profile import load_profile
from ranger_xarm_teleop_record.streamwatch import StreamWatch
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data, QoSProfile,
                       ReliabilityPolicy)
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, JointState

from . import preflight as pf
from .plan import PROBE_NODE

RELIABLE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST, depth=50)
STATE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                       depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
STATUS_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                        depth=1, durability=DurabilityPolicy.VOLATILE)
RESET_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                       depth=10, durability=DurabilityPolicy.VOLATILE)
WATCHED_KINDS = {'teleop_state': (TeleopState, RELIABLE_QOS),
                 'joint_states': (JointState, qos_profile_sensor_data),
                 'camera_info': (CameraInfo, qos_profile_sensor_data)}
RATE_WINDOW_S = 2.0


def watched_streams(profile: dict) -> Dict[str, dict]:
    """Return the required, ungated, robot-side streams the probe watches itself."""
    return {n: s for n, s in profile['streams'].items()
            if s.get('required') and not s.get('gate') and s['kind'] in WATCHED_KINDS}


def image_streams(profile: dict) -> Dict[str, str]:
    """Return required image streams judged by their camera_info (image name -> info name)."""
    return {n: s['info'] for n, s in profile['streams'].items()
            if s.get('required') and s['kind'] == 'image' and s.get('info')}


class Rate:
    """Arrival times of one topic, for a rate over the last few seconds."""

    def __init__(self):
        """Start with no message."""
        self.times: List[float] = []

    def add(self, now: float) -> None:
        """Note one message."""
        self.times.append(now)
        while self.times and self.times[0] < now - RATE_WINDOW_S:
            self.times.pop(0)

    def hz(self, now: float) -> float:
        """Return the rate over the window (0 when silent for over a second)."""
        if len(self.times) < 2 or now - self.times[-1] > 1.0:
            return 0.0
        return (len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-6)


class Probe(Node):
    """Watch the stack; publish nothing."""

    def __init__(self, profile: dict, record: bool, feedback: bool, reset: bool):
        """Subscribe to what the pre-flight needs, for the parts that are on."""
        super().__init__(PROBE_NODE)
        self.profile = profile
        self.record, self.feedback, self.reset = record, feedback, reset
        self.watch: Dict[str, StreamWatch] = {}
        for name, spec in watched_streams(profile).items():
            cls, qos = WATCHED_KINDS[spec['kind']]
            self.watch[name] = StreamWatch(name, spec)
            self.create_subscription(cls, spec['topic'],
                                     lambda m, n=name: self._note(n, m), qos)
        self.teleop = {'seen': False}
        self.episode = {'seen': False}
        self.reset_state = {}
        self.rates = {'/teleop/operator/status': Rate(), '/teleop/reset/state': Rate()}
        if not any(s['kind'] == 'teleop_state' for s in watched_streams(profile).values()):
            self.create_subscription(TeleopState, '/teleop/state', self._on_teleop, RELIABLE_QOS)
        if record:
            self.create_subscription(EpisodeState, '/teleop/episode/state', self._on_episode,
                                     STATE_QOS)
        if feedback:
            self.create_subscription(OperatorStatus, '/teleop/operator/status',
                                     lambda _m: self._rate('/teleop/operator/status'), STATUS_QOS)
        if reset:
            self.create_subscription(ResetState, '/teleop/reset/state', self._on_reset, RESET_QOS)

    def _note(self, name: str, msg) -> None:
        stamp = msg.header.stamp
        self.watch[name].add(stamp.sec + stamp.nanosec * 1e-9, time.monotonic())
        if isinstance(msg, TeleopState):
            self._on_teleop(msg)

    def _on_teleop(self, msg: TeleopState) -> None:
        self.teleop = {'seen': True, 'armed': bool(msg.armed), 'owner': int(msg.owner),
                       'motion_enabled': bool(msg.motion_enabled)}

    def _on_episode(self, msg: EpisodeState) -> None:
        self.episode = {'seen': True, 'state': int(msg.state), 'detail': msg.detail}

    def _on_reset(self, msg: ResetState) -> None:
        self._rate('/teleop/reset/state')
        self.reset_state = {'enabled': bool(msg.enabled),
                            'watchdog_alive': bool(msg.watchdog_alive),
                            'state': int(msg.state), 'reason_text': msg.reason_text}

    def _rate(self, topic: str) -> None:
        self.rates[topic].add(time.monotonic())

    def report(self, started: float) -> dict:
        """Return what the probe saw, in the shape ``preflight`` reads."""
        now = time.monotonic()
        streams = {n: {'problem': w.problem(now)} for n, w in self.watch.items()}
        for image, info in image_streams(self.profile).items():
            if info in streams:
                p = streams[info]['problem']
                streams[image] = {'problem': f'{image} (by its camera_info) {p}' if p else None}
        return {'node': PROBE_NODE, 'seconds': round(now - started, 1), 'streams': streams,
                'teleop': self.teleop, 'episode': self.episode, 'reset': self.reset_state,
                'rates': {t: round(r.hz(now), 2) for t, r in self.rates.items()}}

    def items(self, report: dict) -> List[pf.Item]:
        """Return the stack items of the pre-flight for ``report``."""
        return [pf.streams_item(report), pf.owner_item(report),
                pf.recorder_item(report, self.record, pf.headset_streams(self.profile)),
                pf.feedback_item(report, self.feedback), pf.reset_item(report, self.reset)]


def main(argv=None) -> int:
    """Watch the stack until its items are ok or the timeout; write the JSON report."""
    args_in = remove_ros_args(sys.argv if argv is None else argv)[1:]
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--profile', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--timeout', type=float, default=120.0)
    ap.add_argument('--min-seconds', type=float, default=3.0)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--feedback', action='store_true')
    ap.add_argument('--reset', action='store_true')
    args = ap.parse_args(args_in)
    profile = load_profile(args.profile)
    rclpy.init(args=[])
    node = Probe(profile, args.record, args.feedback, args.reset)
    started = time.monotonic()
    report = {}
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.05)
            elapsed = time.monotonic() - started
            if elapsed < args.min_seconds and elapsed < args.timeout:
                continue
            report = node.report(started)
            done = all(i.status in (pf.OK, pf.OFF) for i in node.items(report))
            if done or elapsed >= args.timeout:
                report['complete'] = done
                break
    except KeyboardInterrupt:
        report = node.report(started)
        report['complete'] = False
        report['interrupted'] = True
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    tmp = args.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=1)
    os.replace(tmp, args.out)
    print(f'probe: {"all stack items ok" if report.get("complete") else "gave up"} after '
          f'{report.get("seconds")} s; left the graph')
    return 0


if __name__ == '__main__':
    sys.exit(main())
