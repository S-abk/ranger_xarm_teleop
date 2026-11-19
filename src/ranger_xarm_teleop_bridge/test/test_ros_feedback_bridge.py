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
Operator feedback through the bridge's relay connection (INV-1, INV-4, INV-7).

A fake relay sends headset frames at 90 Hz over a real WebSocket and keeps what the bridge
sends back. Input timing is the delay from the relay's send (``t_client``, wall clock) to the
bridge reading the frame (``header.stamp``, wall clock: no simulation time here).
Pass criteria fixed before the first run: every frame sent is published, in order, with
feedback off, on, and on with a send that never completes; the p99 delay with feedback on or
stalled is at most max(1.5 x p99 off, p99 off + 2 ms).
"""

import asyncio
import copy
import json
import os
import sys
import threading
import time

import numpy as np
import pytest
from ranger_xarm_teleop_bridge import bridge_node as bn
from ranger_xarm_teleop_bridge.bridge_node import BridgeNode, WebSocketSource
from ranger_xarm_teleop_bridge.frame_parse import parse_text
from ranger_xarm_teleop_interfaces.msg import Haptic, InputStatus, OperatorStatus
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'unit'))
from parse_helpers import valid_frame  # noqa: E402, I100

RUN_S = float(os.environ.get('FEEDBACK_TIMING_RUN_S', '20'))
SESSION = 305419896


class FakeRelay:
    """The relay's place: frames out at 90 Hz, everything the bridge sends kept with its time."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.received = []          # (wall time, message dict)
        self.sent = []              # seq of every frame sent
        self.sending = False
        self.server = None
        self.port = None
        self.connections = set()
        self._seq = 0
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        self.call(self._serve(0))

    def call(self, coro, timeout=10.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _serve(self, port):
        self.server = await websockets.serve(self._handler, '127.0.0.1', port)
        self.port = self.server.sockets[0].getsockname()[1]

    async def _stop_server(self):
        self.server.close()
        for ws in list(self.connections):
            ws.transport.abort()

    def down(self):
        """Stop accepting and cut the bridge's connection abruptly (the link is gone)."""
        self.call(self._stop_server())

    def up(self):
        self.call(self._serve(self.port))

    async def _handler(self, ws, *args):
        self.connections.add(ws)
        feeder = asyncio.ensure_future(self._feed(ws))
        try:
            async for text in ws:
                self.received.append((time.time(), json.loads(text)))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            feeder.cancel()
            self.connections.discard(ws)

    async def _feed(self, ws):
        frame = valid_frame()
        period = 1.0 / 90.0
        next_t = time.monotonic()
        while True:
            next_t += period
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))
            if not self.sending:
                continue
            self._seq += 1
            f = copy.deepcopy(frame)
            f['seq'] = self._seq
            f['t_client'] = time.time() * 1000.0
            f['t_xr'] = 1000.0 + self._seq * 11.1
            f['session_id'] = SESSION
            await ws.send(json.dumps(f))
            self.sent.append(self._seq)

    def of_type(self, kind):
        return [(t, m) for t, m in self.received if m.get('type') == kind]

    def close(self):
        try:
            self.call(self._stop_server())
        except Exception:  # noqa: B902
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(5)


class StallingWs:
    """A connection whose send never completes; everything else as the real one."""

    def __init__(self, ws):
        self._ws = ws
        self.transport = getattr(ws, 'transport', None)

    async def send(self, text):
        await asyncio.Event().wait()


class StalledSource(WebSocketSource):
    async def _send_loop(self, ws, wake):
        return await super()._send_loop(StallingWs(ws), wake)


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


class Rig:
    """The bridge connected to a fake relay, a probe for its input, a feedback publisher."""

    def __init__(self, feedback, source_cls=WebSocketSource):
        self.relay = FakeRelay()
        self.node = BridgeNode(node_name='bridge_feedback_under_test', parameter_overrides=[
            Parameter('relay_url', value=f'ws://127.0.0.1:{self.relay.port}/ws'),
            Parameter('feedback_enabled', value=feedback),
            Parameter('reconnect_interval_s', value=0.2),
            Parameter('stats_period_s', value=0.0)])
        self.probe = rclpy.create_node('bridge_feedback_probe')
        self.status = []
        self.probe.create_subscription(
            InputStatus, '/teleop/input/status',
            lambda m: self.status.append((m.seq,
                                          m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec,
                                          m.client_time_ms / 1000.0)),
            QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE))
        self.haptic_pub = self.probe.create_publisher(Haptic, '/teleop/haptic', bn.HAPTIC_QOS)
        self.status_pub = self.probe.create_publisher(OperatorStatus, '/teleop/operator/status',
                                                      bn.OPERATOR_STATUS_QOS)
        self.executors = []
        for n in (self.node, self.probe):
            ex = SingleThreadedExecutor()
            ex.add_node(n)
            threading.Thread(target=ex.spin, daemon=True).start()
            self.executors.append(ex)
        self.source = source_cls(self.node)
        self.source.start()
        self.haptics_published = []
        deadline = time.monotonic() + 10
        while not self.node.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        assert self.node.connected

    def publish_haptic(self, ident, wall_time=None):
        m = Haptic()
        m.wall_time = time.time() if wall_time is None else wall_time
        m.hand, m.source, m.event, m.priority = 'left', 'episode', 'START', 2
        m.intensity, m.on_ms, m.off_ms = [0.8], [200], [0]
        m.max_age, m.source_count = 0.75, ident
        self.haptic_pub.publish(m)
        self.haptics_published.append(ident)

    def publish_status(self):
        m = OperatorStatus()
        m.wall_time = time.time()
        m.teleop_valid, m.armed, m.hold_text = True, True, 'Ready: hold right grip to move'
        m.teleop_age = 0.01
        self.status_pub.publish(m)

    def run(self, seconds, feedback_traffic=True):
        """Stream frames for ``seconds`` with status at 5 Hz and a vibration every second."""
        self.relay.sending = True
        start = time.monotonic()
        ident = 0
        next_status = start
        while time.monotonic() - start < seconds:
            now = time.monotonic()
            if feedback_traffic and now >= next_status:
                self.publish_status()
                if int((now - start) / 0.2) % 5 == 0:
                    ident += 1
                    self.publish_haptic(ident)
                next_status += 0.2
            time.sleep(0.01)
        self.relay.sending = False
        time.sleep(0.5)

    def close(self):
        self.source.stop()
        for ex in self.executors:
            ex.shutdown()
        self.node.destroy_node()
        self.probe.destroy_node()
        self.relay.close()


def delays(rig):
    sent = list(rig.relay.sent)
    published = [seq for seq, _, _ in rig.status]
    assert published == sent, (len(sent), len(published))      # all, in order
    d = np.array([stamp - t_client for _, stamp, t_client in rig.status])
    return d


def stats(d):
    return {'n': len(d), 'median_ms': 1e3 * float(np.median(d)),
            'p99_ms': 1e3 * float(np.percentile(d, 99)), 'max_ms': 1e3 * float(np.max(d))}


def test_input_timing_is_unchanged_by_feedback_and_a_stalled_send(ros):
    results = {}
    for mode in ('off', 'on', 'stalled'):
        rig = Rig(feedback=mode != 'off',
                  source_cls=StalledSource if mode == 'stalled' else WebSocketSource)
        try:
            rig.run(RUN_S)
            results[mode] = stats(delays(rig))
            if mode == 'on':
                haptics = rig.relay.of_type('teleop_haptic')
                assert [m['id'] for _, m in haptics] == list(
                    range(1, len(rig.haptics_published) + 1))
                n_status = len(rig.relay.of_type('teleop_status'))
                assert 0.9 * 5 * RUN_S <= n_status <= 1.1 * 5 * RUN_S + 1, n_status
            if mode == 'stalled':
                assert len(rig.node.outbox) <= 9           # bounded while nothing drains
            if mode == 'off':
                assert rig.relay.received == []
        finally:
            rig.close()
    print('input timing (relay send -> bridge read):', json.dumps(results, indent=1))
    out = os.environ.get('FEEDBACK_TIMING_OUT')
    if out:
        with open(out, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=1)
    off = results['off']['p99_ms']
    bound = max(1.5 * off, off + 2.0)
    assert results['on']['p99_ms'] <= bound, results
    assert results['stalled']['p99_ms'] <= bound, results


def test_link_not_draining_sends_nothing_and_stays_bounded(ros, monkeypatch):
    rig = Rig(feedback=True)
    try:
        monkeypatch.setattr(bn, '_write_buffer_size', lambda ws: 10 ** 9)
        rig.run(2.0)
        assert rig.relay.received == [] and rig.node.feedback_stalls > 0
        assert len(rig.node.outbox) <= 9
        monkeypatch.setattr(bn, '_write_buffer_size', lambda ws: 0)
        rig.run(1.0)
        assert rig.relay.of_type('teleop_status')
        # Nothing published before the drain resumed arrives older than its maximum age.
        for _, m in rig.relay.received:
            assert m['age_ms'] <= m.get('max_age_ms', 400.0) + 1e-6, m
    finally:
        rig.close()


def test_after_a_reconnect_nothing_old_is_sent_and_nothing_is_doubled(ros):
    rig = Rig(feedback=True)
    try:
        rig.run(1.5, feedback_traffic=False)
        rig.publish_haptic(101)
        time.sleep(0.5)
        rig.relay.down()
        time.sleep(0.2)
        rig.publish_haptic(102)                   # while the link is down
        for _ in range(5):
            rig.publish_status()
            time.sleep(0.2)
        time.sleep(0.8)                           # 102 is now older than 0.75 s
        rig.relay.up()
        deadline = time.monotonic() + 10
        while not rig.node.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        rig.publish_haptic(103)
        rig.publish_status()
        time.sleep(0.5)
        ids = [m['id'] for _, m in rig.relay.of_type('teleop_haptic')]
        assert ids == [1, 3], ids                 # the bridge's ids of 101 and 103: once each
        for _, m in rig.relay.received:
            limit = m['max_age_ms'] if m['type'] == 'teleop_haptic' else 400.0
            assert m['age_ms'] <= limit, m
        statuses = [m['seq'] for _, m in rig.relay.of_type('teleop_status')]
        assert statuses == sorted(set(statuses))  # never resent
    finally:
        rig.close()


@pytest.mark.parametrize('limit', [512, 65536])
def test_a_send_buffer_limit_at_or_above_the_connection_write_limit_is_refused(ros, limit):
    with pytest.raises(ValueError):
        BridgeNode(node_name='bridge_bad_limit', parameter_overrides=[
            Parameter('feedback_enabled', value=True),
            Parameter('send_buffer_limit_bytes', value=limit)])


@pytest.mark.parametrize('limit', [1024, 32768])
def test_the_send_buffer_limit_edges_are_accepted(ros, limit):
    node = BridgeNode(node_name='bridge_edge_limit', parameter_overrides=[
        Parameter('feedback_enabled', value=True),
        Parameter('send_buffer_limit_bytes', value=limit)])
    try:
        assert node.send_buffer_limit == limit
    finally:
        node.destroy_node()


def subscriptions(node):
    return {topic for topic, _ in node.get_subscriber_names_and_types_by_node(
        node.get_name(), node.get_namespace())}


def test_the_bridge_subscribes_only_to_the_two_feedback_topics(ros):
    off = BridgeNode(node_name='bridge_subs_off')
    on = BridgeNode(node_name='bridge_subs_on',
                    parameter_overrides=[Parameter('feedback_enabled', value=True)])
    try:
        deadline = time.monotonic() + 5
        while (subscriptions(on) - subscriptions(off) != {'/teleop/haptic',
                                                          '/teleop/operator/status'}
               and time.monotonic() < deadline):
            time.sleep(0.1)
        assert subscriptions(off) <= {'/parameter_events'}, subscriptions(off)
        assert subscriptions(on) - subscriptions(off) == {'/teleop/haptic',
                                                          '/teleop/operator/status'}
        # Topic types and QoS as published by the feedback node.
        qos = {sub.topic_name: sub.qos_profile for sub in on.subscriptions}
        assert qos['/teleop/haptic'] == bn.HAPTIC_QOS
        assert qos['/teleop/operator/status'] == bn.OPERATOR_STATUS_QOS
        assert bn.OPERATOR_STATUS_QOS.depth == 1      # the contract: reliable, depth 1, volatile
    finally:
        off.destroy_node()
        on.destroy_node()


def test_feedback_off_publishes_exactly_what_feedback_on_publishes(ros):
    """INV-7: the frame path does not depend on the feedback setting."""
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'clip_controller_buttons.jsonl')
    with open(path, encoding='utf-8') as handle:
        texts = [json.dumps(json.loads(line)['msg']) for line in handle if line.strip()]
    off = BridgeNode(node_name='bridge_identity_off')
    on = BridgeNode(node_name='bridge_identity_on',
                    parameter_overrides=[Parameter('feedback_enabled', value=True)])
    try:
        stamp = off.get_clock().now().to_msg()
        n = 0
        for text in texts:
            frame = parse_text(text)
            if getattr(frame, 'seq', None) is None:
                continue
            a, b = off.build_messages(frame, stamp), on.build_messages(frame, stamp)
            assert a.poses.keys() == b.poses.keys()
            for x, y in [(a.poses[k], b.poses[k]) for k in a.poses] + [
                    (a.head, b.head), (a.joy, b.joy), (a.status, b.status)]:
                assert (x is None) == (y is None)
                if x is not None:                  # bytes: NaN axes compare too
                    assert serialize_message(x) == serialize_message(y)
            n += 1
        assert n > 50
        # Published as before; and with feedback off nothing of the feedback path exists.
        assert off.feedback_enabled is False and not hasattr(off, 'outbox')
        off.handle_text(texts[0])
        assert off.last_frame is None
    finally:
        off.destroy_node()
        on.destroy_node()
