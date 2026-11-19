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
The two input sources on their own, against a stand-in for the teleop node (no stack).

``replay_node`` is what a person runs to replay a recording; the stack tests feed their
recordings through the long-lived input player instead (``stack_helpers.start_input``), so
this module is where the real ``replay_node`` is run. The stand-in is a node with the teleop
node's name and its four input subscriptions, on a ROS domain of this module's own.
"""

import json
import os
import threading
import time

from geometry_msgs.msg import PoseStamped
import pytest
from ranger_xarm_teleop_bridge.bridge_node import (HEAD_POSE_TOPIC, JOY_TOPIC, POSE_TOPICS,
                                                   STATUS_TOPIC)
from ranger_xarm_teleop_bridge.joy_layout import BUTTON_NAMES
from ranger_xarm_teleop_interfaces.msg import InputStatus
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from stack_helpers import (child_environment, InputPlayer, lead_in_file, LEAD_IN_SESSION, Process,
                           SERVO_FIXTURES, TELEOP_NODE)

ARM_ONLY = os.path.join(SERVO_FIXTURES, 'arm_only.jsonl')
RECORDS = 270                   # arm_only: seq 0-269 at 90 Hz
SESSION = 195939070             # the XR session id of the servo fixtures
CALIBRATE = BUTTON_NAMES.index('right_primary')         # A, in Joy.buttons
ARM = BUTTON_NAMES.index('right_secondary')             # B
# The stand-in keeps far more than the teleop node does (10 and 5 messages): a stalled test
# process must not be what loses a frame here.
RELIABLE = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE,
                      history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
BEST_EFFORT = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)


class Consumer:
    """A node that takes the input topics the teleop node takes and keeps what arrives."""

    def __init__(self, context, name, topics):
        self.node = rclpy.create_node(name, context=context)
        self.lock = threading.Lock()
        self.status, self.parts = [], {JOY_TOPIC: {}, POSE_TOPICS['right']: {},
                                       HEAD_POSE_TOPIC: {}}
        if STATUS_TOPIC in topics:
            self.node.create_subscription(InputStatus, STATUS_TOPIC, self._on_status, RELIABLE)
        for topic, kind in ((JOY_TOPIC, Joy), (POSE_TOPICS['right'], PoseStamped),
                            (HEAD_POSE_TOPIC, PoseStamped)):
            if topic in topics:
                self.node.create_subscription(
                    kind, topic, lambda msg, topic=topic: self._on_part(topic, msg), BEST_EFFORT)

    @staticmethod
    def _key(stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def _on_status(self, msg):
        with self.lock:
            self.status.append((time.monotonic(), msg))

    def _on_part(self, topic, msg):
        with self.lock:
            self.parts[topic][self._key(msg.header.stamp)] = msg

    def clear(self):
        with self.lock:
            self.status.clear()
            for part in self.parts.values():
                part.clear()

    def frames(self, session=None):
        """Return ``[(arrival, status, complete, buttons)]`` of the frames that arrived."""
        with self.lock:
            out = []
            for arrival, status in self.status:
                if session is not None and status.session_id != session:
                    continue
                key = self._key(status.header.stamp)
                joy = self.parts[JOY_TOPIC].get(key)
                complete = joy is not None and all(key in part for part in self.parts.values())
                out.append((arrival, status, complete, list(joy.buttons) if joy else []))
            return out


@pytest.fixture(scope='module')
def stand_in():
    env = child_environment(offset=14)
    context = rclpy.Context()
    rclpy.init(context=context, domain_id=int(env['ROS_DOMAIN_ID']))
    full = Consumer(context, TELEOP_NODE, (STATUS_TOPIC, JOY_TOPIC, POSE_TOPICS['right'],
                                           HEAD_POSE_TOPIC))
    # A node that does not take the poses: what "not matched" must name.
    half = Consumer(context, 'half_deaf', (STATUS_TOPIC, JOY_TOPIC))
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(full.node)
    executor.add_node(half.node)

    def spin():
        while context.ok():
            try:
                executor.spin_once(timeout_sec=0.1)
            except Exception:  # noqa: B902 - ends with the context
                if not context.ok():
                    break

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield env, full
    finally:
        executor.shutdown(timeout_sec=2.0)
        full.node.destroy_node()
        half.node.destroy_node()
        rclpy.shutdown(context=context)
        thread.join(3.0)


def wait_until(condition, timeout, what):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(f'timed out after {timeout} s waiting for {what}')


def recording_span(first_seq):
    """Return the recorded time (s) from frame ``first_seq`` of arm_only to its last frame."""
    with open(ARM_ONLY, encoding='utf-8') as handle:
        times = [json.loads(line)['t_recv'] for line in handle]
    return times[-1] - times[first_seq]


def test_replay_node_republishes_a_recording_in_order_at_its_rate_and_exits(stand_in, tmp_path):
    env, consumer = stand_in
    consumer.clear()
    process = Process(
        ['ros2', 'run', 'ranger_xarm_teleop_bridge', 'replay_node', '--ros-args',
         '-p', f'file:={ARM_ONLY}', '-p', 'wait_for_subscriber_s:=15.0',
         '-p', 'start_delay_s:=3.0'], env, str(tmp_path / 'replay_node.log'))
    try:
        assert process.wait(40.0) == 0, process.output()
    finally:
        process.stop()
    output = process.output()
    assert 'subscriber(s) on the status topic after' in output, output
    assert f'replayed {RECORDS} records from {ARM_ONLY}' in output, output
    wait_until(lambda: any(s.seq == RECORDS - 1 for _, s, _, _ in consumer.frames()), 5.0,
               'the last frame')
    frames = consumer.frames()
    seqs = [s.seq for _, s, _, _ in frames]
    # The status topic is reliable: from the first frame that arrived every frame to the last
    # one, in order. The first frames of a new process can be lost while it is being
    # discovered (replay_node waits for a subscriber and 3 s more here, on a domain with one
    # other process): at least the last two thirds must have arrived.
    assert seqs == list(range(seqs[0], RECORDS)) and seqs[0] < RECORDS // 3, seqs[:5]
    assert all(s.session_id == SESSION and s.source == 'replay:arm_only.jsonl'
               for _, s, _, _ in frames)
    # Pose, head and joy of a frame carry the status's stamp (they are best effort: most
    # arrive).
    assert sum(complete for _, _, complete, _ in frames) > 0.5 * len(frames)
    # At the recorded rate, by the stamps replay_node gave the frames (its time of
    # publication): the frames that arrived span what they span in the recording.
    stamps = [Consumer._key(s.header.stamp) * 1e-9 for _, s, _, _ in frames]
    span = stamps[-1] - stamps[0]
    assert abs(span - recording_span(seqs[0])) < 0.2, (span, recording_span(seqs[0]))


def test_the_input_player_names_a_topic_its_consumer_does_not_take(stand_in, tmp_path):
    env, _ = stand_in
    player = InputPlayer(env, tmp_path / 'input_player.log')
    try:
        player.wait_event('up', 60.0)
        # The stand-in's process first (both its nodes are one participant): once the full
        # consumer is matched, what the other one does not take is known not to be coming.
        player.send(cmd='match', consumer=TELEOP_NODE, timeout=30.0)
        assert player.wait_event('matched', 40.0)['ok']
        with player.lock:
            player.events.clear()
        player.send(cmd='match', consumer='half_deaf', timeout=3.0)
        report = player.wait_event('matched', 15.0)
        assert not report['ok'], report
        known = {topic: counts[2] for topic, counts in report['topics'].items()}
        assert known == {STATUS_TOPIC: True, JOY_TOPIC: True, POSE_TOPICS['right']: False,
                         HEAD_POSE_TOPIC: False}, report
    finally:
        player.close()
    assert not player.running()


def test_the_input_player_delivers_every_recording_from_its_first_frame(stand_in, tmp_path):
    env, consumer = stand_in
    consumer.clear()
    player = InputPlayer(env, tmp_path / 'input_player.log')
    try:
        player.wait_event('up', 60.0)
        # Matched: every publisher a frame is built from, by every subscription known.
        player.send(cmd='match', consumer=TELEOP_NODE, timeout=30.0)
        report = player.wait_event('matched', 40.0)
        assert report['ok'], report
        assert all(matched >= known >= 1 and consumer_known
                   for matched, known, consumer_known in report['topics'].values()), report
        # Acknowledged: the lead-in is repeated until a frame with its calibrate press has
        # arrived whole (in a stack: until the teleop node has calibrated from it).
        path, count = lead_in_file(tmp_path)
        lead_in = player.play(path, loop=True, source='replay:lead_in')
        try:
            wait_until(lambda: any(complete and buttons[CALIBRATE] for _, _, complete, buttons
                                   in consumer.frames(LEAD_IN_SESSION)),
                       30.0, 'a whole frame of the lead-in with its calibrate press')
            time.sleep(1.0)                     # into a second pass
        finally:
            lead_in.stop()
        assert lead_in.returncode == 0 and lead_in.passes >= 1
        lead = consumer.frames(LEAD_IN_SESSION)
        assert all(s.source == 'replay:lead_in' for _, s, _, _ in lead)
        assert not any(buttons and buttons[ARM] for _, _, _, buttons in lead)   # no arm press
        assert max(s.seq for _, s, _, _ in lead) == count - 1
        # From here on nothing is lost at the start, twice over, with no start delay.
        for attempt in range(2):
            consumer.clear()
            started = time.monotonic()
            replay = player.play(ARM_ONLY, log_path=tmp_path / f'replay.{attempt}.log')
            assert replay.wait(20.0) == 0, replay.output()
            wait_until(lambda: len(consumer.frames(SESSION)) == RECORDS, 5.0, 'every frame')
            frames = consumer.frames(SESSION)
            assert [s.seq for _, s, _, _ in frames] == list(range(RECORDS))
            assert all(s.source == 'replay:arm_only.jsonl' for _, s, _, _ in frames)
            assert frames[0][2], 'the first frame did not arrive whole'
            # Both presses arrived (A 27-44, B 63-80), and nearly every frame whole.
            assert any(complete and buttons[CALIBRATE] for _, _, complete, buttons in frames)
            assert any(complete and buttons[ARM] for _, _, complete, buttons in frames)
            assert sum(complete for _, _, complete, _ in frames) > 0.9 * RECORDS
            stamps = [Consumer._key(s.header.stamp) * 1e-9 for _, s, _, _ in frames]
            assert abs(stamps[-1] - stamps[0] - recording_span(0)) < 0.2
            assert frames[0][0] - started < 1.0         # no start delay, no discovery wait
            assert os.path.getsize(tmp_path / f'replay.{attempt}.log') > 0
    finally:
        player.close()


def test_a_recording_that_cannot_be_read_fails_its_replay_not_the_player(stand_in, tmp_path):
    env, consumer = stand_in
    player = InputPlayer(env, tmp_path / 'input_player.log')
    try:
        player.wait_event('up', 60.0)
        missing = player.play(tmp_path / 'no_such_recording.jsonl')
        assert missing.wait(10.0) == 1, missing.output()
        assert 'no_such_recording' in missing.output()
        # A recording stopped before its start delay ends has sent nothing.
        consumer.clear()
        held = player.play(ARM_ONLY, start_delay=30.0)
        assert held.wait(0.5) is None
        held.stop()
        assert held.returncode == 2 and consumer.frames() == []     # stopped: not a success
        # A recording cut off by the next one has not succeeded either; the next one has.
        cut = player.play(ARM_ONLY)
        wait_until(lambda: len(consumer.frames()) > 20, 10.0, 'the first frames')
        whole = player.play(ARM_ONLY)
        assert cut.wait(5.0) == 2, cut.output()
        cut.stop()                                  # stopping the one cut off: the other plays on
        assert whole.wait(20.0) == 0, whole.output()
        assert player.running()
    finally:
        player.close()
