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
The stack tests' input source: recordings through the bridge's code path, on one node.

A process started for every replay brings new publishers each time, and the first frames of
a recording are lost while the teleop node and the new publishers discover each other (the
calibrate and arm presses are in a recording's first second). This process lives as long as
its stack: its publishers are matched once, and every recording of the module goes through
them. Each record is fed to ``BridgeNode.handle_text`` with the recorded timing, exactly as
``replay_node`` does.

Run by ``stack_helpers.InputPlayer``; not a test module. Commands arrive as JSON lines on
stdin, events leave as JSON lines on stdout (the node's log goes to stderr):

``{"cmd": "match", "consumer": NODE, "timeout": S}``
    Wait until every topic a frame is built from has a subscription of ``NODE`` in the graph
    and this node's publisher on it is matched by every subscription known there. Answers
    ``{"event": "matched", "ok": BOOL, "topics": {TOPIC: [matched, known, consumer known]}}``.
``{"cmd": "play", "id": N, "path": FILE, "start_delay": S, "loop": BOOL, "source": TEXT}``
    Play a JSONL recording once, or again and again until stopped (``loop``). Answers
    ``started`` when the first record is due, ``pass`` after every complete pass, then ``done``
    with the records sent, the passes and whether it was stopped (``error`` first when the
    file cannot be read). A recording still playing is stopped first.
``{"cmd": "stop", "id": N}``
    Stop recording ``N`` if it is the one playing (another one is left alone).
``{"cmd": "quit"}``, or stdin closed
    Leave.
"""

import json
import os
import sys
import threading
import time

from ranger_xarm_teleop_bridge.bridge_node import (BridgeNode, HEAD_POSE_TOPIC, JOY_TOPIC,
                                                   POSE_TOPICS, STATUS_TOPIC)
from ranger_xarm_teleop_bridge.jsonl_io import read_records
import rclpy
from rclpy.executors import ExternalShutdownException

NODE_NAME = 'teleop_test_input'


def say(**event) -> None:
    """Write one event line, with the time (Unix seconds) it was written."""
    sys.stdout.write(json.dumps({**event, 't': round(time.time(), 6)}) + '\n')
    sys.stdout.flush()


class Player:
    """Plays recordings on a bridge node, one at a time, from a background thread."""

    def __init__(self, node: BridgeNode) -> None:
        self.node = node
        # The topics the teleop node builds a frame from (the right controller's pose only).
        self.publishers = {STATUS_TOPIC: node.status_pub, JOY_TOPIC: node.joy_pub,
                           POSE_TOPICS['right']: node.pose_pubs['right'],
                           HEAD_POSE_TOPIC: node.head_pub}
        self.quit = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._playing = None            # id of the recording the thread plays

    def matching(self, consumer: str) -> dict:
        """Return ``{topic: [matched by the publisher, subscriptions known, consumer known]}``."""
        out = {}
        for topic, publisher in self.publishers.items():
            known = self.node.get_subscriptions_info_by_topic(topic)
            out[topic] = [publisher.get_subscription_count(), len(known),
                          any(info.node_name == consumer for info in known)]
        return out

    def match(self, consumer: str, timeout: float) -> None:
        """Report when every frame topic is matched by all its known subscriptions."""
        end = time.monotonic() + timeout
        while True:
            topics = self.matching(consumer)
            ok = all(known_consumer and matched >= known
                     for matched, known, known_consumer in topics.values())
            if ok or time.monotonic() >= end or self.quit.is_set():
                say(event='matched', ok=ok, topics=topics)
                return
            time.sleep(0.05)

    def play(self, ident: int, path: str, start_delay: float, loop: bool, source: str) -> None:
        """Start a recording; the one playing, if any, is stopped first."""
        self.stop()
        try:
            records = read_records(path)
            if not records:
                raise ValueError(f'{path}: no records')
        except (OSError, ValueError) as err:
            say(event='error', id=ident, text=str(err))
            say(event='done', id=ident, records=0, passes=0, stopped=False, failed=True)
            return
        self._stop = threading.Event()
        self._playing = ident
        self._thread = threading.Thread(
            target=self._run, args=(ident, path, records, start_delay, loop, source, self._stop),
            name='play', daemon=True)
        self._thread.start()

    def stop(self, ident=None) -> None:
        """Stop the recording that is playing (only if it is ``ident``, when given)."""
        if ident is not None and ident != self._playing:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)
            self._thread = None

    def _run(self, ident, path, records, start_delay, loop, source, stop) -> None:
        log = self.node.get_logger()
        count = passes = 0
        failed = False
        try:
            if start_delay > 0.0 and stop.wait(start_delay):
                return
            # As replay_node labels its frames; the label is read for every frame.
            self.node.source = source or f'replay:{os.path.basename(path)}'
            say(event='started', id=ident)
            log.info(f'play {ident}: {path}')
            period = (records[-1].t_recv - records[0].t_recv) / max(1, len(records) - 1)
            while not stop.is_set():
                start = time.monotonic()
                first = records[0].t_recv
                for record in records:
                    # Scheduled against the start, as replay_node does, so that sleep
                    # overhead does not add up to a lower rate.
                    delay = start + (record.t_recv - first) - time.monotonic()
                    if delay > 0.0 and stop.wait(delay):
                        break
                    if stop.is_set():
                        break
                    self.node.handle_text(record.text(), record.t_recv)
                    count += 1
                else:
                    passes += 1
                    say(event='pass', id=ident, passes=passes)
                if not loop or stop.wait(period):
                    break
            log.info(f'play {ident}: replayed {count} records from {path} in {passes} '
                     f'pass(es){" (stopped)" if stop.is_set() else ""}')
        except Exception as err:  # noqa: B902 - reported; the player goes on
            failed = True
            say(event='error', id=ident, text=f'{type(err).__name__}: {err}')
        finally:
            say(event='done', id=ident, records=count, passes=passes, stopped=stop.is_set(),
                failed=failed, loop=loop)

    def commands(self) -> None:
        """Read and carry out commands until told to leave or stdin closes."""
        try:
            for line in sys.stdin:
                if not line.strip():
                    continue
                command = json.loads(line)
                name = command.get('cmd')
                if name == 'match':
                    self.match(str(command['consumer']), float(command.get('timeout', 30.0)))
                elif name == 'play':
                    self.play(int(command['id']), str(command['path']),
                              float(command.get('start_delay', 0.0)),
                              bool(command.get('loop', False)), str(command.get('source', '')))
                elif name == 'stop':
                    self.stop(command.get('id'))
                elif name == 'quit':
                    break
                else:
                    say(event='error', id=-1, text=f'unknown command {name!r}')
        except Exception as err:  # noqa: B902 - reported; then the process leaves
            say(event='error', id=-1, text=f'{type(err).__name__}: {err}')
        finally:
            self.stop()
            self.quit.set()


def main() -> None:
    """Run the input source until told to leave."""
    rclpy.init()
    node = BridgeNode(node_name=NODE_NAME, source_label='replay')
    player = Player(node)
    threading.Thread(target=player.commands, name='commands', daemon=True).start()
    say(event='up')
    try:
        while rclpy.ok() and not player.quit.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        player.stop()
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
