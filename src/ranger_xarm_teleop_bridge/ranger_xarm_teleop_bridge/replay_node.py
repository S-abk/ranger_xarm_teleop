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
Republish a JSONL recording of relay text through the same bridge code path.

Each record is fed to ``BridgeNode.handle_text`` with the original inter-record timing
(scaled by ``rate``), so the published topics are what the live bridge would have
published. ``header.stamp`` is ROS time at republication (interface contract); the
client's ``seq`` and frame time are preserved in ``InputStatus``, whose ``source``
field reads ``replay:<file>``.

Parameters: ``file`` (required), ``rate`` (1.0; 0 = as fast as possible), ``loop`` (false),
``wait_for_subscriber_s`` (0.0; wait up to this long for a subscriber on the status topic),
``start_delay_s`` (0.0; then wait this long before the first record). Together they keep
the first frames of a recording from being published before anyone listens.
"""

import os
import threading
import time

from ranger_xarm_teleop_bridge.bridge_node import BridgeNode
from ranger_xarm_teleop_bridge.jsonl_io import iter_records
import rclpy
from rclpy.executors import ExternalShutdownException


class ReplaySource:
    """Feeds a recording to a node from a background thread."""

    def __init__(self, node: BridgeNode, path: str, rate: float, loop: bool,
                 start_delay: float = 0.0, wait_for_subscriber: float = 0.0) -> None:
        self.node = node
        self.path = path
        self.rate = rate
        self.loop = loop
        self.start_delay = max(0.0, start_delay)
        self.wait_for_subscriber = max(0.0, wait_for_subscriber)
        self.done = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='replay', daemon=True)

    def start(self) -> None:
        """Start the background thread."""
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the replay and join the thread."""
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        log = self.node.get_logger()
        try:
            waited = 0.0
            while (waited < self.wait_for_subscriber
                   and self.node.status_pub.get_subscription_count() == 0):
                if self._stop.wait(0.1):
                    return
                waited += 0.1
            if self.wait_for_subscriber > 0.0:
                found = self.node.status_pub.get_subscription_count()
                log.info(f'{found} subscriber(s) on the status topic after {waited:.1f} s')
            if self.start_delay > 0.0 and self._stop.wait(self.start_delay):
                return
            while not self._stop.is_set():
                count = self._play_once()
                log.info(f'replayed {count} records from {self.path}')
                if not self.loop:
                    break
        except Exception as exc:  # noqa: B902 - report, then end the replay
            log.error(f'replay failed: {exc}')
        finally:
            self.done.set()

    def _play_once(self) -> int:
        count = 0
        start = time.monotonic()
        first = None
        for record in iter_records(self.path):
            if self._stop.is_set():
                break
            if first is None:
                first = record.t_recv
            elif self.rate > 0:
                # Schedule against the start time, not the previous record, so sleep
                # overhead does not accumulate into a lower replay rate.
                delay = start + (record.t_recv - first) / self.rate - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    break
            self.node.handle_text(record.text(), record.t_recv)
            count += 1
        return count


def main(args=None) -> None:
    """Run the replay: a JSONL recording in, /teleop/input/* out."""
    rclpy.init(args=args)
    probe = rclpy.create_node('teleop_input_replay_args')
    probe.declare_parameter('file', '')
    probe.declare_parameter('rate', 1.0)
    probe.declare_parameter('loop', False)
    probe.declare_parameter('start_delay_s', 0.0)
    probe.declare_parameter('wait_for_subscriber_s', 0.0)
    path = str(probe.get_parameter('file').value)
    rate = float(probe.get_parameter('rate').value)
    loop = bool(probe.get_parameter('loop').value)
    start_delay = float(probe.get_parameter('start_delay_s').value)
    wait_for_subscriber = float(probe.get_parameter('wait_for_subscriber_s').value)
    probe.destroy_node()
    if not path:
        print('replay_node needs the parameter file:=<recording.jsonl> '
              '(optional: rate:=1.0 loop:=false start_delay_s:=0.0 '
              'wait_for_subscriber_s:=0.0)')
        rclpy.try_shutdown()
        return
    node = BridgeNode(node_name='teleop_input_replay',
                      source_label=f'replay:{os.path.basename(path)}')
    source = ReplaySource(node, path, rate, loop, start_delay, wait_for_subscriber)
    source.start()
    try:
        while rclpy.ok() and not source.done.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        source.stop()
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
