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
Discovery off (as in parallel workers, see conftest.py) hides a process on the same domain.

Pairs of processes on the domain of this session: with local-only discovery on both sides each
sees the other's node and messages, and quickly; with discovery off on either side or on both
(two parallel workers) neither does within a window several times that long.
"""

import os
import select
import subprocess
import sys
import time

HOLDER = """
import sys, time, rclpy
from std_msgs.msg import String
rclpy.init()
node = rclpy.create_node(sys.argv[1])
pub = node.create_publisher(String, sys.argv[2], 10)
print('ready', flush=True)
end = time.monotonic() + float(sys.argv[3])
while time.monotonic() < end:
    pub.publish(String(data='x'))
    rclpy.spin_once(node, timeout_sec=0.1)
"""

LOOKER = """
import sys, time, rclpy
from std_msgs.msg import String
rclpy.init()
node = rclpy.create_node(sys.argv[1] + '_looker')
got = []
node.create_subscription(String, sys.argv[2], got.append, 10)
start = time.monotonic()
end = start + float(sys.argv[3])
seen = False
while time.monotonic() < end and not (seen and got):
    rclpy.spin_once(node, timeout_sec=0.1)
    seen = sys.argv[1] in [name for name, _ in node.get_node_names_and_namespaces()]
print(seen, len(got), round(time.monotonic() - start, 2), flush=True)
"""

#: (holder's range, looker's range, whether the looker must see the holder)
CASES = (('LOCALHOST', 'LOCALHOST', True), ('LOCALHOST', 'OFF', False),
         ('OFF', 'LOCALHOST', False), ('OFF', 'OFF', False))
#: How long a looker that must not see the holder watches for it, s.
WINDOW = 5.0
#: The local-only looker must find the holder and a message within this share of the window.
FOUND_WITHIN = 0.5


def _env(discovery_range):
    """Return this environment (same domain) with the given discovery range and no peers."""
    env = dict(os.environ)
    env['ROS_AUTOMATIC_DISCOVERY_RANGE'] = discovery_range
    env.pop('ROS_STATIC_PEERS', None)
    env.pop('ROS_LOCALHOST_ONLY', None)
    return env


def _ready(process, timeout):
    """Return the first line the process prints within ``timeout`` s ('' if none)."""
    if select.select([process.stdout], [], [], timeout)[0]:
        return process.stdout.readline().strip()
    return ''


def test_off_on_either_or_both_sides_hides_a_process_on_the_same_domain_local_only_does_not():
    tag = f'peer_{os.getpid()}_{time.monotonic_ns() % 1_000_000_000}'
    holders, lookers = [], []
    try:
        for i, (held, _, _) in enumerate(CASES):
            holders.append(subprocess.Popen(
                [sys.executable, '-c', HOLDER, f'{tag}_{i}', f'/{tag}_{i}', '30'],
                env=_env(held), stdout=subprocess.PIPE, text=True))
        assert [_ready(p, 20.0) for p in holders] == ['ready'] * len(CASES)
        for i, (_, looks, seen) in enumerate(CASES):
            lookers.append(subprocess.Popen(
                [sys.executable, '-c', LOOKER, f'{tag}_{i}', f'/{tag}_{i}', str(WINDOW)],
                env=_env(looks), stdout=subprocess.PIPE, text=True))
        results = [p.communicate(timeout=30)[0].split() for p in lookers]
    finally:
        for p in holders + lookers:
            p.kill()
            p.wait()
    for (held, looks, seen), result in zip(CASES, results):
        assert len(result) == 3, f'looker ({held} holder, {looks} looker) printed {result}'
        visible, messages, after = result[0] == 'True', int(result[1]), float(result[2])
        assert (visible, messages > 0) == (seen, seen), (held, looks, result)
        if seen:
            assert after < FOUND_WITHIN * WINDOW, (held, looks, result)
