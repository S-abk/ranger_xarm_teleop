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
Commissioning a machine for recording, without a simulator or robot.

Publishes synthetic images (and their camera_info) at the profile's size and rate for every
image stream the profile records, records them with the profile's recorder settings for a
while, then reports per stream: published, recorded, lost; the recorder's CPU; the disk
rate; and the free space in minutes. Image content ``noise`` does not compress (worst case,
an upper bound for real cameras); ``flat`` compresses almost entirely (a lower bound).

Run it on its own, not during a session: it starts its own publisher and recorder.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from .profile import load_profile, recorded

BYTES = {'rgb8': 3, 'bgr8': 3, 'mono8': 1, '16UC1': 2, 'mono16': 2, '32FC1': 4}


def _proc_ticks(root_pid: int) -> int:
    """Return user+system clock ticks of a process and its children."""
    total = 0
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open(f'/proc/{entry}/stat') as handle:
                fields = handle.read().rsplit(')', 1)[1].split()
        except OSError:
            continue
        pid, ppid = int(entry), int(fields[1])
        if pid == root_pid or ppid == root_pid:
            total += int(fields[11]) + int(fields[12])
    return total


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class Publisher(Node):
    """Synthetic camera streams at the profile's size and rate."""

    def __init__(self, streams, content: str, use_sim_time: bool):
        """Create one image and one camera_info publisher per image stream."""
        super().__init__('record_commission')
        self.items = []
        rng = np.random.default_rng(0)
        for name, spec in streams.items():
            img = spec['image']
            w, h, enc = int(img['width']), int(img['height']), img['encoding']
            nbytes = w * h * BYTES[enc]
            frames = ([rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()
                       for _ in range(4)] if content == 'noise' else [bytes(nbytes)])
            pub = self.create_publisher(Image, spec['topic'], qos_profile_sensor_data)
            info_spec = streams.get(spec.get('info') or '', None)
            info_topic = (info_spec or {}).get('topic') or \
                spec['topic'].rsplit('/', 1)[0] + '/camera_info'
            info = self.create_publisher(CameraInfo, info_topic, qos_profile_sensor_data)
            self.items.append({'name': name, 'w': w, 'h': h, 'enc': enc, 'frames': frames,
                               'pub': pub, 'info': info, 'n': 0, 'stamps': [],
                               'period': 1.0 / float(spec['rate_hz'])})
        self.start = time.monotonic()
        for item in self.items:
            item['next'] = self.start
        self.create_timer(0.002, self.tick)

    def tick(self):
        """Publish what is due."""
        now = time.monotonic()
        for item in self.items:
            if now < item['next']:
                continue
            item['next'] += item['period']
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = item['name']
            msg.width, msg.height, msg.encoding = item['w'], item['h'], item['enc']
            msg.step = item['w'] * BYTES[item['enc']]
            msg.data = item['frames'][item['n'] % len(item['frames'])]
            item['pub'].publish(msg)
            item['stamps'].append(msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)
            info = CameraInfo()
            info.header = msg.header
            info.width, info.height = item['w'], item['h']
            item['info'].publish(info)
            item['n'] += 1


def main(argv=None) -> int:
    """Run the commissioning measurement and print a JSON report."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--profile', default='gazebo')
    ap.add_argument('--seconds', type=float, default=60.0)
    ap.add_argument('--content', choices=('noise', 'flat'), default='noise')
    ap.add_argument('--dir', default=None, help='where the test bag goes (default: the '
                    "profile's data directory); it is deleted afterwards unless --keep")
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--out', default=None, help='also write the report to this JSON file')
    args = ap.parse_args(argv)
    profile = load_profile(args.profile)
    images = {n: s for n, s in recorded(profile).items() if s['kind'] == 'image'}
    infos = {n: s for n, s in profile['streams'].items() if s['kind'] == 'camera_info'}
    if not images:
        print('the profile records no image stream; nothing to commission')
        return 2
    base = os.path.abspath(os.path.expanduser(args.dir or profile['data_dir']))
    os.makedirs(base, exist_ok=True)
    work = tempfile.mkdtemp(prefix='commission-', dir=base)
    bag = os.path.join(work, 'bag')
    rec = profile['recorder']
    topics = sorted({s['topic'] for s in images.values()})
    command = ['ros2', 'bag', 'record', '-o', bag, '-s', rec['storage'],
               '--storage-preset-profile', rec['storage_preset'],
               '--max-cache-size', str(int(rec['max_cache_size_mb']) * 1024 * 1024),
               '--node-name', 'record_commission_recorder'] + topics
    rclpy.init()
    probe = Node('record_commission_probe')
    end = time.monotonic() + 3.0
    while time.monotonic() < end:                       # let discovery settle
        rclpy.spin_once(probe, timeout_sec=0.1)
    busy = [s['topic'] for s in list(images.values()) + list(infos.values())
            if probe.count_publishers(s['topic']) > 0]
    probe.destroy_node()
    if busy:
        rclpy.shutdown()
        print('refused: these topics already have publishers (a camera or a session is '
              f'running here): {", ".join(busy)}. Run the commissioning on its own.')
        return 3
    node = Publisher(dict(images, **infos), args.content, False)
    log = open(os.path.join(work, 'recorder.log'), 'w')
    recorder = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    hz = os.sysconf('SC_CLK_TCK')
    try:
        settle = time.monotonic() + 5.0
        while time.monotonic() < settle:
            rclpy.spin_once(node, timeout_sec=0.01)
        for item in node.items:
            item['n0'] = item['n']
        ticks0, t0, size0 = _proc_ticks(recorder.pid), time.monotonic(), _dir_size(bag)
        end = t0 + args.seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.002)
        ticks1, t1, size1 = _proc_ticks(recorder.pid), time.monotonic(), _dir_size(bag)
        window = {i['name']: i['n'] - i['n0'] for i in node.items}
        sent = {i['name']: list(i['stamps']) for i in node.items}
    finally:
        os.killpg(recorder.pid, signal.SIGINT)
        recorder.wait(60)
        log.close()
        node.destroy_node()
        rclpy.shutdown()
    from rclpy.serialization import deserialize_message
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=rec['storage']),
                rosbag2_py.ConverterOptions('', ''))
    got = {}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        stamp = deserialize_message(data, Image).header.stamp
        got.setdefault(topic, []).append(stamp.sec * 10**9 + stamp.nanosec)
    stats = {}
    for name, stamps in sent.items():
        rec_stamps = sorted(got.get(images[name]['topic'], []))
        first = rec_stamps[0] if rec_stamps else None
        # compare from the first recorded frame on, up to the last frame published
        published = [s for s in stamps if first is not None and s >= first]
        recorded_n = len([s for s in rec_stamps if s <= stamps[-1]]) if stamps else 0
        stats[name] = {'topic': images[name]['topic'], 'published': len(published),
                       'recorded': recorded_n, 'lost': len(published) - recorded_n,
                       'duplicates': len(rec_stamps) - len(set(rec_stamps)),
                       'rate_hz': window[name] / (t1 - t0),
                       'nominal_hz': images[name]['rate_hz']}
    wall = t1 - t0
    final = _dir_size(bag)
    disk_rate = (size1 - size0) / wall if size1 > size0 else final / max(wall, 1e-9)
    free = shutil.disk_usage(base).free
    report = {
        'profile': profile['name'], 'content': args.content, 'seconds': wall,
        'storage_preset': rec['storage_preset'],
        'streams': stats,
        'recorder_cpu_percent': (ticks1 - ticks0) / hz / wall * 100.0,
        'disk_MB_per_s': disk_rate / 1e6,
        'disk_GB_per_min': disk_rate * 60 / 1e9,
        'bag_bytes': final,
        'free_GB': free / 1e9,
        'free_minutes': free / disk_rate / 60.0 if disk_rate > 0 else None,
        'note': 'counted from the first recorded frame of each stream to the last one '
                'published; lost = published - recorded in that window',
    }
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        report['kept'] = os.path.basename(work)
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')
    return 0
