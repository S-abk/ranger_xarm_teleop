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
Stand in for the headset: send a JSONL recording's messages to the relay with their original timing.

    tools/fake_quest.py src/ranger_xarm_teleop_bridge/test/fixtures/dropped_controller.jsonl [--rate 1] [--loop]

The relay rebroadcasts every message to its other clients, so the bridge node sees exactly what
it would see from the Quest. Sends every record (also malformed ones) unless --frames-only.
"""

import argparse
import asyncio
import sys
import time

import websockets

import _pkgpath  # noqa: F401  (sys.path for the package)

from ranger_xarm_teleop_bridge.jsonl_io import read_records  # noqa: E402


async def play(url: str, path: str, rate: float, loop: bool, frames_only: bool) -> int:
    """Send the recording once (or forever with ``loop``); return the number of messages sent."""
    records = read_records(path)
    if frames_only:
        records = [r for r in records if isinstance(r.msg, dict) and r.msg.get('type') == 'xr_frame']
    if not records:
        print('nothing to send', file=sys.stderr)
        return 0
    sent = 0
    async with websockets.connect(url) as ws:
        print(f'connected to {url}; sending {len(records)} records from {path}', file=sys.stderr)
        while True:
            start = time.monotonic()
            t0 = records[0].t_recv
            for rec in records:
                due = start + (rec.t_recv - t0) / rate if rate > 0 else start
                delay = due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(min(delay, 10.0))
                await ws.send(rec.text())
                sent += 1
            if not loop:
                break
    return sent


def main() -> int:
    """Run the fake headset CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='JSONL recording or fixture')
    ap.add_argument('--url', default='ws://127.0.0.1:8443/ws')
    ap.add_argument('--rate', type=float, default=1.0, help='playback speed factor (0 = as fast as possible)')
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--frames-only', action='store_true', help='send only xr_frame messages')
    args = ap.parse_args()
    try:
        sent = asyncio.run(play(args.url, args.path, args.rate, args.loop, args.frames_only))
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        print(f'cannot connect to {args.url}: {exc}', file=sys.stderr)
        return 1
    print(f'sent {sent} messages', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
