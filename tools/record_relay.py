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
Record the relay's WebSocket text to JSONL without ROS.

    tools/record_relay.py --out recordings/test.jsonl [--url ws://127.0.0.1:8443/ws] [--duration 30]

Writes one line per message: {"t_recv": <unix s>, "msg": <json>} (or "raw" for non-JSON).
Useful to check that the headset is streaming before ROS is involved. Ctrl-C stops it.
"""

import argparse
import asyncio
import sys
import time

import websockets

import _pkgpath  # noqa: F401  (sys.path for the package)

from ranger_xarm_teleop_bridge.jsonl_io import RecordWriter  # noqa: E402


async def record(url: str, out: str, duration: float, overwrite: bool) -> int:
    """Connect once and write every message until the duration elapses or the socket closes."""
    count = 0
    deadline = time.monotonic() + duration if duration > 0 else None
    with RecordWriter(out, overwrite=overwrite) as writer:
        async with websockets.connect(url) as ws:
            print(f'connected to {url}, writing {out}', file=sys.stderr)
            while deadline is None or time.monotonic() < deadline:
                timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if isinstance(message, bytes):
                    message = message.decode('utf-8', errors='replace')
                writer.write_text(message, time.time())
                count += 1
                if count % 500 == 0:
                    print(f'{count} messages', file=sys.stderr)
    return count


def main() -> int:
    """Run the recorder CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='ws://127.0.0.1:8443/ws')
    ap.add_argument('--out', required=True, help='output JSONL path')
    ap.add_argument('--duration', type=float, default=0.0, help='seconds to record (0 = until Ctrl-C)')
    ap.add_argument('--overwrite', action='store_true', help='replace an existing output file')
    args = ap.parse_args()
    try:
        count = asyncio.run(record(args.url, args.out, args.duration, args.overwrite))
    except KeyboardInterrupt:
        print('stopped', file=sys.stderr)
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(f'cannot connect to {args.url}: {exc}', file=sys.stderr)
        return 1
    print(f'wrote {count} messages to {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
