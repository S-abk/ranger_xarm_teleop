#!/usr/bin/env python3
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
Cut a time window out of a JSONL recording, copying the lines verbatim.

Used to promote a short clip of a real headset recording to a test fixture:

    tools/clip_recording.py recordings/occlusion.jsonl \\
        src/ranger_xarm_teleop_bridge/test/fixtures/clip_controller_occlusion.jsonl \\
        --from 4.3 --to 6.6

``--from``/``--to`` are seconds relative to the first record's ``t_recv``. Lines
are not re-serialised, so the clip parses exactly like the original.
"""

import argparse
import json
import sys


def main() -> int:
    """Parse the arguments and write the clip."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('source')
    parser.add_argument('out')
    parser.add_argument('--from', dest='t_from', type=float, default=0.0,
                        help='window start, seconds after the first record (default 0)')
    parser.add_argument('--to', dest='t_to', type=float, default=None,
                        help='window end, seconds after the first record (default: end)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    first = None
    kept = 0
    out_lines = []
    with open(args.source, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            t_recv = float(json.loads(line)['t_recv'])
            if first is None:
                first = t_recv
            rel = t_recv - first
            if rel < args.t_from or (args.t_to is not None and rel > args.t_to):
                continue
            out_lines.append(line if line.endswith('\n') else line + '\n')
            kept += 1
    mode = 'w' if args.overwrite else 'x'
    try:
        with open(args.out, mode, encoding='utf-8', newline='') as handle:
            handle.writelines(out_lines)
    except FileExistsError:
        print(f'{args.out} exists; use --overwrite', file=sys.stderr)
        return 2
    print(f'wrote {kept} lines to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
