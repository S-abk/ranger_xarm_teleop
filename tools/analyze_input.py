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
Analyse a JSONL recording of relay text: spread, gaps, absent/emulated intervals, timing.

    tools/analyze_input.py recordings/head_turn_a.jsonl --hand right [--from 2 --to 12] [--json]

No ROS needed. Spread is reported over frames where the hand's pose was present and not
emulated; use --from/--to (seconds from the first record) to select the still part.
"""

import argparse
import json
import sys

import _pkgpath  # noqa: F401  (sys.path for the package)

from ranger_xarm_teleop_bridge.analysis import analyze, format_report  # noqa: E402
from ranger_xarm_teleop_bridge.jsonl_io import iter_records  # noqa: E402


def main() -> int:
    """Run the analysis CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='JSONL recording (from the bridge record_path or tools/record_relay.py)')
    ap.add_argument('--hand', choices=('left', 'right'), default='right')
    ap.add_argument('--from', dest='t_from', type=float, default=None, help='window start, s from first record')
    ap.add_argument('--to', dest='t_to', type=float, default=None, help='window end, s from first record')
    ap.add_argument('--json', action='store_true', help='print the full report as JSON')
    args = ap.parse_args()
    report = analyze(iter_records(args.path), hand=args.hand, t_from=args.t_from, t_to=args.t_to)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        print(format_report(report))
    return 0 if 'error' not in report else 1


if __name__ == '__main__':
    sys.exit(main())
