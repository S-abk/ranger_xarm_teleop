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
Map MoveIt Servo's singularity measure over the tool poses around a posture.

    tools/workspace_map.py SOURCE [--from q1 q2 q3 q4 q5 q6] [--extent 0.30]
                           [--spacing 0.10] [--csv FILE] [--json]

No ROS. ``SOURCE`` is a URDF file or a session recording (its description row is used).
From the posture (default: the ready posture of ``tools/move_arm.py``) the tool is walked
along and turned about each axis of the arm base, and to every point of a cube of tool
positions, and the Jacobian condition number is reported: how far the tool gets before
MoveIt Servo would slow down and stop, the joint speed that needs, and the share of the
cube that stays below each level. ``--csv`` writes the grid, one tool position per row.
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'ranger_xarm_teleop_servo'))

from ranger_xarm_teleop_servo import kinematics, workspace  # noqa: E402
from ranger_xarm_teleop_servo.session import READY_POSTURE  # noqa: E402


def load_urdf(path: str) -> str:
    """Return the URDF text of a URDF file or of a session recording."""
    with open(path, encoding='utf-8') as handle:
        head = handle.read(1)
        handle.seek(0)
        if head != '{':
            return handle.read()
        for line in handle:
            row = json.loads(line)
            if row.get('k') == 'description':
                return row['urdf']
    raise SystemExit(f'{path}: no robot description row')


def main() -> int:
    """Parse the command line and print the map."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('source')
    parser.add_argument('--from', dest='start', nargs=6, type=float, default=READY_POSTURE)
    parser.add_argument('--extent', type=float, default=0.30)
    parser.add_argument('--spacing', type=float, default=0.10)
    parser.add_argument('--csv', default='')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    chain = kinematics.chain_from_urdf(load_urdf(args.source))
    result = workspace.evaluate(chain, args.start, args.extent, args.spacing)
    print(json.dumps(result, indent=2) if args.json else workspace.format_evaluation(result))
    if args.csv:
        with open(args.csv, 'x', encoding='utf-8') as handle:
            handle.write('dx,dy,dz,ended,condition,worst_on_the_way,limit_margin_rad\n')
            for row in workspace.grid(chain, args.start, args.extent, args.spacing):
                value = '' if row['condition'] is None else f"{row['condition']:.2f}"
                handle.write('{:.3f},{:.3f},{:.3f},'.format(*row['offset'])
                             + f"{row['ended']},{value},{row['worst']:.2f},"
                             f"{row['margin']:.3f}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
