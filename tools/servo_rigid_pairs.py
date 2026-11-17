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
Print the self-collision pairs MoveIt Servo checks that no arm motion can change.

    tools/servo_rigid_pairs.py --urdf FILE --srdf FILE [--group xarm6]

No ROS needed. ``--urdf`` is the expanded robot description (for example
``xacro ranger_xarm.urdf.xacro > robot.urdf``), ``--srdf`` the SRDF move_group reads,
``--group`` MoveIt Servo's ``move_group_name``. The output is one ``<disable_collisions>``
line per pair the SRDF leaves enabled whose two links have no joint of the group between
them (``ranger_xarm_teleop_servo/rigid_pairs.py``); nothing when there is none. Exit status
1 when there are such pairs. The teleop node logs the same list at start-up.
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'ranger_xarm_teleop_servo'))

from ranger_xarm_teleop_servo.rigid_pairs import rigid_pairs, srdf_lines  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--srdf', required=True)
    parser.add_argument('--group', default='xarm6')
    args = parser.parse_args()
    with open(args.urdf, encoding='utf-8') as handle:
        urdf = handle.read()
    with open(args.srdf, encoding='utf-8') as handle:
        srdf = handle.read()
    pairs = rigid_pairs(urdf, srdf, args.group)
    for line in srdf_lines(pairs):
        print(line)
    print(f'{len(pairs)} pairs', file=sys.stderr)
    return 1 if pairs else 0


if __name__ == '__main__':
    sys.exit(main())
