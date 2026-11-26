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
r"""
Build the link keep-out's clearance tables of a robot description into the cache (G-15).

One process, under an exclusive lock on the cache directory: a second builder for the same
cache waits, then finds the entries written and builds nothing. Every entry is written whole
(a reader sees all of it or none). The teleop node starts this when an entry is missing and
only loads the cache; it can also be run ahead of time (a first start then needs no build)::

    ros2 run ranger_xarm_teleop_servo keepout_tables_build --description robot.urdf \
        --base link_base --tip link_eef

Progress lines go to standard output; exit status 0 when every entry is in the cache.
No ROS graph is joined: package:// meshes are found through the ament index.
"""

import argparse
import os
import sys
import time
from typing import List, Optional

from . import keepout as ko
from . import keepout_tables as kt


def default_workers() -> int:
    """At most four workers and half the cores (the build is memory-bound: ~0.8 GB each)."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--description', required=True,
                        help='robot description file (URDF text)')
    parser.add_argument('--base', required=True, help='arm base link')
    parser.add_argument('--tip', required=True, help='arm tip link')
    parser.add_argument('--cache', default=None,
                        help=f'cache directory (default {kt.cache_directory()})')
    parser.add_argument('--workers', type=int, default=default_workers())
    parser.add_argument('--with-parent', action='store_true',
                        help='stop when the process that started this one stops')
    parser.add_argument('--test-resolution', action='store_true',
                        help='the coarse tables of the unit tests of made-up robots (still a '
                        'lower bound everywhere, but far more conservative); never for an arm')
    return parser


def resolver():
    from ament_index_python.packages import get_package_share_directory
    return ko.make_resolver(get_package_share_directory)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.with_parent:
        kt._die_with_parent()
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    cache = args.cache or kt.cache_directory()
    started = time.perf_counter()

    def say(text: str) -> None:
        print(f'keepout_tables_build: {text}', flush=True)
    try:
        with open(args.description, encoding='utf-8') as handle:
            urdf = handle.read()
        plan = kt.make_plan(urdf, resolver(), args.base, args.tip,
                            kt.TEST_CONFIG if args.test_resolution else kt.TableConfig())
        for outer, inner, why in plan.unsupported:
            say(f'{outer} - {inner}: no table ({why}); it stays on the sphere covers')
        built = kt.build_missing(plan, cache, max(1, args.workers), say)
    except (OSError, ValueError, ko.ModelError) as err:
        print(f'keepout_tables_build: failed: {err}', file=sys.stderr, flush=True)
        return 1
    left = kt.missing(plan, cache)
    if left:
        print('keepout_tables_build: failed: entries still missing after the build: '
              + ', '.join(f'{o} - {i}' for o, i in left), file=sys.stderr, flush=True)
        return 1
    say(f'done: {built} built, {len(plan.pairs) + len(plan.one) - built} already in {cache}, '
        f'{time.perf_counter() - started:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
