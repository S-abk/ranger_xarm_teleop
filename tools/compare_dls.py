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
Compare the joint-space solver with MoveIt Servo's singularity rule, offline.

    tools/compare_dls.py [URDF_OR_SESSION] [--session recordings/NAME.jsonl] [--skip cube]
                         [--sessions 100] [--segments 30] [--seed 7] [--out FILE]

No ROS. The arm comes from a URDF file or from the description row of a session recording
(the first argument, or the session). Three parts, each printed as a table:

1. a cube of 343 tool positions around the ready posture, each reached by a straight line
   at fixed orientation, the target moving at the speed caps and then held for 1 s: the
   old rule (a model of MoveIt Servo's singularity scaling and hard stop) against the
   solver;
2. the 100-session model of ``tools/workspace_map.py``'s survey (chained one-second motions
   of random direction) driven by the solver, against the old rule's figures;
3. with ``--session``: an offline replay of a recorded session. Each engagement is seeded
   from the nearest joint row and the solver runs over its recorded targets with the
   delayed arm model (150 ms, reference two cycles old). The file is streamed.

The figures are reproducible: the cube and the session model have no random choice beyond
the seed. Which parts run is chosen with ``--skip`` (cube, sessions, replay).
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'ranger_xarm_teleop_servo'))
import _pkgpath  # noqa: F401,E402  (sys.path for the bridge package)

from ranger_xarm_teleop_servo import comparison, workspace  # noqa: E402
from ranger_xarm_teleop_servo.session import READY_POSTURE  # noqa: E402
from ranger_xarm_teleop_servo.solver import SolverParams  # noqa: E402

LINEAR_CAP, ANGULAR_CAP = 0.25, 1.0     # the speed caps of the limits file
JOINT_LIMIT = 2.0                       # rad/s, the arm's joint_limits.yaml
DELAY_TICKS, STALE_TICKS = 8, 3         # 150 ms and a reference two cycles old


def load_urdf(path: str) -> str:
    """Return the URDF text of a URDF file or of a session recording (description row)."""
    with open(path, encoding='utf-8') as handle:
        head = handle.read(1)
        handle.seek(0)
        if head != '{':
            return handle.read()
        for line in handle:
            if '"description"' in line[:80]:
                return json.loads(line)['urdf']
    raise SystemExit(f'{path}: no robot description row')


def recorded_servo_counts(path: str) -> dict:
    """Return how often the recording's MoveIt Servo status entered a halt or a deceleration."""
    halts = decelerating = 0
    last = None
    for row in comparison.iter_rows(open(path, encoding='utf-8'), ('servo',)):
        code = row['code']
        if code != last:
            halts += code in (2, 5, 6)
            decelerating += code in (1, 3, 4)
        last = code
    return {'halts': halts, 'decelerating': decelerating}


def main() -> int:
    """Run the comparison and print (and optionally save) the report."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('source', nargs='?', default='',
                        help='a URDF file or a session recording (default: --session)')
    parser.add_argument('--session', default='', help='a recording to replay offline')
    parser.add_argument('--skip', nargs='*', default=[], choices=['cube', 'sessions', 'replay'])
    parser.add_argument('--from', dest='start', nargs=6, type=float, default=READY_POSTURE)
    parser.add_argument('--spacing', type=float, default=0.10)
    parser.add_argument('--sessions', type=int, default=100)
    parser.add_argument('--segments', type=int, default=30)
    parser.add_argument('--seed', type=int, default=7, help='seed of the session model '
                        '(7 is the one the servo_diag figures were made with)')
    parser.add_argument('--out', default='', help='also write the report to this file')
    args = parser.parse_args()

    if not args.source and not args.session:
        parser.error('give a URDF file or a recording')
    urdf = load_urdf(args.source or args.session)
    arm = comparison.arm_from_description(urdf, JOINT_LIMIT)
    params = SolverParams(linear_speed_cap=LINEAR_CAP, angular_speed_cap=ANGULAR_CAP)
    posture = ' '.join(f'{v:+.2f}' for v in args.start)
    header = (f'arm: {arm.chain.base} -> {arm.chain.tip}; start posture (rad) {posture}; '
              f'speed caps {LINEAR_CAP} m/s, {ANGULAR_CAP} rad/s; joint limit {JOINT_LIMIT} '
              f'rad/s; solver gain {params.gain:g} /s, damping from conditioning '
              f'{params.damping_threshold:g}; old rule thresholds 17 / 30')
    print(header, flush=True)
    lines = [header]

    def part(title: str, text: str, began: float) -> None:
        block = f'\n== {title} ({time.time() - began:.0f} s)\n{text}'
        lines.append(block)
        print(block, flush=True)

    if 'cube' not in args.skip:
        began = time.time()
        offsets = comparison.cube_offsets(0.30, args.spacing)
        rows = workspace.grid(arm.chain, args.start, 0.30, args.spacing, step=0.01)
        reference = workspace.grid_summary(rows, (17.0, 30.0, 100.0))['reached_below']
        reference['complete'] = sum(1 for r in rows if r['ended'] == 'complete') / len(rows)
        result = comparison.cube(arm, params, args.start, offsets)
        part('1. cube of tool positions', comparison.format_cube(result, reference), began)
    if 'sessions' not in args.skip:
        began = time.time()
        old = workspace.wander(arm.chain, args.start, 17.0, 30.0, sessions=args.sessions,
                               segments=args.segments, seed=args.seed,
                               joint_speed_limit=JOINT_LIMIT)
        new = comparison.sessions(arm, params, args.start, args.sessions, args.segments,
                                  seed=args.seed)
        part('2. session model', comparison.format_sessions(new, old), began)
    if args.session and 'replay' not in args.skip:
        began = time.time()
        with open(args.session, encoding='utf-8') as handle:
            rows = comparison.iter_rows(handle, ('state', 'target', 'joints'))
            engagements = comparison.read_engagements(rows)
        seeds = comparison.seed_errors(arm, engagements)
        result = comparison.replay(arm, params, engagements, DELAY_TICKS, STALE_TICKS)
        text = comparison.format_replay(result, seeds, recorded_servo_counts(args.session))
        text += ('\nnote: the recorded targets come from the earlier driving path, which had no '
                 'lead clamp, so the solver error includes the lag behind a hand that '
                 'outran the arm')
        part(f'3. replay of {os.path.basename(args.session)}', text, began)
    notes = ['', 'notes',
             '- old rule (model): the tool steps toward the target each cycle, the step limited '
             'to the speed caps and scaled by the singularity rule (hard stop: nothing moves '
             'again), then to the joint limit; it trails the moving target by one cycle '
             '(5 mm at the cap) even when free, and a motion it only slowed can catch up '
             'during the hold, so its "reached" is higher than the servo_diag walk shares',
             '- solver: kinematic, reference = the arm; the target also gives its own twist '
             'as feedforward. Position has priority, so the error it accepts is in orientation',
             '- the session model ends an old-rule session at its first halt or joint limit; '
             'a solver session never ends, so it spends more time near joint limits (the '
             'solver starts holding a joint 0.15 rad from its bound, the old model stops at '
             '0.1 rad)']
    lines.extend(notes)
    report = '\n'.join(lines)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            handle.write(report + '\n')
        print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
