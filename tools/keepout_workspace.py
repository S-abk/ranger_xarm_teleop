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
What the link keep-out takes away from the arm's workspace, at several margins.

    tools/keepout_workspace.py --out DIR [--samples 200000] [--margins 0 0.01 0.025 0.04]
        [--description FILE | --xacro-args "..."] [--compare-urdf FILE] [--cell 0.02]
        [--useful-samples 200000]

No ROS graph is needed (ROS must be sourced: xacro, and the meshes' package paths). The robot
description is the mock stack's: ``ranger_xarm_description``'s ``ranger_xarm.urdf.xacro``
expanded with ``MOCK_DESCRIPTION_ARGS`` read from
``ranger_xarm_teleop_bringup/launch/teleop_mock.launch.py`` (so the two cannot drift), or a
URDF file given with ``--description``. The model is built as the servo node builds it (base
``xarm_link_base``, tip ``xarm_link_tcp``); the pairs it keeps depend on the margin it is built
with, 0.025 m, for every margin below.

Joint postures are uniform within the joint limits (joints 1, 4 and 6 within +-pi). A posture is
*reachable* when the model's clearance is >= 0 (no contact by the model). For each margin
``m``:

* the share of reachable postures whose clearance is below ``m`` (removed by the keep-out);
* tool positions (the tip frame, in the arm base frame) of the reachable postures are put in
  ``--cell`` (2 cm) cells; a cell is *lost* when every reachable posture in it has clearance
  below ``m``, and the share of cells lost, and where (x, y, z ranges, and the share lost per
  slab of x), is reported. The number of postures per cell is small, so a second figure
  counts only cells with at least three reachable postures.

The same for the *useful* set used before: tool pointing down within 30 degrees of -z (the
tool z axis within 30 degrees of -z), x > 0.15 m, z > 0.05 m.

The useful set is a small part (under 1 %) of uniform postures, so its cells hold few
postures each: ``--useful-samples`` adds a denser run of the useful set alone (postures drawn
uniformly, kept if their tool pose is useful, until that many are kept; the keep-out model is
evaluated only on those).

The model's inflation (it can only give away clearance, never gain it) is part of every
figure: there is no true geometry to compare with offline, so the cost is given at several
margins to show the slope.
"""

import argparse
import ast
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'ranger_xarm_teleop_servo'))

from ranger_xarm_teleop_servo import keepout  # noqa: E402
from ranger_xarm_teleop_servo.kinematics import chain_from_urdf, forward  # noqa: E402

BASE, TIP, MODEL_MARGIN = 'xarm_link_base', 'xarm_link_tcp', 0.025
LAUNCH = os.path.join(REPO, 'src', 'ranger_xarm_teleop_bringup', 'launch', 'teleop_mock.launch.py')
DESCRIPTION_XACRO = os.path.join('urdf', 'ranger_xarm.urdf.xacro')
DOWN_COS = math.cos(math.radians(30.0))
SLABS = np.arange(-1.0, 1.01, 0.1)

_STATE = {}                      # set before the pool forks


# --- pure helpers ----------------------------------------------------------------------------

def mock_description_args(launch_file: str = LAUNCH) -> str:
    """Return ``MOCK_DESCRIPTION_ARGS`` as the launch file defines it (read, not imported)."""
    with open(launch_file, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'MOCK_DESCRIPTION_ARGS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f'{launch_file}: no MOCK_DESCRIPTION_ARGS')


def useful(position: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Return a mask of the useful tool poses (down within 30 deg, x > 0.15, z > 0.05)."""
    return (down > DOWN_COS) & (position[:, 0] > 0.15) & (position[:, 2] > 0.05)


def cell_ids(position: np.ndarray, cell: float):
    """Return ``(unique cell index triples, inverse)`` of ``position`` binned at ``cell`` m."""
    index = np.floor(position / cell).astype(np.int64)
    return np.unique(index, axis=0, return_inverse=True)


def cell_costs(position, clearance, margins, cell, min_samples=1):
    """
    Return, for each margin, the cells lost and the range of the lost cells' centres.

    A cell is lost when its largest clearance is below the margin; only cells with at least
    ``min_samples`` samples count. ``clearance`` is that of the reachable samples only.
    """
    if len(position) == 0:
        return {'cells': 0}
    cells, inverse = cell_ids(position, cell)
    inverse = inverse.reshape(-1)
    best = np.full(len(cells), -np.inf)
    np.maximum.at(best, inverse, clearance)
    count = np.bincount(inverse, minlength=len(cells))
    keep = count >= min_samples
    centres = (cells + 0.5) * cell
    out = {'cells': int(keep.sum()), 'samples_per_cell_median': float(np.median(count)),
           'by_margin': {}}
    for m in margins:
        lost = keep & (best < m)
        entry = {'lost': int(lost.sum()),
                 'share': float(lost.sum() / max(int(keep.sum()), 1))}
        if lost.any():
            lc = centres[lost]
            entry['range'] = {axis: {'min': float(lc[:, i].min()),
                                     'p5': float(np.percentile(lc[:, i], 5)),
                                     'median': float(np.median(lc[:, i])),
                                     'p95': float(np.percentile(lc[:, i], 95)),
                                     'max': float(lc[:, i].max())}
                              for i, axis in enumerate('xyz')}
            slabs = []
            for lo, hi in zip(SLABS[:-1], SLABS[1:]):
                inside = keep & (centres[:, 0] >= lo) & (centres[:, 0] < hi)
                if inside.any():
                    slabs.append({'x_from': round(float(lo), 2), 'x_to': round(float(hi), 2),
                                  'cells': int(inside.sum()),
                                  'lost_share': float((inside & lost).sum() / inside.sum())})
            entry['x_slabs'] = slabs
        out['by_margin'][str(m)] = entry
    return out


def forward_batch(chain, q: np.ndarray):
    """Return the tool positions (n x 3) and down values (n) of postures ``q`` (n x joints)."""
    n = len(q)
    pose = np.broadcast_to(np.eye(4), (n, 4, 4))
    index = 0
    for seg in chain.segments:
        pose = pose @ seg.origin
        if seg.axis is not None:
            x, y, z = seg.axis
            c, s = np.cos(q[:, index]), np.sin(q[:, index])
            k = 1.0 - c
            rot = np.zeros((n, 4, 4))
            rot[:, 0, 0], rot[:, 0, 1], rot[:, 0, 2] = c + x * x * k, x * y * k - z * s, \
                x * z * k + y * s
            rot[:, 1, 0], rot[:, 1, 1], rot[:, 1, 2] = y * x * k + z * s, c + y * y * k, \
                y * z * k - x * s
            rot[:, 2, 0], rot[:, 2, 1], rot[:, 2, 2] = z * x * k - y * s, z * y * k + x * s, \
                c + z * z * k
            rot[:, 3, 3] = 1.0
            pose = pose @ rot
            index += 1
    return pose[:, :3, 3].copy(), -pose[:, 2, 2]


def sample_costs(clearance, margins):
    """Return for each margin the share of the reachable samples with clearance below it."""
    reachable = clearance[clearance >= 0.0]
    return {'reachable': int(len(reachable)),
            'removed_share': {str(m): float((reachable < m).mean()) if len(reachable) else 0.0
                              for m in margins}}


# --- sampling --------------------------------------------------------------------------------

def _work(job):
    """Return ``(clearance, tool position, down)`` for one chunk of postures."""
    seed, count, cap = job
    model, chain = _STATE['model'], _STATE['chain']
    rng = np.random.default_rng(seed)
    lower, upper = _STATE['lower'], _STATE['upper']
    clearance = np.empty(count)
    position = np.empty((count, 3))
    down = np.empty(count)
    given = _STATE.get('given', {}).get(seed)       # postures handed in, not drawn
    for i in range(count):
        q = rng.uniform(lower, upper) if given is None else given[i]
        rows = model.evaluate(q, influence=cap, per_pair=1)
        # Exact up to the cap; above it only a bound that is above the cap too.
        clearance[i] = rows.clearance if rows.exact else max(rows.clearance, cap)
        pose = forward(chain, q)[0]
        position[i] = pose[:3, 3]
        down[i] = -pose[2, 2]
    return clearance, position, down


def expand_description(extra_args: str) -> str:
    """Expand the robot description with xacro and return the URDF text."""
    from ament_index_python.packages import get_package_share_directory
    xacro_file = os.path.join(get_package_share_directory('ranger_xarm_description'),
                              DESCRIPTION_XACRO)
    command = ['xacro', xacro_file] + extra_args.split()
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit('xacro failed:\n' + done.stderr[-3000:])
    return done.stdout


def format_report(r: dict) -> str:
    """Return the text summary of a report."""
    pct = lambda v: f'{100 * v:.1f} %'  # noqa: E731
    lines = [f'Keep-out cost on the workspace ({r["samples"]} uniform joint postures, '
             f'reachable = model clearance >= 0, model built at margin {r["model_margin"]})',
             f'  description: {r["description"]}; model fingerprint {r["fingerprint"][:16]}'
             + (f'; same as {r["compare_urdf"]}: {r["fingerprint_matches"]}'
                if r.get('compare_urdf') else ''),
             f'  tool position cells: {r["cell"] * 100:.0f} cm; ready posture tool '
             f'{r["ready_tool_position"]}']
    titles = {'all': 'all reachable postures',
              'useful': 'useful set (tool within 30 deg of pointing down, x > 0.15, z > 0.05)',
              'useful_dense': 'useful set, dense: only useful postures drawn'}
    for name in ('all', 'useful', 'useful_dense'):
        if name not in r['sets']:
            continue
        part = r['sets'][name]
        title = titles[name]
        lines.append(f'\n[{title}]')
        lines.append(f'  reachable postures: {part["samples"]["reachable"]}; tool cells with '
                     f'reachable postures: {part["cells"]["cells"]} '
                     f'(median {part["cells"]["samples_per_cell_median"]:.0f} postures per cell)')
        lines.append('  margin     postures removed   cells lost   (cells with >= 3 postures)')
        for m in r['margins']:
            k = str(m)
            c = part['cells']['by_margin'][k]
            c3 = part['cells3']['by_margin'][k]
            lines.append(f'  {m * 1000:5.1f} mm   {pct(part["samples"]["removed_share"][k]):>12}'
                         f'   {pct(c["share"]):>9} ({c["lost"]})   {pct(c3["share"]):>6} '
                         f'({c3["lost"]} of {part["cells3"]["cells"]})')
        k = str(min(r['margins'], key=lambda m: abs(m - r['model_margin'])))
        c = part['cells']['by_margin'][k]
        if c['lost']:
            lines.append(f'  where, margin {float(k) * 1000:.0f} mm (centres of the lost cells; '
                         f'min / 5 % / median / 95 % / max):')
            for axis in 'xyz':
                v = c['range'][axis]
                lines.append(f'    {axis}: {v["min"]:+.2f} / {v["p5"]:+.2f} / {v["median"]:+.2f}'
                             f' / {v["p95"]:+.2f} / {v["max"]:+.2f} m')
            lines.append('    share of cells lost per slab of x: ' + ', '.join(
                f'[{s["x_from"]:+.1f},{s["x_to"]:+.1f}) {pct(s["lost_share"])} of {s["cells"]}'
                for s in c['x_slabs'] if s['lost_share'] > 0.0))
    lines.append(f'\nwall time {r["wall_time_s"]:.0f} s')
    return '\n'.join(lines) + '\n'


def main() -> int:
    """Sample, evaluate and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', required=True, help='output directory (created)')
    parser.add_argument('--samples', type=int, default=200000)
    parser.add_argument('--margins', type=float, nargs='+', default=[0.0, 0.010, 0.025, 0.040])
    parser.add_argument('--cell', type=float, default=0.02)
    parser.add_argument('--description', default='', help='URDF file instead of the mock xacro')
    parser.add_argument('--xacro-args', default='', help='xacro arguments (default: mock stack)')
    parser.add_argument('--compare-urdf', default='',
                        help='URDF file whose model fingerprint must equal this one')
    parser.add_argument('--useful-samples', type=int, default=200000)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    os.makedirs(args.out, exist_ok=True)
    margins = sorted(args.margins)

    from ament_index_python.packages import get_package_share_directory
    resolve = keepout.make_resolver(get_package_share_directory)
    if args.description:
        urdf = open(args.description, encoding='utf-8').read()
        source = args.description
    else:
        xacro_args = args.xacro_args or mock_description_args()
        urdf = expand_description(xacro_args)
        source = f'ranger_xarm.urdf.xacro {xacro_args}'
    with open(os.path.join(args.out, 'description.urdf'), 'w', encoding='utf-8') as handle:
        handle.write(urdf)
    model = keepout.build(urdf, resolve, BASE, TIP, MODEL_MARGIN)
    report = {'description': source, 'fingerprint': model.fingerprint, 'samples': args.samples,
              'model_margin': MODEL_MARGIN, 'margins': margins, 'cell': args.cell,
              'seed': args.seed,
              'model': {'pairs': len(model.pairs), 'spheres': model.sphere_count,
                        'largest_sphere_radius_m': model.largest_radius,
                        'excluded': [[e.moving, e.other, e.reason] for e in model.excluded]}}
    if args.compare_urdf:
        other = keepout.fingerprint(open(args.compare_urdf, encoding='utf-8').read(), resolve,
                                    BASE, TIP, MODEL_MARGIN)
        report['compare_urdf'] = args.compare_urdf
        report['fingerprint_matches'] = other == model.fingerprint
        print(f'fingerprint {model.fingerprint[:16]} vs {args.compare_urdf}: '
              f'{"equal" if other == model.fingerprint else "DIFFERENT"}', flush=True)

    chain = chain_from_urdf(urdf, BASE, TIP)
    lower = np.array([j.lower for j in chain.joints])
    upper = np.array([j.upper for j in chain.joints])
    for j in (0, 3, 5):
        lower[j], upper[j] = max(lower[j], -math.pi), min(upper[j], math.pi)
    _STATE.update(model=model, chain=chain, lower=lower, upper=upper)
    report['joint_limits_used'] = [[float(a), float(b)] for a, b in zip(lower, upper)]
    report['ready_tool_position'] = [round(float(v), 3) for v in forward(
        chain, [0.0, -0.6, -0.7, 0.0, 1.3, 0.0])[0][:3, 3]]

    cap = max(margins) + 0.005
    chunk = 2000
    jobs = [(args.seed * 1000003 + i, min(chunk, args.samples - i * chunk), cap)
            for i in range((args.samples + chunk - 1) // chunk)]
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        results = pool.map(_work, jobs, chunksize=1)
    clearance = np.concatenate([r[0] for r in results])
    position = np.concatenate([r[1] for r in results])
    down = np.concatenate([r[2] for r in results])
    print(f'{len(clearance)} postures evaluated in {time.time() - started:.0f} s', flush=True)

    sets = {}
    for name, mask in (('all', np.ones(len(clearance), dtype=bool)),
                       ('useful', useful(position, down))):
        part_clear, part_pos = clearance[mask], position[mask]
        reach = part_clear >= 0.0
        sets[name] = {
            'all_samples_in_set': int(mask.sum()),
            'samples': sample_costs(part_clear, margins),
            'cells': cell_costs(part_pos[reach], part_clear[reach], margins, args.cell),
            'cells3': cell_costs(part_pos[reach], part_clear[reach], margins, args.cell, 3)}
    if args.useful_samples > 0:
        kept, tried = [], 0
        rng = np.random.default_rng(args.seed + 7)
        while sum(len(k) for k in kept) < args.useful_samples:
            q = rng.uniform(lower, upper, size=(1000000, len(lower)))
            pos, dn = forward_batch(chain, q)
            kept.append(q[useful(pos, dn)])
            tried += len(q)
        q = np.concatenate(kept)[:args.useful_samples]
        print(f'useful set: {len(q)} postures out of {tried} drawn '
              f'({100 * len(q) / tried:.2f} %)', flush=True)
        jobs, base_seed = [], 900000001
        _STATE['given'] = {}
        for i in range(0, len(q), chunk):
            _STATE['given'][base_seed + i] = q[i:i + chunk]
            jobs.append((base_seed + i, len(q[i:i + chunk]), cap))
        with multiprocessing.get_context('fork').Pool(args.workers) as pool:
            results = pool.map(_work, jobs, chunksize=1)
        clear = np.concatenate([r[0] for r in results])
        pos = np.concatenate([r[1] for r in results])
        reach = clear >= 0.0
        sets['useful_dense'] = {
            'drawn': tried, 'all_samples_in_set': len(q),
            'samples': sample_costs(clear, margins),
            'cells': cell_costs(pos[reach], clear[reach], margins, args.cell),
            'cells3': cell_costs(pos[reach], clear[reach], margins, args.cell, 3)}
    report['sets'] = sets
    report['wall_time_s'] = time.time() - started
    with open(os.path.join(args.out, 'keepout_workspace.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=1)
    text = format_report(report)
    with open(os.path.join(args.out, 'keepout_workspace.txt'), 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
