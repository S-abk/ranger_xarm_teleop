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
Compare the link keep-out model's clearance with MoveIt's collision checker (INV-4).

    tools/compare_keepout_moveit.py --out DIR [--urdf FILE] [--srdf FILE] [--uniform 2000]
        [--near 2000] [--paths FIXTURE] [--inflation 300] [--workers 6] [--parity 300]
        [--tol 0.0005] [--seed 0] [--reference true|moveit]

INV-4: the model never reports more clearance than MoveIt's collision checker does. Run with
the mock (or any) stack up: the robot description is read from ``/robot_description`` (or
``--urdf``), the SRDF from ``ranger_xarm_moveit_config``, and the model is built with base
``xarm_link_base``, tip ``xarm_link_tcp`` and margin 0.025 m, as the servo node builds it.

**The test.** For every posture and every model pair (moving link, other link) whose model
clearance ``d`` is in [-0.02, 0.10] m: pad every robot link by ``p / 2`` (so a pair is padded
by ``p`` in all) with ``p`` just under ``d - TOL`` and ask MoveIt whether the pair is in
contact. MoveIt pads a mesh by moving its vertices outward by the padding, so a padded mesh
lies inside the Minkowski sum of the mesh and a ball of that radius; a box grows by the padding
on each side, which contains that Minkowski sum. A contact reported at total padding ``p``
therefore means the true distance is at most ``p`` (exactly so for meshes; boxes are
conservative: they can only report extra contacts, false alarms). A contact at
``p = d - TOL`` is thus a violation: the model is more than ``TOL`` above MoveIt. No contact
proves nothing about how much the model gives away: that is measured by the inflation sweep.

**Padding buckets.** The scene is padded once per 0.5 mm bucket, not per posture, so ``p`` is
rounded UP to the bucket: ``p' = ceil((d - TOL) / 0.0005) * 0.0005``, which is >= ``d - TOL``
(the test is not weakened) and, because ``TOL`` is one bucket, ``p' <= d``. The tolerance in
force for a case is then ``d - p'`` which lies in (0, TOL]; the largest is ``TOL`` (0.5 mm),
and it is listed per violation. Pairs with ``d < TOL`` would need a padding <= 0 (negative
padding is never sent) and are counted as not testable by padding.

**Why not the move_group services.** ``/check_state_validity`` checks robot against robot
with ``CollisionRequest::pad_self_collisions = false`` (its default; the service has no field
for it), so padding the planning scene through ``/apply_planning_scene`` has no effect on it:
at 1.0 m on every link the posture still reports no contact (shown by the in-process self
check below, and in the log). Every pair of this robot is robot against robot, so the test
runs MoveIt's own collision environment in process instead (``tools/keepout_moveit``:
``planning_scene::PlanningScene``, FCL, the SRDF's allowed-collision matrix, padded
self-collision check). ``--parity`` shows that at zero padding the helper's contacts equal
the service's on random postures.

Pairs the SRDF allows to collide are never checked by MoveIt: they are listed and excluded.

**Which geometry MoveIt checks.** ``--reference true`` runs MoveIt's checker on the
``/robot_description`` geometry, the one the model is built from (the model's guarantee is
about this). ``--reference moveit`` runs it on move_group's own ``robot_description``
parameter, which may differ: the MoveIt launches grow every fixed-structure box by a planning
margin per side (``planning_margin.xacro``), so MoveIt then reports less clearance than the
true geometry has. Only with ``moveit`` is the helper the same checker as the service, which
``--parity`` then compares.

**Also reported.** (1) FCL's exact distance for every checked pair at every posture, to
tell a false alarm (box padding) from a real violation and to measure inflation directly:
a model clearance above FCL's distance by more than the tolerance is a violation whatever the
padding does (penetrating pairs, where both report contact, are not compared).
(2) The inflation sweep: for a random subsample of (posture, pair) cases with ``d`` in
[0, 0.06] m the smallest bucket at which MoveIt reports the pair in contact, which is an upper
bound of MoveIt's clearance (a contact at p means distance <= p), so ``d_moveit - d_model``
overstates the model's inflation by at most the box conservatism.
"""

import argparse
import concurrent.futures
import json
import math
import os
import queue
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'ranger_xarm_teleop_servo'))

from ranger_xarm_teleop_servo import keepout  # noqa: E402
from ranger_xarm_teleop_servo.kinematics import chain_from_urdf  # noqa: E402

BASE, TIP, MARGIN = 'xarm_link_base', 'xarm_link_tcp', 0.025
STEP = 0.0005                    # padding bucket (m)
SWEEP_TOP = 0.10                 # the inflation sweep's largest total padding (m)
RANGE = (-0.02, 0.10)            # model clearance range that is tested (m)
HELPER_DIR = os.path.join(REPO, 'tools', 'keepout_moveit')
HELPER_BUILD = os.path.join(REPO, 'build', 'keepout_moveit')


# --- pure helpers ----------------------------------------------------------------------------

def bucket_up(padding: float, step: float = STEP) -> int:
    """Return the index of the smallest bucket (multiple of ``step``) not below ``padding``."""
    return int(math.ceil(padding / step - 1e-9))


def pair_key(a: str, b: str) -> str:
    """Return the helper's name of an unordered pair of links."""
    return f'{a}|{b}' if a < b else f'{b}|{a}'


def plan_case(d: float, tol: float, step: float = STEP):
    """
    Return ``(bucket, padding, effective_tolerance)`` for a pair at model clearance ``d``.

    ``None`` when the pair cannot be tested by padding (the padding would be <= 0).
    The padding is ``bucket * step``, at least ``d - tol`` and at most ``d`` when
    ``tol >= step``.
    """
    if d - tol <= 0.0:
        return None
    k = bucket_up(d - tol, step)
    return k, k * step, d - k * step


def stats(values) -> dict:
    """Return count, median, 90th percentile, maximum, minimum of ``values`` (m)."""
    if len(values) == 0:
        return {'n': 0}
    v = np.asarray(values, dtype=float)
    return {'n': int(len(v)), 'min': float(v.min()), 'median': float(np.median(v)),
            'p90': float(np.percentile(v, 90)), 'max': float(v.max())}


# --- the helper process ----------------------------------------------------------------------

class Helper:
    """One ``keepout_moveit`` process (MoveIt's collision environment, driven by lines)."""

    def __init__(self, exe: str, urdf_file: str, srdf_file: str, joints, log_path: str):
        """Start the process on a URDF and SRDF file and tell it the posture's joint names."""
        self.log = open(log_path, 'w', encoding='utf-8')
        self.proc = subprocess.Popen([exe, urdf_file, srdf_file], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True)
        first = self.proc.stdout.readline().strip()
        if first != 'ready':
            raise RuntimeError(f'keepout_moveit did not start: {first!r} (see {log_path})')
        self.ask('joints ' + ' '.join(joints))

    def ask(self, command: str) -> str:
        """Send one command and return the reply line."""
        self.proc.stdin.write(command + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f'keepout_moveit died on: {command[:60]}')
        if line.startswith('error'):
            raise RuntimeError(line.strip())
        return line.strip()

    @staticmethod
    def _values(q) -> str:
        return ' '.join(repr(float(v)) for v in q)

    def contacts(self, q, mode: str = 'padded'):
        """Return the set of pair keys in contact at ``q`` with the current padding."""
        parts = self.ask(f'contacts {mode} {self._values(q)}').split()
        return set(parts[2:])

    def pad(self, per_link: float) -> None:
        """Set the padding of every link (m)."""
        self.ask(f'pad {per_link!r}')

    def distances(self, q) -> dict:
        """Return FCL's distance (m, negative in contact) of every checked pair within 0.2 m."""
        parts = self.ask(f'dist {self._values(q)}').split()[1:]
        return {parts[i]: float(parts[i + 1]) for i in range(0, len(parts), 2)}

    def allowed(self):
        """Return the set of pair keys MoveIt never checks (the SRDF's)."""
        return set(self.ask('acm').split()[1:])

    def links(self):
        """Return the links with collision geometry."""
        return self.ask('links').split()[1:]

    def close(self) -> None:
        """Stop the process."""
        try:
            self.proc.stdin.write('quit\n')
            self.proc.stdin.flush()
        except OSError:
            pass
        self.proc.wait(timeout=10)
        self.log.close()


class Pool:
    """Helpers used from threads: ``run(fn, items)`` gives each call its own helper."""

    def __init__(self, helpers):
        """Take the helpers to share."""
        self.helpers = helpers
        self.free = queue.Queue()
        for h in helpers:
            self.free.put(h)

    def run(self, fn, items):
        """Return ``[fn(helper, item) for item in items]``, calls in parallel."""
        def call(item):
            helper = self.free.get()
            try:
                return fn(helper, item)
            finally:
                self.free.put(helper)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.helpers)) as pool:
            return list(pool.map(call, items))


def ensure_helper(path: str) -> str:
    """Return the helper executable, building it first if it is missing."""
    exe = path or os.path.join(HELPER_BUILD, 'keepout_moveit')
    if os.path.exists(exe):
        return exe
    if path:
        raise SystemExit(f'{path}: no such helper')
    print('building tools/keepout_moveit ...', flush=True)
    for cmd in (['cmake', '-S', HELPER_DIR, '-B', HELPER_BUILD],
                ['cmake', '--build', HELPER_BUILD]):
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise SystemExit(f'{" ".join(cmd)} failed:\n'
                             f'{done.stdout[-2000:]}{done.stderr[-2000:]}')
    return exe


# --- ROS: the description, and the service for the parity check ------------------------------

def read_description(timeout: float = 15.0) -> str:
    """Return the text on ``/robot_description`` (latched topic)."""
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    got = []
    node = rclpy.create_node('compare_keepout_moveit_description')
    node.create_subscription(
        String, '/robot_description', lambda m: got.append(m.data),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                   reliability=ReliabilityPolicy.RELIABLE))
    end = time.time() + timeout
    while not got and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    if not got:
        raise SystemExit('no /robot_description within %.0f s: is the stack up?' % timeout)
    return got[0]


def read_move_group_description(timeout: float = 15.0) -> str:
    """Return move_group's own ``robot_description`` parameter."""
    import rclpy
    from rcl_interfaces.srv import GetParameters
    node = rclpy.create_node('compare_keepout_moveit_params')
    client = node.create_client(GetParameters, '/move_group/get_parameters')
    if not client.wait_for_service(timeout_sec=timeout):
        raise SystemExit('no /move_group/get_parameters service: is move_group up?')
    request = GetParameters.Request()
    request.names = ['robot_description']
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    node.destroy_node()
    reply = future.result()
    if reply is None or not reply.values or not reply.values[0].string_value:
        raise SystemExit('move_group has no robot_description parameter')
    return reply.values[0].string_value


def service_parity(postures, joints, extra, helper: Helper, count: int, rng) -> dict:
    """Compare the helper's unpadded contacts with ``/check_state_validity`` (group "")."""
    import rclpy
    from moveit_msgs.srv import GetStateValidity
    node = rclpy.create_node('compare_keepout_moveit_parity')
    client = node.create_client(GetStateValidity, '/check_state_validity')
    if not client.wait_for_service(timeout_sec=10.0):
        raise SystemExit('no /check_state_validity service')
    helper.pad(0.0)
    names = list(joints) + list(extra)
    same = different = truncated = with_contacts = 0
    examples = []
    for i in rng.choice(len(postures), size=min(count, len(postures)), replace=False):
        q = postures[i]
        request = GetStateValidity.Request()
        request.group_name = ''
        request.robot_state.joint_state.name = names
        request.robot_state.joint_state.position = [float(v) for v in q] + [0.0] * len(extra)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
        reply = future.result()
        if reply is None:
            raise SystemExit('/check_state_validity did not answer')
        theirs = {pair_key(c.contact_body_1, c.contact_body_2) for c in reply.contacts}
        mine = helper.contacts(q, 'plain')
        if mine:
            with_contacts += 1
        if len(reply.contacts) > 40:
            truncated += 1
        if theirs == mine:
            same += 1
        else:
            different += 1
            if len(examples) < 10:
                examples.append({'q': [float(v) for v in q], 'service': sorted(theirs),
                                 'helper': sorted(mine)})
    node.destroy_node()
    return {'postures': same + different, 'identical_contact_sets': same,
            'different': different, 'with_contacts': with_contacts,
            'service_contact_list_possibly_truncated': truncated, 'examples': examples}


# --- the comparison --------------------------------------------------------------------------

def collision_shapes(urdf: str) -> dict:
    """Return link -> sorted set of collision shape kinds (box, mesh, ...)."""
    import xml.etree.ElementTree as ElementTree
    out = {}
    for link in ElementTree.fromstring(urdf).findall('link'):
        kinds = set()
        for collision in link.findall('collision'):
            geometry = collision.find('geometry')
            if geometry is not None and len(geometry):
                kinds.add(geometry[0].tag)
        if kinds:
            out[link.get('name')] = sorted(kinds)
    return out


def sample_postures(model, chain, args, rng):
    """Return ``[(q, source)]``: uniform, near-boundary, and the contact-path points."""
    lower = np.array([j.lower for j in chain.joints])
    upper = np.array([j.upper for j in chain.joints])
    out = [(rng.uniform(lower, upper), 'uniform') for _ in range(args.uniform)]
    tried = 0
    near = []
    while len(near) < args.near:
        q = rng.uniform(lower, upper)
        tried += 1
        rows = model.evaluate(q, influence=0.08, per_pair=1)
        if rows.exact and -0.01 <= rows.clearance <= 0.08:
            near.append((q, 'near'))
    out += near
    n_paths = 0
    if args.paths and os.path.exists(args.paths):
        with open(args.paths, encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                for point in row['positions'][::5]:
                    out.append((np.array(point[:6], dtype=float), 'path:' + row['name']))
                    n_paths += 1
    return out, {'near_tried': tried, 'near_accepted': len(near), 'path_points': n_paths}


def main() -> int:
    """Run the comparison and write the summary and JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', required=True, help='output directory (created)')
    parser.add_argument('--urdf', default='', help='URDF file instead of /robot_description')
    parser.add_argument('--srdf', default='')
    parser.add_argument('--helper', default='', help='keepout_moveit executable')
    parser.add_argument('--uniform', type=int, default=2000)
    parser.add_argument('--near', type=int, default=2000)
    parser.add_argument('--paths', default=os.path.join(
        REPO, 'src', 'ranger_xarm_teleop_servo', 'test', 'fixtures',
        'session3_contact_paths.jsonl'))
    parser.add_argument('--inflation', type=int, default=300)
    parser.add_argument('--parity', type=int, default=300)
    parser.add_argument('--tol', type=float, default=0.0005)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reference', choices=('true', 'moveit'), default='true',
                        help="geometry MoveIt's checker runs on (see above)")
    args = parser.parse_args()
    if args.tol < STEP:
        raise SystemExit(f'--tol must be at least the bucket ({STEP} m)')
    started = time.time()
    os.makedirs(args.out, exist_ok=True)

    from ament_index_python.packages import get_package_share_directory
    import rclpy
    rclpy.init()
    urdf = open(args.urdf, encoding='utf-8').read() if args.urdf else read_description()
    srdf = args.srdf or os.path.join(get_package_share_directory('ranger_xarm_moveit_config'),
                                     'config', 'ranger_xarm.srdf')
    reference = read_move_group_description() if args.reference == 'moveit' else urdf
    urdf_file = os.path.join(args.out, 'robot_description.urdf')
    with open(urdf_file, 'w', encoding='utf-8') as handle:
        handle.write(urdf)
    reference_file = os.path.join(args.out, 'reference_description.urdf')
    with open(reference_file, 'w', encoding='utf-8') as handle:
        handle.write(reference)
    model = keepout.build(urdf, keepout.make_resolver(get_package_share_directory), BASE, TIP,
                          MARGIN)
    chain = chain_from_urdf(urdf, BASE, TIP)
    joints = model.joint_names
    extra = sorted(model.extra_joints)
    print(f'model: {len(model.pairs)} pairs, {model.sphere_count} spheres, largest sphere '
          f'{model.largest_radius * 1000:.1f} mm; excluded: '
          f'{[(e.moving, e.other, e.reason) for e in model.excluded]}', flush=True)

    exe = ensure_helper(args.helper)
    helpers = [Helper(exe, reference_file, srdf, joints, os.path.join(args.out, f'helper{i}.log'))
               for i in range(args.workers)]
    pool = Pool(helpers)
    try:
        report = run(args, model, chain, urdf, reference, joints, extra, helpers, pool, srdf,
                     started)
    finally:
        for h in helpers:
            h.close()
        rclpy.shutdown()
    with open(os.path.join(args.out, 'compare_keepout_moveit.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(report, handle, indent=1)
    text = format_report(report)
    with open(os.path.join(args.out, 'compare_keepout_moveit.txt'), 'w',
              encoding='utf-8') as handle:
        handle.write(text)
    print(text)
    return 0


def run(args, model, chain, urdf, reference_text, joints, extra, helpers, pool, srdf,
        started) -> dict:
    """Do the comparison; return the report (a dict)."""
    rng = np.random.default_rng(args.seed)
    h0 = helpers[0]
    allowed = h0.allowed()
    # In-process self check: the padded check sees padding, the service's call does not.
    q_ready = [0.0, -0.6, -0.7, 0.0, 1.3, 0.0]
    h0.pad(0.0)
    plain0, padded0 = len(h0.contacts(q_ready, 'plain')), len(h0.contacts(q_ready, 'padded'))
    h0.pad(0.5)
    selfcheck = {'padding_per_link_m': 0.5, 'unpadded_contacts': padded0 + plain0,
                 'contacts_service_call_padded': len(h0.contacts(q_ready, 'plain')),
                 'contacts_padded_check': len(h0.contacts(q_ready, 'padded')),
                 'contacts_unpadded_env': len(h0.contacts(q_ready, 'unpadded'))}
    h0.pad(0.0)
    print('self check:', selfcheck, flush=True)

    samples, sampling = sample_postures(model, chain, args, rng)
    postures = [q for q, _ in samples]
    sources = [s for _, s in samples]
    print(f'{len(postures)} postures ({sampling})', flush=True)
    parity = None
    if args.parity > 0 and args.reference == 'moveit':
        parity = service_parity(postures, joints, extra, h0, args.parity, rng)
        print('parity:', {k: v for k, v in parity.items() if k != 'examples'}, flush=True)

    # Model clearance of every pair at every posture.
    pairs = model.pairs
    reference_shapes = collision_shapes(reference_text)
    kinds = [model.pair_list[i][0] for i in range(len(pairs))]
    cases = []                   # (posture, pair index, d)
    for p, q in enumerate(postures):
        values = model.pair_minima(q, limit=RANGE[1] + 1e-6)
        for i in np.nonzero((values <= RANGE[1]) & (values >= RANGE[0]))[0]:
            cases.append((p, int(i), float(values[i])))
    keys = [pair_key(*pairs[i]) for i in range(len(pairs))]
    disabled = {}
    tests = []                   # (posture, pair, d, bucket, padding, tolerance)
    untestable = 0
    for p, i, d in cases:
        if keys[i] in allowed:
            row = disabled.setdefault(keys[i], [])
            row.append(d)
            continue
        plan = plan_case(d, args.tol)
        if plan is None:
            untestable += 1
            continue
        tests.append((p, i, d) + plan)
    print(f'{len(cases)} (posture, pair) cases in range, {len(tests)} testable, {untestable} '
          f'below the tolerance, {sum(len(v) for v in disabled.values())} SRDF-allowed',
          flush=True)

    # The padded check, one scene padding per bucket.
    by_bucket = {}
    for t in tests:
        by_bucket.setdefault(t[3], {}).setdefault(t[0], []).append(t)

    def check_bucket(helper, item):
        k, group = item
        helper.pad(k * STEP / 2.0)
        found = []
        for p, rows in group.items():
            hit = helper.contacts(postures[p], 'padded')
            found += [t for t in rows if keys[t[1]] in hit]
        return found, len(group)

    result = pool.run(check_bucket, sorted(by_bucket.items()))
    violations = [t for found, _ in result for t in found]
    queries = sum(n for _, n in result)
    print(f'padded checks: {queries} posture queries, {len(by_bucket)} paddings, '
          f'{len(violations)} contacts at p = d - TOL', flush=True)

    # FCL's exact distances: false alarms, and inflation against the exact value.
    used = sorted({t[0] for t in tests} | {p for p, _, _ in cases})

    def distances(helper, chunk):
        return [(p, helper.distances(postures[p])) for p in chunk]
    chunks = [used[i::args.workers * 4] for i in range(args.workers * 4)]
    fcl = {}
    for rows in pool.run(distances, chunks):
        fcl.update(dict(rows))
    print(f'FCL distances at {len(fcl)} postures', flush=True)

    def exact(p, i):
        return fcl[p].get(keys[i], math.inf)       # absent: farther than 0.2 m

    genuine, false_alarms = [], []
    for t in violations:
        p, i, d, k, padding, tol = t
        row = {'posture': p, 'source': sources[p], 'q': [float(v) for v in postures[p]],
               'pair': list(pairs[i]), 'd_model': d, 'padding': padding,
               'tolerance_in_force': tol, 'fcl_distance': exact(p, i),
               'shapes': [reference_shapes.get(pairs[i][0], []),
                          reference_shapes.get(pairs[i][1], [])]}
        (genuine if exact(p, i) <= padding + 1e-6 else false_alarms).append(row)

    def shape_class(i):
        kinds = set(reference_shapes.get(pairs[i][0], []) + reference_shapes.get(pairs[i][1], []))
        return 'both meshes (padding test exact)' if kinds <= {'mesh'} else 'involves a box'
    by_class = {}
    for t in tests:
        by_class.setdefault(shape_class(t[1]), {'pair_checks': 0, 'padded_contacts': 0,
                                                'genuine': 0})['pair_checks'] += 1
    for row_list, field in ((violations, 'padded_contacts'), (genuine, 'genuine')):
        for t in row_list:
            i = pairs.index(tuple(t['pair'])) if isinstance(t, dict) else t[1]
            by_class[shape_class(i)][field] += 1
    # Independent of padding: the model above FCL's exact distance by more than the tolerance.
    above = []
    worst = 0.0
    for p, i, d in cases:
        if keys[i] in allowed or d <= 0.0:
            continue                          # both report contact when penetrating
        gap = d - exact(p, i)
        worst = max(worst, gap)
        if gap > args.tol:
            above.append({'posture': p, 'source': sources[p], 'pair': list(pairs[i]),
                          'd_model': d, 'fcl_distance': exact(p, i)})

    inflation = sweep(args, model, pool, postures, cases, keys, allowed, kinds, pairs, exact,
                      rng)

    by_source = {}
    for s in sources:
        by_source[s.split(':')[0]] = by_source.get(s.split(':')[0], 0) + 1
    tested_pairs = {}
    for t in tests:
        tested_pairs[keys[t[1]]] = tested_pairs.get(keys[t[1]], 0) + 1
    return {
        'reference': args.reference, 'reference_differs_from_description': reference_text != urdf,
        'tolerance_m': args.tol, 'bucket_m': STEP, 'range_m': list(RANGE),
        'base': BASE, 'tip': TIP, 'margin': MARGIN, 'seed': args.seed, 'srdf': srdf,
        'model': {'pairs': len(pairs), 'spheres': model.sphere_count,
                  'largest_sphere_radius_m': model.largest_radius,
                  'fingerprint': model.fingerprint,
                  'excluded': [[e.moving, e.other, e.reason] for e in model.excluded]},
        'self_check': selfcheck, 'service_parity': parity,
        'postures': {'total': len(postures), 'by_source': by_source, **sampling},
        'cases_in_range': len(cases), 'pair_checks': len(tests),
        'not_testable_below_tolerance': untestable,
        'posture_queries': queries, 'paddings': len(by_bucket),
        'srdf_allowed_pairs': {
            'listed': sorted(k for k in keys if k in allowed),
            'cases_in_range': {k: stats(v) for k, v in sorted(disabled.items())},
            'all_in_helper': sorted(allowed)},
        'tested_pairs': tested_pairs,
        'violations': {'padded_contacts_at_d_minus_tol': len(violations),
                       'by_shape_class': by_class,
                       'genuine_fcl_distance_le_padding': len(genuine),
                       'false_alarms_fcl_distance_above_padding': len(false_alarms),
                       'genuine_list': genuine, 'false_alarm_list': false_alarms},
        'fcl_cross_check': {'cases_compared': sum(1 for p, i, d in cases
                                                  if keys[i] not in allowed and d > 0.0),
                            'model_above_fcl_by_more_than_tol': len(above),
                            'largest_model_minus_fcl_m': worst,
                            'above_summary': above_summary(above), 'list': above[:2000]},
        'inflation': inflation,
        'wall_time_s': time.time() - started,
    }


def above_summary(above) -> dict:
    """Summarise the cases where the model is above FCL's distance by more than the tolerance."""
    if not above:
        return {'n': 0}
    by_pair = {}
    for row in above:
        by_pair[pair_key(*row['pair'])] = by_pair.get(pair_key(*row['pair']), 0) + 1
    return {'n': len(above), 'model_clearance_m': stats([r['d_model'] for r in above]),
            'excess_m': stats([r['d_model'] - r['fcl_distance'] for r in above]),
            'by_pair': dict(sorted(by_pair.items(), key=lambda kv: -kv[1]))}


def sweep(args, model, pool, postures, cases, keys, allowed, kinds, pairs, exact, rng) -> dict:
    """Bisect-free inflation estimate: the first padding bucket at which MoveIt sees contact."""
    pick = [c for c in cases if 0.0 <= c[2] <= 0.06 and keys[c[1]] not in allowed]
    if len(pick) > args.inflation:
        pick = [pick[j] for j in rng.choice(len(pick), size=args.inflation, replace=False)]
    top = int(round(SWEEP_TOP / STEP))
    by_posture = {}
    for c in pick:
        by_posture.setdefault(c[0], []).append(c)

    def contact_at(helper, k):
        helper.pad(k * STEP / 2.0)
        hit = {p: helper.contacts(postures[p], 'padded') for p in by_posture}
        return k, hit
    first, monotonic_breaks = {}, 0
    seen = {}
    for k, hit in pool.run(contact_at, range(0, top + 1)):
        for p, rows in by_posture.items():
            for c in rows:
                if keys[c[1]] in hit[p]:
                    seen.setdefault((c[0], c[1]), []).append(k)
    for c in pick:
        ks = seen.get((c[0], c[1]), [])
        if ks:
            first[(c[0], c[1])] = ks[0]
            if ks != list(range(ks[0], top + 1)):
                monotonic_breaks += 1
    rows = []
    for p, i, d in pick:
        k = first.get((p, i))
        rows.append({'pair': list(pairs[i]), 'kind': kinds[i], 'd_model': d,
                     'd_moveit_upper_bound': None if k is None else k * STEP,
                     'fcl_distance': exact(p, i)})
    groups = {}
    for r, (p, i, d) in zip(rows, pick):
        if r['d_moveit_upper_bound'] is None:
            groups.setdefault('censored', []).append(r)
            continue
        gap = r['d_moveit_upper_bound'] - d
        for name in ('all', f'{r["kind"]}-pairs', 'moving link ' + r['pair'][0]):
            groups.setdefault(name, []).append(gap)
    gap_fcl = {}
    for r in rows:
        if math.isfinite(r['fcl_distance']):
            for name in ('all', f'{r["kind"]}-pairs', 'moving link ' + r['pair'][0]):
                gap_fcl.setdefault(name, []).append(r['fcl_distance'] - r['d_model'])
    return {
        'cases_sampled': len(pick), 'cases_available': len(
            [c for c in cases if 0.0 <= c[2] <= 0.06 and keys[c[1]] not in allowed]),
        'censored_no_contact_by_0.10': len(groups.get('censored', [])),
        'non_monotonic_in_padding': monotonic_breaks,
        'moveit_minus_model_m': {k: stats(v) for k, v in sorted(groups.items())
                                 if k != 'censored'},
        'fcl_minus_model_m': {k: stats(v) for k, v in sorted(gap_fcl.items())},
        'cases': rows}


def format_report(r: dict) -> str:
    """Return the text summary of a report."""
    mm = lambda v: f'{v * 1000:.1f} mm'  # noqa: E731
    v = r['violations']
    lines = [
        'INV-4: model clearance against MoveIt (padded self-collision check, FCL)',
        f'  MoveIt geometry: {r["reference"]} (differs from /robot_description: '
        f'{r["reference_differs_from_description"]})',
        f'  tolerance {mm(r["tolerance_m"])} (bucket {mm(r["bucket_m"])}; the tolerance in force '
        f'per case is d - p\', in (0, {mm(r["tolerance_m"])}]), model clearance range tested '
        f'{r["range_m"]} m',
        f'  model: {r["model"]["pairs"]} pairs, {r["model"]["spheres"]} spheres, largest sphere '
        f'radius {mm(r["model"]["largest_sphere_radius_m"])}',
        f'  self check (0.5 m on every link, ready posture): {r["self_check"]}',
    ]
    if r['service_parity']:
        p = r['service_parity']
        lines.append(f'  parity with /check_state_validity at zero padding: {p["postures"]} '
                     f'postures, {p["identical_contact_sets"]} identical contact sets, '
                     f'{p["different"]} different ({p["with_contacts"]} postures had contacts)')
    lines += [
        f'  postures: {r["postures"]["total"]} {r["postures"]["by_source"]}',
        f'  (posture, pair) cases with model clearance in range: {r["cases_in_range"]}',
        f'  pair checks (testable by padding): {r["pair_checks"]} in {r["posture_queries"]} '
        f'posture queries at {r["paddings"]} paddings; not testable (d < tolerance): '
        f'{r["not_testable_below_tolerance"]}',
        f'  VIOLATIONS (contact at p = d - TOL): {v["padded_contacts_at_d_minus_tol"]}',
        f'    of which FCL exact distance <= padding (genuine): '
        f'{v["genuine_fcl_distance_le_padding"]}',
        f'    FCL exact distance above the padding (box false alarms): '
        f'{v["false_alarms_fcl_distance_above_padding"]}',
        f'    by shape class: {v["by_shape_class"]}',
        f'  FCL cross-check, independent of padding: {r["fcl_cross_check"]["cases_compared"]} '
        f'cases; model above FCL by more than the tolerance: '
        f'{r["fcl_cross_check"]["model_above_fcl_by_more_than_tol"]}; largest model - FCL '
        f'{mm(r["fcl_cross_check"]["largest_model_minus_fcl_m"])}',
        f'    summary of those: {r["fcl_cross_check"]["above_summary"]}',
        f'  pairs the SRDF allows to collide (never checked by MoveIt, excluded): '
        f'{len(r["srdf_allowed_pairs"]["listed"])} of the model\'s pairs',
    ]
    for key, s in r['srdf_allowed_pairs']['cases_in_range'].items():
        lines.append(f'    {key}: {s["n"]} cases in range, model clearance {mm(s["min"])} .. '
                     f'{mm(s["max"])}')
    for row in (v['genuine_list'] + v['false_alarm_list'])[:30]:
        lines.append(f'    {row["source"]} {row["pair"]} d_model {mm(row["d_model"])} padding '
                     f'{mm(row["padding"])} FCL {mm(row["fcl_distance"])} shapes {row["shapes"]}')
    inf = r['inflation']
    lines.append(f'  inflation: {inf["cases_sampled"]} of {inf["cases_available"]} cases with '
                 f'model clearance in [0, 0.06] m; no contact by 0.10 m: '
                 f'{inf["censored_no_contact_by_0.10"]}; contact not monotonic in padding: '
                 f'{inf["non_monotonic_in_padding"]}')
    lines.append('    MoveIt (first contact padding, an upper bound) minus model:')
    for name, s in inf['moveit_minus_model_m'].items():
        if s['n']:
            lines.append(f'      {name}: n {s["n"]} median {mm(s["median"])} p90 {mm(s["p90"])} '
                         f'max {mm(s["max"])} min {mm(s["min"])}')
    lines.append('    FCL exact distance minus model (same cases):')
    for name, s in inf['fcl_minus_model_m'].items():
        if s['n']:
            lines.append(f'      {name}: n {s["n"]} median {mm(s["median"])} p90 {mm(s["p90"])} '
                         f'max {mm(s["max"])} min {mm(s["min"])}')
    lines.append(f'  wall time {r["wall_time_s"]:.0f} s')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    sys.exit(main())
