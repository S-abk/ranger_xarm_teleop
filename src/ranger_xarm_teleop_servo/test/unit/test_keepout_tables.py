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
Link keep-out between the arm's own links (G-15).

The clearance tables, their cache, self pairs.
"""

import math
import os

from keepout_robot import resolve, robot_urdf
import numpy as np
import pytest
from ranger_xarm_teleop_servo import keepout as ko
from ranger_xarm_teleop_servo import keepout_tables as kt
from ranger_xarm_teleop_servo import solver as dls

BASE, TIP = 'xarm_link_base', 'xarm_link_tcp'
CFG = kt.TEST_CONFIG


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp('tables'))
    urdf = robot_urdf()
    plan = kt.make_plan(urdf, resolve, BASE, TIP, CFG)
    assert kt.build_missing(plan, cache, 1, lambda text: None) == len(plan.pairs) + len(plan.one)
    loaded = kt.load(plan, cache)
    assert loaded is not None
    return urdf, plan, cache, loaded


@pytest.fixture(scope='module')
def model(built):
    urdf, _, _, loaded = built
    return ko.build(urdf, resolve, BASE, TIP, 0.025, near_samples=0, fine_cell_ratio=None,
                    self_margin=0.03, tables=loaded)


def _bodies(urdf):
    return ko.build(urdf, resolve, BASE, TIP, 0.025, near_samples=0, fine_cell_ratio=None)


# --- which pairs ----------------------------------------------------------------------------

def test_pairs_two_joints_apart_get_tables_and_one_joint_apart_the_check(built):
    _, plan, _, _ = built
    pairs = {(s.outer, s.inner): s.joints for s in plan.pairs}
    assert pairs == {('l3', 'l1'): (1, 2), ('l4', 'l2'): (2, 3), ('l5', 'l3'): (3, 4),
                     ('l6', 'l4'): (4, 5), ('camera', 'l4'): (4, 5),
                     ('gripper_base', 'l4'): (4, 5), ('finger_left', 'l4'): (4, 5),
                     ('finger_right', 'l4'): (4, 5)}
    # One joint apart, not joined by it: what is fixed to l6, against l5 (l6 - l5 is joined).
    assert {(s.outer, s.inner) for s in plan.one} == {
        ('camera', 'l5'), ('gripper_base', 'l5'), ('finger_left', 'l5'), ('finger_right', 'l5')}
    assert {s.outer: s.drive for s in plan.pairs if s.drive}.keys() == {'finger_left',
                                                                        'finger_right'}
    assert plan.unsupported == []


def test_the_relative_pose_of_every_spec_is_the_kinematics(built):
    urdf, plan, _, _ = built
    m = _bodies(urdf)
    index = {b.name: i for i, b in enumerate(m.bodies)}
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = rng.uniform(-2.0, 2.0, 6)
        drive = rng.uniform(0.0, 0.04)
        extra = {n: mult * drive + off for n, (_, mult, off) in m.extra_joints.items()}
        frames = m._frames(q, extra)
        for s in plan.pairs:
            rel = np.linalg.inv(m._pose(index[s.inner], frames)) @ m._pose(index[s.outer], frames)
            rot, pos = kt._relative(s, s.s2 @ kt.tail_pose(s.tail, drive),
                                    np.array([q[s.joints[0]]]), np.array([q[s.joints[1]]]))
            assert np.allclose(rot[0], rel[:3, :3], atol=1e-12)
            assert np.allclose(pos[0], rel[:3, 3], atol=1e-12)
        for s in plan.one:
            rel = np.linalg.inv(m._pose(index[s.inner], frames)) @ m._pose(index[s.outer], frames)
            pose = (s.s0 @ kt._motion('revolute', s.axis, q[s.joint]) @ s.s2
                    @ kt.tail_pose(s.tail, drive))
            assert np.allclose(pose, rel, atol=1e-12)


# --- never more clearance than the geometry has -----------------------------------------------

def _points(tris, spacing, most=2500):
    points = ko.sample_triangles(tris, spacing)
    if len(points) > most:
        points = points[np.random.default_rng(0).choice(len(points), most, replace=False)]
    return points


def _sampled_distance(a, b):
    """
    Return the least distance between two sample sets: never less than the surfaces' distance.

    Any subset of the surfaces gives an upper bound of their distance, coarse ones included.
    """
    best = math.inf
    for i in range(0, len(b), 400):
        d = np.linalg.norm(a[:, None, :] - b[None, i:i + 400, :], axis=2)
        best = min(best, float(d.min()))
    return best


def test_a_table_never_reads_more_than_the_geometry_between_its_grid_points(built):
    urdf, plan, _, loaded = built
    rng = np.random.default_rng(2)
    for spec in plan.pairs:
        table = loaded.tables[(spec.outer, spec.inner)].table
        inner = _points(spec.tris_inner, 0.01)
        outer = _points(spec.tris_outer, 0.01)
        (la, ha), (lb, hb) = spec.limits
        for _ in range(8):
            qa = rng.uniform(max(la, -3.0), min(ha, 3.0))
            qb = rng.uniform(max(lb, -3.0), min(hb, 3.0))
            drive = rng.uniform(*spec.drive) if spec.drive else 0.0
            rot, pos = kt._relative(spec, spec.s2 @ kt.tail_pose(spec.tail, drive),
                                    np.array([qa]), np.array([qb]))
            truth_or_more = _sampled_distance(inner, outer @ rot[0].T + pos[0])
            assert table.read(qa, qb)[0] <= truth_or_more + 1e-9, (
                spec.outer, spec.inner, qa, qb)


def test_the_stack_reads_what_each_table_reads_and_its_direction_follows_the_read(built):
    _, plan, _, loaded = built
    keys = list(loaded.tables)
    stack = kt.TableStack([loaded.tables[k].table for k in keys],
                          [loaded.tables[k].joints for k in keys])
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = rng.uniform(-3.0, 3.0, 6)
        value, ga, gb = stack.read(q)
        for i, k in enumerate(keys):
            entry = loaded.tables[k]
            ja, jb = entry.joints
            assert value[i] == pytest.approx(entry.table.read(q[ja], q[jb])[0], abs=1e-12)
            da, db = stack.delta[i]
            assert ga[i] == pytest.approx((entry.table.read(q[ja] + da, q[jb])[0]
                                           - entry.table.read(q[ja] - da, q[jb])[0]) / (2 * da),
                                          abs=1e-9)
            assert gb[i] == pytest.approx((entry.table.read(q[ja], q[jb] + db)[0]
                                           - entry.table.read(q[ja], q[jb] - db)[0]) / (2 * db),
                                          abs=1e-9)
    sub = stack.select([2, 0])
    assert np.allclose(sub.read(q)[0], stack.read(q)[0][[2, 0]])


def test_the_one_joint_check_reads_no_more_than_the_geometry(built):
    _, plan, _, loaded = built
    rng = np.random.default_rng(4)
    for spec in plan.one:
        entry = loaded.one[(spec.outer, spec.inner)]
        assert entry.high >= entry.low and entry.error > 0.0
        inner = _points(spec.tris_inner, 0.01)
        outer = _points(spec.tris_outer, 0.01)
        for q in rng.uniform(*spec.limits, 6):
            drive = rng.uniform(*spec.drive) if spec.drive else 0.0
            pose = (spec.s0 @ kt._motion('revolute', spec.axis, q) @ spec.s2
                    @ kt.tail_pose(spec.tail, drive))
            assert entry.low <= _sampled_distance(inner, outer @ pose[:3, :3].T + pose[:3, 3])


# --- the cache -----------------------------------------------------------------------------

def test_a_second_build_finds_everything_and_leaves_no_partial_file(built):
    _, plan, cache, _ = built
    assert kt.build_missing(plan, cache, 1, lambda text: None) == 0
    assert not [f for f in os.listdir(cache) if f.endswith('.tmp')]


def test_a_damaged_entry_is_rebuilt_never_used(built, tmp_path):
    urdf, plan, cache, _ = built
    copy = str(tmp_path / 'copy')
    os.makedirs(copy)
    for name in os.listdir(cache):
        if name.endswith('.npz'):
            with open(os.path.join(cache, name), 'rb') as src:
                data = src.read()
            with open(os.path.join(copy, name), 'wb') as dst:
                dst.write(data)
    key = plan.keys[('l6', 'l4')]
    with open(os.path.join(copy, key + '.npz'), 'r+b') as handle:
        handle.seek(100)
        handle.write(b'\0' * 64)
    assert kt.load(plan, copy) is None
    assert kt.missing(plan, copy) == [('l6', 'l4')]
    assert kt.build_missing(plan, copy, 1, lambda text: None) == 1
    assert kt.load(plan, copy) is not None


def test_an_entry_under_another_key_is_not_used(built, tmp_path):
    _, plan, cache, _ = built
    key = plan.keys[('l3', 'l1')]
    other = plan.keys[('l4', 'l2')]
    copy = str(tmp_path)
    with open(os.path.join(cache, key + '.npz'), 'rb') as src:
        data = src.read()
    with open(os.path.join(copy, other + '.npz'), 'wb') as dst:
        dst.write(data)                       # the file's own key no longer matches its name
    assert ('l4', 'l2') in kt.missing(plan, copy)


def test_the_keys_follow_the_geometry_of_the_pair_only(built):
    urdf, plan, _, _ = built
    moved = kt.make_plan(robot_urdf(screen_x=-0.35), resolve, BASE, TIP, CFG)
    assert moved.keys == plan.keys            # the fixed structure is not in a self pair
    tool = robot_urdf().replace('<box size="0.09 0.03 0.03"/>', '<box size="0.09 0.03 0.04"/>')
    changed = kt.make_plan(tool, resolve, BASE, TIP, CFG)
    differ = {k for k in plan.keys if changed.keys[k] != plan.keys[k]}
    assert differ == {('camera', 'l4'), ('camera', 'l5')}
    finer = kt.make_plan(urdf, resolve, BASE, TIP, kt.TableConfig())
    assert not set(finer.keys.values()) & set(plan.keys.values())


# --- the model with the tables ---------------------------------------------------------------

def test_self_pairs_by_the_joints_between_them(model):
    kinds = {}
    for kind, a, b in model.pair_list:
        kinds.setdefault(kind, []).append(
            (model.bodies[a].name, model.units.names[b] if kind == 'fixed'
             else model.bodies[b].name))
    assert len(kinds['table']) == 8
    depth = {b.name: b.depth for b in model.bodies}
    assert all(depth[o] - depth[i] >= 3 for o, i in kinds['tool'])
    assert ('l4', 'l1') in kinds['tool'] and ('gripper_base', 'l3') in kinds['tool']
    one = {(e.moving, e.other): e for e in model.excluded if e.reason == ko.ONE_JOINT}
    assert set(one) == {('camera', 'l5'), ('gripper_base', 'l5'), ('finger_left', 'l5'),
                        ('finger_right', 'l5')}
    assert all(math.isfinite(e.low) and e.self_pair and e.joint == 'xarm_joint6'
               for e in one.values())
    neighbours = {(e.moving, e.other) for e in model.excluded
                  if e.reason == 'adjacent' and e.self_pair}
    assert neighbours == {('l2', 'l1'), ('l3', 'l2'), ('l4', 'l3'), ('l5', 'l4'), ('l6', 'l5')}


def test_each_row_carries_its_pairs_margin_and_table_rows_move_only_their_two_joints(model):
    rng = np.random.default_rng(5)
    seen = set()
    for q in rng.uniform(-2.5, 2.5, (300, 6)):
        rows = model.evaluate(q)
        for i, p in enumerate(rows.pair):
            kind, _, _ = model.pair_list[p]
            seen.add(kind)
            assert rows.margin[i] == (0.025 if kind == 'fixed' else 0.03)
            if kind == 'table':
                row = model._table_pairs.tolist().index(p)
                ja, jb = model._tables.ja[row], model._tables.jb[row]
                others = [j for j in range(6) if j not in (ja, jb)]
                assert np.all(rows.grad[i, others] == 0.0)
        assert rows.self_active == any(model.pair_list[p][0] == 'tool' for p in rows.pair)
        if rows.exact:
            assert rows.nearest_margin == (0.025 if rows.nearest[1] in model.units.names
                                           else 0.03)
    assert seen == {'fixed', 'tool', 'table'}


def test_motion_apart_is_never_in_a_self_row(model):
    arm = dls.arm_from_urdf(robot_urdf(), BASE, TIP)
    rng = np.random.default_rng(6)
    for q in rng.uniform(-2.5, 2.5, (200, 6)):
        rows = model.evaluate(q)
        a, lower = dls._rows_normalised(
            arm, rows, model, dls.SolverParams(linear_speed_cap=0.25, angular_speed_cap=1.0))
        assert np.all(lower <= 0.0)          # zero joint motion always satisfies every row
        target = rows.margin + dls.KEEPOUT_TARGET_INSIDE
        inside = rows.distance < target
        assert np.all(lower[inside] == 0.0)  # inside the margin: only approach is limited


def test_pair_minima_and_least_excess_take_each_pairs_own_margin(model):
    q = np.array([0.0, -0.6, -0.7, 0.0, 1.3, 0.0])
    values = model.pair_minima(q, limit=0.2)
    margins = np.array([model.margin_of(i) for i in range(len(model.pair_list))])
    assert model.least_excess(q, within=0.2 - 0.03) == pytest.approx(
        float((values - margins).min()))


def test_nearest_pairs_lists_the_least_clearances_nearest_first(model):
    rng = np.random.default_rng(11)
    for q in rng.uniform(-2.0, 2.0, (50, 6)):
        values = model.pair_minima(q, limit=0.2)
        nearest = model.nearest_pairs(q, count=3, within=0.2)
        expected = sorted(v for v in values if v < 0.2)[:3]
        assert [c for c, _ in nearest] == pytest.approx(expected)
        assert all(values[model.pairs.index(p)] == pytest.approx(c) for c, p in nearest)
    q = np.array([0.0, -0.6, -0.7, 0.0, 1.3, 0.0])
    assert model.nearest_pairs(q, count=3, within=0.0) == []
    assert len(model.nearest_pairs(q, count=5, within=math.inf)) == 5


def test_a_self_margin_beyond_what_the_tables_hold_refuses_the_model(built):
    urdf, _, _, loaded = built
    with pytest.raises(ko.ModelError, match='beyond what the clearance tables hold'):
        ko.build(urdf, resolve, BASE, TIP, 0.025, near_samples=0, fine_cell_ratio=None,
                 self_margin=0.07, tables=loaded)


def test_without_tables_the_pairs_two_joints_apart_are_left_out_and_listed():
    m = _bodies(robot_urdf())
    kinds = [k for k, _, _ in m.pair_list]
    assert 'table' not in kinds and 'tool' in kinds
    assert len([e for e in m.excluded if e.reason == ko.NO_TABLE]) == 8
    assert all(not math.isfinite(e.low) for e in m.excluded if e.reason == ko.ONE_JOINT)


# --- the one-joint rule --------------------------------------------------------------------

def _one(low, high, error=0.005):
    return ko.Excluded('tool_part', 'l5', ko.ONE_JOINT, low, high, True, error, 'xarm_joint6')


@pytest.mark.parametrize('item, refused, words', [
    (_one(0.040, 0.050), False, None),
    (_one(0.015, 0.016), False, 'constant over xarm_joint6: no motion changes it'),
    (_one(0.015, 0.030), True, 'xarm_joint6 changes it'),
    (_one(0.0, 0.0005), True, 'the two can touch'),
    (_one(-0.002, 0.03), True, 'the two can touch'),
    (ko.Excluded('tool_part', 'l5', ko.ONE_JOINT, self_pair=True, joint='xarm_joint6'), True,
     'not checked'),
])
def test_one_joint_apart_is_refused_only_when_the_joint_changes_it_below_the_margin(
        item, refused, words):
    assert ko.one_joint_refused(item, 0.025) is refused
    text = ko.excluded_text(item, 0.025)
    assert text.startswith('tool_part - l5 left out: one joint apart (xarm_joint6')
    if words:
        assert words in text
