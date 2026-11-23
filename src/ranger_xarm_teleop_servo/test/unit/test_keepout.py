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
"""The link keep-out model (G-15): built from the description, never more clearance (INV-3, 4)."""

import math

from keepout_brute import assert_search_is_exhaustive
from keepout_robot import box_stl, JOINTS, MESHES, READY, resolve, robot_urdf
import numpy as np
import pytest
from ranger_xarm_teleop_servo import keepout as ko

MARGIN = 0.025


@pytest.fixture(scope='module')
def model():
    return ko.build(robot_urdf(), resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN)


@pytest.fixture(scope='module')
def close_base():
    """Return the test robot on a mesh base close under its upper arm (l2) at every posture."""
    urdf = robot_urdf(base_mesh_top=0.15)
    return urdf, ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN)


@pytest.fixture(scope='module')
def extra(model):
    return model.extra_values({'finger_left_joint': 0.02})


def _random_on_triangles(tris, count, rng):
    pick = rng.integers(0, len(tris), count)
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    t = tris[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


# --- the sphere cover ------------------------------------------------------------------------

@pytest.mark.parametrize('cell', [0.01, 0.02, 0.04])
def test_sphere_cover_contains_every_point_of_every_triangle(cell):
    rng = np.random.default_rng(3)
    shapes = [ko.read_stl(box_stl((0.1, 0.03, 0.2))), ko._cylinder_triangles(0.05, 0.3),
              ko._box_triangles(np.array([0.01, 0.2, 0.05])),
              rng.normal(0.0, 0.1, (40, 3, 3))]                 # arbitrary triangles
    for tris in shapes:
        centres, radii = ko.sphere_cover(tris, cell, 0.002)
        points = np.concatenate([_random_on_triangles(tris, 5000, rng),
                                 tris.reshape(-1, 3)])
        for chunk in np.array_split(points, max(1, len(points) // 1000)):
            gap = np.linalg.norm(chunk[:, None] - centres[None], axis=2) - radii[None]
            assert np.all(gap.min(axis=1) <= 1e-12)
        assert radii.max() <= cell * math.sqrt(3.0) / 2.0 + 0.002 + 1e-12


def test_circumscribed_prism_contains_the_cylinder():
    tris = ko._cylinder_triangles(0.05, 0.3)
    # Every point of the cylinder's side is inside the prism (its faces are tangent).
    radial = np.linalg.norm(tris.reshape(-1, 3)[:, :2], axis=1)
    apothem = radial.max() * math.cos(math.pi / ko.CYLINDER_SIDES)
    assert apothem >= 0.05 - 1e-12


def test_stl_binary_and_ascii_read_the_same():
    binary = ko.read_stl(box_stl((0.1, 0.2, 0.3)))
    text = ko.read_stl(box_stl((0.1, 0.2, 0.3), binary=False))
    assert np.allclose(binary, text, atol=1e-6)
    with pytest.raises(ko.ModelError):
        ko.read_stl(b'not a mesh at all')


# --- what the model is built from ------------------------------------------------------------

def test_fixed_and_moving_bodies_come_from_the_tree(model):
    assert {b.name for b in model.bodies} == {'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'camera',
                                              'gripper_base', 'finger_left', 'finger_right'}
    assert set(model.units.names) == {'base_link', 'pedestal', 'screen', 'mast', 'sensor',
                                      'xarm_link_base'}
    assert model.not_covered == ['wheel']           # behind a joint that is not fixed
    assert model.joint_names == JOINTS
    assert model.extra_joints == {'finger_left_joint': ('finger_left_joint', 1.0, 0.0),
                                  'finger_right_joint': ('finger_left_joint', 1.0, 0.0)}


def test_pairs_every_moving_body_with_every_fixed_one_and_the_tool_with_the_upper_arm(model):
    pairs = set(model.pairs)
    for body in ('l2', 'l3', 'l4', 'l5', 'l6', 'camera', 'gripper_base', 'finger_left'):
        for fixed in ('base_link', 'pedestal', 'screen', 'mast', 'sensor', 'xarm_link_base'):
            assert (body, fixed) in pairs
    tool = {'l6', 'camera', 'gripper_base', 'finger_left', 'finger_right'}
    for body in tool:
        for arm in ('l1', 'l2', 'l3'):
            assert (body, arm) in pairs
        for arm in ('l4', 'l5'):
            assert (body, arm) not in pairs
    assert ('l1', 'xarm_link_base') not in pairs
    assert [(e.moving, e.other, e.reason) for e in model.excluded] == [
        ('l1', 'xarm_link_base', 'adjacent')]


def test_a_pair_always_inside_the_margin_is_excluded_and_reported():
    collar = ('<link name="collar"><collision><origin xyz="0 0 0.20"/><geometry>'
              '<cylinder radius="0.10" length="0.02"/></geometry></collision></link>'
              '<joint name="base_to_collar" type="fixed"><parent link="xarm_link_base"/>'
              '<child link="collar"/></joint></robot>')
    urdf = robot_urdf().replace('</robot>', collar)
    model = ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN)
    near = [e for e in model.excluded if e.reason == 'structurally near']
    assert ('l1', 'collar') in [(e.moving, e.other) for e in near]
    assert all(e.other == 'collar' and e.low <= e.high < MARGIN for e in near)
    assert all((e.moving, e.other) not in model.pairs for e in near)


def test_a_left_out_pair_says_whether_a_smaller_margin_keeps_it():
    keep = ko.Excluded('xarm_link2', 'xarm_link_base', ko.NEAR, -0.007, 0.048)
    assert ko.kept_by_smaller_margin(keep)
    assert ko.excluded_text(keep, 0.05) == (
        'xarm_link2 - xarm_link_base left out: structurally near (clearance -7.0 to 48.0 mm '
        'over the sampled postures, below the margin of 50.0 mm in every one); a smaller '
        'margin would keep the pair')
    touching = ko.Excluded('l1', 'collar', ko.NEAR, -0.0257, -0.0257)
    assert not ko.kept_by_smaller_margin(touching)
    assert ko.excluded_text(touching, 0.025).endswith(
        '; no margin keeps it (in contact at every sampled posture)')
    adjacent = ko.Excluded('l1', 'xarm_link_base', 'adjacent')
    assert not ko.kept_by_smaller_margin(adjacent)
    assert ko.excluded_text(adjacent, 0.025) == 'l1 - xarm_link_base left out: adjacent'


def test_measured_values_place_the_fingers_and_a_missing_one_is_refused(model):
    with pytest.raises(ko.ModelError, match='finger_left_joint'):
        model.extra_values({})
    open_ = model.extra_values({'finger_left_joint': 0.04})
    assert open_ == {'finger_left_joint': 0.04, 'finger_right_joint': 0.04}
    shut = model.extra_values({'finger_left_joint': 0.0})
    names = [b.name for b in model.bodies]
    i = names.index('finger_right')
    moved = model.world_spheres(READY, open_)[i] - model.world_spheres(READY, shut)[i]
    assert np.allclose(np.linalg.norm(moved, axis=1), 0.04, atol=1e-9)


# --- INV-3: derived from the description, a different one is told apart ---------------------

def test_fingerprint_follows_the_geometry_and_ignores_the_text_around_it():
    def fp(urdf, resolver=resolve):
        return ko.fingerprint(urdf, resolver, 'xarm_link_base', 'xarm_link_tcp', MARGIN)
    base = fp(robot_urdf())
    assert fp(robot_urdf().replace('<robot name="test_robot">',
                                   '<robot name="test_robot"><!-- a comment -->')) == base
    assert fp(robot_urdf(screen_x=-0.32)) != base
    assert fp(robot_urdf(screen_height=0.30)) != base
    assert fp(robot_urdf(extra_tool=True)) != base
    assert fp(robot_urdf(drop_screen=True)) != base
    changed = dict(MESHES)
    changed['package://test_robot/meshes/sensor.stl'] = box_stl((0.03, 0.04, 0.03))
    assert fp(robot_urdf(), lambda name: changed[name]) != base


def test_fingerprint_is_the_built_models_at_its_cell_and_any_margin():
    """The node compares a later description at its own cell; the margin is not geometry."""
    urdf = robot_urdf()
    for cell in (ko.DEFAULT_CELL, 0.03):
        built = ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN, cell=cell,
                         near_samples=0).fingerprint
        assert ko.fingerprint(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN,
                              cell=cell) == built
        assert ko.fingerprint(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', 0.04,
                              cell=cell) == built
    assert ko.fingerprint(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN,
                          cell=0.03) != ko.fingerprint(urdf, resolve, 'xarm_link_base',
                                                       'xarm_link_tcp', MARGIN)


def test_the_model_follows_a_moved_box_and_an_added_tool(model, extra):
    q = [0.0, -1.0, -0.6, 0.0, 1.2, 0.0]
    before = model.pair_clearances(q, extra)[('l2', 'screen')]
    moved = ko.build(robot_urdf(screen_x=-0.27), resolve, 'xarm_link_base', 'xarm_link_tcp',
                     MARGIN)
    after = moved.pair_clearances(q, moved.extra_values({'finger_left_joint': 0.02}))
    assert after[('l2', 'screen')] == pytest.approx(before - 0.03, abs=0.021)
    assert after[('l2', 'screen')] < before - 0.01
    tool = ko.build(robot_urdf(extra_tool=True), resolve, 'xarm_link_base', 'xarm_link_tcp',
                    MARGIN)
    assert ('probe', 'screen') in tool.pairs and ('probe', 'l2') in tool.pairs


@pytest.mark.parametrize('change, message', [
    (lambda u: u.replace('meshes/sensor.stl', 'meshes/missing.stl'), 'cannot be read'),
    (lambda u: u.replace('meshes/sensor.stl', 'meshes/sensor.dae'), 'only STL'),
    (lambda u: u.replace('<cylinder radius="0.02" length="1.0"/>',
                         '<capsule radius="0.02" length="1.0"/>'), 'not supported'),
])
def test_a_description_the_model_cannot_follow_is_refused(change, message):
    with pytest.raises(ko.ModelError, match=message):
        ko.build(change(robot_urdf()), resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN)


def test_an_unreadable_mesh_file_is_refused():
    def broken(name):
        return b'\0' * 100 if name.endswith('sensor.stl') else MESHES[name]
    with pytest.raises(ko.ModelError, match='sensor.stl'):
        ko.build(robot_urdf(), broken, 'xarm_link_base', 'xarm_link_tcp', MARGIN)


def test_resolver_reads_package_file_and_plain_paths(tmp_path):
    (tmp_path / 'pkg' / 'meshes').mkdir(parents=True)
    (tmp_path / 'pkg' / 'meshes' / 'a.stl').write_bytes(b'abc')
    read = ko.make_resolver(lambda package: str(tmp_path / package))
    assert read('package://pkg/meshes/a.stl') == b'abc'
    assert read(f'file://{tmp_path}/pkg/meshes/a.stl') == b'abc'
    assert read(f'{tmp_path}/pkg/meshes/a.stl') == b'abc'


# --- finer cells where the model, not the geometry, holds a pair ------------------------------

def _largest(model, name):
    if name in [b.name for b in model.bodies]:
        return float(next(b for b in model.bodies if b.name == name).radii.max())
    units = model.units
    f = units.names.index(name)
    return max(float(units.mem_rad[units.mem_start[u]:units.mem_start[u] + units.mem_count[u]]
                     .max()) for u in np.nonzero((units.link == f) & (units.mem_count > 0))[0])


def test_a_pair_the_model_holds_gets_finer_cells_on_its_covered_sides(close_base, model):
    urdf, close = close_base
    assert model.refined == [] and model.fine_cell is None      # nothing held: nothing finer
    assert close.refined_by == [('l2', 'xarm_link_base')]
    assert close.refined == ['l2', 'xarm_link_base']           # both sides are covers
    assert close.fine_cell == pytest.approx(ko.FINE_CELL_RATIO * ko.DEFAULT_CELL)
    fine = close.fine_cell * math.sqrt(3.0) / 2.0 + ko.DEFAULT_SPACING
    assert _largest(close, 'l2') <= fine + 1e-12
    assert _largest(close, 'xarm_link_base') <= fine + 1e-12
    for other in ('l1', 'l3', 'l4'):                           # the rest keeps the cell
        assert _largest(close, other) == _largest(model, other) > fine
    coarse = ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN,
                      fine_cell_ratio=None)
    assert coarse.refined == [] and coarse.sphere_count < close.sphere_count
    assert set(coarse.pairs) <= set(close.pairs)
    # It reads the pair closer to its geometry: higher over sampled postures.
    rng = np.random.default_rng(3)
    extra = close.extra_values({'finger_left_joint': 0.02})
    i = close.pairs.index(('l2', 'xarm_link_base'))
    j = coarse.pairs.index(('l2', 'xarm_link_base'))
    gain = [close.pair_minima(q, extra)[i] - coarse.pair_minima(q, extra)[j]
            for q in rng.uniform(-1.5, 1.5, (40, 6))]
    assert np.median(gain) > 0.005


def test_the_choice_follows_the_geometry_not_the_names(close_base):
    urdf, close = close_base
    renamed = ko.build(urdf.replace('"l2"', '"upper_arm"').replace('xarm_link_base',
                                                                   'arm_base'),
                       resolve, 'arm_base', 'xarm_link_tcp', MARGIN)
    assert renamed.refined_by == [('upper_arm', 'arm_base')]
    assert renamed.refined == ['upper_arm', 'arm_base']
    a = next(b for b in close.bodies if b.name == 'l2')
    b = next(b for b in renamed.bodies if b.name == 'upper_arm')
    assert np.array_equal(a.centres, b.centres) and np.array_equal(a.radii, b.radii)


def test_the_finer_cover_is_in_the_fingerprint_and_the_margin_is_not(close_base):
    urdf, close = close_base
    args = (urdf, resolve, 'xarm_link_base', 'xarm_link_tcp')
    assert ko.fingerprint(*args, MARGIN) == close.fingerprint
    assert ko.fingerprint(*args, 0.04) == close.fingerprint
    assert ko.build(*args, MARGIN, fine_cell_ratio=None, near_samples=0).fingerprint \
        != close.fingerprint


def test_a_pair_still_held_at_work_is_reported_with_the_margin_that_frees_it():
    urdf = robot_urdf(base_mesh_top=0.16)
    assert ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN).held == []
    model = ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp', 0.04)
    assert [(h.moving, h.other) for h in model.held] == [('l2', 'xarm_link_base')]
    held = model.held[0]
    assert ko.HELD_SHARE < held.share <= 1.0
    assert held.low < held.free_up_to < 0.04 < held.high
    assert ('l2', 'xarm_link_base') in model.pairs                 # kept, not left out
    text = ko.held_text(held, 0.04)
    assert text.startswith('l2 - xarm_link_base is inside the margin of 40.0 mm at ')
    assert f'up to a margin of {held.free_up_to * 1000.0:.1f} mm' in text
    freed = ko.build(urdf, resolve, 'xarm_link_base', 'xarm_link_tcp',
                     held.free_up_to - 0.001)
    assert freed.held == []


# --- INV-4 within the model: never more clearance than the geometry ---------------------------

def _true_upper_bounds(model, urdf, q, extra, rng):
    """Densely sampled surface distance per fixed pair (an upper bound of the true one)."""
    import xml.etree.ElementTree as ElementTree
    root = ElementTree.fromstring(urdf)
    links = {link.get('name'): link for link in root.findall('link')}
    frames = model._frames(np.asarray(q, dtype=float), extra)
    out = {}
    for b, body in enumerate(model.bodies):
        tris = []
        for collision in links[body.name].findall('collision'):
            local, _ = ko._shape_triangles(collision.find('geometry')[0], resolve,
                                           __import__('hashlib').sha256())
            tris.append(ko._apply(ko._origin(collision), local))
        pose = model._pose(b, frames)
        points = _random_on_triangles(np.concatenate(tris), 1500, rng)
        points = np.concatenate([points, np.concatenate(tris).reshape(-1, 3)])
        world = points @ pose[:3, :3].T + pose[:3, 3]
        units = np.arange(len(model.units.link))
        for f, name in enumerate(model.units.names):
            sel = units[model.units.link == f]
            d = np.full(len(world), np.inf)
            for u in sel:
                if model.units.mem_count[u]:
                    lo = model.units.mem_start[u]
                    hi = lo + model.units.mem_count[u]
                    dd = (np.linalg.norm(world[:, None] - model.units.mem_pos[lo:hi][None],
                                         axis=2) - model.units.mem_rad[lo:hi][None]).min(axis=1)
                else:
                    dd = model.units.pair_distance(world, np.full(len(world), u))[0]
                d = np.minimum(d, dd)
            out[(body.name, name)] = float(d.min())
    return out


def test_model_clearance_is_never_more_than_the_sampled_surface_distance(model, extra):
    _assert_never_more_than_the_surfaces(model, robot_urdf(), extra)


def test_finer_cells_never_read_more_than_the_surfaces_either(close_base):
    urdf, model = close_base
    assert model.refined                     # the finer covers are what is checked here
    _assert_never_more_than_the_surfaces(model, urdf,
                                         model.extra_values({'finger_left_joint': 0.02}))


def test_a_finer_fixed_mesh_cover_still_holds_every_point_of_the_mesh(close_base):
    """The fixed side of INV-1: the refined base mesh's members contain its triangles."""
    _, model = close_base
    units = model.units
    f = units.names.index('xarm_link_base')
    clusters = np.nonzero((units.link == f) & (units.mem_count > 0))[0]
    members = np.concatenate([units.mem_start[u] + np.arange(units.mem_count[u])
                              for u in clusters])
    tris = ko.read_stl(MESHES['package://test_robot/meshes/base_150.stl'])
    tris = tris + np.array([0.0, 0.0, 0.075])                 # its collision origin
    points = np.concatenate([_random_on_triangles(tris, 5000, np.random.default_rng(2)),
                             tris.reshape(-1, 3)])
    gap = (np.linalg.norm(points[:, None] - units.mem_pos[members][None], axis=2)
           - units.mem_rad[members][None]).min(axis=1)
    assert np.all(gap <= 1e-12)
    assert units.mem_rad[members].max() <= model.fine_cell * math.sqrt(3.0) / 2.0 + 0.002


def _assert_never_more_than_the_surfaces(model, urdf, extra):
    rng = np.random.default_rng(7)
    lower = np.array([-3.0, -2.0, -3.9, -3.0, -1.6, -3.0])
    upper = np.array([3.0, 2.0, 0.1, 3.0, 3.1, 3.0])
    checked = 0
    for _ in range(12):
        q = rng.uniform(lower, upper)
        model_pairs = model.pair_clearances(q, extra)
        for key, sampled in _true_upper_bounds(model, urdf, q, extra, rng).items():
            if key in model_pairs and sampled < 0.3:
                assert model_pairs[key] <= sampled + 1e-9, key
                # ... and here gives away no more than about one largest radius per covered
                # side (plus the sampling of this test). The proven bound is twice that per
                # side (module docstring); this checks the covers are as tight as expected.
                assert model_pairs[key] >= sampled - 2 * model.largest_radius - 0.01, key
                checked += 1
    assert checked > 50


# --- per-cycle evaluation ----------------------------------------------------------------------

def test_clearance_is_the_brute_force_minimum_over_every_sphere(model, extra):
    rng = np.random.default_rng(11)
    for _ in range(30):
        q = np.array(READY) + rng.normal(0.0, 0.5, 6)
        rows = model.evaluate(q, extra)
        brute = min(model.pair_minima(q, extra).min(), math.inf)
        if brute <= model.influence:
            assert rows.exact and rows.clearance == pytest.approx(brute, abs=1e-12)
            minima = model.pair_minima(q, extra)
            assert model.pairs.index(rows.nearest) == int(np.argmin(minima))
        else:
            assert not rows.exact and rows.clearance <= brute + 1e-12


@pytest.mark.parametrize('spread', [0.3, 0.8, None])
@pytest.mark.parametrize('robot', ['default', 'close base'])
def test_the_pruned_search_lists_what_a_brute_force_search_finds(model, extra, close_base,
                                                                 spread, robot):
    """Pruning (bodies, groups, a mesh's clusters) never drops a sphere pair within reach."""
    if robot == 'close base':
        model = close_base[1]
        extra = model.extra_values({'finger_left_joint': 0.02})
    rng = np.random.default_rng(13)
    lower = np.array([-3.0, -2.0, -3.9, -3.0, -1.6, -3.0])
    upper = np.array([3.0, 2.0, 0.1, 3.0, 3.1, 3.0])
    for _ in range(15):
        q = (rng.uniform(lower, upper) if spread is None
             else np.array(READY) + rng.normal(0.0, spread, 6))
        assert_search_is_exhaustive(model, q, extra, (MARGIN, model.influence, 0.2))


def test_row_gradients_are_the_derivative_of_the_distance(model, extra):
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(20):
        q = np.array(READY) + rng.normal(0.0, 0.6, 6)
        rows = model.evaluate(q, extra, per_pair=1)
        minima = model.pair_minima(q, extra)
        for i in range(len(rows.distance)):
            pair = rows.pair[i]
            # The row is the nearest sphere of its pair; the pair minimum is smooth unless
            # another sphere takes over, which a tiny step does not do here.
            if abs(minima[pair] - rows.distance[i]) > 1e-12:
                continue
            step = rng.normal(0.0, 1.0, 6)
            step *= 1e-5 / np.linalg.norm(step)
            ahead = model.pair_minima(q + step, extra)[pair]
            behind = model.pair_minima(q - step, extra)[pair]
            numeric = (ahead - behind) / 2.0
            assert rows.grad[i] @ step == pytest.approx(numeric, abs=2e-9)
            checked += 1
    assert checked > 10


def test_a_description_without_collision_geometry_keeps_nothing_out():
    from arm_urdf import arm_urdf
    model = ko.build(arm_urdf(), resolve, 'xarm_link_base', 'xarm_link_tcp', MARGIN)
    assert not model.pairs and model.sphere_count == 0
    rows = model.evaluate(READY)
    assert len(rows.distance) == 0 and rows.clearance == math.inf
    assert len(model.pair_minima(READY)) == 0


def test_a_non_finite_joint_state_is_refused(model):
    """A NaN gripper value would make its rows vanish: it is refused instead."""
    with pytest.raises(ko.ModelError, match='not finite'):
        model.extra_values({'finger_left_joint': float('nan')})
