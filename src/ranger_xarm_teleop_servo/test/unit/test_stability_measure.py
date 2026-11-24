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
"""The stability figure: centre of pressure, edges, bounds, stop and energy (contract)."""

import itertools
import math

import numpy as np
import pytest

from ranger_xarm_teleop_servo import stability as st

G = st.G
RECT = [np.array([0.25, 0.2, 0.0]), np.array([0.25, -0.2, 0.0]), np.array([-0.25, 0.2, 0.0]),
        np.array([-0.25, -0.2, 0.0])]


def edges():
    return st.support_polygon(RECT)


def bodies():
    """Three bodies, centre of mass (0.02, 0.03, 0.4) above the plane z = 0."""
    p = np.array([[0.0, 0.0, 0.2], [0.1, 0.1, 0.6], [-0.03, 0.06, 1.0]])
    m = np.array([50.0, 10.0, 5.0])
    return p, m


def com(p, m):
    return (m[:, None] * p).sum(axis=0) / m.sum()


def test_a_rectangle_has_four_named_edges_inside_is_positive():
    names = sorted(e.name for e in edges())
    assert names == ['front', 'left', 'rear', 'right']
    m, e = st.nearest(np.array([0.0, 0.15]), edges())
    assert e.name == 'left' and m == pytest.approx(0.05)
    m, e = st.nearest(np.array([0.3, 0.0]), edges())
    assert e.name == 'front' and m == pytest.approx(-0.05)       # past the edge: negative


def test_edge_names_follow_the_outward_normal_in_45_degree_sectors():
    assert st.edge_name(np.array([1.0, 0.0])) == ('front', 1)
    assert st.edge_name(np.array([1.0, 1.0]))[0] == 'front-left'
    assert st.edge_name(np.array([0.0, -1.0]))[0] == 'right'
    assert st.edge_name(np.array([-1.0, -0.1]))[0] == 'rear'


def test_collinear_or_too_few_contacts_have_no_polygon():
    with pytest.raises(ValueError):
        st.support_polygon(RECT[:2])
    with pytest.raises(ValueError):
        st.support_polygon([np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])])


def test_at_rest_on_a_level_floor_the_figure_is_the_distance_under_the_centre_of_mass():
    p, m = bodies()
    w = st.gravity_vectors(3, np.array([0.0, 0.0, -G]))
    fig = st.figure(p, m, w, 0.0, edges())
    c = com(p, m)
    assert fig.cop == pytest.approx(c[:2])
    assert fig.edge.name == 'left' and fig.margin == pytest.approx(0.2 - c[1])


def test_a_sideways_acceleration_moves_the_centre_of_pressure_by_h_a_over_g():
    p, m = bodies()
    a = np.array([0.0, 2.0, 0.0])
    w = st.gravity_vectors(3, np.array([0.0, 0.0, -G])) - a
    c = com(p, m)
    cop = st.centre_of_pressure(p, m, w, 0.0)
    assert cop[1] == pytest.approx(c[1] - 2.0 * c[2] / G)        # accelerating left: load right
    assert cop[0] == pytest.approx(c[0])


def test_a_tilted_floor_moves_the_centre_of_pressure_downhill_by_h_tan_theta():
    p, m = bodies()
    th = math.radians(10.0)
    g = G * np.array([math.sin(th), 0.0, -math.cos(th)])         # downhill toward the front
    cop = st.centre_of_pressure(p, m, st.gravity_vectors(3, g), 0.0)
    c = com(p, m)
    assert cop[0] == pytest.approx(c[0] + c[2] * math.tan(th))


def test_the_imu_transfer_at_rest_is_gravity_and_a_rigid_rotation_is_carried_to_each_body():
    p, _ = bodies()
    r_imu = np.array([-0.3, 0.0, 1.1])
    w = st.imu_transfer(p, r_imu, np.array([0.0, 0.0, G]), np.zeros(3), np.zeros(3))
    assert w == pytest.approx(st.gravity_vectors(3, np.array([0.0, 0.0, -G])))
    # A body rolling about x at alpha: points below the IMU accelerate the other way.
    alpha = np.array([2.0, 0.0, 0.0])
    w = st.imu_transfer(p, r_imu, np.array([0.0, 0.0, G]), np.zeros(3), alpha)
    for i in range(3):
        dr = p[i] - r_imu
        assert w[i] == pytest.approx(np.array([0.0, 0.0, -G]) - np.cross(alpha, dr))


def test_the_rate_of_angular_momentum_moves_the_centre_of_pressure_the_right_way():
    p, m = bodies()
    w = st.gravity_vectors(3, np.array([0.0, 0.0, -G]))
    base = st.centre_of_pressure(p, m, w, 0.0)
    # A positive roll acceleration (about +x) needs the floor's reaction on the +y side.
    shifted = st.centre_of_pressure(p, m, w, 0.0, hdot=np.array([5.0, 0.0, 0.0]))
    assert shifted[1] - base[1] == pytest.approx(5.0 / (m.sum() * G))
    shifted = st.centre_of_pressure(p, m, w, 0.0, hdot=np.array([0.0, 5.0, 0.0]))
    assert shifted[0] - base[0] == pytest.approx(-5.0 / (m.sum() * G))


def test_no_figure_when_nothing_presses_the_robot_onto_the_floor():
    p, m = bodies()
    with pytest.raises(ValueError):
        st.centre_of_pressure(p, m, st.gravity_vectors(3, np.array([0.0, 0.0, 1.0])), 0.0)


def test_the_pessimistic_figure_equals_the_nominal_with_no_bounds():
    p, m = bodies()
    w = st.gravity_vectors(3, np.array([0.0, 0.0, -G]))
    nominal = st.figure(p, m, w, 0.0, edges())
    pess, e = st.pessimistic_margin(p, m, m, m, np.zeros(3), w, 0.0, edges())
    assert pess == pytest.approx(nominal.margin) and e.name == nominal.edge.name


def test_a_centre_of_mass_bound_lowers_a_level_static_figure_by_that_bound():
    p, m = bodies()
    w = st.gravity_vectors(3, np.array([0.0, 0.0, -G]))
    nominal = st.figure(p, m, w, 0.0, edges())
    pess, _ = st.pessimistic_margin(p, m, m, m, np.full(3, 0.03), w, 0.0, edges())
    assert pess == pytest.approx(nominal.margin - 0.03)


def brute_force(p, m, lo, hi, b, w, plane_z, edge_list):
    """Every corner of the mass box; each centre of mass pushed toward the edge (along w)."""
    best = math.inf
    for e in edge_list:
        pp = p.copy()
        for i in range(len(p)):
            # The worst displacement of size b: along (-n, n.w_xy / w_z), normalised.
            slope = float(np.dot(w[i, :2], e.inward) / w[i, 2])
            d = np.array([-e.inward[0], -e.inward[1], slope])
            pp[i] = p[i] + b[i] * d / np.linalg.norm(d)
        for corner in itertools.product(*zip(lo, hi)):
            masses = np.array(corner)
            best = min(best, st.distance(st.centre_of_pressure(pp, masses, w, plane_z), e))
    return best


def test_the_pessimistic_figure_is_the_worst_corner_of_the_bounds():
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = 5
        p = np.column_stack([rng.uniform(-0.15, 0.15, n), rng.uniform(-0.15, 0.15, n),
                             rng.uniform(0.1, 1.2, n)])
        m = rng.uniform(1.0, 30.0, n)
        lo, hi = m * rng.uniform(0.6, 1.0, n), m * rng.uniform(1.0, 1.4, n)
        b = rng.uniform(0.0, 0.05, n)
        g = G * np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), -1.0])
        w = g[None, :] - np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
                                          np.zeros(n)])
        pess, _ = st.pessimistic_margin(p, m, lo, hi, b, w, 0.0, edges())
        assert pess == pytest.approx(brute_force(p, m, lo, hi, b, w, 0.0, edges()), abs=1e-9)
        assert pess <= st.figure(p, m, w, 0.0, edges()).margin + 1e-12


def test_overshoot_and_lean_amplification():
    assert st.overshoot(0.2) == pytest.approx(math.exp(-math.pi * 0.2 / math.sqrt(0.96)))
    assert st.overshoot(1.0) == 0.0 and st.overshoot(2.0) == 0.0
    assert st.lean_amplification(math.inf, 80.0, 0.4) == 1.0
    assert st.lean_amplification(2744.0, 85.0, 0.446) == pytest.approx(
        1.0 / (1.0 - 85.0 * G * 0.446 / 2744.0))
    assert st.lean_amplification(100.0, 85.0, 0.446) == math.inf     # no stiffness left


def test_a_stop_throws_the_load_toward_the_motion_by_h_a_f_over_g():
    p, m = bodies()
    g = np.array([0.0, 0.0, -G])
    w = st.stop_vectors(p, g, np.array([0.5, 0.0]), 0.0, 1.0, 3.0, 1.5, 2.0)
    c = com(p, m)
    cop = st.centre_of_pressure(p, m, w, 0.0)
    assert cop[0] == pytest.approx(c[0] + c[2] * 1.0 * 1.5 / G)
    w = st.stop_vectors(p, g, np.array([0.0, -0.5]), 0.0, 1.0, 3.0, 1.5, 2.0)
    cop = st.centre_of_pressure(p, m, w, 0.0)
    assert cop[1] == pytest.approx(c[1] - c[2] * 1.0 * 2.0 / G)
    # Not moving: the stop is the static figure.
    w = st.stop_vectors(p, g, np.zeros(2), 0.0, 1.0, 3.0, 1.5, 2.0)
    assert st.centre_of_pressure(p, m, w, 0.0) == pytest.approx(c[:2])


def test_tilt_toward_an_edge_is_positive_when_that_edge_is_downhill():
    th = math.radians(12.0)
    g = G * np.array([0.0, math.sin(th), -math.cos(th)])
    assert st.tilt_toward(g, np.array([0.0, 1.0])) == pytest.approx(th)
    assert st.tilt_toward(g, np.array([0.0, -1.0])) == pytest.approx(-th)


def test_the_energy_to_tip_on_a_level_floor_and_downhill():
    e = [x for x in edges() if x.name == 'left'][0]
    foot, h, mass = np.array([0.0, 0.05]), 0.4, 100.0
    d0 = 0.15
    assert st.energy_to_tip(foot, h, mass, e, 0.0) == pytest.approx(
        mass * G * (math.hypot(d0, h) - h))
    th = math.radians(15.0)
    assert st.energy_to_tip(foot, h, mass, e, th) == pytest.approx(
        mass * G * (math.hypot(d0, h) - (d0 * math.sin(th) + h * math.cos(th))))
    assert st.energy_to_tip(foot, h, mass, e, th) < st.energy_to_tip(foot, h, mass, e, 0.0)
    assert st.energy_to_tip(np.array([0.0, 0.25]), h, mass, e, 0.0) == 0.0   # already past


def test_the_energy_reserve_takes_the_edges_the_motion_heads_toward_and_the_kinetic_energy():
    p, m = bodies()
    inertias = np.zeros((3, 3, 3))
    g = np.array([0.0, 0.0, -G])
    still, _ = st.energy_reserve(p, m, inertias, 0.0, edges(), g, np.zeros(2), 0.0)
    moving, e = st.energy_reserve(p, m, inertias, 0.0, edges(), g, np.array([0.0, 0.4]), 0.0)
    assert e.name == 'left'
    left = [x for x in edges() if x.name == 'left'][0]
    c = com(p, m)
    tip = st.energy_to_tip(c[:2], c[2], m.sum(), left, 0.0)
    assert moving == pytest.approx(tip - 0.5 * m.sum() * 0.16)
    assert still <= tip
    names = {x.name for x in st.edges_toward(edges(), np.array([0.3, 0.3]))}
    assert names == {'front', 'left'}


def test_the_pessimistic_figure_with_angular_momentum_is_the_worst_corner():
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = 5
        p = np.column_stack([rng.uniform(-0.15, 0.15, n), rng.uniform(-0.15, 0.15, n),
                             rng.uniform(0.1, 1.2, n)])
        m = rng.uniform(1.0, 30.0, n)
        lo, hi = m * rng.uniform(0.6, 1.0, n), m * rng.uniform(1.0, 1.4, n)
        w = st.gravity_vectors(n, np.array([0.0, 0.0, -G]))
        hdot = np.array([rng.uniform(-30, 30), rng.uniform(-30, 30), 0.0])
        pess, _ = st.pessimistic_margin(p, m, lo, hi, np.zeros(n), w, 0.0, edges(), hdot)
        best = math.inf
        for e in edges():
            for corner in itertools.product(*zip(lo, hi)):
                cop = st.centre_of_pressure(p, np.array(corner), w, 0.0, hdot)
                best = min(best, st.distance(cop, e))
        assert pess == pytest.approx(best, abs=1e-9)


def test_a_turn_seen_by_an_imu_off_the_axis_gives_each_body_its_own_centripetal_load():
    p, m = bodies()
    r_imu = np.array([-0.3, 0.0, 1.1])
    omega = np.array([0.0, 0.0, 1.0])
    g = np.array([0.0, 0.0, -G])
    a_imu = -1.0 * np.array([r_imu[0], r_imu[1], 0.0])          # w^2 r toward the axis
    w = st.imu_transfer(p, r_imu, a_imu - g, omega, np.zeros(3))
    direct = g[None, :] + np.column_stack([p[:, 0], p[:, 1], np.zeros(3)])
    assert w == pytest.approx(direct)
