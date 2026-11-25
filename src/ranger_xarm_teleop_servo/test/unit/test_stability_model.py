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
"""The mass model from the description, and the mass file that corrects it (INV-5)."""

import copy
import math

import numpy as np
import pytest
from ranger_xarm_teleop_servo import stability_mass as sm
from ranger_xarm_teleop_servo.stability_model import DescriptionError, RobotModel
from stability_urdf import CHASSIS_KG, LINK2_KG, make_urdf, MASS_FILE, TOTAL_KG, WHEELS


def model(**kw):
    return RobotModel(make_urdf(**kw), 'base_link', WHEELS)


def test_every_inertial_of_the_description_is_a_body():
    m = model()
    assert sum(b.mass for b in m.bodies) == pytest.approx(TOTAL_KG)
    posed = m.pose({})
    assert posed.masses.sum() == pytest.approx(TOTAL_KG)
    chassis = posed.positions[posed.names.index('chassis')]
    assert chassis == pytest.approx([0.0, 0.0, 0.1])               # in base_link, not the root


def test_the_joints_the_figure_needs_exclude_mimic_spin_and_springs():
    assert model().required_joints() == sorted(
        ['arm_joint1', 'arm_joint2'] + [f'{c}_steer_joint' for c in
                                        ('front_left', 'front_right', 'rear_left', 'rear_right')])


def test_bodies_follow_the_measured_joints_and_a_mimic_follows_its_source():
    m = model()
    posed = m.pose({'arm_joint1': math.pi / 2, 'arm_joint2': 0.0})
    link2 = posed.positions[posed.names.index('arm_link2')]
    assert link2 == pytest.approx([0.0, 0.2, 0.6], abs=1e-9)
    finger0 = m.pose({}).positions[posed.names.index('finger')]
    finger1 = m.pose({'arm_joint2': 0.0001}).positions[posed.names.index('finger')]
    assert finger1[1] - finger0[1] == pytest.approx(0.01 * 0.0001, abs=1e-9)


def test_contacts_are_under_the_wheel_centres_and_move_with_an_offset_steering_axis():
    posed = model().pose({})
    assert posed.contacts['front_left_wheel'] == pytest.approx([0.25, 0.2, -0.3])
    assert posed.plane_z == pytest.approx(-0.3)
    centred = model().pose({'front_left_steer_joint': math.pi / 2})
    assert centred.contacts['front_left_wheel'] == pytest.approx([0.25, 0.2, -0.3])
    offset = model(steer_offset=0.05).pose({'front_left_steer_joint': math.pi / 2})
    assert offset.contacts['front_left_wheel'] == pytest.approx([0.25, 0.25, -0.3])


def test_a_cylinder_wheel_gives_its_radius_and_a_box_wheel_is_refused():
    assert model(wheel_collision='cylinder').wheel_radius['rear_left_wheel'] == 0.1
    with pytest.raises(DescriptionError, match='radius'):
        model(wheel_collision='box')


def test_a_description_without_the_wheels_or_the_base_frame_is_refused():
    with pytest.raises(DescriptionError, match='wheel link'):
        RobotModel(make_urdf(), 'base_link', ('front_left_wheel', 'missing_wheel',
                                              'rear_left_wheel'))
    with pytest.raises(DescriptionError, match='base frame'):
        RobotModel(make_urdf(), 'no_such_frame', WHEELS)
    with pytest.raises(DescriptionError, match='three'):
        RobotModel(make_urdf(), 'base_link', WHEELS[:2])
    with pytest.raises(DescriptionError, match='XML'):
        RobotModel('<robot', 'base_link', WHEELS)


def test_springs_and_their_stiffness_in_roll_and_pitch():
    m = model()
    assert len(m.springs) == 4
    s = m.springs['front_left_suspension_joint']
    assert (s.stiffness, s.reference, s.damping, s.lower, s.upper) == (
        20000.0, -0.01, 500.0, -0.015, 0.025)
    roll, pitch = m.suspension_stiffness()
    assert roll == pytest.approx(4 * 20000.0 * 0.2 ** 2)
    assert pitch == pytest.approx(4 * 20000.0 * 0.25 ** 2)
    assert 'front_left_wheel' in m.unsprung_links() and 'chassis' not in m.unsprung_links()
    assert model(springs=False).suspension_stiffness() == (math.inf, math.inf)


# ---------------------------------------------------------------------------- the mass file
def parse(data, simulation=True, **kw):
    return sm.parse(data, model(**kw), simulation)


def test_a_file_that_corrects_nothing_keeps_the_description_and_bounds_every_body():
    mm = parse(MASS_FILE)
    a = sm.as_arrays(mm)
    assert a['mass'].sum() == pytest.approx(TOTAL_KG)
    i = [b.body.name for b in mm.bodies].index('chassis')
    assert (a['mass_low'][i], a['mass_high'][i], a['com_bound'][i]) == (32.0, 48.0, 0.05)
    j = [b.body.name for b in mm.bodies].index('arm_link2')
    assert a['mass_high'][j] == pytest.approx(LINK2_KG * 1.1)
    assert a['com_bound'][j] == 0.02
    assert mm.stiffness == sm.Bound(20000.0, 2000.0, 12000.0)
    assert mm.damping == sm.Bound(500.0, 150.0, 150.0)


def test_a_body_is_corrected_and_one_is_added_to_a_link():
    data = copy.deepcopy(MASS_FILE)
    data['bodies']['chassis'] = {'mass': {'value': 45.0, 'plus': 2.0, 'minus': 2.0},
                                 'com': {'value': [-0.03, 0.0, 0.12], 'bound': 0.01}}
    data['added'] = {'carried_object': {'parent': 'arm_link2',
                                        'mass': {'value': 1.5, 'plus': 0.1, 'minus': 0.1},
                                        'com': {'value': [0.45, 0.0, 0.0], 'bound': 0.01},
                                        'inertia': [0.001, 0.001, 0.001]}}
    mm = parse(data)
    names = [b.body.name for b in mm.bodies]
    assert 'carried_object' in names
    chassis = mm.bodies[names.index('chassis')]
    assert chassis.body.mass == 45.0 and chassis.body.com == pytest.approx([-0.03, 0.0, 0.12])
    m = model()
    posed = m.pose({}, mm.nominal_bodies)
    assert posed.masses.sum() == pytest.approx(TOTAL_KG - CHASSIS_KG + 45.0 + 1.5)
    obj = posed.positions[posed.names.index('carried_object')]
    assert obj == pytest.approx([0.45, 0.0, 0.6])


@pytest.mark.parametrize('change, match', [
    (lambda d: d['bodies'].update({'no_such_link': {'mass': {'plus': 1, 'minus': 1},
                                                    'com': {'bound': 0.1}}}), 'not a body'),
    (lambda d: d.update({'unexpected': 1}), 'unknown key'),
    (lambda d: d['bodies']['chassis']['mass'].pop('plus'), 'missing'),
    (lambda d: d['bodies']['chassis']['com'].pop('bound'), 'missing'),
    (lambda d: d['bodies']['chassis']['mass'].update({'minus': -1.0}), 'below 0'),
    (lambda d: d['bodies']['chassis']['mass'].update({'plus': float('nan')}), 'finite'),
    (lambda d: d['bodies']['chassis']['mass'].update({'minus': 50.0}), 'exceeds'),
    (lambda d: d['bodies']['chassis']['mass'].update({'plus': True}), 'number'),
    (lambda d: d.update({'template': True}), 'template'),
    (lambda d: d.update({'stability_mass_model': 2}), 'format'),
    (lambda d: d.pop('default_uncertainty'), 'missing'),
    (lambda d: d['default_uncertainty'].update({'mass_fraction': 1.0}), 'below 1'),
    (lambda d: d.update({'added': {'chassis': {'parent': 'base_link', 'mass': {
        'value': 1, 'plus': 0, 'minus': 0}, 'com': {'value': [0, 0, 0], 'bound': 0}}}}),
     'exists'),
    (lambda d: d.update({'added': {'thing': {'parent': 'nowhere', 'mass': {
        'value': 1, 'plus': 0, 'minus': 0}, 'com': {'value': [0, 0, 0], 'bound': 0}}}}),
     'parent'),
    (lambda d: d.update({'added': {'thing': {'parent': 'arm_link2', 'mass': {
        'plus': 0, 'minus': 0}, 'com': {'value': [0, 0, 0], 'bound': 0}}}}), 'missing'),
    (lambda d: d['suspension']['stiffness'].update({'minus': 20000.0}), 'exceeds|above 0'),
    (lambda d: d['suspension'].update({'travel': 1}), 'unknown key'),
])
def test_a_wrong_mass_file_is_refused_naming_the_key(change, match):
    data = copy.deepcopy(MASS_FILE)
    change(data)
    with pytest.raises(sm.MassFileError, match=match):
        parse(data)


def test_an_empty_file_and_a_simulation_only_file_outside_simulation_are_refused():
    with pytest.raises(sm.MassFileError, match='empty'):
        parse(None)
    with pytest.raises(sm.MassFileError, match='simulation_only'):
        parse(MASS_FILE, simulation=False)


def test_suspension_bounds_without_springs_are_refused():
    with pytest.raises(sm.MassFileError, match='no springs'):
        parse(MASS_FILE, springs=False)
    data = copy.deepcopy(MASS_FILE)
    data.pop('suspension')
    mm = parse(data, springs=False)
    assert mm.stiffness is None


def test_the_fingerprint_changes_with_every_value_and_only_then():
    base = parse(MASS_FILE).fingerprint
    assert parse(copy.deepcopy(MASS_FILE)).fingerprint == base
    seen = {base}
    for change in (
            lambda d: d['bodies']['chassis']['mass'].update({'plus': 7.0}),
            lambda d: d['bodies']['chassis']['com'].update({'bound': 0.04}),
            lambda d: d['default_uncertainty'].update({'com_m': 0.03}),
            lambda d: d['suspension']['stiffness'].update({'minus': 11000.0}),
            lambda d: d['bodies']['chassis']['mass'].update({'value': 41.0})):
        data = copy.deepcopy(MASS_FILE)
        change(data)
        fp = parse(data).fingerprint
        assert fp not in seen
        seen.add(fp)
    assert len(base) == 64 and all(c in '0123456789abcdef' for c in base)
    # The description is part of the model in force.
    other = sm.parse(MASS_FILE, RobotModel(make_urdf(steer_offset=0.0).replace(
        'mass value="2.0"', 'mass value="2.5"'), 'base_link', WHEELS), True)
    assert other.fingerprint != base
    assert np.isfinite(sm.summary(parse(MASS_FILE))['mass_kg'])


def test_a_continuous_joint_that_carries_mass_is_measured_and_posed():
    urdf = make_urdf().replace('<joint name="arm_joint1" type="revolute">',
                               '<joint name="arm_joint1" type="continuous">')
    m = RobotModel(urdf, 'base_link', WHEELS)
    assert 'arm_joint1' in m.required_joints()
    assert not any(j.endswith('_wheel_joint') for j in m.required_joints())
    posed = m.pose({'arm_joint1': math.pi / 2})
    link2 = posed.positions[posed.names.index('arm_link2')]
    assert link2 == pytest.approx([0.0, 0.2, 0.6], abs=1e-9)


def test_the_fingerprint_covers_the_geometry():
    narrow = make_urdf().replace('0.25 0.2 -0.2', '0.25 0.18 -0.2')
    assert (sm.parse(MASS_FILE, RobotModel(narrow, 'base_link', WHEELS), True).fingerprint
            != parse(MASS_FILE).fingerprint)
