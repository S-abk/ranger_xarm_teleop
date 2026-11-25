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
The stability monitor's mass file: corrections, additions and uncertainties (no ROS).

The description supplies the masses; one file outside the checkout corrects a body, adds one
(a tool, a carried object) and gives every value a bound (INV-5). Nothing has a built-in
default: a value the file does not give comes from the description, and every body gets an
uncertainty, its own or the file's default. Format::

    stability_mass_model: 1
    simulation_only: false            # true: refused outside simulation
    template: false                   # true: always refused (the shipped template)
    default_uncertainty: {mass_fraction: 0.1, com_m: 0.02}
    bodies:                           # by link name of the description
      <link>: {mass: {value: 47.0, plus: 5.0, minus: 5.0}, com: {value: [x, y, z], bound: 0.05}}
    added:                            # fixed to an existing link
      <name>: {parent: <link>, mass: {value: 1.2, plus: 0.1, minus: 0.1},
               com: {value: [x, y, z], bound: 0.01}, inertia: [ixx, iyy, izz]}
    suspension:                       # optional; absent: the description's springs, no bound
      stiffness: {value: 20000.0, plus: 2000.0, minus: 12000.0}
      damping: {value: 570.0, plus: 171.0, minus: 171.0}

Masses in kg, lengths in m (``com`` in the link's own frame), stiffness in N/m, damping in
N s/m; ``plus`` and ``minus`` are absolute amounts. In ``bodies`` the ``value`` keys are
optional (absent: the description's value); the uncertainty keys are required. Anything else
is refused with the key named. The model in force is fingerprinted (``fingerprint``).
"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from ranger_xarm_teleop_servo.stability_model import Body, RobotModel

FORMAT = 1


class MassFileError(ValueError):
    """The mass file is refused; the message names the key."""


@dataclass(frozen=True)
class Bound:
    """A value with an absolute bound below and above."""

    value: float
    plus: float
    minus: float

    @property
    def low(self) -> float:
        return self.value - self.minus

    @property
    def high(self) -> float:
        return self.value + self.plus


@dataclass(frozen=True)
class UncertainBody:
    """A body in force: the body (nominal) with its mass bound and centre-of-mass bound (m)."""

    body: Body
    mass: Bound
    com_bound: float


@dataclass(frozen=True)
class MassModel:
    """The model in force: every body with its bounds, the springs' bounds, the fingerprint."""

    bodies: Tuple[UncertainBody, ...]
    stiffness: Optional[Bound]             # None: no springs (rigid)
    damping: Optional[Bound]
    fingerprint: str
    simulation_only: bool

    @property
    def nominal_bodies(self) -> List[Body]:
        """Return every body in force at its nominal values (for ``RobotModel.pose``)."""
        return [b.body for b in self.bodies]


def _number(value, where: str, minimum: Optional[float] = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassFileError(f'{where}: a number is required (got {value!r})')
    value = float(value)
    if not math.isfinite(value):
        raise MassFileError(f'{where}: must be finite')
    if minimum is not None and value < minimum:
        raise MassFileError(f'{where}: must not be below {minimum:g} (got {value:g})')
    return value


def _keys(section, allowed, where: str, required=()) -> dict:
    if not isinstance(section, dict):
        raise MassFileError(f'{where}: a mapping is required')
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise MassFileError(f'{where}: unknown key(s) {unknown}')
    missing = [k for k in required if k not in section]
    if missing:
        raise MassFileError(f'{where}: missing {missing}')
    return section


def _bound(section, where: str, default_value: Optional[float], value_required: bool,
           positive: bool = False) -> Bound:
    _keys(section, ('value', 'plus', 'minus'), where,
          required=('plus', 'minus') + (('value',) if value_required else ()))
    value = _number(section['value'], f'{where}.value') if 'value' in section else default_value
    if value is None:
        raise MassFileError(f'{where}.value: required (nothing to correct)')
    plus = _number(section['plus'], f'{where}.plus')
    minus = _number(section['minus'], f'{where}.minus')
    if minus > value:
        raise MassFileError(f'{where}: minus {minus:g} exceeds the value {value:g}')
    if positive and not value - minus > 0.0:
        raise MassFileError(f'{where}: a mass and its lower bound must stay above 0 (value '
                            f'{value:g}, minus {minus:g})')
    return Bound(value, plus, minus)


def _com(section, where: str, default: Optional[np.ndarray], value_required: bool):
    _keys(section, ('value', 'bound'), where,
          required=('bound',) + (('value',) if value_required else ()))
    if 'value' in section:
        v = section['value']
        if not isinstance(v, list) or len(v) != 3:
            raise MassFileError(f'{where}.value: a list of three numbers is required')
        com = np.array([_number(x, f'{where}.value', None) for x in v])
    else:
        com = default
    return com, _number(section['bound'], f'{where}.bound')


def parse(data, model: RobotModel, simulation: bool) -> MassModel:
    """Apply a loaded mass file (``yaml.safe_load`` output) to ``model``; refuse what is wrong."""
    if data is None:
        raise MassFileError('the mass file is empty')
    _keys(data, ('stability_mass_model', 'simulation_only', 'template', 'default_uncertainty',
                 'bodies', 'added', 'suspension'), 'mass file',
          required=('stability_mass_model', 'default_uncertainty'))
    if data.get('template'):
        raise MassFileError('this is the template: copy it outside the checkout and give every '
                            'value')
    if data['stability_mass_model'] != FORMAT:
        raise MassFileError(f'stability_mass_model: format {FORMAT} is required')
    simulation_only = bool(data.get('simulation_only', False))
    if simulation_only and not simulation:
        raise MassFileError('this mass file is marked simulation_only: refused outside '
                            'simulation')
    du = _keys(data['default_uncertainty'], ('mass_fraction', 'com_m'), 'default_uncertainty',
               required=('mass_fraction', 'com_m'))
    frac = _number(du['mass_fraction'], 'default_uncertainty.mass_fraction')
    if frac >= 1.0:
        raise MassFileError('default_uncertainty.mass_fraction: must be below 1')
    com_default = _number(du['com_m'], 'default_uncertainty.com_m')
    corrections = data.get('bodies') or {}
    if not isinstance(corrections, dict):
        raise MassFileError('bodies: a mapping is required')
    by_link = {b.link: b for b in model.bodies}
    for link in corrections:
        if link not in by_link:
            raise MassFileError(f'bodies.{link}: not a body of the description (links with an '
                                'inertial only)')
    bodies: List[UncertainBody] = []
    for b in model.bodies:
        if b.link in corrections:
            where = f'bodies.{b.link}'
            entry = _keys(corrections[b.link], ('mass', 'com'), where, required=('mass', 'com'))
            mass = _bound(entry['mass'], f'{where}.mass', b.mass, False, positive=True)
            com, com_bound = _com(entry['com'], f'{where}.com', b.com, False)
            bodies.append(UncertainBody(Body(b.name, b.link, mass.value, com, b.inertia), mass,
                                        com_bound))
        else:
            bodies.append(UncertainBody(b, Bound(b.mass, b.mass * frac, b.mass * frac),
                                        com_default))
    added = data.get('added') or {}
    if not isinstance(added, dict):
        raise MassFileError('added: a mapping is required')
    names = {b.name for b in model.bodies}
    for name, entry in added.items():
        where = f'added.{name}'
        if name in names or name in model.links:
            raise MassFileError(f'{where}: the name exists in the description')
        _keys(entry, ('parent', 'mass', 'com', 'inertia'), where,
              required=('parent', 'mass', 'com'))
        if entry['parent'] not in model.links:
            raise MassFileError(f'{where}.parent: {entry["parent"]!r} is not a link of the '
                                'description')
        mass = _bound(entry['mass'], f'{where}.mass', None, True, positive=True)
        com, com_bound = _com(entry['com'], f'{where}.com', None, True)
        inertia = np.zeros((3, 3))
        if 'inertia' in entry:
            v = entry['inertia']
            if not isinstance(v, list) or len(v) != 3:
                raise MassFileError(f'{where}.inertia: [ixx, iyy, izz] is required')
            inertia = np.diag([_number(x, f'{where}.inertia') for x in v])
        bodies.append(UncertainBody(Body(name, entry['parent'], mass.value, com, inertia), mass,
                                    com_bound))
    stiffness = damping = None
    if model.springs:
        ks = sorted({s.stiffness for s in model.springs.values()})
        cs = sorted({s.damping for s in model.springs.values()})
        if len(ks) != 1 or len(cs) != 1:
            raise MassFileError("the description's springs differ from corner to corner: one "
                                'stiffness and one damping are required')
        stiffness, damping = Bound(ks[0], 0.0, 0.0), Bound(cs[0], 0.0, 0.0)
    if 'suspension' in data:
        if not model.springs:
            raise MassFileError('suspension: the description has no springs to bound')
        sus = _keys(data['suspension'], ('stiffness', 'damping'), 'suspension')
        if 'stiffness' in sus:
            stiffness = _bound(sus['stiffness'], 'suspension.stiffness', stiffness.value, False)
            if stiffness.low <= 0.0:
                raise MassFileError('suspension.stiffness: its lower bound must stay above 0')
        if 'damping' in sus:
            damping = _bound(sus['damping'], 'suspension.damping', damping.value, False)
    return MassModel(tuple(bodies), stiffness, damping,
                     fingerprint(bodies, stiffness, damping, model.description_sha256),
                     simulation_only)


def fingerprint(bodies, stiffness: Optional[Bound], damping: Optional[Bound],
                description_sha256: str = '') -> str:
    """
    Return the sha256 of the model in force.

    Every body with its bounds, the springs' bounds, and the description itself (its geometry:
    contacts, joints, spring positions).
    """
    def r(x):
        return [round(float(v), 9) for v in np.ravel(x)]
    canon = {
        'bodies': sorted([[b.body.name, b.body.link, round(b.mass.value, 9), round(b.mass.plus, 9),
                           round(b.mass.minus, 9), r(b.body.com), round(b.com_bound, 9),
                           r(b.body.inertia)] for b in bodies]),
        'stiffness': None if stiffness is None else r([stiffness.value, stiffness.plus,
                                                       stiffness.minus]),
        'damping': None if damping is None else r([damping.value, damping.plus, damping.minus]),
        'description_sha256': description_sha256,
    }
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


def as_arrays(mm: MassModel) -> Dict[str, np.ndarray]:
    """Return per-body nominal, low and high masses and centre-of-mass bounds (model order)."""
    return {
        'mass': np.array([b.mass.value for b in mm.bodies]),
        'mass_low': np.array([b.mass.low for b in mm.bodies]),
        'mass_high': np.array([b.mass.high for b in mm.bodies]),
        'com_bound': np.array([b.com_bound for b in mm.bodies]),
    }


def summary(mm: MassModel) -> Mapping[str, float]:
    """Return the totals for a log line."""
    a = as_arrays(mm)
    return {'mass_kg': float(a['mass'].sum()), 'low_kg': float(a['mass_low'].sum()),
            'high_kg': float(a['mass_high'].sum()), 'bodies': len(mm.bodies)}
