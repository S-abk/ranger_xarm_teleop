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
The robot's masses, wheel contacts and suspension, from its description (numpy, no ROS).

The stability monitor (contract, "Stability monitor") weighs every body of the robot
description: each link's ``<inertial>`` (mass, centre of mass, inertia) placed by forward
kinematics at the measured joint values. Nothing here is a built-in number: a description
without an inertial on a body simply has no mass there, and the monitor's mass file corrects
or adds bodies (``stability_mass.py``).

* **Frame.** Everything is returned in ``base_frame`` (normally ``base_link``), whatever the
  tree's root is.
* **Wheel contacts.** The links named as wheels carry a sphere or cylinder collision; the
  contact is the wheel centre moved down by that radius along the base frame's z axis, at the
  measured steering angles (so an offset steering axis moves the contact, a centred one does
  not). The contacts lie on the design plane: suspension deflection is not an input.
* **Joint values.** Every moving joint must be measured (``required_joints``) except a fixed
  joint, a mimic joint (it follows its source), a sprung joint (the suspension stays at its
  design position) and a wheel's spin (a continuous joint on the path to a wheel: a spinning
  wheel does not move its centre).
* **Suspension.** A prismatic joint carrying a gz spring (``springStiffness`` and
  ``springReference`` in its ``<gazebo reference>``) is a spring; its ``<dynamics damping>`` is
  the damper. Roll and pitch stiffness follow from the springs' positions.
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ElementTree

import numpy as np


class DescriptionError(ValueError):
    """The description cannot carry a stability figure; the message says why."""


@dataclass(frozen=True)
class Body:
    """One mass: its link, mass (kg), centre of mass and inertia in the link frame."""

    name: str
    link: str
    mass: float
    com: np.ndarray                       # (3,), link frame
    inertia: np.ndarray                   # (3, 3), about the centre of mass, link frame axes


@dataclass(frozen=True)
class Joint:
    """One joint of the tree."""

    name: str
    kind: str                              # fixed, revolute, continuous, prismatic
    parent: str
    child: str
    origin: np.ndarray                     # 4 x 4, parent link to joint frame
    axis: np.ndarray                       # unit, joint frame
    mimic: Optional[Tuple[str, float, float]] = None
    damping: float = 0.0
    lower: float = -math.inf
    upper: float = math.inf


@dataclass(frozen=True)
class Spring:
    """A sprung joint of the suspension."""

    joint: str
    stiffness: float                       # N/m
    reference: float                       # m, the joint value where the spring force is zero
    damping: float                         # N s/m
    lower: float
    upper: float


@dataclass(frozen=True)
class Posed:
    """The bodies and contacts at one set of joint values, in the base frame."""

    names: Tuple[str, ...]
    masses: np.ndarray                     # (N,)
    positions: np.ndarray                  # (N, 3) centres of mass
    inertias: np.ndarray                   # (N, 3, 3) about each centre of mass, base axes
    contacts: Dict[str, np.ndarray]        # wheel link -> (3,) contact point
    plane_z: float                         # height of the contact plane (mean of the contacts)


def _floats(text: Optional[str], default: Sequence[float]) -> List[float]:
    return [float(v) for v in text.split()] if text else list(default)


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(element) -> np.ndarray:
    o = element.find('origin') if element is not None else None
    xyz = _floats(o.get('xyz') if o is not None else None, (0.0, 0.0, 0.0))
    rpy = _floats(o.get('rpy') if o is not None else None, (0.0, 0.0, 0.0))
    t = np.eye(4)
    t[:3, :3] = _rpy(*rpy)
    t[:3, 3] = xyz
    return t


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


class RobotModel:
    """The mass model of one robot description."""

    def __init__(self, urdf: str, base_frame: str, wheel_links: Sequence[str]):
        try:
            root = ElementTree.fromstring(urdf)
        except ElementTree.ParseError as err:
            raise DescriptionError(f'the robot description is not valid XML: {err}') from err
        self.base_frame = base_frame
        self.description_sha256 = hashlib.sha256(urdf.encode('utf-8')).hexdigest()
        self.links = {link.get('name'): link for link in root.findall('link')}
        if base_frame not in self.links:
            raise DescriptionError(f'base frame {base_frame!r} is not a link of the description')
        self.bodies: List[Body] = []
        for name, link in self.links.items():
            inertial = link.find('inertial')
            if inertial is None or inertial.find('mass') is None:
                continue
            mass = float(inertial.find('mass').get('value'))
            t = _origin(inertial)
            it = inertial.find('inertia')
            inertia = np.zeros((3, 3))
            if it is not None:
                def g(key):
                    return float(it.get(key, '0'))
                inertia = np.array([[g('ixx'), g('ixy'), g('ixz')],
                                    [g('ixy'), g('iyy'), g('iyz')],
                                    [g('ixz'), g('iyz'), g('izz')]])
            rot = t[:3, :3]
            self.bodies.append(Body(name, name, mass, t[:3, 3].copy(), rot @ inertia @ rot.T))
        self.joints: Dict[str, Joint] = {}
        self._child_joint: Dict[str, str] = {}
        for j in root.findall('joint'):
            axis_el = j.find('axis')
            axis = np.array(_floats(axis_el.get('xyz') if axis_el is not None else None,
                                    (1.0, 0.0, 0.0)))
            norm = np.linalg.norm(axis)
            axis = axis / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
            mim = j.find('mimic')
            mimic = None
            if mim is not None:
                mimic = (mim.get('joint'), float(mim.get('multiplier', '1')),
                         float(mim.get('offset', '0')))
            dyn = j.find('dynamics')
            lim = j.find('limit')
            joint = Joint(
                name=j.get('name'), kind=j.get('type'), parent=j.find('parent').get('link'),
                child=j.find('child').get('link'), origin=_origin(j), axis=axis, mimic=mimic,
                damping=float(dyn.get('damping', '0')) if dyn is not None else 0.0,
                lower=float(lim.get('lower', '-inf')) if lim is not None else -math.inf,
                upper=float(lim.get('upper', 'inf')) if lim is not None else math.inf)
            self.joints[joint.name] = joint
            self._child_joint[joint.child] = joint.name
        roots = [name for name in self.links if name not in self._child_joint]
        if len(roots) != 1:
            raise DescriptionError(f'the description is not one tree (roots: {sorted(roots)})')
        self.root = roots[0]
        self.springs: Dict[str, Spring] = {}
        for gz in root.findall('gazebo'):
            ref = gz.get('reference')
            k_el, r_el = gz.find('springStiffness'), gz.find('springReference')
            if ref in self.joints and k_el is not None and r_el is not None:
                j = self.joints[ref]
                if j.kind == 'prismatic':
                    self.springs[ref] = Spring(ref, float(k_el.text), float(r_el.text),
                                               j.damping, j.lower, j.upper)
        self.wheel_links = list(wheel_links)
        self.wheel_radius: Dict[str, float] = {}
        for w in self.wheel_links:
            if w not in self.links:
                raise DescriptionError(f'wheel link {w!r} is not in the description (no wheel '
                                       'contacts: is the base modelled with its wheels?)')
            self.wheel_radius[w] = self._radius(self.links[w], w)
        if len(self.wheel_links) < 3:
            raise DescriptionError('fewer than three wheel links are configured: no support '
                                   'polygon')
        self._order = self._topological()
        self.wheel_path_joints = set()       # the joints between the base and each wheel
        for w in self.wheel_links:
            link = w
            while link in self._child_joint:
                jn = self._child_joint[link]
                self.wheel_path_joints.add(jn)
                link = self.joints[jn].parent

    @staticmethod
    def _radius(link, name: str) -> float:
        for col in link.findall('collision'):
            geom = col.find('geometry')
            if geom is None:
                continue
            for kind in ('sphere', 'cylinder'):
                g = geom.find(kind)
                if g is not None and g.get('radius'):
                    return float(g.get('radius'))
        raise DescriptionError(f'wheel link {name!r} has no sphere or cylinder collision: '
                               'its radius is unknown')

    def _topological(self) -> List[str]:
        order, frontier = [], [self.root]
        children: Dict[str, List[str]] = {}
        for j in self.joints.values():
            children.setdefault(j.parent, []).append(j.name)
        while frontier:
            link = frontier.pop(0)
            for jn in sorted(children.get(link, [])):
                order.append(jn)
                frontier.append(self.joints[jn].child)
        return order

    def required_joints(self) -> List[str]:
        """Return the joints whose measured value the figure needs (see the module text)."""
        return sorted(j.name for j in self.joints.values()
                      if j.kind in ('revolute', 'prismatic', 'continuous') and j.mimic is None
                      and j.name not in self.springs and not self._is_wheel_spin(j))

    def _is_wheel_spin(self, joint: Joint) -> bool:
        return joint.kind == 'continuous' and joint.name in self.wheel_path_joints

    def _value(self, joint: Joint, q: Mapping[str, float]) -> float:
        if joint.kind == 'fixed':
            return 0.0
        if joint.name in self.springs:
            return 0.0                      # the design position: deflection is not an input
        if joint.mimic is not None:
            src, mult, off = joint.mimic
            return q.get(src, 0.0) * mult + off
        if self._is_wheel_spin(joint):
            return 0.0                      # a wheel's spin does not move its centre
        return q.get(joint.name, 0.0)

    def link_transforms(self, q: Mapping[str, float]) -> Dict[str, np.ndarray]:
        """Return every link's 4 x 4 transform in the base frame."""
        out = {self.root: np.eye(4)}
        for jn in self._order:
            j = self.joints[jn]
            t = out[j.parent] @ j.origin
            v = self._value(j, q)
            if v != 0.0:
                m = np.eye(4)
                if j.kind in ('revolute', 'continuous'):
                    m[:3, :3] = _axis_rotation(j.axis, v)
                elif j.kind == 'prismatic':
                    m[:3, 3] = j.axis * v
                t = t @ m
            out[j.child] = t
        base_inv = np.linalg.inv(out[self.base_frame])
        return {name: base_inv @ t for name, t in out.items()}

    def pose(self, q: Mapping[str, float], bodies: Optional[Sequence[Body]] = None) -> Posed:
        """
        Return the bodies and the contacts at the joint values ``q``.

        ``bodies`` replaces the description's own (the mass file's model in force, whose
        added bodies are fixed to existing links).
        """
        tf = self.link_transforms(q)
        bodies = list(self.bodies if bodies is None else bodies)
        names, masses, positions, inertias = [], [], [], []
        for b in bodies:
            t = tf[b.link]
            names.append(b.name)
            masses.append(b.mass)
            positions.append(t[:3, :3] @ b.com + t[:3, 3])
            inertias.append(t[:3, :3] @ b.inertia @ t[:3, :3].T)
        contacts = {}
        for w in self.wheel_links:
            centre = tf[w][:3, 3]
            contacts[w] = centre - np.array([0.0, 0.0, self.wheel_radius[w]])
        plane_z = float(np.mean([c[2] for c in contacts.values()]))
        return Posed(tuple(names), np.array(masses, dtype=float), np.array(positions, dtype=float),
                     np.array(inertias, dtype=float).reshape(len(bodies), 3, 3), contacts, plane_z)

    def unsprung_links(self) -> List[str]:
        """Return the links below a spring (they do not lean with the body)."""
        below = set()
        for jn in self.springs:
            stack = [self.joints[jn].child]
            while stack:
                link = stack.pop()
                below.add(link)
                stack.extend(j.child for j in self.joints.values() if j.parent == link)
        return sorted(below)

    def suspension_stiffness(self) -> Tuple[float, float]:
        """
        Return the roll and pitch stiffness (N m/rad) of the springs about the contacts' centre.

        Each spring acts at its own position in the base frame; with no springs the robot is
        rigid on its wheels and both are infinite.
        """
        if not self.springs:
            return math.inf, math.inf
        tf = self.link_transforms({})
        xs, ys, ks = [], [], []
        for s in self.springs.values():
            p = tf[self.joints[s.joint].child][:3, 3]
            xs.append(p[0])
            ys.append(p[1])
            ks.append(s.stiffness)
        xs, ys, ks = np.array(xs), np.array(ys), np.array(ks)
        xc = float(np.sum(ks * xs) / np.sum(ks))
        yc = float(np.sum(ks * ys) / np.sum(ks))
        return float(np.sum(ks * (ys - yc) ** 2)), float(np.sum(ks * (xs - xc) ** 2))
