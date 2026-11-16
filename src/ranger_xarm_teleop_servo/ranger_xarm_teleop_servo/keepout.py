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
Link keep-out model (G-15): clearance of the arm from the robot's own fixed structure. No ROS.

Built at start-up from the robot description, never written by hand:

* **Fixed bodies:** every link joined to the arm base through fixed joints only (in either
  direction of the tree): base, mounts, sensors, anything a project bolts on. Their box,
  cylinder and sphere collision shapes are used exactly; a mesh gets a sphere cover.
* **Moving bodies:** every link downstream of the first arm joint: arm links, gripper,
  cameras, any tool a project attaches. Each is covered by spheres (below). Links moved by
  joints that are not arm joints (gripper fingers; ``<mimic>`` resolved) are placed with
  those joints' measured values.
* **Pairs:** every moving body against every fixed body, and the tool (bodies moved by the
  last arm joint) against arm links at least ``TOOL_ARM_JOINTS`` joints up the chain.
  Excluded, and reported: a moving link attached directly to a fixed link (adjacent), and
  a pair whose clearance stays below the margin over the whole sampled range of the
  joints it depends on (structurally near: it cannot be kept clear).

**Sphere cover, and why the model never reports more clearance than the geometry has.**
Every collision shape of a moving body (and every fixed mesh) is turned into triangles; a
box, cylinder or sphere into a polyhedron that contains it. Each triangle is sampled on a
barycentric grid whose sub-triangles have edges no longer than ``spacing``: every point of
the triangle is then within ``spacing`` of a sample (it lies in a sub-triangle, and every
point of a triangle is within its longest edge of each corner). The samples are binned
into cubic cells of size ``cell``; each occupied cell becomes one sphere centred on its
samples' centroid, with radius = farthest sample from that centre + ``spacing``. So every
point of the surface is inside some sphere, and the distance from the sphere union to any
fixed shape is at most the distance from the surface to it. A fixed shape is used exactly
(or, for a mesh, covered the same way). The model's clearance is therefore never larger
than the true distance between the collision geometries; what it gives away (inflation)
is at most the largest sphere radius, ``cell * sqrt(3) / 2 + spacing``.

STL meshes only (binary or ASCII); any other mesh format refuses the model.
"""

from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ElementTree

import numpy as np

from .kinematics import chain_from_urdf

TOOL_ARM_JOINTS = 3            # tool against arm links at least this many joints up
DEFAULT_CELL = 0.020           # m
DEFAULT_SPACING = 0.002        # m
NEAR_SAMPLES = 3000            # postures sampled to find structurally near pairs
CYLINDER_SIDES = 16


class ModelError(ValueError):
    """The description cannot be turned into a keep-out model (it is refused)."""


# --- description ---------------------------------------------------------------------------

def _floats(text: Optional[str], default: Sequence[float]) -> List[float]:
    return [float(v) for v in text.split()] if text else list(default)


def _transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                   [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                   [-sp, cp * sr, cp * cr]]
    out[:3, 3] = xyz
    return out


def _origin(element) -> np.ndarray:
    origin = element.find('origin') if element is not None else None
    if origin is None:
        return np.eye(4)
    return _transform(_floats(origin.get('xyz'), (0, 0, 0)), _floats(origin.get('rpy'), (0, 0, 0)))


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    out = np.eye(4)
    out[:3, :3] = [[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                   [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                   [z * x * k - y * s, z * y * k + x * s, c + z * z * k]]
    return out


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    mimic: Optional[Tuple[str, float, float]]      # (joint, multiplier, offset)
    text: str                                      # canonical text for the fingerprint


def _joints(root) -> Dict[str, _Joint]:
    out = {}
    for joint in root.findall('joint'):
        parent, child = joint.find('parent'), joint.find('child')
        if parent is None or child is None:
            continue
        axis_element = joint.find('axis')
        axis = np.array(_floats(axis_element.get('xyz') if axis_element is not None else None,
                                (1, 0, 0)))
        mimic_element = joint.find('mimic')
        mimic = None
        if mimic_element is not None:
            mimic = (mimic_element.get('joint'), float(mimic_element.get('multiplier', '1')),
                     float(mimic_element.get('offset', '0')))
        origin = _origin(joint)
        kind = joint.get('type')
        norm = float(np.linalg.norm(axis))
        text = (f'{joint.get("name")}|{kind}|{parent.get("link")}|{child.get("link")}|'
                f'{np.round(origin, 9).tolist()}|{axis.tolist()}|{mimic}')
        out[joint.get('name')] = _Joint(joint.get('name'), kind, parent.get('link'),
                                        child.get('link'), origin,
                                        axis / norm if norm > 0 else axis, mimic, text)
    return out


def make_resolver(share_directory: Callable[[str], str]) -> Callable[[str], bytes]:
    """
    Return a mesh reader for ``package://``, ``file://`` and plain paths.

    ``share_directory(package)`` gives a package's share directory (the node passes
    ament_index's lookup, which keeps this module free of ROS).
    """
    def resolve(name: str) -> bytes:
        if name.startswith('package://'):
            package, _, rest = name[len('package://'):].partition('/')
            path = share_directory(package) + '/' + rest
        elif name.startswith('file://'):
            path = name[len('file://'):]
        else:
            path = name
        with open(path, 'rb') as handle:
            return handle.read()
    return resolve


# --- geometry ------------------------------------------------------------------------------

def read_stl(data: bytes) -> np.ndarray:
    """Return the triangles (N x 3 x 3) of a binary or ASCII STL file."""
    if len(data) >= 84:
        count = int.from_bytes(data[80:84], 'little')
        if len(data) == 84 + 50 * count and count > 0:
            record = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
            return np.frombuffer(data[84:], dtype=record)['v'].astype(float)
    text = data.decode('ascii', errors='replace')
    if text.lstrip().startswith('solid') and 'vertex' in text:
        values = [line.split()[1:4] for line in text.splitlines()
                  if line.strip().startswith('vertex')]
        points = np.array(values, dtype=float)
        if len(points) and len(points) % 3 == 0:
            return points.reshape(-1, 3, 3)
    raise ModelError('not a readable STL file')


def _box_triangles(half: np.ndarray) -> np.ndarray:
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                       dtype=float) * half
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return corners[np.array(faces)]


def _cylinder_triangles(radius: float, length: float) -> np.ndarray:
    # A prism whose cross-section circumscribes the circle: it contains the cylinder.
    r = radius / math.cos(math.pi / CYLINDER_SIDES)
    angles = np.arange(CYLINDER_SIDES) * 2.0 * math.pi / CYLINDER_SIDES
    ring = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    h = length / 2.0
    tris = []
    for i in range(CYLINDER_SIDES):
        a, b = ring[i], ring[(i + 1) % CYLINDER_SIDES]
        a0, b0, a1, b1 = [a[0], a[1], -h], [b[0], b[1], -h], [a[0], a[1], h], [b[0], b[1], h]
        tris += [[a0, b0, b1], [a0, b1, a1], [[0, 0, -h], b0, a0], [[0, 0, h], a1, b1]]
    return np.array(tris, dtype=float)


def _sphere_triangles(radius: float) -> np.ndarray:
    # The circumscribed cube contains the sphere.
    return _box_triangles(np.array([radius] * 3))


def _apply(pose: np.ndarray, tris: np.ndarray) -> np.ndarray:
    return tris @ pose[:3, :3].T + pose[:3, 3]


def sample_triangles(tris: np.ndarray, spacing: float) -> np.ndarray:
    """Return points on the triangles such that every surface point is within ``spacing``."""
    tris = np.asarray(tris, dtype=float)
    if len(tris) == 0:
        return np.zeros((0, 3))
    edges = np.stack([np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1),
                      np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1),
                      np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)], axis=1).max(axis=1)
    divisions = np.maximum(1, np.ceil(edges / spacing)).astype(int)
    out = []
    for n in np.unique(divisions):
        group = tris[divisions == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
        keep = (i + j) <= n
        u, v = i[keep] / n, j[keep] / n
        w = 1.0 - u - v
        pts = (w[None, :, None] * group[:, None, 0] + u[None, :, None] * group[:, None, 1]
               + v[None, :, None] * group[:, None, 2])
        out.append(pts.reshape(-1, 3))
    return np.concatenate(out)


def sphere_cover(tris: np.ndarray, cell: float = DEFAULT_CELL,
                 spacing: float = DEFAULT_SPACING) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(centres M x 3, radii M)`` of spheres that contain every triangle."""
    points = sample_triangles(tris, spacing)
    if len(points) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    keys = np.floor(points / cell).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    count = len(unique)
    sums = np.zeros((count, 3))
    np.add.at(sums, inverse, points)
    # Each sphere is centred on its samples' centroid or on the cell centre, whichever
    # needs the smaller radius to hold every sample of the cell.
    options = [sums / np.bincount(inverse, minlength=count)[:, None], (unique + 0.5) * cell]
    best_c, best_r = None, None
    for centres in options:
        radii = np.zeros(count)
        np.maximum.at(radii, inverse, np.linalg.norm(points - centres[inverse], axis=1))
        if best_r is None:
            best_c, best_r = centres, radii
        else:
            better = radii < best_r
            best_c = np.where(better[:, None], centres, best_c)
            best_r = np.where(better, radii, best_r)
    return best_c, best_r + spacing


def _groups(centres: np.ndarray, radii: np.ndarray, size: float):
    """Return per-sphere group index and each group's bounding sphere (centre, radius)."""
    if len(centres) == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 3)), np.zeros(0)
    _, group = np.unique(np.floor(centres / size).astype(np.int64), axis=0,
                         return_inverse=True)
    group = group.reshape(-1)
    count = int(group.max()) + 1
    lo = np.full((count, 3), np.inf)
    hi = np.full((count, 3), -np.inf)
    np.minimum.at(lo, group, centres)
    np.maximum.at(hi, group, centres)
    gc = (lo + hi) / 2.0
    gr = np.zeros(count)
    np.maximum.at(gr, group, np.linalg.norm(centres - gc[group], axis=1) + radii)
    return group, gc, gr


# --- the model -----------------------------------------------------------------------------

GROUP_SIZE = 0.06               # m: spheres are pruned in groups of about this size
CRITICAL = 0.010                # m beyond the margin: rows kept beyond per_pair ...
CRITICAL_ROWS = 48              # ... up to this many per pair


@dataclass
class _Body:
    """A moving link: its path from the arm base and its spheres in its own frame."""

    name: str
    depth: int                       # arm joints on its path (it moves with joints 0..depth-1)
    prefix: int                      # chain segments shared with the arm chain
    tail: List[_Joint]               # joints after the shared prefix
    centres: np.ndarray              # M x 3, link frame
    radii: np.ndarray                # M
    group: np.ndarray                # M, group index
    group_centre: np.ndarray         # G x 3, link frame
    group_radius: np.ndarray         # G
    bound_centre: np.ndarray         # 3, link frame
    bound_radius: float


@dataclass
class _Units:
    """Every fixed collision shape in the arm base frame, by kind, with its fixed link."""

    names: List[str]                                    # fixed link names
    box_rot: np.ndarray                                 # K x 3 x 3
    box_pos: np.ndarray                                 # K x 3
    box_half: np.ndarray                                # K x 3
    cyl_rot: np.ndarray
    cyl_pos: np.ndarray
    cyl_size: np.ndarray                                # K x 2: radius, half length
    sph_pos: np.ndarray
    sph_rad: np.ndarray
    link: np.ndarray                                    # U: fixed link of each unit
    kind: np.ndarray                                    # U: BOX, CYLINDER or SPHERE
    index: np.ndarray                                   # U: index within its kind
    bound_centre: np.ndarray                            # U x 3
    bound_radius: np.ndarray                            # U
    # A fixed mesh's spheres come in clusters: the unit is the cluster's bounding sphere
    # (a lower bound for pruning), its members are the spheres themselves.
    mem_start: np.ndarray                               # U
    mem_count: np.ndarray                               # U (0: not a cluster)
    mem_pos: np.ndarray                                 # members x 3
    mem_rad: np.ndarray                                 # members

    @staticmethod
    def make(names, boxes, cylinders, spheres) -> '_Units':
        """Return the units of ``(link, rot, pos, size)`` lists per kind."""
        def stack(items, width):
            if not items:
                return (np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros((0, width)))
            return (np.array([i[1] for i in items]), np.array([i[2] for i in items]),
                    np.array([i[3] for i in items], dtype=float).reshape(len(items), width))
        br, bp, bh = stack(boxes, 3)
        cr, cp, cs = stack(cylinders, 2)
        _, sp, sr = stack(spheres, 1)
        link = np.array([i[0] for i in boxes] + [i[0] for i in cylinders]
                        + [i[0] for i in spheres], dtype=int)
        kind = np.array([BOX] * len(boxes) + [CYLINDER] * len(cylinders)
                        + [SPHERE] * len(spheres), dtype=int)
        index = np.concatenate([np.arange(len(boxes)), np.arange(len(cylinders)),
                                np.arange(len(spheres))]).astype(int)
        centre = np.concatenate([bp, cp, sp]) if len(link) else np.zeros((0, 3))
        radius = np.concatenate([np.linalg.norm(bh, axis=1), np.hypot(cs[:, 0], cs[:, 1]),
                                 sr[:, 0]]) if len(link) else np.zeros(0)
        count = np.zeros(len(link), dtype=int)
        mem_pos, mem_rad = [np.zeros((0, 3))], [np.zeros(0)]
        first = len(boxes) + len(cylinders)
        for i, item in enumerate(spheres):
            if len(item) > 4:
                count[first + i] = len(item[5])
                mem_pos.append(np.asarray(item[4], dtype=float))
                mem_rad.append(np.asarray(item[5], dtype=float))
        start = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(int)
        return _Units(list(names), br, bp, bh, cr, cp, cs, sp, sr[:, 0], link, kind, index,
                      centre, radius, start, count, np.concatenate(mem_pos),
                      np.concatenate(mem_rad))

    def members(self, entries: np.ndarray, units: np.ndarray):
        """Return ``(entry, member)`` index pairs expanding the cluster units of ``entries``."""
        counts = self.mem_count[units[entries]]
        total = int(counts.sum())
        rep = np.repeat(entries, counts)
        offsets = np.repeat(self.mem_start[units[entries]]
                            - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        return rep, np.arange(total) + offsets

    def pair_distance(self, points: np.ndarray, units: np.ndarray):
        """Return signed distance (P) and unit normal (P x 3) of ``points[i]`` to ``units[i]``."""
        d = np.empty(len(units))
        n = np.empty((len(units), 3))
        kinds = self.kind[units]
        sel = np.nonzero(kinds == BOX)[0]
        if len(sel):
            k = self.index[units[sel]]
            rot = self.box_rot[k]
            local = np.einsum('pij,pi->pj', rot, points[sel] - self.box_pos[k])
            excess = np.abs(local) - self.box_half[k]
            outside = np.maximum(excess, 0.0)
            out_norm = np.linalg.norm(outside, axis=1)
            nl = np.sign(local) * outside / np.maximum(out_norm, 1e-12)[:, None]
            deep = out_norm <= 1e-12
            if deep.any():
                pick = np.zeros((int(deep.sum()), 3))
                axis = np.argmax(excess[deep], axis=1)
                pick[np.arange(len(axis)), axis] = np.where(
                    local[deep, axis] >= 0.0, 1.0, -1.0)
                nl[deep] = pick
            d[sel] = out_norm + np.minimum(excess.max(axis=1), 0.0)
            n[sel] = np.einsum('pij,pj->pi', rot, nl)
        sel = np.nonzero(kinds == CYLINDER)[0]
        if len(sel):
            k = self.index[units[sel]]
            rot = self.cyl_rot[k]
            radius, half = self.cyl_size[k, 0], self.cyl_size[k, 1]
            local = np.einsum('pij,pi->pj', rot, points[sel] - self.cyl_pos[k])
            rho = np.linalg.norm(local[:, :2], axis=1)
            radial = np.where(rho[:, None] > 1e-12, local[:, :2] / np.maximum(rho, 1e-12)[:, None],
                              np.array([1.0, 0.0]))
            dr, dz = rho - radius, np.abs(local[:, 2]) - half
            out_r, out_z = np.maximum(dr, 0.0), np.maximum(dz, 0.0)
            out_norm = np.hypot(out_r, out_z)
            outside = out_norm > 1e-12
            wr = np.where(outside, out_r / np.maximum(out_norm, 1e-12), (dr >= dz).astype(float))
            wz = np.where(outside, out_z / np.maximum(out_norm, 1e-12), (dz > dr).astype(float))
            nl = np.concatenate([radial * wr[:, None],
                                 (np.where(local[:, 2] >= 0.0, 1.0, -1.0) * wz)[:, None]], axis=1)
            d[sel] = out_norm + np.minimum(np.maximum(dr, dz), 0.0)
            n[sel] = np.einsum('pij,pj->pi', rot, nl)
        sel = np.nonzero(kinds == SPHERE)[0]
        if len(sel):
            k = self.index[units[sel]]
            diff = points[sel] - self.sph_pos[k]
            norm = np.linalg.norm(diff, axis=1)
            d[sel] = norm - self.sph_rad[k]
            n[sel] = np.where(norm[:, None] > 1e-12, diff / np.maximum(norm, 1e-12)[:, None],
                              np.array([0.0, 0.0, 1.0]))
        return d, n


@dataclass(frozen=True)
class Excluded:
    """A pair left out of the keep-out, and why."""

    moving: str
    other: str
    reason: str                      # 'adjacent' or 'structurally near'
    low: float = math.nan            # clearance range over the sampled postures (m)
    high: float = math.nan


@dataclass(frozen=True)
class Rows:
    """Keep-out rows at one posture: ``d_dot = grad @ qdot`` for each sphere pair."""

    grad: np.ndarray                 # k x n (per rad/s of each arm joint)
    distance: np.ndarray             # k, m
    pair: np.ndarray                 # k, index into ``Model.pairs``
    clearance: float                 # smallest clearance over all pairs (m)
    nearest: Tuple[str, str]         # the pair with that clearance, ('', '') if not exact
    exact: bool = True               # False: nothing within ``report``; clearance is a bound


@dataclass
class _Frames:
    """Per-posture data: chain axes and points, and each body's pose (computed lazily)."""

    poses: list
    axes: np.ndarray
    points: np.ndarray
    extra: Mapping[str, float]
    body_poses: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class Model:
    """The keep-out model of one robot description (see the module docstring)."""

    base: str
    joint_names: List[str]
    margin: float
    influence: float                         # rows only for pairs closer than this
    bodies: List[_Body]
    units: _Units
    pair_list: List[Tuple[str, int, int]]    # ('fixed', body, fixed link) / ('tool', body, body)
    extra_joints: Dict[str, Tuple[str, float, float]]   # joint -> (source, mult, offset)
    excluded: List[Excluded]
    not_covered: List[str]
    fingerprint: str
    cell: float
    spacing: float
    chain_segments: list

    @property
    def pairs(self) -> List[Tuple[str, str]]:
        """Return the included pairs as (moving body, other body) names."""
        return [self._names(p) for p in self.pair_list]

    def _names(self, pair) -> Tuple[str, str]:
        kind, a, b = pair
        other = self.units.names[b] if kind == 'fixed' else self.bodies[b].name
        return self.bodies[a].name, other

    @property
    def sphere_count(self) -> int:
        """Return the number of spheres of the moving bodies."""
        return int(sum(len(b.radii) for b in self.bodies))

    @property
    def largest_radius(self) -> float:
        """Return the largest sphere radius: the bound on the model's inflation (m)."""
        radii = [b.radii.max() for b in self.bodies if len(b.radii)]
        return float(max(radii)) if radii else 0.0

    # -- kinematics --
    def extra_values(self, measured: Optional[Mapping[str, float]]) -> Dict[str, float]:
        """Return the value of every non-arm joint the model needs, mimic resolved."""
        measured = measured or {}
        out = {}
        missing = []
        for name, (source, mult, offset) in self.extra_joints.items():
            if name in measured:
                out[name] = float(measured[name])
            elif source in measured:
                out[name] = mult * float(measured[source]) + offset
            else:
                missing.append(name)
        if missing:
            raise ModelError('no joint state for ' + ', '.join(sorted(missing)))
        bad = sorted(name for name, value in out.items() if not math.isfinite(value))
        if bad:
            raise ModelError('joint state not finite for ' + ', '.join(bad))
        return out

    def _frames(self, q: np.ndarray, extra: Optional[Mapping[str, float]]) -> _Frames:
        if len(q) != len(self.joint_names):
            raise ValueError(f'{len(self.joint_names)} joint angles are needed, got {len(q)}')
        poses = [np.eye(4)]
        axes, points = [], []
        pose = np.eye(4)
        index = 0
        for seg in self.chain_segments:
            pose = pose @ seg.origin
            if seg.axis is not None:
                axes.append(pose[:3, :3] @ seg.axis)
                points.append(pose[:3, 3].copy())
                pose = pose @ _rotation(seg.axis, float(q[index]))
                index += 1
            poses.append(pose)
        if extra is None:
            extra = {name: 0.0 for name in self.extra_joints}
        return _Frames(poses, np.array(axes), np.array(points), extra)

    def _pose(self, b: int, frames: _Frames) -> np.ndarray:
        pose = frames.body_poses.get(b)
        if pose is None:
            body = self.bodies[b]
            pose = frames.poses[body.prefix]
            for joint in body.tail:
                pose = pose @ joint.origin
                if joint.kind in ('revolute', 'continuous'):
                    pose = pose @ _rotation(joint.axis, frames.extra[joint.name])
                elif joint.kind == 'prismatic':
                    step = np.eye(4)
                    step[:3, 3] = joint.axis * frames.extra[joint.name]
                    pose = pose @ step
            frames.body_poses[b] = pose
        return pose

    def world_spheres(self, q: Sequence[float], extra: Optional[Mapping[str, float]] = None):
        """Return each body's sphere centres in the arm base frame (list of M x 3)."""
        frames = self._frames(np.asarray(q, dtype=float), extra)
        out = []
        for b, body in enumerate(self.bodies):
            pose = self._pose(b, frames)
            out.append(body.centres @ pose[:3, :3].T + pose[:3, 3])
        return out

    def _index(self) -> None:
        """Build the flat lookups the per-cycle evaluation uses, from ``pair_list``."""
        nb, nf = len(self.bodies), len(self.units.names)
        self._allowed_fixed = np.zeros((nb, nf), dtype=bool)
        self._allowed_tool = np.zeros((nb, nb), dtype=bool)
        self._pair_index = {}
        for i, (kind, a, b) in enumerate(self.pair_list):
            self._pair_index[(kind, a, b)] = i
            if kind == 'fixed':
                self._allowed_fixed[a, b] = True
            else:
                self._allowed_tool[a, b] = True
        self._pair_of_fixed = np.full((nb, max(nf, 1)), -1, dtype=int)
        self._pair_of_tool = np.full((nb, nb), -1, dtype=int)
        for (kind, a, b), i in self._pair_index.items():
            (self._pair_of_fixed if kind == 'fixed' else self._pair_of_tool)[a, b] = i
        # All spheres and groups of all bodies, flat.
        sphere_body, centres, radii, group_of = [], [], [], []
        group_body, group_centre, group_radius = [], [], []
        offset = 0
        for b, body in enumerate(self.bodies):
            sphere_body.append(np.full(len(body.radii), b))
            centres.append(body.centres)
            radii.append(body.radii)
            group_of.append(body.group + offset)
            group_body.append(np.full(len(body.group_radius), b))
            group_centre.append(body.group_centre)
            group_radius.append(body.group_radius)
            offset += len(body.group_radius)
        cat = (lambda x, shape: np.concatenate(x) if x else np.zeros(shape))
        self._sphere_body = cat(sphere_body, (0,)).astype(int)
        self._sphere_centre = cat(centres, (0, 3))
        self._sphere_radius = cat(radii, (0,))
        group = cat(group_of, (0,)).astype(int)
        self._group_body = cat(group_body, (0,)).astype(int)
        self._group_centre = cat(group_centre, (0, 3))
        self._group_radius = cat(group_radius, (0,))
        self._members = np.argsort(group, kind='stable')
        self._counts = np.bincount(group, minlength=len(self._group_radius))
        self._starts = np.concatenate([[0], np.cumsum(self._counts)[:-1]]).astype(int)
        self._depth = np.array([body.depth for body in self.bodies], dtype=int)
        self._body_centre = np.array([b.bound_centre for b in self.bodies]).reshape(-1, 3)
        self._body_radius = np.array([b.bound_radius for b in self.bodies])
        self._group_members = np.argsort(self._group_body, kind='stable')
        self._group_count = np.bincount(self._group_body, minlength=len(self.bodies))
        self._group_start = np.concatenate([[0], np.cumsum(self._group_count)[:-1]]).astype(int)

    def _expand(self, groups: np.ndarray) -> np.ndarray:
        counts = self._counts[groups]
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, dtype=int)
        offsets = np.repeat(self._starts[groups] - np.concatenate([[0], np.cumsum(counts)[:-1]]),
                            counts)
        return self._members[np.arange(total) + offsets]

    def _poses(self, frames: _Frames):
        rot = np.array([self._pose(b, frames)[:3, :3] for b in range(len(self.bodies))])
        pos = np.array([self._pose(b, frames)[:3, 3] for b in range(len(self.bodies))])
        return rot, pos

    def _compute(self, frames: _Frames, limit: float):
        """
        Return every sphere pair that may be closer than ``limit``, and a lower bound.

        ``(pair, d, normal, point, other, lower)``: per pair of bodies (index into
        ``pair_list``) and sphere of the moving body, its clearance, the unit normal away
        from the other body, its centre and, for a tool pair, the other sphere's centre
        (NaN for a fixed body). ``lower`` is a lower bound of every clearance not listed.
        """
        rot, pos = self._poses(frames)
        gb = self._group_body
        gc = np.einsum('gij,gj->gi', rot[gb], self._group_centre) + pos[gb]
        lower = math.inf
        out_pair, out_d, out_n, out_p, out_o = [], [], [], [], []
        units = self.units
        if len(units.link) and len(gc):
            # Bodies against shapes, then the groups of the bodies in reach.
            # Lower bounds: exact distance from a bounding sphere's centre to each shape,
            # less its radius (every sphere it bounds is inside it).
            bc = np.einsum('bij,bj->bi', rot, self._body_centre) + pos
            bi, ui = np.nonzero(self._allowed_fixed[:, units.link])
            gap = units.pair_distance(bc[bi], ui)[0] - self._body_radius[bi]
            near = gap <= limit
            if (~near).any():
                lower = min(lower, float(gap[~near].min()))
            bi, ui = bi[near], ui[near]
            counts = self._group_count[bi]
            gi = self._group_members[np.repeat(self._group_start[bi] - np.concatenate(
                [[0], np.cumsum(counts)[:-1]]), counts) + np.arange(int(counts.sum()))] \
                if len(bi) else np.zeros(0, dtype=int)
            ui = np.repeat(ui, counts)
            gap = units.pair_distance(gc[gi], ui)[0] - self._group_radius[gi]
            near = gap <= limit
            if (~near).any():
                lower = min(lower, float(gap[~near].min()))
            gi, ui = gi[near], ui[near]
            if len(gi):
                spheres = self._expand(gi)
                pair_units = np.repeat(ui, self._counts[gi])
                body = self._sphere_body[spheres]
                world = np.einsum('pij,pj->pi', rot[body], self._sphere_centre[spheres]) \
                    + pos[body]
                d, normal = units.pair_distance(world, pair_units)
                cluster = units.mem_count[pair_units] > 0
                if cluster.any():
                    # Replace each cluster's lower bound by its member spheres.
                    rep, mem = units.members(np.nonzero(cluster)[0], pair_units)
                    diff = world[rep] - units.mem_pos[mem]
                    norm = np.linalg.norm(diff, axis=1)
                    keep = ~cluster
                    d = np.concatenate([d[keep], norm - units.mem_rad[mem]])
                    normal = np.concatenate([normal[keep],
                                             diff / np.maximum(norm, 1e-12)[:, None]])
                    spheres = np.concatenate([spheres[keep], spheres[rep]])
                    pair_units = np.concatenate([pair_units[keep], pair_units[rep]])
                    body = np.concatenate([body[keep], body[rep]])
                    world = np.concatenate([world[keep], world[rep]])
                d = d - self._sphere_radius[spheres]
                within = d <= limit
                if not within.all():
                    lower = min(lower, float(d[~within].min()))
                    d, normal, spheres = d[within], normal[within], spheres[within]
                    pair_units, body, world = pair_units[within], body[within], world[within]
                # One entry per (sphere, shape): each is smooth in the posture, so a row
                # never jumps from one shape to another.
                out_pair.append(self._pair_of_fixed[body, units.link[pair_units]])
                out_d.append(d)
                out_n.append(normal)
                out_p.append(world)
                out_o.append(np.full((len(d), 3), np.nan))
        if self._allowed_tool.any():
            tool_groups = np.nonzero(self._allowed_tool[gb].any(axis=1))[0]
            arm_groups = np.nonzero(self._allowed_tool[:, gb].any(axis=0))[0]
            gap = (np.linalg.norm(gc[tool_groups][:, None] - gc[arm_groups][None], axis=2)
                   - self._group_radius[tool_groups][:, None]
                   - self._group_radius[arm_groups][None])
            allowed = self._allowed_tool[gb[tool_groups]][:, gb[arm_groups]]
            gap = np.where(allowed, gap, np.inf)
            ti, ai = np.nonzero(gap <= limit)
            far = gap[gap > limit]
            if len(far):
                lower = min(lower, float(far.min()))
            if len(ti):
                tg, ag = tool_groups[ti], arm_groups[ai]
                # Every sphere of each tool group against every sphere of the arm group.
                ts = self._expand(tg)
                rep = np.repeat(np.arange(len(tg)), self._counts[tg])
                counts_a = self._counts[ag][rep]
                ts = np.repeat(ts, counts_a)
                as_ = self._expand(np.repeat(ag, self._counts[tg]))
                bt, ba = self._sphere_body[ts], self._sphere_body[as_]
                pt = np.einsum('pij,pj->pi', rot[bt], self._sphere_centre[ts]) + pos[bt]
                pa = np.einsum('pij,pj->pi', rot[ba], self._sphere_centre[as_]) + pos[ba]
                diff = pt - pa
                norm = np.linalg.norm(diff, axis=1)
                d = norm - self._sphere_radius[ts] - self._sphere_radius[as_]
                within = d <= limit
                if not within.all():
                    lower = min(lower, float(d[~within].min()))
                    d, diff, norm = d[within], diff[within], norm[within]
                    ts, bt, ba, pt, pa = ts[within], bt[within], ba[within], pt[within], \
                        pa[within]
                # One entry per pair of spheres (smooth, see above).
                out_pair.append(self._pair_of_tool[bt, ba])
                out_d.append(d)
                out_n.append(diff / np.maximum(norm, 1e-12)[:, None])
                out_p.append(pt)
                out_o.append(pa)
        if out_pair:
            return (np.concatenate(out_pair), np.concatenate(out_d), np.concatenate(out_n),
                    np.concatenate(out_p), np.concatenate(out_o), lower)
        return (np.zeros(0, dtype=int), np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)),
                np.zeros((0, 3)), lower)

    def _jacobian(self, depth: np.ndarray, points: np.ndarray, frames: _Frames) -> np.ndarray:
        """Return the Jacobians (k x 3 x n) of base-frame points moved by joints < depth."""
        n = len(self.joint_names)
        out = np.zeros((len(points), 3, n))
        for i in range(n):
            moving = depth > i
            if moving.any():
                out[moving, :, i] = np.cross(frames.axes[i], points[moving] - frames.points[i])
        return out

    def evaluate(self, q: Sequence[float], extra: Optional[Mapping[str, float]] = None,
                 influence: Optional[float] = None, per_pair: int = 16) -> Rows:
        """
        Return the keep-out rows at ``q`` and the smallest clearance.

        Rows: per pair of bodies the up to ``per_pair`` closest sphere pairs (a sphere and
        a fixed shape, or two spheres) nearer than
        ``influence`` (default: the model's). The clearance is exact when it is within
        ``influence``; otherwise it is a lower bound (``Rows.exact`` False).
        """
        n = len(self.joint_names)
        if not self.pair_list:
            # Nothing to keep apart (no collision geometry on one side).
            return Rows(np.zeros((0, n)), np.zeros(0), np.zeros(0, dtype=int), math.inf,
                        ('', ''), False)
        frames = self._frames(np.asarray(q, dtype=float), extra)
        reach = self.influence if influence is None else influence
        pair, d, normal, point, other, lower = self._compute(frames, reach)
        if len(d) and d.min() <= reach:
            k = int(np.argmin(d))
            best, nearest, exact = float(d[k]), self.pairs[pair[k]], True
        else:
            best = float(min(lower, d.min() if len(d) else math.inf))
            nearest, exact = ('', ''), False
        near = np.nonzero(d < reach)[0]
        if len(near) == 0:
            return Rows(np.zeros((0, n)), np.zeros(0), np.zeros(0, dtype=int), best, nearest,
                        exact)
        # Up to per_pair rows per pair, closest first, and up to CRITICAL_ROWS per pair of
        # those within CRITICAL of the margin (a farther row can approach faster and bind).
        # Bounded, so a posture deep in contact does not make the correction large.
        order = near[np.lexsort((d[near], pair[near]))]
        p = pair[order]
        start = np.concatenate([[True], p[1:] != p[:-1]])
        rank = np.arange(len(order)) - np.maximum.accumulate(
            np.where(start, np.arange(len(order)), 0))
        keep = order[(rank < per_pair)
                     | ((d[order] < self.margin + CRITICAL) & (rank < CRITICAL_ROWS))]
        body = np.array([self.pair_list[i][1] for i in pair[keep]], dtype=int)
        jac = self._jacobian(self._depth[body], point[keep], frames)
        tool = ~np.isnan(other[keep, 0])
        if tool.any():
            arm = np.array([self.pair_list[i][2] for i in pair[keep][tool]], dtype=int)
            jac[tool] -= self._jacobian(self._depth[arm], other[keep][tool], frames)
        grad = np.einsum('ki,kin->kn', normal[keep], jac)
        return Rows(grad, d[keep], pair[keep], best, nearest, exact)

    def clearance(self, q: Sequence[float],
                  extra: Optional[Mapping[str, float]] = None,
                  within: Optional[float] = None) -> Tuple[float, Tuple[str, str]]:
        """
        Return the smallest clearance and the pair.

        Exact up to ``within`` (default: the influence distance); beyond, a lower bound
        and no pair. A larger ``within`` costs much more where many spheres are near.
        """
        rows = self.evaluate(q, extra, influence=within, per_pair=1)
        return rows.clearance, rows.nearest

    def pair_minima(self, q: Sequence[float], extra: Optional[Mapping[str, float]] = None,
                    limit: float = math.inf) -> np.ndarray:
        """Return each pair's clearance, or ``limit`` when it is certainly not below it."""
        out = np.full(len(self.pair_list), limit if math.isfinite(limit) else math.inf)
        if not self.pair_list:
            return out
        frames = self._frames(np.asarray(q, dtype=float), extra)
        pair, d, _, _, _, _ = self._compute(frames, limit)
        if len(d):
            np.minimum.at(out, pair, d)
        return out

    def subset(self, indices: Sequence[int]) -> 'Model':
        """Return a copy of the model that evaluates only the pairs ``indices``."""
        out = replace(self, pair_list=[self.pair_list[int(i)] for i in indices])
        out._index()
        return out

    def pair_clearance(self, index: int, q: Sequence[float],
                       extra: Optional[Mapping[str, float]] = None,
                       limit: float = math.inf) -> float:
        """Return pair ``index``'s clearance, or ``limit`` when it is certainly above it."""
        return float(self.pair_minima(q, extra, limit)[index])

    def pair_clearances(self, q: Sequence[float],
                        extra: Optional[Mapping[str, float]] = None
                        ) -> Dict[Tuple[str, str], float]:
        """Return the clearance of every included pair of bodies (m)."""
        return {self.pairs[i]: self.pair_clearance(i, q, extra)
                for i in range(len(self.pair_list))}


def _shape_triangles(geometry, resolve: Callable[[str], bytes], hasher):
    """Return ``(triangles, exact)`` of one URDF geometry element (``exact`` for primitives)."""
    kind = geometry.tag
    if kind == 'box':
        half = np.array(_floats(geometry.get('size'), (0, 0, 0))) / 2.0
        hasher.update(f'box{half.tolist()}'.encode())
        return _box_triangles(half), (BOX, half)
    if kind == 'cylinder':
        radius, length = float(geometry.get('radius')), float(geometry.get('length'))
        hasher.update(f'cylinder{radius},{length}'.encode())
        return _cylinder_triangles(radius, length), (CYLINDER, np.array([radius, length / 2.0]))
    if kind == 'sphere':
        radius = float(geometry.get('radius'))
        hasher.update(f'sphere{radius}'.encode())
        return _sphere_triangles(radius), (SPHERE, np.array([radius]))
    if kind == 'mesh':
        name = geometry.get('filename', '')
        if not name.lower().endswith('.stl'):
            raise ModelError(f'mesh {name}: only STL collision meshes are supported')
        try:
            data = resolve(name)
        except Exception as err:  # noqa: B902 - any failure to read refuses the model
            raise ModelError(f'mesh {name}: cannot be read ({err})') from err
        scale = np.array(_floats(geometry.get('scale'), (1, 1, 1)))
        hasher.update(f'mesh{name}{scale.tolist()}'.encode())
        hasher.update(hashlib.sha256(data).digest())
        try:
            tris = read_stl(data)
        except ModelError as err:
            raise ModelError(f'mesh {name}: {err}') from err
        return tris * scale, None
    raise ModelError(f'collision geometry <{kind}> is not supported')


BOX, CYLINDER, SPHERE = 0, 1, 2


def build(urdf: str, resolve: Callable[[str], bytes], base: str, tip: str, margin: float,
          influence_extra: float = 0.06, cell: float = DEFAULT_CELL,
          fixed_cell: Optional[float] = None,
          spacing: float = DEFAULT_SPACING, near_samples: int = NEAR_SAMPLES,
          seed: int = 0) -> Model:
    """
    Build the keep-out model of the arm ``base`` -> ``tip`` in URDF text.

    ``resolve(filename)`` returns a mesh file's bytes (it handles ``package://``).
    ``margin`` is the keep-out margin (m), used here only to find structurally near pairs.
    """
    if not (math.isfinite(margin) and margin > 0.0):
        raise ModelError('the margin must be a positive number')
    try:
        root = ElementTree.fromstring(urdf)
    except ElementTree.ParseError as err:
        raise ModelError(f'not a URDF: {err}') from err
    chain = chain_from_urdf(urdf, base, tip)
    joints = _joints(root)
    arm_names = chain.joint_names
    arm_set = set(arm_names)
    by_parent: Dict[str, List[_Joint]] = {}
    by_child: Dict[str, _Joint] = {}
    for joint in joints.values():
        by_parent.setdefault(joint.parent, []).append(joint)
        by_child[joint.child] = joint
    links = {link.get('name'): link for link in root.findall('link')}
    hasher = hashlib.sha256()
    hasher.update(f'{base}|{tip}|{arm_names}|{cell}|{fixed_cell}|{spacing}|'
                  f'{TOOL_ARM_JOINTS}'.encode())
    for seg in chain.segments:
        hasher.update(joints[seg.name].text.encode())

    # Moving: everything below the first arm joint.
    moving, stack = [], [joints[arm_names[0]].child]
    while stack:
        link = stack.pop()
        moving.append(link)
        stack.extend(j.child for j in by_parent.get(link, []))
    moving_set = set(moving)
    # Fixed: joined to the base through fixed joints only, either direction; each fixed
    # link's pose in the base frame is found on the way.
    fixed_poses: Dict[str, np.ndarray] = {base: np.eye(4)}
    stack, seen = [base], {base}
    while stack:
        link = stack.pop()
        here = fixed_poses[link]
        neighbours = [(j, j.child, here @ j.origin) for j in by_parent.get(link, [])]
        if link in by_child:
            up = by_child[link]
            neighbours.append((up, up.parent, here @ np.linalg.inv(up.origin)))
        for joint, other, pose in neighbours:
            if joint.kind == 'fixed' and other not in seen and other not in moving_set:
                seen.add(other)
                fixed_poses[other] = pose
                stack.append(other)
    not_covered = sorted(name for name in links if name not in moving_set and name not in seen
                         and links[name].find('collision') is not None)

    chain_names = [s.name for s in chain.segments]
    bodies: List[_Body] = []
    extra_joints: Dict[str, Tuple[str, float, float]] = {}
    for link in sorted(moving):
        element = links.get(link)
        collisions = element.findall('collision') if element is not None else []
        if not collisions:
            continue
        path = []
        cursor = link
        while cursor != base:
            joint = by_child[cursor]
            path.append(joint)
            cursor = joint.parent
        path.reverse()
        prefix = 0
        while prefix < len(path) and prefix < len(chain_names) \
                and path[prefix].name == chain_names[prefix]:
            prefix += 1
        tail = path[prefix:]
        for joint in tail:
            hasher.update(joint.text.encode())
            if joint.kind in ('revolute', 'continuous', 'prismatic'):
                if joint.name in arm_set:
                    raise ModelError(f'{link}: arm joint {joint.name} off the arm chain')
                extra_joints[joint.name] = joint.mimic or (joint.name, 1.0, 0.0)
            elif joint.kind != 'fixed':
                raise ModelError(f'{link}: joint {joint.name} of type {joint.kind} is not '
                                 'supported')
        depth = sum(1 for j in path[:prefix] if j.name in arm_set)
        tris = []
        hasher.update(f'moving:{link}'.encode())
        for collision in collisions:
            geometry = collision.find('geometry')
            if geometry is None or len(geometry) == 0:
                raise ModelError(f'{link}: collision without geometry')
            pose = _origin(collision)
            hasher.update(np.round(pose, 9).tobytes())
            local, _ = _shape_triangles(geometry[0], resolve, hasher)
            tris.append(_apply(pose, local))
        centres, radii = sphere_cover(np.concatenate(tris), cell, spacing)
        group, gc, gr = _groups(centres, radii, GROUP_SIZE)
        centre = (centres.min(axis=0) + centres.max(axis=0)) / 2.0
        reach = float((np.linalg.norm(centres - centre, axis=1) + radii).max())
        bodies.append(_Body(link, depth, prefix, tail, centres, radii, group, gc, gr, centre,
                            reach))

    fixed_names: List[str] = []
    boxes, cylinders, spheres = [], [], []
    for link in sorted(seen):
        element = links.get(link)
        collisions = element.findall('collision') if element is not None else []
        if not collisions:
            continue
        f = len(fixed_names)
        fixed_names.append(link)
        hasher.update(f'fixed:{link}{np.round(fixed_poses[link], 9).tolist()}'.encode())
        for collision in collisions:
            geometry = collision.find('geometry')
            if geometry is None or len(geometry) == 0:
                raise ModelError(f'{link}: collision without geometry')
            pose = fixed_poses[link] @ _origin(collision)
            hasher.update(np.round(_origin(collision), 9).tobytes())
            tris, exact = _shape_triangles(geometry[0], resolve, hasher)
            if exact is None:
                c, r = sphere_cover(_apply(pose, tris), fixed_cell or cell, spacing)
                group, gcentre, gradius = _groups(c, r, GROUP_SIZE)
                for g in range(len(gradius)):
                    inside = group == g
                    spheres.append((f, np.eye(3), gcentre[g], [gradius[g]], c[inside],
                                    r[inside]))
            elif exact[0] == BOX:
                boxes.append((f, pose[:3, :3], pose[:3, 3], exact[1]))
            elif exact[0] == CYLINDER:
                cylinders.append((f, pose[:3, :3], pose[:3, 3], exact[1]))
            else:
                spheres.append((f, np.eye(3), pose[:3, 3], exact[1]))
    units = _Units.make(fixed_names, boxes, cylinders, spheres)

    n_arm = len(arm_names)
    adjacent = {(j.child, j.parent) for j in joints.values()
                if j.name in arm_set and j.child in moving_set}
    pair_list: List[Tuple[str, int, int]] = []
    excluded: List[Excluded] = []
    for b, body in enumerate(bodies):
        for f, name in enumerate(fixed_names):
            if (body.name, name) in adjacent:
                excluded.append(Excluded(body.name, name, 'adjacent'))
            else:
                pair_list.append(('fixed', b, f))
    for t, tool in enumerate(bodies):
        if tool.depth != n_arm:
            continue
        for a, other in enumerate(bodies):
            if 1 <= other.depth <= n_arm - TOOL_ARM_JOINTS:
                pair_list.append(('tool', t, a))
    model = Model(base=base, joint_names=list(arm_names), margin=margin,
                  influence=margin + influence_extra, bodies=bodies, units=units,
                  pair_list=pair_list, extra_joints=extra_joints, excluded=excluded,
                  not_covered=not_covered, fingerprint='', cell=cell, spacing=spacing,
                  chain_segments=list(chain.segments))
    model._index()

    # Structurally near: below the margin at every sampled posture.
    if near_samples > 0 and pair_list:
        rng = np.random.default_rng(seed)
        lower = np.array([j.lower for j in chain.joints])
        upper = np.array([j.upper for j in chain.joints])
        lower = np.where(np.isfinite(lower), lower, -math.pi)
        upper = np.where(np.isfinite(upper), upper, math.pi)
        samples = rng.uniform(lower, upper, size=(near_samples, n_arm))
        candidates = np.arange(len(pair_list))
        for q in samples:
            if len(candidates) == 0:
                break
            sub = model.subset(candidates)
            candidates = candidates[sub.pair_minima(q, limit=margin) < margin]
        if len(candidates):
            sub = model.subset(candidates)
            values = np.array([sub.pair_minima(q) for q in samples])
            for column, i in enumerate(candidates):
                moving_name, other = model.pairs[i]
                excluded.append(Excluded(moving_name, other, 'structurally near',
                                         float(values[:, column].min()),
                                         float(values[:, column].max())))
        drop = {int(i) for i in candidates}
        model.pair_list = [p for i, p in enumerate(pair_list) if i not in drop]
        model._index()
    hasher.update(f'{sorted(extra_joints.items())}'.encode())
    model.fingerprint = hasher.hexdigest()
    return model


def fingerprint(urdf: str, resolve: Callable[[str], bytes], base: str, tip: str,
                margin: float) -> str:
    """Return the fingerprint of the model ``urdf`` gives (no pair sampling)."""
    return build(urdf, resolve, base, tip, margin, near_samples=0).fingerprint
