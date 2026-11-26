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
Clearance tables of the link keep-out (G-15).

For arm bodies two joints apart, and the check of bodies one joint apart. No ROS.

**Why tables.** Two moving bodies with exactly two arm joints between them (the inner body A,
the outer body B) have a relative pose that depends on those two joints only, and on the
non-arm joints that move B (a gripper). Their clearance is a function of two angles; it is
computed once, from the description, on a grid, and read per cycle in microseconds.

**A lower bound everywhere, not only at the grid points.**

* *Distance grid of A.* A's collision surface is sampled so that every surface point is within
  ``s`` of a sample; each sample marks its nearest voxel centre (spacing ``g``); ``E`` is the
  exact Euclidean distance from every voxel centre to the nearest marked one (separable, windowed
  to the grid's cap). For any point p and its nearest voxel centre v:
  ``dist(p, A) >= E(v) - s - g sqrt(3) / 2 - |p - v|`` (each surface point is within
  ``s + g sqrt(3) / 2`` of a marked centre; distance is 1-Lipschitz). Stored rounded down.
* *B's cover.* Spheres that contain B's surface (``keepout.sphere_cover``), or for a body the
  gripper moves, spheres at the occupied cells of B's samples over the gripper's whole range
  (radius: half the cell diagonal, ``s``, and the most a surface point moves between two drive
  samples). A sphere's clearance is at least ``dist(centre, A) - radius``.
* *Between grid points.* Turning joint a by ``x`` moves every point of B by at most
  ``La |x|``, joint b by at most ``Lb |x|`` (``La``, ``Lb``: upper bounds of the distance of B's
  points from each axis, over every value of the other joint, computed from the description, not
  sampled), so the clearance is ``(La, Lb)``-Lipschitz. A bilinear read of node values that are
  lower bounds is then above the clearance by at most ``(La ha + Lb hb) / 2``: the table's read
  is the bilinear value less that slack. Beyond a joint's tabled range the read is the edge's
  value less ``L`` times the distance to it.

**Cost.** Two levels: a coarse table everywhere (with points whose neighbourhood is certainly
farther than the table's cap filled from a bound, not evaluated), and the fine table evaluated
finely only where the coarse one reads less than ``band``; elsewhere the fine table holds the
coarse read (also a lower bound). Groups of B's spheres are pruned by their bounds.

**One joint apart.** Two bodies joined through one arm joint either touch at that joint (direct
neighbours: the joint is between them) or are a link and something fixed beyond the next link
(link 5 and a gripper on link 6). The solver has no motion that moves either apart: their
relative motion is that one joint. ``one_joint_check`` reports the least clearance of the
second kind over the joint's whole range and the gripper's, whether it changes by more than the
check's own error, and that error.
"""

from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import inspect
import io
import json
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import zipfile

import numpy as np

SQ3 = math.sqrt(3.0)
PAIR_BUDGET = 2_000_000            # sphere lookups evaluated at once
GROUP_BUDGET = 4_000_000           # node x group lookups at once
TABLE_VERSION = 1
GRADIENT_BASELINE = 0.010           # m: B's motion each way for the rows' direction


@dataclass(frozen=True)
class TableConfig:
    """How finely the tables are made (part of the cache key)."""

    slack_coarse: float = 0.005     # m, the coarse table's read below its nodes' bound
    slack_fine: float = 0.0015      # m, the fine table's
    band: float = 0.06              # m, fine evaluation where the coarse read is below this
    table_cap: float = 0.12         # m, values above it are only known to be above it
    coarse_g: float = 0.002
    coarse_s: float = 0.0005
    coarse_cap: float = 0.17
    fine_g: float = 0.001
    fine_s: float = 0.00025
    fine_cap: float = 0.09
    coarse_cell: float = 0.003
    coarse_spacing: float = 0.0015
    fine_cell: float = 0.0015
    fine_spacing: float = 0.0005
    coarse_group: float = 0.02
    fine_group: float = 0.01
    pad_coarse: float = 0.001       # m, most a gripper-moved point moves between drive samples
    pad_fine: float = 0.00025
    stride: int = 4                 # coarse points evaluated first, to skip the far ones
    extra: float = 0.05             # rad tabled beyond each limit of a joint with a range
    one_joint_step: float = math.radians(1.0)

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


# Coarse enough for unit tests of made-up robots (seconds, not minutes); never for an arm.
TEST_CONFIG = TableConfig(slack_coarse=0.006, slack_fine=0.003, band=0.06, table_cap=0.12,
                          coarse_g=0.004, coarse_s=0.0015, coarse_cap=0.17, fine_g=0.002,
                          fine_s=0.0006, fine_cap=0.1, coarse_cell=0.006, coarse_spacing=0.002,
                          fine_cell=0.003, fine_spacing=0.001, coarse_group=0.03,
                          fine_group=0.015, pad_coarse=0.0015, pad_fine=0.0006,
                          one_joint_step=math.radians(2.0))


# --- geometry helpers ---------------------------------------------------------------------

def _rot_batch(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    k = 1.0 - c
    out = np.empty((len(q), 3, 3))
    out[:, 0, 0] = c + x * x * k
    out[:, 0, 1] = x * y * k - z * s
    out[:, 0, 2] = x * z * k + y * s
    out[:, 1, 0] = y * x * k + z * s
    out[:, 1, 1] = c + y * y * k
    out[:, 1, 2] = y * z * k - x * s
    out[:, 2, 0] = z * x * k - y * s
    out[:, 2, 1] = z * y * k + x * s
    out[:, 2, 2] = c + z * z * k
    return out


def _motion(kind: str, axis: np.ndarray, value: float) -> np.ndarray:
    out = np.eye(4)
    if kind == 'prismatic':
        out[:3, 3] = axis * value
    elif kind in ('revolute', 'continuous'):
        out[:3, :3] = _rot_batch(np.asarray(axis, float), np.array([value]))[0]
    return out


@dataclass
class TailJoint:
    """One joint of B's path beyond the arm chain (fixed, or moved by a non-arm joint)."""

    kind: str
    origin: np.ndarray
    axis: np.ndarray
    mult: float = 0.0              # value = mult * drive + offset (moving joints)
    offset: float = 0.0


def tail_pose(tail: Sequence[TailJoint], drive: float) -> np.ndarray:
    pose = np.eye(4)
    for j in tail:
        pose = pose @ j.origin
        if j.kind != 'fixed':
            pose = pose @ _motion(j.kind, j.axis, j.mult * drive + j.offset)
    return pose


@dataclass
class PairSpec:
    """
    Two moving bodies with two arm joints (a, b) between them.

    B's pose in A's frame is ``s0 R(axis_a, qa) s1 R(axis_b, qb) s2 tail(drive)``.
    """

    inner: str
    outer: str
    joints: Tuple[int, int]
    s0: np.ndarray
    axis_a: np.ndarray
    s1: np.ndarray
    axis_b: np.ndarray
    s2: np.ndarray
    tail: List[TailJoint]
    limits: Tuple[Tuple[float, float], Tuple[float, float]]
    drive: Optional[Tuple[float, float]]      # the gripper's range when it moves B, else None
    tris_inner: np.ndarray
    tris_outer: np.ndarray


@dataclass
class OneJointSpec:
    """
    Two moving bodies with one arm joint between them that it does not join directly.

    B's pose in A's frame is ``s0 R(axis, q) s2 tail(drive)``.
    """

    inner: str
    outer: str
    joint: int
    s0: np.ndarray
    axis: np.ndarray
    s2: np.ndarray
    tail: List[TailJoint]
    limits: Tuple[float, float]
    drive: Optional[Tuple[float, float]]
    tris_inner: np.ndarray
    tris_outer: np.ndarray


# --- A's distance grid ----------------------------------------------------------------------

class Field:
    """Lower bound of the distance to A's collision surface, on a voxel grid (module docstring)."""

    UNIT = 1e-5                     # m per stored step (rounded down)

    def __init__(self, tris: np.ndarray, g: float, s: float, cap: float, chunk: int = 2000):
        from .keepout import sample_triangles
        verts = np.asarray(tris, float).reshape(-1, 3)
        lo = verts.min(0) - cap - 2.0 * g
        hi = verts.max(0) + cap + 2.0 * g
        shape = np.ceil((hi - lo) / g).astype(int) + 1
        occ = np.zeros(shape, dtype=bool)
        for i in range(0, len(tris), chunk):
            pts = sample_triangles(tris[i:i + chunk], s)
            idx = np.floor((pts - lo) / g + 0.5).astype(np.int64)
            occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        self.lo, self.g, self.s, self.cap, self.shape = lo, g, s, cap, shape
        window = int(math.ceil(cap / g)) + 1
        big = np.float32((window + 1) ** 2)
        d = np.where(occ, np.float32(0.0), big).astype(np.float32)
        del occ
        for axis in range(3):
            d = self._pass(d, axis, window, big)
        exact = np.minimum(np.sqrt(d) * g, cap)
        del d
        self.E = np.floor(np.maximum(exact - 1e-7, 0.0) / self.UNIT).astype(np.uint16)
        self.loss = s + g * SQ3 / 2.0

    @staticmethod
    def _pass(d, axis, window, big):
        d = np.moveaxis(d, axis, 0)
        out = d.copy()
        n = d.shape[0]
        for k in range(1, min(window, n - 1) + 1):
            kk = np.float32(k * k)
            np.minimum(out[k:], d[:-k] + kk, out=out[k:])
            np.minimum(out[:-k], d[k:] + kk, out=out[:-k])
        np.minimum(out, big, out=out)
        return np.moveaxis(out, 0, axis)

    def lower(self, p: np.ndarray) -> np.ndarray:
        """Lower bound of the distance from points ``p`` (... x 3, A's frame) to A, at most cap."""
        f = (p - self.lo) / self.g
        i = np.floor(f + 0.5).astype(np.int64)
        inside = np.all((i >= 0) & (i < self.shape), axis=-1)
        ic = np.clip(i, 0, self.shape - 1)
        e = self.E[ic[..., 0], ic[..., 1], ic[..., 2]].astype(float) * self.UNIT
        v = self.lo + ic * self.g
        val = e - self.loss - np.linalg.norm(p - v, axis=-1)
        return np.where(inside, np.minimum(val, self.cap), self.cap)

    @property
    def nbytes(self) -> int:
        return int(self.E.nbytes)


# --- B's cover ---------------------------------------------------------------------------

def _finish_cover(c: np.ndarray, r: np.ndarray, group: float) -> dict:
    from .keepout import _groups
    gi, gc, gr = _groups(c, r, group)
    order = np.argsort(gi, kind='stable')
    c, r, gi = c[order], r[order], gi[order]
    count = np.bincount(gi, minlength=len(gr))
    start = np.searchsorted(gi, np.arange(len(gr)))
    mx = int(count.max()) if len(count) else 0
    pad = np.full((len(gr), max(mx, 1)), -1, dtype=np.int64)
    for g, (s0, k) in enumerate(zip(start, count)):
        pad[g, :k] = np.arange(s0, s0 + k)
    centre = (c.min(0) + c.max(0)) / 2.0
    return {'c': c, 'r': r, 'gc': gc, 'gr': gr, 'start': start, 'count': count, 'pad': pad,
            'bc': centre, 'br': float((np.linalg.norm(c - centre, axis=1) + r).max())}


def cover(tris: np.ndarray, cell: float, spacing: float, group: float) -> dict:
    from .keepout import sphere_cover
    c, r = sphere_cover(tris, cell, spacing)
    return _finish_cover(c, r, group)


def drive_rate(tail: Sequence[TailJoint], reach: float) -> float:
    """
    Upper bound of a B point's speed per unit of drive (m per unit), from the description.

    A point at most ``reach`` from B's origin; for each moving joint of the tail its speed is at
    most ``|mult|`` times the point's distance from that joint, bounded by the offsets of the
    joints after it plus ``reach`` (and ``|mult|`` for a prismatic joint).
    """
    rate = 0.0
    for k, j in enumerate(tail):
        if j.kind == 'fixed' or j.mult == 0.0:
            continue
        if j.kind == 'prismatic':
            rate += abs(j.mult)
            continue
        after = sum(float(np.linalg.norm(t.origin[:3, 3])) for t in tail[k + 1:])
        rate += abs(j.mult) * (after + reach)
    return rate


def swept_cover(tris: np.ndarray, s2: np.ndarray, tail: Sequence[TailJoint],
                drive: Tuple[float, float], cell: float, s: float, pad: float, group: float,
                chunk: int = 2000) -> Tuple[dict, float]:
    """
    Cover of B over the whole drive range, in the frame after joint b (module docstring).

    Returns (cover, rate): drive samples are close enough that no surface point moves more
    than ``pad`` from one to the nearest.
    """
    from .keepout import sample_triangles
    verts = np.asarray(tris, float).reshape(-1, 3)
    reach = float(np.linalg.norm(verts, axis=1).max())
    rate = drive_rate(tail, reach)
    lo, hi = drive
    n = max(2, int(math.ceil((hi - lo) * rate / (2.0 * pad))) + 1)
    step = (hi - lo) / (n - 1)
    keys = []
    for g in np.linspace(lo, hi, n):
        pose = s2 @ tail_pose(tail, float(g))
        for i in range(0, len(tris), chunk):
            pts = sample_triangles(tris[i:i + chunk], s) @ pose[:3, :3].T + pose[:3, 3]
            keys.append(np.unique(np.floor(pts / cell).astype(np.int64), axis=0))
        if len(keys) > 64:
            keys = [np.unique(np.concatenate(keys), axis=0)]
    keys = np.unique(np.concatenate(keys), axis=0)
    c = (keys + 0.5) * cell
    r = np.full(len(c), cell * SQ3 / 2.0 + s + rate * step / 2.0)
    return _finish_cover(c, r, group), rate


# --- evaluation ----------------------------------------------------------------------------

def node_values(F: Field, cov: dict, R: np.ndarray, t: np.ndarray, ceiling: float) -> np.ndarray:
    """
    Least lower-bounded clearance of B's spheres posed (R, t) in A's frame, at most ``ceiling``.

    ``F.lower`` is 1-Lipschitz up to its nearest-voxel rounding (``g sqrt3``), so every sphere
    of a group (or of the body) reads at least the group's centre reading less its radius and
    that rounding: a body or group whose bound is at or above the running value is skipped.
    """
    K = len(R)
    out = np.full(K, float(ceiling))
    if K == 0:
        return out
    sl = F.g * SQ3
    bc = np.einsum('kij,j->ki', R, cov['bc']) + t
    live = np.nonzero(F.lower(bc) - cov['br'] - sl < ceiling)[0]
    if not len(live):
        return out
    step = max(1, GROUP_BUDGET // max(1, len(cov['gr'])))
    for k0 in range(0, len(live), step):
        idx = live[k0:k0 + step]
        out[idx] = _node_values(F, cov, R[idx], t[idx], ceiling, sl)
    return out


def _node_values(F, cov, R, t, ceiling, sl):
    K = len(R)
    gcb = np.einsum('kij,gj->kgi', R, cov['gc']) + t[:, None, :]
    glb = F.lower(gcb) - cov['gr'][None] - sl
    best = glb.argmin(1)
    members = cov['pad'][best]
    valid = members >= 0
    safe = np.where(valid, members, 0)
    p = np.einsum('kij,kmj->kmi', R, cov['c'][safe]) + t[:, None, :]
    v = F.lower(p) - cov['r'][safe]
    out = np.minimum(np.where(valid, v, np.inf).min(1), ceiling)
    glb[np.arange(K), best] = np.inf
    k_i, g_i = np.nonzero(glb < out[:, None])
    if not len(k_i):
        return out
    n = cov['count'][g_i]
    ends = np.cumsum(n)
    lo = 0
    while lo < len(n):
        base = ends[lo - 1] if lo else 0
        hi = max(int(np.searchsorted(ends, base + PAIR_BUDGET, side='right')), lo + 1)
        kk_, gg_ = k_i[lo:hi], g_i[lo:hi]
        nn = cov['count'][gg_]
        rep = np.repeat(np.arange(len(kk_)), nn)
        ib = cov['start'][gg_][rep] + (np.arange(int(nn.sum()))
                                       - np.repeat(np.cumsum(nn) - nn, nn))
        kk = kk_[rep]
        c = cov['c'][ib]
        pp = np.empty_like(c)
        for i in range(3):
            pp[:, i] = (R[kk, i, :] * c).sum(1) + t[kk, i]
        np.minimum.at(out, kk, F.lower(pp) - cov['r'][ib])
        lo = hi
    return out


# --- the table ----------------------------------------------------------------------------

class Table:
    """Node values (lower bounds) on a regular grid over two joints; read = bilinear - slack."""

    def __init__(self, lo, h, n, periodic, values, slack, lever, cap=math.inf):
        self.cap = float(cap)                       # values at or above it: only known above it
        self.lo = np.asarray(lo, float)
        self.h = np.asarray(h, float)
        self.n = np.asarray(n, int)
        self.periodic = np.asarray(periodic, bool)
        self.v = np.asarray(values, float)
        self.slack = float(slack)
        self.lever = np.asarray(lever, float)       # (La, Lb)

    def _coord(self, q, i):
        x = (q - self.lo[i]) / self.h[i]
        if self.periodic[i]:
            x = np.mod(x, self.n[i])
            j = np.minimum(np.floor(x).astype(int), self.n[i] - 1)
            return j, (j + 1) % self.n[i], x - j, np.zeros_like(x)
        top = self.n[i] - 1
        over = (np.maximum(x - top, 0.0) + np.maximum(-x, 0.0)) * self.h[i]
        x = np.clip(x, 0.0, top - 1e-12)
        j = np.floor(x).astype(int)
        return j, j + 1, x - j, over

    def read(self, qa, qb):
        """Return (lower bound, d/dqa, d/dqb); beyond the tabled range, less lever x overshoot."""
        a0, a1, fa, oa = self._coord(np.asarray(qa, float), 0)
        b0, b1, fb, ob = self._coord(np.asarray(qb, float), 1)
        v = self.v
        v00, v01, v10, v11 = v[a0, b0], v[a0, b1], v[a1, b0], v[a1, b1]
        val = (v00 * (1 - fa) * (1 - fb) + v10 * fa * (1 - fb) + v01 * (1 - fa) * fb
               + v11 * fa * fb)
        ga = ((v10 - v00) * (1 - fb) + (v11 - v01) * fb) / self.h[0]
        gb = ((v01 - v00) * (1 - fa) + (v11 - v10) * fa) / self.h[1]
        return val - self.slack - self.lever[0] * oa - self.lever[1] * ob, ga, gb

    def to_arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        return {f'{prefix}lo': self.lo, f'{prefix}h': self.h, f'{prefix}n': self.n,
                f'{prefix}periodic': self.periodic, f'{prefix}v': self.v.astype(np.float32),
                f'{prefix}slack': np.array(self.slack), f'{prefix}lever': self.lever,
                f'{prefix}cap': np.array(self.cap)}

    @staticmethod
    def from_arrays(z, prefix: str) -> 'Table':
        return Table(z[f'{prefix}lo'], z[f'{prefix}h'], z[f'{prefix}n'], z[f'{prefix}periodic'],
                     z[f'{prefix}v'].astype(float), float(z[f'{prefix}slack']),
                     z[f'{prefix}lever'], float(z[f'{prefix}cap']))


def lever_arms(spec: PairSpec, cov: dict, s2: np.ndarray, tail_drive: Optional[float]):
    """
    Upper bounds (La, Lb) of B's points' distance from axis a (any qb) and from axis b.

    In the frame after s0 (axis a through its origin), a point y of the frame after joint b
    sits at ``s1 (R(b) y)``; with y = y_par + y_perp about axis b, its distance from axis a is at
    most ``|P_a(t1 + R1 y_par)| + |y_perp|`` for every qb. A sphere adds its radius.
    """
    if tail_drive is None:
        c, r = cov['c'], cov['r']
    else:
        pose = s2 @ tail_pose(spec.tail, tail_drive)
        c = cov['c'] @ pose[:3, :3].T + pose[:3, 3]
        r = cov['r']
    ab = spec.axis_b / np.linalg.norm(spec.axis_b)
    par = c @ ab
    perp = np.linalg.norm(c - par[:, None] * ab, axis=1)
    lb = float((perp + r).max())
    aa = spec.axis_a / np.linalg.norm(spec.axis_a)
    x = spec.s1[:3, 3] + (par[:, None] * ab) @ spec.s1[:3, :3].T
    pa = np.linalg.norm(x - (x @ aa)[:, None] * aa, axis=1)
    la = float((pa + perp + r).max())
    return la, lb


def _axes(limits, la, lb, slack, extra):
    out = []
    for (lo, hi), lever in zip(limits, (la, lb)):
        h = slack / lever
        if hi - lo >= 2.0 * math.pi - 1e-9:
            n = int(math.ceil(2.0 * math.pi / h))
            out.append((-math.pi, 2.0 * math.pi / n, n, True))
        else:
            span = hi - lo + 2.0 * extra
            n = int(math.ceil(span / h)) + 1
            out.append((lo - extra, span / (n - 1), n, False))
    return out


def _nodes(ax):
    qa = ax[0][0] + ax[0][1] * np.arange(ax[0][2])
    qb = ax[1][0] + ax[1][1] * np.arange(ax[1][2])
    QA, QB = np.meshgrid(qa, qb, indexing='ij')
    return QA.ravel(), QB.ravel()


def _relative(spec: PairSpec, s2tail: np.ndarray, qa: np.ndarray, qb: np.ndarray):
    Ra, Rb = _rot_batch(spec.axis_a, qa), _rot_batch(spec.axis_b, qb)
    R = np.einsum('ij,kjl,lm,kmn,np->kip', spec.s0[:3, :3], Ra, spec.s1[:3, :3], Rb,
                  s2tail[:3, :3])
    t = np.einsum('kij,j->ki', Rb, s2tail[:3, 3])
    t = t @ spec.s1[:3, :3].T + spec.s1[:3, 3]
    t = np.einsum('kij,kj->ki', Ra, t)
    t = t @ spec.s0[:3, :3].T + spec.s0[:3, 3]
    return R, t


def _evaluate(F, cov, spec, s2tail, QA, QB, ceiling, batch=2000):
    out = np.empty(len(QA))
    for i in range(0, len(QA), batch):
        R, t = _relative(spec, s2tail, QA[i:i + batch], QB[i:i + batch])
        out[i:i + batch] = node_values(F, cov, R, t, ceiling)
    return out


def build_table(spec: PairSpec, cfg: TableConfig, coarse: Field,
                fine: Field) -> Tuple[Table, dict]:
    """Build one pair's table (module docstring, "Cost"); returns it and a few counts."""
    if spec.drive is not None:
        cov_c, _ = swept_cover(spec.tris_outer, spec.s2, spec.tail, spec.drive, cfg.coarse_cell,
                               cfg.coarse_spacing, cfg.pad_coarse, cfg.coarse_group)
        cov_f, _ = swept_cover(spec.tris_outer, spec.s2, spec.tail, spec.drive, cfg.fine_cell,
                               cfg.fine_spacing, cfg.pad_fine, cfg.fine_group)
        s2tail, drive_for_lever = np.eye(4), None
    else:
        s2tail = spec.s2 @ tail_pose(spec.tail, 0.0)
        cov_c = cover(spec.tris_outer, cfg.coarse_cell, cfg.coarse_spacing, cfg.coarse_group)
        cov_f = cover(spec.tris_outer, cfg.fine_cell, cfg.fine_spacing, cfg.fine_group)
        drive_for_lever = 0.0
    la, lb = lever_arms(spec, cov_c, spec.s2, drive_for_lever)
    axc = _axes(spec.limits, la, lb, cfg.slack_coarse, cfg.extra)
    na, nb = axc[0][2], axc[1][2]
    stride = max(1, int(cfg.stride))
    ia, ib = np.arange(0, na, stride), np.arange(0, nb, stride)
    if not axc[0][3] and ia[-1] != na - 1:
        ia = np.append(ia, na - 1)
    if not axc[1][3] and ib[-1] != nb - 1:
        ib = np.append(ib, nb - 1)

    def nearest(idx, n, periodic):
        i = np.arange(n)
        d = np.abs(i[:, None] - idx[None, :])
        if periodic:
            d = np.minimum(d, n - d)
        k = d.argmin(1)
        return k, d[i, k]
    ka, da = nearest(ia, na, axc[0][3])
    kb, db = nearest(ib, nb, axc[1][3])
    reach = la * axc[0][1] * int(da.max()) + lb * axc[1][1] * int(db.max())
    QA, QB = np.meshgrid(axc[0][0] + axc[0][1] * ia, axc[1][0] + axc[1][1] * ib, indexing='ij')
    vs = _evaluate(coarse, cov_c, spec, s2tail, QA.ravel(), QB.ravel(),
                   cfg.table_cap + reach + 1e-6).reshape(len(ia), len(ib))
    bound = vs[ka][:, kb] - (la * axc[0][1] * da)[:, None] - (lb * axc[1][1] * db)[None, :]
    far = bound >= cfg.table_cap
    vc = np.where(far, bound, 0.0).ravel()
    QA, QB = _nodes(axc)
    todo = ~far.ravel()
    vc[todo] = _evaluate(coarse, cov_c, spec, s2tail, QA[todo], QB[todo], cfg.table_cap)
    slack_c = (la * axc[0][1] + lb * axc[1][1]) / 2.0
    tc = Table([a[0] for a in axc], [a[1] for a in axc], [a[2] for a in axc],
               [a[3] for a in axc], vc.reshape(na, nb), slack_c, (la, lb))
    axf = _axes(spec.limits, la, lb, cfg.slack_fine, cfg.extra)
    QA, QB = _nodes(axf)
    vf = tc.read(QA, QB)[0]
    inb = vf < cfg.band
    if inb.any():
        # ceiling above the band by the coarse table's own loss: no flat stretch at the band
        fine_ceiling = min(fine.cap - float(cov_f['gr'].max()) - fine.g * SQ3,
                           cfg.band + 2.0 * cfg.slack_coarse + coarse.loss
                           + float(cov_c['r'].max()))
        vf[inb] = np.maximum(vf[inb], _evaluate(fine, cov_f, spec, s2tail, QA[inb], QB[inb],
                                                fine_ceiling))
    slack_f = (la * axf[0][1] + lb * axf[1][1]) / 2.0
    values = np.floor((vf - 1e-6) / 1e-6) * 1e-6        # never raised by the float32 cast
    tf = Table([a[0] for a in axf], [a[1] for a in axf], [a[2] for a in axf],
               [a[3] for a in axf], values.astype(np.float32).astype(float).reshape(
                   axf[0][2], axf[1][2]), slack_f, (la, lb), cfg.table_cap)
    # How far the read may be below the truth inside the band: A's grid on both sides of the
    # nearest-voxel step, B's cover and the slack (for the selection of pairs held at work).
    error = 2.0 * fine.loss + fine.g * SQ3 + 2.0 * float(cov_f['r'].max()) + slack_f
    stats = {'error': float(error), 'coarse': int(na * nb), 'coarse_far': int(far.sum()),
             'fine': int(len(vf)), 'fine_evaluated': int(inb.sum()),
             'slack_mm': round(slack_f * 1000.0, 3), 'lever_m': [round(la, 4), round(lb, 4)],
             'spheres': [int(len(cov_c['r'])), int(len(cov_f['r']))]}
    return tf, stats


def one_joint_check(spec: OneJointSpec, cfg: TableConfig, fine: Field) -> dict:
    """
    Least and largest lower-bounded clearance over the joint's range (and the gripper's).

    Returns ``low``, ``high`` (m, lower bounds at the samples less the slack between them) and
    ``error`` (m): how far the check may read below the truth (A's grid, B's cover, the step).
    """
    if spec.drive is not None:
        cov, _ = swept_cover(spec.tris_outer, spec.s2, spec.tail, spec.drive, cfg.fine_cell,
                             cfg.fine_spacing, cfg.pad_fine, cfg.fine_group)
        s2tail = np.eye(4)
    else:
        cov = cover(spec.tris_outer, cfg.fine_cell, cfg.fine_spacing, cfg.fine_group)
        s2tail = spec.s2 @ tail_pose(spec.tail, 0.0)
    lo, hi = spec.limits
    if hi - lo >= 2.0 * math.pi - 1e-9:
        lo, hi = -math.pi, math.pi
    n = max(2, int(math.ceil((hi - lo) / cfg.one_joint_step)) + 1)
    q = np.linspace(lo, hi, n)
    h = (hi - lo) / (n - 1)
    ax = spec.axis / np.linalg.norm(spec.axis)
    c = cov['c'] @ s2tail[:3, :3].T + s2tail[:3, 3]
    lever = float((np.linalg.norm(c - (c @ ax)[:, None] * ax, axis=1) + cov['r']).max())
    R = np.einsum('ij,kjl,lm->kim', spec.s0[:3, :3], _rot_batch(ax, q), s2tail[:3, :3])
    t = np.einsum('kij,j->ki', _rot_batch(ax, q), s2tail[:3, 3]) @ spec.s0[:3, :3].T \
        + spec.s0[:3, 3]
    vals = node_values(fine, cov, R, t, fine.cap - float(cov['gr'].max()) - fine.g * SQ3)
    step = lever * h / 2.0
    error = fine.loss + fine.g * SQ3 / 2.0 + 2.0 * float(cov['r'].max()) + step
    capped = bool(vals.max() >= fine.cap - float(cov['gr'].max()) - fine.g * SQ3 - 1e-9)
    return {'low': float(vals.min() - step), 'high': float(vals.max()), 'error': float(error),
            'capped': capped}


# --- per-cycle read of every table at once ---------------------------------------------------

class TableStack:
    """
    The tables of a model, read together each cycle (one gather for all pairs).

    The value is the table's read (a lower bound everywhere); the rows' direction is a central
    difference of the same read over ``GRADIENT_BASELINE`` of B's motion each way per joint: the
    one-cell bilinear gradient follows the grid's nearest-voxel ripple and is often flat or
    wrong-signed near the margin, the wide one follows the true distance.
    """

    def __init__(self, tables: Sequence[Table], joints: Sequence[Tuple[int, int]],
                 _shared=None, _index=None):
        if _shared is None:
            sizes = [t.v.size for t in tables]
            flat = (np.concatenate([np.asarray(t.v, np.float32).ravel() for t in tables])
                    if tables else np.zeros(0, np.float32))
            offset = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64) \
                if tables else np.zeros(0, np.int64)

            def arr(f, shape):
                return (np.array([f(t) for t in tables], dtype=float).reshape(shape)
                        if tables else np.zeros(shape))

            _shared = {
                'flat': flat, 'offset': offset,
                'lo': arr(lambda t: t.lo, (-1, 2)), 'h': arr(lambda t: t.h, (-1, 2)),
                'n': arr(lambda t: t.n, (-1, 2)).astype(np.int64),
                'per': arr(lambda t: t.periodic, (-1, 2)).astype(bool),
                'slack': arr(lambda t: t.slack, (-1,)),
                'lever': arr(lambda t: t.lever, (-1, 2)),
                'cap': arr(lambda t: t.cap, (-1,)),
                'joints': np.array(joints, dtype=np.int64).reshape(-1, 2)}
            _index = np.arange(len(tables))
        self._shared = _shared
        self.index = np.asarray(_index, dtype=np.int64)
        sh, i = _shared, self.index
        self.offset, self.lo, self.h, self.n = sh['offset'][i], sh['lo'][i], sh['h'][i], sh['n'][i]
        self.per, self.slack, self.lever = sh['per'][i], sh['slack'][i], sh['lever'][i]
        self.cap = sh['cap'][i]
        self.ja, self.jb = sh['joints'][i, 0], sh['joints'][i, 1]
        self.delta = GRADIENT_BASELINE / np.maximum(self.lever, 1e-9)
        self._steps = np.array([[0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0]])

    def __len__(self) -> int:
        return len(self.index)

    def select(self, members: Sequence[int]) -> 'TableStack':
        """Return the stack of ``members`` (positions in this stack); the values are shared."""
        return TableStack((), (), self._shared, self.index[np.asarray(members, dtype=np.int64)])

    def _axis(self, q, i):
        lo, h = self.lo[:, i, None], self.h[:, i, None]
        n, per = self.n[:, i, None], self.per[:, i, None]
        x = (q - lo) / h
        xp = np.mod(x, n)
        jp = np.minimum(np.floor(xp), n - 1)
        top = n - 1
        xc = np.clip(x, 0.0, top - 1e-12)
        jc = np.floor(xc)
        j0 = np.where(per, jp, jc)
        f = np.where(per, xp - jp, xc - jc)
        j1 = np.where(per, np.mod(j0 + 1, n), j0 + 1)
        over = np.where(per, 0.0, (np.maximum(x - top, 0.0) + np.maximum(-x, 0.0)) * h)
        return j0.astype(np.int64), j1.astype(np.int64), f, over

    def _read(self, qa, qb):
        a0, a1, fa, oa = self._axis(qa, 0)
        b0, b1, fb, ob = self._axis(qb, 1)
        flat, base, nb = self._shared['flat'], self.offset[:, None], self.n[:, 1, None]
        v00, v01 = flat[base + a0 * nb + b0], flat[base + a0 * nb + b1]
        v10, v11 = flat[base + a1 * nb + b0], flat[base + a1 * nb + b1]
        val = (v00 * (1 - fa) * (1 - fb) + v10 * fa * (1 - fb) + v01 * (1 - fa) * fb
               + v11 * fa * fb)
        return (val - self.slack[:, None] - self.lever[:, 0, None] * oa
                - self.lever[:, 1, None] * ob)

    def read(self, q: np.ndarray):
        """Return (lower bounds k, d/dq_a k, d/dq_b k) of every table at the arm posture ``q``."""
        q = np.asarray(q, dtype=float)
        qa = q[self.ja][:, None] + self.delta[:, 0, None] * self._steps[0]
        qb = q[self.jb][:, None] + self.delta[:, 1, None] * self._steps[1]
        v = self._read(qa, qb)
        ga = (v[:, 1] - v[:, 2]) / (2.0 * self.delta[:, 0])
        gb = (v[:, 3] - v[:, 4]) / (2.0 * self.delta[:, 1])
        return v[:, 0], ga, gb

    @property
    def nbytes(self) -> int:
        return int(self._shared['flat'].nbytes)


# --- the pairs of a description --------------------------------------------------------------

@dataclass
class Plan:
    """
    The self pairs of one description that the tables or the one-joint check cover.

    Tables cover two joints apart; the one-joint check covers one joint apart, not direct
    neighbours. The cache keys of the pairs are kept as well.
    """

    pairs: List[PairSpec]
    one: List[OneJointSpec]
    unsupported: List[Tuple[str, str, str]]          # (outer, inner, why): left on spheres
    keys: Dict[Tuple[str, str], str]                 # (outer, inner) -> cache key
    cfg: TableConfig
    parse_seconds: float = 0.0


def _source_limits(urdf: str, joint: str) -> Tuple[float, float]:
    import xml.etree.ElementTree as ElementTree
    for element in ElementTree.fromstring(urdf).findall('joint'):
        if element.get('name') == joint:
            limit = element.find('limit')
            if limit is None or limit.get('lower') is None or limit.get('upper') is None:
                break
            return float(limit.get('lower')), float(limit.get('upper'))
    raise ValueError(f'joint {joint} has no position limits')


def _round(a) -> bytes:
    return np.round(np.asarray(a, dtype=float), 9).tobytes()


def _code_hash() -> str:
    """Hash of the code the tables' values depend on (not of the per-cycle read)."""
    from . import keepout
    h = hashlib.sha256(f'tables {TABLE_VERSION}'.encode())
    for item in (keepout.sample_triangles, keepout._unique_rows, keepout.sphere_cover,
                 keepout._groups, _rot_batch, _motion, tail_pose, Field, _finish_cover, cover,
                 drive_rate, swept_cover, node_values, _node_values, lever_arms, _axes, _nodes,
                 _relative, _evaluate, Table, build_table, one_joint_check):
        h.update(inspect.getsource(item).encode())
    return h.hexdigest()


def _key(kind: str, spec, cfg: TableConfig, code: str) -> str:
    h = hashlib.sha256(f'{kind}|{cfg.key()}|{code}'.encode())
    h.update(_round(spec.tris_inner))
    h.update(_round(spec.tris_outer))
    if kind == 'pair':
        parts = (spec.s0, spec.axis_a, spec.s1, spec.axis_b, spec.s2, spec.limits)
    else:
        parts = (spec.s0, spec.axis, spec.s2, spec.limits)
    for part in parts:
        h.update(_round(part))
    for j in spec.tail:
        h.update(f'{j.kind}|{j.mult}|{j.offset}'.encode())
        h.update(_round(j.origin))
        h.update(_round(j.axis))
    h.update(repr(spec.drive).encode())
    return h.hexdigest()


def make_plan(urdf: str, resolve: Callable[[str], bytes], base: str, tip: str,
              cfg: TableConfig = TableConfig()) -> Plan:
    """Return the table and one-joint pairs of a description (no table is built)."""
    import time
    from . import keepout
    started = time.perf_counter()
    parts = keepout.build(urdf, resolve, base, tip, 1.0, _parts_only=True)
    segments = parts['segments']
    bodies, tris = parts['bodies'], parts['tris']
    limits = parts['limits']
    pairs, one, unsupported = [], [], []
    code = _code_hash()
    keys = {}
    for b_body in bodies:
        for a_body in bodies:
            gap = b_body.depth - a_body.depth
            if a_body.depth < 1 or gap not in (1, 2):
                continue
            if gap == 1 and (b_body.name, a_body.name) in parts['adjacent']:
                continue
            if b_body.prefix < a_body.prefix:
                continue
            if any(j.kind != 'fixed' for j in a_body.tail):
                unsupported.append((b_body.name, a_body.name,
                                    'the inner body moves with a joint off the arm chain'))
                continue
            tail_a = np.eye(4)
            for j in a_body.tail:
                tail_a = tail_a @ j.origin
            pose = np.linalg.inv(tail_a)
            cuts = []
            for k in range(a_body.prefix, b_body.prefix):
                seg = segments[k]
                pose = pose @ seg.origin
                if seg.axis is not None:
                    cuts.append((pose, np.asarray(seg.axis, float), parts['seg_joint'][k]))
                    pose = np.eye(4)
            tail, sources = [], set()
            for j in b_body.tail:
                if j.kind == 'fixed':
                    tail.append(TailJoint('fixed', j.origin, np.asarray(j.axis, float)))
                else:
                    source, mult, offset = parts['extra'][j.name]
                    sources.add(source)
                    tail.append(TailJoint(j.kind, j.origin, np.asarray(j.axis, float), mult,
                                          offset))
            if len(sources) > 1:
                unsupported.append((b_body.name, a_body.name,
                                    'the outer body moves with more than one joint off the arm'))
                continue
            drive = _source_limits(urdf, sources.pop()) if sources else None
            ja = [c[2] for c in cuts]
            jl = [limits[j] for j in ja]
            if gap == 2:
                spec = PairSpec(a_body.name, b_body.name, (ja[0], ja[1]), cuts[0][0], cuts[0][1],
                                cuts[1][0], cuts[1][1], pose, tail, (jl[0], jl[1]), drive,
                                tris[a_body.name], tris[b_body.name])
                pairs.append(spec)
                keys[(b_body.name, a_body.name)] = _key('pair', spec, cfg, code)
            else:
                spec = OneJointSpec(a_body.name, b_body.name, ja[0], cuts[0][0], cuts[0][1],
                                    pose, tail, jl[0], drive, tris[a_body.name],
                                    tris[b_body.name])
                one.append(spec)
                keys[(b_body.name, a_body.name)] = _key('one', spec, cfg, code)
    return Plan(pairs, one, unsupported, keys, cfg, time.perf_counter() - started)


# --- the cache -------------------------------------------------------------------------------

def cache_directory() -> str:
    """Return the user's cache directory for the tables (outside any checkout)."""
    root = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(root, 'ranger_xarm_teleop', 'keepout_tables')


@dataclass
class TableEntry:
    inner: str
    outer: str
    joints: Tuple[int, int]
    table: Table
    error: float
    stats: dict = field(default_factory=dict)


@dataclass
class OneJointEntry:
    inner: str
    outer: str
    joint: int
    low: float
    high: float
    error: float
    capped: bool


@dataclass
class Loaded:
    """Every entry of a plan, read from the cache and checked against its key."""

    tables: Dict[Tuple[str, str], TableEntry]
    one: Dict[Tuple[str, str], OneJointEntry]
    unsupported: List[Tuple[str, str, str]]
    nbytes: int


def _path(cache: str, key: str) -> str:
    return os.path.join(cache, key + '.npz')


def _write(cache: str, key: str, arrays: dict) -> None:
    """Write one entry atomically (a reader sees the whole entry or none)."""
    os.makedirs(cache, exist_ok=True)
    buffer = io.BytesIO()
    np.savez(buffer, key=np.array(key), **arrays)
    tmp = os.path.join(cache, f'.{key}.{os.getpid()}.tmp')
    with open(tmp, 'wb') as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, _path(cache, key))


def _read(cache: str, key: str):
    try:
        with np.load(_path(cache, key), allow_pickle=False) as z:
            if str(z['key']) != key:
                return None
            return {name: z[name] for name in z.files}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        # Absent, unreadable or damaged: the entry is built again, never used.
        return None


def _entry(plan: Plan, cache: str, spec, kind: str):
    key = plan.keys[(spec.outer, spec.inner)]
    z = _read(cache, key)
    if z is None or str(z.get('kind')) != kind:
        return None
    try:
        if kind == 'pair':
            table = Table.from_arrays(z, 't_')
            if (table.v.shape != tuple(int(n) for n in table.n) or not np.all(np.isfinite(table.v))
                    or not np.all(table.h > 0) or not np.all(np.isfinite(table.lever))):
                return None
            return TableEntry(spec.inner, spec.outer, spec.joints, table, float(z['error']),
                              json.loads(str(z['stats'])))
        values = (float(z['low']), float(z['high']), float(z['error']))
        if not all(math.isfinite(v) for v in values):
            return None
        return OneJointEntry(spec.inner, spec.outer, spec.joint, *values, bool(z['capped']))
    except (KeyError, ValueError, TypeError):
        return None


def missing(plan: Plan, cache: str) -> List[Tuple[str, str]]:
    """Return the (outer, inner) pairs whose entry is absent or does not pass its check."""
    out = []
    for kind, specs in (('pair', plan.pairs), ('one', plan.one)):
        for spec in specs:
            if _entry(plan, cache, spec, kind) is None:
                out.append((spec.outer, spec.inner))
    return out


def load(plan: Plan, cache: str) -> Optional[Loaded]:
    """Return every entry of ``plan``, or None if any is absent or fails its check."""
    tables, one, nbytes = {}, {}, 0
    for spec in plan.pairs:
        entry = _entry(plan, cache, spec, 'pair')
        if entry is None:
            return None
        tables[(spec.outer, spec.inner)] = entry
        nbytes += entry.table.v.size * 4
    for spec in plan.one:
        entry = _entry(plan, cache, spec, 'one')
        if entry is None:
            return None
        one[(spec.outer, spec.inner)] = entry
    return Loaded(tables, one, list(plan.unsupported), nbytes)


# --- the build (a separate process: keepout_build) -------------------------------------------

_POOL_PLAN: Optional[Plan] = None
_POOL_FIELDS: dict = {}


def _fields(spec, levels):
    name = spec.inner
    have = _POOL_FIELDS.get('name')
    if have != name or not set(levels) <= set(_POOL_FIELDS):
        if have != name:
            _POOL_FIELDS.clear()
            _POOL_FIELDS['name'] = name
        cfg = _POOL_PLAN.cfg
        for level in levels:
            if level not in _POOL_FIELDS:
                if level == 'coarse':
                    _POOL_FIELDS[level] = Field(spec.tris_inner, cfg.coarse_g, cfg.coarse_s,
                                                cfg.coarse_cap)
                else:
                    _POOL_FIELDS[level] = Field(spec.tris_inner, cfg.fine_g, cfg.fine_s,
                                                cfg.fine_cap)
    return [_POOL_FIELDS[level] for level in levels]


def _work(item):
    import time
    kind, index, cache = item
    started = time.perf_counter()
    plan = _POOL_PLAN
    if kind == 'pair':
        spec = plan.pairs[index]
        coarse, fine = _fields(spec, ('coarse', 'fine'))
        table, stats = build_table(spec, plan.cfg, coarse, fine)
        arrays = dict(kind=np.array('pair'), error=np.array(stats['error']),
                      stats=np.array(json.dumps(stats)), **table.to_arrays('t_'))
    else:
        spec = plan.one[index]
        (fine,) = _fields(spec, ('fine',))
        result = one_joint_check(spec, plan.cfg, fine)
        arrays = {'kind': np.array('one'), 'low': np.array(result['low']),
                  'high': np.array(result['high']), 'error': np.array(result['error']),
                  'capped': np.array(result['capped'])}
    _write(cache, plan.keys[(spec.outer, spec.inner)], arrays)
    return f'{spec.outer} - {spec.inner}', time.perf_counter() - started


def _child_init():
    _die_with_parent()
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass


def _die_with_parent():
    """Ask the kernel to stop this process when its parent dies (Linux; else nothing)."""
    try:
        import ctypes
        import signal
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, signal.SIGTERM)          # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


class BuildLock:
    """An exclusive lock on the cache directory (one builder at a time, across processes)."""

    def __init__(self, cache: str):
        os.makedirs(cache, exist_ok=True)
        self.handle = open(os.path.join(cache, 'build.lock'), 'a')

    def __enter__(self):
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def build_missing(plan: Plan, cache: str, workers: int,
                  progress: Callable[[str], None] = print) -> int:
    """Build the absent entries under the cache lock; return how many were built."""
    import multiprocessing
    global _POOL_PLAN
    with BuildLock(cache):
        todo = set(missing(plan, cache))     # again under the lock: another builder may have run
        items = [('pair', i, cache) for i, s in enumerate(plan.pairs)
                 if (s.outer, s.inner) in todo]
        items += [('one', i, cache) for i, s in enumerate(plan.one) if (s.outer, s.inner) in todo]
        name = (lambda it: (plan.pairs if it[0] == 'pair' else plan.one)[it[1]].inner)
        items.sort(key=name)
        if not items:
            return 0
        _POOL_PLAN = plan
        progress(f'building {len(items)} entries on {workers} workers')
        done = 0
        if workers <= 1:
            for item in items:
                pair, took = _work(item)
                done += 1
                progress(f'{done}/{len(items)} {pair} {took:.0f} s')
        else:
            context = multiprocessing.get_context('fork')
            with context.Pool(workers, initializer=_child_init) as pool:
                for pair, took in pool.imap_unordered(_work, items):
                    done += 1
                    progress(f'{done}/{len(items)} {pair} {took:.0f} s')
        _POOL_FIELDS.clear()
        return done
