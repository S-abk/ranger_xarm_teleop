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
A brute-force reference of the keep-out model's per-cycle search (tests only).

Every sphere against every shape (a fixed mesh's every member sphere) and every tool sphere
against every arm sphere its pair allows, with nothing pruned: what ``Model._compute`` must
list within a limit.
"""

import math

import numpy as np


def brute_compute(model, frames, limit):
    """Return ``Model._compute``'s tuple, found by checking every sphere pair."""
    rot, pos = model._poses(frames)
    body = model._sphere_body
    world = np.einsum('pij,pj->pi', rot[body], model._sphere_centre) + pos[body]
    radius = model._sphere_radius
    units = model.units
    parts = []
    if len(units.link) and len(body):
        s, u = np.nonzero(model._allowed_fixed[body][:, units.link])
        plain = units.mem_count[u] == 0
        d, n = units.pair_distance(world[s[plain]], u[plain])
        parts.append((s[plain], u[plain], d - radius[s[plain]], n))
        for k in np.nonzero(units.mem_count > 0)[0]:
            ss = s[u == k]
            first, count = units.mem_start[k], units.mem_count[k]
            mem = np.arange(first, first + count)
            ss, mem = np.repeat(ss, count), np.tile(mem, len(ss))
            diff = world[ss] - units.mem_pos[mem]
            norm = np.linalg.norm(diff, axis=1)
            parts.append((ss, np.full(len(ss), k), norm - units.mem_rad[mem] - radius[ss],
                          diff / np.maximum(norm, 1e-12)[:, None]))
    out_pair, out_d, out_n, out_p, out_o = [], [], [], [], []
    for s, u, d, n in parts:
        out_pair.append(model._pair_of_fixed[body[s], units.link[u]])
        out_d.append(d)
        out_n.append(n)
        out_p.append(world[s])
        out_o.append(np.full((len(s), 3), np.nan))
    ts, as_ = np.nonzero(model._allowed_tool[body][:, body])
    if len(ts):
        diff = world[ts] - world[as_]
        norm = np.linalg.norm(diff, axis=1)
        out_pair.append(model._pair_of_tool[body[ts], body[as_]])
        out_d.append(norm - radius[ts] - radius[as_])
        out_n.append(diff / np.maximum(norm, 1e-12)[:, None])
        out_p.append(world[ts])
        out_o.append(world[as_])
    if not out_pair:
        return (np.zeros(0, dtype=int), np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)),
                np.zeros((0, 3)), math.inf)
    pair, d, n, p, o = (np.concatenate(x) for x in (out_pair, out_d, out_n, out_p, out_o))
    keep = d <= limit
    lower = float(d[~keep].min()) if (~keep).any() else math.inf
    return pair[keep], d[keep], n[keep], p[keep], o[keep], lower


def assert_search_is_exhaustive(model, q, extra, limits):
    """Assert ``Model._compute`` lists what the brute force finds, and ``evaluate`` agrees."""
    frames = model._frames(np.asarray(q, dtype=float), extra)
    for limit in limits:
        pair, d, _, _, _, lower = model._compute(frames, limit)
        b_pair, b_d, _, _, _, b_lower = brute_compute(model, frames, limit)
        got = np.lexsort((d, pair))
        want = np.lexsort((b_d, b_pair))
        assert np.array_equal(pair[got], b_pair[want])
        assert np.allclose(d[got], b_d[want], rtol=0.0, atol=1e-12)
        assert lower > limit and lower <= b_lower + 1e-12
    reference = _with_compute(model, brute_compute)
    for per_pair in (1, 16):
        a = model.evaluate(q, extra, per_pair=per_pair)
        b = reference.evaluate(q, extra, per_pair=per_pair)
        assert (a.nearest, a.exact) == (b.nearest, b.exact)
        assert a.clearance == b.clearance or abs(a.clearance - b.clearance) <= 1e-12
        got, want = np.lexsort((a.distance, a.pair)), np.lexsort((b.distance, b.pair))
        assert np.array_equal(a.pair[got], b.pair[want])
        assert np.allclose(a.distance[got], b.distance[want], rtol=0.0, atol=1e-12)


def _with_compute(model, compute):
    import copy
    out = copy.copy(model)
    out._compute = lambda frames, limit: compute(model, frames, limit)
    return out
