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
The stability monitor's reference in Gazebo: the wheels' loads, and the rule INV-3 is judged by.

* **The loads** (independent of the monitor, which never reads the suspension): each corner's
  sprung joint carries a gz spring and damper, so its load is A k (x - x_ref) + c xdot + B, with
  k, x_ref and c read from the description. A and B are fitted on the run's own rest windows
  (the four loads sum to the weight and their centre of pressure sits under the description's
  centre of mass); measured: the simulated springs read as about 0.88-0.91 k (0.888 k with the
  description's present masses), with the running model's masses equal to the description's.
  The figure from the loads is the signed distance of their centre of pressure to the nearest
  edge of the contacts' hull. The fit knows nothing of the droop stop: a corner hanging at its
  stop carries nothing, yet reads A k (x_stop - x_ref) + B, which is not zero (B alone when its
  reference sits at the stop; more, a positive load, when its reference lies past the stop).
  So once a wheel pair has lifted the figure has a floor and cannot go lower; the floor is
  above zero when a reference lies past the droop stop, and a lifted pair does not show.
* **The rule (frozen before this module ran):** per manoeuvre, the monitor's smallest nominal
  figure within +-0.1 s of the loads' smallest lies in [loads - 0.12, loads + 0.01] m; the
  pessimistic figure never above the nominal; the stop figure judged only for stops from at
  most 0.25 m/s (a turn by its wheels' speed): its smallest value over the 60 ms before the zero
  against the loads' smallest after the zero, in the same band. Reported, not judged: the most
  the figure reads safer at any instant, and its lag against the loads.
* **A step under a wheel** (a window marked ``step``): the nominal figure is reported, not
  judged (the monitor was not shown to follow a step); the pessimistic figure is judged there
  instead: its smallest within +-0.1 s of the loads' smallest must not lie above the loads.
"""

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from ranger_xarm_teleop_servo import stability as st
from ranger_xarm_teleop_servo.stability_model import RobotModel

SAFER, LOWER = 0.01, 0.12          # the frozen band (m)
STOP_JUDGED_UP_TO = 0.25           # m/s
WINDOW_S = 0.1
TURN_WHEEL_RADIUS = 0.307          # the wheels' distance from the turning axis (sim)
WHEELS = ('front_left_wheel_link', 'front_right_wheel_link', 'rear_left_wheel_link',
          'rear_right_wheel_link')


class LoadReference:
    """The wheels' loads from the suspension joints, and the figure they give."""

    def __init__(self, urdf: str, q_arm: Dict[str, float]):
        self.model = RobotModel(urdf, 'base_link', WHEELS)
        self.springs = {}
        for w in WHEELS:
            joint = w.replace('_wheel_link', '_suspension_joint')
            self.springs[w] = self.model.springs[joint]
        posed = self.model.pose(q_arm)
        self.contacts = {w: posed.contacts[w] for w in WHEELS}
        self.edges = st.support_polygon(list(self.contacts.values()))
        self.a, self.b = 1.0, 0.0

    def spring_force(self, positions: Dict[str, float]) -> np.ndarray:
        return np.array([self.springs[w].stiffness * (positions[self.joint(w)]
                                                      - self.springs[w].reference)
                         for w in WHEELS])

    @staticmethod
    def joint(w: str) -> str:
        return w.replace('_wheel_link', '_suspension_joint')

    def calibrate(self, rests: Sequence[Tuple[Dict[str, float], np.ndarray, float]]) -> float:
        """
        Fit A and B on rest windows; return the worst centre-of-pressure error (m).

        Each rest: (mean suspension positions, the model's centre of pressure (x, y), the
        weight along the body's z).
        """
        c = np.array([self.contacts[w][:2] for w in WHEELS])
        rows, rhs = [], []
        for positions, cop, weight in rests:
            s = self.spring_force(positions)
            rows.append([s.sum() / weight, 4.0 / weight])
            rhs.append(1.0)
            for j in range(2):
                rows.append([(s * (c[:, j] - cop[j])).sum() / weight * 10.0,
                             (c[:, j] - cop[j]).sum() / weight * 10.0])
                rhs.append(0.0)
        (self.a, self.b), *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        worst = 0.0
        for positions, cop, _ in rests:
            worst = max(worst, float(np.linalg.norm(self.cop(positions, {}) - cop)))
        return worst

    def loads(self, positions: Dict[str, float], speeds: Dict[str, float]) -> np.ndarray:
        s = self.spring_force(positions)
        damp = np.array([self.springs[w].damping * speeds.get(self.joint(w), 0.0)
                         for w in WHEELS])
        return self.a * s + damp + self.b

    def cop(self, positions, speeds) -> np.ndarray:
        f = self.loads(positions, speeds)
        c = np.array([self.contacts[w][:2] for w in WHEELS])
        return (f[:, None] * c).sum(axis=0) / f.sum()

    def figure(self, positions, speeds) -> float:
        return st.nearest(self.cop(positions, speeds), self.edges)[0]


def model_cop_at_rest(urdf: str, q: Dict[str, float],
                      down: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return the description's centre of pressure for gravity ``down`` (unit, body axes)."""
    model = RobotModel(urdf, 'base_link', WHEELS)
    posed = model.pose(q)
    w = st.gravity_vectors(len(posed.masses), st.G * down)
    cop = st.centre_of_pressure(posed.positions, posed.masses, w, posed.plane_z)
    return cop, float(posed.masses.sum() * st.G * -down[2])


def judge(windows: List[dict], ref_t: np.ndarray, ref: np.ndarray, mon_t: np.ndarray,
          nominal: np.ndarray, pessimistic: np.ndarray, stop: np.ndarray,
          speed: np.ndarray, turn: np.ndarray) -> List[dict]:
    """
    Judge every manoeuvre window by the frozen rule; return one row per window.

    ``windows``: {'name', 'start', 'end', 'zero' (or None), 'speed' (the commanded speed; a
    turn's at its wheels), 'judged' (bool), 'step' (optional bool: a step under a wheel)} in
    the same clock as ``ref_t`` and ``mon_t`` (simulation time). Each window ends after its own
    settling: no window reaches into the next.
    """
    rows = []
    for w in windows:
        step = bool(w.get('step'))
        sel = (ref_t >= w['start']) & (ref_t <= w['end'])
        if sel.sum() < 5:
            rows.append({'name': w['name'], 'judged': w['judged'], 'step': step,
                         'loads_min': math.nan, 'dev': math.nan, 'pess_dev': math.nan,
                         'safer_anywhere': math.nan, 'pess_ok': True, 'stop': None,
                         'problems': ['no reference in the window']})
            continue
        j = np.where(sel)[0][np.argmin(ref[sel])]
        near = ((mon_t >= max(ref_t[j] - WINDOW_S, w['start']))
                & (mon_t <= min(ref_t[j] + WINDOW_S, w['end'])))
        row = {'name': w['name'], 'judged': w['judged'], 'step': step,
               'loads_min': float(ref[j]), 'dev': math.nan, 'pess_dev': math.nan,
               'safer_anywhere': math.nan, 'pess_ok': True, 'stop': None, 'problems': []}
        if not near.any():
            row['problems'].append("no monitor figure at the loads' minimum")
            rows.append(row)
            continue
        row['dev'] = float(np.min(nominal[near]) - ref[j])
        row['pess_dev'] = float(np.min(pessimistic[near]) - ref[j])
        inside = (mon_t >= w['start']) & (mon_t <= w['end'])
        row['pess_ok'] = bool(np.all(pessimistic[inside] <= nominal[inside] + 1e-9))
        on_ref = np.interp(ref_t[sel], mon_t[inside], nominal[inside]) if inside.sum() > 1 \
            else np.full(sel.sum(), np.nan)
        row['safer_anywhere'] = float(np.nanmax(on_ref - ref[sel]))
        stop_judged = (w.get('zero') is not None and w['judged']
                       and w['speed'] <= STOP_JUDGED_UP_TO)         # a stop names its speed
        if w.get('zero') is not None:
            before = (mon_t >= w['zero'] - 0.06) & (mon_t <= w['zero'])
            after = (ref_t >= w['zero']) & (ref_t <= w['end'])
            if before.any() and after.any():
                row['stop'] = (w['speed'],
                               float(np.min(stop[before]) - np.min(ref[after])))
            elif stop_judged:
                row['problems'].append('no stop figure before the zero')
        if w['judged']:
            if step:
                if row['pess_dev'] > 0.0:
                    row['problems'].append(f'pessimistic figure {row["pess_dev"]:+.3f} m above '
                                           'the loads at a step')
            elif not (-LOWER <= row['dev'] <= SAFER):
                row['problems'].append(f'figure {row["dev"]:+.3f} m from the loads')
            if not row['pess_ok']:
                row['problems'].append('pessimistic above nominal')
            if stop_judged and row['stop'] and not (-LOWER <= row['stop'][1] <= SAFER):
                row['problems'].append(f'stop figure {row["stop"][1]:+.3f} m from the loads')
        rows.append(row)
    return rows


def lag(ref_t, ref, mon_t, nominal) -> Tuple[float, float]:
    """Return the shift (s) that best aligns the figure with the loads (+: the figure lags)."""
    lo, hi = max(ref_t[0], mon_t[0]) + 1.0, min(ref_t[-1], mon_t[-1]) - 1.0
    grid = np.arange(lo, hi, 0.005)
    a = np.interp(grid, ref_t, ref)
    best = (0.0, math.inf)
    for shift in np.arange(-0.1, 0.1001, 0.005):
        e = float(np.sqrt(np.mean((a - np.interp(grid + shift, mon_t, nominal)) ** 2)))
        if e < best[1]:
            best = (float(shift), e)
    return best
