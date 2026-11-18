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
Refit the consistency tolerances from kept episodes (no ROS).

Rule (the one the scripted Gazebo tolerances were fitted with):

- Take the episodes in recording order (session, then index); only kept episodes whose
  arm consistency could be judged and that fail nothing but consistency are eligible.
- Fit on the first ``fraction`` of them (7 of 10 at the default 0.7) and hold out the rest.
- Each tolerance is the worst value on the fit set times ``factor`` (1.5), rounded up to its
  step (0.5 mm, 0.1 deg, 0.005 rad).
- Judge the held-out episodes with the proposed values and report every one that fails.

The result is a proposal: nothing here writes a profile or a tolerance file. Fewer eligible
episodes than ``min_episodes`` give no proposal at all.
"""

import math
from typing import Dict, List, Optional, Sequence

# tolerance key -> (consistency part, metric of that part, rounding step)
FIT_KEYS: Dict[str, tuple] = {
    'arm_rms_m': ('arm', 'rms_m', 0.0005),
    'arm_p99_m': ('arm', 'p99_m', 0.0005),
    'arm_rot_p99_deg': ('arm', 'rot_p99_deg', 0.1),
    'gripper_abs_rad': ('gripper', 'p99_rad', 0.005),
}
FACTOR = 1.5
FRACTION = 0.7
MIN_EPISODES = 5


def round_up(value: float, step: float) -> float:
    """Return ``value`` rounded up to a multiple of ``step`` (float noise ignored)."""
    return round(math.ceil(value / step - 1e-9) * step, 10)


def n_fit(n: int, fraction: float = FRACTION) -> int:
    """Return how many of ``n`` ordered episodes form the fit set (at least one held out)."""
    if n < 2:
        raise ValueError('need at least two episodes: one to fit, one to hold out')
    return max(1, min(n - 1, int(math.floor(fraction * n + 0.5))))


def metrics(consistency_report: dict) -> Dict[str, Optional[float]]:
    """Return the fitted metrics of one consistency report (None where not judged)."""
    out: Dict[str, Optional[float]] = {}
    for key, (part, metric, _) in FIT_KEYS.items():
        entry = consistency_report.get(part) or {}
        judged = entry.get('verdict') in ('PASS', 'FAIL')
        if part == 'gripper' and not entry.get('applicable', True):
            judged = False
        value = entry.get(metric) if judged else None
        out[key] = None if value is None else float(value)
    return out


def judge(values: Dict[str, Optional[float]], limits: Dict[str, float]) -> List[str]:
    """Return the keys whose value exceeds its limit."""
    return [k for k, v in values.items()
            if v is not None and k in limits and v > float(limits[k])]


def fit(episodes: Sequence[dict], current: Dict[str, float], fraction: float = FRACTION,
        factor: float = FACTOR, min_episodes: int = MIN_EPISODES) -> dict:
    """
    Fit the tolerances on ``episodes`` (in recording order).

    Each episode is ``{'id': str, 'metrics': metrics(...)}``. ``current`` holds the tolerances
    in force, reported beside the proposal. Returns ``{'ok', 'reason', 'fit', 'held_out',
    'worst_fit', 'worst_held_out', 'proposed', 'current', 'held_out_failures', ...}``.
    """
    n = len(episodes)
    rule = {'fraction': fraction, 'factor': factor, 'min_episodes': min_episodes,
            'steps': {k: v[2] for k, v in FIT_KEYS.items()},
            'text': (f'worst on the first {fraction:g} of the eligible episodes in recording '
                     f'order x {factor:g}, rounded up to the step; the rest held out')}
    out = {'ok': False, 'reason': '', 'rule': rule, 'n_eligible': n,
           'current': {k: current.get(k) for k in FIT_KEYS}}
    if n < max(2, min_episodes):
        out['reason'] = (f'{n} eligible episode(s), at least {max(2, min_episodes)} needed: '
                         'no proposal')
        return out
    k = n_fit(n, fraction)
    fit_set, held = list(episodes[:k]), list(episodes[k:])
    out['fit'] = [e['id'] for e in fit_set]
    out['held_out'] = [e['id'] for e in held]

    def worst(group, key):
        values = [e['metrics'].get(key) for e in group]
        values = [v for v in values if v is not None]
        return (max(values), len(values)) if values else (None, 0)

    proposed, worst_fit, worst_held, counts, not_fitted = {}, {}, {}, {}, {}
    for key, (_, _, step) in FIT_KEYS.items():
        w, m = worst(fit_set, key)
        worst_fit[key] = w
        worst_held[key] = worst(held, key)[0]
        counts[key] = m
        if w is None:
            not_fitted[key] = 'no fit episode has this metric judged'
            continue
        proposed[key] = round_up(factor * w, step)
    out.update(ok=True, proposed=proposed, worst_fit=worst_fit, worst_held_out=worst_held,
               n_fit_values=counts, not_fitted=not_fitted)
    out['held_out_failures'] = {e['id']: judge(e['metrics'], proposed) for e in held
                                if judge(e['metrics'], proposed)}
    out['changes'] = {key: {'current': current.get(key), 'proposed': value,
                            'direction': ('same' if current.get(key) is None
                                          or math.isclose(value, float(current[key]))
                                          else 'tighter' if value < float(current[key])
                                          else 'looser')}
                      for key, value in proposed.items()}
    return out


def profile_block(result: dict, provenance: str) -> dict:
    """Return the proposal as a profile ``checks``/``provisional`` block (nothing provisional)."""
    return {'checks': {'consistency': dict(result.get('proposed') or {})},
            'fit': {'rule': result['rule']['text'], 'fit': result.get('fit'),
                    'held_out': result.get('held_out'),
                    'held_out_failures': result.get('held_out_failures'),
                    'source': provenance}}
