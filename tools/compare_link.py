#!/usr/bin/env python3
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
Compare the headset link of groups of episodes in one session (no ROS).

    tools/compare_link.py SESSION_DIR --group hidden=1-3 --group shown=4-6 [--group off=7-9]

Reads each episode's sidecar (``episodes/*.json``, the ``link`` block the episode node writes)
and prints, per episode and per group (mean of the episodes, and the worst):
  - XR frame period (median, ms) and its jitter (std, p99 of the deviation, ms): what the page's
    own frame loop did; the panel's drawing cost shows here first;
  - input age at the teleop node (median, p99, max, ms) while armed;
  - sequence gaps, missing frames, steps back, reconnects.
Used to judge the cost of the operator panel on the headset: record the same motion with the
panel hidden, shown, and with feedback off, then compare the groups.
"""

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional

COLUMNS = (
    # (header, path in the link block, scale)
    ('period_ms', ('xr_frame_time', 'median_period_ms'), 1.0),
    ('jit_std_ms', ('xr_frame_time', 'std_ms'), 1.0),
    ('jit_p99_ms', ('xr_frame_time', 'p99_ms'), 1.0),
    ('age_med_ms', ('input_age_s', 'median'), 1e3),
    ('age_p99_ms', ('input_age_s', 'p99'), 1e3),
    ('age_max_ms', ('input_age_s', 'max'), 1e3),
    ('seq_gaps', ('seq_gaps',), 1.0),
    ('missing', ('seq_missing',), 1.0),
    ('back', ('seq_backwards',), 1.0),
    ('reconn', ('reconnects',), 1.0),
)


def parse_range(text: str) -> List[int]:
    """Return the episode indices of '1-3,5' style text."""
    out: List[int] = []
    for part in text.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def value(link: dict, path, scale: float) -> Optional[float]:
    v = link
    for key in path:
        if not isinstance(v, dict) or key not in v:
            return None
        v = v[key]
    if v is None:
        return None
    v = float(v) * scale
    return v if math.isfinite(v) else None


def load(session: str) -> Dict[int, dict]:
    """Return {episode index: link block} of every episode sidecar in the session."""
    out = {}
    for path in sorted(glob.glob(os.path.join(session, 'episodes', '*.json'))):
        with open(path, encoding='utf-8') as handle:
            sidecar = json.load(handle)
        ep = sidecar.get('episode', {})
        if 'index' in ep and isinstance(sidecar.get('link'), dict):
            out[int(ep['index'])] = sidecar['link']
    return out


def summary(links: List[dict]) -> Dict[str, Dict[str, Optional[float]]]:
    """Return per column the mean over the episodes and the worst (largest) value."""
    out = {}
    for name, path, scale in COLUMNS:
        vals = [v for v in (value(link, path, scale) for link in links) if v is not None]
        out[name] = {'mean': sum(vals) / len(vals) if vals else None,
                     'worst': max(vals) if vals else None}
    return out


def fmt(v: Optional[float]) -> str:
    return '-' if v is None else f'{v:.2f}'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('session', help='the session directory (holds episodes/)')
    ap.add_argument('--group', action='append', required=True,
                    help='NAME=RANGE, e.g. hidden=1-3 (repeatable)')
    args = ap.parse_args(argv)
    links = load(args.session)
    if not links:
        print(f'no episode sidecars with a link block in {args.session}', file=sys.stderr)
        return 1
    header = ['group', 'episode'] + [c[0] for c in COLUMNS]
    print('\t'.join(header))
    groups = {}
    for spec in args.group:
        name, _, rng = spec.partition('=')
        wanted = parse_range(rng)
        missing = [i for i in wanted if i not in links]
        if missing:
            print(f'{name}: episodes {missing} not found', file=sys.stderr)
        chosen = [links[i] for i in wanted if i in links]
        for i in wanted:
            if i in links:
                row = [fmt(value(links[i], p, s)) for _, p, s in COLUMNS]
                print('\t'.join([name, str(i)] + row))
        groups[name] = summary(chosen)
    print()
    print('\t'.join(['group', 'stat'] + [c[0] for c in COLUMNS]))
    for name, s in groups.items():
        for stat in ('mean', 'worst'):
            print('\t'.join([name, stat] + [fmt(s[c[0]][stat]) for c in COLUMNS]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
