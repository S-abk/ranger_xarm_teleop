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
Command-line tools for recorded sessions. None of them joins a ROS graph.

- ``episode_labels``: type the task label and operator for the next episodes.
- ``episode_check``: check episodes of a session, write ``checks/NNNN.json``, print a table.
- ``episode_export``: export one episode at a fixed rate into ``exports/NNNN/``.
- ``episode_purge``: list discarded episodes; with ``--delete`` remove their bag files
  (the sidecar stays, with the purge recorded). Nothing is ever deleted automatically.
- ``episode_fit``: propose consistency tolerances from kept episodes (fit and hold-out rule
  of ``fit.py``); prints the proposal and writes it only to a file named with ``--out``.

Tolerances: checks.yaml with the overrides of the package's current profile of the episode's
profile name (``--profile`` names another; without a package profile of that name, the
episode's recorded copy). The check report records which source was used.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

from . import metadata as md
from .profile import effective_checks, load_checks, load_profile, package_profile


def _data_dir(args) -> str:
    if args.data_dir:
        return os.path.abspath(os.path.expanduser(args.data_dir))
    return os.path.abspath(os.path.expanduser(load_profile(args.profile)['data_dir']))


def labels_main(argv=None) -> int:
    """Write the labels file (prompting for what is not given)."""
    ap = argparse.ArgumentParser(description='Set the task label and operator for the next '
                                 'episodes. Stored only in the data directory.')
    ap.add_argument('--profile', default='mock')
    ap.add_argument('--data-dir', default='')
    ap.add_argument('--task', default=None)
    ap.add_argument('--operator', default=None)
    args = ap.parse_args(argv)
    data_dir = _data_dir(args)
    current, _ = md.read_labels(data_dir)
    current = current or {}
    task = args.task if args.task is not None else input(
        f'task label [{current.get("task", "")}]: ').strip() or current.get('task', '')
    operator = args.operator if args.operator is not None else input(
        f'operator [{current.get("operator", "")}]: ').strip() or current.get('operator', '')
    if not task or not operator:
        print('both a task label and an operator are needed', file=sys.stderr)
        return 2
    os.makedirs(data_dir, exist_ok=True)
    path = md.write_labels(data_dir, task, operator)
    print(f'labels written to {path}; they apply from the next episode')
    return 0


def tolerances(base: dict, recorded_profile: dict, profile_arg: Optional[str] = None) -> dict:
    """Return the tolerances that judge an episode recorded under ``recorded_profile``."""
    if profile_arg:
        return effective_checks(base, load_profile(profile_arg), f'profile {profile_arg} (given)')
    name = str(recorded_profile.get('name') or '')
    current = package_profile(name) if name else None
    if current is not None:
        return effective_checks(base, current, f'package profile {name} (current)')
    return effective_checks(base, recorded_profile, 'recorded profile (episode sidecar)')


def _episodes(session: str, which):
    paths = list(md.iter_episodes(session))
    if not which:
        return paths
    wanted = {md.episode_name(int(w)) for w in which}
    return [p for p in paths if os.path.basename(p)[:-5] in wanted]


def check_main(argv=None) -> int:
    """Check episodes; exit 1 when a kept episode fails."""
    from .bag_io import read_episode
    from .checker import check_episode, stability_lines
    ap = argparse.ArgumentParser(description='Check recorded episodes (rates, gaps, clocks, '
                                 'losses, consistency). Never changes the data.')
    ap.add_argument('session')
    ap.add_argument('episodes', nargs='*', help='indices (default: all)')
    ap.add_argument('--checks', default=None, help='tolerance file (default: the package one)')
    ap.add_argument('--profile', default=None, help='profile whose tolerance overrides apply '
                    "(default: the package's profile of the episode's profile name)")
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args(argv)
    base = load_checks(args.checks)
    session = os.path.abspath(args.session)
    bad = 0
    print(f'{"ep":>4} {"status":<11} {"verdict":<8} {"usable":<6} flags / reasons')
    for path in _episodes(session, args.episodes):
        sidecar = md.read_json(path)
        status = sidecar.get('episode', {}).get('status')
        if status in ('recording', 'failed') or sidecar.get('episode', {}).get('purged'):
            print(f'{os.path.basename(path)[:-5]:>4} {status:<11} skipped')
            continue
        ep = read_episode(session, path)
        report = check_episode(ep, tolerances(base, ep.profile, args.profile))
        report['checked_wall'] = time.time()
        if not args.no_write:
            md.write_json_atomic(os.path.join(session, 'checks', os.path.basename(path)),
                                 report)
        flags = ','.join(sorted({f['flag'] for f in report['flags']}))
        provisional = report['consistency'].get('provisional') or []
        if provisional and report['consistency'].get('verdict') != 'NOT_JUDGED':
            flags += f' [provisional tolerance: {",".join(provisional)}]'
        print(f'{os.path.basename(path)[:-5]:>4} {status:<11} {report["verdict"]:<8} '
              f'{str(report["usable"]):<6} {flags} {"; ".join(report["reasons"])[:200]}')
        if 'stability' in report:           # not in the report of an episode with no timeline
            for line in stability_lines(report['stability']):
                print(f'     {line}')
        if status == 'kept' and report['verdict'] != 'PASS':
            bad += 1
    return 1 if bad else 0


def export_main(argv=None) -> int:
    """Export one episode at a fixed rate."""
    from .bag_io import read_episode
    from .checker import check_episode
    from .export import build_export, write_export
    ap = argparse.ArgumentParser(description='Export one episode at a fixed rate (causal '
                                 'hold, nothing interpolated).')
    ap.add_argument('session')
    ap.add_argument('episode', type=int)
    ap.add_argument('--rate', type=float, default=None,
                    help='fixed tick rate (default: checks.yaml export.rate_hz)')
    ap.add_argument('--tick-stream', default=None, metavar='CAMERA',
                    help='put the ticks on the frame stamps of this image stream (e.g. '
                    'camera_wrist), so each of its frames is used exactly once')
    ap.add_argument('--checks', default=None)
    ap.add_argument('--profile', default=None, help='profile whose tolerance overrides apply')
    ap.add_argument('--out', default=None, help='default: <session>/exports/NNNN')
    args = ap.parse_args(argv)
    if args.rate and args.tick_stream:
        ap.error('--rate and --tick-stream exclude each other')
    session = os.path.abspath(args.session)
    path = md.episode_path(session, args.episode)
    ep = read_episode(session, path)
    checks = tolerances(load_checks(args.checks), ep.profile, args.profile)
    report = check_episode(ep, checks)
    arrays, meta = build_export(ep, checks, rate_hz=args.rate, verdict=report,
                                tick_stream=args.tick_stream)
    out = args.out or os.path.join(session, 'exports', md.episode_name(args.episode))
    written = write_export(out, arrays, meta, ep)
    ticks = meta['ticks']
    grid = (f'at {ticks["rate_hz"]:g} Hz' if ticks['mode'] == 'fixed'
            else f'on the frames of {ticks["stream"]}')
    print(f'exported episode {args.episode} ({report["verdict"]}) to {out}: '
          f'{len(written)} files, {len(arrays["t"])} ticks {grid}')
    for name, info in (meta.get('images') or {}).items():
        print(f'  {name}: {info["frames_used"]} of {info["frames_recorded"]} frames used, '
              f'{info["frames_unused"]} unused, {info["ticks_repeating_frame"]} ticks '
              'repeating the frame before')
    return 0


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def purge_main(argv=None) -> int:
    """List discarded episodes; delete their bag files only with --delete."""
    ap = argparse.ArgumentParser(description='List discarded episodes and the space their '
                                 'bag files take. Dry run unless --delete is given; the '
                                 'sidecars always stay, with the purge recorded.')
    ap.add_argument('path', help='a data directory or one session')
    ap.add_argument('--delete', action='store_true', help='really delete the bag files')
    args = ap.parse_args(argv)
    root = os.path.abspath(os.path.expanduser(args.path))
    sessions = ([root] if os.path.exists(os.path.join(root, 'session.json')) else
                sorted(os.path.join(root, d) for d in os.listdir(root)
                       if os.path.exists(os.path.join(root, d, 'session.json'))))
    total = 0
    for session in sessions:
        sidecars = [(p, md.read_json(p)) for p in md.iter_episodes(session)]
        # A file any other episode refers to is never deleted (two episodes can share one
        # when the recorder failed to split between them).
        others = {}
        for _, sc in sidecars:
            e = sc.get('episode', {})
            if e.get('status') != 'discarded':
                others.setdefault(e.get('file'), []).append(e.get('index'))
        for path, sidecar in sidecars:
            e = sidecar.get('episode', {})
            if e.get('status') != 'discarded' or e.get('purged'):
                continue
            bag = os.path.join(session, e.get('file') or '')
            if not e.get('file') or e.get('file') in others:
                shared = others.get(e.get('file'))
                print(f'kept: {os.path.basename(session)}/{e.get("file")} (episode '
                      f'{e.get("index")}) is also the file of episode(s) {shared}')
                continue
            size = _size(bag)
            total += size
            action = 'would delete'
            if args.delete and os.path.isfile(bag):
                os.unlink(bag)
                e['purged'] = {'wall': time.time(), 'bytes': size, 'file': e.get('file')}
                md.write_json_atomic(path, sidecar)
                action = 'deleted'
            print(f'{action} {os.path.basename(session)}/{e.get("file")} '
                  f'({size / 1e6:.1f} MB), episode {e.get("index")}')
    print(json.dumps({'sessions': len(sessions), 'bytes': total, 'deleted': bool(args.delete)}))
    if not args.delete:
        print('dry run: nothing deleted (add --delete to delete these files)')
    return 0


def _session_order(path: str):
    try:
        return (float(md.read_json(os.path.join(path, 'session.json')).get('started_wall')),
                path)
    except (OSError, TypeError, ValueError):
        return (math.inf, path)


def fit_main(argv=None) -> int:
    """Propose consistency tolerances from kept episodes; exit 2 when there is no proposal."""
    from .bag_io import read_episode
    from .checker import check_episode
    from . import fit
    ap = argparse.ArgumentParser(description='Propose the consistency tolerances from kept '
                                 'episodes: worst of the first 70 percent (recording order) '
                                 'x 1.5, rounded up; the rest held out and judged. Prints the '
                                 'proposal; never edits a profile or the sessions.')
    ap.add_argument('sessions', nargs='+', help='session directories (recording order is '
                    'taken from their start times)')
    ap.add_argument('--profile', default=None, help='profile whose tolerances are current '
                    "(default: the package's profile of the episodes' profile name)")
    ap.add_argument('--checks', default=None)
    ap.add_argument('--last', type=int, default=None, help='use only the last N eligible '
                    'episodes')
    ap.add_argument('--fraction', type=float, default=fit.FRACTION)
    ap.add_argument('--factor', type=float, default=fit.FACTOR)
    ap.add_argument('--min-episodes', type=int, default=fit.MIN_EPISODES)
    ap.add_argument('--out', default=None, help='also write the proposal (YAML) to this file')
    args = ap.parse_args(argv)
    base = load_checks(args.checks)
    eligible, skipped, names, current = [], [], set(), None
    for session in sorted((os.path.abspath(p) for p in args.sessions), key=_session_order):
        for path in md.iter_episodes(session):
            ident = f'{os.path.basename(session)}/{os.path.basename(path)[:-5]}'
            sidecar = md.read_json(path)
            if sidecar.get('episode', {}).get('status') != 'kept':
                continue
            ep = read_episode(session, path)
            names.add(str(ep.profile.get('name')))
            checks = tolerances(base, ep.profile, args.profile)
            current = current or {k: checks['consistency'].get(k) for k in fit.FIT_KEYS}
            report = check_episode(ep, checks)
            other = [r for r in report['reasons'] if not r.startswith('CONSISTENCY:')]
            values = fit.metrics(report['consistency'])
            if other:
                skipped.append((ident, 'fails other checks: ' + '; '.join(other)[:160]))
            elif values['arm_rms_m'] is None:
                skipped.append((ident, 'arm consistency not judged'))
            else:
                eligible.append({'id': ident, 'metrics': values})
    if len(names) > 1:
        print(f'episodes of more than one profile ({sorted(names)}): fit one deployment at a '
              'time', file=sys.stderr)
        return 2
    if args.last:
        eligible = eligible[-args.last:]
    result = fit.fit(eligible, current or {}, args.fraction, args.factor, args.min_episodes)
    for ident, why in skipped:
        print(f'skipped {ident}: {why}')
    fit_ids = set(result.get('fit') or [])
    print(f'{"episode":<32} {"set":<5} ' + ' '.join(f'{k:>16}' for k in fit.FIT_KEYS))
    for e in eligible:
        role = '-' if not result['ok'] else 'fit' if e['id'] in fit_ids else 'held'
        cells = ' '.join(f'{"-" if v is None else f"{v:.5g}":>16}'
                         for v in e['metrics'].values())
        print(f'{e["id"]:<32} {role:<5} {cells}')
    if not result['ok']:
        print(result['reason'])
        return 2
    for key, change in result['changes'].items():
        held = result['worst_held_out'][key]
        held = '-' if held is None else f'{held:.5g}'
        print(f'{key}: proposed {change["proposed"]:g} (current {change["current"]}, '
              f'{change["direction"]}; worst fit {result["worst_fit"][key]:.5g}, worst held '
              f'out {held})')
    for key, why in result['not_fitted'].items():
        print(f'{key}: not fitted ({why}); keep the current value')
    if result['held_out_failures']:
        for ident, keys in result['held_out_failures'].items():
            print(f'held-out episode {ident} FAILS the proposal on {", ".join(keys)}')
    else:
        print(f'all {len(result["held_out"])} held-out episodes pass the proposal')
    if args.out:
        import yaml
        block = fit.profile_block(result, f'{len(eligible)} kept episodes of profile '
                                  f'{next(iter(names), "?")}')
        with open(args.out, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(block, handle, sort_keys=False)
        print(f'proposal written to {args.out}; copy its checks block into the profile by hand')
    return 0
