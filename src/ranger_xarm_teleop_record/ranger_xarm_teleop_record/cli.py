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
"""

import argparse
import json
import os
import sys
import time

from . import metadata as md
from .profile import load_checks, load_profile


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


def _episodes(session: str, which):
    paths = list(md.iter_episodes(session))
    if not which:
        return paths
    wanted = {md.episode_name(int(w)) for w in which}
    return [p for p in paths if os.path.basename(p)[:-5] in wanted]


def check_main(argv=None) -> int:
    """Check episodes; exit 1 when a kept episode fails."""
    from .bag_io import read_episode
    from .checker import check_episode
    ap = argparse.ArgumentParser(description='Check recorded episodes (rates, gaps, clocks, '
                                 'losses, consistency). Never changes the data.')
    ap.add_argument('session')
    ap.add_argument('episodes', nargs='*', help='indices (default: all)')
    ap.add_argument('--checks', default=None, help='tolerance file (default: the package one)')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args(argv)
    checks = load_checks(args.checks)
    session = os.path.abspath(args.session)
    bad = 0
    print(f'{"ep":>4} {"status":<11} {"verdict":<8} {"usable":<6} flags / reasons')
    for path in _episodes(session, args.episodes):
        sidecar = md.read_json(path)
        status = sidecar.get('episode', {}).get('status')
        if status in ('recording', 'failed') or sidecar.get('episode', {}).get('purged'):
            print(f'{os.path.basename(path)[:-5]:>4} {status:<11} skipped')
            continue
        report = check_episode(read_episode(session, path), checks)
        report['checked_wall'] = time.time()
        if not args.no_write:
            md.write_json_atomic(os.path.join(session, 'checks', os.path.basename(path)),
                                 report)
        flags = ','.join(sorted({f['flag'] for f in report['flags']}))
        print(f'{os.path.basename(path)[:-5]:>4} {status:<11} {report["verdict"]:<8} '
              f'{str(report["usable"]):<6} {flags} {"; ".join(report["reasons"])[:200]}')
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
    ap.add_argument('--rate', type=float, default=None)
    ap.add_argument('--checks', default=None)
    ap.add_argument('--out', default=None, help='default: <session>/exports/NNNN')
    args = ap.parse_args(argv)
    session = os.path.abspath(args.session)
    path = md.episode_path(session, args.episode)
    checks = load_checks(args.checks)
    ep = read_episode(session, path)
    report = check_episode(ep, checks)
    arrays, meta = build_export(ep, checks, rate_hz=args.rate, verdict=report['verdict'])
    out = args.out or os.path.join(session, 'exports', md.episode_name(args.episode))
    written = write_export(out, arrays, meta, ep)
    print(f'exported episode {args.episode} ({report["verdict"]}) to {out}: '
          f'{len(written)} files, {len(arrays["t"])} ticks at {meta["rate_hz"]:g} Hz')
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
