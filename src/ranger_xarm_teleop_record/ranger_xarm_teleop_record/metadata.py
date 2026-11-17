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
Session layout, the episode sidecar's required keys, and the labels file (no ROS).

Layout of a data directory (every path inside a session is relative to the session, so a
session directory can be moved to another machine and checked or exported there)::

    <data_dir>/labels.json            task label and operator, typed in at run time
    <data_dir>/<session>/session.json
    <data_dir>/<session>/robot_description.urdf, robot_description.srdf, limits.yaml
    <data_dir>/<session>/bag/         rosbag2 MCAP, one file per episode
    <data_dir>/<session>/episodes/NNNN.json
    <data_dir>/<session>/checks/NNNN.json
    <data_dir>/<session>/exports/NNNN/
    <data_dir>/<session>/recorder.log
"""

import json
import os
import tempfile
from typing import Iterable, List, Optional

FORMAT = 'ranger_xarm_teleop.episode/1'
SESSION_FORMAT = 'ranger_xarm_teleop.session/1'
STATUSES = ('recording', 'pending', 'kept', 'discarded', 'interrupted', 'failed')

# INV-3: what an episode must carry to be interpreted without the repo's current state.
# Dotted paths into the episode sidecar; a missing key, None or '' counts as missing.
REQUIRED = (
    'format',
    'episode.index', 'episode.status', 'episode.file', 'episode.labels.task',
    'episode.labels.operator', 'episode.start.ros', 'episode.start.wall', 'episode.stop.ros',
    'episode.stop.wall', 'episode.stop_reason',
    'software.teleop_repo.commit', 'software.teleop_repo.dirty',
    'software.robot_workspace.commit', 'software.robot_workspace.dirty',
    'robot.description_sha256', 'robot.description_file', 'robot.semantic_sha256',
    'limits.file', 'limits.sha256', 'limits.parameters.teleop',
    'calibration.calibrated', 'calibration.heading', 'calibration.session_id',
    'calibration.reset_count', 'calibration.changed_during_episode',
    'frames.base', 'frames.tool', 'frames.input',
    'scale.at_start', 'scale.values_used',
    'time.use_sim_time', 'time.recording_clock', 'time.streams', 'time.headset',
    'time.clock_pairs',
    'base.mode',
    'counts',
    'profile',
)
# Required for every recorded, required image stream's camera.
CAMERA_REQUIRED = ('intrinsics.width', 'intrinsics.height', 'intrinsics.k',
                   'intrinsics.distortion_model', 'intrinsics.frame_id',
                   'extrinsics.mount_link', 'extrinsics.in_mount')


def lookup(data: dict, dotted: str):
    """Return the value at a dotted path, or None when any part is missing."""
    node = data
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def is_missing(value) -> bool:
    """Return True for None, '' and an empty container (False and 0 are values)."""
    if value is None:
        return True
    if isinstance(value, (str, list, dict, tuple)) and len(value) == 0:
        return True
    return False


def missing(sidecar: dict) -> List[str]:
    """Return the required keys this sidecar lacks (INV-3); empty when complete."""
    out = [k for k in REQUIRED if is_missing(lookup(sidecar, k))]
    profile = sidecar.get('profile') or {}
    for name, stream in (profile.get('streams') or {}).items():
        if stream.get('kind') != 'image' or not stream.get('required') \
                or not stream.get('record', True):
            continue
        camera = stream.get('camera') or name
        for key in CAMERA_REQUIRED:
            path = f'cameras.{camera}.{key}'
            if is_missing(lookup(sidecar, path)):
                out.append(path)
    if sidecar.get('format') not in (None, FORMAT):
        out.append(f'format (unknown: {sidecar.get("format")!r})')
    return out


def episode_name(index: int) -> str:
    """Return the file stem of episode ``index`` (1-based)."""
    return f'{int(index):04d}'


def episode_path(session_dir: str, index: int) -> str:
    """Return the sidecar path of episode ``index`` in a session."""
    return os.path.join(session_dir, 'episodes', episode_name(index) + '.json')


def write_json_atomic(path: str, data: dict) -> None:
    """Write JSON so a reader never sees a half-written file."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.tmp-', dir=folder)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, indent=1, sort_keys=False, allow_nan=True)
            handle.write('\n')
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: str) -> dict:
    """Read one JSON file."""
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def git_root(path: str) -> Optional[str]:
    """Return the enclosing git working tree of ``path`` (a .git dir or file), or None."""
    p = os.path.abspath(os.path.expanduser(path))
    while True:
        if os.path.exists(os.path.join(p, '.git')):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def _ignored(root: str, path: str) -> bool:
    """Return True when git would ignore ``path`` in the work tree ``root``."""
    import subprocess
    try:
        done = subprocess.run(['git', '-C', root, 'check-ignore', '-q', path],
                              capture_output=True, timeout=10.0, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def data_dir_problem(path: str, code_root: Optional[str] = None) -> Optional[str]:
    """
    Return why ``path`` may not hold recordings, or None.

    Never inside this code's own checkout (``code_root``, default: the tree holding this
    file), even in an ignored folder; inside any other git work tree only where that
    tree ignores the path, so a recording can never be committed by accident.
    """
    p = os.path.abspath(os.path.expanduser(path))
    own = code_root if code_root is not None else git_root(os.path.realpath(__file__))
    if own is not None and (p == own or p.startswith(own.rstrip(os.sep) + os.sep)):
        return (f'the data directory {path} is inside the code checkout {own}; '
                'recordings must stay out of the repository')
    root = git_root(p)
    if root is not None and not _ignored(root, p):
        return (f'the data directory {path} is inside the git work tree {root} and not '
                'ignored there; recordings must never be committable')
    return None


LABEL_KEYS = ('task', 'operator')


def labels_path(data_dir: str) -> str:
    """Return the labels file of a data directory."""
    return os.path.join(os.path.expanduser(data_dir), 'labels.json')


def read_labels(data_dir: str):
    """Return (labels, problem): the labels in force, or None and why not."""
    path = labels_path(data_dir)
    try:
        data = read_json(path)
    except FileNotFoundError:
        return None, f'no labels file {path}: run episode_labels first'
    except (OSError, ValueError) as exc:
        return None, f'labels file {path} unreadable: {exc}'
    if not isinstance(data, dict):
        return None, f'labels file {path}: not a JSON object'
    absent = [k for k in LABEL_KEYS if not str(data.get(k) or '').strip()]
    if absent:
        return None, f'labels file {path}: no {" or ".join(absent)}'
    return {k: str(data[k]).strip() for k in LABEL_KEYS}, None


def write_labels(data_dir: str, task: str, operator: str) -> str:
    """Write the labels file; return its path."""
    problem = data_dir_problem(data_dir)
    if problem:
        raise ValueError(problem)
    path = labels_path(data_dir)
    write_json_atomic(path, {'task': task.strip(), 'operator': operator.strip()})
    return path


def iter_episodes(session_dir: str) -> Iterable[str]:
    """Yield the sidecar paths of a session in index order."""
    folder = os.path.join(session_dir, 'episodes')
    if not os.path.isdir(folder):
        return
    for name in sorted(os.listdir(folder)):
        if name.endswith('.json') and not name.startswith('.'):
            yield os.path.join(folder, name)


def relative_inside(session_dir: str, path: str) -> str:
    """Return ``path`` relative to the session; raise when it lies outside it."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(session_dir))
    if rel.startswith('..') or os.path.isabs(rel):
        raise ValueError(f'{path} is outside the session {session_dir}')
    return rel.replace(os.sep, '/')
