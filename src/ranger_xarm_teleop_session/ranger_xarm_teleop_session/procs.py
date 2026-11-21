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
The session's processes, found by their marker; the state file; what a stop leaves. No ROS.

Every process the launcher starts gets ``TELEOP_SESSION_MARK=<mark>`` in its environment, and
every process those start inherits it (gz, the recorder, the nodes). The stop signals only
processes that carry this session's own mark, found in ``/proc/<pid>/environ``: never a process
matched by its name or its group alone. A process that changed its environment would be missed
and is never signalled; it shows up nowhere, so the stop cannot claim it was stopped.

The state file holds the supervisor's pid, the mark, the children (name, pid, stop order) and the
session directory. It lets ``teleop_session stop`` find a session whose supervisor is gone, and
keeps a second session from starting while one runs.
"""

from dataclasses import dataclass
import json
import os
import tempfile
from typing import Dict, Iterable, List, Optional

MARK_VAR = 'TELEOP_SESSION_MARK'
STATE_DIR_VAR = 'TELEOP_SESSION_STATE_DIR'      # tests point it at their own directory


@dataclass(frozen=True)
class Proc:
    """One process of the session."""

    pid: int
    pgid: int
    ppid: int
    name: str
    cmdline: str


def _read(path: str, binary: bool = False):
    try:
        with open(path, 'rb' if binary else 'r') as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return None


def mark_of(pid: int, proc_root: str = '/proc') -> Optional[str]:
    """Return the session mark in a process's environment, or None (also when unreadable)."""
    raw = _read(os.path.join(proc_root, str(pid), 'environ'), binary=True)
    if not raw:
        return None
    prefix = (MARK_VAR + '=').encode()
    for entry in raw.split(b'\0'):
        if entry.startswith(prefix):
            return entry[len(prefix):].decode('utf-8', 'replace')
    return None


def _stat(pid: int, proc_root: str):
    raw = _read(os.path.join(proc_root, str(pid), 'stat'))
    if not raw or ')' not in raw:
        return None
    name = raw[raw.index('(') + 1:raw.rindex(')')]
    fields = raw[raw.rindex(')') + 2:].split()
    # fields[0] state, [1] ppid, [2] pgrp
    if fields[0] == 'Z':
        return None                     # a zombie holds nothing and cannot be signalled
    return name, int(fields[1]), int(fields[2])


def marked(mark: str, proc_root: str = '/proc', exclude: Iterable[int] = ()) -> List[Proc]:
    """Return the live processes that carry ``mark`` (never the caller's own process)."""
    skip = set(exclude) | {os.getpid()}
    out = []
    try:
        pids = [int(p) for p in os.listdir(proc_root) if p.isdigit()]
    except OSError:
        return out
    for pid in sorted(pids):
        if pid in skip or mark_of(pid, proc_root) != mark:
            continue
        st = _stat(pid, proc_root)
        if st is None:
            continue
        cmd = (_read(os.path.join(proc_root, str(pid), 'cmdline'), binary=True) or b'')
        out.append(Proc(pid, st[2], st[1], st[0],
                        cmd.replace(b'\0', b' ').decode('utf-8', 'replace').strip()))
    return out


def descendants_of(root_pid: int, procs: Iterable[Proc]) -> List[Proc]:
    """Return ``procs`` that are ``root_pid`` or descend from it (by parent pid)."""
    procs = list(procs)
    by_parent: Dict[int, List[Proc]] = {}
    for p in procs:
        by_parent.setdefault(p.ppid, []).append(p)
    out, todo = [], [root_pid]
    seen = set()
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out += [p for p in procs if p.pid == pid]
        todo += [c.pid for c in by_parent.get(pid, [])]
    return out


def alive(pid: Optional[int]) -> bool:
    """Return True for a live process (not a zombie)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    st = _stat(pid, '/proc')
    return st is not None


def open_files_under(path: str, proc_root: str = '/proc') -> List[str]:
    """Return 'pid name: file' for every open file under ``path`` (any readable process)."""
    root = os.path.realpath(path).rstrip(os.sep) + os.sep
    out = []
    try:
        pids = [p for p in os.listdir(proc_root) if p.isdigit()]
    except OSError:
        return out
    for pid in pids:
        fd_dir = os.path.join(proc_root, pid, 'fd')
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target.startswith(root):
                st = _stat(int(pid), proc_root)
                out.append(f'{pid} {st[0] if st else "?"}: {target}')
    return sorted(out)


def episodes_still_recording(sidecars: Dict[str, dict]) -> List[str]:
    """Return the sidecars (path -> content) whose episode is still marked recording."""
    return sorted(p for p, s in sidecars.items()
                  if ((s or {}).get('episode') or {}).get('status') == 'recording')


# ------------------------------------------------------------------ the state file
def state_dir(env: Optional[Dict[str, str]] = None) -> str:
    """Return the directory of the state file (per user, outside the checkout)."""
    env = os.environ if env is None else env
    if env.get(STATE_DIR_VAR):
        return env[STATE_DIR_VAR]
    base = env.get('XDG_RUNTIME_DIR') or os.path.join(tempfile.gettempdir(),
                                                      f'teleop_session-{os.getuid()}')
    return os.path.join(base, 'teleop_session')


def state_path(env: Optional[Dict[str, str]] = None) -> str:
    """Return the state file's path."""
    return os.path.join(state_dir(env), 'state.json')


def read_state(path: str) -> Optional[dict]:
    """Return the state file's content, or None when there is none or it is unreadable."""
    raw = _read(path)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return {'unreadable': True}
    return data if isinstance(data, dict) else {'unreadable': True}


def write_state(path: str, state: dict) -> None:
    """Write the state file atomically."""
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(state, handle, indent=1)
    os.replace(tmp, path)


def clear_state(path: str, mark: Optional[str] = None) -> bool:
    """Remove the state file (only this session's when ``mark`` is given); return True if gone."""
    current = read_state(path)
    if current is None:
        return True
    if mark is not None and current.get('mark') != mark:
        return False
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    return True


_LOCK = []      # the open lock file of this supervisor, held for its whole life


def take_lock(state_file: str) -> bool:
    """
    Take the one-session lock beside the state file; return False when another holds it.

    An exclusive flock, held until this process ends (also when it is killed): two starts at
    the same moment cannot both pass. The file is not inherited by the children.
    """
    import fcntl
    path = os.path.join(os.path.dirname(state_file), 'lock')
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    handle = open(path, 'a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False
    _LOCK.append(handle)
    return True


def session_is_live(state: Optional[dict], proc_root: str = '/proc') -> bool:
    """Return True when a live process carries the state file's mark."""
    if not state or not state.get('mark'):
        return False
    return bool(marked(state['mark'], proc_root))
