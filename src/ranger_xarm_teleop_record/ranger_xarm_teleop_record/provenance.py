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
Provenance of a session: commits, dirty flags and fingerprints (no ROS).

Only names, commits and hashes are recorded, never an absolute path, a host or a user name,
so a session can be moved to another machine without carrying where it was made.
"""

import functools
import hashlib
import os
import shutil
import subprocess
from typing import Optional

from .metadata import git_root


def sha256_text(text: str) -> str:
    """Return the sha256 of a string (UTF-8)."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_file(path: str) -> str:
    """Return the sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


MASS_FILE_COPY = 'stability_mass.yaml'


def stability_record(mass_file: str, session_dir: str) -> Optional[dict]:
    """
    Return the optional sidecar key ``stability`` for the stability monitor's mass file.

    ``mass_file`` empty: None (the sidecar then has no such key). Otherwise the file is copied
    into ``session_dir`` as ``stability_mass.yaml`` (as the limits file is) and the record holds
    ``mass_file`` (that copy's name, relative to the session), ``mass_file_sha256`` and
    ``source_name`` (the original's file name, never its directory). A file that cannot be read
    gives ``mass_file`` and ``mass_file_sha256`` None and ``error`` saying why; the caller logs
    it.
    """
    if not mass_file:
        return None
    record = {'mass_file': None, 'mass_file_sha256': None,
              'source_name': os.path.basename(mass_file)}
    try:
        digest = sha256_file(mass_file)
        shutil.copyfile(mass_file, os.path.join(session_dir, MASS_FILE_COPY))
    except OSError as err:
        record['error'] = f'mass file {record["source_name"]} not recorded: {err}'
        return record
    record.update({'mass_file': MASS_FILE_COPY, 'mass_file_sha256': digest})
    return record


def _git(root: str, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(['git', '-C', root, *args], capture_output=True, text=True,
                              timeout=10.0, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def git_info(path: Optional[str]) -> dict:
    """
    Return {'name', 'commit', 'dirty', 'changed'} of the git tree holding ``path``.

    ``commit`` is None when ``path`` is not in a git tree (the checker then refuses the
    episode). ``changed`` lists the modified tracked files (relative), at most 50.
    """
    root = git_root(path) if path else None
    if root is None:
        return {'name': None, 'commit': None, 'dirty': None, 'changed': []}
    commit = (_git(root, 'rev-parse', 'HEAD') or '').strip() or None
    status = _git(root, 'status', '--porcelain', '--untracked-files=no')
    changed = [line[3:] for line in (status or '').splitlines() if line.strip()]
    return {'name': os.path.basename(root), 'commit': commit,
            'dirty': None if status is None else bool(changed), 'changed': changed[:50]}


def _package_root() -> str:
    """Return this package's directory (the source tree when installed with symlinks)."""
    return os.path.dirname(os.path.realpath(__file__))


@functools.lru_cache(maxsize=1)
def checking_code() -> dict:
    """
    Return what identifies the code that checks and exports: its git commit and a digest.

    ``commit``, ``dirty`` and ``changed`` come from the git tree holding this package (None
    when it is installed outside one). ``source_sha256`` covers every module of this package
    and its packaged profiles and tolerance file, so a verdict from uncommitted code can still
    be matched to the code that gave it. Computed once per process.
    """
    here = _package_root()
    files = []
    for folder in (here, os.path.join(os.path.dirname(here), 'profiles'),
                   os.path.join(os.path.dirname(here), 'config')):
        if os.path.isdir(folder):
            files += [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                      if f.endswith(('.py', '.yaml'))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(os.path.relpath(path, os.path.dirname(here)).encode('utf-8'))
        digest.update(sha256_file(path).encode('ascii'))
    info = git_info(here)
    return {'package': 'ranger_xarm_teleop_record', 'repo': info['name'],
            'commit': info['commit'], 'dirty': info['dirty'], 'changed': info['changed'],
            'source_sha256': digest.hexdigest(), 'files': len(files)}
