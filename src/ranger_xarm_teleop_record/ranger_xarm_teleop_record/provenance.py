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

import hashlib
import os
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
