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
tools/relay_setup.sh: a .venv without pip is rebuilt, and a failed venv says what to install.

The script runs from a copy in a temporary tree, with a stand-in ``python3`` first on PATH
(it makes, or fails to make, the venv) and a stand-in pip that only logs: nothing touches the
checkout's .venv or the network.
"""

import os
import shutil
import stat
import subprocess

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
SCRIPT = os.path.join(REPO, 'tools', 'relay_setup.sh')
HINT = 'sudo apt install python3-venv'

# The stand-in python3: ``-m venv DIR`` makes DIR/bin, with pip when FAKE_VENV says so.
FAKE_PYTHON = r"""#!/bin/sh
echo "python3 $*" >> "$FAKE_LOG"
[ "$1" = "-m" ] && [ "$2" = "venv" ] || exit 2
mkdir -p "$3/bin"
case "$FAKE_VENV" in
  ok)
    printf '#!/bin/sh\necho "pip $*" >> "%s"\n' "$FAKE_LOG" > "$3/bin/pip"
    chmod +x "$3/bin/pip" ;;
  nopip) ;;
  fail) exit 1 ;;
esac
"""


def _executable(path, text):
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


@pytest.fixture
def tree(tmp_path):
    os.makedirs(tmp_path / 'tools')
    shutil.copy(SCRIPT, tmp_path / 'tools' / 'relay_setup.sh')
    os.makedirs(tmp_path / 'third_party' / 'vr-teleop-kit')
    (tmp_path / 'third_party' / 'vr-teleop-kit' / 'requirements-relay.txt').write_text('')
    os.makedirs(tmp_path / 'fakebin')
    _executable(tmp_path / 'fakebin' / 'python3', FAKE_PYTHON)
    return tmp_path


def run(tree, venv_mode):
    env = dict(os.environ, PATH=f"{tree / 'fakebin'}:/usr/bin:/bin", FAKE_VENV=venv_mode,
               FAKE_LOG=str(tree / 'calls.log'))
    done = subprocess.run(['bash', str(tree / 'tools' / 'relay_setup.sh')], env=env,
                          capture_output=True, text=True, timeout=30, check=False)
    log = tree / 'calls.log'
    calls = log.read_text().splitlines() if log.exists() else []
    return done, calls


def test_a_venv_without_pip_is_made_again(tree):
    os.makedirs(tree / '.venv' / 'bin')
    _executable(tree / '.venv' / 'bin' / 'python', '#!/bin/sh\n')
    done, calls = run(tree, 'ok')
    assert done.returncode == 0, done.stderr
    assert 'incomplete' in done.stderr
    assert calls[0] == 'python3 -m venv .venv'
    assert calls[1].startswith('pip install -r third_party/vr-teleop-kit/')
    assert 'Relay environment ready' in done.stdout


def test_a_ready_venv_is_kept(tree):
    os.makedirs(tree / '.venv' / 'bin')
    _executable(tree / '.venv' / 'bin' / 'pip',
                f"#!/bin/sh\necho \"pip $*\" >> \"{tree / 'calls.log'}\"\n")
    done, calls = run(tree, 'fail')
    assert done.returncode == 0, done.stderr
    assert [c.split()[0] for c in calls] == ['pip']


def test_a_failed_venv_names_the_package_and_leaves_nothing(tree):
    done, calls = run(tree, 'fail')
    assert done.returncode != 0
    assert HINT in done.stderr
    assert not (tree / '.venv').exists()
    assert not any(c.startswith('pip') for c in calls)


def test_a_venv_made_without_pip_names_the_package_and_leaves_nothing(tree):
    done, _ = run(tree, 'nopip')
    assert done.returncode != 0
    assert HINT in done.stderr
    assert not (tree / '.venv').exists()
