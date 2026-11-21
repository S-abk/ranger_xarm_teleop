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
"""INV-6 for the reset feedback module: no ROS imports, loads without ROS."""

import os
import subprocess
import sys

import test_no_ros_imports as base


def test_reset_feedback_source_has_no_ros_imports():
    source = base.source_of('reset_feedback')
    assert base.IMPORT_RE.search(source) is None
    for name in base.FORBIDDEN:
        assert f'import {name}' not in source and f'from {name}' not in source


def test_reset_feedback_loads_in_isolated_interpreter():
    code = (
        'import sys\n'
        f'sys.path.insert(0, {base.PKG_DIR!r})\n'
        'import ranger_xarm_teleop_bridge.reset_feedback\n'
        f'forbidden = {base.FORBIDDEN!r}\n'
        'loaded = [m for m in sys.modules if m.split(".")[0] in forbidden]\n'
        'assert not loaded, loaded\n'
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONPATH', 'AMENT_PREFIX_PATH', 'LD_LIBRARY_PATH')}
    result = subprocess.run([sys.executable, '-I', '-c', code], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
