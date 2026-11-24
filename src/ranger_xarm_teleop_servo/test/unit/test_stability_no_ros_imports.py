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
"""The stability monitor's pure modules import no ROS packages and load without ROS (INV-6)."""

import os
import subprocess
import sys

import pytest
from test_servo_no_ros_imports import BRIDGE_DIR, FORBIDDEN, IMPORT_RE, PKG_DIR, source_of  # noqa

MODULES = ['stability', 'stability_filters', 'stability_mass', 'stability_model',
           'stability_monitor']


@pytest.mark.parametrize('module', MODULES)
def test_source_has_no_ros_imports(module):
    assert IMPORT_RE.search(source_of(module)) is None
    for name in FORBIDDEN:
        assert f'import {name}' not in source_of(module)
        assert f'from {name}' not in source_of(module)


@pytest.mark.parametrize('module', MODULES)
def test_module_loads_in_isolated_interpreter(module):
    code = (
        'import sys\n'
        f'sys.path[:0] = [{PKG_DIR!r}, {BRIDGE_DIR!r}]\n'
        f'import ranger_xarm_teleop_servo.{module}\n'
        f'forbidden = {FORBIDDEN!r}\n'
        'loaded = [m for m in sys.modules if m.split(".")[0] in forbidden]\n'
        'assert not loaded, loaded\n'
        'assert not any("/opt/ros" in p for p in sys.path), sys.path\n'
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONPATH', 'AMENT_PREFIX_PATH', 'LD_LIBRARY_PATH')}
    result = subprocess.run([sys.executable, '-I', '-c', code], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
