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
"""INV-11: the pure modules import no ROS packages and load without ROS."""

import os
import re
import subprocess
import sys

import pytest

PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BRIDGE_DIR = os.path.join(os.path.dirname(PKG_DIR), 'ranger_xarm_teleop_bridge')
MODULES = ['rotations', 'mapping', 'calibration', 'params', 'gate', 'ownership', 'frames',
           'static_frames', 'simcheck', 'kinematics', 'halts', 'workspace', 'session']
FORBIDDEN = ['rclpy', 'geometry_msgs', 'sensor_msgs', 'std_msgs', 'std_srvs', 'tf2_ros',
             'moveit_msgs', 'control_msgs', 'controller_manager_msgs', 'action_msgs',
             'trajectory_msgs', 'ranger_xarm_teleop_interfaces', 'scipy']
IMPORT_RE = re.compile(r'^\s*(?:from|import)\s+(?:%s)\b' % '|'.join(FORBIDDEN), re.MULTILINE)


def source_of(module):
    path = os.path.join(PKG_DIR, 'ranger_xarm_teleop_servo', f'{module}.py')
    with open(path, encoding='utf-8') as handle:
        return handle.read()


@pytest.mark.parametrize('module', MODULES)
def test_source_has_no_ros_imports(module):
    assert IMPORT_RE.search(source_of(module)) is None
    for name in FORBIDDEN:
        assert f'import {name}' not in source_of(module)
        assert f'from {name}' not in source_of(module)


def test_import_pattern_detects_forbidden_imports():
    assert IMPORT_RE.search('import rclpy')
    assert IMPORT_RE.search('    from tf2_ros import Buffer')
    assert IMPORT_RE.search('import numpy') is None


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
