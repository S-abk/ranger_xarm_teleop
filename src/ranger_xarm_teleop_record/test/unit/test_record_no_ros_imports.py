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
"""The record package's pure modules import no ROS packages and load without ROS."""

import os
import re
import subprocess
import sys

import pytest

PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# The pure modules of the servo and bridge packages they use (rotations, static frames,
# the Joy layout) are pure as well.
SERVO_DIR = os.path.join(os.path.dirname(PKG_DIR), 'ranger_xarm_teleop_servo')
BRIDGE_DIR = os.path.join(os.path.dirname(PKG_DIR), 'ranger_xarm_teleop_bridge')
MODULES = ['streams', 'profile', 'metadata', 'checker', 'consistency', 'export', 'episode',
           'extrinsics', 'linkstats', 'provenance', 'sim_check']
FORBIDDEN = ['rclpy', 'geometry_msgs', 'sensor_msgs', 'std_msgs', 'std_srvs', 'tf2_ros',
             'tf2_msgs', 'nav_msgs', 'moveit_msgs', 'control_msgs', 'controller_manager_msgs',
             'action_msgs', 'trajectory_msgs', 'rcl_interfaces', 'builtin_interfaces',
             'ranger_xarm_teleop_interfaces', 'rosbag2_py', 'rosidl_runtime_py', 'scipy',
             'cv2', 'ament_index_python']
# profile.py imports ament_index_python inside a function, for the installed package only.
ALLOWED = {'profile': {'ament_index_python'}}


def pattern(names):
    """Return the regex matching an import of any of ``names``."""
    return re.compile(r'^\s*(?:from|import)\s+(?:%s)\b' % '|'.join(names), re.MULTILINE)


def source_of(module):
    path = os.path.join(PKG_DIR, 'ranger_xarm_teleop_record', f'{module}.py')
    with open(path, encoding='utf-8') as handle:
        return handle.read()


@pytest.mark.parametrize('module', MODULES)
def test_source_has_no_ros_imports(module):
    forbidden = [n for n in FORBIDDEN if n not in ALLOWED.get(module, set())]
    assert pattern(forbidden).search(source_of(module)) is None


def test_the_allowed_import_is_inside_a_function():
    # Only the guarded import in profile.py is let through, and it is indented (not module level).
    lines = [line for line in source_of('profile').splitlines()
             if re.match(r'\s*(?:from|import)\s+ament_index_python\b', line)]
    assert len(lines) == 1 and lines[0].startswith(' ')


def test_import_pattern_detects_forbidden_imports():
    assert pattern(FORBIDDEN).search('import rclpy')
    assert pattern(FORBIDDEN).search('    from rosbag2_py import SequentialReader')
    assert pattern(FORBIDDEN).search('import cv2')
    assert pattern(FORBIDDEN).search('import numpy') is None


@pytest.mark.parametrize('module', MODULES)
def test_module_loads_in_isolated_interpreter(module):
    code = (
        'import sys\n'
        f'sys.path[:0] = [{PKG_DIR!r}, {SERVO_DIR!r}, {BRIDGE_DIR!r}]\n'
        f'import ranger_xarm_teleop_record.{module}\n'
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
