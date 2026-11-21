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
The launcher runs without ROS where it must.

Its modules (all but the probe) import no ROS, and the front command runs on a machine
without ROS: the relay role's requirement (INV-8 for the pre-flight logic).
"""

import os
import re
import subprocess
import sys
import time

import pytest

PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CHECKOUT = os.path.dirname(os.path.dirname(PKG_DIR))
MODULES = ['plan', 'preflight', 'logwatch', 'procs', 'facts', 'supervisor', 'cli']
FORBIDDEN = ['rclpy', 'rcl_interfaces', 'sensor_msgs', 'std_msgs', 'ranger_xarm_teleop_interfaces',
             'rosidl_runtime_py', 'launch', 'launch_ros', 'ament_index_python',
             'ranger_xarm_teleop_record', 'yaml']
# Imported inside functions, for the stack role only (ROS sourced there).
LAZY = {'cli': {'ranger_xarm_teleop_record'}, 'supervisor': {'ament_index_python'}}
NO_ROS_ENV = {k: v for k, v in os.environ.items()
              if k not in ('PYTHONPATH', 'AMENT_PREFIX_PATH', 'LD_LIBRARY_PATH', 'ROS_DISTRO',
                           'COLCON_PREFIX_PATH', 'CMAKE_PREFIX_PATH')}
NO_ROS_ENV['PATH'] = ':'.join(p for p in os.environ.get('PATH', '').split(':')
                              if '/opt/ros/' not in p and '/install/' not in p)


def module_level(source, names):
    return re.search(r'^(?:from|import)\s+(?:%s)\b' % '|'.join(map(re.escape, names)), source,
                     re.MULTILINE)


@pytest.mark.parametrize('module', MODULES)
def test_no_module_level_ros_import(module):
    with open(os.path.join(PKG_DIR, 'ranger_xarm_teleop_session', f'{module}.py'),
              encoding='utf-8') as handle:
        source = handle.read()
    assert module_level(source, FORBIDDEN) is None
    for name in LAZY.get(module, ()):
        assert re.search(r'^\s+from %s\b' % re.escape(name), source, re.MULTILINE)


@pytest.mark.parametrize('module', MODULES)
def test_module_loads_without_ros(module):
    code = (f'import sys\nsys.path.insert(0, {PKG_DIR!r})\n'
            f'import ranger_xarm_teleop_session.{module}\n'
            f'bad = [m for m in sys.modules if m.split(".")[0] in {FORBIDDEN!r}]\n'
            'assert not bad, bad\n')
    done = subprocess.run([sys.executable, '-I', '-c', code], env=NO_ROS_ENV,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_the_front_command_runs_without_ros(tmp_path):
    env = dict(NO_ROS_ENV, TELEOP_SESSION_STATE_DIR=str(tmp_path))
    tool = os.path.join(CHECKOUT, 'tools', 'teleop_session')
    done = subprocess.run([sys.executable, '-I', tool, 'status'], env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'no session'
    done = subprocess.run([sys.executable, '-I', tool, 'start', '--help'], env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and '--role {stack,relay}' in done.stdout


@pytest.mark.skipif(not os.path.exists(os.path.join(CHECKOUT, '.venv', 'bin', 'python')),
                    reason="the relay's .venv is not set up (tools/relay_setup.sh)")
def test_the_relay_role_starts_and_stops_without_ros(tmp_path):
    env = {'HOME': os.environ.get('HOME', str(tmp_path)), 'PATH': '/usr/bin:/bin',
           'TELEOP_SESSION_STATE_DIR': str(tmp_path / 'state'),
           'ROS_LOG_DIR': str(tmp_path / 'log'), 'TELEOP_SESSION_NO_ADB': '1'}
    tool = os.path.join(CHECKOUT, 'tools', 'teleop_session')
    out = open(tmp_path / 'out.txt', 'w+')
    start = subprocess.Popen([sys.executable, '-I', tool, 'start', '--role', 'relay', '--port',
                              str(18470 + os.getpid() % 20), '--yes'], env=env, stdout=out,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    try:
        end = time.monotonic() + 30
        while 'listens on this machine only' not in open(tmp_path / 'out.txt').read():
            assert time.monotonic() < end and start.poll() is None, \
                open(tmp_path / 'out.txt').read()
            time.sleep(0.2)
        done = subprocess.run([sys.executable, '-I', tool, 'stop'], env=env,
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0 and 'PASS' in done.stdout, done.stdout + done.stderr
        assert start.wait(30) == 0
    finally:
        if start.poll() is None:
            start.kill()
            start.wait()
    text = open(tmp_path / 'out.txt').read()
    assert 'role relay: the relay only' in text and 'stopping relay' in text


def test_the_stack_role_says_plainly_that_it_needs_ros(tmp_path):
    env = dict(NO_ROS_ENV, TELEOP_SESSION_STATE_DIR=str(tmp_path),
               ROS_LOG_DIR=str(tmp_path / 'log'))
    tool = os.path.join(CHECKOUT, 'tools', 'teleop_session')
    done = subprocess.run([sys.executable, '-I', tool, 'start', '--profile', 'mock'], env=env,
                          capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert done.returncode == 2, done.stdout + done.stderr
    assert 'needs ROS sourced' in done.stdout
    assert not os.path.exists(tmp_path / 'state.json')
