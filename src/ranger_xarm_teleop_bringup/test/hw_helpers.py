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
Helpers of the hardware-path stack tests (hardware-shaped mock, teleop_hardware.launch.py).

No limits file the hardware path would accept is shipped (not even as a fixture): each test
writes its own into its temporary directory, derived from the simulation-only values, and
it is deleted with that directory.
"""

import glob
import json
import os
import time

from ament_index_python.packages import get_package_share_directory
from ranger_xarm_teleop_bringup.limits import HARDWARE_REQUIRED, load_limits
from stack_helpers import Process
import yaml

SIM_LIMITS = os.path.join(get_package_share_directory('ranger_xarm_teleop_bringup'), 'config',
                          'SIMULATION_ONLY_limits.yaml')
# Low values for the tests, as a first hardware session would choose them (fractions of the
# simulation values); test material only, written to a temporary file per test.
FRACTIONS = {'linear_speed_cap_m_s': 0.4, 'angular_speed_cap_rad_s': 0.4,
             'max_offset_translation_m': 0.5, 'max_offset_rotation_deg': 0.5}


def write_hardware_limits(directory, **overrides):
    """Write a complete hardware limits file into ``directory``; return its path."""
    sim = load_limits(SIM_LIMITS)
    values = {}
    for name in HARDWARE_REQUIRED:
        if name == 'tool_pose_timeout_s':
            values[name] = 0.1
        else:
            values[name] = float(sim[name]) * FRACTIONS.get(name, 1.0)
    values['translation_scale_coarse'] = 1.0
    values.update(overrides)
    path = os.path.join(str(directory), 'hardware_limits_test.yaml')
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump({'/**': {'ros__parameters': values}}, handle)
    return path


def start_hwmock(env, stage, logs):
    """Start the hardware-shaped mock bring-up at ``stage``; return the ``Process``."""
    return Process(['ros2', 'launch', 'ranger_xarm_teleop_hardware', 'hwmock.launch.py',
                    f'stage:={stage}'], env, os.path.join(str(logs), f'hwmock_{stage}.log'))


def start_attach(env, stage, limits, logs, label, *extra):
    """Start teleop_hardware.launch.py on the running bring-up; return the ``Process``."""
    return Process(['ros2', 'launch', 'ranger_xarm_teleop_hardware',
                    'teleop_hardware.launch.py', f'stage:={stage}', f'limits_file:={limits}',
                    *extra], env, os.path.join(str(logs), f'attach_{label}.log'))


def wait_for_controllers(probe, expected, timeout=90.0):
    """Wait until the controller manager reports at least ``expected`` (name -> state)."""
    deadline = time.monotonic() + timeout
    seen = {}
    while time.monotonic() < deadline:
        try:
            seen = probe.controllers()
        except AssertionError:
            seen = {}
        if all(seen.get(k) == v for k, v in expected.items()):
            return seen
        time.sleep(0.5)
    raise AssertionError(f'controllers {seen}, expected {expected}')


def publishers_of(probe, topic):
    """Return the node names publishing ``topic``."""
    return sorted(i.node_name for i in probe.node.get_publishers_info_by_topic(topic))


def start_check(env, stage, logs, label, *extra):
    """Start the read-only check tool; return ``(Process, data dir)``."""
    data_dir = os.path.join(str(logs), f'checks_{label}')
    process = Process(['ros2', 'run', 'ranger_xarm_teleop_check', 'hw_check', stage,
                       '--data-dir', data_dir, *extra], env,
                      os.path.join(str(logs), f'check_{label}.log'))
    return process, data_dir


def check_report(process, data_dir, timeout):
    """Wait for the check tool; return ``(exit status, {rule: verdict}, report, output)``."""
    code = process.wait(timeout)
    output = process.output()
    assert code is not None, f'the check tool did not finish in {timeout} s:\n{output[-3000:]}'
    paths = sorted(glob.glob(os.path.join(data_dir, 'hardware_checks', '*.json')))
    assert paths, f'the check tool wrote no report (exit {code}):\n{output[-3000:]}'
    with open(paths[-1], encoding='utf-8') as handle:
        report = json.load(handle)
    return code, {r['rule']: r['verdict'] for r in report['rules']}, report, output
