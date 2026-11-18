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
The ROS domains of parallel test workers and stack test modules (pytest-xdist).

Every domain is derived from the session's own; for every session domain 10-99 no two
workers of the bridge and servo packages, no stack test module and the session itself share
one, and none is in 0-9. A worker without the session's domain fails.
"""

import importlib.util
import os
import re

import pytest

HERE = os.path.dirname(os.path.realpath(__file__))
SRC = os.path.dirname(os.path.dirname(HERE))
BRINGUP_TEST = os.path.join(SRC, 'ranger_xarm_teleop_bringup', 'test')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVO = _load('servo_test_conftest', os.path.join(HERE, 'conftest.py'))
BRIDGE = _load('bridge_test_conftest',
               os.path.join(SRC, 'ranger_xarm_teleop_bridge', 'test', 'conftest.py'))
STACK = _load('bringup_stack_helpers', os.path.join(BRINGUP_TEST, 'stack_helpers.py'))


def _stack_offsets():
    """Return every offset a stack test module asks child_environment for."""
    found = set()
    for name in os.listdir(BRINGUP_TEST):
        if name.startswith('test_') and name.endswith('.py'):
            with open(os.path.join(BRINGUP_TEST, name), encoding='utf-8') as handle:
                found |= {int(v) for v in re.findall(r'child_environment\(offset=(\d+)\)',
                                                     handle.read())}
    return sorted(found)


def test_the_stack_modules_ask_for_the_offsets_this_test_expects():
    assert _stack_offsets() == list(range(1, 13))


@pytest.mark.parametrize('base', range(10, 100))
def test_no_two_workers_stack_modules_or_the_session_share_a_domain(base):
    domains = [base]                                                     # serial node tests
    domains += [STACK.band_domain(base, offset) for offset in _stack_offsets()]
    domains += [BRIDGE.worker_domain(base, i) for i in range(BRIDGE.MAX_WORKERS)]
    domains += [SERVO.worker_domain(base, i) for i in range(SERVO.MAX_WORKERS)]
    assert len(domains) == len(set(domains))
    assert all(10 <= d <= 99 for d in domains)


@pytest.mark.parametrize('environ', [{}, {'ROS_DOMAIN_ID': ''}, {'ROS_DOMAIN_ID': '0'},
                                     {'ROS_DOMAIN_ID': '9'}, {'ROS_DOMAIN_ID': '100'},
                                     {'ROS_DOMAIN_ID': 'x'}])
def test_a_worker_without_the_sessions_domain_fails(environ):
    for conftest in (SERVO, BRIDGE):
        with pytest.raises(pytest.UsageError, match='ROS_DOMAIN_ID'):
            conftest.move_worker(dict(environ), 'gw0')


def test_a_worker_beyond_the_packages_range_fails():
    with pytest.raises(pytest.UsageError, match='workers'):
        SERVO.move_worker({'ROS_DOMAIN_ID': '50'}, f'gw{SERVO.MAX_WORKERS}')


def test_a_worker_keeps_discovery_on_this_machine_and_both_packages_derive_from_the_session():
    environ = {'ROS_DOMAIN_ID': '77', 'ROS_AUTOMATIC_DISCOVERY_RANGE': 'SUBNET',
               'ROS_STATIC_PEERS': '192.0.2.1'}
    assert SERVO.move_worker(environ, 'gw3') == SERVO.worker_domain(77, 3)
    assert environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] == 'LOCALHOST'
    assert 'ROS_STATIC_PEERS' not in environ
    # The other package's conftest in the same worker starts again from the session's domain.
    assert BRIDGE.move_worker(environ, 'gw3') == BRIDGE.worker_domain(77, 3)


def test_a_parallel_stack_module_without_the_sessions_domain_fails(monkeypatch):
    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw0')
    monkeypatch.delenv('ROS_DOMAIN_ID', raising=False)
    with pytest.raises(AssertionError, match='ROS_DOMAIN_ID'):
        STACK.child_environment(offset=1)
    monkeypatch.setenv('ROS_DOMAIN_ID', '77')
    env = STACK.child_environment(offset=1)
    assert env['ROS_DOMAIN_ID'] == '78' and env['ROS_AUTOMATIC_DISCOVERY_RANGE'] == 'LOCALHOST'
