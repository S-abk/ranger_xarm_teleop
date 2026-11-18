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
Give each parallel test worker (pytest-xdist) its own ROS domain.

The node tests call ``rclpy.init()`` on the session's domain. Run in one process, as
before, they keep it. Run by several workers at once, each worker that collects any ROS test
moves to a domain of its own, derived from the session's: worker ``i`` of this package takes
offset 40 + i (bridge 20-39, servo 40-59; the stack tests' modules use 1-12) in the band
10-99, so never 0-9. Discovery is forced to this machine. A worker that cannot have its own
domain fails; it never falls back to a default one. The pure-Python unit tests need no ROS and
are left as they are.
"""

import os

import pytest

DOMAIN_OFFSET = 40          # this package's first worker offset from the session's domain
MAX_WORKERS = 20            # the next package's range starts DOMAIN_OFFSET + 20 further on
SESSION_DOMAIN = 'TELEOP_SESSION_ROS_DOMAIN'    # the session's own domain, kept aside


def band_domain(base: int, offset: int) -> int:
    """Return the domain ``offset`` places after ``base`` in the band 10-99."""
    return 10 + (base - 10 + offset) % 90


def worker_domain(base: int, index: int) -> int:
    """Return worker ``index``'s domain for the session's domain ``base``."""
    return band_domain(base, DOMAIN_OFFSET + index)


def move_worker(environ, worker_id: str) -> int:
    """Move a worker's ROS settings in ``environ`` to its own domain; return the domain."""
    index = int(worker_id.lstrip('gw'))
    if index >= MAX_WORKERS:
        raise pytest.UsageError(f'at most {MAX_WORKERS} workers: each needs its own ROS domain')
    # Another package's conftest in the same run must not shift an already shifted domain.
    raw = environ.get(SESSION_DOMAIN, environ.get('ROS_DOMAIN_ID', ''))
    if not raw.isdigit() or not 10 <= int(raw) <= 99:
        raise pytest.UsageError(
            'parallel ROS tests need ROS_DOMAIN_ID set to the domain of this checkout, 10-99 '
            f'(got {raw!r}); each worker derives its own domain from it')
    environ[SESSION_DOMAIN] = raw
    environ['ROS_DOMAIN_ID'] = str(worker_domain(int(raw), index))
    environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    environ.pop('ROS_STATIC_PEERS', None)
    return int(environ['ROS_DOMAIN_ID'])


def pytest_collection_modifyitems(config, items):
    workerinput = getattr(config, 'workerinput', None)
    if workerinput is None:
        return                                   # not a parallel worker: unchanged
    if all(f'{os.sep}unit{os.sep}' in str(item.path) for item in items):
        return                                   # pure-Python tests only: no ROS domain used
    move_worker(os.environ, workerinput['workerid'])
