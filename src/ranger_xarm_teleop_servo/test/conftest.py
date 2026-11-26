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
Node tests in parallel (pytest-xdist): each worker keeps its ROS graph inside its own process.

A serial run is left exactly as the environment sets it. In a worker, automatic discovery is
switched off before any test starts ROS: the nodes of one test still see each other (one
process), and nothing else, not another worker, another checkout or a running stack, even on
the same domain. The domain is never changed. If discovery cannot be shown to be off, every
test of the worker errors with the reason; none runs with discovery on.
"""

import ctypes
import os

import pytest

#: Cleared in a worker: each would name peers to find, and no test needs one.
CLEARED = ('ROS_STATIC_PEERS', 'ROS_LOCALHOST_ONLY')
#: Refused in a worker: each configures discovery or transports inside the middleware (a
#: discovery server, an XML or JSON profile, the built-in transports), which switching ROS
#: discovery off is not known to override.
REFUSED = ('ROS_DISCOVERY_SERVER', 'ROS_SUPER_CLIENT', 'FASTRTPS_DEFAULT_PROFILES_FILE',
           'FASTDDS_DEFAULT_PROFILES_FILE', 'FASTDDS_ENVIRONMENT_FILE',
           'FASTDDS_BUILTIN_TRANSPORTS', 'CYCLONEDDS_URI')
#: Fast DDS loads this profile from the working directory when no profile file is named.
DEFAULT_PROFILE = 'DEFAULT_FASTRTPS_PROFILES.xml'
#: The only middleware for which discovery off has been shown to hide other processes (the
#: two-process test); another one (its own discovery settings, a router) is refused.
MIDDLEWARE = ('', 'rmw_fastrtps_cpp')
#: RMW_AUTOMATIC_DISCOVERY_RANGE_OFF (rmw/discovery_options.h).
RANGE_OFF = 1

_problem = None     # why discovery could not be switched off in this worker, or None


class _DiscoveryOptions(ctypes.Structure):
    """rmw_discovery_options_t: the range is the first field; the rest is room for the others."""

    _fields_ = [('automatic_discovery_range', ctypes.c_int), ('static_peers', ctypes.c_void_p),
                ('static_peers_count', ctypes.c_size_t), ('allocator', ctypes.c_byte * 64)]


def _rclpy_started():
    """Return whether rclpy's default context is initialised (False without rclpy at all)."""
    try:
        import rclpy
    except ImportError:
        return False
    return rclpy.ok()


def _range_rcl_reads():
    """Return the discovery range rcl_init will take from this environment, or None without rcl."""
    try:
        import rclpy  # noqa: F401  (loads librcl; without rclpy no test can start ROS)
    except ImportError:
        return None
    rcl = ctypes.CDLL('librcl.so')
    options = _DiscoveryOptions()
    if rcl.rcl_get_automatic_discovery_range(ctypes.byref(options)) != 0:
        raise RuntimeError('rcl_get_automatic_discovery_range failed')
    return options.automatic_discovery_range


def pytest_configure(config):
    """Switch discovery off in an xdist worker, before any test module starts ROS."""
    global _problem
    config.addinivalue_line('markers', 'real_table_build: the node starts the keep-out '
                            'table builder process itself')
    if 'PYTEST_XDIST_WORKER' not in os.environ:
        return
    refused = [name for name in REFUSED if os.environ.get(name)]
    if os.path.exists(DEFAULT_PROFILE):
        refused.append(os.path.abspath(DEFAULT_PROFILE))
    if os.environ.get('RMW_IMPLEMENTATION', '') not in MIDDLEWARE:
        refused.append('RMW_IMPLEMENTATION=' + os.environ['RMW_IMPLEMENTATION'])
    if refused:
        _problem = ('refused in a parallel worker (it may re-enable discovery): '
                    + ', '.join(refused))
        return
    if _rclpy_started():
        _problem = 'rclpy was initialised before discovery could be switched off'
        return
    for name in CLEARED:
        os.environ.pop(name, None)
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'OFF'


@pytest.fixture(scope='session', autouse=True)
def _discovery_off_in_parallel_workers():
    """Error every test of a worker in which discovery is not off; nothing in a serial run."""
    if 'PYTEST_XDIST_WORKER' not in os.environ:
        return
    if _problem is not None:
        pytest.fail(f'discovery not switched off: {_problem}', pytrace=False)
    if _rclpy_started():
        pytest.fail('discovery not switched off: rclpy was initialised before the first test',
                    pytrace=False)
    leftover = [name for name in CLEARED if os.environ.get(name)]
    seen = _range_rcl_reads()
    if leftover or seen not in (None, RANGE_OFF):
        pytest.fail(f'discovery not switched off: rcl reads range {seen}, still set: {leftover}',
                    pytrace=False)


@pytest.fixture(scope='session', autouse=True)
def _keepout_table_cache(tmp_path_factory):
    """Keep the link keep-out's clearance tables of the tests out of the user's cache."""
    before = os.environ.get('XDG_CACHE_HOME')
    os.environ['XDG_CACHE_HOME'] = str(tmp_path_factory.mktemp('cache'))
    yield
    if before is None:
        os.environ.pop('XDG_CACHE_HOME', None)
    else:
        os.environ['XDG_CACHE_HOME'] = before


@pytest.fixture(autouse=True)
def _keepout_tables_in_process(request, monkeypatch):
    """
    Node tests of made-up robots: coarse tables (TEST_CONFIG), built in the test's process.

    The builder process finds meshes through the ament index, which the made-up robots' meshes
    are not in. A test marked ``real_table_build`` runs the node's own path (the process).
    """
    import sys
    node = sys.modules.get('ranger_xarm_teleop_servo.teleop_node')
    if node is None:
        yield
        return
    from ranger_xarm_teleop_servo import keepout_tables as kt
    monkeypatch.setattr(node, 'TABLE_CONFIG', kt.TEST_CONFIG)
    if request.node.get_closest_marker('real_table_build') is None:
        def build_here(self, plan, cache):
            kt.build_missing(plan, cache, 1, lambda text: None)
            self._make_keepout(plan, kt.load(plan, cache), 'built in the test')
        monkeypatch.setattr(node.TeleopNode, '_start_table_build', build_here)
    yield
