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
Keep a failed stack test's logs: its launch and process output and its nodes' own log files.

With ``TELEOP_TEST_KEEP_LOGS`` set to a directory, a test that fails (in setup or in its
body) gets a new subdirectory there with everything its module registered through
``stack_helpers`` (``Process`` output files, the ``ROS_LOG_DIR`` of each child environment).
Unset, nothing is copied, and the nodes' log files of a failed test are lost when the session
ends (the temporary ROS log directories are removed then; process output stays in pytest's own
temporary directories for a few runs). Keeping logs is opt-in: the project's test line sets it.
"""

import os
import shutil
import sys

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    # The stack tests import stack_helpers (ROS); the pure unit tests under test/unit must run
    # without ROS, so it is imported by the test modules, never here.
    helpers = sys.modules.get('stack_helpers')
    if helpers is None:
        return
    keep_logs, KEEP_LOGS_ENV = helpers.keep_logs, helpers.KEEP_LOGS_ENV
    destination = os.environ.get(KEEP_LOGS_ENV)
    if destination and report.failed and report.when in ('setup', 'call'):
        module = item.nodeid.split('::')[0]
        try:
            target = keep_logs(module, item.name, destination)
            report.sections.append(('kept logs', target))
            print(f'\nlogs of the failed test kept in {target}')
        except OSError as err:  # never turn a test failure into a different error
            print(f'\ncould not keep the logs: {err}')


def pytest_sessionfinish(session, exitstatus):
    """Remove the temporary ROS log directories (a failed test's were copied already)."""
    helpers = sys.modules.get('stack_helpers')
    if helpers is None:
        return
    for path in helpers.TEMP_LOG_DIRS:
        shutil.rmtree(path, ignore_errors=True)
