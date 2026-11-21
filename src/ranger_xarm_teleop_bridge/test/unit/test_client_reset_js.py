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
"""Run the reset-line tests of the headset panel (node) as part of the quick tier."""

import os
import subprocess

from test_client_feedback_js import node_or_skip

JS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'js'))


def test_reset_line_js_unit_tests():
    """The node:test suite for the reset line of feedback.js passes."""
    node = node_or_skip()
    proc = subprocess.run([node, '--test', os.path.join(JS_DIR, 'reset_line.test.js')],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
