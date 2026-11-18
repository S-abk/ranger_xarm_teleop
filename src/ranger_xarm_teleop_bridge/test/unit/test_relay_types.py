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
"""The vendored relay rebroadcasts the operator-feedback message types."""

import ast
import os

SERVER = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', 'third_party', 'vr-teleop-kit', 'src',
    'vr_teleop_kit', 'relay', 'server.py')


def relay_types():
    """Return the string members of RELAY_TYPES, read from the source (fastapi is not imported)."""
    with open(SERVER, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'RELAY_TYPES' for t in node.targets):
            assert isinstance(node.value, ast.Set)
            return {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    raise AssertionError('RELAY_TYPES not found in server.py')


def test_feedback_types_are_relayed():
    """Unknown types are echoed back to the sender, so the two new types must be listed."""
    types = relay_types()
    assert {'teleop_haptic', 'teleop_status'} <= types


def test_existing_types_are_still_relayed():
    """The input stream and the settings types keep being rebroadcast."""
    assert {'xr_frame', 'ik_state', 'config_update', 'request_settings'} <= relay_types()
