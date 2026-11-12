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
"""Make this package and the bridge package importable when pytest runs without colcon."""

import os
import sys

_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_BRIDGE_DIR = os.path.join(os.path.dirname(_PKG_DIR), 'ranger_xarm_teleop_bridge')
for _path in (_BRIDGE_DIR, _PKG_DIR, os.path.dirname(__file__)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
