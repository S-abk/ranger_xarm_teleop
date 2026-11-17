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
"""INV-9: the keep-out and retreat modules import no ROS packages and load without ROS."""

import pytest
import test_servo_no_ros_imports as purity

MODULES = ['keepout', 'retreat']


@pytest.mark.parametrize('module', MODULES)
def test_source_has_no_ros_imports(module):
    purity.test_source_has_no_ros_imports(module)


@pytest.mark.parametrize('module', MODULES)
def test_module_loads_in_isolated_interpreter(module):
    purity.test_module_loads_in_isolated_interpreter(module)
