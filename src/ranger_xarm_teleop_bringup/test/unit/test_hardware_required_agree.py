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
"""The launch's and the teleop node's lists of hardware values are the same list."""

from ranger_xarm_teleop_bringup.limits import HARDWARE_REQUIRED, RESERVED
from ranger_xarm_teleop_servo import params


def test_launch_and_node_require_the_same_hardware_values():
    assert set(HARDWARE_REQUIRED) == set(params.HARDWARE_REQUIRED)


def test_the_hardware_guards_cannot_be_set_by_an_extra_file():
    assert 'hardware_limits' in RESERVED and 'refuse_teleop' in RESERVED
