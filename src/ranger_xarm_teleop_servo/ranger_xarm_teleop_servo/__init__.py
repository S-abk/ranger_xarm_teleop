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
Teleop servo: clutch-relative mapping, safety gate, command mux.

Pure modules (no ROS imports, unit-tested without ROS): ``rotations``, ``mapping``,
``calibration``, ``params``, ``gate``, ``ownership``, ``frames``. ROS wrappers:
``teleop_node``, ``arm_mux_node``.
"""
