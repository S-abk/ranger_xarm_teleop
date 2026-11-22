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
"""G-17 on the mux's side: a teleop controller active without a switch by the mux."""

from ranger_xarm_teleop_servo.ownership import outside_activation

TELEOP = ['teleop_arm_controller']


def test_a_teleop_controller_the_mux_switched_is_not_outside():
    states = {'teleop_arm_controller': 'active', 'xarm6_traj_controller': 'inactive'}
    assert outside_activation(states, TELEOP, switched_by_mux=True) == []


def test_a_teleop_controller_active_without_the_mux_is_outside():
    states = {'teleop_arm_controller': 'active', 'xarm6_traj_controller': 'inactive'}
    assert outside_activation(states, TELEOP, switched_by_mux=False) == TELEOP


def test_inactive_or_missing_controllers_are_never_outside():
    for states in ({'teleop_arm_controller': 'inactive'}, {}, {'xarm6_traj_controller': 'active'}):
        assert outside_activation(states, TELEOP, switched_by_mux=False) == []


def test_with_a_gripper_pair_only_the_active_ones_are_named():
    names = ['teleop_arm_controller', 'teleop_gripper_controller']
    states = {'teleop_arm_controller': 'inactive', 'teleop_gripper_controller': 'active'}
    assert outside_activation(states, names, False) == ['teleop_gripper_controller']
