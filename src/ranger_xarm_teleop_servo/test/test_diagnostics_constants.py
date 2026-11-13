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
"""The codes the pure diagnostics modules name are those of the message definitions."""

from moveit_msgs.msg import ServoStatus
from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
from ranger_xarm_teleop_servo import halts, session, workspace
from ranger_xarm_teleop_servo.teleop_node import SERVO_HALT_CODES


def constants(message, prefix):
    return {getattr(message, name): name[len(prefix):] for name in dir(message)
            if name.startswith(prefix)}


def test_hold_reasons_match_teleop_state():
    assert halts.HOLD_REASON == constants(TeleopState, 'HOLD_')
    assert halts.HOLD_SERVO_HALTED == TeleopState.HOLD_SERVO_HALTED


def test_servo_status_codes_match_moveit_msgs():
    names = {getattr(ServoStatus, name): name for name in dir(ServoStatus)
             if name.isupper() and isinstance(getattr(ServoStatus, name), int)}
    assert halts.SERVO_STATUS == names
    assert tuple(sorted(halts.HALT_CODES)) == tuple(sorted(SERVO_HALT_CODES))
    assert (workspace.NO_WARNING, workspace.APPROACHING, workspace.HALT, workspace.LEAVING) == (
        ServoStatus.NO_WARNING, ServoStatus.DECELERATE_FOR_APPROACHING_SINGULARITY,
        ServoStatus.HALT_FOR_SINGULARITY, ServoStatus.DECELERATE_FOR_LEAVING_SINGULARITY)


def test_owner_codes_match_arm_owner():
    assert (session.OWNER_NONE, session.OWNER_MOVEIT, session.OWNER_TELEOP,
            session.OWNER_SWITCHING) == (ArmOwner.OWNER_NONE, ArmOwner.OWNER_MOVEIT,
                                         ArmOwner.OWNER_TELEOP, ArmOwner.OWNER_SWITCHING)
