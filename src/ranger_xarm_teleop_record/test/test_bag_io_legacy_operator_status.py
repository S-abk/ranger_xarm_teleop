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
"""Recordings made before OperatorStatus grew its reset part still read (bag_io.LEGACY)."""

from ranger_xarm_teleop_interfaces.msg import OperatorStatus
from ranger_xarm_teleop_record.bag_io import LEGACY
from rclpy.serialization import deserialize_message

TYPE = 'ranger_xarm_teleop_interfaces/msg/OperatorStatus'
OLD_HASH = 'RIHS01_3f696d9029b87656c81b77a84b7ccf0b499ef5edeaa9e87fc4a5fc5de8507759'
# One OperatorStatus serialized with the definition before the reset part (built from the
# interfaces package at that commit): hold_text 28 x 'x', last_refusal_text 'rr',
# episode_index 5, disk_minutes_left 49.4883412249289.
OLD = bytes.fromhex(
    '0001000000000000000000000100000000000000f61b7ef172dec74101000000000000000000000000000000'
    '00000000000000000000000000000000000000001d0000007878787878787878787878787878787878787878'
    '78787878787878780000790003000000797900000000000000000000007a7a7a060000007a7a7a7a7a000000'
    '0000000002000000670000006700000000000000000000000000000000000000000000000000000000000000'
    '0500000000000000000000000000000000000000000000000100000000656565650000000000000000000000'
    '030000007272007200000000000000000000000000000000da2d1bf781be4840')


def test_an_old_operator_status_reads_with_the_reset_part_not_recorded():
    pad, missing = LEGACY[TYPE][OLD_HASH]
    msg = deserialize_message(OLD + pad, OperatorStatus)
    assert msg.hold_text == 'x' * 28 and msg.last_refusal_text == 'rr'
    assert msg.episode_index == 5 and msg.disk_minutes_left == 49.4883412249289
    assert not msg.reset_valid and msg.reset_text == ''
    assert set(missing) == {'reset_valid', 'reset_age', 'reset_state', 'reset_reason',
                            'reset_text', 'reset_event_age', 'base_valid', 'base_age',
                            'base_state', 'base_reason', 'base_text', 'base_event_age'}


def test_without_the_padding_an_old_operator_status_does_not_read():
    try:
        deserialize_message(OLD, OperatorStatus)
    except Exception:                   # noqa: B902 - rclpy raises its own RMW error type
        return
    raise AssertionError('an old OperatorStatus read without its padding')
