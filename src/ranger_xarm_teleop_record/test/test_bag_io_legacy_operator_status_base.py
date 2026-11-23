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
"""Recordings made before OperatorStatus grew its base part still read (bag_io.LEGACY)."""

from ranger_xarm_teleop_interfaces.msg import OperatorStatus
from ranger_xarm_teleop_record.bag_io import LEGACY
from rclpy.serialization import deserialize_message

TYPE = 'ranger_xarm_teleop_interfaces/msg/OperatorStatus'
OLD_HASH = 'RIHS01_9c96c4ec27f83ebf3bb6635dd5e587559d28263820c90dd5ebb4e15c7516b66c'
# One OperatorStatus serialized with the definition before the base part (built from the
# interfaces package at origin/main 124abb6): hold_text 28 x 'x', last_refusal_text 'rr',
# episode_index 5, disk_minutes_left 49.4883412249289, reset_valid True, reset_age 0.25,
# reset_state 4, reset_reason 30, reset_text 'Reset: moving', reset_event_age 1.5.
OLD = bytes.fromhex(
    '0001000000000000000000000100000000016412000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000086e0b1d0000007878787878787878787878787878787878787878'
    '7878787878787878000000000100000000000000000000000000000000000000010000000000000001000000'
    '00047c0f7c026409000000000000000000047c0f7c026404000000000000000000086a1f0500000000000000'
    '000000000000000000000000000000000100000000087c085300744100000000000000000300000072720000'
    '00000000000000000000000000000000da2d1bf781be48400100000000000000000000000000d03f041e7c0a'
    '0e00000052657365743a206d6f76696e67000000000000000000f83f')


def test_an_operator_status_before_the_base_part_reads_with_it_not_recorded():
    pad, missing = LEGACY[TYPE][OLD_HASH]
    msg = deserialize_message(OLD + pad, OperatorStatus)
    assert msg.hold_text == 'x' * 28 and msg.last_refusal_text == 'rr'
    assert msg.episode_index == 5 and msg.disk_minutes_left == 49.4883412249289
    assert msg.reset_valid and msg.reset_state == 4 and msg.reset_reason == 30
    assert msg.reset_text == 'Reset: moving' and msg.reset_event_age == 1.5
    assert not msg.base_valid and msg.base_text == ''
    assert set(missing) == {'base_valid', 'base_age', 'base_state', 'base_reason', 'base_text',
                            'base_event_age'}


def test_without_the_padding_an_operator_status_before_the_base_part_does_not_read():
    try:
        deserialize_message(OLD, OperatorStatus)
    except Exception:                   # noqa: B902 - rclpy raises its own RMW error type
        return
    raise AssertionError('an OperatorStatus without the base part read without its padding')
