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
Recordings made before TeleopState grew its later fields still read (bag_io.LEGACY).

Each older definition is the current one without a suffix of fields; its bytes are the
current message's bytes without that suffix, as long as the suffix needs no alignment of its
own (a Time after a 4-aligned end, a string after a 4-aligned end, one-byte bools). The test
builds such an old message by cutting the current one's suffix, pads it as bag_io does, and
checks every field that existed then reads back unchanged.
"""

import pytest
from ranger_xarm_teleop_interfaces.msg import TeleopState
from ranger_xarm_teleop_record.bag_io import LEGACY
from rclpy.serialization import deserialize_message, serialize_message

TYPE = 'ranger_xarm_teleop_interfaces/msg/TeleopState'
NEWEST_OLDER = 'RIHS01_62c8052500962ebf0f9bdb7fdacd986e4cb56a91cee592a03daa4a52dc26e053'


def _current(reason=''):
    m = TeleopState()
    m.armed = True
    m.hold_reason = TeleopState.HOLD_CLUTCH_RELEASED
    m.gripper_measured = 0.25
    m.keepout_link = 'link'
    m.measured_tool_pose_stamp.sec = 0
    m.preview_reason = reason
    return m


@pytest.mark.parametrize('old_hash', sorted(LEGACY[TYPE]))
def test_an_older_teleop_state_reads_with_its_later_fields_not_recorded(old_hash):
    pad, missing = LEGACY[TYPE][old_hash]
    data = serialize_message(_current())
    # The suffix the older definition lacks is exactly the pad's length, and its bytes for
    # default values are the pad: cutting it gives the old layout.
    assert data[-len(pad):] == pad
    old = data[:-len(pad)]
    msg = deserialize_message(old + pad, TeleopState)
    assert msg.armed and msg.hold_reason == TeleopState.HOLD_CLUTCH_RELEASED
    assert msg.gripper_measured == 0.25 and msg.keepout_link == 'link'
    assert not msg.joint_acceleration_limited and not msg.stopping
    assert {'joint_acceleration_limited', 'stopping'} <= set(missing)


@pytest.mark.parametrize('reason', ['', 'abc', 'abcdefg'])
def test_the_newest_older_definition_reads_whatever_the_string_before_the_bools(reason):
    pad, _ = LEGACY[TYPE][NEWEST_OLDER]
    data = serialize_message(_current(reason))
    msg = deserialize_message(data[:-2] + pad, TeleopState)
    assert msg.preview_reason == reason and not msg.stopping
