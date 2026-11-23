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
"""The base command and base state streams read with their own stamps (bag_io, needs rclpy)."""

import math

from geometry_msgs.msg import TwistStamped
from ranger_xarm_teleop_interfaces.msg import BaseState
from ranger_xarm_teleop_record import bag_io
from ranger_xarm_teleop_record import profile


def rows_of(name, msg, recv):
    """Return the rows ``bag_io`` makes of one message of the profile stream ``name``."""
    spec = profile.load_profile('gazebo')['streams'][name]
    rows = bag_io._Rows(name, spec)
    bag_io._add(rows, msg, recv)
    return rows


def test_a_stamped_twist_reads_with_its_header_stamp_and_its_twist():
    msg = TwistStamped()
    msg.header.stamp.sec = 12
    msg.header.stamp.nanosec = 500000000
    msg.header.frame_id = 'base_link'
    msg.twist.linear.x = 0.3
    msg.twist.linear.y = -0.1
    msg.twist.angular.z = 0.2
    rows = rows_of('base_cmd', msg, 12.75)
    assert rows.stamp == [12.5] and rows.recv == [12.75] and rows.frame_id == ['base_link']
    assert rows.fields['twist'] == [[0.3, -0.1, 0.0, 0.0, 0.0, 0.2]]


def test_a_base_state_reads_every_field_the_checker_and_the_report_use():
    msg = BaseState()
    msg.header.stamp.sec = 40
    msg.header.stamp.nanosec = 250000000
    msg.enabled = True
    msg.state = BaseState.STATE_DRIVING
    msg.mode = BaseState.MODE_TURN
    msg.reason = BaseState.REASON_RELEASED
    msg.driving = True
    msg.command.linear.x = 0.25
    msg.command.angular.z = -0.5
    msg.at_rest = False
    msg.odom_age = 0.04
    msg.active_limit = BaseState.LIMIT_CAP
    msg.tool_speed_scale = math.nan
    msg.event_count = 7
    rows = rows_of('base_state', msg, 40.3)
    assert rows.stamp == [40.25] and rows.recv == [40.3]
    f = rows.fields
    assert (f['state'], f['mode'], f['reason'], f['active_limit'], f['event_count']) == (
        [2], [2], [3], [1], [7])
    assert f['driving'] == [True] and f['at_rest'] == [False] and f['enabled'] == [True]
    assert f['odom_age'] == [0.04] and math.isnan(f['tool_speed_scale'][0])
    assert f['command'] == [[0.25, 0.0, 0.0, 0.0, 0.0, -0.5]]
