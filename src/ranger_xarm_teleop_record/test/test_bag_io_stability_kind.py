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
"""The stability state read with its own stamp and every field the checker uses (needs rclpy)."""

import math

from ranger_xarm_teleop_interfaces.msg import StabilityState
from ranger_xarm_teleop_record import bag_io
from ranger_xarm_teleop_record import profile


def rows_of(msg, recv):
    """Return the rows ``bag_io`` makes of one message of the gazebo stream."""
    spec = profile.load_profile('gazebo')['streams']['stability_state']
    rows = bag_io._Rows('stability_state', spec)
    bag_io._add(rows, msg, recv)
    return rows


def test_a_stability_state_reads_every_field_the_checker_uses():
    msg = StabilityState()
    msg.header.stamp.sec = 30
    msg.header.stamp.nanosec = 500000000
    msg.header.frame_id = 'base_link'
    msg.available = True
    msg.reason = StabilityState.REASON_OK
    msg.reason_text = 'fine'
    msg.tilt_source = StabilityState.TILT_LEVEL_ASSUMED
    msg.margin_m = 0.21
    msg.edge = StabilityState.EDGE_REAR_LEFT
    msg.margin_pessimistic_m = 0.18
    msg.edge_pessimistic = StabilityState.EDGE_REAR
    msg.stop_margin_m = 0.12
    msg.stop_margin_pessimistic_m = 0.1
    msg.stop_edge = StabilityState.EDGE_FRONT
    msg.stop_decel_m_s2 = 0.6
    msg.energy_reserve_j = 14.0
    msg.energy_edge = StabilityState.EDGE_FRONT_RIGHT
    msg.speed_m_s = 0.3
    msg.at_rest = False
    msg.model_sha256 = 'c' * 64
    rows = rows_of(msg, 30.6)
    assert rows.stamp == [30.5] and rows.recv == [30.6] and rows.frame_id == ['base_link']
    f = rows.fields
    assert f['available'] == [True] and f['at_rest'] == [False]
    assert (f['reason'], f['tilt_source'], f['edge'], f['edge_pessimistic'], f['stop_edge'],
            f['energy_edge']) == ([0], [2], [4], [5], [1], [8])
    assert f['margin_m'] == [0.21] and f['margin_pessimistic_m'] == [0.18]
    assert f['stop_margin_m'] == [0.12] and f['stop_margin_pessimistic_m'] == [0.1]
    assert f['stop_decel_m_s2'] == [0.6] and f['energy_reserve_j'] == [14.0]
    assert f['speed_m_s'] == [0.3]
    assert rows.labels['reason_text'] == ['fine'] and rows.labels['model_sha256'] == ['c' * 64]


def test_an_unavailable_state_keeps_its_nan_figures_and_its_reason():
    msg = StabilityState()
    msg.available = False
    msg.reason = StabilityState.REASON_NO_MASS_FILE
    msg.margin_m = math.nan
    rows = rows_of(msg, 1.0)
    assert rows.fields['available'] == [False] and rows.fields['reason'] == [3]
    assert math.isnan(rows.fields['margin_m'][0])
    assert rows.frame_id == ['']


def test_the_kind_is_the_last_of_the_tables():
    assert list(bag_io.KIND_FIELDS)[-1] == 'stability_state'
    assert list(bag_io.KIND_FIELDS)[-2] == 'base_state'
    assert 'ranger_xarm_teleop_interfaces/msg/StabilityState' not in bag_io.LEGACY
