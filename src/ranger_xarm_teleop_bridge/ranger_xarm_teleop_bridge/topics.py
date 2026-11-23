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
Names and QoS of the input and operator-feedback topics, shared by the bridge and its readers.

The arm-side nodes (the teleop node, the reset node) read the headset's topics by these names.
This module imports nothing outside ROS, so those nodes do not need the bridge's WebSocket
library; ``bridge_node`` re-exports every name here.
"""

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

HANDS = ('left', 'right')
POSE_TOPICS = {hand: f'/teleop/input/{hand}/pose' for hand in HANDS}
HEAD_POSE_TOPIC = '/teleop/input/head/pose'
JOY_TOPIC = '/teleop/input/joy'
STATUS_TOPIC = '/teleop/input/status'
HAPTIC_TOPIC = '/teleop/haptic'
OPERATOR_STATUS_TOPIC = '/teleop/operator/status'

# Interface contract: poses and joy best effort, depth 1; status reliable, depth 10.
POSE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                      depth=1, durability=DurabilityPolicy.VOLATILE)
JOY_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                     depth=1, durability=DurabilityPolicy.VOLATILE)
STATUS_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                        depth=10, durability=DurabilityPolicy.VOLATILE)

# Operator feedback in: as published by the operator feedback node.
HAPTIC_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                        depth=10, durability=DurabilityPolicy.VOLATILE)
OPERATOR_STATUS_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 history=HistoryPolicy.KEEP_LAST, depth=1,
                                 durability=DurabilityPolicy.VOLATILE)
