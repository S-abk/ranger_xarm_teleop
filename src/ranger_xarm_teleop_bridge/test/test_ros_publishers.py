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
"""INV-5: topic names, types and QoS of the bridge's publishers match the interface contract."""

import pytest
from ranger_xarm_teleop_bridge.bridge_node import BridgeNode
import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

EXPECTED_TYPES = {
    '/teleop/input/left/pose': ['geometry_msgs/msg/PoseStamped'],
    '/teleop/input/right/pose': ['geometry_msgs/msg/PoseStamped'],
    '/teleop/input/joy': ['sensor_msgs/msg/Joy'],
    '/teleop/input/status': ['ranger_xarm_teleop_interfaces/msg/InputStatus'],
}
# (reliability, durability, history, depth) per topic, from the contract.
EXPECTED_QOS = {
    '/teleop/input/left/pose': (ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.VOLATILE,
                                HistoryPolicy.KEEP_LAST, 1),
    '/teleop/input/right/pose': (ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.VOLATILE,
                                 HistoryPolicy.KEEP_LAST, 1),
    '/teleop/input/joy': (ReliabilityPolicy.BEST_EFFORT, DurabilityPolicy.VOLATILE,
                          HistoryPolicy.KEEP_LAST, 1),
    '/teleop/input/status': (ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE,
                             HistoryPolicy.KEEP_LAST, 10),
}


@pytest.fixture(scope='module')
def node():
    rclpy.init()
    n = BridgeNode(node_name='bridge_publishers_under_test')
    yield n
    n.destroy_node()
    rclpy.shutdown()


def test_publishes_exactly_the_contract_topics(node):
    pubs = dict(node.get_publisher_names_and_types_by_node(node.get_name(), node.get_namespace()))
    teleop = {k: v for k, v in pubs.items() if k.startswith('/teleop')}
    assert teleop == EXPECTED_TYPES


def test_topic_qos_as_seen_on_the_graph(node):
    for topic, (reliability, durability, _history, _depth) in EXPECTED_QOS.items():
        infos = [i for i in node.get_publishers_info_by_topic(topic)
                 if i.node_name == node.get_name()]
        assert len(infos) == 1, topic
        assert infos[0].qos_profile.reliability == reliability, topic
        assert infos[0].qos_profile.durability == durability, topic


def test_topic_history_and_depth_on_the_publishers(node):
    """DDS discovery does not carry history/depth; check the publisher objects themselves."""
    publishers = {node.pose_pubs['left'].topic_name: node.pose_pubs['left'],
                  node.pose_pubs['right'].topic_name: node.pose_pubs['right'],
                  node.joy_pub.topic_name: node.joy_pub,
                  node.status_pub.topic_name: node.status_pub}
    assert set(publishers) == set(EXPECTED_QOS)
    for topic, (reliability, durability, history, depth) in EXPECTED_QOS.items():
        qos = publishers[topic].qos_profile
        assert (qos.reliability, qos.durability, qos.history, qos.depth) == \
            (reliability, durability, history, depth), topic


def test_default_parameters(node):
    assert node.get_parameter('relay_url').value == 'ws://127.0.0.1:8443/ws'
    assert node.get_parameter('frame_id').value == 'quest_world'
    assert node.get_parameter('record_path').value == ''
    assert node.get_parameter('source').value == 'relay'
