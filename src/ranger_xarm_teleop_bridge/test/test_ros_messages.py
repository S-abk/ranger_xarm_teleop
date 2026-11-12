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
"""Message-level checks of the bridge node: INV-1 to INV-4, INV-7 and the identical-stamp rule."""

import os

from builtin_interfaces.msg import Time
import pytest
from ranger_xarm_teleop_bridge import bridge_node
from ranger_xarm_teleop_bridge.axes import xr_pose_to_ros
from ranger_xarm_teleop_bridge.bridge_node import BridgeNode
from ranger_xarm_teleop_bridge.frame_parse import parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.jsonl_io import read_records
import rclpy

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
STAMP = Time(sec=1234, nanosec=567)


def frames_of(name):
    """Return the ParsedFrames of a fixture, in order."""
    out = []
    for rec in read_records(os.path.join(FIXTURES, name + '.jsonl')):
        parsed = parse_text(rec.text())
        if isinstance(parsed, ParsedFrame):
            out.append(parsed)
    return out


def pose_tuple(pose):
    p, o = pose.pose.position, pose.pose.orientation
    return (p.x, p.y, p.z), (o.x, o.y, o.z, o.w)


@pytest.fixture(scope='module')
def node():
    rclpy.init()
    n = BridgeNode(node_name='bridge_messages_under_test', source_label='test')
    yield n
    n.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def published(node, monkeypatch):
    """Capture what the node would publish instead of sending it to the graph."""
    calls = []
    monkeypatch.setattr(node, 'publish_frame', calls.append)
    return calls


def test_both_hands_present_all_messages_share_the_stamp(node):
    frame = next(f for f in frames_of('still_controller') if f.left and f.right)
    msgs = node.build_messages(frame, STAMP)
    assert set(msgs.poses) == {'left', 'right'}
    for m in [*msgs.poses.values(), msgs.joy, msgs.status]:
        assert m.header.stamp == STAMP
        assert m.header.frame_id == 'quest_world'
    assert msgs.status.left.pose_present and msgs.status.right.pose_present
    assert not msgs.status.left.emulated_position and not msgs.status.right.emulated_position
    assert msgs.status.viewer_present
    assert msgs.status.source == 'test'


def test_inv1_conversion_called_once_per_present_pose(node, monkeypatch):
    calls = []

    def counting(position, orientation):
        calls.append((tuple(position), tuple(orientation)))
        return xr_pose_to_ros(position, orientation)
    monkeypatch.setattr(bridge_node, 'xr_pose_to_ros', counting)
    both = next(f for f in frames_of('still_controller') if f.left and f.right)
    node.build_messages(both, STAMP)
    assert len(calls) == 2
    one = next(f for f in frames_of('dropped_controller') if f.left and f.right is None)
    calls.clear()
    node.build_messages(one, STAMP)
    assert calls == [(one.left.position, one.left.orientation)]


def test_inv1_pose_values_are_the_converted_sample(node):
    frame = next(f for f in frames_of('moving_controller') if f.right)
    msgs = node.build_messages(frame, STAMP)
    assert pose_tuple(msgs.poses['right']) == xr_pose_to_ros(frame.right.position,
                                                             frame.right.orientation)


def test_inv2_absent_hand_has_no_pose_and_nothing_substituted(node):
    frame = next(f for f in frames_of('dropped_controller') if f.left and f.right is None)
    msgs = node.build_messages(frame, STAMP)
    assert set(msgs.poses) == {'left'}
    assert msgs.status.right.pose_present is False
    assert msgs.status.right.emulated_position is False
    assert msgs.status.left.pose_present is True
    assert pose_tuple(msgs.poses['left']) == xr_pose_to_ros(frame.left.position,
                                                            frame.left.orientation)
    assert list(msgs.joy.axes[4:8]) == [0.0, 0.0, 0.0, 0.0]
    assert list(msgs.joy.buttons[5:10]) == [0, 0, 0, 0, 0]


def test_inv3_viewer_only_frame_publishes_no_pose(node):
    frame = next(f for f in frames_of('malformed')
                 if f.left is None and f.right is None and f.viewer)
    msgs = node.build_messages(frame, STAMP)
    assert msgs.poses == {}
    assert msgs.status.viewer_present is True
    assert not msgs.status.left.pose_present and not msgs.status.right.pose_present
    assert list(msgs.joy.axes) == [0.0] * 8 and list(msgs.joy.buttons) == [0] * 10


def test_inv3_headset_pose_never_on_a_controller_topic(node):
    for name in ('still_controller', 'dropped_controller', 'moving_controller'):
        for frame in frames_of(name):
            if frame.viewer is None:
                continue
            viewer_ros = xr_pose_to_ros(frame.viewer.position, frame.viewer.orientation)
            msgs = node.build_messages(frame, STAMP)
            for hand, pose in msgs.poses.items():
                assert pose_tuple(pose) != viewer_ros, (name, hand, frame.seq)


def test_hand_tracked_source_is_not_present(node):
    frames = frames_of('malformed')
    frame = next(f for f in frames if f.left is not None and f.right is None)
    assert frame.right_hand_tracked
    msgs = node.build_messages(frame, STAMP)
    assert 'right' not in msgs.poses
    assert msgs.status.right.pose_present is False
    assert msgs.status.right.hand_tracked is True
    assert msgs.status.right.emulated_position is False
    assert msgs.status.left.pose_present is True
    assert msgs.status.left.hand_tracked is False
    # Joy block of a tracked hand is zeros, like an absent hand.
    assert list(msgs.joy.axes[4:]) == [0.0] * 4 and list(msgs.joy.buttons[5:]) == [0] * 5


def test_real_hands_clip_publishes_no_pose_and_flags_hand_tracked(node):
    frames = frames_of('clip_hands_head_turn')
    assert len(frames) == 45
    for frame in frames:
        assert frame.left is not None and frame.right is not None    # headset sent poses
        msgs = node.build_messages(frame, STAMP)
        assert msgs.poses == {}
        for side in ('left', 'right'):
            status = getattr(msgs.status, side)
            assert status.pose_present is False and status.hand_tracked is True
        assert list(msgs.joy.buttons) == [0] * 10 and list(msgs.joy.axes) == [0.0] * 8


def test_real_controller_clip_publishes_the_right_pose(node):
    frames = frames_of('clip_controller_occlusion')
    for frame in frames:
        msgs = node.build_messages(frame, STAMP)
        assert set(msgs.poses) == {'right'}
        assert msgs.status.right.pose_present and not msgs.status.right.hand_tracked
        assert msgs.status.right.emulated_position is frame.right.emulated_position
        assert not msgs.status.left.pose_present and not msgs.status.left.hand_tracked
        assert msgs.status.session_id == 128324305


def test_hand_tracked_is_false_for_controllers_and_absent_hands(node):
    present = node.build_messages(frames_of('still_controller')[0], STAMP)
    assert not present.status.left.hand_tracked and not present.status.right.hand_tracked
    absent = next(f for f in frames_of('dropped_controller') if f.right is None)
    msgs = node.build_messages(absent, STAMP)
    assert msgs.status.right.pose_present is False
    assert msgs.status.right.hand_tracked is False


def test_session_id_preserved_and_change_visible(node):
    frames = frames_of('new_session')
    before, after = frames[179], frames[180]
    assert node.build_messages(before, STAMP).status.session_id == before.session_id == 0x12345678
    assert node.build_messages(after, STAMP).status.session_id == after.session_id == 0xCAFE0042
    top = next(f for f in frames_of('malformed') if f.session_id == 2 ** 32 - 1)
    assert node.build_messages(top, STAMP).status.session_id == 2 ** 32 - 1


def test_emulated_pose_is_published_with_the_flag(node):
    frame = next(f for f in frames_of('dropped_controller')
                 if f.right and f.right.emulated_position)
    msgs = node.build_messages(frame, STAMP)
    assert 'right' in msgs.poses
    assert msgs.status.right.pose_present is True
    assert msgs.status.right.emulated_position is True


def test_inv4_client_times_and_counters_preserved(node):
    frame = frames_of('reset_event')[-1]
    msgs = node.build_messages(frame, STAMP)
    assert msgs.status.seq == frame.seq
    assert msgs.status.xr_time_ms == frame.xr_time_ms
    assert msgs.status.client_time_ms == frame.client_time_ms
    assert msgs.status.reset_count == frame.reset_count == 1


def test_handle_text_publishes_with_ros_time_and_identical_stamps(node, published):
    rec = next(r for r in read_records(os.path.join(FIXTURES, 'still_controller.jsonl')))
    before = node.get_clock().now().to_msg()
    node.handle_text(rec.text())
    after = node.get_clock().now().to_msg()
    assert len(published) == 1
    msgs = published[0]
    stamps = {(m.header.stamp.sec, m.header.stamp.nanosec)
              for m in [*msgs.poses.values(), msgs.joy, msgs.status]}
    assert len(stamps) == 1
    (sec, nanosec), = stamps
    assert (before.sec, before.nanosec) <= (sec, nanosec) <= (after.sec, after.nanosec)


def test_inv7_malformed_fixture_never_raises_and_is_counted(node, published):
    from fixtures.make_fixtures import expected  # noqa: I100 - local test helper
    records = read_records(os.path.join(FIXTURES, 'malformed.jsonl'))
    kinds = expected()
    assert len(kinds) == len(records)
    malformed0, ignored0, frames0 = node.malformed, node.ignored, node.frames
    for rec in records:
        node.handle_text(rec.text())
    assert len(published) == kinds.count('frame')
    assert node.malformed - malformed0 == kinds.count('malformed')
    assert node.ignored - ignored0 == kinds.count('ignored')
    assert node.frames - frames0 == kinds.count('frame')
    # The last fixture line is a valid frame: its status carries the running counters.
    assert published[-1].status.malformed_count == node.malformed
    assert published[-1].status.ignored_count == node.ignored


def test_handle_text_garbage_never_raises(node, published):
    for text in ['', '\x00', '{', '[]', 'null', '{"type": "xr_frame"}', '{"type": 5}',
                 'x' * 10000]:
        node.handle_text(text)
    assert published == []
