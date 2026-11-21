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
"""The pairing of one headset frame's messages by stamp (input_frames), and its frame."""

import math
import os
import subprocess
import sys
from types import SimpleNamespace as NS

from ranger_xarm_teleop_servo import input_frames
from ranger_xarm_teleop_servo.frames import (ARM_BUTTON, CALIBRATE_BUTTON, CLUTCH_BUTTON,
                                             SCALE_BUTTON, TRIGGER_AXIS)
from ranger_xarm_teleop_servo.input_frames import FrameAssembler, gate_frame, stamp_key
from ranger_xarm_teleop_servo.rotations import Pose

WAIT = input_frames.PAIR_WAIT_S
POSE = Pose((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
HEAD = Pose((0.0, 0.0, 1.6), (0.0, 0.0, 0.0, 1.0))


def stamp(seq):
    return NS(sec=1, nanosec=seq * 10_000_000)


def status(seq, pose=True, head=True, session=7, reset=0, emulated=False):
    right = NS(pose_present=pose, emulated_position=emulated)
    return NS(header=NS(stamp=stamp(seq)), right=right,
              viewer_present=head, viewer_emulated_position=False, xr_time_ms=1000.0 + seq,
              session_id=session, seq=seq, reset_count=reset)


def joy(pressed=(), trigger=0.0, buttons=10, axes=8):
    b = [0] * buttons
    for index in pressed:
        b[index] = 1
    a = [0.0] * axes
    if axes > TRIGGER_AXIS:
        a[TRIGGER_AXIS] = trigger
    return NS(buttons=b, axes=a)


def key(seq):
    return stamp_key(stamp(seq))


def test_stamp_key_is_integer_nanoseconds():
    assert stamp_key(NS(sec=2, nanosec=5)) == 2_000_000_005


def test_a_complete_frame_is_handed_on_at_once_in_any_arrival_order():
    frames = FrameAssembler()
    frames.add_status(1.0, status(1))
    assert frames.take(1.0) is None             # nothing else has arrived
    frames.add_pose(key(1), POSE)
    frames.add_joy(key(1), joy())
    assert frames.take(1.001) is None           # the head is announced and missing
    frames.add_head(key(1), HEAD)
    paired = frames.take(1.002)
    assert paired.key == key(1) and paired.arrival == 1.0
    assert math.isclose(paired.waited, 0.002)
    assert frames.parts(key(1))[1:] == (POSE, HEAD)
    assert frames.take(1.002) is None


def test_parts_not_announced_by_the_status_are_not_waited_for():
    frames = FrameAssembler()
    frames.add_joy(key(1), joy())
    frames.add_status(1.0, status(1, pose=False, head=False))
    assert frames.take(1.0) is not None


def test_a_missing_part_is_given_up_after_the_wait_and_reads_as_absent():
    frames = FrameAssembler()
    frames.add_status(1.0, status(1))
    frames.add_joy(key(1), joy())
    frames.add_head(key(1), HEAD)
    assert frames.take(1.0 + WAIT * 0.9) is None
    paired = frames.take(1.0 + WAIT)
    assert paired is not None and math.isclose(paired.waited, WAIT)
    assert not frames.has_pose(key(1)) and frames.has_joy(key(1))
    assert frames.parts(key(1))[1] is None


def test_frames_leave_in_status_order_and_a_late_one_holds_back_the_next():
    frames = FrameAssembler()
    frames.add_status(1.0, status(1))           # incomplete
    for part in (frames.add_pose, frames.add_head):
        part(key(2), POSE)
    frames.add_joy(key(2), joy())
    frames.add_status(1.005, status(2))         # complete
    assert frames.take(1.006) is None           # frame 2 waits behind frame 1
    first = frames.take(1.0 + WAIT)
    second = frames.take(1.0 + WAIT)            # the same "now" for both, as the node does
    assert (first.key, second.key) == (key(1), key(2))
    assert math.isclose(second.waited, 1.0 + WAIT - 1.005)


def test_done_drops_the_parts_of_that_frame_and_of_every_earlier_stamp():
    frames = FrameAssembler()
    for seq in (1, 2, 3):
        frames.add_pose(key(seq), POSE)
        frames.add_joy(key(seq), joy())
        frames.add_head(key(seq), HEAD)
    frames.done(key(2))
    assert not frames.has_pose(key(1)) and not frames.has_joy(key(2))
    assert frames.parts(key(2)) == (None, None, None)
    assert frames.has_pose(key(3)) and frames.has_joy(key(3))


def test_gate_frame_takes_the_buttons_and_the_trigger_from_the_joy():
    s = status(4, reset=2, emulated=True)
    frame = gate_frame(s, joy((CLUTCH_BUTTON, ARM_BUTTON), trigger=0.7), POSE, HEAD, 3.0)
    assert (frame.arrival, frame.seq, frame.session_id, frame.reset_count) == (3.0, 4, 7, 2)
    assert math.isclose(frame.xr_time, 1.004)
    assert frame.controller == POSE and frame.controller_present and frame.controller_emulated
    assert frame.head == HEAD and frame.head_present
    assert frame.grip and frame.arm and not frame.calibrate and not frame.scale_toggle
    assert math.isclose(frame.trigger, 0.7)
    frame = gate_frame(s, joy((CALIBRATE_BUTTON, SCALE_BUTTON)), POSE, HEAD, 3.0)
    assert frame.calibrate and frame.scale_toggle and not frame.grip


def test_without_buttons_the_controller_counts_as_not_present():
    s = status(4)
    for missing in (None, joy(buttons=CLUTCH_BUTTON)):     # no joy, or too few buttons
        frame = gate_frame(s, missing, POSE, HEAD, 3.0)
        assert frame.controller is None and not frame.controller_present
        assert not frame.grip and not frame.arm
    assert gate_frame(s, None, POSE, HEAD, 3.0).trigger is None
    assert gate_frame(s, joy(axes=TRIGGER_AXIS), POSE, HEAD, 3.0).trigger is None
    assert math.isnan(gate_frame(s, joy(trigger=math.nan), POSE, HEAD, 3.0).trigger)


def test_a_pose_or_head_the_status_does_not_announce_is_not_present():
    frame = gate_frame(status(4, pose=False, head=False), joy(), POSE, HEAD, 3.0)
    assert not frame.controller_present and not frame.head_present
    frame = gate_frame(status(4), joy(), None, None, 3.0)
    assert not frame.controller_present and not frame.head_present


def test_input_frames_has_no_ros_imports():
    pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    bridge = os.path.join(os.path.dirname(pkg), 'ranger_xarm_teleop_bridge')
    code = ('import sys\n'
            f'sys.path[:0] = [{pkg!r}, {bridge!r}]\n'
            'import ranger_xarm_teleop_servo.input_frames\n'
            "bad = [m for m in sys.modules if m.split('.')[0] in ('rclpy', 'sensor_msgs', "
            "'std_msgs', 'geometry_msgs', 'ranger_xarm_teleop_interfaces')]\n"
            'assert not bad, bad\n')
    result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
