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
Build gate frames from parsed relay frames, and run the gate over a recording.

Used by the unit tests and the offline tools: a JSONL recording goes through the same
parser, axis conversion and Joy layout as the live bridge, then through the same gate
functions as the node, against an ideal arm that reaches every target. No ROS imports.
"""

from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable, List, Optional, Tuple

from ranger_xarm_teleop_bridge.axes import xr_pose_to_ros
from ranger_xarm_teleop_bridge.frame_parse import parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.joy_layout import build_joy_arrays, BUTTON_NAMES
from ranger_xarm_teleop_bridge.jsonl_io import iter_records

from .gate import Env, Frame, GateState, on_frame, Output, tick
from .params import GateParams
from .rotations import Pose

# Buttons of this task (Joy indices, fixed by the bridge's layout).
CLUTCH_BUTTON = BUTTON_NAMES.index('right_grip')
CALIBRATE_BUTTON = BUTTON_NAMES.index('right_primary')
ARM_BUTTON = BUTTON_NAMES.index('right_secondary')


def frame_from_parsed(parsed: ParsedFrame, arrival: float) -> Frame:
    """Return the gate frame for one parsed relay frame received at ``arrival``."""
    sample = parsed.controller('right')
    controller = None
    if sample is not None:
        controller = Pose(*xr_pose_to_ros(sample.position, sample.orientation))
    head = None
    if parsed.viewer is not None:
        head = Pose(*xr_pose_to_ros(parsed.viewer.position, parsed.viewer.orientation))
    _, buttons = build_joy_arrays(parsed)
    return Frame(
        arrival=float(arrival),
        xr_time=float(parsed.xr_time_ms) / 1000.0,
        session_id=int(parsed.session_id),
        seq=int(parsed.seq),
        reset_count=int(parsed.reset_count),
        controller=controller,
        controller_present=sample is not None,
        controller_emulated=bool(sample is not None and sample.emulated_position),
        head=head,
        head_present=parsed.viewer is not None,
        head_emulated=bool(parsed.viewer is not None and parsed.viewer.emulated_position),
        grip=bool(buttons[CLUTCH_BUTTON]),
        calibrate=bool(buttons[CALIBRATE_BUTTON]),
        arm=bool(buttons[ARM_BUTTON]),
    )


def frames_of_recording(path: str) -> List[Frame]:
    """Return the gate frames of a JSONL recording; arrival is ``t_recv`` from its start."""
    frames = []
    start = None
    for record in iter_records(path):
        parsed = parse_text(record.text())
        if not isinstance(parsed, ParsedFrame):
            continue
        if start is None:
            start = record.t_recv
        frames.append(frame_from_parsed(parsed, record.t_recv - start))
    return frames


@dataclass(frozen=True)
class Step:
    """One tick of a simulated run."""

    t: float
    output: Output
    tool: Pose


def simulate(frames: Iterable[Frame], params: GateParams, tool_start: Pose,
             state: Optional[GateState] = None,
             edit: Optional[Callable[[Frame], Frame]] = None,
             owner_is_teleop: bool = True,
             servo_delay_ticks: int = 1,
             tail: float = 0.3) -> Tuple[List[Step], GateState]:
    """
    Run the gate over ``frames`` with ticks at the target rate against an ideal arm.

    The ideal arm is at the last published target (``tool_start`` before the first).
    MoveIt Servo becomes active ``servo_delay_ticks`` ticks after the gate asks for it.
    ``edit`` may rewrite each frame (for example to hold the grip on a real clip).
    Ticks continue for ``tail`` seconds after the last frame so a stalled stream is seen.
    """
    state = state if state is not None else GateState()
    frames = list(frames)
    steps: List[Step] = []
    tool = tool_start
    wanted_for = 0
    index = 0
    end = (frames[-1].arrival if frames else 0.0) + tail
    count = int(math.floor(end / params.target_period)) + 1
    for k in range(count):
        now = k * params.target_period
        while index < len(frames) and frames[index].arrival <= now:
            frame = frames[index]
            state = on_frame(state, edit(frame) if edit else frame, params)
            index += 1
        env = Env(now=now, owner_is_teleop=owner_is_teleop, tool_pose=tool, tool_pose_age=0.0,
                  servo_active=wanted_for >= servo_delay_ticks)
        state, output = tick(state, env, params)
        wanted_for = wanted_for + 1 if output.want_servo_active else 0
        if output.target is not None:
            tool = output.target
        steps.append(Step(now, output, tool))
    return steps, state


def hold_grip(frame: Frame) -> Frame:
    """Edit for ``simulate``: the grip is held in every frame."""
    return replace(frame, grip=True)
