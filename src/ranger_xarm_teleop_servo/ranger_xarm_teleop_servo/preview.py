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
The target preview while motion is disabled (contract, "Arm ownership"). No ROS imports.

With motion disabled (every hardware profile starts so) arming is refused and the gate never
computes a target. So that the operator can still check the calibration and the mapping on
screen, ``step`` computes the target an engagement would command, while the grip is held:
the same clutch-relative mapping, scale, filter and G-6 bounds (offset limit and keep-out
planes) as ``gate.tick``, from the measured tool pose at the grip press, under a valid
calibration and with fresh joint states, exactly as an armed target needs them.

What it leaves out, because the arm does not move: the lead clamp (G-14; against an arm that
stands still it would pin the preview to within the lead limit of the measured pose), MoveIt
Servo, the solver and the speed caps. A step beyond the per-cycle limit (G-5) ends the preview
engagement until the grip is released, as it would release the clutch.

Nothing here feeds the command path: the result is published on its own topic, only while
motion is disabled, and the gate's own state is read, never changed.
"""

from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple

from .gate import _pose_reason, bound, Env, GateState, Hold, low_pass
from .mapping import apply_pivot, clutch_relative
from .params import GateParams
from .rotations import Pose, q_between, v_norm, v_sub

# Why no preview is published (TeleopState.preview_reason); empty while one is.
NOT_DISABLED = 'motion is enabled: no preview'
NO_INPUT = 'no fresh input from the headset'
NO_CALIBRATION = 'not calibrated for this headset session: point the right controller ' \
                 'forward and press A'
NO_JOINT_STATE = 'no fresh joint state: the measured tool pose is unknown'
GRIP_RELEASED = 'grip released: hold the grip to see the target'
STEP = 'the target jumped beyond the per-cycle limit: release the grip'


@dataclass(frozen=True)
class PreviewState:
    """What the preview remembers between cycles; the gate's own state is separate."""

    engaged: bool = False
    latched: str = ''                 # a step reject holds until the grip is released
    clutch: Optional[Pose] = None
    tool0: Optional[Pose] = None
    heading: float = 0.0
    scale: float = 1.0
    filtered: Optional[Pose] = None
    last_target: Optional[Pose] = None


def step(preview: PreviewState, gate: GateState, env: Env,
         params: GateParams) -> Tuple[PreviewState, Optional[Pose], str]:
    """Return the new preview state, the previewed target (or None) and why not."""
    if params.motion_enabled:
        return PreviewState(), None, NOT_DISABLED
    frame = gate.frame
    if frame is None or not env.now - frame.arrival <= params.input_timeout:
        return PreviewState(), None, NO_INPUT
    grip = frame.controller_present and frame.grip
    if not grip:
        return PreviewState(), None, GRIP_RELEASED
    if preview.latched:
        return preview, None, preview.latched
    reason = _pose_reason(frame, params)
    if reason != Hold.NONE:
        return replace(PreviewState(), latched=f'{reason.name.lower()}: release the grip'), \
            None, f'{reason.name.lower()}: release the grip'
    calibration = gate.calibration
    if calibration is None or not calibration.valid_for(frame.session_id, frame.reset_count):
        return PreviewState(), None, NO_CALIBRATION
    if env.tool_pose is None or not env.tool_pose_age <= params.tool_pose_timeout:
        return PreviewState(), None, NO_JOINT_STATE
    controller = apply_pivot(frame.controller, params.pivot_offset)
    if not preview.engaged or preview.heading != calibration.heading:
        preview = PreviewState(engaged=True, clutch=controller, tool0=env.tool_pose,
                               heading=calibration.heading,
                               scale=params.translation_scale(False),
                               filtered=env.tool_pose, last_target=env.tool_pose)
    raw = clutch_relative(controller, preview.clutch, preview.heading, preview.tool0,
                          preview.scale)
    filtered = low_pass(preview.filtered, raw, params)
    target, _ = bound(filtered, preview.tool0, preview.last_target, params)
    jump = v_norm(v_sub(target.p, preview.last_target.p))
    turn = q_between(preview.last_target.q, target.q)
    if jump > params.step_translation_limit or turn > params.step_rotation_limit:
        return replace(PreviewState(), latched=STEP), None, STEP
    if not all(math.isfinite(v) for v in target.p + target.q):
        return PreviewState(), None, NO_JOINT_STATE
    return replace(preview, filtered=filtered, last_target=target), target, ''
