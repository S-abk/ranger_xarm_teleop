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
The whole driving path against a model arm with a delay: gate, solver, MoveIt Servo, arm.

No ROS imports. The model is what the node does each cycle, with the robot side reduced to
what matters for stability and for the lead limit:

* MoveIt Servo integrates each joint velocity command from its own commanded state
  (``commanded``). The teleop node sees that state as Servo's output, ``stale_ticks``
  cycles late (the reference of the solver).
* The measured joint state is the commanded state ``delay`` seconds earlier (transport,
  Servo's window, the controller), and the measured tool pose is its forward kinematics.
* The gate sees that measured tool pose: at engagement, and for the lead limit (G-14).
"""

from dataclasses import dataclass
import math
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from .gate import Env, Frame, GateState, on_frame, Output, tick
from .params import GateParams
from .rotations import Pose, pose_from_matrix, pose_to_matrix
from .solver import Arm, Solution, solve, SolverParams, tool_pose


@dataclass(frozen=True)
class ModelStep:
    """One cycle of the model."""

    t: float
    output: Output
    solution: Optional[Solution]        # None when no target was published
    qdot: np.ndarray                    # the joint command sent this cycle (zeros on hold)
    commanded: np.ndarray               # Servo's commanded state after this cycle
    measured: np.ndarray                # measured joint state used this cycle
    tool: Pose                          # measured tool pose (forward kinematics)


def run(frames: Iterable[Frame], gate_params: GateParams, arm: Arm, solver_params: SolverParams,
        q0: Sequence[float], delay: float = 0.15, stale_ticks: int = 1,
        state: Optional[GateState] = None, tail: float = 1.0,
        edit: Optional[Callable[[Frame], Frame]] = None) -> List[ModelStep]:
    """Run the gate and the solver over ``frames`` against the delayed model arm."""
    period = gate_params.target_period
    delay_ticks = max(0, int(round(delay / period)))
    state = state if state is not None else GateState()
    frames = list(frames)
    commanded = np.array(q0, dtype=float)
    history = [commanded.copy()]           # commanded state at the end of each cycle
    steps: List[ModelStep] = []
    index = 0
    wanted_for = 0
    end = (frames[-1].arrival if frames else 0.0) + tail
    reference: Optional[np.ndarray] = None
    for k in range(int(math.floor(end / period)) + 1):
        now = k * period
        while index < len(frames) and frames[index].arrival <= now:
            frame = frames[index]
            state = on_frame(state, edit(frame) if edit else frame, gate_params)
            index += 1
        measured = history[max(0, len(history) - 1 - delay_ticks)]
        tool = pose_from_matrix(tool_pose(arm, measured))
        env = Env(now=now, owner_is_teleop=True, tool_pose=tool, tool_pose_age=0.0,
                  servo_active=wanted_for >= 1)
        engaged_before = state.engaged
        state, output = tick(state, env, gate_params)
        wanted_for = wanted_for + 1 if output.want_servo_active else 0
        qdot = np.zeros_like(commanded)
        solution = None
        if output.target is not None:
            if not engaged_before or reference is None:
                reference = measured.copy()          # seeded from the measurement
            else:
                reference = history[max(0, len(history) - 1 - (stale_ticks - 1))].copy() \
                    if stale_ticks > 0 else commanded.copy()
            solution = solve(arm, reference, pose_to_matrix(output.target), solver_params,
                             output.feedforward)
            qdot = solution.qdot
        else:
            reference = None
        commanded = commanded + qdot * period
        history.append(commanded.copy())
        steps.append(ModelStep(now, output, solution, qdot, commanded.copy(), measured.copy(),
                               tool))
    return steps
