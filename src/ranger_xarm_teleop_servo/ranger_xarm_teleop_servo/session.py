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
Rows of a teleop session recording and the rule for starting and stopping one (no ROS).

The recorder (``tools/record_session.py``) must not disturb teleoperation: a ROS tool
that starts or stops while the clutch is held can stall the input stream and release
the clutch. So a recording starts only when the teleop node reports a released clutch,
and a request to stop is carried out only once the clutch is released.

The message arguments are duck-typed, so this module imports no message package.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

STATE_FRESH_S = 1.0         # a teleop state older than this says nothing about the clutch
ARM_JOINTS = tuple(f'xarm_joint{i}' for i in range(1, 7))
OWNER_NONE, OWNER_MOVEIT, OWNER_TELEOP, OWNER_SWITCHING = range(4)      # ArmOwner.OWNER_*
READY_POSTURE = (0.0, -0.6, -0.7, 0.0, 1.3, 0.0)


def rejection_reason(owner: Optional[int]) -> str:
    """Say why a trajectory goal to MoveIt's arm controller was rejected."""
    if owner == OWNER_TELEOP:
        return ('teleop owns the arm: disarm first (press B, or '
                '`ros2 service call /teleop/disarm std_srvs/srv/Trigger`), '
                'or run tools/recover_arm.py, which disarms and then moves')
    if owner == OWNER_SWITCHING:
        return 'the arm is being handed over right now: try again in a second'
    if owner == OWNER_NONE:
        return ('no arm controller is active (command mux owner NONE): '
                'check the controller manager')
    if owner == OWNER_MOVEIT:
        return ('MoveIt owns the arm, so ownership is not the reason: check --controller '
                '(mock: xarm6_traj_controller, simulator: xarm_xarm6_traj_controller) '
                'and the controller log')
    return ('the command mux reports no owner (is the teleop stack running?): check '
            '--controller and that the controller is active')


def may_recover(owner: Optional[int], armed: Optional[bool]) -> Tuple[bool, str]:
    """
    Allow a recovery motion only when MoveIt owns the arm and teleop is not armed.

    ``owner`` is the command mux's fresh report (None: none received); ``armed`` the
    teleop node's fresh state (None: no teleop node heard).
    """
    if owner != OWNER_MOVEIT:
        name = {OWNER_NONE: 'NONE', OWNER_TELEOP: 'TELEOP',
                OWNER_SWITCHING: 'SWITCHING'}.get(owner, 'not reported')
        return False, f'MoveIt does not own the arm (owner: {name})'
    if armed:
        return False, 'teleop is still armed'
    return True, ''


class ClutchWatch:
    """What the recorder knows about the clutch, from ``/teleop/state``."""

    def __init__(self) -> None:
        """Start with no evidence."""
        self.seen_at: Optional[float] = None
        self.engaged = False

    def update(self, now: float, engaged: bool) -> None:
        """Take in a teleop state received at ``now`` (monotonic seconds)."""
        self.seen_at = now
        self.engaged = bool(engaged)

    def may_start(self, now: float) -> Tuple[bool, str]:
        """Allow a start only on fresh evidence that the clutch is released."""
        if self.seen_at is None or now - self.seen_at > STATE_FRESH_S:
            return False, ('no /teleop/state received: start the teleop stack first '
                           '(the recorder must know the clutch is released)')
        if self.engaged:
            return False, 'the clutch is engaged: release the grip, then start the recorder'
        return True, ''

    def may_stop(self, now: float) -> bool:
        """Allow a stop unless a fresh state says the clutch is engaged."""
        if self.seen_at is None or now - self.seen_at > STATE_FRESH_S:
            return True         # the teleop node is gone: nothing left to disturb
        return not self.engaged


def _stamp(msg: Any) -> int:
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def _pose(pose: Any) -> List[List[float]]:
    return [[pose.position.x, pose.position.y, pose.position.z],
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]]


def description_row(urdf: str) -> Dict[str, Any]:
    """Row holding the robot description, so a recording can be analysed without ROS."""
    return {'k': 'description', 'urdf': urdf}


def state_row(msg: Any) -> Dict[str, Any]:
    """Row for a ``TeleopState``."""
    return {
        'k': 'state', 'stamp': _stamp(msg), 'motion': bool(msg.motion_enabled),
        'armed': bool(msg.armed), 'engaged': bool(msg.clutch_engaged),
        'owner': int(msg.owner), 'mux': int(msg.mux_status), 'hold': int(msg.hold_reason),
        'limit': int(msg.active_limit), 'servo': int(msg.servo_status),
        'calibrated': bool(msg.calibrated), 'age': float(msg.input_age),
        'seq': int(msg.input_seq),
        'tool': _pose(msg.measured_tool_pose) if msg.measured_tool_pose_valid else None,
        # The joint-space solver (G-8) and the lead limit (G-14).
        'conditioning': float(msg.conditioning), 'damped': bool(msg.damped),
        'damping': float(msg.damping),
        'solver_error_translation': float(msg.solver_error_translation),
        'solver_error_rotation': float(msg.solver_error_rotation),
        'lead_translation': float(msg.lead_translation),
        'lead_rotation': float(msg.lead_rotation),
        'commanded_linear_speed': float(msg.commanded_linear_speed),
        'commanded_angular_speed': float(msg.commanded_angular_speed)}


def pose_row(kind: str, msg: Any) -> Dict[str, Any]:
    """Row for a ``PoseStamped`` (kind ``target``, ``input`` or ``head``)."""
    p, q = _pose(msg.pose)
    return {'k': kind, 'stamp': _stamp(msg), 'frame': str(msg.header.frame_id), 'p': p, 'q': q}


def joints_row(msg: Any, names: Sequence[str] = ARM_JOINTS) -> Optional[Dict[str, Any]]:
    """Row for a ``JointState``; None when it does not carry every arm joint."""
    index = {name: i for i, name in enumerate(msg.name)}
    if not all(name in index for name in names):
        return None
    velocity = ([float(msg.velocity[index[name]]) for name in names]
                if len(msg.velocity) == len(msg.name) else None)
    return {'k': 'joints', 'stamp': _stamp(msg),
            'q': [float(msg.position[index[name]]) for name in names], 'v': velocity}


def command_row(msg: Any) -> Dict[str, Any]:
    """Row for the node's joint command, a ``control_msgs/JointJog``: names and rad/s."""
    return {'k': 'command', 'stamp': _stamp(msg), 'names': [str(n) for n in msg.joint_names],
            'v': [float(v) for v in msg.velocities]}


def servo_row(msg: Any) -> Dict[str, Any]:
    """Row for a ``moveit_msgs/ServoStatus``."""
    return {'k': 'servo', 'code': int(msg.code), 'msg': str(msg.message)}


def owner_row(msg: Any) -> Dict[str, Any]:
    """Row for an ``ArmOwner``."""
    return {'k': 'owner', 'owner': int(msg.owner), 'status': int(msg.status),
            'detail': str(msg.detail)}
