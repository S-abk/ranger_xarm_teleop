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
Explain the MoveIt Servo halts of a recorded teleop session (numpy, no ROS).

Input: the rows written by ``tools/record_session.py`` (one dict per line, ``t`` on the
recorder's monotonic clock, ``k`` the kind):

    description  urdf
    state        engaged, armed, hold, limit, owner, servo, tool ([p, q] or None), ...
    target       p, q            the tool target of /teleop/target_pose
    joints       q, v            the six arm joints
    servo        code, msg       moveit_msgs/ServoStatus
    owner        owner, status, detail

The singularity measure is not published by MoveIt Servo; it is computed here from the
recorded joint angles with ``kinematics.condition_number`` (the same definition).
"""

from bisect import bisect_right
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ranger_xarm_teleop_servo import kinematics
from ranger_xarm_teleop_servo.rotations import q_conj, q_mul

# moveit_msgs/ServoStatus codes (moveit_servo/utils/datatypes.hpp).
SERVO_STATUS = {
    -1: 'INVALID', 0: 'NO_WARNING', 1: 'DECELERATE_FOR_APPROACHING_SINGULARITY',
    2: 'HALT_FOR_SINGULARITY', 3: 'DECELERATE_FOR_LEAVING_SINGULARITY',
    4: 'DECELERATE_FOR_COLLISION', 5: 'HALT_FOR_COLLISION', 6: 'JOINT_BOUND'}
HALT_CODES = (2, 5, 6)
DECELERATE_CODES = (1, 3, 4)
# TeleopState.HOLD_* (checked against the message definition by a test).
HOLD_REASON = {
    0: 'NONE', 1: 'MOTION_DISABLED', 2: 'NOT_ARMED', 3: 'NOT_OWNER', 4: 'NOT_CALIBRATED',
    5: 'CLUTCH_RELEASED', 6: 'INPUT_STALE', 7: 'POSE_ABSENT', 8: 'POSE_EMULATED',
    9: 'OUT_OF_VIEW', 10: 'HEAD_ABSENT', 11: 'HEAD_EMULATED', 12: 'RESET_CHANGED',
    13: 'SESSION_CHANGED', 14: 'SEQ_BACKWARDS', 15: 'INPUT_JUMP', 16: 'STEP_REJECTED',
    17: 'TOOL_POSE_STALE', 18: 'SERVO_NOT_READY', 19: 'SERVO_HALTED'}
HOLD_SERVO_HALTED = 19
MERGE_S = 0.5               # a status halt and the hold it causes are one event
TREND_S = 1.0               # how far back the trend of the measure is reported
VELOCITY_S = 0.2            # window for the commanded velocity
DEFAULT_LOWER, DEFAULT_HARD_STOP = 17.0, 30.0
WHAT_IF = (17.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0)


class SessionError(ValueError):
    """The recording cannot be analysed (for example it has no robot description)."""


class _Series:
    """Rows of one kind, looked up by time."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        """Index the rows, which are in time order."""
        self.rows = rows
        self.times = [row['t'] for row in rows]

    def at(self, t: float, max_age: float = math.inf) -> Optional[Dict[str, Any]]:
        """Return the newest row not after ``t`` (and not older than ``max_age``)."""
        index = bisect_right(self.times, t) - 1
        if index < 0 or t - self.times[index] > max_age:
            return None
        return self.rows[index]

    def after(self, t: float, within: float) -> List[Dict[str, Any]]:
        """Return the rows in ``(t, t + within]``."""
        return self.rows[bisect_right(self.times, t):bisect_right(self.times, t + within)]

    def between(self, start: float, end: float) -> List[Dict[str, Any]]:
        """Return the rows in ``[start, end]``."""
        return [row for row in self.rows[bisect_right(self.times, start - 1e-12):]
                if row['t'] <= end]


def chain_of(records: Sequence[Dict[str, Any]], urdf: Optional[str] = None) -> kinematics.Chain:
    """Build the arm chain from the recording's description row (or the given URDF)."""
    if urdf is None:
        for row in records:
            if row.get('k') == 'description':
                urdf = row['urdf']
                break
    if not urdf:
        raise SessionError('the recording has no robot description row; pass a URDF')
    return kinematics.chain_from_urdf(urdf)


def _unit(vector: np.ndarray) -> List[float]:
    norm = float(np.linalg.norm(vector))
    return [0.0, 0.0, 0.0] if norm < 1e-12 else [float(v) / norm for v in vector]


def _rotation_vector(q_from: Sequence[float], q_to: Sequence[float]) -> np.ndarray:
    """Rotation taking ``q_from`` to ``q_to`` as axis times angle, in the base frame."""
    x, y, z, w = q_mul(q_to, q_conj(q_from))
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    sine = math.sqrt(x * x + y * y + z * z)
    if sine < 1e-12:
        return np.zeros(3)
    return np.array([x, y, z]) / sine * (2.0 * math.atan2(sine, w))


def _events(states: _Series, servo: _Series) -> List[Dict[str, Any]]:
    """Find the halts: Servo entering a halt status, or the gate holding for a halt."""
    found = []
    last = None
    for row in servo.rows:
        if row['code'] in HALT_CODES and last not in HALT_CODES:
            found.append({'t': row['t'], 'code': row['code'], 'msg': row.get('msg', '')})
        last = row['code']
    last = None
    for row in states.rows:
        if row['hold'] == HOLD_SERVO_HALTED and last != HOLD_SERVO_HALTED:
            code = row.get('servo', -1)
            found.append({'t': row['t'], 'code': code if code in HALT_CODES else None,
                          'msg': ''})
        last = row['hold']
    found.sort(key=lambda event: event['t'])
    merged: List[Dict[str, Any]] = []
    for event in found:
        if merged and event['t'] - merged[-1]['last'] <= MERGE_S:
            merged[-1]['last'] = event['t']
            if merged[-1]['code'] is None:
                merged[-1].update(code=event['code'], msg=event['msg'])
            continue
        merged.append({**event, 'last': event['t']})
    return merged


def _measure(chain: kinematics.Chain, row: Optional[Dict[str, Any]]) -> Optional[float]:
    return None if row is None else kinematics.condition_number(chain, row['q'])


def _describe(event, chain, states, targets, joints, servo) -> Dict[str, Any]:
    t = event['t']
    out: Dict[str, Any] = {
        't': t, 'servo_code': event['code'],
        'servo_status': SERVO_STATUS.get(event['code'], 'not recorded'),
        'servo_message': event['msg']}
    holds = [row['hold'] for row in states.after(t - 0.1, MERGE_S + 0.1) if row['hold'] != 0]
    hold = HOLD_SERVO_HALTED if HOLD_SERVO_HALTED in holds else (holds[0] if holds else None)
    out['hold_reason'] = None if hold is None else HOLD_REASON.get(hold, str(hold))
    before = [row['code'] for row in servo.between(t - TREND_S, t - 1e-9)]
    out['servo_codes_before'] = sorted(set(before))
    joint_row = joints.at(t, 0.5)
    if joint_row is None:
        out['joints'] = None
        return out
    q = joint_row['q']
    out['joints'] = [float(v) for v in q]
    out['joint_velocity'] = joint_row.get('v')
    out['condition_number'] = kinematics.condition_number(chain, q)
    window = [(row['t'] - t, kinematics.condition_number(chain, row['q']))
              for row in joints.between(t - TREND_S, t)]
    out['trend'] = {
        'at_minus_1.0s': _measure(chain, joints.at(t - 1.0, 0.2)),
        'at_minus_0.5s': _measure(chain, joints.at(t - 0.5, 0.2)),
        'at_minus_0.2s': _measure(chain, joints.at(t - 0.2, 0.2)),
        'max_in_window': max((value for _, value in window), default=None),
        'rate_per_s': (float(np.polyfit([a for a, _ in window[-10:]],
                                        [b for _, b in window[-10:]], 1)[0])
                       if len(window) >= 3 and window[-1][0] - window[-10:][0][0] > 1e-3
                       else None)}
    name, margin = kinematics.limit_margin(chain, q)
    out['nearest_limit'] = {'joint': name, 'margin_rad': margin}
    shape = kinematics.resemblance(chain, q)
    out['resembles'] = shape.label(DEFAULT_LOWER)
    out['joint_shares'] = {'wrist': shape.wrist_share, 'shoulder': shape.shoulder_share,
                           'elbow': shape.elbow_share}
    out['wrist_deg'], out['elbow_deg'], out['shoulder_m'] = (
        shape.wrist_deg, shape.elbow_deg, shape.shoulder_m)
    # The task direction the arm can move least in (last left singular vector).
    left, values, _ = np.linalg.svd(kinematics.jacobian(chain, q))
    weak = left[:, -1]
    out['weak_direction'] = {'linear': [float(v) for v in weak[:3]],
                             'angular': [float(v) for v in weak[3:]],
                             'smallest_singular_value': float(values[-1])}
    target = targets.at(t, 0.5)
    state = states.at(t, 0.5)
    tool = state.get('tool') if state else None
    command: Dict[str, Any] = {}
    if target is not None and tool is not None:
        lead = np.array(target['p']) - np.array(tool[0])
        turn = _rotation_vector(tool[1], target['q'])
        command.update(
            lead_mm=float(np.linalg.norm(lead)) * 1000.0, lead_direction=_unit(lead),
            lead_rotation_deg=math.degrees(float(np.linalg.norm(turn))),
            lead_rotation_axis=_unit(turn))
        demand = np.concatenate([lead, turn])
        norm = float(np.linalg.norm(demand))
        command['along_weak_direction'] = (
            abs(float(np.dot(demand, weak))) / norm if norm > 1e-9 else 0.0)
    earlier = targets.at(t - VELOCITY_S, 0.2)
    if target is not None and earlier is not None and target['t'] > earlier['t']:
        span = target['t'] - earlier['t']
        velocity = (np.array(target['p']) - np.array(earlier['p'])) / span
        spin = _rotation_vector(earlier['q'], target['q']) / span
        command.update(
            speed_m_s=float(np.linalg.norm(velocity)), direction=_unit(velocity),
            turn_rate_deg_s=math.degrees(float(np.linalg.norm(spin))), turn_axis=_unit(spin))
    out['command'] = command or None
    return out


def _engaged_measure(chain, states: _Series, joints: _Series, servo: _Series):
    """Condition number and Servo status over engaged time, weighted by duration."""
    values, weights, codes = [], [], []
    for row, following in zip(joints.rows, joints.rows[1:]):
        state = states.at(row['t'], 0.2)
        if state is None or not state['engaged']:
            continue
        step = following['t'] - row['t']
        if step <= 0.0 or step > 0.2:       # a hole in the stream is not engaged time
            continue
        values.append(kinematics.condition_number(chain, row['q']))
        weights.append(step)
        status = servo.at(row['t'], 0.2)
        codes.append(status['code'] if status else -1)
    return np.array(values), np.array(weights), np.array(codes)


def _percentile(values: np.ndarray, weights: np.ndarray, fraction: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, fraction * cumulative[-1]))
    return float(values[order][min(index, len(values) - 1)])


def analyze(records: Sequence[Dict[str, Any]], urdf: Optional[str] = None,
            lower: float = DEFAULT_LOWER, hard_stop: float = DEFAULT_HARD_STOP
            ) -> Dict[str, Any]:
    """Return every halt of the session and the summary figures."""
    chain = chain_of(records, urdf)
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for row in sorted((r for r in records if 't' in r), key=lambda r: r['t']):
        by_kind.setdefault(row.get('k', ''), []).append(row)
    states = _Series(by_kind.get('state', []))
    targets = _Series(by_kind.get('target', []))
    joints = _Series(by_kind.get('joints', []))
    servo = _Series(by_kind.get('servo', []))
    halts = [_describe(event, chain, states, targets, joints, servo)
             for event in _events(states, servo)]
    counts: Dict[str, int] = {}
    for halt in halts:
        key = halt['servo_status'] + (f" / {halt['resembles']}" if halt.get('resembles') else '')
        counts[key] = counts.get(key, 0) + 1
    values, weights, codes = _engaged_measure(chain, states, joints, servo)
    total = float(weights.sum()) if len(weights) else 0.0
    summary: Dict[str, Any] = {
        'halts': len(halts), 'by_reason': counts, 'engaged_s': total,
        'thresholds': {'lower': lower, 'hard_stop': hard_stop}}
    if total > 0.0:
        summary['condition_number'] = {
            'min': float(values.min()), 'max': float(values.max()),
            **{f'p{int(f * 100)}': _percentile(values, weights, f)
               for f in (0.5, 0.9, 0.95, 0.99)}}
        summary['share_above'] = {
            str(level): float(weights[values >= level].sum()) / total for level in WHAT_IF}
        summary['share_decelerating'] = float(
            weights[np.isin(codes, DECELERATE_CODES)].sum()) / total
        summary['share_above_lower'] = float(weights[values > lower].sum()) / total
        summary['share_above_hard_stop'] = float(weights[values >= hard_stop].sum()) / total
    return {'halts': halts, 'summary': summary, 'joint_names': chain.joint_names}


def _number(value: Optional[float], spec: str = '.1f') -> str:
    return 'n/a' if value is None or not math.isfinite(value) else format(value, spec)


def _vector(values: Optional[Sequence[float]]) -> str:
    return 'n/a' if values is None else '(' + ' '.join(f'{v:+.2f}' for v in values) + ')'


def format_report(result: Dict[str, Any]) -> str:
    """Render ``analyze``'s result as text: one block per halt, then the summary."""
    lines: List[str] = []
    start = result['halts'][0]['t'] if result['halts'] else 0.0
    for number, halt in enumerate(result['halts'], 1):
        lines.append(f"halt {number}  t={halt['t'] - start:+.2f} s  "
                     f"servo {halt['servo_code']} {halt['servo_status']}  "
                     f"hold {halt['hold_reason']}")
        if halt.get('servo_message'):
            lines.append(f"  message: {halt['servo_message']}")
        if halt.get('joints') is None:
            lines.append('  no joint state near this time')
            continue
        trend = halt['trend']
        lines.append('  joints (rad): ' + ' '.join(f'{v:+.3f}' for v in halt['joints']))
        lines.append(
            f"  condition number {_number(halt['condition_number'])}  "
            f"(1.0 s before {_number(trend['at_minus_1.0s'])}, "
            f"0.5 s {_number(trend['at_minus_0.5s'])}, 0.2 s {_number(trend['at_minus_0.2s'])}; "
            f"rate {_number(trend['rate_per_s'])} /s)")
        limit = halt['nearest_limit']
        lines.append(f"  nearest joint limit: {limit['joint']}, margin "
                     f"{limit['margin_rad']:.3f} rad")
        lines.append(
            f"  resembles: {halt['resembles'] or 'no singularity (measure below 17)'}  "
            f"(axes 4-6 {halt['wrist_deg']:.1f} deg apart, elbow {halt['elbow_deg']:.1f} deg "
            f"from straight, wrist {halt['shoulder_m'] * 1000:.0f} mm from axis 1)")
        weak = halt['weak_direction']
        lines.append(f"  weakest direction: linear {_vector(weak['linear'])} angular "
                     f"{_vector(weak['angular'])}")
        command = halt.get('command')
        if command and 'lead_mm' in command:
            lines.append(
                f"  target ahead of the tool: {command['lead_mm']:.0f} mm "
                f"{_vector(command['lead_direction'])}, {command['lead_rotation_deg']:.1f} deg "
                f"about {_vector(command['lead_rotation_axis'])}; "
                f"{command['along_weak_direction'] * 100:.0f} % along the weakest direction")
        if command and 'speed_m_s' in command:
            lines.append(
                f"  commanded motion: {command['speed_m_s']:.3f} m/s "
                f"{_vector(command['direction'])}, {command['turn_rate_deg_s']:.0f} deg/s "
                f"about {_vector(command['turn_axis'])}")
        if not command:
            lines.append('  commanded motion: no target or tool pose near this time')
    summary = result['summary']
    lines.append('')
    lines.append(f"halts: {summary['halts']}   engaged time: {summary['engaged_s']:.1f} s")
    for reason, count in sorted(summary['by_reason'].items()):
        lines.append(f'  {count:3d}  {reason}')
    measure = summary.get('condition_number')
    if measure:
        lines.append(
            'condition number over engaged time: '
            f"min {measure['min']:.1f}  median {measure['p50']:.1f}  90 % {measure['p90']:.1f}  "
            f"95 % {measure['p95']:.1f}  99 % {measure['p99']:.1f}  max {measure['max']:.1f}")
        lines.append('share of engaged time at or above: ' + '  '.join(
            f'{level}: {share * 100:.1f} %' for level, share in summary['share_above'].items()))
        lines.append('share of engaged time Servo reported decelerating: '
                     f"{summary['share_decelerating'] * 100:.1f} %")
    return '\n'.join(lines)
