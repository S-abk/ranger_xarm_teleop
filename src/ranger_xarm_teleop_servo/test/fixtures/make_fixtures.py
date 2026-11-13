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
Generate the synthetic clutch fixtures (deterministic; the JSONL files are committed).

Every fixture is a relay recording in the bridge's format (WebXR axes: x right, y up,
z toward the user), 90 frames per second, right controller only. Common opening:
the calibration button (A) is pressed at 0.3 s with the controller yawed ``CAL_YAW_DEG``
to the left, the arm button (B) at 0.7 s, the grip closes at ``GRIP_ON``.

Run ``python3 make_fixtures.py`` to rewrite the files next to this script.
"""

import json
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

RATE_HZ = 90
T_RECV_START = 1700000000.0
T_XR_START_MS = 5000.0
SESSION_ID = 0x0BADCAFE
SESSION_ID_2 = 0x5EED0001
CAL_YAW_DEG = 20.0
GRIP_ON = 1.2
TRIGGER_AT = 2.5

HEAD = (0.0, 1.6, 0.0)
START = (0.15, 1.35, -0.5)

Vec = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def yaw(deg: float) -> Quat:
    """Rotation about the WebXR up axis (+y), positive to the left."""
    half = math.radians(deg) / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


def qmul(a: Quat, b: Quat) -> Quat:
    """Hamilton product."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def smooth(t: float, start: float, duration: float) -> float:
    """Minimum-jerk ramp from 0 to 1 over ``duration`` starting at ``start``."""
    s = min(max((t - start) / duration, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


# Direction the calibrated controller points along, in WebXR axes (forward is -z).
FORWARD = (-math.sin(math.radians(CAL_YAW_DEG)), 0.0, -math.cos(math.radians(CAL_YAW_DEG)))
# A tilted grip, to show the mapping does not depend on how the controller is held.
TILT = (math.sin(math.radians(-25.0)), 0.0, 0.0, math.cos(math.radians(-25.0)))
HELD = qmul(yaw(CAL_YAW_DEG), TILT)


def along(distance: float) -> Vec:
    """Return START moved ``distance`` along the calibrated forward direction."""
    return tuple(START[k] + distance * FORWARD[k] for k in range(3))


def buttons(grip: bool, a: bool, b: bool) -> List[Dict[str, Any]]:
    """Return seven xr-standard buttons: [1] grip, [4] A, [5] B."""
    pressed = {1: grip, 4: a, 5: b}
    return [{'p': pressed.get(k, False), 't': pressed.get(k, False),
             'v': 1.0 if pressed.get(k, False) else 0.0} for k in range(7)]


def hand(position: Vec, orientation: Quat, grip: bool, a: bool = False, b: bool = False,
         emulated: bool = False) -> Dict[str, Any]:
    """Return one controller entry of a frame."""
    return {'position': [round(v, 6) for v in position],
            'orientation': [round(v, 8) for v in orientation],
            'emulated_position': emulated,
            'buttons': buttons(grip, a, b),
            'axes': [0.0, 0.0, 0.0, 0.0],
            'profiles': ['meta-quest-touch-plus', 'generic-trigger-squeeze-thumbstick']}


def viewer(orientation: Quat = (0.0, 0.0, 0.0, 1.0), emulated: bool = False) -> Dict[str, Any]:
    """Return the headset entry of a frame."""
    return {'position': list(HEAD), 'orientation': [round(v, 8) for v in orientation],
            'emulated_position': emulated}


def line(i: int, right: Optional[Dict[str, Any]], head: Optional[Dict[str, Any]],
         reset_count: int = 0, seq: Optional[int] = None, session_id: int = SESSION_ID) -> str:
    """Return one recording line for frame index ``i``."""
    t_xr = round(T_XR_START_MS + i * 1000.0 / RATE_HZ, 6)
    jitter = 0.0004 * math.sin(1.7 * i) + 0.0002 * math.sin(0.37 * i)
    frame = {'type': 'xr_frame', 't_client': round(t_xr + 0.4, 6), 't_xr': t_xr,
             'seq': i if seq is None else seq, 'reset_count': reset_count,
             'session_id': session_id,
             'controllers': {} if right is None else {'right': right}, 'viewer': head}
    return json.dumps({'t_recv': round(T_RECV_START + i / RATE_HZ + jitter, 6), 'msg': frame},
                      ensure_ascii=False, allow_nan=False)


Motion = Callable[[float], Tuple[Vec, Quat]]


def opening(t: float, motion: Motion, grip: Callable[[float], bool]) -> Dict[str, Any]:
    """Return the controller entry at ``t``: calibration and arming, then ``motion``."""
    if t < GRIP_ON - 0.2:
        return hand(START, yaw(CAL_YAW_DEG), False, a=0.3 <= t < 0.5, b=0.7 <= t < 0.9)
    position, orientation = motion(t)
    return hand(position, orientation, grip(t))


def record(duration: float, motion: Motion, grip: Callable[[float], bool],
           change: Optional[Callable[[int, float, Dict[str, Any]], Optional[str]]] = None
           ) -> List[str]:
    """Build a recording; ``change`` may return a replacement line or '' to drop a frame."""
    lines = []
    for i in range(int(round(duration * RATE_HZ))):
        t = i / RATE_HZ
        right = opening(t, motion, grip)
        replaced = change(i, t, right) if change else None
        if replaced == '':
            continue
        lines.append(replaced if replaced is not None else line(i, right, viewer()))
    return lines


def held_until(end: float) -> Callable[[float], bool]:
    """Grip held from GRIP_ON to ``end``."""
    return lambda t: GRIP_ON <= t < end


def line_motion(t: float) -> Tuple[Vec, Quat]:
    """Move 0.20 m forward between 1.5 s and 3.5 s, held tilted."""
    return along(0.20 * smooth(t, 1.5, 2.0)), HELD


def slow_motion(t: float) -> Tuple[Vec, Quat]:
    """Move 0.10 m forward between 1.5 s and 4.5 s."""
    return along(0.10 * smooth(t, 1.5, 3.0)), HELD


def clutch_line() -> List[str]:
    """Grip 1.2-4.2 s; 0.20 m forward, hold, release."""
    return record(4.6, line_motion, held_until(4.2))


def clutch_circle() -> List[str]:
    """Grip 1.2-7.2 s; one and a half horizontal circles of radius 0.06 m."""
    def motion(t: float) -> Tuple[Vec, Quat]:
        angle = 3.0 * math.pi * smooth(t, 1.5, 5.0)
        return (START[0] + 0.06 * math.sin(angle), START[1],
                START[2] - 0.06 * (1.0 - math.cos(angle))), HELD
    return record(7.6, motion, held_until(7.2))


def clutch_rotation() -> List[str]:
    """Grip 1.2-4.2 s; turn 40 deg to the left about the vertical and back, no translation."""
    def motion(t: float) -> Tuple[Vec, Quat]:
        angle = 40.0 * (smooth(t, 1.5, 1.2) - smooth(t, 2.8, 1.2))
        return START, qmul(yaw(angle), HELD)
    return record(4.6, motion, held_until(4.2))


def clutch_reengage() -> List[str]:
    """Two engagements of 0.10 m forward each; the hand returns and re-grips differently."""
    def motion(t: float) -> Tuple[Vec, Quat]:
        distance = 0.10 * (smooth(t, 1.4, 1.2) - smooth(t, 2.8, 0.6) + smooth(t, 3.7, 1.2))
        orientation = HELD if t < 2.75 else qmul(yaw(35.0), HELD)
        return along(distance), orientation
    return record(5.4, motion, lambda t: GRIP_ON <= t < 2.7 or 3.5 <= t < 5.0)


def clutch_jump() -> List[str]:
    """As clutch_line, but from 2.5 s the reported position is 290 mm to the right."""
    def motion(t: float) -> Tuple[Vec, Quat]:
        position, orientation = line_motion(t)
        if t >= TRIGGER_AT:
            position = (position[0] + 0.29, position[1], position[2])
        return position, orientation
    return record(4.6, motion, held_until(4.2))


def clutch_stall() -> List[str]:
    """As clutch_line, but no frame arrives between 2.5 s and 3.0 s (grip held)."""
    return record(4.6, line_motion, held_until(4.2),
                  lambda i, t, right: '' if TRIGGER_AT <= t < 3.0 else None)


def clutch_release_moving() -> List[str]:
    """As clutch_line, but the grip opens at 2.5 s, at the peak hand speed (0.19 m/s)."""
    return record(4.0, line_motion, held_until(TRIGGER_AT))


def clutch_into_body() -> List[str]:
    """Grip 1.2-6.0 s; the hand pulls 0.28 m straight back (the tool toward the arm base)."""
    def motion(t: float) -> Tuple[Vec, Quat]:
        return along(-0.28 * smooth(t, 1.5, 3.0)), HELD

    def change(i: int, t: float, right: Dict[str, Any]) -> Optional[str]:
        # The operator looks 25 deg down at the hand, which keeps it in view as it
        # comes closer.
        down = math.radians(-25.0) / 2.0
        return line(i, right, viewer((math.sin(down), 0.0, 0.0, math.cos(down))))
    return record(6.4, motion, held_until(6.0), change)


def pitch_up(deg: float) -> Quat:
    """Rotation about the calibrated right-hand axis: positive raises the pointing end."""
    half = math.radians(deg) / 2.0
    c = math.radians(CAL_YAW_DEG)
    return (math.cos(c) * math.sin(half), 0.0, -math.sin(c) * math.sin(half), math.cos(half))


def clutch_singularity() -> List[str]:
    """
    Tilt into a wrist singularity, let go, grip again and tilt back; no translation.

    Grip 1.2-5.0 s while the hand pitches up 25 deg (1.5-4.5 s); grip again 5.6-8.8 s
    while it pitches back down by 20 deg (5.9-8.4 s). For an arm that starts with its
    wrist 0.5 rad from aligned (joints 0 -0.6 -0.7 0 0.5 0), where 15 deg of pitch brings
    MoveIt Servo's singularity measure to its default hard stop.
    """
    def motion(t: float) -> Tuple[Vec, Quat]:
        angle = 25.0 * smooth(t, 1.5, 3.0) - 20.0 * smooth(t, 5.9, 2.5)
        return START, qmul(pitch_up(angle), HELD)
    return record(9.2, motion, lambda t: GRIP_ON <= t < 5.0 or 5.6 <= t < 8.8)


# ---- Fixtures for the joint-space solver: poor regions of the arm and hard inputs ----
#
# Hand offsets are in the tool's own axes, which the calibrated forward direction maps to the
# arm base: ``dx`` forward, ``dy`` to the left, ``dz`` up. The motions are meant for an arm
# that starts at the ready posture (joints 0 -0.6 -0.7 0 1.3 0, tool pointing down).

LOOK_DOWN = (math.sin(math.radians(-25.0) / 2.0), 0.0, 0.0, math.cos(math.radians(-25.0) / 2.0))
LEFT = (-math.cos(math.radians(CAL_YAW_DEG)), 0.0, math.sin(math.radians(CAL_YAW_DEG)))

Leg = Tuple[float, float, float]        # start, duration, change of one hand offset


def at_offset(dx: float, dy: float = 0.0, dz: float = 0.0) -> Vec:
    """Return START moved ``dx`` forward, ``dy`` to the left and ``dz`` up (tool axes)."""
    return tuple(START[k] + dx * FORWARD[k] + dy * LEFT[k] + (dz if k == 1 else 0.0)
                 for k in range(3))


def moves(t: float, legs: List[Leg]) -> float:
    """Return a hand offset at ``t``: the sum of minimum-jerk legs (start, duration, change)."""
    return sum(change * smooth(t, start, duration) for start, duration, change in legs)


def linear(t: float, start: float, duration: float, change: float) -> float:
    """Return a hand offset that changes by ``change`` at constant speed from ``start``."""
    return change * min(max((t - start) / duration, 0.0), 1.0)


def looking_down(i: int, t: float, right: Dict[str, Any]) -> Optional[str]:
    """Replace a frame by one whose headset looks 25 deg down, at the hand."""
    return line(i, right, viewer(LOOK_DOWN))


def clutch_shoulder() -> List[str]:
    """
    Pull the tool back until the wrist is over the axis of joint 1, rising to clear the body.

    Grip 1.2-6.4 s: the tool rises 0.20 m, then comes back 0.12 m. The hand returns while the
    grip is open. Grip 7.2-11.8 s: back another 0.17 m (0.29 m in all; each engagement is
    inside the 0.30 m offset limit), hold, release.
    """
    def motion(t: float) -> Tuple[Vec, Quat]:
        dz = moves(t, [(1.5, 2.0, 0.20), (6.6, 0.5, -0.20)])
        dx = moves(t, [(3.8, 2.0, -0.12), (6.6, 0.5, 0.12), (7.7, 2.5, -0.17)])
        return at_offset(dx, 0.0, dz), HELD
    return record(12.4, motion, lambda t: GRIP_ON <= t < 6.4 or 7.2 <= t < 11.8, looking_down)


def clutch_fast() -> List[str]:
    """Grip 1.2-8.0 s; 0.24 m at 0.6 m/s (2.4 x the speed cap), still for 2 s, then 40 mm back."""
    def motion(t: float) -> Tuple[Vec, Quat]:
        dx = linear(t, 1.5, 0.4, 0.24) + linear(t, 4.4, 0.5, -0.04)
        return at_offset(dx), HELD
    return record(8.4, motion, held_until(8.0))


def clutch_step() -> List[str]:
    """Grip 1.2-7.6 s; 3 s still, then the hand jumps 40 mm in 0.1 s and holds for 3 s."""
    def motion(t: float) -> Tuple[Vec, Quat]:
        return at_offset(linear(t, 4.2, 0.1, 0.04)), HELD
    return record(8.0, motion, held_until(7.6))


def clutch_joint_limit() -> List[str]:
    """Grip 1.2-6.0 s; tilt the pointing end up by 55 deg (inside the 60 deg offset limit)."""
    def motion(t: float) -> Tuple[Vec, Quat]:
        return START, qmul(pitch_up(55.0 * smooth(t, 1.5, 3.0)), HELD)
    return record(6.4, motion, held_until(6.0))


def clutch_reach() -> List[str]:
    """
    Reach out past the arm's maximum reach, hold, then bring the hand back 0.10 m.

    Three engagements of 0.30, 0.30 and 0.15 m forward (the offset limit is 0.30 m per
    engagement; the hand returns while the grip is open): grip 1.2-5.2 s, 6.0-10.0 s and
    10.8-19.0 s. In the last, 2 s still at the far point, then 0.10 m back.
    """
    def motion(t: float) -> Tuple[Vec, Quat]:
        dx = moves(t, [(1.5, 3.0, 0.30), (5.3, 0.5, -0.30), (6.3, 3.0, 0.30),
                       (10.1, 0.5, -0.30), (11.1, 2.0, 0.15)]) + linear(t, 16.5, 0.5, -0.10)
        return at_offset(dx), HELD
    return record(19.6, motion,
                  lambda t: GRIP_ON <= t < 5.2 or 6.0 <= t < 10.0 or 10.8 <= t < 19.0)


def gate_absent() -> List[str]:
    """Grip held; the controller is absent from 2.5 s to 3.0 s."""
    def change(i: int, t: float, right: Dict[str, Any]) -> Optional[str]:
        return line(i, None, viewer()) if TRIGGER_AT <= t < 3.0 else None
    return record(3.6, slow_motion, held_until(9.0), change)


def gate_emulated() -> List[str]:
    """Emulated position 2.5-3.0 s; grip held to 3.6 s, released, pressed again at 3.8 s."""
    def change(i: int, t: float, right: Dict[str, Any]) -> Optional[str]:
        if TRIGGER_AT <= t < 3.0:
            right = dict(right, emulated_position=True)
            return line(i, right, viewer())
        return None
    return record(4.4, slow_motion, lambda t: GRIP_ON <= t < 3.6 or t >= 3.8, change)


def gate_out_of_view() -> List[str]:
    """Grip held; from 2.5 s the head turns 90 deg away within 0.3 s and stays there."""
    def change(i: int, t: float, right: Dict[str, Any]) -> Optional[str]:
        return line(i, right, viewer(yaw(-90.0 * smooth(t, TRIGGER_AT, 0.3))))
    return record(3.6, slow_motion, held_until(9.0), change)


def gate_head_emulated() -> List[str]:
    """Grip held; the headset position is emulated from 2.5 s to 3.0 s."""
    def change(i: int, t: float, right: Dict[str, Any]) -> Optional[str]:
        return line(i, right, viewer(emulated=TRIGGER_AT <= t < 3.0))
    return record(3.6, slow_motion, held_until(9.0), change)


def gate_reset() -> List[str]:
    """Grip held; the reset count goes from 0 to 1 at 2.5 s (poses unchanged)."""
    def change(i: int, t: float, right: Dict[str, Any]) -> Optional[str]:
        return line(i, right, viewer(), reset_count=1 if t >= TRIGGER_AT else 0)
    return record(3.6, slow_motion, held_until(9.0), change)


def gate_session() -> List[str]:
    """Grip held; the session id changes at 2.5 s (poses and seq continue)."""
    def change(i: int, t: float, right: Dict[str, Any]) -> Optional[str]:
        return line(i, right, viewer(),
                    session_id=SESSION_ID_2 if t >= TRIGGER_AT else SESSION_ID)
    return record(3.6, slow_motion, held_until(9.0), change)


def gate_seq_backwards() -> List[str]:
    """Grip held; at 2.5 s the sequence number steps back by 50 and counts on from there."""
    def change(i: int, t: float, right: Dict[str, Any]) -> Optional[str]:
        return line(i, right, viewer(), seq=i - 50 if t >= TRIGGER_AT else i)
    return record(3.6, slow_motion, held_until(9.0), change)


def generate() -> Dict[str, List[str]]:
    """Return fixture name -> exact file lines (without newlines)."""
    makers = [clutch_line, clutch_circle, clutch_rotation, clutch_reengage, clutch_jump,
              clutch_stall, clutch_release_moving, clutch_into_body, clutch_singularity,
              clutch_shoulder, clutch_fast, clutch_step, clutch_joint_limit, clutch_reach,
              gate_absent,
              gate_emulated, gate_out_of_view, gate_head_emulated, gate_reset, gate_session,
              gate_seq_backwards]
    return {maker.__name__: maker() for maker in makers}


def main() -> None:
    """Write every fixture as ``<name>.jsonl`` next to this script or into argv[1]."""
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    for name, lines in generate().items():
        path = os.path.join(out_dir, f'{name}.jsonl')
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            for text in lines:
                handle.write(text + '\n')
        print(f'wrote {path} ({len(lines)} lines)')


if __name__ == '__main__':
    main()
