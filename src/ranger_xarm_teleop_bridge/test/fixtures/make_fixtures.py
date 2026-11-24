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
Deterministic generator of the synthetic JSONL fixtures used by the unit tests.

Run ``python3 make_fixtures.py [output_dir]`` to rewrite the files. Output is a
pure function of the code: no ``random``, closed-form noise (sums of sines) and
fixed-precision rounding, so regeneration is byte-identical.

All positions are metres in WebXR axes; quaternions are xyzw; time is the
headset rate of 72 Hz.

``generate()`` holds the fixtures the bridge tests use. ``generate_base()`` holds the
base-driving fixtures (``base_*``): a session that calibrates (A) and arms (B) as the servo
package's clutch fixtures do, then drives with the left grip and thumbstick, and one fixture
per fault that must stop a drive. Their frames are valid; the faults are in what the
stream does (a stall, a session change, a sequence step back, a missing hand). ``main()``
writes both sets.
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

RATE_HZ = 72
T_RECV_START = 1700000000.0
T_XR_START_MS = 1000.0
CLIENT_OFFSET_MS = 0.4
NUM_BUTTONS = 7
SESSION_ID = 0x12345678
SESSION_ID_2 = 0xCAFE0042

# The movement reported with the reset in ``reset_event``: 0.1 m along -x, 0.05 m along -z
# (0.1118 m) and 10 degrees about +y, in WebXR axes.
RESET_TRANSFORM = {'position': [-0.1, 0.0, -0.05],
                   'orientation': [0.0, round(math.sin(math.radians(5.0)), 8), 0.0,
                                   round(math.cos(math.radians(5.0)), 8)]}

RIGHT_STILL = (0.25, 0.95, -0.45)
LEFT_STILL = (-0.25, 0.95, -0.45)
VIEWER_BASE = (0.0, 1.6, 0.0)

Quat = Tuple[float, float, float, float]
Vec = Tuple[float, float, float]


def _axis_angle(axis: Vec, angle: float) -> Quat:
    norm = math.sqrt(sum(a * a for a in axis))
    s = math.sin(angle / 2.0) / norm
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0))


def _qmul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _yaw(angle: float) -> Quat:
    return _axis_angle((0.0, 1.0, 0.0), angle)


RIGHT_QUAT = _axis_angle((0.3, 0.5, 0.8), math.radians(35.0))
LEFT_QUAT = _axis_angle((-0.6, 0.2, 0.1), math.radians(-20.0))


def _r(values, digits: int) -> List[float]:
    return [round(v, digits) + 0.0 for v in values]


def _noise(i: int, phase: float, amplitude: float) -> float:
    """Closed-form noise in [-amplitude, amplitude]."""
    return amplitude * (0.7 * math.sin(0.91 * i + phase) + 0.3 * math.sin(2.37 * i + 3.0 * phase))


def _noisy_pos(base: Vec, i: int, phase: float, amplitude: float) -> Vec:
    return tuple(base[k] + _noise(i, phase + 1.3 * k, amplitude) for k in range(3))


def _noisy_quat(base: Quat, i: int, phase: float, amplitude_deg: float) -> Quat:
    # Rotation-vector noise whose norm stays within amplitude_deg.
    amp = math.radians(amplitude_deg) / math.sqrt(3.0)
    rv = [_noise(i, phase + 0.7 * k, amp) for k in range(3)]
    angle = math.sqrt(sum(c * c for c in rv))
    if angle == 0.0:
        return base
    q = _qmul(base, _axis_angle((rv[0], rv[1], rv[2]), angle))
    norm = math.sqrt(sum(c * c for c in q))
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


def _idx(seconds: float) -> int:
    return int(round(seconds * RATE_HZ))


def _buttons(grip: bool = False, trigger: float = 0.0) -> List[Dict[str, Any]]:
    buttons = []
    for k in range(NUM_BUTTONS):
        pressed, touched, value = False, False, 0.0
        if k == 0:
            pressed, touched, value = trigger > 0.9, trigger > 0.0, trigger
        elif k == 1:
            pressed, touched, value = grip, grip, 1.0 if grip else 0.0
        buttons.append({'p': pressed, 't': touched, 'v': round(value, 6) + 0.0})
    return buttons


def _hand(position: Vec, orientation: Quat, emulated: bool = False, grip: bool = False,
          trigger: float = 0.0, stick: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, Any]:
    return {
        'position': _r(position, 6),
        'orientation': _r(orientation, 8),
        'emulated_position': emulated,
        'buttons': _buttons(grip, trigger),
        'axes': [0.0, 0.0, stick[0], stick[1]],
    }


def _viewer(position: Vec, orientation: Quat) -> Dict[str, Any]:
    return {'position': _r(position, 6), 'orientation': _r(orientation, 8),
            'emulated_position': False}


def _frame(i: int, left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]],
           viewer: Optional[Dict[str, Any]], reset_count: int = 0, seq: Optional[int] = None,
           session_id: int = SESSION_ID,
           reset_event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    t_xr = round(T_XR_START_MS + i * 1000.0 / RATE_HZ, 6)
    controllers: Dict[str, Any] = {}
    if left is not None:
        controllers['left'] = left
    if right is not None:
        controllers['right'] = right
    frame = {
        'type': 'xr_frame',
        't_client': round(t_xr + CLIENT_OFFSET_MS, 6),
        't_xr': t_xr,
        'seq': i if seq is None else seq,
        'reset_count': reset_count,
        'session_id': session_id,
        'controllers': controllers,
        'viewer': viewer,
    }
    if reset_event is not None:
        frame['reset_event'] = reset_event
    return frame


def _t_recv(i: int) -> float:
    jitter = 0.0004 * math.sin(1.7 * i) + 0.0002 * math.sin(0.37 * i)
    return round(T_RECV_START + i / RATE_HZ + jitter, 6)


def _line(i: int, frame: Dict[str, Any]) -> str:
    return json.dumps({'t_recv': _t_recv(i), 'msg': frame}, ensure_ascii=False,
                      allow_nan=False)


def _static_viewer() -> Dict[str, Any]:
    return _viewer(VIEWER_BASE, (0.0, 0.0, 0.0, 1.0))


def still_controller() -> List[str]:
    """Both controllers still; the headset turns +-90 deg with a 0.3 m sideways sway."""
    lines = []
    for i in range(10 * RATE_HZ):
        t = i / RATE_HZ
        s = math.sin(2.0 * math.pi * t / 6.0)
        viewer = _viewer((VIEWER_BASE[0] + 0.3 * s, VIEWER_BASE[1], VIEWER_BASE[2]),
                         _yaw(math.radians(90.0) * s))
        right = _hand(_noisy_pos(RIGHT_STILL, i, 0.0, 0.0003),
                      _noisy_quat(RIGHT_QUAT, i, 0.5, 0.05))
        left = _hand(_noisy_pos(LEFT_STILL, i, 2.0, 0.0003),
                     _noisy_quat(LEFT_QUAT, i, 2.5, 0.05))
        lines.append(_line(i, _frame(i, left, right, viewer)))
    return lines


def moving_controller() -> List[str]:
    """Right controller moves 0.4 m along +x and turns 90 deg about +y; buttons change."""
    lines = []
    n = 6 * RATE_HZ
    for i in range(n):
        t = i / RATE_HZ
        s = i / (n - 1)
        position = (RIGHT_STILL[0] + 0.4 * s, RIGHT_STILL[1], RIGHT_STILL[2])
        trigger = min(max((t - 2.0) / 2.0, 0.0), 1.0)
        stick = (0.5, -0.5) if i >= _idx(3.0) else (0.0, 0.0)
        right = _hand(position, _yaw(math.radians(90.0) * s), grip=i >= _idx(1.0),
                      trigger=trigger, stick=stick)
        left = _hand(LEFT_STILL, LEFT_QUAT)
        lines.append(_line(i, _frame(i, left, right, _static_viewer())))
    return lines


def dropped_controller() -> List[str]:
    """Right controller absent 2-3 s, emulated with 2 cm drift 4-5 s."""
    lines = []
    for i in range(8 * RATE_HZ):
        base = _noisy_pos(RIGHT_STILL, i, 0.0, 0.0002)
        right: Optional[Dict[str, Any]] = _hand(base, RIGHT_QUAT)
        if _idx(2.0) <= i < _idx(3.0):
            right = None
        elif _idx(4.0) <= i < _idx(5.0):
            drift = 0.02 * (i - _idx(4.0)) / (_idx(5.0) - _idx(4.0))
            right = _hand((base[0] + drift, base[1], base[2]), RIGHT_QUAT, emulated=True)
        left = _hand(LEFT_STILL, LEFT_QUAT)
        lines.append(_line(i, _frame(i, left, right, _static_viewer())))
    return lines


def reset_event() -> List[str]:
    """
    Both controllers still; at 2.5 s reset_count becomes 1 and positions jump.

    The frame where the count rises carries the reset event once, with the movement the
    headset reports (``RESET_TRANSFORM``).
    """
    lines = []
    jump = (0.1, 0.0, 0.05)
    for i in range(5 * RATE_HZ):
        after = i >= _idx(2.5)
        shift = jump if after else (0.0, 0.0, 0.0)
        right = _hand(tuple(RIGHT_STILL[k] + shift[k] for k in range(3)), RIGHT_QUAT)
        left = _hand(tuple(LEFT_STILL[k] + shift[k] for k in range(3)), LEFT_QUAT)
        event = ({'count': 1, 'transform': RESET_TRANSFORM} if i == _idx(2.5) else None)
        lines.append(_line(i, _frame(i, left, right, _static_viewer(), 1 if after else 0,
                                     reset_event=event)))
    return lines


def new_session() -> List[str]:
    """
    Both controllers still; at 2.5 s the page is reloaded.

    The session id changes, seq restarts at 0, reset_count stays 0 and the positions jump
    because local-floor was redefined.
    """
    lines = []
    jump = (-0.08, 0.0, 0.12)
    reload_at = _idx(2.5)
    for i in range(5 * RATE_HZ):
        after = i >= reload_at
        shift = jump if after else (0.0, 0.0, 0.0)
        right = _hand(tuple(RIGHT_STILL[k] + shift[k] for k in range(3)), RIGHT_QUAT)
        left = _hand(tuple(LEFT_STILL[k] + shift[k] for k in range(3)), LEFT_QUAT)
        frame = _frame(i, left, right, _static_viewer(), seq=i - reload_at if after else i,
                       session_id=SESSION_ID_2 if after else SESSION_ID)
        lines.append(_line(i, frame))
    return lines


def _mutated(i: int, mutate) -> Dict[str, Any]:
    """Build a valid still frame, apply ``mutate`` to it and return it."""
    frame = _frame(i, _hand(LEFT_STILL, LEFT_QUAT), _hand(RIGHT_STILL, RIGHT_QUAT),
                   _static_viewer())
    mutate(frame)
    return frame


def _set(path: Tuple[Any, ...], value: Any):
    def apply(frame: Dict[str, Any]) -> None:
        node = frame
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return apply


def _delete(path: Tuple[Any, ...]):
    def apply(frame: Dict[str, Any]) -> None:
        node = frame
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return apply


def _stock_client(frame: Dict[str, Any]) -> None:
    for key in ('seq', 't_xr', 'reset_count', 'session_id'):
        del frame[key]
    for hand in frame['controllers'].values():
        del hand['emulated_position']
    del frame['viewer']


def _extra_keys(frame: Dict[str, Any]) -> None:
    frame['extra'] = {'a': [1, 2, 3]}
    frame['controllers']['head'] = {'position': 'ignored'}
    frame['controllers']['left']['handedness'] = 'left'
    frame['controllers']['right']['profiles'] = ['meta-quest-touch-plus']
    frame['controllers']['right']['buttons'][0]['extra'] = None
    frame['viewer']['extra'] = 1


def _viewer_only(frame: Dict[str, Any]) -> None:
    frame['controllers'] = {}


def _hand_tracked_right(frame: Dict[str, Any]) -> None:
    frame['controllers']['right'] = {'hand_tracked': True}


def _hand_as_controller(frame: Dict[str, Any]) -> None:
    """Turn the left hand into what the Quest 3 sends for a tracked hand (10 buttons, 0 axes)."""
    frame['controllers']['left']['buttons'] = _buttons()[:7] + _buttons()[:3]
    frame['controllers']['left']['axes'] = []


def _tracked_with_junk(frame: Dict[str, Any]) -> None:
    frame['controllers']['left'] = {'hand_tracked': True, 'position': [1, 2], 'buttons': 7}


# (kind, payload, expected). kind 'msg' payload is a callable building the message
# (or a plain JSON value); kind 'raw' payload is the received text.
def _malformed_cases() -> List[Tuple[str, Any, str]]:
    return [
        ('frame', _set(('seq',), 0), 'frame'),
        ('raw', 'not json at all', 'malformed'),
        ('raw', '{"type": "xr_frame", ', 'malformed'),
        ('value', [1, 2, 3], 'malformed'),
        ('value', 'xr_frame', 'malformed'),
        ('value', {'foo': 1}, 'malformed'),
        ('value', {'type': 123}, 'malformed'),
        ('value', {'type': 'camera_list', 'cameras': []}, 'ignored'),
        ('value', {'type': 'echo', 'echo': {'t_client': 1}}, 'ignored'),
        ('frame', _stock_client, 'malformed'),
        ('frame', _set(('seq',), -1), 'malformed'),
        ('frame', _set(('seq',), True), 'malformed'),
        ('frame', _set(('t_xr',), 'abc'), 'malformed'),
        ('raw', '{"type":"xr_frame","t_client":1000.4,"t_xr":NaN,"seq":3,"reset_count":0,'
                '"controllers":{},"viewer":null}', 'malformed'),
        ('frame', _set(('controllers', 'right', 'position'), [0.1, 0.2]), 'malformed'),
        ('frame', _set(('controllers', 'right', 'orientation'), [0.0, 0.0, 1.0]), 'malformed'),
        ('frame', _set(('controllers', 'left', 'orientation'), [0.0, 0.0, 0.0, 2.0]),
         'malformed'),
        ('frame', _delete(('controllers', 'right', 'emulated_position')), 'malformed'),
        ('frame', _set(('controllers', 'right', 'emulated_position'), 'yes'), 'malformed'),
        ('frame', _set(('controllers', 'right', 'buttons'), {'0': {'p': True}}), 'malformed'),
        ('frame', _delete(('controllers', 'right', 'buttons', 2, 'p')), 'malformed'),
        ('frame', _set(('controllers', 'left', 'axes'), [0.0, 0.0, 'x', 0.0]), 'malformed'),
        ('frame', _set(('controllers', 'right'), None), 'malformed'),
        ('frame', _delete(('viewer',)), 'malformed'),
        ('frame', _viewer_only, 'frame'),
        ('frame', _hand_tracked_right, 'frame'),
        ('frame', _extra_keys, 'frame'),
        ('frame', _set(('t_client',), None), 'malformed'),
        ('frame', _set(('seq',), 1.5), 'malformed'),
        ('frame', _set(('viewer', 'orientation'), [0.0, 0.0, 0.0]), 'malformed'),
        ('frame', _tracked_with_junk, 'frame'),
        ('raw', '{"type":"xr_frame","t_client":1,"t_xr":Infinity,"seq":3,"reset_count":0,'
                '"controllers":{},"viewer":null}', 'malformed'),
        ('frame', _set(('seq',), 7), 'frame'),
        ('frame', _delete(('session_id',)), 'malformed'),
        ('frame', _set(('session_id',), -1), 'malformed'),
        ('frame', _set(('session_id',), 2 ** 32), 'malformed'),
        ('frame', _set(('session_id',), 1.0), 'malformed'),
        ('frame', _set(('session_id',), 2 ** 32 - 1), 'frame'),
        ('frame', _set(('controllers', 'right', 'profiles'), 'meta-quest-touch-plus'),
         'malformed'),
        ('frame', _set(('controllers', 'right', 'profiles'), ['meta-quest-touch-plus', 3]),
         'malformed'),
        ('frame', _set(('controllers', 'right', 'profiles'), ['meta-quest-touch-plus']), 'frame'),
        ('frame', _set(('reset_event',), {'count': 2, 'transform': None}), 'frame'),
        ('frame', _set(('reset_event',), {'count': 2, 'transform': RESET_TRANSFORM}), 'frame'),
        ('frame', _set(('reset_event',), None), 'malformed'),
        ('frame', _set(('reset_event',), [1, 2]), 'malformed'),
        ('frame', _set(('reset_event',), {'transform': None}), 'malformed'),
        ('frame', _set(('reset_event',), {'count': -1, 'transform': None}), 'malformed'),
        ('frame', _set(('reset_event',), {'count': 1.5, 'transform': None}), 'malformed'),
        ('frame', _set(('reset_event',), {'count': 1}), 'malformed'),
        ('frame', _set(('reset_event',), {'count': 1, 'transform': 'moved'}), 'malformed'),
        ('frame', _set(('reset_event',), {'count': 1, 'transform': {
            'position': [0.1, 0.0], 'orientation': [0.0, 0.0, 0.0, 1.0]}}), 'malformed'),
        ('frame', _set(('reset_event',), {'count': 1, 'transform': {
            'position': [0.1, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.0, 2.0]}}), 'malformed'),
        ('frame', _set(('reset_event',), {'count': 1, 'transform': {
            'position': [0.1, 0.0, 0.0]}}), 'malformed'),
        ('frame', _hand_as_controller, 'frame'),
    ]


def malformed() -> List[str]:
    """Return lines with valid frames at both ends and odd messages between them."""
    lines = []
    for n, (kind, payload, _expected) in enumerate(_malformed_cases()):
        t_recv = _t_recv(n)
        if kind == 'raw':
            lines.append(json.dumps({'t_recv': t_recv, 'raw': payload}, ensure_ascii=False,
                                    allow_nan=False))
            continue
        msg = _mutated(n, payload) if kind == 'frame' else payload
        lines.append(json.dumps({'t_recv': t_recv, 'msg': msg}, ensure_ascii=False,
                                allow_nan=False))
    return lines


def expected() -> List[str]:
    """Return 'frame', 'ignored' or 'malformed' for each line of the malformed fixture."""
    return [case[2] for case in _malformed_cases()]


# ---- Base-driving fixtures ---------------------------------------------------------------
#
# Every fixture opens as the servo package's clutch fixtures do: the calibration button (A,
# xr-standard slot 4) is pressed on the right controller at 0.3 s while it points
# ``BASE_CAL_YAW_DEG`` to the left, the arm button (B, slot 5) at 0.7 s. The left controller is
# present throughout, at rest, with its trigger at 0.0. Left grip is slot 1, the thumbsticks
# are ``axes[2:4]`` (x right, y down: pushed up, forward, is negative).

BASE_CAL_YAW_DEG = 20.0
BASE_START = (0.15, 1.35, -0.5)
BASE_FORWARD = (-math.sin(math.radians(BASE_CAL_YAW_DEG)), 0.0,
                -math.cos(math.radians(BASE_CAL_YAW_DEG)))
BASE_TILT = (math.sin(math.radians(-25.0)), 0.0, 0.0, math.cos(math.radians(-25.0)))
BASE_HELD = _qmul(_yaw(math.radians(BASE_CAL_YAW_DEG)), BASE_TILT)
STICK_FORWARD = (0.0, -1.0)
STICK_RIGHT = (1.0, 0.0)
STICK_CENTRE = (0.0, 0.0)

Stick = Tuple[float, float]


def _smooth(t: float, start: float, duration: float) -> float:
    """Minimum-jerk ramp from 0 to 1 over ``duration`` starting at ``start``."""
    s = min(max((t - start) / duration, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def _along(distance: float) -> Vec:
    return tuple(BASE_START[k] + distance * BASE_FORWARD[k] for k in range(3))


def _window(i: int, start: float, end: float) -> bool:
    """Return True for frame ``i`` from ``start`` s up to (not including) ``end`` s."""
    return _idx(start) <= i < _idx(end)


def _arm_hand(i: int, grip: bool, stick: Stick, reach: float) -> Dict[str, Any]:
    """Return the right controller: calibrate and arm, then held at ``reach`` m forward."""
    t = i / RATE_HZ
    if t < 1.0:
        pressed = {4: _window(i, 0.3, 0.5), 5: _window(i, 0.7, 0.9)}
        position, orientation = BASE_START, _yaw(math.radians(BASE_CAL_YAW_DEG))
    else:
        pressed = {}
        position, orientation = _along(reach), BASE_HELD
    buttons = []
    for k in range(NUM_BUTTONS):
        down = grip if k == 1 else pressed.get(k, False)
        buttons.append({'p': down, 't': down, 'v': 1.0 if down else 0.0})
    return {'position': _r(position, 6), 'orientation': _r(orientation, 8),
            'emulated_position': False, 'buttons': buttons,
            'axes': [0.0, 0.0, stick[0], stick[1]]}


def _stroke(i: int, start: float, back: float) -> float:
    """Return a stroke: 0.12 m out over 1.2 s from ``start``, back over 0.6 s from ``back``."""
    t = i / RATE_HZ
    return 0.12 * (_smooth(t, start, 1.2) - _smooth(t, back, 0.6))


def _base_lines(duration: float, left_grip=None, left_stick=None, right_grip=None,
                right_stick=None, reach=None, left_present=None, session=None, seq=None,
                dropped=None) -> List[str]:
    """
    Return a recording of ``duration`` s. Each argument is a function of the frame index.

    ``left_grip``, ``right_grip``: pressed; ``left_stick``, ``right_stick``: (x, y);
    ``reach``: the right hand's distance forward (m); ``left_present``: the left controller is
    in the frame; ``session``: the session id; ``seq``: the sequence number; ``dropped``: the
    frame is not sent at all (its index is skipped, as a stall does).
    """
    lines = []
    for i in range(_idx(duration)):
        if dropped is not None and dropped(i):
            continue
        left = None
        if left_present is None or left_present(i):
            left = _hand(LEFT_STILL, LEFT_QUAT, grip=bool(left_grip and left_grip(i)),
                         trigger=0.0, stick=left_stick(i) if left_stick else STICK_CENTRE)
        right = _arm_hand(i, bool(right_grip and right_grip(i)),
                          right_stick(i) if right_stick else STICK_CENTRE,
                          reach(i) if reach else 0.0)
        frame = _frame(i, left, right, _static_viewer(),
                       seq=seq(i) if seq else None,
                       session_id=session(i) if session else SESSION_ID)
        lines.append(_line(i, frame))
    return lines


def _forward_until(end: float, start: float = 1.5):
    """Return a left stick function: forward from ``start`` s to ``end`` s, else centred."""
    return lambda i: STICK_FORWARD if _window(i, start, end) else STICK_CENTRE


def base_drive_between_strokes() -> List[str]:
    """
    Drive forward, arm stroke, drive sideways, turn, arm stroke: 15 s.

    Left grip with the left stick forward 1.5-3.0 s; right grip stroke 4.0-6.0 s (0.12 m
    forward, hand back 6.2-6.8 s); left grip with the stick right 7.0-8.5 s; left grip with the
    left stick centred and the right stick x at -1.0 (turn left) 9.5-11.0 s; right grip stroke
    12.0-14.0 s (hand back 14.2-14.8 s).
    """
    def left_grip(i):
        return any(_window(i, a, b) for a, b in ((1.5, 3.0), (7.0, 8.5), (9.5, 11.0)))

    def left_stick(i):
        return STICK_FORWARD if _window(i, 1.5, 3.0) else (
            STICK_RIGHT if _window(i, 7.0, 8.5) else STICK_CENTRE)

    def right_grip(i):
        return _window(i, 4.0, 6.0) or _window(i, 12.0, 14.0)

    def reach(i):
        return _stroke(i, 4.3, 6.2) + _stroke(i, 12.3, 14.2)

    def right_stick(i):
        return (-1.0, 0.0) if _window(i, 9.5, 11.0) else STICK_CENTRE
    return _base_lines(15.0, left_grip, left_stick, right_grip, right_stick, reach)


def base_fault_release() -> List[str]:
    """Drive forward 1.5-3.0 s; the left grip is released at 3.0 s, the stick centres 3.3 s."""
    return _base_lines(4.5, lambda i: _window(i, 1.5, 3.0), _forward_until(3.3))


def base_fault_stale() -> List[str]:
    """Drive forward 1.5-4.0 s; no frame at all arrives 3.0-3.5 s; the grip stays held."""
    return _base_lines(4.5, lambda i: _window(i, 1.5, 4.0), _forward_until(4.0),
                       dropped=lambda i: _window(i, 3.0, 3.5))


def base_fault_left_missing() -> List[str]:
    """Drive forward 1.5-4.0 s; the left controller is absent from 3.0 to 3.5 s, then back."""
    return _base_lines(4.5, lambda i: _window(i, 1.5, 4.0), _forward_until(4.0),
                       left_present=lambda i: not _window(i, 3.0, 3.5))


def base_fault_session() -> List[str]:
    """Drive forward 1.5-4.0 s; the session id changes at 3.0 s (seq and poses continue)."""
    return _base_lines(4.5, lambda i: _window(i, 1.5, 4.0), _forward_until(4.0),
                       session=lambda i: SESSION_ID_2 if i >= _idx(3.0) else SESSION_ID)


def base_fault_seq_backwards() -> List[str]:
    """Drive forward 1.5-4.0 s; at 3.0 s the sequence number steps back by 50 and counts on."""
    return _base_lines(4.5, lambda i: _window(i, 1.5, 4.0), _forward_until(4.0),
                       seq=lambda i: i - 50 if i >= _idx(3.0) else i)


def base_both_grips_same_frame() -> List[str]:
    """Both grips pressed in the same frame at 1.5 s, left stick forward, held to 4.0 s."""
    return _base_lines(4.5, lambda i: _window(i, 1.5, 4.0), _forward_until(4.0),
                       lambda i: _window(i, 1.5, 4.0))


def base_right_grip_during_drive() -> List[str]:
    """Drive forward 1.5-4.0 s; the right grip is pressed at 3.0 s while the left is held."""
    return _base_lines(4.5, lambda i: _window(i, 1.5, 4.0), _forward_until(4.0),
                       lambda i: _window(i, 3.0, 4.0))


def base_left_grip_during_stroke() -> List[str]:
    """Right grip 1.5-4.0 s (hand 0.12 m out 1.8-3.0 s); left grip and stick forward 3.0-4.0 s."""
    return _base_lines(4.5, lambda i: _window(i, 3.0, 4.0), _forward_until(4.0, 3.0),
                       lambda i: _window(i, 1.5, 4.0),
                       reach=lambda i: 0.12 * _smooth(i / RATE_HZ, 1.8, 1.2))


def base_reset_pull_while_driving() -> List[str]:
    """Drive forward 1.5-4.0 s; the right stick is pulled fully back (y +1.0) 2.5-3.5 s."""
    return _base_lines(4.5, lambda i: _window(i, 1.5, 4.0), _forward_until(4.0),
                       right_stick=lambda i: (0.0, 1.0) if _window(i, 2.5, 3.5)
                       else STICK_CENTRE)


def base_turn_diagonal() -> List[str]:
    """Left grip, left stick centred, right stick (-0.9, 0.7), a turn not a pull, 1.5-4.0 s."""
    return _base_lines(4.5, lambda i: _window(i, 1.5, 4.0),
                       right_stick=lambda i: (-0.9, 0.7) if _window(i, 1.5, 4.0)
                       else STICK_CENTRE)


def generate_base() -> Dict[str, List[str]]:
    """Return the base-driving fixture name -> exact file lines (without newlines)."""
    makers = [base_drive_between_strokes, base_fault_release, base_fault_stale,
              base_fault_left_missing, base_fault_session, base_fault_seq_backwards,
              base_both_grips_same_frame, base_right_grip_during_drive,
              base_left_grip_during_stroke, base_reset_pull_while_driving,
              base_turn_diagonal]
    return {maker.__name__: maker() for maker in makers}


def generate() -> Dict[str, List[str]]:
    """Return fixture name -> exact file lines (without newlines)."""
    return {
        'still_controller': still_controller(),
        'moving_controller': moving_controller(),
        'dropped_controller': dropped_controller(),
        'reset_event': reset_event(),
        'new_session': new_session(),
        'malformed': malformed(),
    }


def main() -> None:
    """Write every fixture as ``<name>.jsonl`` next to this script or into argv[1]."""
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    for name, lines in {**generate(), **generate_base()}.items():
        path = os.path.join(out_dir, f'{name}.jsonl')
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            for line in lines:
                handle.write(line + '\n')
        print(f'wrote {path} ({len(lines)} lines)')


if __name__ == '__main__':
    main()
