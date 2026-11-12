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
           session_id: int = SESSION_ID) -> Dict[str, Any]:
    t_xr = round(T_XR_START_MS + i * 1000.0 / RATE_HZ, 6)
    controllers: Dict[str, Any] = {}
    if left is not None:
        controllers['left'] = left
    if right is not None:
        controllers['right'] = right
    return {
        'type': 'xr_frame',
        't_client': round(t_xr + CLIENT_OFFSET_MS, 6),
        't_xr': t_xr,
        'seq': i if seq is None else seq,
        'reset_count': reset_count,
        'session_id': session_id,
        'controllers': controllers,
        'viewer': viewer,
    }


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
    """Both controllers still; at 2.5 s reset_count becomes 1 and positions jump."""
    lines = []
    jump = (0.1, 0.0, 0.05)
    for i in range(5 * RATE_HZ):
        after = i >= _idx(2.5)
        shift = jump if after else (0.0, 0.0, 0.0)
        right = _hand(tuple(RIGHT_STILL[k] + shift[k] for k in range(3)), RIGHT_QUAT)
        left = _hand(tuple(LEFT_STILL[k] + shift[k] for k in range(3)), LEFT_QUAT)
        lines.append(_line(i, _frame(i, left, right, _static_viewer(), 1 if after else 0)))
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
    for name, lines in generate().items():
        path = os.path.join(out_dir, f'{name}.jsonl')
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            for line in lines:
                handle.write(line + '\n')
        print(f'wrote {path} ({len(lines)} lines)')


if __name__ == '__main__':
    main()
