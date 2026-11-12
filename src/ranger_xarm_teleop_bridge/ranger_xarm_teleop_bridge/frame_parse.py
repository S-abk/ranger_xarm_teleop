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
Strict parsing of the vr-teleop-kit relay messages (no ROS, INV-6).

One WebSocket text message is one headset frame. ``parse_text`` classifies it as a
``ParsedFrame``, an ``Ignored`` message (another relay message type) or a
``Malformed`` one with a specific reason. It never raises (INV-7). A frame is
accepted only if every present hand is valid; otherwise the whole frame is
``Malformed`` and nothing may be published for it.

A hand that is omitted, or that is a tracked hand (``{"hand_tracked": true}``)
is ``None``: not present; the tracked-hand case is flagged in ``left_hand_tracked``
/ ``right_hand_tracked``. No other pose is ever substituted (INV-2).
``session_id``, ``seq``, ``t_xr`` and ``t_client`` are carried through unchanged
(INV-4). A frame without ``session_id`` (or any other field the patched client
adds) is malformed.
Positions are WebXR axes in metres and quaternions are xyzw; conversion to ROS
axes happens elsewhere (``axes.py``).
"""

from dataclasses import dataclass
import json
import math
from typing import Any, Optional, Tuple, Union

ORIENTATION_NORM_TOLERANCE = 1e-3
UINT32_MAX = 2 ** 32 - 1
PATCHED_CLIENT_KEYS = ('session_id', 'seq', 't_xr', 'reset_count')


@dataclass(frozen=True)
class Button:
    """One gamepad button."""

    pressed: bool
    touched: bool
    value: float


@dataclass(frozen=True)
class HandSample:
    """Pose, buttons and axes of one tracked controller."""

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    emulated_position: bool
    buttons: Tuple[Button, ...]
    axes: Tuple[float, ...]


@dataclass(frozen=True)
class ViewerSample:
    """Headset pose."""

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    emulated_position: bool


@dataclass(frozen=True)
class ParsedFrame:
    """A valid ``xr_frame`` message."""

    session_id: int
    seq: int
    xr_time_ms: float
    client_time_ms: float
    reset_count: int
    left: Optional[HandSample]
    right: Optional[HandSample]
    viewer: Optional[ViewerSample]
    # True when the input source was a tracked hand; the hand itself is then None.
    left_hand_tracked: bool = False
    right_hand_tracked: bool = False


@dataclass(frozen=True)
class Ignored:
    """A well-formed relay message that is not an ``xr_frame``."""

    type: str  # noqa: A003


@dataclass(frozen=True)
class Malformed:
    """A message that is not a valid frame; ``reason`` names the offending field."""

    reason: str


ParseResult = Union[ParsedFrame, Ignored, Malformed]


class _Invalid(Exception):
    """Internal: carries the reason a frame is malformed."""


def _reject_constant(token: str) -> Any:
    raise ValueError(f'non-finite JSON token {token}')


def loads_strict(text: str) -> Any:
    """
    Decode JSON, rejecting the NaN, Infinity and -Infinity tokens.

    Raises ``ValueError`` (or ``RecursionError``) when ``text`` is not strict JSON.
    """
    return json.loads(text, parse_constant=_reject_constant)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any) -> Optional[float]:
    """Return ``value`` as a float if it is a finite JSON number, else None."""
    if not _is_number(value):
        return None
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _number(value: Any, name: str) -> float:
    result = _finite(value)
    if result is None:
        raise _Invalid(f'{name}: expected a finite number')
    return result


def _count(obj: dict, key: str, maximum: Optional[int] = None) -> int:
    value = obj[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _Invalid(f'{key}: expected an integer >= 0')
    if maximum is not None and value > maximum:
        raise _Invalid(f'{key}: expected an integer <= {maximum}')
    return int(value)


def _vector(value: Any, size: int, name: str) -> Tuple[float, ...]:
    if isinstance(value, list) and len(value) == size:
        result = tuple(_finite(v) for v in value)
        if all(v is not None for v in result):
            return result  # type: ignore[return-value]
    raise _Invalid(f'{name}: expected {size} finite numbers')


def _position(obj: dict, prefix: str) -> Tuple[float, float, float]:
    if 'position' not in obj:
        raise _Invalid(f'{prefix}.position: missing')
    return _vector(obj['position'], 3, f'{prefix}.position')  # type: ignore[return-value]


def _orientation(obj: dict, prefix: str) -> Tuple[float, float, float, float]:
    name = f'{prefix}.orientation'
    if 'orientation' not in obj:
        raise _Invalid(f'{name}: missing')
    quat = _vector(obj['orientation'], 4, name)
    norm = math.hypot(*quat)
    if abs(norm - 1.0) > ORIENTATION_NORM_TOLERANCE:
        raise _Invalid(f'{name}: expected a unit quaternion, norm is {norm!r}')
    return quat  # type: ignore[return-value]


def _emulated(obj: dict, prefix: str) -> bool:
    name = f'{prefix}.emulated_position'
    if 'emulated_position' not in obj:
        raise _Invalid(f'{name}: missing')
    value = obj['emulated_position']
    if not isinstance(value, bool):
        raise _Invalid(f'{name}: expected bool')
    return value


def _button(value: Any, name: str) -> Button:
    if not isinstance(value, dict):
        raise _Invalid(f'{name}: expected an object')
    if 'p' not in value:
        raise _Invalid(f'{name}.p: missing')
    if not isinstance(value['p'], bool):
        raise _Invalid(f'{name}.p: expected bool')
    touched = value.get('t', False)
    if not isinstance(touched, bool):
        raise _Invalid(f'{name}.t: expected bool')
    analog = _number(value['v'], f'{name}.v') if 'v' in value else 0.0
    return Button(pressed=value['p'], touched=touched, value=analog)


def _hand(value: Any, name: str) -> Tuple[Optional[HandSample], bool]:
    """Return (sample, hand_tracked); a tracked hand gives (None, True)."""
    if not isinstance(value, dict):
        raise _Invalid(f'{name}: expected an object')
    if value.get('hand_tracked') is True:
        return None, True
    position = _position(value, name)
    orientation = _orientation(value, name)
    emulated = _emulated(value, name)
    buttons = value.get('buttons')
    if not isinstance(buttons, list):
        raise _Invalid(f'{name}.buttons: expected a list')
    parsed_buttons = tuple(_button(b, f'{name}.buttons[{i}]') for i, b in enumerate(buttons))
    axes = value.get('axes')
    if not isinstance(axes, list):
        raise _Invalid(f'{name}.axes: expected a list')
    parsed_axes = tuple(_number(a, f'{name}.axes[{i}]') for i, a in enumerate(axes))
    return HandSample(position, orientation, emulated, parsed_buttons, parsed_axes), False


def _viewer(value: Any) -> Optional[ViewerSample]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _Invalid('viewer: expected an object or null')
    return ViewerSample(_position(value, 'viewer'), _orientation(value, 'viewer'),
                        _emulated(value, 'viewer'))


def _frame(obj: dict) -> ParsedFrame:
    for key in PATCHED_CLIENT_KEYS:
        if key not in obj:
            raise _Invalid(f'client not patched: missing {key}')
    session_id = _count(obj, 'session_id', UINT32_MAX)
    seq = _count(obj, 'seq')
    xr_time = _number(obj['t_xr'], 't_xr')
    if 't_client' not in obj:
        raise _Invalid('t_client: missing')
    client_time = _number(obj['t_client'], 't_client')
    reset_count = _count(obj, 'reset_count')
    controllers = obj.get('controllers')
    if not isinstance(controllers, dict):
        raise _Invalid('controllers: expected an object')
    if 'viewer' not in obj:
        raise _Invalid('viewer: missing (expected an object or null)')
    hands = {}
    tracked = {}
    for side in ('left', 'right'):
        hands[side], tracked[side] = None, False
        if side in controllers:
            hands[side], tracked[side] = _hand(controllers[side], f'controllers.{side}')
    viewer = _viewer(obj['viewer'])
    return ParsedFrame(session_id, seq, xr_time, client_time, reset_count,
                       hands['left'], hands['right'], viewer,
                       left_hand_tracked=tracked['left'], right_hand_tracked=tracked['right'])


def parse_message(obj: Any) -> ParseResult:
    """Classify an already decoded JSON value; never raises."""
    try:
        if not isinstance(obj, dict):
            return Malformed(f'message: expected a JSON object, got {type(obj).__name__}')
        if 'type' not in obj:
            return Malformed('type: missing')
        msg_type = obj['type']
        if not isinstance(msg_type, str):
            return Malformed('type: expected a string')
        if msg_type != 'xr_frame':
            return Ignored(msg_type)
        return _frame(obj)
    except _Invalid as err:
        return Malformed(str(err))
    except Exception as err:  # INV-7: a bug here must not escape as a raise
        return Malformed(f'unexpected {type(err).__name__} while parsing: {err}')


def parse_text(text: str) -> ParseResult:
    """Classify one relay text message; never raises."""
    try:
        obj = loads_strict(text)
    except (ValueError, RecursionError, TypeError) as err:
        return Malformed(f'not strict JSON: {err}')
    return parse_message(obj)
