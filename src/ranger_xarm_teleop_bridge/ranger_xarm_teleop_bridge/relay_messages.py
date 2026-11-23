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
The two message types the bridge sends to the headset over the relay (no ROS).

``teleop_haptic``: one vibration request (a Haptic message). ``teleop_status``: what the
panel shows (an OperatorStatus message), sent at its own rate. Both are built from any object
with the message's field names (the ROS message, or the pure ``Snapshot`` in tests), and get
their age and the ``echo`` only when they are sent. JSON never carries NaN (the page's
JSON.parse rejects it): every non-finite number becomes null.

``echo`` lets the page age a message on its own clock without clock sync: it names the newest
headset frame the bridge received (``t_client`` is the page's clock in that frame) and how long
the bridge has held it, so ``now - (t_client + held_ms)`` on the page is the uplink plus
downlink time: an over-estimate of the delivery delay.
"""

import json
import math
from typing import Any, Dict, Optional, Sequence, Tuple

from . import base_feedback
from .episode_feedback import EVENT_NAMES
from .operator_status import EPISODE_STATE_NAMES
from .reset_feedback import REASON_NAMES, STATE_NAMES

PROTOCOL_VERSION = 1
HAPTIC_TYPE = 'teleop_haptic'
STATUS_TYPE = 'teleop_status'

# Newest frame: (session_id, seq, t_client ms, monotonic arrival s).
FrameInfo = Tuple[int, int, float, float]


def _num(value) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def clean(value: Any) -> Any:
    """Return ``value`` with every non-finite float replaced by None, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def haptic_payload(msg, run: int, ident: int) -> Dict[str, Any]:
    """Build a ``teleop_haptic`` body (without age and echo) from a Haptic-like object."""
    intensity: Sequence[float] = list(msg.intensity)
    on_ms: Sequence[int] = list(msg.on_ms)
    off_ms: Sequence[int] = list(msg.off_ms)
    if not len(intensity) == len(on_ms) == len(off_ms):
        raise ValueError('intensity, on_ms and off_ms differ in length')
    # Rounded: the message holds float32 (0.8 arrives as 0.800000011920929).
    pulses = [[round(min(1.0, max(0.0, float(i))), 4), int(on), int(off)]
              for i, on, off in zip(intensity, on_ms, off_ms)]
    return {
        'type': HAPTIC_TYPE, 'v': PROTOCOL_VERSION, 'run': int(run), 'id': int(ident),
        'hand': str(msg.hand), 'source': str(msg.source), 'event': str(msg.event),
        'priority': int(msg.priority), 'pulses': pulses,
        'max_age_ms': 1000.0 * float(msg.max_age), 'missed': int(msg.missed),
        'source_count': int(msg.source_count),
    }


def status_payload(msg, run: int, seq: int, panel: Dict[str, Any], stale_ms: float,
                   period_ms: float, sim_text: str = '') -> Dict[str, Any]:
    """
    Build a ``teleop_status`` body (without age and echo) from an OperatorStatus-like object.

    ``sim_text``: the "simulator slow" line (operator_status.SimSpeed), '' when not slow.
    """
    teleop = None
    if msg.teleop_valid:
        fault_current = msg.last_fault == msg.hold_reason
        teleop = {
            'armed': bool(msg.armed), 'clutch': bool(msg.clutch_engaged),
            'calibrated': bool(msg.calibrated), 'motion_enabled': bool(msg.motion_enabled),
            'scale': _num(msg.translation_scale),
            'hold': str(msg.hold_text),
            # A fault is listed apart from the hold only once it no longer applies.
            'fault': '' if fault_current else str(msg.last_fault_text),
            'fault_age_s': None if fault_current else _num(msg.last_fault_age),
            'limit': str(msg.limit_text),
            'gripper': str(msg.gripper_text),
        }
    episode = None
    if msg.episode_valid:
        episode = {
            'state': EPISODE_STATE_NAMES.get(int(msg.episode_state), 'NOT_READY'),
            'index': int(msg.episode_index), 'elapsed_s': _num(msg.elapsed),
            'kept': int(msg.kept_count), 'discarded': int(msg.discarded_count),
            'discard_armed': bool(msg.discard_armed),
            'last_event': '' if int(msg.last_event) == 0
            else EVENT_NAMES.get(int(msg.last_event), ''),
            'last_event_text': str(msg.last_event_text),
            'last_event_age_s': _num(msg.last_event_age),
            'refusal': str(msg.last_refusal_text),
            'refusal_age_s': _num(msg.last_refusal_age),
            'missed': int(msg.events_missed),
            'disk_minutes_left': _num(msg.disk_minutes_left),
        }
    body = {
        'type': STATUS_TYPE, 'v': PROTOCOL_VERSION, 'run': int(run), 'seq': int(seq),
        'stale_ms': float(stale_ms), 'period_ms': float(period_ms),
        'teleop': teleop, 'teleop_age_s': _num(msg.teleop_age),
        'episode': episode, 'episode_age_s': _num(msg.episode_age),
        'sim_text': str(sim_text),
        'panel': dict(panel),
    }
    # The reset section exists only once a reset state has been seen: with the feature off the
    # body is what it always was.
    if math.isfinite(float(getattr(msg, 'reset_age', math.nan))):
        valid = bool(msg.reset_valid)
        body['reset'] = {
            'valid': valid,
            'state': STATE_NAMES.get(int(msg.reset_state), f'UNKNOWN_{int(msg.reset_state)}')
            if valid else None,
            'reason': REASON_NAMES.get(int(msg.reset_reason), f'UNKNOWN_{int(msg.reset_reason)}')
            if valid else None,
            'text': str(msg.reset_text) if valid else '',
            'event_age_s': _num(msg.reset_event_age) if valid else None,
        }
    # Likewise the base section: only once a base state has been seen.
    if math.isfinite(float(getattr(msg, 'base_age', math.nan))):
        valid = bool(msg.base_valid)
        body['base'] = {
            'valid': valid,
            'state': base_feedback.STATE_NAMES.get(int(msg.base_state),
                                                   f'UNKNOWN_{int(msg.base_state)}')
            if valid else None,
            'reason': base_feedback.REASON_NAMES.get(int(msg.base_reason),
                                                     f'UNKNOWN_{int(msg.base_reason)}')
            if valid else None,
            'text': str(msg.base_text) if valid else '',
            'event_age_s': _num(msg.base_event_age) if valid else None,
        }
    return body


def echo_of(frame: Optional[FrameInfo], now: float) -> Optional[Dict[str, Any]]:
    """Return the ``echo`` for a message sent at monotonic ``now``, or None without a frame."""
    if frame is None:
        return None
    session_id, seq, t_client, arrived = frame
    return {'session_id': int(session_id), 'seq': int(seq), 't_client': float(t_client),
            'held_ms': max(0.0, 1000.0 * (now - arrived))}


def encode(payload: Dict[str, Any], age_s: float, echo: Optional[Dict[str, Any]]) -> str:
    """Return the JSON text to send: the payload with its age (ms) and echo."""
    body = dict(payload)
    body['age_ms'] = round(1000.0 * max(0.0, float(age_s)), 3)
    body['echo'] = echo
    return json.dumps(clean(body), allow_nan=False, separators=(',', ':'))


def panel_config(anchor: str, offset: Sequence[float], width_m: float,
                 visible_default: bool) -> Dict[str, Any]:
    """Return the panel placement sent with every status (validated)."""
    if anchor not in ('left_controller', 'head'):
        raise ValueError(f"panel_anchor must be 'left_controller' or 'head', not {anchor!r}")
    offset = [float(v) for v in offset]
    if len(offset) != 3 or not all(math.isfinite(v) and abs(v) <= 2.0 for v in offset):
        raise ValueError('panel_offset must be three finite numbers within 2 m')
    if not 0.05 <= float(width_m) <= 0.6:          # the page's own range (feedback.js)
        raise ValueError('panel_width_m must be 0.05..0.6')
    return {'anchor': anchor, 'offset': offset, 'width_m': float(width_m),
            'visible_default': bool(visible_default)}
