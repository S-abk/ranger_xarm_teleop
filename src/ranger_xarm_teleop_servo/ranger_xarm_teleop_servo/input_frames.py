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
Pair the messages of one headset frame by their stamp (no ROS).

Every message published from one headset frame (both poses, joy, status) carries the same
``header.stamp`` (the contract's "Pairing poses with status"). A frame is complete when its
status has arrived together with its joy, its right controller pose if the status announces
one, and its headset pose if the status announces one. A status waits at most
``PAIR_WAIT_S`` for its parts; after that the frame is handed on without them, and a missing
part then counts as absent. Frames are handed on in the order their status arrived.

Any node that judges headset input uses this one assembler, so that all of them see the same
frames. The message arguments are duck-typed, so this module imports no message package.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

from .frames import ARM_BUTTON, CALIBRATE_BUTTON, CLUTCH_BUTTON, SCALE_BUTTON, TRIGGER_AXIS
from .gate import Frame

PAIR_WAIT_S = 0.02          # how long a status waits for the pose and joy of its frame


def stamp_key(stamp) -> int:
    """Return a header stamp as integer nanoseconds (the pairing key)."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass(frozen=True)
class Paired:
    """One headset frame as the assembler hands it on."""

    arrival: float          # when its status arrived (the caller's monotonic clock)
    status: Any             # the InputStatus message
    key: int                # its stamp, as ``stamp_key``
    waited: float           # how long the status had waited when the frame was handed on


class FrameAssembler:
    """
    Pairs status, joy and poses by stamp; hands on complete frames, or late ones after a wait.

    Use: ``add_*`` from the message callbacks; then ``take(now)`` until it returns None. While
    a frame taken is being handled its parts can be read with ``parts``; ``done`` then drops
    every part stored for it or for an earlier stamp.
    """

    def __init__(self, wait_s: float = PAIR_WAIT_S) -> None:
        """Start empty; a status waits at most ``wait_s`` for its parts."""
        self.wait_s = wait_s
        self._poses: Dict[int, Any] = {}
        self._heads: Dict[int, Any] = {}
        self._joys: Dict[int, Any] = {}
        self._pending: Deque[Tuple[float, Any]] = deque()

    def add_pose(self, key: int, pose) -> None:
        """Store the right controller pose of the frame ``key``."""
        self._poses[key] = pose

    def add_head(self, key: int, pose) -> None:
        """Store the headset pose of the frame ``key``."""
        self._heads[key] = pose

    def add_joy(self, key: int, joy) -> None:
        """Store the Joy message of the frame ``key``."""
        self._joys[key] = joy

    def add_status(self, arrival: float, status) -> None:
        """Queue a status message that arrived at ``arrival``."""
        self._pending.append((arrival, status))

    def take(self, now: float) -> Optional[Paired]:
        """
        Return the oldest queued frame if it is complete or has waited long enough, else None.

        ``now`` is the same clock as the arrivals; a caller handing on several frames in one
        go passes the same ``now`` to each call.
        """
        if not self._pending:
            return None
        arrival, status = self._pending[0]
        key = stamp_key(status.header.stamp)
        complete = (key in self._joys
                    and (not status.right.pose_present or key in self._poses)
                    and (not status.viewer_present or key in self._heads))
        if not complete and now - arrival < self.wait_s:
            return None
        self._pending.popleft()
        return Paired(arrival, status, key, now - arrival)

    def has_pose(self, key: int) -> bool:
        """Return True when the right controller pose of the frame ``key`` arrived."""
        return key in self._poses

    def has_joy(self, key: int) -> bool:
        """Return True when the Joy message of the frame ``key`` arrived."""
        return key in self._joys

    def parts(self, key: int) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Return ``(joy, controller pose, headset pose)`` of the frame ``key`` (None: missing)."""
        return self._joys.get(key), self._poses.get(key), self._heads.get(key)

    def done(self, key: int) -> None:
        """Drop the parts stored for the frame ``key`` and for every earlier stamp."""
        for store in (self._poses, self._heads, self._joys):
            for old in [k for k in store if k <= key]:
                del store[old]


def gate_frame(status, joy, controller, head, arrival: float) -> Frame:
    """
    Build the gate frame of one status message from the parts with its stamp.

    ``controller`` and ``head`` are gate poses (None when missing). A pose or joy message that
    did not arrive with the same stamp is treated as absent: without the buttons the grip is
    unknown, so the controller counts as not present.
    """
    buttons = list(joy.buttons) if joy is not None else []
    have_buttons = len(buttons) > max(CLUTCH_BUTTON, CALIBRATE_BUTTON, ARM_BUTTON,
                                      SCALE_BUTTON)
    controller = controller if have_buttons else None
    trigger = None
    if joy is not None and len(joy.axes) > TRIGGER_AXIS:
        trigger = float(joy.axes[TRIGGER_AXIS])
    return Frame(
        arrival=arrival,
        xr_time=float(status.xr_time_ms) / 1000.0,
        session_id=int(status.session_id),
        seq=int(status.seq),
        reset_count=int(status.reset_count),
        controller=controller,
        controller_present=bool(status.right.pose_present) and controller is not None,
        controller_emulated=bool(status.right.emulated_position),
        head=head,
        head_present=bool(status.viewer_present) and head is not None,
        head_emulated=bool(status.viewer_emulated_position),
        grip=have_buttons and bool(buttons[CLUTCH_BUTTON]),
        calibrate=have_buttons and bool(buttons[CALIBRATE_BUTTON]),
        arm=have_buttons and bool(buttons[ARM_BUTTON]),
        scale_toggle=have_buttons and bool(buttons[SCALE_BUTTON]),
        trigger=trigger,
    )
