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
Arrival stamps of one subscribed stream, judged for readiness. No ROS imports.

Moved unchanged from episode_node.py, so that the episode node and the session pre-flight judge
a stream's rate and silence by the same rule.
"""

from typing import Dict, List, Optional, Sequence

from .streams import FieldPresence, OPTIONAL_FIELDS

RATE_WINDOW_S = 2.0


class StreamWatch:
    """Arrival stamps of one subscribed stream: rate for readiness, counts for an episode."""

    def __init__(self, name: str, spec: dict):
        """Watch the profile stream ``name``."""
        self.name, self.spec = name, spec
        self.recent: List[float] = []          # header stamps, last RATE_WINDOW_S
        self.last_arrival: Optional[float] = None
        self.episode: Optional[List[float]] = None
        # Only joint states are fed field by field (on_joint_states); another kind gets no
        # presence, so a required field there cannot hold START refused forever.
        keys = OPTIONAL_FIELDS['joint_states'] if spec.get('kind') == 'joint_states' else ()
        self.presence = FieldPresence(keys) if keys else None       # since start-up
        self.episode_presence: Optional[FieldPresence] = None        # in the episode

    def fields(self, values: Dict[str, Sequence[float]]) -> None:
        """Note which optional fields of one message have a finite value."""
        if self.presence is not None:
            self.presence.add(values)
            if self.episode_presence is not None:
                self.episode_presence.add(values)

    def add(self, stamp: float, arrival: float) -> None:
        """Note one message."""
        self.last_arrival = arrival
        self.recent.append(stamp)
        cut = stamp - RATE_WINDOW_S
        while self.recent and (self.recent[0] < cut or self.recent[0] > stamp + 1.0):
            self.recent.pop(0)
        if self.episode is not None:
            self.episode.append(stamp)

    def problem(self, now: float) -> Optional[str]:
        """Return why this stream is not usable now, or None."""
        if self.last_arrival is None:
            return f'{self.name}: no message on {self.spec["topic"]}'
        if now - self.last_arrival > 1.0:
            return f'{self.name}: silent for {now - self.last_arrival:.1f} s'
        for key in self.spec.get('required_fields') or []:
            if self.presence is not None and not self.presence.last.get(key, False):
                return (f'{self.name}: required field {key} has no finite value in the '
                        'newest message')
        rate = self.spec.get('rate_hz')
        if rate is None or len(self.recent) < 2:
            return None if rate is None else f'{self.name}: too few messages to judge the rate'
        span = self.recent[-1] - self.recent[0]
        measured = (len(self.recent) - 1) / span if span > 0 else 0.0
        if measured < 0.9 * float(rate):
            return f'{self.name}: {measured:.1f} Hz, below its nominal {float(rate):g} Hz'
        return None
