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
Fixed transforms between frames, from what was published on ``/tf_static``. No ROS imports.

The teleop node converts the tool-frame target to the pose of the servo group's tip
link. That is only sound when the two frames are rigidly connected, so the transform is
taken from a graph that holds static transforms and nothing else: if no path of static
edges joins the frames, there is no answer.
"""

from collections import deque
from typing import Dict, Optional, Tuple

from .rotations import compose, IDENTITY_POSE, inverse, Pose


class StaticGraph:
    """Static transforms as an undirected graph of frames."""

    def __init__(self) -> None:
        self._edges: Dict[str, Dict[str, Pose]] = {}

    def add(self, parent: str, child: str, child_in_parent: Pose) -> None:
        """Record the pose of ``child`` in ``parent`` (a newer value replaces an older)."""
        self._edges.setdefault(parent, {})[child] = child_in_parent
        self._edges.setdefault(child, {})[parent] = inverse(child_in_parent)

    def transform(self, source: str, target: str) -> Optional[Pose]:
        """
        Return the pose of ``target`` in ``source``, or ``None``.

        ``None`` means the frames are not joined by static transforms only.
        """
        if source == target:
            return IDENTITY_POSE
        if source not in self._edges or target not in self._edges:
            return None
        seen = {source}
        queue: deque = deque([(source, IDENTITY_POSE)])
        while queue:
            frame, pose = queue.popleft()
            for neighbour, step in self._edges[frame].items():
                if neighbour in seen:
                    continue
                reached = compose(pose, step)
                if neighbour == target:
                    return reached
                seen.add(neighbour)
                queue.append((neighbour, reached))
        return None

    def frames(self) -> Tuple[str, ...]:
        """Return the known frame names."""
        return tuple(sorted(self._edges))
