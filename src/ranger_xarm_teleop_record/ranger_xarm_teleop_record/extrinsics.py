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
Camera extrinsics from the robot description and the static transforms (no ROS).

A camera's optical frame is joined to the robot by fixed joints of the description and by
static transforms (the optical frames are often published separately). Its extrinsics are
recorded as the pose of the optical frame in the link it rides on (the first link above it
that a movable joint moves, or the root): constant for the episode. When the base or the
tool frame lies on the same rigid body, the pose in that frame is recorded too; otherwise
it depends on the joints and is given by the recorded joint states and the description.
"""

import math
from typing import Dict, Iterable, Optional, Tuple
import xml.etree.ElementTree as ET

from ranger_xarm_teleop_servo.rotations import Pose, q_from_matrix
from ranger_xarm_teleop_servo.static_frames import StaticGraph

FIXED = 'fixed'


def _floats(text, default):
    return [float(v) for v in text.split()] if text else list(default)


def rpy_pose(xyz, rpy) -> Pose:
    """Return the pose of a URDF origin (fixed-axis roll, pitch, yaw)."""
    roll, pitch, yaw = rpy
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    m = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]]
    return Pose(tuple(float(v) for v in xyz), q_from_matrix(m))


def parse_urdf(urdf: str) -> Dict[str, Tuple[str, str, Pose]]:
    """Return child link -> (parent link, joint type, child pose in parent)."""
    root = ET.fromstring(urdf)
    out = {}
    for joint in root.iter('joint'):
        parent, child = joint.find('parent'), joint.find('child')
        if parent is None or child is None:
            continue
        origin = joint.find('origin')
        xyz = _floats(origin.get('xyz') if origin is not None else None, (0, 0, 0))
        rpy = _floats(origin.get('rpy') if origin is not None else None, (0, 0, 0))
        out[child.get('link')] = (parent.get('link'), joint.get('type', ''), rpy_pose(xyz, rpy))
    return out


def pose_dict(pose: Optional[Pose]) -> Optional[dict]:
    """Return a pose as {'xyz': [...], 'xyzw': [...]} (JSON), or None."""
    if pose is None:
        return None
    return {'xyz': [float(v) for v in pose.p], 'xyzw': [float(v) for v in pose.q]}


class Frames:
    """The rigid part of the robot's frame tree: fixed joints plus static transforms."""

    def __init__(self, urdf: str, static: Iterable[Tuple[str, str, Pose]] = ()):
        """Build from a URDF string and (parent, child, child-in-parent) static transforms."""
        self.parents: Dict[str, Tuple[str, str]] = {}
        self.graph = StaticGraph()
        for child, (parent, kind, pose) in parse_urdf(urdf).items():
            self.parents[child] = (parent, kind)
            if kind == FIXED:
                self.graph.add(parent, child, pose)
        for parent, child, pose in static:
            if child not in self.parents:
                self.parents[child] = (parent, FIXED)
            self.graph.add(parent, child, pose)

    def mount(self, frame: str) -> str:
        """Return the link ``frame`` rides on: climb fixed joints until a movable one."""
        seen = set()
        while frame in self.parents and frame not in seen:
            seen.add(frame)
            parent, kind = self.parents[frame]
            if kind != FIXED:
                return frame
            frame = parent
        return frame

    def extrinsics(self, frame: str, base: str, tool: str) -> dict:
        """Return the extrinsics record of one optical frame (JSON)."""
        known = frame in self.parents or frame in self.graph.frames()
        mount = self.mount(frame) if known else None
        return {
            'frame': frame,
            'known': known,
            'mount_link': mount,
            'in_mount': pose_dict(self.graph.transform(mount, frame)) if mount else None,
            'in_base': pose_dict(self.graph.transform(base, frame)) if known else None,
            'in_tool': pose_dict(self.graph.transform(tool, frame)) if known else None,
            'base_frame': base,
            'tool_frame': tool,
            'source': 'robot description fixed joints + /tf_static',
        }
