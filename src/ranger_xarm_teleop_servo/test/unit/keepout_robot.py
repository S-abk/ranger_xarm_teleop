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
A made-up robot for the keep-out tests: the six-joint chain of ``arm_urdf`` with collision
shapes, on a mobile base with a pedestal, a screen box behind the arm, a sensor mast, a
mesh sensor, a two-finger gripper (one driven joint, one mimic) and a wrist camera.

Nothing here is the real robot's geometry; it only has the same kinds of parts.
"""

import struct

import numpy as np

_ARM = [
    # name, parent, child, xyz, rpy, lower, upper
    ('xarm_joint1', 'xarm_link_base', 'l1', '0 0 0.267', '0 0 0', -6.283185307179586,
     6.283185307179586),
    ('xarm_joint2', 'l1', 'l2', '0 0 0', '-1.5708 0 0', -2.059, 2.0944),
    ('xarm_joint3', 'l2', 'l3', '0.0535 -0.2845 0', '0 0 0', -3.927, 0.19198),
    ('xarm_joint4', 'l3', 'l4', '0.0775 0.3425 0', '-1.5708 0 0', -6.283185307179586,
     6.283185307179586),
    ('xarm_joint5', 'l4', 'l5', '0 0 0', '1.5708 0 0', -1.69297, 3.141592653589793),
    ('xarm_joint6', 'l5', 'l6', '0.076 0.097 0', '-1.5708 0 0', -6.283185307179586,
     6.283185307179586),
]
READY = [0.0, -0.6, -0.7, 0.0, 1.3, 0.0]
JOINTS = [j[0] for j in _ARM]

# Collision shapes of the arm links, in each link's frame (rough cylinders and boxes).
_ARM_SHAPES = {
    'xarm_link_base': '<origin xyz="0 0 0.06"/><geometry><cylinder radius="0.07" '
                      'length="0.12"/></geometry>',
    'l1': '<origin xyz="0 0 -0.04"/><geometry><cylinder radius="0.06" length="0.12"/>'
          '</geometry>',
    'l2': '<origin xyz="0.027 -0.14 0" rpy="1.5708 0 0"/><geometry><cylinder radius="0.05" '
          'length="0.32"/></geometry>',
    'l3': '<origin xyz="0.05 0.03 0"/><geometry><box size="0.10 0.12 0.10"/></geometry>',
    'l4': '<origin xyz="0 0 -0.15"/><geometry><cylinder radius="0.045" length="0.25"/>'
          '</geometry>',
    'l5': '<origin xyz="0.04 0.05 0"/><geometry><box size="0.08 0.10 0.08"/></geometry>',
    'l6': '<origin xyz="0 0 0.01"/><geometry><cylinder radius="0.04" length="0.03"/>'
          '</geometry>',
}


def box_stl(half, binary=True) -> bytes:
    """Return an STL of a box with half extents ``half`` centred on the origin."""
    h = np.asarray(half, dtype=float)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    if not binary:
        lines = ['solid box']
        for f in faces:
            lines.append(' facet normal 0 0 0\n  outer loop')
            lines += [f'   vertex {x} {y} {z}' for x, y, z in corners[list(f)]]
            lines.append('  endloop\n endfacet')
        lines.append('endsolid box')
        return '\n'.join(lines).encode()
    out = bytearray(b'\0' * 80) + struct.pack('<I', len(faces))
    for f in faces:
        out += struct.pack('<3f', 0, 0, 0)
        for v in corners[list(f)]:
            out += struct.pack('<3f', *v)
        out += b'\0\0'
    return bytes(out)


MESHES = {
    'package://test_robot/meshes/sensor.stl': box_stl((0.03, 0.04, 0.02)),
    'package://test_robot/meshes/finger.stl': box_stl((0.008, 0.01, 0.025), binary=False),
}


def resolve(name: str) -> bytes:
    """Return the bytes of one of the made-up meshes."""
    return MESHES[name]


def robot_urdf(screen_x=-0.30, screen_height=0.27, extra_tool=False, drop_screen=False,
               gripper=True) -> str:
    """Return the URDF text of the test robot (keyword arguments vary its geometry)."""
    parts = ['<robot name="test_robot">', '<link name="base_footprint"/>',
             '<link name="base_link"><collision><origin xyz="0 0 -0.18"/><geometry>'
             '<box size="0.8 0.5 0.36"/></geometry></collision></link>',
             '<joint name="footprint" type="fixed"><parent link="base_footprint"/>'
             '<child link="base_link"/><origin xyz="0 0 0.33"/></joint>',
             # Pedestal under the arm.
             '<link name="pedestal"><collision><origin xyz="0 0 0.10"/><geometry>'
             '<box size="0.29 0.29 0.20"/></geometry></collision></link>',
             '<joint name="base_to_pedestal" type="fixed"><parent link="base_link"/>'
             '<child link="pedestal"/><origin xyz="0 0 0"/></joint>',
             '<link name="xarm_mount"/>',
             '<joint name="base_to_mount" type="fixed"><parent link="base_link"/>'
             '<child link="xarm_mount"/><origin xyz="0 0 0.20"/></joint>',
             '<joint name="mount_to_arm" type="fixed"><parent link="xarm_mount"/>'
             '<child link="xarm_link_base"/></joint>',
             # A mast with a mesh sensor on top, behind the arm.
             '<link name="mast"><collision><geometry><cylinder radius="0.02" length="1.0"/>'
             '</geometry></collision></link>',
             '<joint name="base_to_mast" type="fixed"><parent link="base_link"/>'
             '<child link="mast"/><origin xyz="-0.38 0.2 0.5"/></joint>',
             '<link name="sensor"><collision><geometry><mesh '
             'filename="package://test_robot/meshes/sensor.stl"/></geometry></collision>'
             '</link>',
             '<joint name="mast_to_sensor" type="fixed"><parent link="mast"/>'
             '<child link="sensor"/><origin xyz="0 0 0.52"/></joint>',
             # A wheel on a continuous joint: not fixed, not moving with the arm.
             '<link name="wheel"><collision><geometry><cylinder radius="0.1" length="0.05"/>'
             '</geometry></collision></link>',
             '<joint name="wheel_joint" type="continuous"><parent link="base_link"/>'
             '<child link="wheel"/><origin xyz="0.3 0.3 -0.25"/><axis xyz="0 1 0"/></joint>']
    if not drop_screen:
        parts += ['<link name="screen"><collision><origin xyz="0 0 0.011"/><geometry>'
                  '<box size="0.28 0.39 0.052"/></geometry></collision><collision>'
                  f'<origin xyz="-0.125 0 {screen_height / 2 + 0.007}"/><geometry>'
                  f'<box size="0.042 0.39 {screen_height}"/></geometry></collision></link>',
                  '<joint name="base_to_screen" type="fixed"><parent link="base_link"/>'
                  f'<child link="screen"/><origin xyz="{screen_x} 0 0.198" rpy="0 0 3.14159"/>'
                  '</joint>']
    for name, shape in _ARM_SHAPES.items():
        parts.append(f'<link name="{name}"><collision>{shape}</collision></link>')
    for name, parent, child, xyz, rpy, lower, upper in _ARM:
        parts.append(
            f'<joint name="{name}" type="revolute"><parent link="{parent}"/>'
            f'<child link="{child}"/><origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="0 0 1"/>'
            f'<limit lower="{lower}" upper="{upper}" velocity="3.14" effort="20"/></joint>')
    parts.append('<link name="xarm_link_tcp"/>')
    parts.append('<joint name="tool" type="fixed"><parent link="l6"/>'
                 '<child link="xarm_link_tcp"/><origin xyz="0 0 0.172"/></joint>')
    parts.append('<link name="camera"><collision><origin xyz="0 0.06 0.01"/><geometry>'
                 '<box size="0.09 0.03 0.03"/></geometry></collision></link>')
    parts.append('<joint name="l6_to_camera" type="fixed"><parent link="l6"/>'
                 '<child link="camera"/></joint>')
    if gripper:
        parts.append('<link name="gripper_base"><collision><origin xyz="0 0 0.05"/>'
                     '<geometry><box size="0.06 0.12 0.08"/></geometry></collision></link>')
        parts.append('<joint name="l6_to_gripper" type="fixed"><parent link="l6"/>'
                     '<child link="gripper_base"/></joint>')
        for side, sign in (('left', 1), ('right', -1)):
            mimic = '' if side == 'left' else \
                '<mimic joint="finger_left_joint" multiplier="1" offset="0"/>'
            parts.append(f'<link name="finger_{side}"><collision><origin xyz="0 0 0.02"/>'
                         '<geometry><mesh filename="package://test_robot/meshes/finger.stl"/>'
                         '</geometry></collision></link>')
            parts.append(f'<joint name="finger_{side}_joint" type="prismatic">'
                         f'<parent link="gripper_base"/><child link="finger_{side}"/>'
                         f'<origin xyz="0 {0.02 * sign} 0.10"/><axis xyz="0 {sign} 0"/>'
                         f'<limit lower="0" upper="0.04" velocity="0.1" effort="10"/>'
                         f'{mimic}</joint>')
    if extra_tool:
        parts.append('<link name="probe"><collision><origin xyz="0 0 0.25"/><geometry>'
                     '<cylinder radius="0.01" length="0.2"/></geometry></collision></link>')
        parts.append('<joint name="l6_to_probe" type="fixed"><parent link="l6"/>'
                     '<child link="probe"/></joint>')
    parts.append('</robot>')
    return ''.join(parts)
