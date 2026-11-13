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
"""A six-joint arm (the xArm 6 chain with a 172 mm tool) as URDF text, for the unit tests."""

_JOINTS = [
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


def arm_urdf(tool='0 0 0.172'):
    """Return the URDF text; ``tool`` is the fixed offset from the last link to the tip."""
    parts = ['<robot name="arm">']
    for name, parent, child, xyz, rpy, lower, upper in _JOINTS:
        parts.append(
            f'<joint name="{name}" type="revolute"><parent link="{parent}"/>'
            f'<child link="{child}"/><origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="0 0 1"/>'
            f'<limit lower="{lower}" upper="{upper}" velocity="3.14" effort="20"/></joint>')
    parts.append('<joint name="tool" type="fixed"><parent link="l6"/>'
                 f'<child link="xarm_link_tcp"/><origin xyz="{tool}" rpy="0 0 0"/></joint>')
    parts.append('</robot>')
    return ''.join(parts)
