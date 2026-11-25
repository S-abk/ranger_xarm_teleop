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
A small wheeled robot description for the stability monitor's unit tests (generic names).

``base_footprint`` (root, on the floor) -> ``base_link`` 0.3 m up -> a 40 kg chassis; four
corners at (+-0.25, +-0.2) with a sprung vertical joint (20 kN/m, reference -0.01 m, damping
500 N s/m), a steering joint and a 3 kg sphere wheel of radius 0.1 whose contact is on the
floor (base_link z = -0.3); a two-joint arm on a mount at (0, 0, 0.4) with a mimic finger; an
IMU link at (-0.3, 0, 1.1).
"""

WHEELS = ('front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel')
CORNERS = (('front_left', 0.25, 0.2), ('front_right', 0.25, -0.2), ('rear_left', -0.25, 0.2),
           ('rear_right', -0.25, -0.2))
CHASSIS_KG, SUSP_KG, STEER_KG, WHEEL_KG = 40.0, 0.3, 1.0, 3.0
LINK1_KG, LINK2_KG, FINGER_KG = 2.0, 2.0, 0.1


def _inertial(mass, xyz='0 0 0', i=0.01):
    return (f'<inertial><origin xyz="{xyz}"/><mass value="{mass}"/>'
            f'<inertia ixx="{i}" ixy="0" ixz="0" iyy="{i}" iyz="0" izz="{i}"/></inertial>')


def make_urdf(steer_offset=0.0, springs=True, wheel_collision='sphere'):
    """Return the description text; ``steer_offset`` puts each wheel ahead of its steer axis."""
    parts = ['<robot name="test_robot">',
             '<link name="base_footprint"/>',
             '<link name="base_link"/>',
             '<joint name="footprint_joint" type="fixed"><parent link="base_footprint"/>'
             '<child link="base_link"/><origin xyz="0 0 0.3"/></joint>',
             f'<link name="chassis">{_inertial(CHASSIS_KG, "0 0 0.1", 1.0)}</link>',
             '<joint name="chassis_joint" type="fixed"><parent link="base_link"/>'
             '<child link="chassis"/></joint>']
    for name, x, y in CORNERS:
        parts.append(f'<link name="{name}_susp">{_inertial(SUSP_KG)}</link>')
        parts.append(f'<joint name="{name}_suspension_joint" type="prismatic">'
                     f'<parent link="base_link"/><child link="{name}_susp"/>'
                     f'<origin xyz="{x} {y} -0.2"/><axis xyz="0 0 1"/>'
                     '<limit lower="-0.015" upper="0.025" effort="5000" velocity="2"/>'
                     '<dynamics damping="500" friction="0"/></joint>')
        if springs:
            parts.append(f'<gazebo reference="{name}_suspension_joint">'
                         '<springReference>-0.01</springReference>'
                         '<springStiffness>20000.0</springStiffness></gazebo>')
        parts.append(f'<link name="{name}_steer">{_inertial(STEER_KG)}</link>')
        parts.append(f'<joint name="{name}_steer_joint" type="revolute">'
                     f'<parent link="{name}_susp"/><child link="{name}_steer"/>'
                     '<axis xyz="0 0 1"/>'
                     '<limit lower="-1.7" upper="1.7" effort="10" velocity="5"/></joint>')
        geom = ('<sphere radius="0.1"/>' if wheel_collision == 'sphere' else
                '<cylinder radius="0.1" length="0.07"/>' if wheel_collision == 'cylinder' else
                '<box size="0.2 0.07 0.2"/>')
        parts.append(f'<link name="{name}_wheel">{_inertial(WHEEL_KG)}'
                     f'<collision><geometry>{geom}</geometry></collision></link>')
        parts.append(f'<joint name="{name}_wheel_joint" type="continuous">'
                     f'<parent link="{name}_steer"/><child link="{name}_wheel"/>'
                     f'<origin xyz="{steer_offset} 0 0"/><axis xyz="0 1 0"/></joint>')
    parts += [
        '<link name="arm_mount"/>',
        '<joint name="arm_mount_joint" type="fixed"><parent link="base_link"/>'
        '<child link="arm_mount"/><origin xyz="0 0 0.4"/></joint>',
        f'<link name="arm_link1">{_inertial(LINK1_KG, "0 0 0.1")}</link>',
        '<joint name="arm_joint1" type="revolute"><parent link="arm_mount"/>'
        '<child link="arm_link1"/><axis xyz="0 0 1"/>'
        '<limit lower="-3" upper="3" effort="10" velocity="2"/></joint>',
        f'<link name="arm_link2">{_inertial(LINK2_KG, "0.2 0 0")}</link>',
        '<joint name="arm_joint2" type="revolute"><parent link="arm_link1"/>'
        '<child link="arm_link2"/><origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>'
        '<limit lower="-2" upper="2" effort="10" velocity="2"/></joint>',
        f'<link name="finger">{_inertial(FINGER_KG)}</link>',
        '<joint name="finger_joint" type="prismatic"><parent link="arm_link2"/>'
        '<child link="finger"/><origin xyz="0.4 0 0"/><axis xyz="0 1 0"/>'
        '<limit lower="0" upper="0.04" effort="10" velocity="1"/>'
        '<mimic joint="arm_joint2" multiplier="0.01" offset="0"/></joint>',
        '<link name="imu_link"/>',
        '<joint name="imu_joint" type="fixed"><parent link="base_link"/>'
        '<child link="imu_link"/><origin xyz="-0.3 0 1.1"/></joint>',
        '</robot>']
    return '\n'.join(parts)


TOTAL_KG = CHASSIS_KG + 4 * (SUSP_KG + STEER_KG + WHEEL_KG) + LINK1_KG + LINK2_KG + FINGER_KG

MASS_FILE = {
    'stability_mass_model': 1,
    'simulation_only': True,
    'default_uncertainty': {'mass_fraction': 0.1, 'com_m': 0.02},
    'bodies': {'chassis': {'mass': {'plus': 8.0, 'minus': 8.0}, 'com': {'bound': 0.05}}},
    'suspension': {'stiffness': {'plus': 2000.0, 'minus': 12000.0},
                   'damping': {'plus': 150.0, 'minus': 150.0}},
}
