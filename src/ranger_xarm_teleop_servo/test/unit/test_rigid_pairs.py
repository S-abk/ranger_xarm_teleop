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
"""Self-collision pairs no arm motion can change (gripper_fix INV-2; pure)."""

import itertools

import pytest
from ranger_xarm_teleop_servo.rigid_pairs import (DescriptionError, disabled_pairs, group_joints,
                                                  rigid_pairs, srdf_lines)

from keepout_robot import robot_urdf  # noqa: I100

ARM = [f'xarm_joint{i}' for i in range(1, 7)]
WRIST = ['camera', 'finger_left', 'finger_right', 'gripper_base', 'l6']


def srdf(disabled=(('l5', 'l6'), ('gripper_base', 'l6')), group=None):
    group = group or '<chain base_link="xarm_link_base" tip_link="xarm_link_tcp"/>'
    lines = ''.join(f'<disable_collisions link1="{a}" link2="{b}" reason="Adjacent"/>'
                    for a, b in disabled)
    return (f'<robot name="test_robot"><group name="arm">{group}</group>'
            f'<group name="hand"><joint name="finger_left_joint"/></group>{lines}</robot>')


def pairs_among(links, without=()):
    return sorted(p for p in itertools.combinations(sorted(links), 2)
                  if frozenset(p) not in {frozenset(w) for w in without})


def test_the_arm_group_is_the_chain_without_its_fixed_joints():
    assert group_joints(robot_urdf(), srdf(), 'arm') == set(ARM)


def test_the_pairs_are_the_wrist_links_left_enabled():
    # Everything mounted on l6 (camera, gripper base, both fingers) and l6 itself: no arm
    # joint between any two of them. The fingers move only with their own joint.
    assert rigid_pairs(robot_urdf(), srdf(), 'arm') == pairs_among(
        WRIST, without=[('gripper_base', 'l6')])


def test_pairs_across_an_arm_joint_are_never_derived():
    pairs = {frozenset(p) for p in rigid_pairs(robot_urdf(), srdf(), 'arm')}
    for a, b in pairs:
        assert a in WRIST and b in WRIST
    # The wrist links against l5 (joint 6 between), the structure, or the other links.
    assert not any('l5' in p or 'l4' in p or 'mast' in p or 'base_link' in p for p in pairs)


def test_links_the_arm_does_not_move_are_left_out():
    # base_link, pedestal, mast, sensor, wheel: no arm joint between them either, but
    # MoveIt Servo checks only pairs with a link the arm moves.
    pairs = rigid_pairs(robot_urdf(), srdf(), 'arm')
    static = {'base_link', 'pedestal', 'mast', 'sensor', 'wheel', 'screen', 'xarm_link_base'}
    assert not any(a in static or b in static for a, b in pairs)


def test_a_gripper_mounted_one_link_earlier_changes_the_pairs():
    # Modified description: the gripper hangs from l5. Joint 6 now lies between the gripper
    # and l6 / the camera, so those pairs are checked again; the gripper pairs with l5 are
    # the rigid ones.
    urdf = robot_urdf().replace('<joint name="l6_to_gripper" type="fixed"><parent link="l6"/>',
                                '<joint name="l6_to_gripper" type="fixed"><parent link="l5"/>')
    pairs = rigid_pairs(urdf, srdf(), 'arm')
    assert pairs == sorted(
        pairs_among(['finger_left', 'finger_right', 'gripper_base', 'l5'], without=[])
        + [('camera', 'l6')])


def test_a_finger_on_an_arm_joint_is_not_rigid_against_the_gripper():
    # Modified group: the finger joint is part of the servoed group, so the finger moves
    # with "the arm" and its pairs with the wrist are checked.
    group = ('<chain base_link="xarm_link_base" tip_link="xarm_link_tcp"/>'
             '<joint name="finger_left_joint"/>')
    pairs = rigid_pairs(robot_urdf(), srdf(group=group), 'arm')
    # The right finger mimics the left one: it moves with the group as well.
    assert group_joints(robot_urdf(), srdf(group=group), 'arm') == set(
        ARM + ['finger_left_joint', 'finger_right_joint'])
    assert pairs == pairs_among(['camera', 'gripper_base', 'l6'],
                                without=[('gripper_base', 'l6')])


def test_pairs_the_srdf_disables_are_not_listed_again():
    every = rigid_pairs(robot_urdf(), srdf(disabled=()), 'arm')
    assert every == pairs_among(WRIST)
    assert rigid_pairs(robot_urdf(), srdf(disabled=every), 'arm') == []
    assert disabled_pairs(srdf()) == {frozenset(('l5', 'l6')), frozenset(('gripper_base', 'l6'))}


def test_without_a_gripper_only_the_camera_remains():
    assert rigid_pairs(robot_urdf(gripper=False), srdf(disabled=()), 'arm') == [('camera', 'l6')]


def test_srdf_lines():
    assert srdf_lines([('a', 'b')]) == ['<disable_collisions link1="a" link2="b" reason="Rigid"/>']


@pytest.mark.parametrize('urdf, semantic, group, message', [
    (robot_urdf(), srdf(), 'nope', 'no group'),
    (robot_urdf(), srdf(group='<link name="l6"/>'), 'arm', 'no <chain> or <joint>'),
    (robot_urdf(), srdf(group='<chain base_link="l6" tip_link="l1"/>'), 'arm', 'not below'),
    (robot_urdf(), srdf(group='<joint name="ghost"/>'), 'arm', 'does not have'),
    (robot_urdf(), srdf(group='<joint name="tool"/>'), 'arm', 'no movable joint'),
    ('<robot', srdf(), 'arm', 'not valid XML'),
    (robot_urdf(), '<robot', 'arm', 'not valid XML'),
])
def test_descriptions_it_cannot_use_are_refused(urdf, semantic, group, message):
    with pytest.raises(DescriptionError, match=message):
        rigid_pairs(urdf, semantic, group)
