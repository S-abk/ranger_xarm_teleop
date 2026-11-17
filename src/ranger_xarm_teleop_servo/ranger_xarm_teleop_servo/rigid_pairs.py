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
Self-collision pairs that arm motion cannot change. No ROS imports.

MoveIt Servo slows every joint while any pair its allowed collision matrix leaves enabled
is within ``self_collision_proximity_threshold``, whatever the direction of motion. A pair
whose two links have no joint of the servoed arm between them keeps its distance however
the arm moves: two links of a gripper, or a finger and the wrist link the gripper is
mounted on. When such a pair is near (a closed gripper), Servo slows the arm for good
although no arm motion can bring the two links any closer.

``rigid_pairs`` derives those pairs from the robot description alone (no link names):
the links of every pair have collision geometry, are moved by the arm (Servo checks only
pairs with a link the group moves), have no movable joint of the group on the tree path
between them, and are left enabled by the SRDF. ``srdf_lines`` writes them as
``<disable_collisions>`` lines for the SRDF.

What disabling them gives up: only collisions that the arm cannot cause or avoid. Motion
of other joints between the two links (the gripper's own) can still change their
distance; whether that motion can bring them into contact is a property of the gripper,
not of the arm, and is not checked here.
"""

from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from xml.etree import ElementTree

Pair = Tuple[str, str]


class DescriptionError(ValueError):
    """The URDF or SRDF cannot be used to derive the pairs."""


def _parse(text: str, what: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(text)
    except ElementTree.ParseError as err:
        raise DescriptionError(f'the {what} is not valid XML: {err}') from err


def _tree(urdf: ElementTree.Element) -> Tuple[Dict[str, Tuple[str, str, str]], Dict[str, str]]:
    """Return child link -> (parent link, joint, type) and joint -> the joint it mimics."""
    parent_of, mimics = {}, {}
    for joint in urdf.findall('joint'):
        name, kind = joint.get('name'), joint.get('type')
        parent, child = joint.find('parent'), joint.find('child')
        if name is None or kind is None or parent is None or child is None:
            raise DescriptionError(f'joint {name!r} has no name, type, parent or child')
        if child.get('link') in parent_of:
            raise DescriptionError(f"link {child.get('link')!r} has two parent joints")
        parent_of[child.get('link')] = (parent.get('link'), name, kind)
        mimic = joint.find('mimic')
        if mimic is not None and mimic.get('joint'):
            mimics[name] = mimic.get('joint')
    return parent_of, mimics


def _chain_joints(parent_of: Dict[str, Tuple[str, str, str]], base: str, tip: str) -> List[str]:
    joints, link = [], tip
    while link != base:
        if link not in parent_of:
            raise DescriptionError(f'tip link {tip!r} is not below base link {base!r}')
        link, joint, _ = parent_of[link]
        joints.append(joint)
    return joints


def group_joints(urdf_text: str, srdf_text: str, group: str) -> Set[str]:
    """
    Return the movable joints of SRDF group ``group``, and the joints that mimic them.

    The group is read from its ``<chain>`` and ``<joint>`` elements. A group with any other
    element (``<link>``, ``<group>``) is refused (``DescriptionError``): a joint given that
    way would be missed, and pairs across it listed.
    """
    urdf, srdf = _parse(urdf_text, 'URDF'), _parse(srdf_text, 'SRDF')
    parent_of, mimics = _tree(urdf)
    kinds = {joint: kind for _, joint, kind in parent_of.values()}
    element = next((g for g in srdf.findall('group') if g.get('name') == group), None)
    if element is None:
        raise DescriptionError(f'the SRDF has no group {group!r}')
    others = [child.tag for child in element if child.tag not in ('chain', 'joint')]
    if others:
        raise DescriptionError(f'group {group!r} has {sorted(set(others))} elements: only '
                               '<chain> and <joint> groups are derived (a joint given another '
                               'way would be missed)')
    names: List[str] = []
    for chain in element.findall('chain'):
        names += _chain_joints(parent_of, chain.get('base_link'), chain.get('tip_link'))
    for joint in element.findall('joint'):
        if joint.get('name') not in kinds:
            raise DescriptionError(f"group {group!r} names joint {joint.get('name')!r}, "
                                   'which the URDF does not have')
        names.append(joint.get('name'))
    if not names:
        raise DescriptionError(f'group {group!r} has no <chain> or <joint>: its joints are '
                               'not derived')
    movable = {n for n in names if kinds.get(n) != 'fixed'}
    if not movable:
        raise DescriptionError(f'group {group!r} has no movable joint')
    # A joint that mimics an arm joint moves with the arm.
    changed = True
    while changed:
        extra = {j for j, m in mimics.items() if m in movable and j not in movable}
        movable |= extra
        changed = bool(extra)
    return movable


def disabled_pairs(srdf_text: str) -> Set[FrozenSet[str]]:
    """Return the pairs the SRDF disables (``<disable_collisions>``)."""
    srdf = _parse(srdf_text, 'SRDF')
    return {frozenset((d.get('link1'), d.get('link2')))
            for d in srdf.findall('disable_collisions')}


def _anchor(link: str, parent_of: Dict[str, Tuple[str, str, str]],
            arm: Set[str]) -> Optional[str]:
    """Return the nearest arm joint above ``link`` (None: the arm does not move it)."""
    seen = set()
    while link in parent_of:
        if link in seen:
            raise DescriptionError(f'the URDF tree has a loop at link {link!r}')
        seen.add(link)
        parent, joint, _ = parent_of[link]
        if joint in arm:
            return joint
        link = parent
    return None


def rigid_pairs(urdf_text: str, srdf_text: str, group: str) -> List[Pair]:
    """
    Return the enabled pairs whose distance no motion of ``group`` can change, sorted.

    Two links have no joint of the group on the tree path between them exactly when the
    nearest group joint above each is the same joint. Pairs not moved by the group at all
    are left out (MoveIt Servo does not check them).
    """
    urdf = _parse(urdf_text, 'URDF')
    parent_of, _ = _tree(urdf)
    arm = group_joints(urdf_text, srdf_text, group)
    disabled = disabled_pairs(srdf_text)
    with_geometry = sorted(link.get('name') for link in urdf.findall('link')
                           if link.find('collision') is not None)
    anchors = {link: _anchor(link, parent_of, arm) for link in with_geometry}
    pairs = []
    for i, a in enumerate(with_geometry):
        for b in with_geometry[i + 1:]:
            if (anchors[a] is not None and anchors[a] == anchors[b]
                    and frozenset((a, b)) not in disabled):
                pairs.append((a, b))
    return pairs


def srdf_lines(pairs: List[Pair]) -> List[str]:
    """Return ``<disable_collisions>`` lines for ``pairs``."""
    return [f'<disable_collisions link1="{a}" link2="{b}" reason="Rigid"/>' for a, b in pairs]
