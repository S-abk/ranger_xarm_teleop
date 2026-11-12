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
"""Static-frame graph (tool to tip conversion) and the simulation-only check (INV-10)."""

import math

import pytest
from ranger_xarm_teleop_servo.rotations import (compose, inverse, Pose, q_between,
                                                q_from_axis_angle, q_from_yaw, v_norm, v_sub)
from ranger_xarm_teleop_servo.simcheck import (hardware_plugins, NotASimulator,
                                               require_simulator, SIMULATOR_PLUGINS)
from ranger_xarm_teleop_servo.static_frames import StaticGraph

A = Pose((0.0, 0.0, 0.1), q_from_yaw(0.5))
B = Pose((0.05, 0.0, 0.0), q_from_axis_angle((0.0, 1.0, 0.0), 0.3))


def same(a, b):
    return v_norm(v_sub(a.p, b.p)) < 1e-12 and q_between(a.q, b.q) < 1e-7


def test_static_chain_in_both_directions():
    graph = StaticGraph()
    graph.add('link6', 'eef', A)
    graph.add('eef', 'tcp', B)
    graph.add('tcp', 'tool', Pose((0.0, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)))
    assert same(graph.transform('link6', 'tcp'), compose(A, B))
    assert same(graph.transform('tcp', 'link6'), inverse(compose(A, B)))
    assert same(graph.transform('tool', 'tcp'), Pose((0.0, 0.0, -0.2), (0.0, 0.0, 0.0, 1.0)))
    assert graph.transform('tcp', 'tcp') == Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert graph.frames() == ('eef', 'link6', 'tcp', 'tool')


def test_frames_not_joined_by_static_transforms_have_no_transform():
    graph = StaticGraph()
    graph.add('base', 'pedestal', A)            # static
    graph.add('eef', 'tcp', B)                  # static, but link5 -> link6 is a joint
    assert graph.transform('base', 'tcp') is None
    assert graph.transform('tcp', 'unknown') is None
    assert graph.transform('unknown', 'tcp') is None


def test_newer_static_transform_replaces_the_older():
    graph = StaticGraph()
    graph.add('tcp', 'tool', A)
    graph.add('tcp', 'tool', B)
    assert same(graph.transform('tcp', 'tool'), B)
    assert same(graph.transform('tool', 'tcp'), inverse(B))


def urdf(*plugins):
    blocks = ''.join(
        f'<ros2_control name="c{i}" type="system"><hardware>'
        + (f'<plugin>{plugin}</plugin>' if plugin is not None else '')
        + '</hardware></ros2_control>' for i, plugin in enumerate(plugins))
    return f'<robot name="r"><link name="base"/>{blocks}</robot>'


@pytest.mark.parametrize('plugin', SIMULATOR_PLUGINS)
def test_simulator_plugins_are_accepted(plugin):
    assert require_simulator(urdf(plugin)) == [plugin]
    assert require_simulator(urdf(plugin, 'mock_components/GenericSystem'))


@pytest.mark.parametrize('description', [
    urdf('uf_robot_hardware/UFRobotSystemHardware'),
    urdf('mock_components/GenericSystem', 'example_vendor/RealArmHardware'),
    urdf(None),
    urdf(),
    '<robot name="r"/>',
    'not xml',
    '',
])
def test_anything_else_refuses_simulation_only_values(description):
    with pytest.raises(NotASimulator):
        require_simulator(description)


def test_hardware_plugins_lists_every_block():
    assert hardware_plugins(urdf('a/B', ' c/D ')) == ['a/B', 'c/D']
    assert math.isfinite(len(hardware_plugins(urdf())))
