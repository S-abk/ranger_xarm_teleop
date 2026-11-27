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
"""MoveIt for a driven base: the arguments and semantic description chosen; the refusals."""

import ast
import os

import pytest
from ranger_xarm_teleop_bringup import drive_base as db

MODULE = os.path.join(os.path.dirname(db.__file__), 'drive_base.py')


def test_with_the_base_off_moveit_gets_no_argument_at_all():
    assert db.moveit_arguments(False) == {}


def test_with_the_base_driven_moveit_gets_drive_base_only():
    assert db.moveit_arguments(True) == {'drive_base': 'true'}


def test_never_a_planning_frame_or_an_obstacle_map():
    for base in (False, True):
        assert not {'planning_frame', 'obstacle_map'} & set(db.moveit_arguments(base))


def test_servo_loads_the_shipped_file_unless_the_base_is_driven():
    assert db.semantic_file(False) == 'ranger_xarm.srdf'
    assert db.semantic_file(True) == 'ranger_xarm_mobile_base_wheels.srdf'


def test_command_line_moveit_arguments_are_refused_saying_why():
    text = db.refused_arguments(['base', 'planning_frame', 'record'], db.NOT_OFFERED)
    assert text.startswith('refused: planning_frame:= is not an argument here')
    assert "workspace's default" in text
    both = db.refused_arguments(['obstacle_map', 'drive_base'], db.NOT_OFFERED)
    assert 'drive_base:=' in both and 'obstacle_map:=' in both


def test_no_refusal_without_those_arguments():
    assert db.refused_arguments(['base', 'reset', 'start_moveit'], db.NOT_OFFERED) is None
    assert db.refused_arguments([], db.NOT_OFFERED) is None


def test_the_mock_refuses_drive_base_only():
    assert db.refused_arguments(['drive_base'], db.MOCK_NOT_OFFERED).startswith('refused: drive')
    assert db.refused_arguments(['planning_frame'], db.MOCK_NOT_OFFERED) is None


def test_the_truth_words_are_the_launch_files():
    assert db.TRUE_WORDS == ('true', '1', 'yes')


def test_drive_base_off_is_accepted_everywhere():
    for base in (False, True):
        for simulation in (False, True):
            db.check_drive_base(False, base, simulation)


def test_drive_base_with_the_base_in_simulation_is_accepted():
    db.check_drive_base(True, True, True)


def test_drive_base_outside_simulation_is_refused_saying_why():
    for base in (False, True):
        with pytest.raises(ValueError, match='refused outside simulation'):
            db.check_drive_base(True, base, False)


def test_drive_base_without_the_base_is_refused_saying_why():
    with pytest.raises(ValueError, match='needs base:=true'):
        db.check_drive_base(True, False, True)


def test_no_ros_imports():
    tree = ast.parse(open(MODULE, encoding='utf-8').read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split('.')[0])
    assert names <= {'typing'}, names
