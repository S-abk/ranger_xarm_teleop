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
What the launch files pass, evaluated without starting anything (``launch_eval.py``).

With the base off, every node this stack starts and every argument it passes to an included
launch file is as before MoveIt was set up for a driven base: the reference
(``fixtures/launch_reference.jsonl``) was written from the launch files of that time with
``python3 launch_eval.py --write fixtures/launch_reference.jsonl``. With the base driven in
simulation, move_group is asked for a driven base (``drive_base:=true`` and nothing else) and
MoveIt Servo loads the same semantic description. The robot workspace's own nodes are not
compared, only what this stack passes to its launch files.
"""

import copy
import json
import os

from launch_eval import BASE_CASES, CASES, evaluate, launch_path
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, 'fixtures', 'launch_reference.jsonl')
DRIVEN_SRDF = {'srdf_file': 'ranger_xarm_mobile_base_wheels.srdf'}
SHIPPED_SRDF = {'srdf_file': 'ranger_xarm.srdf'}


def reference():
    with open(REFERENCE, encoding='utf-8') as handle:
        return {line['case']: line for line in map(json.loads, filter(str.strip, handle))}


def semantic_of(nodes, executable):
    found = [p['robot_description_semantic'] for n in nodes if n.get('executable') == executable
             for p in n['parameters'] if 'robot_description_semantic' in p]
    assert len(found) == 1, (executable, found)
    return found[0]


def include(result, launch_file):
    found = [i for i in result['includes'] if i['file'].endswith('/launch/' + launch_file)]
    assert len(found) == 1, (launch_file, [i['file'] for i in result['includes']])
    return found[0]


def test_the_reference_holds_every_case():
    assert sorted(reference()) == sorted(case for case, _, _ in CASES + BASE_CASES)


@pytest.mark.parametrize('case,name,args', CASES, ids=[c[0] for c in CASES])
def test_with_the_base_off_or_mocked_everything_passed_is_as_before(case, name, args, tmp_path):
    expected = reference()[case]
    got = evaluate(launch_path(name), args, str(tmp_path))
    assert got['includes'] == expected['includes']
    assert len(got['nodes']) == len(expected['nodes'])
    for g, e in zip(got['nodes'], expected['nodes']):
        assert g == e, (e.get('executable'), e.get('package'))


def test_with_the_base_driven_move_group_gets_drive_base_and_nothing_else(tmp_path):
    got = evaluate(launch_path('teleop_sim.launch.py'), ['base:=true'], str(tmp_path))
    assert include(got, 'moveit.launch.py')['launch_arguments'] == [['drive_base', 'true']]
    core = dict(include(got, 'teleop_core.launch.py')['launch_arguments'])
    assert core['drive_base'] == 'true' and core['base'] == 'true'
    assert semantic_of(got['nodes'], 'servo_node') == DRIVEN_SRDF


@pytest.mark.parametrize('case,name,args', BASE_CASES, ids=[c[0] for c in BASE_CASES])
def test_with_the_base_driven_only_the_driven_base_changes_differ_from_before(
        case, name, args, tmp_path):
    """Undo exactly the three intended changes; the rest must equal the reference."""
    got = copy.deepcopy(evaluate(launch_path(name), args, str(tmp_path)))
    moveit = include(got, 'moveit.launch.py')
    assert moveit['launch_arguments'] == [['drive_base', 'true']]
    moveit['launch_arguments'] = []
    core = include(got, 'teleop_core.launch.py')
    assert core['launch_arguments'][-1] == ['drive_base', 'true']
    core['launch_arguments'] = core['launch_arguments'][:-1]
    servo = [p for n in got['nodes'] if n.get('executable') == 'servo_node'
             for p in n['parameters'] if 'robot_description_semantic' in p]
    assert [p['robot_description_semantic'] for p in servo] == [DRIVEN_SRDF]
    servo[0]['robot_description_semantic'] = SHIPPED_SRDF
    expected = reference()[case]
    assert got['includes'] == expected['includes']
    assert got['nodes'] == expected['nodes']


def test_moveit_arguments_on_the_sim_command_line_are_refused(tmp_path):
    for given in ('planning_frame:=base_footprint', 'obstacle_map:=true', 'drive_base:=true'):
        with pytest.raises(ValueError, match='teleop_sim.launch.py refused: '
                                             + given.split(':=')[0]):
            evaluate(launch_path('teleop_sim.launch.py'), ['base:=true', given], str(tmp_path))


def test_the_mock_refuses_drive_base(tmp_path):
    with pytest.raises(ValueError, match='teleop_mock.launch.py refused: drive_base'):
        evaluate(launch_path('teleop_mock.launch.py'), ['base:=true', 'drive_base:=true'],
                 str(tmp_path))


def test_start_moveit_false_starts_no_move_group_with_the_base_driven(tmp_path):
    got = evaluate(launch_path('teleop_sim.launch.py'), ['base:=true', 'start_moveit:=false'],
                   str(tmp_path))
    assert not [i for i in got['includes'] if i['file'].endswith('/moveit.launch.py')]
    assert semantic_of(got['nodes'], 'servo_node') == DRIVEN_SRDF


def test_the_core_refuses_drive_base_without_the_base(tmp_path):
    limits = os.path.join(os.path.dirname(launch_path('x')), '..', 'config',
                          'SIMULATION_ONLY_limits.yaml')
    with pytest.raises(ValueError, match='needs base:=true'):
        evaluate(launch_path('teleop_core.launch.py'),
                 ['simulation:=true', f'limits_file:={limits}', 'drive_base:=true'],
                 str(tmp_path))
