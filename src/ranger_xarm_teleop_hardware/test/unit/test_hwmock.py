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
"""The hardware-shaped mock's pure helpers, against the real bring-up launch files."""

import ast
import os
from pathlib import Path

import pytest

from ranger_xarm_teleop_hardware import hwmock

PKG_DIR = Path(__file__).resolve().parents[2]
LAUNCH = PKG_DIR / 'launch' / 'hwmock.launch.py'
REAL_LAUNCH_FILES = {'check': 'real_hardware_check.launch.py',
                     'execution': 'real_execution.launch.py'}


def _workspace():
    """Return the robot workspace's moveit config launch directory, or None."""
    candidates = []
    if os.environ.get('RANGER_XARM_WS'):
        candidates.append(Path(os.environ['RANGER_XARM_WS']))
    candidates += [parent / 'ranger_xarm_ws' for parent in PKG_DIR.parents]
    for root in candidates:
        launch = root / 'src' / 'ranger_xarm_moveit_config' / 'launch'
        if launch.is_dir():
            return launch
    return None


needs_workspace = pytest.mark.skipif(
    _workspace() is None,
    reason='the robot workspace (ranger_xarm_ws, or RANGER_XARM_WS) is not present, so the '
           'real launch files cannot be read')


def _call_name(node):
    func = node.func
    return func.id if isinstance(func, ast.Name) else getattr(func, 'attr', '')


def _real_tree(stage):
    return ast.parse((_workspace() / REAL_LAUNCH_FILES[stage]).read_text(encoding='utf-8'))


def _declared_defaults(tree):
    """Return {launch argument name: default text or None} of the DeclareLaunchArgument calls."""
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == 'DeclareLaunchArgument':
            default = None
            for keyword in node.keywords:
                if keyword.arg == 'default_value':
                    default = ast.literal_eval(keyword.value)
            defaults[ast.literal_eval(node.args[0])] = default
    return defaults


def _real_xacro_args(stage):
    """Return the real launch's xacro arguments, with launch arguments at their defaults."""
    tree = _real_tree(stage)
    defaults = _declared_defaults(tree)
    configs = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and _call_name(node.value) == 'LaunchConfiguration'):
            configs[node.targets[0].id] = ast.literal_eval(node.value.args[0])
    command = None
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and any(
                isinstance(e, ast.Constant) and e.value == 'xacro ' for e in node.elts):
            command = node.elts
    assert command is not None, 'no xacro command list in the real launch file'
    words = []
    pending = ''
    for element in command:
        if isinstance(element, ast.Constant):
            text = element.value.strip()
            if text.endswith(':='):
                pending = text
            elif text.startswith('xacro'):
                continue
            else:
                words.append(text)
        elif isinstance(element, ast.Name) and pending:
            default = defaults[configs[element.id]]
            words.append(pending + ('<no default>' if default is None else default))
            pending = ''
    return words


def _real_spawner_names(stage):
    names = []
    for node in ast.walk(_real_tree(stage)):
        if isinstance(node, ast.Call) and _call_name(node) == 'Node':
            keywords = {k.arg: k.value for k in node.keywords}
            executable = keywords.get('executable')
            if isinstance(executable, ast.Constant) and executable.value == 'spawner':
                names.append(ast.literal_eval(keywords['arguments']).pop(0)
                             if isinstance(keywords['arguments'].elts[0], ast.Constant)
                             else None)
    return names


@needs_workspace
@pytest.mark.parametrize('stage', hwmock.STAGES)
def test_xacro_args_are_the_real_launchs_with_only_the_plugin_changed(stage):
    real = _real_xacro_args(stage)
    mock = hwmock.description_args(stage)
    assert 'xarm_ros2_control_plugin:=' + hwmock.MOCK_PLUGIN in mock
    real_plugin = [a for a in real if a.startswith('xarm_ros2_control_plugin:=')]
    assert len(real_plugin) == 1 and 'uf_robot_hardware' in real_plugin[0]
    swapped = [('xarm_ros2_control_plugin:=' + hwmock.MOCK_PLUGIN
                if a.startswith('xarm_ros2_control_plugin:=') else a) for a in real]
    # The one exclusion: the robot address, which only the real plugin reads.
    assert [a for a in swapped if a not in mock] == ['xarm_robot_ip:=<no default>']
    assert [a for a in mock if a not in swapped] == []
    assert [a for a in mock if a in swapped] == [a for a in swapped if a in mock]


@needs_workspace
def test_planning_margin_is_the_real_launchs_default():
    defaults = _declared_defaults(_real_tree('execution'))
    assert hwmock.PLANNING_MARGIN == defaults['planning_margin']
    assert hwmock.planning_description_args('execution')[-1] == (
        'planning_margin:=' + hwmock.PLANNING_MARGIN)


@needs_workspace
@pytest.mark.parametrize('stage', hwmock.STAGES)
def test_real_controllers_are_spawned_by_the_mock(stage):
    real = [name for name in _real_spawner_names(stage) if name]
    mock = [name for name, _ in hwmock.spawner_specs(stage)]
    assert real and all(name in mock for name in real)
    assert [name for name in mock if name not in real] == (
        ['teleop_arm_controller'] if stage == 'execution' else [])


def test_description_args_stay_real_hardware_free():
    for stage in hwmock.STAGES:
        text = ' '.join(hwmock.description_args(stage) + hwmock.planning_description_args(stage))
        assert 'uf_robot_hardware' not in text
        assert 'robot_ip' not in text
    assert 'add_gripper:=false' in hwmock.description_args('check')
    assert 'add_gripper:=true' in hwmock.description_args('execution')
    assert 'xarm_velocity_control:=false' in hwmock.description_args('execution')


def test_controllers_per_stage():
    assert hwmock.spawner_specs('check') == [('joint_state_broadcaster', ())]
    assert hwmock.spawner_specs('execution') == [
        ('joint_state_broadcaster', ()), ('xarm6_traj_controller', ()),
        ('teleop_arm_controller', ('--inactive',))]
    arguments = hwmock.spawner_arguments('teleop_arm_controller', ('--inactive',))
    assert arguments[:2] == ['teleop_arm_controller', '--inactive']
    assert '--controller-manager' in arguments


def test_teleop_fragment_is_passed_in_execution_only():
    assert hwmock.TELEOP_FRAGMENT in hwmock.own_config_files('execution')
    assert hwmock.own_config_files('check') == []
    for name in hwmock.own_config_files('execution'):
        assert (PKG_DIR / name).is_file(), name
    # The launch file names no controller file itself: only through the helper.
    assert 'teleop_arm_controller_real' not in LAUNCH.read_text(encoding='utf-8')


def test_stage_is_refused_with_one_line():
    for stage in hwmock.STAGES:
        assert hwmock.check_stage(stage) == stage
    for stage in ('', 'real', 'Check', None):
        with pytest.raises(ValueError) as error:
            hwmock.check_stage(stage)
        assert '\n' not in str(error.value) and str(error.value).startswith('hwmock refused')
    with pytest.raises(ValueError):
        hwmock.description_args('real')


def test_launch_starts_with_the_mock_log_line_then_the_guard():
    text = LAUNCH.read_text(encoding='utf-8')
    assert 'hwmock: HARDWARE-SHAPED MOCK (mock_components, no robot) - stage ' in text
    tree = ast.parse(text)
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == 'generate_launch_description')
    calls = [_call_name(n) for n in ast.walk(function) if isinstance(n, ast.Call)]
    assert calls.index('LogInfo') < calls.index('OpaqueFunction')


def test_refusal_reason():
    assert hwmock.refusal_reason(['/other/service'], 0) is None
    assert hwmock.refusal_reason(['/x', hwmock.CONTROLLER_MANAGER_SERVICE], 0) == hwmock.REFUSAL
    assert hwmock.refusal_reason([], 1) == hwmock.REFUSAL
    assert '\n' not in hwmock.REFUSAL
    assert hwmock.REFUSAL == (
        'hwmock refused: a controller manager is already answering on this ROS domain '
        '(a real bring-up or another stack); hwmock never starts beside one')


def test_broadcaster_file_restricts_to_the_arm_joints():
    import yaml
    data = yaml.safe_load((PKG_DIR / hwmock.BROADCASTER_FILE).read_text(encoding='utf-8'))
    params = data['joint_state_broadcaster']['ros__parameters']
    assert params['joints'] == list(hwmock.ARM_JOINTS)
    # Without interfaces the broadcaster ignores joints and publishes everything.
    assert params['interfaces'] == ['position', 'velocity']


def test_gripper_clamp():
    assert hwmock.clamp_position(-0.3) == 0.0
    assert hwmock.clamp_position(0.4) == 0.4
    assert hwmock.clamp_position(2.0) == 0.849


def test_gripper_step_toward():
    assert hwmock.step_toward(0.0, 0.5, 1.0, 0.1) == pytest.approx(0.1)
    assert hwmock.step_toward(0.5, 0.0, 1.0, 0.1) == pytest.approx(0.4)
    assert hwmock.step_toward(0.45, 0.5, 1.0, 0.1) == 0.5
    assert hwmock.step_toward(0.2, 0.2, 1.0, 0.1) == 0.2
    assert hwmock.step_toward(0.0, 0.5, 1.0, 0.0) == 0.0
    assert hwmock.step_toward(0.0, 0.5, 1.0, -1.0) == 0.0
