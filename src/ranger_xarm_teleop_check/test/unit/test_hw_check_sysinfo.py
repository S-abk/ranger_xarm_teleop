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
"""Machine state, files, robot description and the driver gate (outside effects faked)."""

import os
import subprocess

import pytest
from ranger_xarm_teleop_check import sysinfo
from ranger_xarm_teleop_record.provenance import sha256_text


def _fake_sysfs(root, governors):
    for i, gov in enumerate(governors):
        d = root / f'cpu{i}' / 'cpufreq'
        d.mkdir(parents=True)
        (d / 'scaling_governor').write_text(gov + '\n')
    (root / 'cpufreq').mkdir(exist_ok=True)       # not a cpu: ignored
    (root / 'cpuidle').mkdir(exist_ok=True)


def test_governors_read_from_fake_sysfs(tmp_path):
    _fake_sysfs(tmp_path, ['performance', 'powersave'])
    assert sysinfo.read_governors(str(tmp_path)) == {'cpu0': 'performance', 'cpu1': 'powersave'}
    info = sysinfo.machine_info(str(tmp_path), str(tmp_path / 'nope'))
    assert info['governor_set'] == ['performance', 'powersave'] and info['loadavg'] is None


def test_governors_absent(tmp_path):
    assert sysinfo.read_governors(str(tmp_path)) == {}


def test_loadavg(tmp_path):
    f = tmp_path / 'loadavg'
    f.write_text('0.52 0.40 0.30 1/200 12345\n')
    assert sysinfo.read_loadavg(str(f)) == (0.52, 0.40, 0.30)
    f.write_text('garbage')
    assert sysinfo.read_loadavg(str(f)) is None


def test_package_versions(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'package.xml').write_text('<package><name>a</name><version>4.48.1'
                                                '</version></package>')
    (tmp_path / 'b').mkdir()
    (tmp_path / 'b' / 'package.xml').write_text('not xml')

    def share(name):
        if name == 'c':
            raise LookupError(name)
        return str(tmp_path / name)

    assert sysinfo.package_versions(['a', 'b', 'c'], share) == {'a': '4.48.1', 'b': None,
                                                                'c': None}


URDF = """<robot name="r">
  <ros2_control name="s" type="system"><hardware><plugin>{plugin}</plugin>
    <param name="velocity_control">{vel}</param></hardware></ros2_control>
  <joint name="xarm_joint1"/><joint name="xarm_drive_joint"/>
</robot>"""


def test_description_hardware_position_control():
    text = URDF.format(plugin='xarm_control/UFRobotSystemHardware', vel='False')
    d = sysinfo.description_info(text)
    assert d['sha256'] == sha256_text(text)
    assert d['plugins'] == ['xarm_control/UFRobotSystemHardware']
    assert d['simulator'] is False and d['gripper'] is True
    assert d['velocity_control'] == ['false'] and d['error'] is None


def test_description_simulator_and_no_param():
    d = sysinfo.description_info(URDF.format(plugin='mock_components/GenericSystem',
                                             vel='true'))
    assert d['simulator'] is True and d['velocity_control'] == ['true']
    d = sysinfo.description_info('<robot><joint name="xarm_joint1"/></robot>')
    assert d['simulator'] is None and d['velocity_control'] == [] and d['gripper'] is False


def test_description_not_xml():
    d = sysinfo.description_info('<robot')
    assert d['error'] and d['plugins'] == []


def test_limit_parameter_any_nesting():
    data = {'/**': {'ros__parameters': {'a': 1, 'group': {'tool_pose_timeout_s': 0.25}}}}
    assert sysinfo.limit_parameter(data, 'tool_pose_timeout_s') == 0.25
    assert sysinfo.limit_parameter(data, 'missing') is None
    assert sysinfo.limit_parameter({'tool_pose_timeout_s': 1.0}, 'tool_pose_timeout_s') is None
    assert sysinfo.limit_parameter(None, 'x') is None


def test_is_number():
    assert sysinfo.is_number(1) and sysinfo.is_number(0.5)
    assert not sysinfo.is_number(True) and not sysinfo.is_number(float('nan'))
    assert not sysinfo.is_number('1') and not sysinfo.is_number(None)


def test_joint_acceleration_limits():
    data = {'joint_limits': {
        'xarm_joint1': {'has_acceleration_limits': True, 'max_acceleration': 2.5},
        'xarm_joint2': {'has_acceleration_limits': False, 'max_acceleration': 9.0},
        'xarm_joint3': {'has_acceleration_limits': True},
    }}
    assert sysinfo.joint_acceleration_limits(data) == {
        'xarm_joint1': 2.5, 'xarm_joint2': None, 'xarm_joint3': None}
    assert sysinfo.joint_acceleration_limits({}) == {}


def test_load_yaml_file(tmp_path):
    f = tmp_path / 'x.yaml'
    f.write_text('a: [1, 2')
    data, problem = sysinfo.load_yaml_file(str(f))
    assert data is None and 'cannot read' in problem
    data, problem = sysinfo.load_yaml_file(str(tmp_path / 'none.yaml'))
    assert data is None and problem
    f.write_text('a: 1')
    assert sysinfo.load_yaml_file(str(f)) == ({'a': 1}, None)


# ---- the driver gate --------------------------------------------------------------------

def _make_runner(replies):
    """Return a runner answering by the git verb; ``replies`` maps verb -> (rc, out)."""
    calls = []

    def run(args):
        calls.append(args)
        verb = [a for a in args if a in ('rev-parse', 'describe', 'cat-file', 'merge-base')][0]
        return replies.get(verb, (1, ''))

    run.calls = calls
    return run


def _workspace(tmp_path, source=None):
    repo = tmp_path / 'src' / 'xarm_ros2'
    repo.mkdir(parents=True)
    if source is not None:
        path = repo / sysinfo.DRIVER_SOURCE
        path.parent.mkdir(parents=True)
        path.write_text(source)
    return str(tmp_path)


def test_gate_workspace_not_given():
    g = sysinfo.driver_gate(None)
    assert g['verdict'] == 'NOT JUDGED' and 'pass --workspace' in g['reason']
    assert sysinfo.driver_gate('/nonexistent/ws')['verdict'] == 'NOT JUDGED'


def test_gate_ancestor_passes(tmp_path):
    run = _make_runner({'rev-parse': (0, 'abc123\n'), 'describe': (0, 'v1-3-gabc\n'),
                        'cat-file': (0, ''), 'merge-base': (0, '')})
    g = sysinfo.driver_gate(_workspace(tmp_path), run)
    assert g['verdict'] == 'PASS' and g['head'] == 'abc123' and g['describe'] == 'v1-3-gabc'
    assert 'ancestor' in g['evidence']


def test_gate_known_but_not_ancestor_fails_without_fallback(tmp_path):
    ws = _workspace(tmp_path, '// 20250318\n get_joint_states(')
    run = _make_runner({'rev-parse': (0, 'abc\n'), 'cat-file': (0, ''), 'merge-base': (1, '')})
    assert sysinfo.driver_gate(ws, run)['verdict'] == 'FAIL'


def test_gate_unknown_object_uses_source_markers(tmp_path):
    ws = _workspace(tmp_path, '// 20250318, get joint\n x = driver.get_joint_states();')
    run = _make_runner({'rev-parse': (0, 'abc\n'), 'cat-file': (128, '')})
    g = sysinfo.driver_gate(ws, run)
    assert g['verdict'] == 'PASS' and 'source markers' in g['evidence']
    ws2 = _workspace(tmp_path / 'two', '// 20250318 only')
    g = sysinfo.driver_gate(ws2, run)
    assert g['verdict'] == 'FAIL' and 'get_joint_states(' in g['evidence']


def test_gate_source_unreadable_is_not_judged(tmp_path):
    run = _make_runner({'rev-parse': (128, '')})
    g = sysinfo.driver_gate(_workspace(tmp_path), run)
    assert g['verdict'] == 'NOT JUDGED' and 'cannot read' in g['reason']


@pytest.mark.skipif(subprocess.run(['git', '--version'], capture_output=True).returncode != 0,
                    reason='git missing')
def test_gate_with_a_real_repository(tmp_path, monkeypatch):
    repo = tmp_path / 'src' / 'xarm_ros2'
    repo.mkdir(parents=True)
    env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@e',
               GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@e')

    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True, env=env,
                              capture_output=True, text=True).stdout

    git('init', '-q')
    (repo / 'f').write_text('1')
    git('add', 'f')
    git('commit', '-q', '-m', 'one')
    # cc44f9f is unknown in this repository: the gate falls back to the source markers
    g = sysinfo.driver_gate(str(tmp_path))
    assert g['head'] == git('rev-parse', 'HEAD').strip()
    assert g['verdict'] == 'NOT JUDGED' and 'cannot read' in g['reason']
    # with a commit that exists: an ancestor passes, a commit on another branch fails
    fix = git('rev-parse', 'HEAD').strip()
    monkeypatch.setattr(sysinfo, 'DRIVER_FIX_COMMIT', fix)
    (repo / 'f').write_text('2')
    git('commit', '-q', '-am', 'two')
    g = sysinfo.driver_gate(str(tmp_path))
    assert g['verdict'] == 'PASS' and g['describe']
    git('checkout', '-q', '--orphan', 'other')
    git('commit', '-q', '-m', 'unrelated')
    assert sysinfo.driver_gate(str(tmp_path))['verdict'] == 'FAIL'


def test_choose_qos():
    assert sysinfo.choose_qos([], 'reliable', 'volatile') == ('reliable', 'volatile')
    assert sysinfo.choose_qos([('best_effort', 'volatile')], 'reliable', 'volatile') == (
        'best_effort', 'volatile')
    assert sysinfo.choose_qos([('reliable', 'volatile')], 'reliable', 'transient_local') == (
        'reliable', 'volatile')
    assert sysinfo.choose_qos([('reliable', 'transient_local')], 'reliable',
                              'transient_local') == ('reliable', 'transient_local')
