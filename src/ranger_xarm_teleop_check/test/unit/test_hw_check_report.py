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
"""The report: path, refusal, content, printing; and that the tool stays read-only."""

import datetime
import json
import math
import os
import re
import subprocess
import sys

import pytest
from ranger_xarm_teleop_check import report as rep
from ranger_xarm_teleop_check import rules

PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODULE_DIR = os.path.join(PKG_DIR, 'ranger_xarm_teleop_check')
PURE = ('rules', 'stats', 'report', 'sysinfo')


def _git(path, *args):
    subprocess.run(['git', '-C', str(path), *args], check=True, capture_output=True)


def test_banner_is_exact():
    assert rep.BANNER == ('read-only: this tool only subscribes; it creates no publisher, '
                          'client, action or parameter call')


def test_report_path():
    when = datetime.datetime(2026, 3, 4, 5, 6, 7)
    path = rep.report_path('/data/x', 'idle', when)
    assert path == '/data/x/hardware_checks/20260304-050607_idle.json'
    assert rep.report_dir('/data/x') == '/data/x/hardware_checks'


def test_report_dir_outside_git_is_allowed(tmp_path):
    assert rep.report_dir_problem(str(tmp_path / 'data')) is None


def test_report_dir_in_unignored_git_tree_is_refused(tmp_path):
    _git(tmp_path, 'init', '-q')
    problem = rep.report_dir_problem(str(tmp_path / 'data'))
    assert problem and 'not ignored' in problem


def test_report_dir_in_ignoring_git_tree_is_allowed(tmp_path):
    _git(tmp_path, 'init', '-q')
    (tmp_path / '.gitignore').write_text('data/\n')
    assert rep.report_dir_problem(str(tmp_path / 'data')) is None


def test_report_dir_in_the_code_checkout_is_refused():
    problem = rep.report_dir_problem(os.path.join(PKG_DIR, 'data'))
    assert problem and 'code checkout' in problem


def test_sanitize_replaces_non_finite_numbers():
    out = rep.sanitize({'a': math.nan, 'b': [math.inf, 1.0], 'c': (2,), 3: 'x'})
    assert out == {'a': None, 'b': [None, 1.0], 'c': [2], '3': 'x'}


def test_build_and_write_report(tmp_path):
    start = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    end = start + datetime.timedelta(seconds=30)
    results = [rules.Result('rate', rules.FAIL, '10 Hz', '>= 5 Hz', 'do it'),
               rules.Result('x', rules.PASS, '1', '2')]
    doc = rep.build_report('idle', start, end, {'v': math.nan}, results,
                           {'packages': {'a': '1'}}, ['hw_check', 'idle'], ['w'], 'abc', 1)
    path = rep.write_report(rep.report_path(str(tmp_path), 'idle', start), doc)
    back = json.loads(open(path).read())
    assert back['stage'] == 'idle' and back['code_commit'] == 'abc'
    assert back['start_time'].startswith('2026-01-02T03:04:05')
    assert back['end_time'].startswith('2026-01-02T03:04:35')
    assert back['measures'] == {'v': None} and back['argv'] == ['hw_check', 'idle']
    assert back['rules'][0] == {'rule': 'rate', 'verdict': 'FAIL', 'measured': '10 Hz',
                                'threshold': '>= 5 Hz', 'reason': 'do it'}
    assert back['versions'] == {'packages': {'a': '1'}} and back['exit_status'] == 1
    assert {'local_time', 'utc_offset', 'epoch_s'} <= set(back['host_clock'])
    assert os.path.basename(path) == '20260102-030405_idle.json'


def test_code_commit_unknown_outside_a_repository(tmp_path):
    (tmp_path / 'f.py').write_text('')
    assert rep.code_commit(str(tmp_path / 'f.py')) == 'unknown'


def test_code_commit_of_this_checkout():
    assert re.fullmatch(r'[0-9a-f]{40}|unknown', rep.code_commit())


def test_format_results():
    text = rep.format_results([rules.Result('a', rules.PASS, '1', '2'),
                               rules.Result('b', rules.NOT_JUDGED, '3', '4', 'why')])
    lines = text.splitlines()
    assert lines == ['PASS  a: 1 (rule: 2)', 'NOT JUDGED  b: 3 (rule: 4) -- to do: why']


@pytest.mark.parametrize('module', PURE)
def test_pure_modules_do_not_import_ros_or_the_node(module):
    source = open(os.path.join(MODULE_DIR, module + '.py'), encoding='utf-8').read()
    names = ('rclpy', 'rcl_interfaces', 'ament_index_python', 'sensor_msgs', 'std_msgs',
             'geometry_msgs', 'control_msgs', 'controller_manager_msgs',
             'ranger_xarm_teleop_interfaces', 'hw_check')
    pattern = r'^\s*(?:from|import)\s+(?:%s)\b' % '|'.join(map(re.escape, names))
    assert not re.search(pattern, source, re.MULTILINE)
    assert not re.search(r'^\s*(?:from|import)\b.*\bhw_check\b', source, re.MULTILINE)


def test_pure_modules_import_without_ros():
    code = ('import sys\nsys.path.insert(0, %r)\n' % PKG_DIR
            + 'sys.path[:0] = [%r, %r, %r]\n' % tuple(
                os.path.join(os.path.dirname(PKG_DIR), p)
                for p in ('ranger_xarm_teleop_record', 'ranger_xarm_teleop_servo',
                          'ranger_xarm_teleop_bridge'))
            + 'import ranger_xarm_teleop_check.rules, ranger_xarm_teleop_check.report\n'
              'bad = [m for m in sys.modules if m.split(".")[0] == "rclpy"]\n'
              'bad += [m for m in sys.modules if m.endswith(".hw_check")]\n'
              'print(bad)\nsys.exit(1 if bad else 0)\n')
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


# INV-3: the tool only subscribes. These calls must never appear in the node module.
FORBIDDEN = ('create_publisher', 'create_client', 'create_service', 'ActionClient',
             'ActionServer', 'declare_parameter', 'get_parameter', 'set_parameters',
             'list_parameters', 'create_timer', 'create_guard_condition', 'ParameterClient',
             'AsyncParametersClient', 'SyncParametersClient', 'wait_for_service')


def _code_lines(path):
    """Return the source with comments and docstring text removed."""
    import io
    import tokenize
    out = []
    with open(path, encoding='utf-8') as handle:
        for tok in tokenize.generate_tokens(handle.readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING, tokenize.NL,
                                tokenize.NEWLINE):
                out.append(tok.string)
            elif tok.type == tokenize.STRING and not tok.string.startswith(('"""', "'''")):
                out.append(tok.string)
    return ' '.join(out) if out else io.StringIO().getvalue()


@pytest.mark.parametrize('module', ('hw_check',) + PURE)
def test_no_endpoint_creation_but_subscriptions(module):
    code = _code_lines(os.path.join(MODULE_DIR, module + '.py'))
    assert 'def ' in code
    assert module != 'hw_check' or 'create_subscription' in code
    for word in FORBIDDEN:
        assert word not in code, f'{module}: {word}'


def test_the_node_has_no_parameter_services():
    source = open(os.path.join(MODULE_DIR, 'hw_check.py'), encoding='utf-8').read()
    assert 'start_parameter_services=False' in source
    assert len(re.findall(r'\bNode\(', source)) == 1
    assert source.count('create_subscription(') == 2      # the helper and the camera streams


def _reject(constant):
    raise ValueError('not JSON: ' + constant)


def _populated_data():
    from test_hw_check_rules import _cam, _teleop_data, arm_samples
    data = _teleop_data()
    data['arm'] = arm_samples(n=600, speed=0.5)
    data['cameras'] = {'cam': _cam()}
    return data


@pytest.mark.parametrize('stage', rules.STAGES)
def test_every_stage_report_round_trips(stage, tmp_path):
    streams = [{'name': 'cam', 'topic': '/cam/image', 'rate_hz': 30.0,
                'image': {'width': 848, 'height': 480, 'encoding': 'rgb8'}}]
    opts = rules.Options(streams=streams, limits_given=True, tool_pose_timeout_s=1.0,
                         max_lead_translation_m=0.05)
    measures, results = rules.evaluate(stage, _populated_data(), opts)
    start = datetime.datetime(2026, 1, 2, 3, 4, 5).astimezone()
    doc = rep.build_report(stage, start, start, measures, results, {}, ['x'], [], 'abc',
                           rules.exit_status(results))
    path = rep.write_report(rep.report_path(str(tmp_path), stage, start), doc)
    back = json.loads(open(path).read(), parse_constant=_reject)
    assert back['stage'] == stage and len(back['rules']) == len(results)
    assert all(r['verdict'] in ('PASS', 'FAIL', 'NOT JUDGED', 'INFO') for r in back['rules'])
    assert stage != 'teleop' or 'lead' in back['measures']['teleop']
