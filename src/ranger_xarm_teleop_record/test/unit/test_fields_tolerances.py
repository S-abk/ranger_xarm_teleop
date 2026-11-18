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
"""Absent and required fields, episodes without action, profile tolerances, refitting."""

import copy
import math
import os
import shutil

import numpy as np
import pytest
from ranger_xarm_teleop_record import checker, cli, fit, profile, provenance
from ranger_xarm_teleop_record.streams import FieldPresence
import synth

CHECKS = profile.load_checks()


def reasons(report):
    """Return all reasons as one string."""
    return ' | '.join(report['reasons'])


def without_streams(ep, names):
    """Return a copy of ``ep`` without the streams ``names`` (and their counts)."""
    out = copy.deepcopy(ep)
    for name in names:
        out.streams.pop(name, None)
        out.sidecar['counts'].pop(name, None)
    return out


# ------------------------------------------------------------------ episode without action

def test_an_episode_without_an_engaged_cycle_says_so_once():
    ep = synth.build()
    ep.streams['state'].fields['clutch_engaged'][:] = False
    ep = without_streams(ep, ['target', 'joint_command'])
    report = checker.check_episode(ep, CHECKS)
    assert report['verdict'] == 'FAIL'
    no_action = [r for r in report['reasons'] if r.startswith('NO_ACTION')]
    assert len(no_action) == 1
    assert 'no teleop arm action in the episode' in no_action[0]
    assert 'target' in no_action[0] and 'joint_command' in no_action[0]
    assert 'required stream missing' not in reasons(report)
    assert report['usable'] is False


def test_a_missing_target_while_engaged_is_still_a_missing_stream():
    ep = without_streams(synth.build(), ['target'])
    report = checker.check_episode(ep, CHECKS)
    assert report['verdict'] == 'FAIL'
    assert 'target: required stream missing' in reasons(report)
    assert 'NO_ACTION' not in reasons(report)


# ------------------------------------------------------------------ absent and required fields

def nan_effort(ep):
    """Return ``ep`` with every joint effort NaN (what Gazebo and mock execution publish)."""
    js = ep.streams['joint_states']
    js.fields['effort'] = np.full_like(js.fields['effort'], np.nan)
    return ep


def test_absent_effort_is_information_only():
    report = checker.check_episode(nan_effort(synth.build()), CHECKS)
    assert report['verdict'] == 'PASS', report['reasons']
    assert report['fields']['joint_states'] == {'absent': ['effort'], 'required_fields': []}


def test_present_fields_are_not_listed_as_absent():
    report = checker.check_episode(synth.build(), CHECKS)
    assert report['fields']['joint_states']['absent'] == []


def test_a_profile_that_requires_effort_fails_without_it():
    ep = nan_effort(synth.build())
    ep.profile = copy.deepcopy(ep.profile)
    ep.profile['streams']['joint_states']['required_fields'] = ['effort']
    report = checker.check_episode(ep, CHECKS)
    assert report['verdict'] == 'FAIL'
    assert 'FIELD_MISSING: joint_states: required field effort' in reasons(report)
    # With effort published, the same profile passes.
    ok = synth.build()
    ok.profile = ep.profile
    assert checker.check_episode(ok, CHECKS)['verdict'] == 'PASS'


def test_required_fields_are_validated():
    raw = profile.load_yaml(profile.resolve('mock'))
    bad = copy.deepcopy(raw)
    bad['streams']['joint_states']['required_fields'] = ['torque']
    with pytest.raises(profile.ProfileError, match='torque'):
        profile.validate(bad)
    bad = copy.deepcopy(raw)
    bad['streams']['state']['required_fields'] = ['effort']
    with pytest.raises(profile.ProfileError, match='effort'):
        profile.validate(bad)
    good = copy.deepcopy(raw)
    good['streams']['joint_states']['required_fields'] = ['position', 'effort']
    assert profile.validate(good)['streams']['joint_states']['required_fields'] == \
        ['position', 'effort']


def test_field_presence_tells_empty_from_nan():
    p = FieldPresence(('position', 'velocity', 'effort'))
    assert p.report() == {'messages': 0, 'absent': ['position', 'velocity', 'effort'],
                          'how': {'position': 'no message', 'velocity': 'no message',
                                  'effort': 'no message'}}
    p.add({'position': [0.1, 0.2], 'velocity': [], 'effort': [math.nan, math.nan]})
    p.add({'position': [0.1, math.nan], 'velocity': [], 'effort': [math.nan, math.nan]})
    assert p.absent() == ['velocity', 'effort']
    assert p.report()['how'] == {'velocity': 'empty', 'effort': 'NaN'}
    assert p.last == {'position': True, 'velocity': False, 'effort': False}
    p.add({'position': [0.1], 'velocity': [0.0], 'effort': [1.0]})
    assert p.absent() == [] and p.last['effort'] is True


# ------------------------------------------------------------------ profile tolerances

def rotated_measured(ep, degrees):
    """Return ``ep`` with the measured tool orientation turned by ``degrees`` about z."""
    st = ep.streams['state']
    q = synth.qaxis((0.0, 0.0, 1.0), math.radians(degrees))
    pose = st.fields['measured_tool_pose'].copy()
    for i in range(len(pose)):
        pose[i, 3:] = synth.qmul(q, pose[i, 3:])
    st.fields['measured_tool_pose'] = pose
    return ep


def test_gazebo_sets_a_provisional_rotation_limit_and_mock_does_not():
    gazebo = profile.effective_checks(CHECKS, profile.load_profile('gazebo'), 'test')
    mock = profile.effective_checks(CHECKS, profile.load_profile('mock'), 'test')
    assert gazebo['consistency']['arm_rot_p99_deg'] == 1.5
    assert 'consistency.arm_rot_p99_deg' in gazebo['_tolerances']['provisional']
    assert mock['consistency']['arm_rot_p99_deg'] == CHECKS['consistency']['arm_rot_p99_deg']
    assert mock['_tolerances']['provisional'] == {}
    # Every other tolerance is the file's.
    for section, values in CHECKS.items():
        for key, value in values.items():
            if (section, key) != ('consistency', 'arm_rot_p99_deg'):
                assert gazebo[section][key] == value


def test_the_rotation_limit_judges_by_the_profile_in_force():
    ep = rotated_measured(synth.build(profile_name='gazebo'), 1.0)
    report = checker.check_episode(ep, CHECKS)          # the episode's own profile: gazebo
    assert report['verdict'] == 'PASS', report['reasons']
    assert report['consistency']['limits']['arm_rot_p99_deg'] == 1.5
    assert report['consistency']['provisional'] == ['arm_rot_p99_deg']
    assert report['tolerances']['source'] == 'recorded profile (episode sidecar)'
    mock = profile.effective_checks(CHECKS, profile.load_profile('mock'), 'mock')
    report = checker.check_episode(ep, mock)
    assert report['verdict'] == 'FAIL'
    assert 'rotation p99' in reasons(report) and 'exceeds 0.80 deg' in reasons(report)
    assert report['consistency']['provisional'] == []


def test_an_override_of_an_unknown_tolerance_is_an_error():
    prof = copy.deepcopy(profile.load_profile('gazebo'))
    prof['checks'] = {'consistency': {'arm_rot_p99_degrees': 1.5}}
    with pytest.raises(profile.ProfileError, match='arm_rot_p99_degrees'):
        profile.effective_checks(CHECKS, prof, 'test')
    prof['checks'] = {'no_section': {'x': 1}}
    with pytest.raises(profile.ProfileError, match='no_section'):
        profile.effective_checks(CHECKS, prof, 'test')


@pytest.mark.parametrize('override', [
    {'flags': {'fail_flags': []}},                     # would stop DATA_LOST from failing
    {'streams': {'gap_max_s': 10.0}},
    {'consistency': {'lag_max_s': 5.0}},
    {'consistency': {'arm_rot_p99_deg': 0}},
    {'consistency': {'arm_rot_p99_deg': 'loose'}},
])
def test_a_profile_cannot_switch_off_a_failure(override):
    prof = copy.deepcopy(profile.load_profile('gazebo'))
    prof['checks'], prof['provisional'] = override, {}
    with pytest.raises(profile.ProfileError):
        profile.effective_checks(CHECKS, prof, 'test')
    # The checker refuses to run on such a recorded profile rather than apply it.
    ep = synth.build()
    ep.profile = prof
    with pytest.raises(profile.ProfileError):
        checker.check_episode(ep, CHECKS)


def test_provisional_names_a_tolerance_the_profile_sets():
    raw = profile.load_yaml(profile.resolve('gazebo'))
    raw['provisional'] = {'consistency.arm_rms_m': 'not set here'}
    with pytest.raises(profile.ProfileError, match='arm_rms_m'):
        profile.validate(raw)


def test_the_tools_use_the_current_package_profile_for_old_episodes():
    old = copy.deepcopy(profile.load_profile('gazebo'))
    old['checks'], old['provisional'] = {}, {}         # recorded before the override existed
    checks = cli.tolerances(CHECKS, old)
    assert checks['consistency']['arm_rot_p99_deg'] == 1.5
    assert checks['_tolerances']['source'] == 'package profile gazebo (current)'
    other = dict(old, name='a_downstream_profile')
    checks = cli.tolerances(CHECKS, other)
    assert checks['_tolerances']['source'] == 'recorded profile (episode sidecar)'
    assert checks['consistency']['arm_rot_p99_deg'] == CHECKS['consistency']['arm_rot_p99_deg']
    checks = cli.tolerances(CHECKS, old, 'mock')
    assert checks['_tolerances']['source'] == 'profile mock (given)'


# ------------------------------------------------------------------ refitting

def episodes(values):
    """Return fit inputs with the given rotation p99 values (other metrics fixed)."""
    return [{'id': f'e{i:02d}',
             'metrics': {'arm_rms_m': 0.001, 'arm_p99_m': 0.003, 'arm_rot_p99_deg': v,
                         'gripper_abs_rad': 0.04}}
            for i, v in enumerate(values)]


def test_fit_on_seven_of_ten_and_round_up():
    assert fit.n_fit(10) == 7 and fit.n_fit(5) == 4 and fit.n_fit(2) == 1
    values = [0.1, 0.2, 0.3, 0.52, 0.4, 0.3, 0.2, 0.9, 0.1, 0.1]
    result = fit.fit(episodes(values), {'arm_rot_p99_deg': 1.5})
    assert result['ok']
    assert result['fit'] == [f'e{i:02d}' for i in range(7)]
    assert result['held_out'] == ['e07', 'e08', 'e09']
    assert result['proposed']['arm_rot_p99_deg'] == 0.8          # 0.52 x 1.5 = 0.78 -> 0.8
    assert result['proposed']['arm_rms_m'] == 0.0015              # 0.0015 exactly, not 0.002
    assert result['proposed']['gripper_abs_rad'] == 0.06
    assert result['changes']['arm_rot_p99_deg']['direction'] == 'tighter'
    # The held-out episode at 0.9 deg fails the proposal, and that is reported.
    assert result['held_out_failures'] == {'e07': ['arm_rot_p99_deg']}


def test_fit_refuses_too_few_episodes():
    result = fit.fit(episodes([0.5, 0.9]), {})
    assert not result['ok'] and 'at least 5' in result['reason']
    assert 'proposed' not in result


def test_fit_skips_metrics_that_were_not_judged():
    report = {'arm': {'verdict': 'PASS', 'rms_m': 0.001, 'p99_m': 0.002, 'rot_p99_deg': 0.3},
              'gripper': {'verdict': 'NOT_JUDGED', 'applicable': True, 'p99_rad': None}}
    assert fit.metrics(report) == {'arm_rms_m': 0.001, 'arm_p99_m': 0.002,
                                   'arm_rot_p99_deg': 0.3, 'gripper_abs_rad': None}
    eps = episodes([0.2] * 6)
    for e in eps:
        e['metrics']['gripper_abs_rad'] = None
    result = fit.fit(eps, {})
    assert 'gripper_abs_rad' not in result['proposed']
    assert 'gripper_abs_rad' in result['not_fitted']


def test_the_proposal_block_is_a_profile_checks_block():
    result = fit.fit(episodes([0.3] * 10), {})
    block = fit.profile_block(result, 'test')
    prof = copy.deepcopy(profile.load_profile('gazebo'))
    prof['checks'] = block['checks']
    prof['provisional'] = {}
    checks = profile.effective_checks(CHECKS, prof, 'refit')
    assert checks['consistency']['arm_rot_p99_deg'] == 0.5


# ------------------------------------------------------------------ reproducible verdicts

def test_every_report_names_the_checking_code_and_the_values_in_force():
    import subprocess
    report = checker.check_episode(synth.build(), CHECKS)
    by = report['checked_by']
    here = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(['git', '-C', here, 'rev-parse', 'HEAD'], capture_output=True,
                          text=True, check=False).stdout.strip()
    assert by['commit'] == head and isinstance(by['dirty'], bool)
    assert len(by['source_sha256']) == 64 and by['files'] > 10
    values = report['tolerances']['values']
    expected = profile.effective_checks(CHECKS, synth.build().profile, 'x')
    assert values == {k: v for k, v in expected.items() if k != '_tolerances'}
    assert values['consistency']['arm_rot_p99_deg'] == 1.5


def test_the_source_digest_changes_with_the_checking_code(tmp_path, monkeypatch):
    package = os.path.dirname(os.path.dirname(os.path.abspath(provenance.__file__)))
    copy_root = tmp_path / 'pkg'
    shutil.copytree(package, copy_root, ignore=shutil.ignore_patterns('test', '__pycache__'))
    inner = copy_root / 'ranger_xarm_teleop_record'
    monkeypatch.setattr(provenance, '_package_root', lambda: str(inner))
    first = provenance.checking_code.__wrapped__()
    assert first['commit'] is None                       # not in a git tree: said so
    with open(inner / 'checker.py', 'a', encoding='utf-8') as handle:
        handle.write('\n# changed\n')
    assert provenance.checking_code.__wrapped__()['source_sha256'] != first['source_sha256']
