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
"""The command-versus-measured consistency check: it must catch each injected fault."""

import copy
import math

import numpy as np
import pytest
from ranger_xarm_teleop_record import consistency, profile
import synth

CHECKS = profile.load_checks()


@pytest.fixture(scope='module')
def clean():
    """One clean episode, never modified (tests deep-copy it)."""
    return synth.build()


def run(ep):
    """Run the consistency check with the default tolerances."""
    return consistency.check(ep, CHECKS)


def test_clean_episode_passes(clean):
    result = run(clean)
    assert result['verdict'] == 'PASS', result['reasons']
    arm = result['arm']
    assert arm['lag_s'] == pytest.approx(synth.LAG_ARM, abs=0.006)
    assert arm['lag_identifiable'] is True
    assert arm['n_clean'] >= 25 and arm['n_limited'] == 0
    assert arm['rms_m'] < 0.002 and arm['p99_m'] < 0.004
    assert arm['rot_p99_deg'] < 0.01
    assert arm['min_rms_curve_m'] == pytest.approx(arm['rms_m'], abs=1e-9)
    assert arm['rms_lag0_m'] > arm['rms_m']
    gripper = result['gripper']
    assert gripper['verdict'] == 'PASS' and gripper['closed_direction'] == 1
    assert 0.04 <= gripper['lag_s'] <= 0.30
    assert gripper['n_grasp'] == 0


def test_limited_samples_are_reported_not_judged():
    ep = synth.build(limited_share=0.3)
    result = run(ep)
    arm = result['arm']
    assert result['verdict'] == 'PASS', result['reasons']
    assert arm['n_limited'] > 0.2 * (arm['n_clean'] + arm['n_limited'])
    assert arm['limited']['rms_m'] > 0.02
    assert arm['limited']['max_m'] > 0.04
    assert arm['rms_m'] < 0.005
    assert arm['lag_s'] == pytest.approx(synth.LAG_ARM, abs=0.006)


@pytest.mark.parametrize('heading_deg', [30.0, 90.0])
def test_target_rotated_by_a_heading_fails(clean, heading_deg):
    ep = copy.deepcopy(clean)
    pose = synth.rotate_poses_about_z(ep.streams['target'].fields['pose'],
                                      math.radians(heading_deg))
    ep.streams['target'].fields['pose'] = pose
    result = run(ep)
    assert result['verdict'] == 'FAIL'
    assert result['arm']['verdict'] == 'FAIL'
    text = ' | '.join(result['arm']['reasons'])
    assert 'translation RMS' in text
    assert 'rotation p99' in text
    assert result['arm']['frame_ok'] is True


def test_target_in_the_tool_frame_fails():
    result = run(synth.build(tool_frame_id=True))
    assert result['verdict'] == 'FAIL'
    assert any('target frame' in r for r in result['arm']['reasons'])
    assert result['arm']['frame_ok'] is False
    # Everything else about the samples is fine: only the frame fails.
    assert len(result['arm']['reasons']) == 1


def test_target_frame_falls_back_to_the_profile(clean):
    ep = copy.deepcopy(clean)
    ep.sidecar['frames'].pop('base')
    assert run(ep)['arm']['frame_expected'] == ep.profile['frames']['base']


def test_target_stamps_shifted_half_a_second_fail(clean):
    result = run(synth.shift_stamps(clean, 'target', 0.5))
    assert result['verdict'] == 'FAIL'
    assert result['arm']['lag_s'] < synth.LAG_ARM - 0.1


def test_target_stamps_shifted_a_little_fail_on_the_lag(clean):
    result = run(synth.shift_stamps(clean, 'target', 0.1))
    arm = result['arm']
    assert result['verdict'] == 'FAIL'
    assert arm['lag_s'] == pytest.approx(0.02, abs=0.006)
    assert any('lag' in r for r in arm['reasons'])
    assert arm['rms_m'] < 0.005       # the residual itself is fine at the fitted lag


def test_target_and_measured_swapped_fail_with_negative_lag(clean):
    ep = copy.deepcopy(clean)
    state = ep.streams['state']
    target = ep.streams['target']
    rows = np.searchsorted(state.stamp, target.stamp - 1e-6)
    old_measured = state.fields['measured_tool_pose'].copy()
    state.fields['measured_tool_pose'] = np.array([synth.target_pose(t) for t in state.stamp])
    target.fields['pose'] = old_measured[rows]
    result = run(ep)
    assert result['verdict'] == 'FAIL'
    assert result['arm']['lag_s'] == pytest.approx(-synth.LAG_ARM, abs=0.006)
    assert any('lag' in r for r in result['arm']['reasons'])


def test_gripper_command_and_measured_swapped_fail(clean):
    ep = copy.deepcopy(clean)
    state = ep.streams['state']
    command = ep.streams['gripper_command']
    rows = np.searchsorted(state.stamp, command.stamp - 1e-6)
    old_measured = state.fields['gripper_measured'].copy()
    state.fields['gripper_measured'] = state.fields['gripper_command'].copy()
    command.fields['position'] = old_measured[rows]
    result = run(ep)
    assert result['verdict'] == 'FAIL'
    assert result['gripper']['verdict'] == 'FAIL'
    assert result['gripper']['lag_s'] == pytest.approx(-synth.LAG_GRIPPER, abs=0.012)
    assert result['arm']['verdict'] == 'PASS'


def test_gripper_closed_direction_is_inferred_from_the_request(clean):
    # Mirror the opening about zero: closed is now the negative direction, still passes.
    ep = copy.deepcopy(clean)
    ep.streams['gripper_command'].fields['position'] = \
        -ep.streams['gripper_command'].fields['position']
    state = ep.streams['state']
    state.fields['gripper_measured'] = -state.fields['gripper_measured']
    result = run(ep)
    assert result['gripper']['closed_direction'] == -1
    assert result['gripper']['verdict'] == 'PASS', result['gripper']['reasons']


def test_grasp_interval_is_excluded_and_reported(clean):
    ep = copy.deepcopy(clean)
    command = ep.streams['gripper_command']
    position = command.fields['position'].copy()
    late = (command.stamp - synth.T0 > 3.0) & (command.stamp - synth.T0 < 3.5)
    position[late] = 1.0            # asks for more than the object lets it close
    command.fields['position'] = position
    result = run(ep)
    assert result['gripper']['n_grasp'] > 10
    assert result['gripper']['verdict'] == 'PASS', result['gripper']['reasons']
    # The same shortfall while the measured opening is still moving is not a grasp.
    moving = copy.deepcopy(clean)
    pos = moving.streams['gripper_command'].fields['position'].copy()
    early = (moving.streams['gripper_command'].stamp - synth.T0 > 2.0) \
        & (moving.streams['gripper_command'].stamp - synth.T0 < 2.2)
    pos[early] += 0.3
    moving.streams['gripper_command'].fields['position'] = pos
    out = run(moving)
    assert out['gripper']['n_grasp'] == 0
    assert out['gripper']['verdict'] == 'FAIL'


def test_gripper_not_applicable_without_command_rows(clean):
    ep = copy.deepcopy(clean)
    del ep.streams['gripper_command']
    result = run(ep)
    assert result['gripper']['applicable'] is False
    assert result['verdict'] == 'PASS'


def test_few_clean_samples_are_not_judged(clean):
    ep = copy.deepcopy(clean)
    target = ep.streams['target']
    keep = np.zeros(len(target), dtype=bool)
    keep[:10] = True
    ep.streams['target'] = synth.select_rows(target, keep)
    result = run(ep)
    assert result['arm']['verdict'] == 'NOT_JUDGED'
    assert 'clean samples' in result['arm']['reason']
    assert result['verdict'] == 'NOT_JUDGED'


def test_few_samples_do_not_hide_a_frame_error(clean):
    ep = copy.deepcopy(clean)
    target = ep.streams['target']
    keep = np.zeros(len(target), dtype=bool)
    keep[:10] = True
    ep.streams['target'] = synth.select_rows(target, keep)
    ep.streams['target'].frame_id = [synth.TOOL_FRAME] * 10
    assert run(ep)['verdict'] == 'FAIL'


def test_short_target_path_does_not_identify_the_lag(clean):
    ep = copy.deepcopy(clean)
    state = ep.streams['state']
    target = ep.streams['target']
    still = np.tile(np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]), (len(target), 1))
    target.fields['pose'] = still
    measured = np.tile(still[0], (len(state), 1))
    measured[:, :3] += np.random.default_rng(3).normal(0.0, 0.0005, (len(state), 3))
    state.fields['measured_tool_pose'] = measured
    result = run(ep)
    arm = result['arm']
    assert arm['lag_identifiable'] is False
    assert arm['lag_s'] == pytest.approx(0.17)
    assert result['verdict'] == 'PASS', result['reasons']


def test_no_target_rows_is_not_judged(clean):
    ep = copy.deepcopy(clean)
    del ep.streams['target']
    result = run(ep)
    assert result['arm']['verdict'] == 'NOT_JUDGED'
    assert result['verdict'] == 'NOT_JUDGED'


def test_consistency_does_not_modify_the_episode(clean):
    ep = copy.deepcopy(clean)
    before = synth.snapshot(ep)
    run(ep)
    assert synth.same(before, synth.snapshot(ep))


def test_quat_angle_between():
    a = np.array([synth.qaxis((0, 0, 1), 0.3)] * 2)
    b = np.array([synth.qaxis((0, 0, 1), 0.5), synth.qaxis((0, 0, 1), 0.3)])
    angle = consistency.quat_angle_between(a, b)
    assert angle[0] == pytest.approx(0.2)
    assert angle[1] == pytest.approx(0.0, abs=1e-12)
    # q and -q are the same rotation.
    assert consistency.quat_angle_between(a, -b)[0] == pytest.approx(0.2)


def test_closed_direction():
    position = np.array([0.0, 0.5, 0.8, 0.4, 0.0])
    request = np.array([1, 2, 2, 1, 1])
    assert consistency.closed_direction(position, request) == 1
    assert consistency.closed_direction(-position, request) == -1
    assert consistency.closed_direction(position, np.zeros(5)) is None
    # only closing rows: compared with all rows
    assert consistency.closed_direction(position, np.array([0, 2, 2, 0, 0])) == 1
