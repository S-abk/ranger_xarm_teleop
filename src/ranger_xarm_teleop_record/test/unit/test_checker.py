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
"""The episode checker on synthetic episodes: every injected fault must be named."""

import copy
import json

import numpy as np
import pytest
from ranger_xarm_teleop_record import checker, metadata, profile
import synth

CHECKS = profile.load_checks()


@pytest.fixture(scope='module')
def clean():
    """One clean episode, never modified (tests deep-copy it)."""
    return synth.build()


def run(ep):
    """Check an episode with the default tolerances."""
    return checker.check_episode(ep, CHECKS)


def flags(report, name):
    """Return the flags of one kind."""
    return [f for f in report['flags'] if f['flag'] == name]


def reasons(report):
    """Return all reasons as one string."""
    return ' | '.join(report['reasons'])


def test_clean_episode_passes(clean):
    report = run(clean)
    assert report['verdict'] == 'PASS', report['reasons']
    assert report['reasons'] == []
    assert report['usable'] is True
    assert report['flags'] == []
    assert report['missing_metadata'] == []
    assert report['consistency']['verdict'] == 'PASS'
    assert report['rtf']['mean'] == pytest.approx(1.0)
    assert report['base']['mode'] == 'parked: no base topics recorded'
    for key in ('verdict', 'reasons', 'flags', 'usable', 'streams', 'link', 'rtf', 'base',
                'consistency', 'missing_metadata'):
        assert key in report
    json.dumps(report, allow_nan=False)


def test_report_has_per_stream_numbers(clean):
    streams = run(clean)['streams']
    assert streams['joint_states']['rate_hz'] == pytest.approx(142.857, rel=1e-3)
    assert streams['state']['rate_wall_hz'] == pytest.approx(50.0)
    assert streams['target']['n_spans'] == 3
    assert streams['camera_gantry']['offset_s'] == pytest.approx(0.0, abs=1e-6)
    assert streams['status']['median_latency_s'] == pytest.approx(0.002, abs=1e-6)
    assert streams['status']['count']['counted'] == streams['status']['count']['recorded']


def test_not_kept_is_not_usable():
    ep = synth.build()
    ep.sidecar['episode']['status'] = 'pending'
    report = run(ep)
    assert report['verdict'] == 'PASS'
    assert report['usable'] is False


@pytest.mark.parametrize('key', list(metadata.REQUIRED))
def test_each_required_key_refuses(clean, key):
    ep = copy.deepcopy(clean)
    node = ep.sidecar
    parts = key.split('.')
    for part in parts[:-1]:
        node = node[part]
    del node[parts[-1]]
    report = run(ep)
    assert report['verdict'] == 'REFUSED'
    assert key in report['missing_metadata']
    assert any(key in r for r in report['reasons'])
    assert report['usable'] is False


def test_missing_camera_key_refuses(clean):
    ep = copy.deepcopy(clean)
    del ep.sidecar['cameras']['gantry_d455']['intrinsics']['k']
    report = run(ep)
    assert report['verdict'] == 'REFUSED'
    assert 'cameras.gantry_d455.intrinsics.k' in report['missing_metadata']


def test_gap_in_joint_states_fails(clean):
    report = run(synth.drop_rows(clean, 'joint_states', range(500, 520)))
    assert report['verdict'] == 'FAIL'
    assert 'joint_states: largest gap' in reasons(report)
    assert report['streams']['joint_states']['gap_ok'] is False


def test_gap_in_images_fails(clean):
    report = run(synth.drop_rows(clean, 'camera_gantry', range(100, 106)))
    assert report['verdict'] == 'FAIL'
    assert 'camera_gantry: largest gap' in reasons(report)
    assert 'camera_wrist' not in reasons(report)


def test_rate_drop_fails(clean):
    # The publisher itself sent half as many: the episode node counted half as many too.
    ep = synth.drop_rows(clean, 'joint_command', range(0, 428, 2))
    if 'joint_command' in ep.sidecar['counts']:
        ep.sidecar['counts']['joint_command']['count'] -= 214
    report = run(ep)
    assert report['verdict'] == 'FAIL'
    assert 'joint_command: rate' in reasons(report)
    assert report['streams']['joint_command']['rate_ok'] is False
    assert flags(report, 'DATA_LOST') == []


def test_actions_lost_by_the_recorder_are_data_lost(clean):
    ep = synth.drop_rows(clean, 'target', range(40, 43))          # three actions in a row
    report = run(ep)
    assert report['verdict'] == 'FAIL'
    assert 'target: ' in flags(report, 'DATA_LOST')[0]['reason']


def test_short_gate_span_is_not_rate_checked(clean):
    # A span shorter than min_gate_span_s: the rate is not judged, the gaps still are.
    ep = copy.deepcopy(clean)
    state = ep.streams['state']
    engaged = np.zeros(len(state), dtype=bool)
    engaged[50:60] = True           # 0.2 s
    state.fields['clutch_engaged'] = engaged
    target = ep.streams['target']
    keep = (target.stamp >= state.stamp[50] - 1e-6) & (target.stamp <= state.stamp[59] + 1e-6)
    ep.streams['target'] = synth.select_rows(target, keep)
    stream = run(ep)['streams']['target']
    assert stream['rate_hz'] is None and stream['gap_ok'] is True and stream['n_spans'] == 1
    keep = np.ones(10, dtype=bool)
    keep[2:8] = False                # 0.12 s hole inside the span
    ep.streams['target'] = synth.select_rows(ep.streams['target'], keep)
    assert run(ep)['streams']['target']['gap_ok'] is False


def test_offset_injection_fails(clean):
    report = run(synth.shift_recv(clean, 'joint_states', 0.05))
    assert report['verdict'] == 'FAIL'
    assert 'joint_states: offset to the action stream' in reasons(report)
    assert 'ONE_HOST' not in reasons(report)
    assert report['streams']['joint_states']['offset_s'] == pytest.approx(0.05, abs=1e-6)


def test_robot_stream_on_another_clock_fails_one_host(clean):
    report = run(synth.shift_stamps(clean, 'joint_states', 1000.0))
    assert report['verdict'] == 'FAIL'
    one_host = [r for r in report['reasons'] if r.startswith('ONE_HOST:')]
    assert len(one_host) == 1 and 'joint_states' in one_host[0]
    assert 'robot-side streams on more than one clock' in one_host[0]
    assert 'state' not in one_host[0].replace('joint_states', '')


def test_wall_clock_stream_mapped_with_clock_pairs():
    # Receive time and mapped stamp agree at rtf 0.6, so the wall-clock streams are in domain.
    report = run(synth.build(rtf=0.6))
    assert report['streams']['status']['domain_ok'] is True
    assert report['streams']['joy']['median_latency_s'] == pytest.approx(0.002, abs=1e-6)
    assert report['streams']['state']['rate_wall_hz'] == pytest.approx(30.0, rel=1e-3)


def test_clock_pairs_extrapolate_at_the_ends(clean):
    ep = synth.drop_rows(clean, 'episode_state', [0, 1, 2, 58, 59, 60])
    report = run(ep)
    assert report['verdict'] == 'PASS', report['reasons']
    assert report['streams']['status']['median_latency_s'] == pytest.approx(0.002, abs=1e-6)


def test_extrap_interp_extrapolates_linearly():
    x = np.array([-1.0, 0.0, 0.5, 2.0, 5.0])
    y = checker.extrap_interp(x, [0.0, 1.0, 2.0], [10.0, 12.0, 16.0])
    assert y.tolist() == pytest.approx([8.0, 10.0, 11.0, 16.0, 28.0])
    with pytest.raises(ValueError):
        checker.extrap_interp(x, [0.0], [1.0])


def test_no_clock_pairs_fails_wall_streams(clean):
    ep = copy.deepcopy(clean)
    del ep.streams['episode_state']
    report = run(ep)
    assert report['verdict'] == 'FAIL'
    assert 'NO_CLOCK_PAIRS' in reasons(report)
    assert 'status' in [r for r in report['reasons'] if 'NO_CLOCK_PAIRS' in r][0]
    assert report['rtf']['mean'] is None


def test_non_monotonic_stamps_fail_and_are_not_repaired(clean):
    ep = copy.deepcopy(clean)
    stamp = ep.streams['target'].stamp.copy()
    stamp[10], stamp[11] = stamp[11], stamp[10]
    ep.streams['target'].stamp = stamp
    before = synth.snapshot(ep)
    report = run(ep)
    assert report['verdict'] == 'FAIL'
    assert report['streams']['target']['non_monotonic'] == 1
    assert 'target: 1 stamps earlier' in reasons(report)
    assert synth.same(before, synth.snapshot(ep))


def test_counts_shortfall_is_data_lost(clean):
    ep = synth.drop_rows(clean, 'joint_states', [500, 501, 502])
    report = run(ep)
    assert report['verdict'] == 'FAIL'
    lost = flags(report, 'DATA_LOST')
    assert len(lost) == 1 and 'joint_states' in lost[0]['reason']
    assert all(r.startswith('DATA_LOST') for r in report['reasons'])
    assert report['streams']['joint_states']['count']['counted'] \
        - report['streams']['joint_states']['count']['recorded'] == 3


def test_dropped_image_with_counted_camera_info_is_data_lost(clean):
    report = run(synth.drop_rows(clean, 'camera_gantry', [100]))
    assert report['verdict'] == 'FAIL'
    lost = flags(report, 'DATA_LOST')
    assert len(lost) == 1 and lost[0]['reason'].startswith('camera_gantry:')
    assert all(r.startswith('DATA_LOST') for r in report['reasons'])
    info = report['streams']['camera_gantry_info']['count']
    assert info['recorded'] == info['counted'] and info['recorded_camera_gantry'] == 360


def test_counts_recorded_above_counted_is_only_reported(clean):
    ep = copy.deepcopy(clean)
    ep.sidecar['counts']['joint_states']['count'] -= 1
    report = run(ep)
    assert report['verdict'] == 'PASS'
    assert flags(report, 'DATA_LOST') == []
    count = report['streams']['joint_states']['count']
    assert count['recorded'] == count['counted'] + 1


def test_missing_required_stream_fails(clean):
    ep = copy.deepcopy(clean)
    del ep.streams['camera_wrist']
    del ep.streams['joint_command']
    report = run(ep)
    assert report['verdict'] == 'FAIL'
    assert 'camera_wrist: required stream missing' in reasons(report)
    assert 'joint_command: required stream missing' in reasons(report)
    assert report['streams']['camera_wrist']['present'] is False


def test_gated_stream_absent_fails_even_when_not_required(clean):
    ep = copy.deepcopy(clean)
    del ep.streams['gripper_command']
    assert ep.profile['streams']['gripper_command']['required'] is False
    report = run(ep)
    assert report['verdict'] == 'FAIL'
    assert 'gripper_command: gated stream absent' in reasons(report)


def test_state_without_rows_does_not_crash(clean):
    ep = copy.deepcopy(clean)
    del ep.streams['state']
    report = run(ep)
    assert report['verdict'] == 'FAIL'
    assert 'state' in reasons(report)


@pytest.mark.parametrize('hold', [3, 4] + list(range(6, 21)))
def test_fault_release_flag_names_the_hold(clean, hold):
    ep = copy.deepcopy(clean)
    state = ep.streams['state']
    row = int(np.searchsorted(state.stamp, synth.T0 + 4.0 + 1e-6)) + 0
    assert not state.fields['clutch_engaged'][row]
    reason_field = state.fields['hold_reason'].copy()
    reason_field[row] = hold
    state.fields['hold_reason'] = reason_field
    report = run(ep)
    found = flags(report, 'FAULT_RELEASE')
    assert len(found) == 1
    assert found[0]['reason'] == checker.HOLD_NAMES[hold]
    assert found[0]['time'] == pytest.approx(state.stamp[row])
    assert report['verdict'] == 'PASS'


def test_operator_release_is_not_a_fault(clean):
    assert flags(run(clean), 'FAULT_RELEASE') == []


def test_fault_release_in_built_episode():
    report = run(synth.build(release_reason={1: 6}))
    found = flags(report, 'FAULT_RELEASE')
    assert [f['reason'] for f in found] == ['HOLD_INPUT_STALE']
    assert found[0]['time'] == pytest.approx(synth.T0 + 8.02)


def test_limit_bound_with_limited_cycles_still_passes():
    report = run(synth.build(limited_share=0.3))
    found = flags(report, 'LIMIT_BOUND')
    assert len(found) == 1
    assert report['verdict'] == 'PASS', report['reasons']
    assert report['limits']['bound_share'] > 0.25
    assert report['limits']['per_limit']['LIMIT_KEEP_OUT'] == pytest.approx(
        report['limits']['bound_share'])
    assert report['consistency']['arm']['n_limited'] > 0


def test_limit_bound_absent_when_clean(clean):
    report = run(clean)
    assert flags(report, 'LIMIT_BOUND') == []
    assert report['limits']['bound_share'] == 0.0
    assert report['limits']['per_limit'] == {'LIMIT_NONE': 1.0}


def test_disarmed_flag_from_the_state_stream(clean):
    ep = copy.deepcopy(clean)
    armed = ep.streams['state'].fields['armed'].copy()
    armed[450:] = False
    ep.streams['state'].fields['armed'] = armed
    report = run(ep)
    found = flags(report, 'DISARMED')
    assert len(found) == 1
    assert found[0]['time'] == pytest.approx(ep.streams['state'].stamp[450])


def test_disarmed_flag_from_the_stop_reason(clean):
    ep = copy.deepcopy(clean)
    ep.sidecar['episode']['stop_reason'] = 'disarmed'
    assert len(flags(run(ep), 'DISARMED')) == 1
    assert flags(run(clean), 'DISARMED') == []


def test_low_rtf_flag():
    report = run(synth.build(rtf=0.6))
    assert len(flags(report, 'LOW_RTF')) == 1
    assert report['rtf']['mean'] == pytest.approx(0.6)
    assert report['rtf']['min_1s'] == pytest.approx(0.6)
    assert report['verdict'] == 'PASS', report['reasons']


def test_low_rtf_skipped_when_null():
    ep = synth.build(rtf=0.6)
    ep.profile['low_rtf'] = None
    report = run(ep)
    assert flags(report, 'LOW_RTF') == []
    assert report['rtf']['mean'] == pytest.approx(0.6)


def test_base_parked_with_topics():
    report = run(synth.build(base='parked'))
    assert report['base']['mode'] == 'parked: base topics present, no motion'
    assert flags(report, 'BASE_MOVED') == []
    assert report['base']['streams']['odom']['travel_m'] == 0.0


def test_base_moved_by_odometry_and_cmd_vel():
    report = run(synth.build(base='moved'))
    assert report['base']['mode'] == 'moved'
    assert len(flags(report, 'BASE_MOVED')) == 1
    assert report['base']['streams']['odom']['travel_m'] == pytest.approx(0.1)
    assert report['base']['streams']['cmd_vel']['nonzero_rows'] == 10
    assert report['verdict'] == 'PASS', report['reasons']


def test_base_moved_by_cmd_vel_alone():
    ep = synth.build(base='parked')
    twist = ep.streams['cmd_vel'].fields['twist'].copy()
    twist[5, 5] = 0.2
    ep.streams['cmd_vel'].fields['twist'] = twist
    report = run(ep)
    assert len(flags(report, 'BASE_MOVED')) == 1
    assert report['base']['streams']['odom']['travel_m'] == 0.0


def test_base_moved_by_rotation_alone():
    ep = synth.build(base='parked')
    pose = ep.streams['odom'].fields['pose'].copy()
    pose[20:, 3:] = synth.qaxis((0, 0, 1), 0.05)
    ep.streams['odom'].fields['pose'] = pose
    report = run(ep)
    assert len(flags(report, 'BASE_MOVED')) == 1
    assert report['base']['streams']['odom']['yaw_change_rad'] == pytest.approx(0.05)


def test_link_numbers_are_always_reported(clean):
    link = run(clean)['link']
    assert link['input_age_p99_s'] < 0.1
    assert link['seq_gaps'] == 0 and link['reconnects'] == 0
    assert link['xr_period_ms'] == pytest.approx(1000.0 / 60.0)
    assert link['xr_jitter_p99_ms'] == pytest.approx(0.0, abs=1e-6)


def test_link_degraded_by_sequence_gap(clean):
    ep = copy.deepcopy(clean)
    seq = ep.streams['status'].fields['seq'].copy()
    seq[300:] += 5
    ep.streams['status'].fields['seq'] = seq
    report = run(ep)
    found = flags(report, 'LINK_DEGRADED')
    assert len(found) == 1 and 'sequence gaps' in found[0]['reason']
    assert report['link']['seq_gaps'] == 1
    assert report['verdict'] == 'PASS'


def test_link_degraded_by_reconnect_and_session_change(clean):
    ep = copy.deepcopy(clean)
    status = ep.streams['status']
    stamp = status.stamp.copy()
    stamp[400:] += 3.0
    status.stamp = stamp
    ep.sidecar['counts']['status']['window'][1] += 10.0
    session = status.fields['session_id'].copy()
    session[500:] = 2
    status.fields['session_id'] = session
    report = run(ep)
    assert report['link']['status_stamp_gaps'] == 1
    assert report['link']['session_changes'] == 1
    assert report['link']['reconnects'] == 2
    assert 'reconnects' in flags(report, 'LINK_DEGRADED')[0]['reason']


def test_link_degraded_by_input_age(clean):
    ep = copy.deepcopy(clean)
    ep.streams['state'].fields['input_age'] = np.full(len(ep.streams['state']), 0.5)
    report = run(ep)
    assert 'input age' in flags(report, 'LINK_DEGRADED')[0]['reason']
    assert report['link']['input_age_max_s'] == pytest.approx(0.5)


def test_sidecar_flags_are_copied(clean):
    ep = copy.deepcopy(clean)
    ep.sidecar['episode']['flags'] = [
        {'flag': 'DISK_LOW', 'reason': 'free space below the limit', 'time': 5.0},
        {'flag': 'TELEOP_STATE_LOST', 'reason': 'state stopped', 'time': None}]
    report = run(ep)
    assert [f['flag'] for f in report['flags']] == ['DISK_LOW', 'TELEOP_STATE_LOST']
    assert report['flags'][0]['time'] == 5.0 and report['flags'][1]['time'] is None
    # TELEOP_STATE_LOST is a fail flag (checks.yaml); DISK_LOW alone is not
    assert report['verdict'] == 'FAIL'
    assert any(r.startswith('TELEOP_STATE_LOST') for r in report['reasons'])
    ep.sidecar['episode']['flags'] = ep.sidecar['episode']['flags'][:1]
    assert run(ep)['verdict'] == 'PASS'


@pytest.mark.parametrize('where', ['start', 'end'])
def test_a_stall_at_the_start_or_end_of_the_window_is_a_gap(clean, where):
    ep = copy.deepcopy(clean)
    w0, w1 = ep.streams['state'].stamp[0], ep.streams['state'].stamp[-1]
    for name in ('camera_wrist', 'camera_wrist_info'):
        s = ep.streams[name]
        keep = s.stamp > w0 + 0.5 if where == 'start' else s.stamp < w1 - 0.5
        ep.streams[name] = type(s)(name=s.name, topic=s.topic, kind=s.kind, stamp=s.stamp[keep],
                                   recv=s.recv[keep],
                                   fields={k: v[keep] for k, v in s.fields.items()},
                                   frame_id=[f for f, k in zip(s.frame_id, keep) if k],
                                   labels={k: [x for x, kk in zip(v, keep) if kk]
                                           for k, v in s.labels.items()},
                                   joint_names=s.joint_names, has_header=s.has_header)
    for key in ('camera_wrist_info',):
        ep.sidecar['counts'].pop(key, None)
    report = run(ep)
    assert report['verdict'] == 'FAIL'
    assert any('camera_wrist' in r and 'gap' in r for r in report['reasons']), report['reasons']


def test_consistency_failure_fails_the_episode(clean):
    ep = copy.deepcopy(clean)
    ep.streams['target'].frame_id = [synth.TOOL_FRAME] * len(ep.streams['target'])
    report = run(ep)
    assert report['verdict'] == 'FAIL'
    assert report['consistency']['verdict'] == 'FAIL'
    assert 'CONSISTENCY' in reasons(report)


def test_checker_never_mutates_the_episode(clean):
    for ep in (copy.deepcopy(clean), synth.drop_rows(clean, 'joint_states', range(500, 520)),
               synth.build(limited_share=0.3, base='moved')):
        before = synth.snapshot(ep)
        run(ep)
        assert synth.same(before, synth.snapshot(ep))


def test_report_is_strict_json_even_when_failing(clean):
    ep = synth.shift_stamps(clean, 'joint_states', 1000.0)
    report = run(ep)
    json.dumps(report, allow_nan=False)
