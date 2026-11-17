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
"""Export: strictly causal holds (no negative age) and ticks on one camera's frames."""

import numpy as np
import pytest
from ranger_xarm_teleop_record import export, profile
from ranger_xarm_teleop_record.streams import Stream
import synth

CHECKS = profile.load_checks()


def live_like():
    """
    Return an episode whose measured gripper sample is newer than its state cycle.

    Seen in Gazebo: the teleop node publishes a cycle stamped before the newest joint state
    it read, so gripper_measured_stamp is up to a joint period after the cycle stamp. Also
    a cycle without an input frame (input_age -1).
    """
    ep = synth.build()
    st, js = ep.streams['state'], ep.streams['joint_states']
    after = np.minimum(np.searchsorted(js.stamp, st.stamp + 1e-9, side='right'), len(js) - 1)
    ahead = np.arange(len(st)) % 3 == 0
    stamp = st.fields['gripper_measured_stamp'].copy()
    stamp[ahead] = js.stamp[after[ahead]]
    assert np.any(stamp > st.stamp)
    st.fields['gripper_measured_stamp'] = stamp
    st.fields['gripper_measured'] = np.where(ahead, js.fields['position'][after, 6],
                                             st.fields['gripper_measured'])
    st.fields['input_age'][5] = -1.0
    return ep


def jittered_camera(ep, name, first_offset=-0.010, duplicate_at=None):
    """Replace ``name`` with frames 32-34 ms apart (ms stamps), starting before the state."""
    st = ep.streams[name]
    span = float(np.max(ep.streams['state'].stamp) - np.min(ep.streams['state'].stamp))
    steps = np.tile([0.032, 0.033, 0.034, 0.033, 0.034, 0.032], 200)
    stamps = synth.T0 + first_offset + np.concatenate([[0.0], np.cumsum(steps)])
    stamps = np.round(stamps[stamps <= synth.T0 + span], 3)
    if duplicate_at is not None:
        stamps = np.insert(stamps, duplicate_at, stamps[duplicate_at])
    n = len(stamps)
    ep.streams[name] = Stream(
        name=name, topic=st.topic, kind=st.kind, stamp=stamps, recv=stamps + synth.LATENCY,
        fields={k: np.repeat(v[:1], n, axis=0) for k, v in st.fields.items()},
        labels={k: [v[0]] * n for k, v in st.labels.items()})
    return ep


def ages(arrays):
    """Return every exported age array, by name."""
    return {k: v for k, v in arrays.items() if k.endswith('_age')}


@pytest.mark.parametrize('tick_stream', [None, 'camera_wrist'])
def test_no_age_in_any_field_is_negative(tick_stream):
    ep = jittered_camera(live_like(), 'camera_wrist')
    arrays, _ = export.build_export(ep, CHECKS, tick_stream=tick_stream)
    found = ages(arrays)
    assert {'state_age', 'gripper_measured_age', 'measured_row_age', 'input_age',
            'joint_state_age', 'action_pose_age', 'image_camera_wrist_age'} <= set(found)
    for name, value in found.items():
        finite = value[np.isfinite(value)]
        assert len(finite), name
        assert np.all(finite >= 0.0), (name, finite.min())


def test_measured_sample_is_never_later_than_the_tick():
    ep = live_like()
    st = ep.streams['state']
    arrays, _ = export.build_export(ep, CHECKS)
    t = arrays['t']
    # Every exported measured opening is one whose own stamp is not after the tick.
    ok = arrays['gripper_measured_valid']
    held_stamp = t - arrays['gripper_measured_age']
    assert np.all(held_stamp[ok] <= t[ok] + 1e-12)
    for k in np.flatnonzero(ok)[:200]:
        row = np.flatnonzero(np.isclose(st.fields['gripper_measured_stamp'], held_stamp[k],
                                         rtol=0.0, atol=1e-9))
        assert len(row)
        assert arrays['gripper_measured'][k] in st.fields['gripper_measured'][row]
    # And where the cycle's own sample was later than the tick, an earlier row is used.
    assert np.any(arrays['measured_row_age'] > arrays['state_age'] + 1e-9)


def test_a_sample_one_ms_after_the_tick_is_never_held():
    stamps = np.array([0.0, 1.0 / 30.0 + 0.001, 2.0 / 30.0])
    t = np.arange(3) / 30.0
    idx = export.hold_index(stamps, t)
    assert list(idx) == [0, 0, 2]
    assert np.all(export._age(stamps, idx, t) >= 0.0)
    # Float rounding of the grid (well inside TIME_EPS) is held, with age 0, not negative.
    noisy = t + np.array([0.0, 2e-13, 0.0])
    idx = export.hold_index(noisy, t)
    assert list(idx) == [0, 1, 2]
    assert export._age(noisy, idx, t)[1] == 0.0


def test_input_age_without_an_input_frame_is_nan():
    arrays, meta = export.build_export(live_like(), CHECKS)
    assert np.isnan(arrays['input_age']).sum() >= 1
    info = {f['name']: f for f in meta['fields']}
    assert 'NaN' in info['input_age']['description']


def test_ticks_on_a_camera_use_every_frame_exactly_once():
    ep = jittered_camera(synth.build(), 'camera_wrist', duplicate_at=40)
    n = len(ep.streams['camera_wrist'])
    arrays, meta = export.build_export(ep, CHECKS, tick_stream='camera_wrist')
    index = arrays['image_camera_wrist_index']
    assert len(arrays['t']) == n
    assert sorted(index) == list(range(n))                   # each frame once, none skipped
    assert np.all(arrays['image_camera_wrist_age'] == 0.0)
    assert np.all(np.diff(arrays['t']) >= 0.0)
    ticks = meta['ticks']
    assert ticks['mode'] == 'stream' and ticks['stream'] == 'camera_wrist'
    assert ticks['ticks_before_first_state'] == 1
    assert 0.0 <= ticks['dt_min_s'] <= 0.032 + 1e-9 and ticks['dt_max_s'] >= 0.034 - 1e-9
    info = meta['images']['camera_wrist']
    assert (info['frames_used'], info['frames_unused'], info['ticks_repeating_frame']) == \
        (n, 0, 0)


def test_a_tick_before_the_first_state_holds_nothing():
    ep = jittered_camera(synth.build(), 'camera_wrist')
    arrays, _ = export.build_export(ep, CHECKS, tick_stream='camera_wrist')
    first = arrays['t'] < np.min(ep.streams['state'].stamp)
    assert first.sum() == 1
    # No wrap-around to the last state row: nothing is held, the tick is invalid.
    assert not arrays['tick_valid'][first].any()
    assert not arrays['clutch_engaged'][first].any()
    assert not arrays['action_valid'][first].any()
    assert not arrays['gripper_action_valid'][first].any()
    assert np.isnan(arrays['state_age'][first]).all()
    assert np.isnan(arrays['gripper_command'][first]).all()
    assert (arrays['hold_reason'][first] == -1).all()


def test_a_fixed_grid_against_a_jittered_camera_skips_and_repeats_frames():
    ep = jittered_camera(synth.build(), 'camera_wrist')
    _, meta = export.build_export(ep, CHECKS)
    info = meta['images']['camera_wrist']
    assert info['frames_unused'] > 0 or info['ticks_repeating_frame'] > 0
    assert info['frames_used'] + info['frames_unused'] == info['frames_recorded']
    assert meta['ticks']['mode'] == 'fixed' and meta['ticks']['rate_hz'] == 30.0


def test_tick_stream_must_be_a_recorded_image_stream():
    ep = synth.build()
    with pytest.raises(ValueError):
        export.build_export(ep, CHECKS, tick_stream='joint_states')
    with pytest.raises(ValueError):
        export.build_export(ep, CHECKS, tick_stream='no_such_stream')
    with pytest.raises(ValueError):
        export.build_export(ep, CHECKS, rate_hz=15.0, tick_stream='camera_wrist')


def test_absent_fields_are_named_in_the_export():
    ep = synth.build()
    js = ep.streams['joint_states']
    js.fields['effort'] = np.full_like(js.fields['effort'], np.nan)
    arrays, meta = export.build_export(ep, CHECKS)
    assert meta['fields_absent'] == {'joint_states': ['effort']}
    assert np.isnan(arrays['joint_effort']).all()
