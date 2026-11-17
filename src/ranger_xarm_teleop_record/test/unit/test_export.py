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
"""The fixed-rate export: causal hold, ages, validity flags and the written files."""

import copy
import json
import os
import socket

import numpy as np
from PIL import Image
import pytest
from ranger_xarm_teleop_record import checker, export, profile
import synth

CHECKS = profile.load_checks()
T = int(30 * 12) + 1


@pytest.fixture(scope='module')
def clean():
    """One clean episode, never modified (tests deep-copy it)."""
    return synth.build()


@pytest.fixture(scope='module')
def built(clean):
    """Return the export of the clean episode."""
    return export.build_export(clean, CHECKS)


def ticks(arrays, lo, hi):
    """Return the tick indices whose time lies in ``[lo, hi]`` seconds after the start."""
    u = arrays['t'] - synth.T0
    return np.flatnonzero((u >= lo) & (u <= hi))


def test_shapes_and_timeline(built):
    arrays, meta = built
    assert meta['rate_hz'] == 30.0 and meta['n_ticks'] == T
    assert arrays['t'].shape == (T,)
    assert arrays['t'][0] == synth.T0 and arrays['t'][-1] == pytest.approx(synth.T0 + 12.0)
    assert np.diff(arrays['t']) == pytest.approx(1.0 / 30.0)
    assert arrays['t_wall'][0] == pytest.approx(synth.WALL0)
    assert arrays['t_wall'][-1] == pytest.approx(synth.WALL0 + 12.0)
    constants = {'joint_names', 'joint_command_names'}
    for name, array in arrays.items():
        if name not in constants:
            assert array.shape[0] == T, name
        assert array.dtype != object, name
    assert arrays['joint_position'].shape == (T, 7)
    assert arrays['joint_velocity'].shape == (T, 7)
    assert arrays['joint_effort'].shape == (T, 7)
    assert arrays['joint_command'].shape == (T, 6)
    assert arrays['action_pose'].shape == (T, 7)
    assert arrays['tool_pose_measured'].shape == (T, 7)
    assert arrays['joint_names'].tolist() == synth.JOINTS
    assert meta['constants']['joint_names'] == synth.JOINTS
    assert meta['constants']['joint_command_names'] == synth.JOINTS[:6]
    for name in synth.CAMERAS:
        assert arrays[f'image_{name}_index'].dtype == np.int64
        assert arrays[f'image_{name}_age'].shape == (T,)
    for name in ('clutch_engaged', 'action_valid', 'gripper_action_valid', 'damped', 'armed',
                 'tick_valid', 'tool_pose_measured_valid', 'joint_command_valid'):
        assert arrays[name].dtype == bool, name
    # Streams on the host wall clock are inputs, not exported.
    assert not any('status' in k or 'joy' in k for k in arrays)
    assert 'base_odom_pose' not in arrays


def test_rate_argument(clean):
    arrays, meta = export.build_export(clean, CHECKS, rate_hz=10.0)
    assert len(arrays['t']) == 121 and meta['rate_hz'] == 10.0


def test_values_are_the_held_samples(clean, built):
    arrays, _ = built
    state = clean.streams['state']
    joints = clean.streams['joint_states']
    for k in (0, 17, 100, 250, 360):
        t = arrays['t'][k]
        row = np.flatnonzero(state.stamp <= t + 1e-9)[-1]
        assert arrays['state_age'][k] == pytest.approx(t - state.stamp[row])
        assert arrays['armed'][k] == state.fields['armed'][row]
        assert arrays['tool_pose_measured'][k] == pytest.approx(
            state.fields['measured_tool_pose'][row])
        jrow = np.flatnonzero(joints.stamp <= t + 1e-9)[-1]
        assert arrays['joint_position'][k] == pytest.approx(joints.fields['position'][jrow])
        assert arrays['joint_state_age'][k] == pytest.approx(t - joints.stamp[jrow])
        assert arrays['gripper_measured_age'][k] == pytest.approx(
            t - state.fields['gripper_measured_stamp'][row])


def test_ages_are_within_one_source_period(built):
    arrays, _ = built
    assert np.all(arrays['state_age'] >= 0.0) and np.all(arrays['state_age'] <= 0.02 + 1e-9)
    assert np.all(arrays['joint_state_age'] >= 0.0)
    assert np.all(arrays['joint_state_age'] <= 1.0 / 142.857 + 1e-6)
    for name in synth.CAMERAS:
        age = arrays[f'image_{name}_age']
        assert np.all(age >= -1e-9) and np.all(age <= 1.0 / 30.0 + 1e-9)
    assert np.all(arrays['tick_valid'])


def test_nothing_comes_from_the_future(clean):
    arrays, _ = export.build_export(clean, CHECKS)
    k = 150
    t_k = arrays['t'][k]
    ep = copy.deepcopy(clean)
    for stream in ep.streams.values():
        later = stream.stamp > t_k + 1e-6
        for key, value in stream.fields.items():
            changed = value.copy()
            if changed.dtype == bool:
                changed[later] = ~changed[later]
            else:
                changed[later] = changed[later] + 123
            stream.fields[key] = changed
        stream.stamp = stream.stamp.copy()
    perturbed, _ = export.build_export(ep, CHECKS)
    later = slice(k + 5, None)
    assert not np.array_equal(arrays['joint_position'][later], perturbed['joint_position'][later])
    for name, array in arrays.items():
        if name.endswith('_names'):
            continue
        assert np.array_equal(array[:k + 1], perturbed[name][:k + 1], equal_nan=True), name


def test_hold_index_is_causal_and_stable():
    stamp = np.array([0.0, 1.0, 0.5, 2.0])      # one stamp out of order
    t = np.array([-0.1, 0.0, 0.49, 0.5, 1.4, 2.0, 9.0])
    assert export.hold_index(stamp, t).tolist() == [-1, 0, 0, 2, 1, 3, 3]


def test_released_cycles_hold_with_no_action(built):
    arrays, _ = built
    for lo, hi in ((0.0, 0.9), (4.1, 4.9), (8.1, 8.9), (11.6, 12.0)):
        sel = ticks(arrays, lo, hi)
        assert len(sel) > 5
        assert not arrays['clutch_engaged'][sel].any()
        assert not arrays['action_valid'][sel].any()
        assert not arrays['joint_command_valid'][sel].any()
    for lo, hi in ((1.1, 3.9), (5.1, 7.9), (9.1, 11.4)):
        sel = ticks(arrays, lo, hi)
        assert arrays['clutch_engaged'][sel].all()
        assert arrays['action_valid'][sel].all()
        assert arrays['joint_command_valid'][sel].all()
    # No target yet: NaN; after the first span the last target is held, with its age.
    assert np.isnan(arrays['action_pose'][ticks(arrays, 0.0, 0.9)]).all()
    held = ticks(arrays, 4.5, 4.9)
    assert not np.isnan(arrays['action_pose'][held]).any()
    assert (arrays['action_pose_age'][held] > 0.4 * 0.5).all()
    assert arrays['action_pose_age'][ticks(arrays, 2.0, 3.0)].max() <= 0.02 + 1e-9


def test_action_valid_false_when_the_cycle_has_no_target(clean):
    dropped = synth.drop_rows(clean, 'target', range(60, 70))
    arrays, _ = export.build_export(dropped, CHECKS)
    stamps = clean.streams['target'].stamp[60:70]
    lo, hi = stamps[0] - synth.T0, stamps[-1] - synth.T0
    inside = ticks(arrays, lo + 0.04, hi - 0.02)
    assert len(inside) > 0
    assert arrays['clutch_engaged'][inside].all()
    assert not arrays['action_valid'][inside].any()
    # tick_valid is about observations, not actions.
    assert arrays['tick_valid'].all()


def test_gripper_action_valid_is_false_for_a_measured_stand_in(built, clean):
    arrays, _ = built
    stand_in = ticks(arrays, 0.0, 1.4)
    assert (arrays['gripper_source'][stand_in] == 3).all()
    assert not arrays['gripper_action_valid'][stand_in].any()
    assert arrays['gripper_command'][stand_in] == pytest.approx(0.0, abs=1e-12)
    active = ticks(arrays, 1.6, 3.4)
    assert arrays['gripper_action_valid'][active].all()
    assert (arrays['gripper_request'][active] == 2).all()
    between = ticks(arrays, 3.6, 5.4)
    assert not arrays['gripper_action_valid'][between].any()
    assert arrays['gripper_command'][between] == pytest.approx(0.8)
    assert arrays['gripper_measured'][between][-1] == pytest.approx(0.8)
    # And the flag follows the state stream's gripper_active, whatever the source says.
    ep = copy.deepcopy(clean)
    ep.streams['state'].fields['gripper_active'] = np.zeros(len(ep.streams['state']), dtype=bool)
    off, _ = export.build_export(ep, CHECKS)
    assert not off['gripper_action_valid'].any()
    # Commanding from the measured opening (source MEASURED while active) is masked too.
    ep = copy.deepcopy(clean)
    ep.streams['state'].fields['gripper_reference_source'] = np.full(
        len(ep.streams['state']), 3)
    measured, _ = export.build_export(ep, CHECKS)
    assert not measured['gripper_action_valid'].any()


def test_tick_valid_false_across_a_joint_state_gap(clean):
    dropped = synth.drop_rows(clean, 'joint_states', range(500, 560))
    arrays, _ = export.build_export(dropped, CHECKS)
    gap_lo = clean.streams['joint_states'].stamp[499] - synth.T0
    gap_hi = clean.streams['joint_states'].stamp[560] - synth.T0
    inside = ticks(arrays, gap_lo + 0.02, gap_hi - 0.001)
    assert len(inside) > 5
    assert not arrays['tick_valid'][inside].any()
    assert arrays['tick_valid'][ticks(arrays, 0.0, gap_lo - 0.02)].all()
    assert arrays['tick_valid'][ticks(arrays, gap_hi + 0.05, 12.0)].all()
    assert arrays['joint_state_age'][inside].max() > 0.2


def test_tick_valid_false_across_an_image_gap(clean):
    dropped = synth.drop_rows(clean, 'camera_wrist', range(100, 130))
    arrays, _ = export.build_export(dropped, CHECKS)
    lo = clean.streams['camera_wrist'].stamp[100] - synth.T0
    hi = clean.streams['camera_wrist'].stamp[130] - synth.T0
    inside = ticks(arrays, lo + 0.1, hi - 0.05)
    assert len(inside) > 5
    assert not arrays['tick_valid'][inside].any()
    assert arrays['image_camera_wrist_age'][inside].max() > 0.5
    # The image held is still the last real one (nothing invented).
    assert (arrays['image_camera_wrist_index'][inside] == 99).all()


def test_image_index_is_minus_one_before_the_first_image(clean):
    ep = synth.drop_rows(clean, 'camera_gantry', range(0, 10))
    arrays, _ = export.build_export(ep, CHECKS)
    assert (arrays['image_camera_gantry_index'][:10] == -1).all()
    assert np.isnan(arrays['image_camera_gantry_age'][:10]).all()
    assert not arrays['tick_valid'][:10].any()
    assert arrays['image_camera_gantry_index'][10] == 0     # rows of the recorded stream


def test_base_odometry_is_exported_when_recorded():
    arrays, meta = export.build_export(synth.build(base='moved'), CHECKS)
    assert arrays['base_odom_pose'].shape == (T, 7)
    assert arrays['base_odom_pose'][-1, 0] == pytest.approx(0.1, abs=0.01)
    assert arrays['base_odom_age'].shape == (T,)
    assert meta['base_odometry_stream'] == 'odom'


def test_meta_describes_the_export(clean):
    report = checker.check_episode(clean, CHECKS)
    arrays, meta = export.build_export(clean, CHECKS, verdict=report)
    assert meta['format'] == 'ranger_xarm_teleop.export/1'
    assert meta['rule'] == export.RULE
    assert 'causal zero-order hold' in meta['rule'] and 'nothing is interpolated' in meta['rule']
    assert meta['frames']['base'] == synth.BASE_FRAME
    assert 'xyzw' in meta['frames']['poses']
    by_name = {f['name']: f for f in meta['fields']}
    assert set(by_name) == set(arrays)
    assert by_name['action_pose']['shape'] == [T, 7]
    assert by_name['action_pose']['frame'] == synth.BASE_FRAME
    assert by_name['t']['unit'] == 's' and by_name['joint_position']['dtype'] == 'float64'
    assert by_name['image_camera_gantry_index']['dtype'] == 'int64'
    assert meta['episode'] == {'index': 1, 'status': 'kept', 'session': 'test-session',
                               'labels': {'task': 'test-task', 'operator': 'test-operator'}}
    assert meta['source']['bag_file'] == 'bag/0001/0001_0.mcap'
    assert meta['checker']['verdict'] == 'PASS' and meta['checker']['usable'] is True
    assert meta['images']['camera_gantry']['path'] == 'images/camera_gantry/<row:06d>.png'
    arrays2, meta2 = export.build_export(clean, CHECKS, verdict='FAIL')
    assert meta2['checker'] == {'verdict': 'FAIL'}
    assert export.build_export(clean, CHECKS)[1]['checker'] is None


def test_absolute_bag_path_is_reduced_to_the_file_name(clean):
    ep = copy.deepcopy(clean)
    ep.sidecar['episode']['file'] = '/somewhere/else/0001_0.mcap'
    assert export.build_export(ep, CHECKS)[1]['source']['bag_file'] == '0001_0.mcap'


def strings(node):
    """Yield every string in a JSON document."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from strings(value)


def test_write_export_round_trip(clean, tmp_path):
    report = checker.check_episode(clean, CHECKS)
    arrays, meta = export.build_export(clean, CHECKS, verdict=report)
    before = synth.snapshot(clean)
    out = tmp_path / 'export' / '0001'
    written = export.write_export(str(out), arrays, meta, clean)
    assert synth.same(before, synth.snapshot(clean))
    assert 'data.npz' in written and 'export.json' in written
    assert all(not os.path.isabs(w) for w in written)
    assert sorted(written) == sorted(
        os.path.relpath(os.path.join(d, f), out)
        for d, _, files in os.walk(out) for f in files)
    with np.load(out / 'data.npz', allow_pickle=False) as data:
        assert set(data.files) == set(arrays)
        for name in arrays:
            assert data[name].dtype == arrays[name].dtype, name
            nan_ok = arrays[name].dtype.kind == 'f'
            assert np.array_equal(data[name], arrays[name], equal_nan=nan_ok), name
    for name in synth.CAMERAS:
        rows = np.unique(arrays[f'image_{name}_index'])
        rows = rows[rows >= 0]
        pngs = sorted(w for w in written if w.startswith(f'images/{name}/'))
        assert pngs == [f'images/{name}/{r:06d}.png' for r in rows]
        for r in (rows[0], rows[len(rows) // 2], rows[-1]):
            decoded = np.array(Image.open(out / 'images' / name / f'{r:06d}.png'))
            assert np.array_equal(decoded, synth.image_pixels(name, r))
    text = (out / 'export.json').read_text(encoding='utf-8')
    document = json.loads(text)
    assert document['format'] == 'ranger_xarm_teleop.export/1'
    assert document['files'][-1] == 'export.json'
    for value in strings(document):
        assert not value.startswith('/'), value
    assert os.path.expanduser('~') not in text
    assert socket.gethostname() not in text
    assert str(tmp_path) not in text


def test_write_export_pixel_encodings(clean, tmp_path):
    ep = copy.deepcopy(clean)
    ep.profile['streams']['camera_gantry']['image']['encoding'] = 'bgr8'
    ep.profile['streams']['camera_wrist']['image']['encoding'] = '32FC1'
    for name in synth.CAMERAS:
        ep.streams[name].labels = {}
    float_image = np.linspace(0.0, 3.0, 48, dtype=np.float32).reshape(6, 8)
    ep.images['camera_wrist'] = lambda row: float_image + row
    arrays, meta = export.build_export(ep, CHECKS)
    assert meta['images']['camera_wrist']['file_extension'] == 'npy'
    assert 'not a PNG' in meta['images']['camera_wrist']['note']
    written = export.write_export(str(tmp_path), arrays, meta, ep)
    assert any(w.startswith('images/camera_wrist/') and w.endswith('.npy') for w in written)
    assert not any(w.startswith('images/camera_wrist/') and w.endswith('.png') for w in written)
    loaded = np.load(tmp_path / 'images' / 'camera_wrist' / '000007.npy')
    assert loaded.dtype == np.float32 and np.array_equal(loaded, float_image + 7)
    decoded = np.array(Image.open(tmp_path / 'images' / 'camera_gantry' / '000003.png'))
    assert np.array_equal(decoded, synth.image_pixels('camera_gantry', 3)[..., ::-1])


def test_write_export_mono_and_16_bit(clean, tmp_path):
    ep = copy.deepcopy(clean)
    ep.profile['streams']['camera_gantry']['image']['encoding'] = 'mono8'
    ep.profile['streams']['camera_wrist']['image']['encoding'] = '16UC1'
    for name in synth.CAMERAS:
        ep.streams[name].labels = {}
    gray = np.arange(48, dtype=np.uint8).reshape(6, 8)
    deep = (np.arange(48, dtype=np.uint16).reshape(6, 8) * 1000)
    ep.images['camera_gantry'] = lambda row: gray
    ep.images['camera_wrist'] = lambda row: deep
    arrays, meta = export.build_export(ep, CHECKS)
    export.write_export(str(tmp_path), arrays, meta, ep, image_streams=list(synth.CAMERAS))
    assert np.array_equal(np.array(Image.open(tmp_path / 'images' / 'camera_gantry'
                                              / '000000.png')), gray)
    assert np.array_equal(np.array(Image.open(tmp_path / 'images' / 'camera_wrist'
                                              / '000000.png')), deep)


def test_build_export_does_not_mutate_the_episode(clean):
    ep = copy.deepcopy(clean)
    before = synth.snapshot(ep)
    export.build_export(ep, CHECKS)
    assert synth.same(before, synth.snapshot(ep))


def test_missing_state_is_an_error(clean):
    ep = copy.deepcopy(clean)
    del ep.streams['state']
    with pytest.raises(ValueError):
        export.build_export(ep, CHECKS)
