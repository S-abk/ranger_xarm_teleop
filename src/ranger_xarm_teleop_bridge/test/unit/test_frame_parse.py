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
"""INV-2, INV-4, INV-7: strict parsing of relay messages."""

import json
import os

from parse_helpers import button, hand, load_make_fixtures, valid_frame
import pytest

from ranger_xarm_teleop_bridge.frame_parse import (
    Ignored, Malformed, parse_message, parse_text, ParsedFrame)
from ranger_xarm_teleop_bridge.jsonl_io import read_records

FIX = load_make_fixtures()


def parse(frame):
    return parse_text(json.dumps(frame))


def assert_malformed(frame, fragment):
    result = parse(frame)
    assert isinstance(result, Malformed), result
    assert result.reason
    assert fragment in result.reason, result.reason


# ---- INV-2: absent hands ---------------------------------------------------------

def test_both_hands_present():
    result = parse(valid_frame())
    assert isinstance(result, ParsedFrame)
    assert result.left.position == (-0.25, 0.95, -0.45)
    assert result.right.position == (0.25, 0.95, -0.45)
    assert result.right.emulated_position is False
    assert result.viewer.position == (0.0, 1.6, 0.0)


@pytest.mark.parametrize('absent,present', [('left', 'right'), ('right', 'left')])
def test_absent_hand_is_none_and_other_unaffected(absent, present):
    full = parse(valid_frame())
    frame = valid_frame()
    del frame['controllers'][absent]
    result = parse(frame)
    assert isinstance(result, ParsedFrame)
    assert getattr(result, absent) is None
    assert getattr(result, present) == getattr(full, present)


def test_viewer_only_frame_has_no_hands():
    frame = valid_frame()
    frame['controllers'] = {}
    result = parse(frame)
    assert isinstance(result, ParsedFrame)
    assert result.left is None and result.right is None
    assert result.viewer is not None


def test_hand_tracked_is_not_present_whatever_else_it_contains():
    frame = valid_frame()
    frame['controllers']['right'] = {'hand_tracked': True}
    result = parse(frame)
    assert result.right is None
    assert result.left is not None
    frame['controllers']['right'] = dict(hand(), hand_tracked=True)
    assert parse(frame).right is None
    frame['controllers']['right'] = {'hand_tracked': True, 'position': [1], 'buttons': 3}
    assert parse(frame).right is None


def test_hand_tracked_false_is_a_normal_controller():
    frame = valid_frame()
    frame['controllers']['right'] = dict(hand(), hand_tracked=False)
    assert parse(frame).right is not None


def test_unknown_controller_keys_are_ignored():
    frame = valid_frame()
    frame['controllers']['head'] = 'junk'
    frame['controllers']['gaze'] = None
    result = parse(frame)
    assert isinstance(result, ParsedFrame)


def test_viewer_is_never_a_hand():
    frame = valid_frame()
    del frame['controllers']['right']
    result = parse(frame)
    assert result.right is None
    assert result.viewer is not None


def test_null_viewer_is_ok():
    frame = valid_frame()
    frame['viewer'] = None
    result = parse(frame)
    assert isinstance(result, ParsedFrame)
    assert result.viewer is None


def test_hand_value_null_is_malformed():
    frame = valid_frame()
    frame['controllers']['right'] = None
    assert_malformed(frame, 'controllers.right')


def test_invalid_present_hand_makes_whole_frame_malformed():
    frame = valid_frame()
    frame['controllers']['right']['position'] = [0.0, 1.0]
    assert_malformed(frame, 'controllers.right.position: expected 3 finite numbers')


# ---- INV-4: exact values ---------------------------------------------------------

def test_times_and_sequence_preserved_exactly():
    frame = valid_frame()
    text = ('{"type":"xr_frame","t_client":123457.18900000002,"t_xr":123456.78900000001,'
            '"seq":9007199254740993,"reset_count":3,"controllers":{},"viewer":null}')
    result = parse_text(text)
    assert isinstance(result, ParsedFrame)
    assert result.xr_time_ms == 123456.78900000001
    assert result.xr_time_ms.hex() == float('123456.78900000001').hex()
    assert result.client_time_ms == 123457.18900000002
    assert result.seq == 9007199254740993
    assert result.reset_count == 3
    assert frame['seq'] == 5


def test_integer_time_becomes_float_and_seq_stays_int():
    frame = valid_frame()
    frame['t_xr'] = 1000
    result = parse(frame)
    assert result.xr_time_ms == 1000.0
    assert type(result.xr_time_ms) is float
    assert type(result.seq) is int


def test_output_types_are_plain_python():
    frame = valid_frame()
    frame['controllers']['right']['buttons'][1] = button(True, 1)
    result = parse(frame)
    for hand_sample in (result.left, result.right):
        assert type(hand_sample.emulated_position) is bool
        assert all(type(v) is float for v in hand_sample.position)
        assert all(type(v) is float for v in hand_sample.orientation)
        assert all(type(v) is float for v in hand_sample.axes)
        for b in hand_sample.buttons:
            assert type(b.pressed) is bool and type(b.touched) is bool
            assert type(b.value) is float
    for value in (result.xr_time_ms, result.client_time_ms):
        assert type(value) is float
    assert type(result.seq) is int and type(result.reset_count) is int
    assert all(type(v) is float for v in result.viewer.position + result.viewer.orientation)
    assert type(result.viewer.emulated_position) is bool
    assert isinstance(result.right.position, tuple) and isinstance(result.right.buttons, tuple)


def test_button_defaults():
    frame = valid_frame()
    frame['controllers']['right']['buttons'] = [{'p': True}]
    result = parse(frame)
    assert result.right.buttons[0].pressed is True
    assert result.right.buttons[0].touched is False
    assert result.right.buttons[0].value == 0.0


def test_extra_keys_ignored():
    frame = valid_frame()
    frame['junk'] = [1]
    frame['controllers']['right']['more'] = {}
    frame['viewer']['more'] = 1
    frame['controllers']['right']['buttons'][0]['x'] = 1
    assert isinstance(parse(frame), ParsedFrame)


# ---- Ignored and message-level errors --------------------------------------------

@pytest.mark.parametrize('msg_type', ['camera_list', 'echo', 'ik_state', 'config_update', ''])
def test_other_types_are_ignored(msg_type):
    assert parse_text(json.dumps({'type': msg_type, 'x': 1})) == Ignored(msg_type)


def test_parse_message_matches_parse_text():
    frame = valid_frame()
    assert parse_message(frame) == parse(frame)


@pytest.mark.parametrize('text,fragment', [
    ('', 'not strict JSON'),
    ('not json at all', 'not strict JSON'),
    ('{"type": "xr_frame", ', 'not strict JSON'),
    ('[1, 2]', 'JSON object'),
    ('"xr_frame"', 'JSON object'),
    ('null', 'JSON object'),
    ('3', 'JSON object'),
    ('{}', 'type: missing'),
    ('{"type": 123}', 'type: expected a string'),
    ('{"type": null}', 'type: expected a string'),
    ('[' * 100000, 'not strict JSON'),
    ('1' * 10000, 'not strict JSON'),
])
def test_message_level_errors(text, fragment):
    result = parse_text(text)
    assert isinstance(result, Malformed)
    assert fragment in result.reason


# ---- strictness ------------------------------------------------------------------

def test_stock_client_frame_is_reported_as_unpatched():
    frame = valid_frame()
    for key in ('seq', 't_xr', 'reset_count'):
        del frame[key]
    for h in frame['controllers'].values():
        del h['emulated_position']
    assert_malformed(frame, 'client not patched: missing seq')
    frame = valid_frame()
    del frame['reset_count']
    assert_malformed(frame, 'client not patched: missing reset_count')
    frame = valid_frame()
    del frame['t_xr']
    assert_malformed(frame, 'client not patched: missing t_xr')


@pytest.mark.parametrize('field,value,fragment', [
    ('seq', -1, 'seq: expected an integer >= 0'),
    ('seq', True, 'seq'),
    ('seq', 1.0, 'seq'),
    ('seq', '1', 'seq'),
    ('seq', None, 'seq'),
    ('reset_count', -1, 'reset_count'),
    ('reset_count', False, 'reset_count'),
    ('reset_count', 0.5, 'reset_count'),
    ('t_xr', 'abc', 't_xr: expected a finite number'),
    ('t_xr', True, 't_xr'),
    ('t_xr', None, 't_xr'),
    ('t_client', 'abc', 't_client'),
    ('t_client', None, 't_client'),
    ('t_client', True, 't_client'),
    ('controllers', None, 'controllers'),
    ('controllers', [], 'controllers'),
    ('viewer', 'x', 'viewer'),
    ('viewer', [], 'viewer'),
])
def test_top_level_field_strictness(field, value, fragment):
    frame = valid_frame()
    frame[field] = value
    text = json.dumps(frame)
    assert fragment in parse_text(text).reason


def test_missing_t_client_and_viewer():
    frame = valid_frame()
    del frame['t_client']
    assert_malformed(frame, 't_client')
    frame = valid_frame()
    del frame['viewer']
    assert_malformed(frame, 'viewer: missing')
    frame = valid_frame()
    del frame['controllers']
    assert_malformed(frame, 'controllers')


@pytest.mark.parametrize('token', ['NaN', 'Infinity', '-Infinity'])
def test_non_finite_tokens_rejected(token):
    text = ('{"type":"xr_frame","t_client":1,"t_xr":%s,"seq":0,"reset_count":0,'
            '"controllers":{},"viewer":null}' % token)
    result = parse_text(text)
    assert isinstance(result, Malformed)
    assert token in result.reason


def test_overflowing_t_xr_rejected():
    text = json.dumps(valid_frame()).replace('"t_xr": 1000.0', '"t_xr": 1e999')
    assert 't_xr: expected a finite number' in parse_text(text).reason


def test_overflowing_number_rejected():
    frame = valid_frame()
    text = json.dumps(frame).replace('0.95', '1e999', 1)
    assert 'controllers.left.position' in parse_text(text).reason


@pytest.mark.parametrize('path,value,fragment', [
    (('position',), [0.1, 0.2], 'position: expected 3 finite numbers'),
    (('position',), [0.1, 0.2, 0.3, 0.4], 'position: expected 3 finite numbers'),
    (('position',), [0.1, 0.2, 'x'], 'position: expected 3 finite numbers'),
    (('position',), [0.1, 0.2, True], 'position: expected 3 finite numbers'),
    (('position',), [0.1, 0.2, None], 'position: expected 3 finite numbers'),
    (('position',), 'abc', 'position: expected 3 finite numbers'),
    (('position',), None, 'position: expected 3 finite numbers'),
    (('orientation',), [0.0, 0.0, 1.0], 'orientation: expected 4 finite numbers'),
    (('orientation',), [0.0, 0.0, 0.0, 2.0], 'orientation: expected a unit quaternion'),
    (('orientation',), [0.0, 0.0, 0.0, 0.0], 'orientation: expected a unit quaternion'),
    (('orientation',), [0.0, 0.0, 0.0, 1.002], 'orientation: expected a unit quaternion'),
    (('emulated_position',), 'yes', 'emulated_position: expected bool'),
    (('emulated_position',), 1, 'emulated_position: expected bool'),
    (('emulated_position',), None, 'emulated_position: expected bool'),
    (('buttons',), {}, 'buttons: expected a list'),
    (('buttons',), None, 'buttons: expected a list'),
    (('buttons',), [3], 'buttons[0]: expected an object'),
    (('buttons',), [{'t': True}], 'buttons[0].p: missing'),
    (('buttons',), [{'p': 1}], 'buttons[0].p: expected bool'),
    (('buttons',), [{'p': True, 't': 'x'}], 'buttons[0].t: expected bool'),
    (('buttons',), [{'p': True, 'v': 'x'}], 'buttons[0].v: expected a finite number'),
    (('buttons',), [{'p': True, 'v': None}], 'buttons[0].v: expected a finite number'),
    (('axes',), [0.0, 'x'], 'axes[1]: expected a finite number'),
    (('axes',), 'abc', 'axes: expected a list'),
    (('axes',), [True], 'axes[0]: expected a finite number'),
])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_hand_field_strictness(side, path, value, fragment):
    frame = valid_frame()
    frame['controllers'][side][path[0]] = value
    assert_malformed(frame, f'controllers.{side}.{fragment}')


@pytest.mark.parametrize('key', ['position', 'orientation', 'emulated_position', 'buttons',
                                 'axes'])
def test_missing_hand_field(key):
    frame = valid_frame()
    del frame['controllers']['right'][key]
    assert_malformed(frame, f'controllers.right.{key}')


def test_orientation_norm_tolerance_edge():
    frame = valid_frame()
    frame['controllers']['right']['orientation'] = [0.0, 0.0, 0.0, 1.0009]
    assert isinstance(parse(frame), ParsedFrame)


def test_viewer_fields_validated():
    frame = valid_frame()
    frame['viewer']['orientation'] = [0.0, 0.0, 0.0]
    assert_malformed(frame, 'viewer.orientation')
    frame = valid_frame()
    del frame['viewer']['emulated_position']
    assert_malformed(frame, 'viewer.emulated_position')
    frame = valid_frame()
    frame['viewer']['position'] = [0, 1]
    assert_malformed(frame, 'viewer.position')


# ---- INV-7: the malformed fixture and fuzzing ------------------------------------

def malformed_records():
    return read_records(os.path.join(os.path.dirname(FIX.__file__), 'malformed.jsonl'))


def test_malformed_fixture_matches_expected():
    records = malformed_records()
    expected = FIX.expected()
    assert len(records) == len(expected)
    kinds = {ParsedFrame: 'frame', Ignored: 'ignored', Malformed: 'malformed'}
    for n, (record, want) in enumerate(zip(records, expected)):
        result = parse_text(record.text())
        assert kinds[type(result)] == want, (n, record, result)
        if isinstance(result, Malformed):
            assert result.reason
        if isinstance(result, ParsedFrame):
            assert result.seq >= 0
    assert expected[0] == 'frame' and expected[-1] == 'frame'
    assert {'frame', 'ignored', 'malformed'} <= set(expected)


def test_malformed_fixture_has_raw_lines():
    assert any(r.raw is not None for r in malformed_records())


def _paths(node, prefix=()):
    yield prefix
    if isinstance(node, dict):
        for key, child in node.items():
            yield from _paths(child, prefix + (key,))
    elif isinstance(node, list):
        for index, child in enumerate(node):
            yield from _paths(child, prefix + (index,))


def _mutate(frame, path, replacement):
    if not path:
        return None if replacement is _DELETE else replacement
    node = frame
    for key in path[:-1]:
        node = node[key]
    if replacement is _DELETE:
        if isinstance(node, dict):
            del node[path[-1]]
        else:
            node.pop(path[-1])
    else:
        node[path[-1]] = replacement
    return frame


_DELETE = object()
_REPLACEMENTS = [_DELETE, None, 'x', [], {}, [1], -1, True, 2.5e300]


def test_fuzz_mutations_never_raise():
    base = valid_frame()
    base['controllers']['right']['buttons'][1] = button(True, 1.0)
    count = 0
    for path in _paths(base):
        for replacement in _REPLACEMENTS:
            frame = _mutate(json.loads(json.dumps(base)), path, replacement)
            for result in (parse_text(json.dumps(frame)), parse_message(frame)):
                assert isinstance(result, (ParsedFrame, Ignored, Malformed))
                if isinstance(result, Malformed):
                    assert result.reason
            count += 1
    assert count >= 200


def test_every_deleted_required_key_is_malformed():
    base = valid_frame()
    required = [('seq',), ('t_xr',), ('t_client',), ('reset_count',), ('controllers',),
                ('viewer',), ('controllers', 'right', 'position'),
                ('controllers', 'right', 'orientation'),
                ('controllers', 'right', 'emulated_position'),
                ('controllers', 'right', 'buttons'), ('controllers', 'right', 'axes'),
                ('controllers', 'left', 'position'), ('viewer', 'position'),
                ('viewer', 'orientation'), ('viewer', 'emulated_position')]
    for path in required:
        frame = _mutate(json.loads(json.dumps(base)), path, _DELETE)
        assert isinstance(parse_message(frame), Malformed), path


def test_every_truncation_never_raises():
    text = json.dumps(valid_frame())
    for end in range(len(text) + 1):
        result = parse_text(text[:end])
        if end < len(text):
            assert isinstance(result, Malformed), end
            assert result.reason
        else:
            assert isinstance(result, ParsedFrame)


@pytest.mark.parametrize('value', [None, 3, 'x', [], [[]], {}, {'type': ['xr_frame']}, True,
                                   float('nan')])
def test_parse_message_never_raises_on_any_value(value):
    assert isinstance(parse_message(value), (Ignored, Malformed))
