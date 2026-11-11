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
"""JSONL recording: round trip, raw fallback, corrupt lines."""

import json

from parse_helpers import valid_frame
import pytest

from ranger_xarm_teleop_bridge.frame_parse import parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.jsonl_io import iter_records, read_records, Record, RecordWriter


def write(path, texts, start=100.0):
    with RecordWriter(str(path)) as writer:
        for n, text in enumerate(texts):
            writer.write_text(text, t_recv=start + n)


def test_round_trip_text_record_text(tmp_path):
    path = tmp_path / 'rec.jsonl'
    text = json.dumps(valid_frame())
    write(path, [text])
    (record,) = read_records(str(path))
    assert record.t_recv == 100.0
    assert record.raw is None
    assert record.msg == json.loads(text)
    assert json.loads(record.text()) == json.loads(text)
    assert parse_text(record.text()) == parse_text(text)
    assert isinstance(parse_text(record.text()), ParsedFrame)


def test_file_layout_is_one_object_per_line(tmp_path):
    path = tmp_path / 'rec.jsonl'
    write(path, ['{"a": 1}', 'not json'])
    lines = path.read_text(encoding='utf-8').splitlines()
    assert [json.loads(line) for line in lines] == [
        {'t_recv': 100.0, 'msg': {'a': 1}}, {'t_recv': 101.0, 'raw': 'not json'}]


@pytest.mark.parametrize('text', [
    'not json', '', '{"type": "xr_frame", ', '{"t_xr": NaN}', '[Infinity]', '[1e999]',
    '{"a": "\\ud800"}'])
def test_invalid_or_unrepresentable_text_is_kept_raw(tmp_path, text):
    path = tmp_path / 'rec.jsonl'
    write(path, [text])
    (record,) = read_records(str(path))
    if record.raw is None:
        # Strict JSON that only needs escaping to be stored: it must still round trip.
        assert text == '{"a": "\\ud800"}'
        assert record.msg == {'a': '\ud800'}
    else:
        assert record.raw == text
        assert record.text() == text


def test_raw_record_text_feeds_parser(tmp_path):
    path = tmp_path / 'rec.jsonl'
    write(path, ['{"type": "xr_frame", '])
    (record,) = read_records(str(path))
    assert record.raw == '{"type": "xr_frame", '
    assert parse_text(record.text()).reason


def test_non_json_values_round_trip(tmp_path):
    path = tmp_path / 'rec.jsonl'
    write(path, ['null', '[1, 2]', '"s"', '3'])
    records = read_records(str(path))
    assert [r.msg for r in records] == [None, [1, 2], 's', 3]
    assert [r.text() for r in records] == ['null', '[1,2]', '"s"', '3']


def test_unicode_is_stored_unescaped(tmp_path):
    path = tmp_path / 'rec.jsonl'
    write(path, ['{"name": "café"}'])
    assert 'café' in path.read_text(encoding='utf-8')
    assert read_records(str(path))[0].msg == {'name': 'café'}


def test_default_t_recv_is_now_and_parent_dirs_are_created(tmp_path):
    import time
    path = tmp_path / 'a' / 'b' / 'rec.jsonl'
    before = time.time()
    with RecordWriter(str(path)) as writer:
        writer.write_text('{}')
    (record,) = read_records(str(path))
    assert before <= record.t_recv <= time.time()


def test_flush_on_every_line(tmp_path):
    path = tmp_path / 'rec.jsonl'
    writer = RecordWriter(str(path))
    writer.write_text('{"a": 1}')
    assert len(path.read_text(encoding='utf-8').splitlines()) == 1
    writer.close()
    writer.close()


def test_corrupt_line_raises_with_line_number(tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text('{"t_recv": 1.0, "msg": {}}\n{"t_recv": 2.0, "msg"\n', encoding='utf-8')
    with pytest.raises(ValueError, match=r'bad\.jsonl:2:'):
        read_records(str(path))
    iterator = iter_records(str(path))
    assert next(iterator) == Record(t_recv=1.0, msg={})
    with pytest.raises(ValueError, match=r'bad\.jsonl:2:'):
        next(iterator)


@pytest.mark.parametrize('line', [
    '[1]', '{"msg": {}}', '{"t_recv": "x", "msg": {}}', '{"t_recv": true, "msg": {}}',
    '{"t_recv": 1.0}', '{"t_recv": 1.0, "raw": 5}', '{"t_recv": NaN, "msg": {}}'])
def test_invalid_record_shapes_raise(tmp_path, line):
    path = tmp_path / 'bad.jsonl'
    path.write_text('\n' + line + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match=r'bad\.jsonl:2:'):
        read_records(str(path))


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / 'blank.jsonl'
    path.write_text('\n{"t_recv": 1.0, "msg": 1}\n   \n\n{"t_recv": 2.0, "raw": "x"}\n\n',
                    encoding='utf-8')
    records = read_records(str(path))
    assert records == [Record(1.0, msg=1), Record(2.0, raw='x')]


def test_iter_and_read_are_equivalent(tmp_path):
    path = tmp_path / 'rec.jsonl'
    write(path, [json.dumps(valid_frame()), 'junk', '[1]'])
    assert list(iter_records(str(path))) == read_records(str(path))
    assert len(read_records(str(path))) == 3


def test_record_defaults_and_frozen():
    record = Record(t_recv=1.0)
    assert record.msg is None and record.raw is None
    assert record.text() == 'null'
    with pytest.raises(AttributeError):
        record.t_recv = 2.0
