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
"""Synthetic fixtures: byte-exact regeneration and the properties the brief states."""

import itertools
import math
import os

from parse_helpers import FIXTURE_DIR, load_make_fixtures
import pytest

from ranger_xarm_teleop_bridge.frame_parse import parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.jsonl_io import read_records, RecordWriter

FIX = load_make_fixtures()
FRAME_FIXTURES = ['still_controller', 'moving_controller', 'dropped_controller',
                  'reset_event']
DURATIONS_S = {'still_controller': 10, 'moving_controller': 6, 'dropped_controller': 8,
               'reset_event': 5}


def path_of(name):
    return os.path.join(FIXTURE_DIR, f'{name}.jsonl')


def frames(name):
    result = []
    for record in read_records(path_of(name)):
        parsed = parse_text(record.text())
        assert isinstance(parsed, ParsedFrame), (name, record)
        result.append((record.t_recv, parsed))
    return result


def seconds(frame):
    return (frame.xr_time_ms - 1000.0) / 1000.0


def test_generate_matches_committed_files_byte_for_byte():
    generated = FIX.generate()
    assert set(generated) == set(FRAME_FIXTURES) | {'malformed'}
    for name, lines in generated.items():
        with open(path_of(name), 'rb') as handle:
            committed = handle.read()
        assert committed == ''.join(line + '\n' for line in lines).encode('utf-8'), name


def test_generation_is_deterministic():
    assert FIX.generate() == FIX.generate()


def test_main_writes_files_to_directory(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['make_fixtures.py', str(tmp_path)])
    FIX.main()
    for name in FIX.generate():
        with open(path_of(name), 'rb') as committed:
            assert (tmp_path / f'{name}.jsonl').read_bytes() == committed.read()


@pytest.mark.parametrize('name', ['still_controller', 'moving_controller',
                                  'dropped_controller', 'reset_event', 'malformed'])
def test_recording_writer_reproduces_fixture(tmp_path, name):
    out = tmp_path / 'out.jsonl'
    with RecordWriter(str(out)) as writer:
        for record in read_records(path_of(name)):
            writer.write_text(record.text(), t_recv=record.t_recv)
    with open(path_of(name), encoding='utf-8') as handle:
        assert out.read_text(encoding='utf-8') == handle.read()


@pytest.mark.parametrize('name', FRAME_FIXTURES)
def test_every_line_parses_and_seq_increases(name):
    parsed = frames(name)
    assert len(parsed) == DURATIONS_S[name] * 72
    seqs = [f.seq for _, f in parsed]
    assert seqs == list(range(len(seqs)))
    t_recv = [t for t, _ in parsed]
    assert all(b > a for a, b in zip(t_recv, t_recv[1:]))
    assert t_recv[0] == pytest.approx(1700000000.0, abs=1e-3)
    assert (t_recv[-1] - t_recv[0]) == pytest.approx((len(t_recv) - 1) / 72, abs=2e-3)
    assert parsed[0][1].xr_time_ms == 1000.0
    for _, f in parsed:
        assert f.client_time_ms == pytest.approx(f.xr_time_ms + 0.4, abs=1e-6)
        assert len(f.right.buttons if f.right else f.left.buttons) == 7
        assert f.viewer is not None


def test_no_fixture_has_an_absent_hand_except_dropped():
    for name in ('still_controller', 'moving_controller', 'reset_event'):
        assert all(f.left and f.right for _, f in frames(name))


def angle_deg(a, b):
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def test_still_controller_spread_is_small():
    parsed = [f for _, f in frames('still_controller')]
    for side, centre in (('right', (0.25, 0.95, -0.45)), ('left', (-0.25, 0.95, -0.45))):
        samples = [getattr(f, side) for f in parsed]
        for axis in range(3):
            values = [s.position[axis] for s in samples]
            assert max(values) - min(values) < 1e-3
            assert sum(values) / len(values) == pytest.approx(centre[axis], abs=1e-4)
        quats = [s.orientation for s in samples]
        worst = max(angle_deg(a, b) for a, b in itertools.combinations(quats, 2))
        assert 0.0 < worst < 0.1
        assert all(not s.emulated_position for s in samples)
    # Right orientation is a fixed non-identity rotation.
    assert angle_deg(parsed[0].right.orientation, (0.0, 0.0, 0.0, 1.0)) > 10.0


def test_still_controller_viewer_sweeps_and_controllers_do_not_follow():
    parsed = [f for _, f in frames('still_controller')]
    yaws = [math.degrees(2.0 * math.atan2(f.viewer.orientation[1], f.viewer.orientation[3]))
            for f in parsed]
    xs = [f.viewer.position[0] for f in parsed]
    assert max(yaws) == pytest.approx(90.0, abs=1.0)
    assert min(yaws) == pytest.approx(-90.0, abs=1.0)
    assert max(xs) == pytest.approx(0.3, abs=0.005)
    assert min(xs) == pytest.approx(-0.3, abs=0.005)
    assert all(f.viewer.position[1] == 1.6 and f.viewer.position[2] == 0.0 for f in parsed)
    # Sine of period 6 s: back to zero at 3 s.
    assert abs(yaws[3 * 72]) < 1e-3


def test_moving_controller():
    parsed = [f for _, f in frames('moving_controller')]
    start, end = parsed[0].right, parsed[-1].right
    assert end.position[0] - start.position[0] == pytest.approx(0.4, abs=1e-6)
    assert end.position[1] == start.position[1] and end.position[2] == start.position[2]
    assert angle_deg(start.orientation, end.orientation) == pytest.approx(90.0, abs=1e-3)
    for f in parsed:
        t = seconds(f)
        b = f.right.buttons
        assert b[1].pressed == (t >= 1.0 - 1e-9)
        if b[1].pressed:
            assert b[1].value == 1.0
        assert b[0].value == pytest.approx(min(max((t - 2.0) / 2.0, 0.0), 1.0), abs=1e-6)
        stick = tuple(f.right.axes[2:4])
        assert stick == ((0.5, -0.5) if t >= 3.0 - 1e-9 else (0.0, 0.0))
        assert len(f.right.axes) == 4
    assert parsed[0].left.position == parsed[-1].left.position


def test_dropped_controller_intervals():
    parsed = [f for _, f in frames('dropped_controller')]
    for f in parsed:
        t = seconds(f)
        assert f.left is not None and not f.left.emulated_position
        if 2.0 - 1e-9 <= t < 3.0 - 1e-9:
            assert f.right is None
        else:
            assert f.right is not None
            assert f.right.emulated_position == (4.0 - 1e-9 <= t < 5.0 - 1e-9)
    absent = [f.seq for f in parsed if f.right is None]
    emulated = [f.seq for f in parsed if f.right is not None and f.right.emulated_position]
    assert (absent[0], absent[-1], len(absent)) == (144, 215, 72)
    assert (emulated[0], emulated[-1], len(emulated)) == (288, 359, 72)
    before = [f.right.position for f in parsed[:144]]
    mean = [sum(p[k] for p in before) / len(before) for k in range(3)]
    for f in parsed[216:288] + parsed[360:]:
        assert all(abs(f.right.position[k] - mean[k]) < 1e-3 for k in range(3))
    drift = [f.right.position[0] - mean[0] for f in parsed[288:360]]
    assert max(drift) == pytest.approx(0.02, abs=1e-3)
    assert drift[-1] > drift[0]


def test_reset_event():
    parsed = [f for _, f in frames('reset_event')]
    for f in parsed:
        assert f.reset_count == (1 if seconds(f) >= 2.5 - 1e-9 else 0)
    before, after = parsed[179], parsed[180]
    assert (before.reset_count, after.reset_count) == (0, 1)
    for side in ('left', 'right'):
        a, b = getattr(before, side).position, getattr(after, side).position
        assert b[0] - a[0] == pytest.approx(0.1, abs=1e-6)
        assert b[1] - a[1] == pytest.approx(0.0, abs=1e-6)
        assert b[2] - a[2] == pytest.approx(0.05, abs=1e-6)
    assert parsed[-1].right.position[0] == pytest.approx(0.35, abs=1e-6)
    assert len({f.right.position for f in parsed[180:]}) == 1


def test_expected_has_one_entry_per_malformed_line():
    assert len(FIX.expected()) == len(FIX.generate()['malformed'])
    assert set(FIX.expected()) == {'frame', 'ignored', 'malformed'}
