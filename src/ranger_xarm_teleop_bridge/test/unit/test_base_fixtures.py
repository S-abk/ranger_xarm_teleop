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
"""Base-driving fixtures: byte-exact regeneration and what each one does at which second."""

import math
import os

from parse_helpers import FIXTURE_DIR, load_make_fixtures
import pytest

from ranger_xarm_teleop_bridge.frame_parse import parse_text, ParsedFrame
from ranger_xarm_teleop_bridge.joy_layout import build_joy_arrays
from ranger_xarm_teleop_bridge.jsonl_io import read_records

FIX = load_make_fixtures()
NAMES = ['base_drive_between_strokes', 'base_fault_release', 'base_fault_stale',
         'base_fault_left_missing', 'base_fault_session', 'base_fault_seq_backwards',
         'base_both_grips_same_frame', 'base_right_grip_during_drive',
         'base_left_grip_during_stroke', 'base_reset_pull_while_driving', 'base_turn_diagonal']
# Joy axes and buttons (joy_layout): left trigger, grip, stick x, y; right trigger, grip, x, y.
LT, LG, LX, LY, RT, RG, RX, RY = range(8)
B_LEFT_GRIP, B_RIGHT_GRIP, B_RIGHT_A, B_RIGHT_B = 1, 6, 8, 9


def path_of(name):
    return os.path.join(FIXTURE_DIR, f'{name}.jsonl')


def frames(name):
    """Return (t, ParsedFrame, joy axes, joy buttons) of a fixture, t in seconds."""
    out = []
    for record in read_records(path_of(name)):
        parsed = parse_text(record.text())
        assert isinstance(parsed, ParsedFrame), (name, record)
        axes, buttons = build_joy_arrays(parsed)
        out.append(((parsed.xr_time_ms - 1000.0) / 1000.0, parsed, axes, buttons))
    return out


def at(rows, t):
    """Return the row nearest to ``t`` s."""
    return min(rows, key=lambda r: abs(r[0] - t))


def windows(rows, test):
    """Return the (first, last) times of each run of rows where ``test(row)`` holds."""
    runs, start, prev = [], None, None
    for row in rows:
        if test(row) and start is None:
            start = row[0]
        if not test(row) and start is not None:
            runs.append((round(start, 2), round(prev, 2)))
            start = None
        prev = row[0]
    if start is not None:
        runs.append((round(start, 2), round(prev, 2)))
    return runs


def left_grip(row):
    return row[3][B_LEFT_GRIP] == 1


def right_grip(row):
    return row[3][B_RIGHT_GRIP] == 1


def test_generate_base_matches_committed_files_byte_for_byte():
    generated = FIX.generate_base()
    assert list(generated) == NAMES
    for name, lines in generated.items():
        with open(path_of(name), 'rb') as handle:
            assert handle.read() == ''.join(line + '\n' for line in lines).encode('utf-8'), name


def test_the_existing_fixtures_are_not_in_generate_base_and_not_changed_by_it():
    assert not set(FIX.generate_base()) & set(FIX.generate())
    assert FIX.generate_base() == FIX.generate_base()


def test_main_writes_the_base_fixtures_too(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['make_fixtures.py', str(tmp_path)])
    FIX.main()
    for name in NAMES:
        with open(path_of(name), 'rb') as committed:
            assert (tmp_path / f'{name}.jsonl').read_bytes() == committed.read()


@pytest.mark.parametrize('name', NAMES)
def test_every_frame_parses_at_the_headset_rate_with_both_hands_but_the_missing_one(name):
    rows = frames(name)
    assert rows[0][1].xr_time_ms == 1000.0 and rows[0][1].seq == 0
    assert rows[0][1].session_id == FIX.SESSION_ID
    t_recv = [rec.t_recv for rec in read_records(path_of(name))]
    assert len(t_recv) == len(rows)
    assert all(b > a for a, b in zip(t_recv, t_recv[1:]))
    assert t_recv[0] == pytest.approx(1700000000.0, abs=1e-3)
    for t, f, axes, buttons in rows:
        assert f.right is not None and len(f.right.buttons) == 7 and f.viewer is not None
        if f.left is None:
            assert name == 'base_fault_left_missing' and 3.0 - 1e-6 <= t < 3.5 - 1e-6
            assert math.isnan(axes[LT]) and axes[LG] == axes[LX] == axes[LY] == 0.0
            assert buttons[:5] == (0, 0, 0, 0, 0)
        else:
            assert axes[LT] == 0.0          # a present left controller's trigger value: 0.0


def test_the_session_opening_calibrates_then_arms_then_holds_still():
    rows = frames('base_fault_release')
    a = windows(rows, lambda r: r[3][B_RIGHT_A] == 1)[0]
    b = windows(rows, lambda r: r[3][B_RIGHT_B] == 1)[0]
    assert a == pytest.approx((0.3, 0.49), abs=0.02) and b == pytest.approx((0.7, 0.89), abs=0.02)
    assert not any(right_grip(r) for r in rows)
    assert at(rows, 4.0)[1].right.position == at(rows, 1.2)[1].right.position


def test_drive_between_strokes_has_five_actions_at_their_seconds():
    rows = frames('base_drive_between_strokes')
    assert len(rows) == 15 * 72
    assert windows(rows, left_grip) == [(1.5, 2.99), (7.0, 8.49), (9.5, 10.99)]
    assert windows(rows, right_grip) == [(4.0, 5.99), (12.0, 13.99)]
    # Forward: y negative, nothing sideways; right: x positive; the turn: right stick x negative
    # with the left stick centred.
    for t, f, axes, _ in rows:
        stick = (axes[LX], axes[LY])
        if 1.5 <= t < 3.0:
            assert stick == (0.0, -1.0) and (axes[RX], axes[RY]) == (0.0, 0.0)
        elif 7.0 <= t < 8.5:
            assert stick == (1.0, 0.0) and (axes[RX], axes[RY]) == (0.0, 0.0)
        elif 9.5 <= t < 11.0:
            assert stick == (0.0, 0.0) and (axes[RX], axes[RY]) == (-1.0, 0.0)
        else:
            assert stick == (0.0, 0.0) and (axes[RX], axes[RY]) == (0.0, 0.0)
        assert not (left_grip((t, f, axes, _)) and right_grip((t, f, axes, _)))
    # The arm strokes move the right hand 0.12 m forward and back; the drives do not.
    start = at(rows, 3.5)[1].right.position
    assert math.dist(at(rows, 5.5)[1].right.position, start) == pytest.approx(0.12, abs=1e-3)
    assert math.dist(at(rows, 7.5)[1].right.position, start) == pytest.approx(0.0, abs=1e-6)
    assert math.dist(at(rows, 13.6)[1].right.position, start) == pytest.approx(0.12, abs=1e-3)
    assert at(rows, 2.0)[1].right.position == at(rows, 3.5)[1].right.position


@pytest.mark.parametrize('name', ['base_fault_release', 'base_fault_stale',
                                  'base_fault_left_missing', 'base_fault_session',
                                  'base_fault_seq_backwards'])
def test_each_fault_fixture_starts_with_a_forward_drive_at_1_5_s(name):
    rows = frames(name)
    row = at(rows, 2.0)
    assert left_grip(row) and (row[2][LX], row[2][LY]) == (0.0, -1.0)
    assert not left_grip(at(rows, 1.4)) and not right_grip(at(rows, 2.0))


def test_release_ends_the_grip_at_3_s_and_the_stick_centres_after():
    rows = frames('base_fault_release')
    assert windows(rows, left_grip) == [(1.5, 2.99)]
    assert at(rows, 3.1)[2][LY] == -1.0 and at(rows, 3.4)[2][LY] == 0.0


def test_stale_has_no_frame_for_half_a_second_and_the_grip_stays_held():
    rows = frames('base_fault_stale')
    gap = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
    assert max(gap) == pytest.approx(0.5 + 1 / 72, abs=1e-3)
    assert sum(1 for g in gap if g > 0.1) == 1
    assert left_grip(at(rows, 2.9)) and left_grip(at(rows, 3.6))
    assert {r[1].session_id for r in rows} == {FIX.SESSION_ID}


def test_left_missing_drops_the_hand_from_3_to_3_5_s_with_the_grip_held_after():
    rows = frames('base_fault_left_missing')
    assert len(rows) == 4.5 * 72
    assert [round(r[0], 2) for r in rows if r[1].left is None][0] == 3.0
    assert sum(1 for r in rows if r[1].left is None) == 36
    assert left_grip(at(rows, 3.6)) and at(rows, 3.6)[2][LY] == -1.0


def test_session_changes_at_3_s_and_the_sequence_keeps_counting():
    rows = frames('base_fault_session')
    change = [(r[0], r[1].session_id) for a, r in zip(rows, rows[1:])
              if a[1].session_id != r[1].session_id]
    assert change == [(3.0, FIX.SESSION_ID_2)]
    assert [r[1].seq for r in rows] == list(range(len(rows)))
    assert left_grip(at(rows, 3.9))


def test_the_sequence_steps_back_by_50_at_3_s_and_the_grip_stays_held():
    rows = frames('base_fault_seq_backwards')
    seqs = [r[1].seq for r in rows]
    back = [i for i in range(1, len(seqs)) if seqs[i] < seqs[i - 1]]
    assert back == [216] and seqs[216] == 166 and seqs[215] == 215
    assert [b - a for a, b in zip(seqs[216:], seqs[217:])] == [1] * (len(seqs) - 217)
    assert left_grip(at(rows, 3.9)) and {r[1].session_id for r in rows} == {FIX.SESSION_ID}


def test_both_grips_are_pressed_in_the_same_frame_with_the_stick_forward():
    rows = frames('base_both_grips_same_frame')
    assert windows(rows, left_grip) == windows(rows, right_grip) == [(1.5, 3.99)]
    assert at(rows, 1.5)[2][LY] == -1.0


def test_right_grip_joins_a_drive_at_3_s_while_the_left_is_held():
    rows = frames('base_right_grip_during_drive')
    assert windows(rows, left_grip) == [(1.5, 3.99)]
    assert windows(rows, right_grip) == [(3.0, 3.99)]


def test_left_grip_is_pressed_at_3_s_during_a_right_grip_stroke():
    rows = frames('base_left_grip_during_stroke')
    assert windows(rows, right_grip) == [(1.5, 3.99)]
    assert windows(rows, left_grip) == [(3.0, 3.99)]
    assert at(rows, 2.0)[2][LY] == 0.0 and at(rows, 3.5)[2][LY] == -1.0
    assert math.dist(at(rows, 3.5)[1].right.position, at(rows, 1.2)[1].right.position) \
        == pytest.approx(0.12, abs=1e-3)


def test_reset_pull_is_the_right_stick_fully_back_for_a_second_during_the_drive():
    rows = frames('base_reset_pull_while_driving')
    pulled = windows(rows, lambda r: r[2][RY] == 1.0)
    assert pulled == [(2.5, 3.49)]
    assert all(r[2][RX] == 0.0 and r[2][LY] == -1.0 for r in rows if 2.5 <= r[0] < 3.5)
    assert windows(rows, left_grip) == [(1.5, 3.99)]


def test_turn_diagonal_is_not_a_reset_pull_and_the_left_stick_is_centred():
    rows = frames('base_turn_diagonal')
    assert windows(rows, left_grip) == [(1.5, 3.99)]
    for row in rows:
        assert row[2][LX] == row[2][LY] == 0.0
        assert (row[2][RX], row[2][RY]) == ((-0.9, 0.7) if 1.5 <= row[0] < 4.0 else (0.0, 0.0))
    assert not any(r[2][RY] == 1.0 for r in rows)
