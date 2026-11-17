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
"""
Recording INV-8: the left controller's episode buttons change nothing in teleop.

The episode fixtures press X, Y and the left trigger throughout; the same recordings with
the left controller removed must give the gate identical frames and identical outputs.
"""

import json
import os
import sys

from helpers import params, TOOL_START
import pytest
from ranger_xarm_teleop_servo.frames import frames_of_recording, simulate

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fixtures')


def generated(name):
    sys.path.insert(0, FIXTURES)
    try:
        import make_fixtures
    finally:
        sys.path.remove(FIXTURES)
    return getattr(make_fixtures, name)()


def without_left(lines):
    out = []
    for text in lines:
        record = json.loads(text)
        record['msg']['controllers'].pop('left', None)
        out.append(json.dumps(record, ensure_ascii=False, allow_nan=False))
    return out


def write(path, lines):
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return str(path)


@pytest.mark.parametrize('name', ['episode_marks_engaged', 'episode_disarm', 'episodes_ten'])
def test_left_buttons_leave_the_gate_bit_identical(name, tmp_path):
    lines = generated(name)
    assert any('"left"' in text for text in lines)
    with_left = frames_of_recording(write(tmp_path / 'with.jsonl', lines))
    plain = frames_of_recording(write(tmp_path / 'plain.jsonl', without_left(lines)))
    assert len(with_left) == len(plain)
    a, _ = simulate(with_left, params(), TOOL_START)
    b, _ = simulate(plain, params(), TOOL_START)
    assert len(a) == len(b)
    assert any(s.output.engaged for s in a)
    for x, y in zip(a, b):
        assert x == y


def test_the_fixtures_press_every_episode_input():
    lines = generated('episode_marks_engaged')
    left = [json.loads(t)['msg']['controllers']['left'] for t in lines]
    assert any(h['buttons'][4]['p'] for h in left)            # X
    assert any(h['buttons'][5]['p'] for h in left)            # Y
    assert any(h['buttons'][0]['v'] > 0.6 for h in left)      # left trigger
