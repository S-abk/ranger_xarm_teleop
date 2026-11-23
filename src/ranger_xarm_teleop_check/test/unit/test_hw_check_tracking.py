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
Teleop stage: is the arm held back from following the hand (INFO, not judged yet).

On the first hardware stage 4 run every rule passed and the arm did not track: a pair at the
link keep-out's margin at the working posture held joints 1 and 2 still, and the tool pivoted
about the wrist. The lead rules cannot see it: the gate absorbs the lead the keep-out blocks
(G-14). These lines report what can: how often the keep-out shaped the command, how close to
the margin, which pair, and the target's rotation lead.
"""

import math

from ranger_xarm_teleop_check import rules
from ranger_xarm_teleop_check.rules import Options

MARGIN = 0.04
PAIR = ('xarm_link2', 'xarm_link_base')


def row(engaged=True, bound=False, clearance=0.06, margin=MARGIN, pair=('', ''),
        rotation=0.0, limit=0):
    """One /teleop/state sample as hw_check keeps it (``data['tracking']``)."""
    return (engaged, limit, bound, clearance, margin, pair[0], pair[1], 0.0, rotation)


def evaluate(tracking):
    return rules.tracking_measures({'tracking': tracking})


def test_the_keepout_shaping_the_command_is_counted_with_the_room_and_the_pair():
    """The first stage 4 run: link 2 against the base held at a 40 mm margin, all along."""
    held = [row(bound=True, clearance=MARGIN + 0.0005 * (k % 3), pair=PAIR) for k in range(80)]
    free = [row() for _ in range(20)]
    km = evaluate(held + free + [row(engaged=False, bound=True, clearance=0.01)] * 50)
    assert km['engaged'] == 100 and km['bound'] == 80
    assert km['pairs'] == {'xarm_link2 - xarm_link_base': 80}
    result = rules.info_keepout_hold(km)
    assert result.verdict == rules.INFO
    assert result.measured == (
        "80 % of 100 engaged cycles (80); clearance at the command's reference minus the "
        'margin: median 0.5 mm, least 0.0 mm; nearest pair at the measured joints: '
        'xarm_link2 - xarm_link_base (80)')
    assert 'threshold waits for stage 4 data' in result.threshold


def test_a_pair_inside_the_margin_shows_a_negative_room():
    km = evaluate([row(bound=True, clearance=0.0352, pair=PAIR)])
    assert 'median -4.8 mm, least -4.8 mm' in rules.info_keepout_hold(km).measured


def test_open_space_shows_none():
    km = evaluate([row() for _ in range(100)])
    assert rules.info_keepout_hold(km).measured == '0 % of 100 engaged cycles (0)'


def test_a_bound_cycle_without_a_clearance_or_a_pair_is_still_counted():
    km = evaluate([row(bound=True, clearance=math.nan)])
    assert km['bound'] == 1 and km['room_p50_m'] is None
    assert rules.info_keepout_hold(km).measured == (
        '100 % of 1 engaged cycles (1); nearest pair at the measured joints: not reported')


def test_without_an_engaged_cycle_it_says_so():
    km = evaluate([row(engaged=False)])
    assert rules.info_keepout_hold(km).measured == 'no engaged cycle'
    assert rules.info_rotation_lead(km).measured == 'no engaged cycle'


def test_the_rotation_lead_is_shown_in_degrees():
    rotations = [math.radians(d) for d in range(100)]
    km = evaluate([row(rotation=r) for r in rotations])
    result = rules.info_rotation_lead(km)
    assert result.verdict == rules.INFO
    assert result.measured == 'p50 50.0 p99 99.0 max 99.0 deg over 100 engaged cycles'


def test_the_teleop_stage_reports_both_lines_and_judges_nothing_by_them():
    data = {'tracking': [row(bound=True, clearance=0.03, pair=('a', 'b'))] * 10}
    _, results = rules.evaluate('teleop', data, Options())
    names = {r.rule: r.verdict for r in results}
    assert names[rules.KEEPOUT_HOLD_INFO] == rules.INFO
    assert names[rules.ROTATION_LEAD_INFO] == rules.INFO
