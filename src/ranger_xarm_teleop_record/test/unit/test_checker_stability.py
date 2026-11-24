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
"""The stability section of the checker: information only, never a reason or a flag (no ROS)."""

import copy
import math
import os
import re

import numpy as np
import pytest
from ranger_xarm_teleop_record import checker, profile
from ranger_xarm_teleop_record.streams import Stream
import synth

CHECKS = profile.load_checks()
T0 = synth.T0
MSG = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                   'ranger_xarm_teleop_interfaces', 'msg', 'StabilityState.msg')


def with_stability(ep, rows=200, hz=50.0, **columns):
    """Record the stability state in ``ep`` (rows from T0); ``columns`` override the defaults."""
    spec = ep.profile['streams']['stability_state']
    stamp = T0 + np.arange(rows) / hz
    fields = {
        'available': np.ones(rows, dtype=bool), 'reason': np.zeros(rows, dtype=int),
        'tilt_source': np.full(rows, 1), 'margin_m': np.full(rows, 0.30),
        'edge': np.full(rows, 5), 'margin_pessimistic_m': np.full(rows, 0.25),
        'edge_pessimistic': np.full(rows, 5), 'stop_margin_m': np.full(rows, 0.20),
        'stop_margin_pessimistic_m': np.full(rows, 0.15), 'stop_edge': np.full(rows, 1),
        'stop_decel_m_s2': np.full(rows, 0.5), 'energy_reserve_j': np.full(rows, 40.0),
        'energy_edge': np.full(rows, 1), 'speed_m_s': np.zeros(rows),
        'at_rest': np.ones(rows, dtype=bool)}
    fields.update({k: np.asarray(v) for k, v in columns.items()})
    labels = {'reason_text': [''] * rows, 'model_sha256': ['a' * 64] * rows}
    ep.streams['stability_state'] = Stream(
        name='stability_state', topic=spec['topic'], kind=spec['kind'], stamp=stamp,
        recv=stamp + synth.LATENCY, fields=fields, labels=labels, has_header=True,
        frame_id=[synth.BASE_FRAME] * rows)
    return ep


def section(ep):
    """Return the stability section of the checker's report."""
    return checker.check_episode(ep, CHECKS)['stability']


def column(rows, value, at, base):
    """Return ``rows`` copies of ``base`` with ``value`` at the indices ``at``."""
    out = np.full(rows, base, dtype=float)
    out[list(at)] = value
    return out


def test_an_episode_without_the_stream_says_not_recorded_and_nothing_else_changes():
    plain = checker.check_episode(synth.build(), CHECKS)
    assert plain['stability'] == {'recorded': False}
    assert checker.stability_lines(plain['stability']) == ['stability: not recorded']
    without = {k: v for k, v in plain.items() if k != 'stability'}
    again = checker.check_episode(synth.build(), CHECKS)
    assert without == {k: v for k, v in again.items() if k != 'stability'}


def test_a_recorded_stream_reports_the_smallest_figures_with_edge_and_time_from_the_start():
    ep = synth.build()
    start = float(np.min(ep.streams['state'].stamp))
    rows = 200
    with_stability(
        ep, rows=rows,
        margin_m=column(rows, 0.05, [100], 0.30), edge=column(rows, 7, [100], 5),
        margin_pessimistic_m=column(rows, -0.02, [50], 0.25),
        edge_pessimistic=column(rows, 2, [50], 5),
        stop_margin_m=column(rows, 0.01, [150], 0.20), stop_edge=column(rows, 4, [150], 1),
        stop_decel_m_s2=column(rows, 0.8, [150], 0.5),
        energy_reserve_j=column(rows, -3.0, [20], 40.0), energy_edge=column(rows, 8, [20], 1))
    out = section(ep)
    assert out['recorded'] is True and out['n'] == rows
    assert out['unavailable'] == {'n': 0, 'share': 0.0, 'by_reason': {}}
    assert out['margin_m']['value'] == pytest.approx(0.05)
    assert out['margin_m']['edge'] == 'right'
    assert out['margin_m']['time_s'] == pytest.approx(T0 + 100 / 50.0 - start)
    assert out['margin_pessimistic_m']['value'] == pytest.approx(-0.02)
    assert out['margin_pessimistic_m']['edge'] == 'front-left'
    assert out['margin_pessimistic_m']['time_s'] == pytest.approx(T0 + 1.0 - start)
    assert out['stop_margin_m']['value'] == pytest.approx(0.01)
    assert out['stop_margin_m']['edge'] == 'rear-left'
    assert out['stop_margin_m']['stop_decel_m_s2'] == pytest.approx(0.8)
    assert out['energy_reserve_j']['value'] == pytest.approx(-3.0)
    assert out['energy_reserve_j']['edge'] == 'front-right'
    assert out['tilt_sources'] == {'TILT_MEASURED': rows}
    assert out['level_floor_assumed'] is False
    assert out['model_sha256'] == ['a' * 64]
    text = '\n'.join(checker.stability_lines(out))
    assert 'right edge' in text and 'level floor assumed' not in text


def test_the_unavailable_share_is_counted_per_reason_and_its_nan_figures_are_skipped():
    ep = synth.build()
    rows = 100
    unavailable = list(range(0, 10)) + list(range(50, 55))
    available = np.ones(rows, dtype=bool)
    available[unavailable] = False
    reason = np.zeros(rows, dtype=int)
    reason[0:10] = 3
    reason[50:55] = 9
    margin = column(rows, math.nan, unavailable, 0.4)
    margin[70] = 0.12
    with_stability(ep, rows=rows, available=available, reason=reason, margin_m=margin,
                   margin_pessimistic_m=column(rows, math.nan, unavailable, 0.3),
                   stop_margin_m=column(rows, math.nan, unavailable, 0.3),
                   energy_reserve_j=column(rows, math.nan, unavailable, 30.0))
    out = section(ep)
    assert out['unavailable']['n'] == 15
    assert out['unavailable']['share'] == pytest.approx(0.15)
    assert out['unavailable']['by_reason'] == {'REASON_NO_MASS_FILE': 10, 'REASON_NO_TILT': 5}
    assert out['margin_m']['value'] == pytest.approx(0.12)
    assert out['energy_reserve_j']['value'] == pytest.approx(30.0)
    lines = checker.stability_lines(out)
    assert any('15' in line and 'REASON_NO_MASS_FILE 10' in line for line in lines)


def test_a_stream_that_was_never_available_has_no_smallest_figure_and_the_line_says_so():
    ep = synth.build()
    rows = 30
    nan = np.full(rows, math.nan)
    with_stability(ep, rows=rows, available=np.zeros(rows, dtype=bool),
                   reason=np.full(rows, 1), margin_m=nan, margin_pessimistic_m=nan,
                   stop_margin_m=nan, energy_reserve_j=nan)
    out = section(ep)
    assert out['unavailable']['share'] == 1.0
    assert out['margin_m'] is None and out['stop_margin_m'] is None
    assert out['energy_reserve_j'] is None
    assert any('no finite value' in line for line in checker.stability_lines(out))


def test_level_floor_assumed_is_said_when_any_message_had_no_tilt_source():
    ep = synth.build()
    rows = 60
    tilt = np.full(rows, 1)
    tilt[40:] = 2
    out = section(with_stability(ep, rows=rows, tilt_source=tilt))
    assert out['tilt_sources'] == {'TILT_MEASURED': 40, 'TILT_LEVEL_ASSUMED': 20}
    assert out['level_floor_assumed'] is True
    assert 'level floor assumed' in '\n'.join(checker.stability_lines(out))


def test_two_mass_model_fingerprints_are_both_listed():
    ep = synth.build()
    rows = 40
    with_stability(ep, rows=rows)
    ep.streams['stability_state'].labels['model_sha256'] = (
        ['b' * 64] * 20 + ['a' * 64] * 19 + [''])
    out = section(ep)
    assert out['model_sha256'] == ['a' * 64, 'b' * 64]
    assert 'mass models: 2' in '\n'.join(checker.stability_lines(out))


def test_the_section_is_information_only_it_changes_no_verdict_reason_or_flag():
    plain = checker.check_episode(synth.build(), CHECKS)
    ep = synth.build()
    rows = 600
    with_stability(ep, rows=rows, margin_m=np.full(rows, -1.0),
                   margin_pessimistic_m=np.full(rows, -2.0),
                   stop_margin_m=np.full(rows, -3.0), energy_reserve_j=np.full(rows, -9.0),
                   available=np.arange(rows) % 2 == 0, tilt_source=np.full(rows, 2))
    report = checker.check_episode(ep, copy.deepcopy(CHECKS))
    assert report['stability']['recorded'] is True
    assert report['verdict'] == plain['verdict'] == 'PASS'
    assert report['reasons'] == plain['reasons']
    assert report['flags'] == plain['flags']
    assert report['usable'] == plain['usable']


def test_the_edge_and_reason_tables_match_the_message_definition():
    text = open(MSG).read()
    constants = {name: int(value) for name, value in re.findall(
        r'^uint8 ((?:EDGE|REASON|TILT)_[A-Z_]+) = (\d+)', text, re.M)}
    edges = {v: k[len('EDGE_'):].lower().replace('_', '-') for k, v in constants.items()
             if k.startswith('EDGE_')}
    assert edges == checker.EDGE_NAMES
    assert checker.EDGE_FRONT == 1 and checker.EDGE_FRONT_RIGHT == 8
    reasons = {v: k for k, v in constants.items() if k.startswith('REASON_')}
    assert reasons == checker.STABILITY_REASON_NAMES
    tilts = {v: k for k, v in constants.items() if k.startswith('TILT_')}
    assert tilts == checker.STABILITY_TILT_NAMES


def test_an_edge_number_outside_the_table_is_named_not_dropped():
    assert checker.edge_name(3) == 'left'
    assert checker.edge_name(99) == 'unknown 99'
