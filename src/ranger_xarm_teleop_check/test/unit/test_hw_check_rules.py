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
"""Every rule with its PASS, FAIL and NOT JUDGED cases, and the stage evaluation."""

import math

from ranger_xarm_teleop_check import rules, stats
from ranger_xarm_teleop_check.rules import FAIL, NOT_JUDGED, Options, PASS


def arm_samples(n=600, rate=150.0, speed=0.0, hold=1, offset=0.0, gaps=()):
    """Return ``data['arm']`` tuples for a steady stream (``gaps``: {index: extra seconds})."""
    out = []
    t = 1000.0
    for i in range(n):
        t += 1.0 / rate + gaps.get(i, 0.0) if hasattr(gaps, 'get') else 1.0 / rate
        k = (i // hold) * hold
        pos = tuple(speed * k / rate * (j + 1) for j in range(6))
        out.append((t, t + offset, t, pos, None, False))
    return out


def by_rule(results):
    return {r.rule: r for r in results}


def idle_data(**kw):
    gripper = [(1000.0 + i / 10, 1000.0 + i / 10) for i in range(30)]
    data = {'arm': arm_samples(), 'gripper': gripper,
            'js_publishers': ['/controller_manager'], 'clock_publishers': [],
            'activity': {'xarm6_traj_controller': 'active'}, 'controller_state': {},
            'description': None, 'states': [], 'targets': [],
            'rosout': {'overruns': 0, 'read_failures': 0},
            'machine': {'governors': {'cpu0': 'performance'}, 'governor_set': ['performance']},
            'versions': {'controller_manager': '4.48.1'},
            'gate': {'verdict': PASS, 'head': 'abc', 'describe': 'x', 'evidence': 'git'}}
    data.update(kw)
    return data


def test_exit_status_order():
    P, F, N = (rules.Result('r', v, '', '') for v in (PASS, FAIL, NOT_JUDGED))
    I = rules.Result('r', rules.INFO, '', '')   # noqa: E741
    assert rules.exit_status([P, I]) == 0
    assert rules.exit_status([P, N]) == 2
    assert rules.exit_status([P, N, F]) == 1
    assert rules.exit_status([]) == 0


def test_result_line_format():
    r = rules.Result('rate', FAIL, '10 Hz', '>= 5 Hz', 'do it')
    assert r.line().splitlines()[0] == 'FAIL  rate: 10 Hz (rule: >= 5 Hz)'
    assert 'do it' in r.line().splitlines()[1]
    assert '\n' not in rules.Result('rate', PASS, '1', '2', 'x').line()
    assert rules.Result('r', NOT_JUDGED, 'm', 't', 'why').as_dict()['reason'] == 'why'


def test_rate_pass_and_fail_and_none():
    m = rules.joint_state_measures(idle_data(), 150.0)
    assert rules.rule_rate(m).verdict == PASS
    slow = idle_data(arm=arm_samples(rate=100.0))
    r = rules.rule_rate(rules.joint_state_measures(slow, 150.0))
    assert r.verdict == FAIL and 'if 150 Hz is not held' in r.advice
    r = rules.rule_rate(rules.joint_state_measures(idle_data(arm=[]), 150.0))
    assert r.verdict == NOT_JUDGED


def test_rate_uses_the_worse_of_arrival_and_stamps():
    arm = [(t, t, t * 0.5, p, v, e) for (t, _, _, p, v, e) in arm_samples()]
    r = rules.rule_rate(rules.joint_state_measures(idle_data(arm=arm), 150.0))
    assert r.verdict == PASS
    arm = [(a, b, c * 1.2, p, v, e) for (a, b, c, p, v, e) in arm_samples()]   # stamps 125 Hz
    assert rules.rule_rate(rules.joint_state_measures(idle_data(arm=arm), 150.0)
                           ).verdict == FAIL


def test_gap_cases():
    gappy = idle_data(arm=arm_samples(gaps={300: 0.2}))
    m = rules.joint_state_measures(gappy, 150.0)
    no_limits = rules.rule_gap(m, Options())
    assert no_limits.verdict == NOT_JUDGED and 'suggested minimum timeout' in no_limits.measured
    ok = rules.rule_gap(m, Options(limits_given=True, tool_pose_timeout_s=1.0))
    assert ok.verdict == PASS
    bad = rules.rule_gap(m, Options(limits_given=True, tool_pose_timeout_s=0.3))
    assert bad.verdict == FAIL
    for value in (None, 'x', True, math.nan, 0, -1):
        r = rules.rule_gap(m, Options(limits_given=True, tool_pose_timeout_s=value))
        assert r.verdict == NOT_JUDGED and 'tool_pose_timeout_s in your limits file' in r.advice
    empty = rules.joint_state_measures(idle_data(arm=[]), 150.0)
    assert rules.rule_gap(empty, Options()).verdict == NOT_JUDGED


def test_effort_is_informational_only():
    assert 'absent' in rules.info_effort(idle_data()).measured
    arm = [s[:5] + (True,) for s in arm_samples(10)]
    r = rules.info_effort(idle_data(arm=arm))
    assert r.verdict == rules.INFO and r.measured.startswith('present')
    assert rules.info_effort(idle_data(arm=[])).verdict == rules.INFO


def test_controllers_report_and_expectations():
    data = idle_data(activity={'a': 'active', 'b': 'inactive'}, controller_state={'a': 5})
    opts = Options(controllers=('a', 'b'), expect_active=('a',), expect_inactive=('b',))
    r = by_rule(rules.rules_controllers(data, opts))
    assert r['controller a active'].verdict == PASS
    assert r['controller b inactive'].verdict == PASS
    opts = Options(controllers=('a', 'b'), expect_active=('b', 'c'), expect_inactive=('a',))
    r = by_rule(rules.rules_controllers(data, opts))
    assert r['controller b active'].verdict == FAIL
    assert r['controller c active'].verdict == FAIL and 'not loaded' in r[
        'controller c active'].measured
    assert r['controller a inactive'].verdict == FAIL
    assert rules.rules_controllers(data, Options(controllers=('a',)))[0].verdict == rules.INFO


def test_controllers_without_activity_never_asks_the_service():
    data = idle_data(activity=None, controller_state={'a': 3, 'b': 0})
    opts = Options(controllers=('a', 'b'), expect_active=('a', 'b'), expect_inactive=('b',))
    r = rules.rules_controllers(data, opts)
    assert 'older controller_manager' in r[0].measured
    named = by_rule(r)
    assert named['controller a active'].verdict == PASS
    assert named['controller b active'].verdict == FAIL
    assert named['controller b inactive'].verdict == PASS


def _desc(**kw):
    d = {'sha256': 'ab' * 32, 'plugins': ['p/Hardware'], 'simulator': False, 'gripper': True,
         'velocity_control': ['false'], 'error': None}
    d.update(kw)
    return d


def test_description_rules():
    r = by_rule(rules.rules_description(_desc(), Options()))
    assert r['velocity_control'].verdict == PASS
    assert 'HARDWARE' in r['description backend'].measured
    assert 'fingerprint' not in ' '.join(r)
    r = by_rule(rules.rules_description(_desc(velocity_control=['true']), Options()))
    assert r['velocity_control'].verdict == FAIL
    assert 'xarm_velocity_control:=false' in r['velocity_control'].advice
    r = by_rule(rules.rules_description(_desc(velocity_control=[]), Options()))
    assert r['velocity_control'].verdict == NOT_JUDGED
    r = by_rule(rules.rules_description(_desc(simulator=True), Options()))
    assert 'MOCK/SIMULATOR' in r['description backend'].measured
    r = by_rule(rules.rules_description(_desc(error='bad xml', plugins=[], simulator=None),
                                        Options()))
    assert r['velocity_control'].verdict == NOT_JUDGED


def test_description_fingerprint():
    ok = by_rule(rules.rules_description(_desc(), Options(expect_fingerprint='AB' * 32)))
    assert ok['description fingerprint'].verdict == PASS
    bad = by_rule(rules.rules_description(_desc(), Options(expect_fingerprint='00')))
    assert bad['description fingerprint'].verdict == FAIL
    assert 'changed since' in bad['description fingerprint'].advice
    none = by_rule(rules.rules_description(None, Options(expect_fingerprint='00')))
    assert {v.verdict for v in none.values()} == {NOT_JUDGED}
    assert 'description fingerprint' in none


def test_clock_rules():
    m = rules.joint_state_measures(idle_data(), 150.0)
    r = rules.rule_clock(idle_data(), m)
    assert [x.verdict for x in r] == [PASS, PASS]
    r = rules.rule_clock(idle_data(clock_publishers=['/sim']), m)
    assert r[0].verdict == FAIL and 'use_sim_time' in r[0].advice
    shifted = idle_data(arm=arm_samples(offset=0.2))
    r = rules.rule_clock(shifted, rules.joint_state_measures(shifted, 150.0))
    assert r[1].verdict == FAIL
    empty = idle_data(arm=[])
    r = rules.rule_clock(empty, rules.joint_state_measures(empty, 150.0))
    assert r[1].verdict == NOT_JUDGED


def test_governor_rule():
    ok = {'governors': {'cpu0': 'performance', 'cpu1': 'performance'},
          'governor_set': ['performance']}
    assert rules.rule_governor(ok).verdict == PASS
    bad = {'governors': {'cpu0': 'performance', 'cpu1': 'powersave'},
           'governor_set': ['performance', 'powersave']}
    r = rules.rule_governor(bad)
    assert r.verdict == FAIL and 'performance' in r.advice
    assert rules.rule_governor({'governors': {}}).verdict == NOT_JUDGED
    assert rules.rule_governor({}).verdict == NOT_JUDGED


def _states(period=0.02, n=200, extra=None):
    t = 50.0
    out = []
    for i in range(n):
        t += period + (extra or {}).get(i, 0.0)
        out.append((t, t, False, True, (0.0, 0.0, 0.0)))
    return out


def test_teleop_cycle_rules():
    r = rules.rules_teleop_cycle(idle_data(states=_states()))
    assert [x.verdict for x in r] == [PASS] * 3
    r = rules.rules_teleop_cycle(idle_data(states=_states(extra={50: 0.1})))
    assert [x.verdict for x in r] == [PASS, FAIL, FAIL]   # p99 fine; max 0.12 and gap 0.12
    jitter = {i: 0.015 for i in range(0, 200, 3)}
    r = rules.rules_teleop_cycle(idle_data(states=_states(extra=jitter)))
    assert r[0].verdict == FAIL
    r = rules.rules_teleop_cycle(idle_data(states=[]))
    assert [x.verdict for x in r] == [NOT_JUDGED] * 3
    assert r[0].advice == 'the teleop stack is not running'


def test_overruns_rule():
    ok = rules.rule_overruns({'rosout': {'overruns': 0, 'read_failures': 0}})
    assert ok.verdict == PASS
    bad = rules.rule_overruns({'rosout': {'overruns': 2, 'read_failures': 1,
                                          'overrun_examples': ['cm: Overrun detected'],
                                          'read_examples': ['drv: Read() returns 3']}})
    assert bad.verdict == FAIL and 'Overrun detected' in bad.measured
    assert rules.rule_overruns({}).verdict == PASS


def test_driver_gate_rule():
    gate = {'verdict': PASS, 'head': 'abcdef0123456789', 'describe': 'v1', 'evidence': 'git'}
    assert rules.rule_driver_gate(gate).verdict == PASS
    r = rules.rule_driver_gate(dict(gate, verdict=FAIL))
    assert r.verdict == FAIL and 'cc44f9f' in r.advice and 'update the driver' in r.advice
    reason = 'workspace not given: pass --workspace'
    r = rules.rule_driver_gate({'verdict': NOT_JUDGED, 'reason': reason})
    assert r.verdict == NOT_JUDGED and 'workspace not given' in r.measured


def motion_data(**kw):
    return idle_data(arm=arm_samples(n=600, speed=0.5, **kw))


def test_stale_rule():
    r = rules.rule_stale(rules.motion_measures(motion_data()))
    assert r.verdict == PASS
    r = rules.rule_stale(rules.motion_measures(motion_data(hold=30)))
    assert r.verdict == FAIL and 'procedure stage 3' in r.advice
    r = rules.rule_stale(rules.motion_measures(idle_data()))
    assert r.verdict == NOT_JUDGED and 'no motion seen' in r.advice


def test_acceleration_rule():
    mm = rules.motion_measures(motion_data())
    assert rules.rule_acceleration(mm, Options()).verdict == NOT_JUDGED
    assert 'rad/s^2' in rules.rule_acceleration(mm, Options()).measured
    limits = {f'xarm_joint{i}': 5.0 for i in range(1, 7)}
    ok = Options(joint_limits_given=True, joint_acceleration_limits=limits)
    assert rules.rule_acceleration(mm, ok).verdict == PASS
    mm['peak_acceleration']['xarm_joint2'] = 9.0
    r = rules.rule_acceleration(mm, ok)
    assert r.verdict == FAIL and 'xarm_joint2' in r.measured
    none = Options(joint_limits_given=True, joint_acceleration_limits={'xarm_joint1': None})
    assert rules.rule_acceleration(mm, none).verdict == NOT_JUDGED


def test_oscillation_rule():
    mm = rules.motion_measures(motion_data())
    assert rules.rule_oscillation(mm).verdict == PASS
    t = [1000.0 + i / 150 for i in range(600)]
    arm = []
    for i, x in enumerate(t):
        wrist = 0.1 * math.sin(2 * math.pi * 4 * x) / (2 * math.pi * 4) * 5
        arm.append((x, x, x, (0.0, 0.0, 0.0, wrist, 0.0, 0.0), None, False))
    mm = rules.motion_measures(idle_data(arm=arm))
    r = rules.rule_oscillation(mm)
    assert r.verdict == FAIL and 'limit cycle near the wrist singularity' in r.advice
    assert rules.rule_oscillation(rules.motion_measures(idle_data(arm=[]))
                                  ).verdict == NOT_JUDGED


def _teleop_data(lead=0.01, limits=None):
    n = 300
    t = 100.0
    states, targets = [], []
    for i in range(n):
        t += 0.02
        x = 0.1 * math.sin(i / 20)
        states.append((t, t, True, True, (x - lead, 0.0, 0.0)))
        targets.append((t, (x, 0.0, 0.0)))
    return idle_data(states=states, targets=targets)


def test_lead_rule():
    tm = rules.teleop_measures(_teleop_data(lead=0.01))
    assert abs(tm['lead']['p99'] - 0.01) < 1e-9 and tm['lead']['count'] == 300
    ok = Options(limits_given=True, max_lead_translation_m=0.05)
    assert rules.rules_lead(tm, ok)[0].verdict == PASS
    tight = Options(limits_given=True, max_lead_translation_m=0.005)
    assert rules.rules_lead(tm, tight)[0].verdict == FAIL
    assert rules.rules_lead(tm, Options())[0].verdict == NOT_JUDGED
    missing = Options(limits_given=True, max_lead_translation_m=None)
    r = rules.rules_lead(tm, missing)[0]
    assert r.verdict == NOT_JUDGED and 'max_lead_translation_m in your limits' in r.advice
    empty = rules.teleop_measures(idle_data())
    assert rules.rules_lead(empty, ok)[0].verdict == NOT_JUDGED
    assert rules.rules_lead(tm, ok)[1].verdict == rules.INFO


def _cam(rate=30.0, n=300, width=848, gaps=0, enc='rgb8'):
    t = [i / rate for i in range(n)]
    for k in range(gaps):
        j = 10 + 20 * k
        t = t[:j] + [x + 1.0 for x in t[j:]]
    return {'mono': t, 'stamps': list(t), 'first': (width, 480, enc)}


STREAMS = [{'name': 'cam', 'topic': '/cam/image', 'rate_hz': 30.0,
            'image': {'width': 848, 'height': 480, 'encoding': 'rgb8'}}]


def _cams(rec):
    data = {'cameras': {'cam': rec}}
    cm = rules.camera_measures(data, STREAMS)
    return by_rule(rules.rules_cameras(cm, STREAMS))


def test_camera_rules_pass():
    r = _cams(_cam())
    assert {x.verdict for x in r.values()} == {PASS}


def test_camera_rules_fail_cases():
    assert _cams(_cam(rate=20.0))['cam rate'].verdict == FAIL
    assert _cams(_cam(gaps=5))['cam gaps'].verdict == FAIL        # 5 of 300 = 1.7 %
    assert _cams(_cam(gaps=2))['cam gaps'].verdict == PASS        # 0.67 %
    assert _cams(_cam(width=640))['cam format'].verdict == FAIL
    assert _cams(_cam(enc='bgr8'))['cam format'].verdict == FAIL


def test_camera_rules_without_frames():
    r = _cams({'mono': [], 'stamps': [], 'first': None})
    assert {x.verdict for x in r.values()} == {NOT_JUDGED}


def test_camera_stage_evaluates_cameras_only():
    opts = Options(streams=STREAMS)
    measures, results = rules.evaluate('cameras', {'cameras': {'cam': _cam()}}, opts)
    assert set(measures) == {'cameras'} and rules.exit_status(results) == 0


def test_stage_composition():
    opts = Options()
    names = {s: {r.rule for r in rules.evaluate(s, motion_data() if s != 'idle'
                                                else idle_data(), opts)[1]}
             for s in ('idle', 'motion', 'teleop')}
    assert 'joint_states rate' in names['idle'] and 'wrist oscillation' not in names['idle']
    assert names['idle'] < names['motion'] < names['teleop']
    assert 'target lag' in names['teleop'] and 'stale repeats while moving' in names['motion']


def test_idle_with_a_clean_stack_is_judged_apart_from_the_unjudgeable():
    measures, results = rules.evaluate('idle', idle_data(), Options())
    verdicts = {r.rule: r.verdict for r in results}
    assert verdicts['joint_states rate'] == PASS and verdicts['no /clock publisher'] == PASS
    assert verdicts['joint_states largest gap'] == NOT_JUDGED
    assert rules.exit_status(results) == 2
    assert measures['joint_states']['gripper_messages'] == 30
    assert abs(measures['joint_states']['gripper_rate_hz'] - 10.0) < 1e-6


def test_gripper_only_messages_do_not_count_as_arm():
    kind = stats.classify_joint_state(['xarm_drive_joint'], [0.0])[0]
    data = idle_data(arm=[])
    assert kind == 'gripper' and rules.joint_state_measures(data, 150.0)['arm_messages'] == 0
