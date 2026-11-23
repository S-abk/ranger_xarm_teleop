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
"""The arm-state rule on the driver's xarm_msgs/msg/RobotMsg topic, from synthetic captures."""

from types import SimpleNamespace

from ranger_xarm_teleop_check import hw_check, rules
from ranger_xarm_teleop_check.rules import FAIL, INFO, NOT_JUDGED, Options, PASS
from test_hw_check_rules import idle_data

RULE = 'arm state (xarm_msgs/RobotMsg)'
TOPIC = '/xarm_xarm/robot_states'     # the real driver's, in its hardware namespace
TYPE = 'xarm_msgs/msg/RobotMsg'
THRESHOLD = 'err 0, mode 1 (servo), state <= 2 (1 running, 2 ready) in every message'


def message(t, err=0, warn=0, state=2, mode=1, brake=63, able=63):
    return (t, err, warn, state, mode, brake, able)


def capture(n=50, **last):
    """Return ``data['robot_states']`` for ``n`` messages at 5 Hz; ``last`` edits the last."""
    messages = [message(1000.0 + 0.2 * i) for i in range(n)]
    if last:
        messages[-1] = message(1000.0 + 0.2 * (n - 1), **last)
    return {'topics': {TOPIC: ['/xarm/driver']}, 'topic': TOPIC, 'unavailable': None,
            'messages': messages}


def judged(robot_states):
    data = idle_data(robot_states=robot_states)
    _, results = rules.evaluate('idle', data, Options())
    return next(r for r in results if r.rule == RULE)


def test_all_messages_good_pass():
    r = judged(capture())
    assert r.verdict == PASS and r.threshold == THRESHOLD
    assert f'50 messages on {TOPIC} at 5.0 Hz' in r.measured
    assert 'last err 0, warn 0, state 2, mode 1' in r.measured
    assert '0 of 50 messages broke the rule' in r.measured
    assert r.line().startswith(f'PASS  {RULE}: ')


def test_state_1_running_passes():
    assert judged(capture(state=1)).verdict == PASS


def test_one_message_with_an_error_fails_and_names_it():
    r = judged(capture(err=3))
    assert r.verdict == FAIL
    assert 'err != 0 in 1 of 50 messages' in r.measured
    assert 'last err 3' in r.measured and '1 of 50 messages broke the rule' in r.measured
    assert "clear the error in the vendor's software and restart from stage 2" in r.advice
    assert r.advice.startswith('stop')
    assert 'err 3' in r.advice
    assert 'to do: stop' in r.line()


def test_mode_not_servo_fails():
    r = judged(capture(mode=0))
    assert r.verdict == FAIL
    assert 'mode != 1 in 1 of 50' in r.measured and 'last err 0' in r.measured
    assert 'set servo mode (mode 1) as the procedure says' in r.advice
    assert 'error' not in r.advice


def test_state_above_two_fails():
    r = judged(capture(state=4))
    assert r.verdict == FAIL
    assert 'state > 2 in 1 of 50' in r.measured and 'state 4' in r.measured
    assert 'the arm is not ready (stopped, paused or configuration changed)' in r.advice


def test_a_message_breaking_two_ways_counts_once_and_gives_both_advices():
    r = judged(capture(err=7, state=5))
    assert r.verdict == FAIL
    assert '1 of 50 messages broke the rule' in r.measured
    assert 'err != 0 in 1' in r.measured and 'state > 2 in 1' in r.measured
    assert 'vendor' in r.advice and 'not ready' in r.advice


def test_no_topic_is_info_not_available_and_leaves_a_simulation_clean():
    r = judged(None)
    assert r.verdict == INFO and r.measured == f'not available: no publisher of {TYPE}'
    absent = judged({'topics': {}, 'topic': None, 'unavailable': None, 'messages': []})
    assert absent.verdict == INFO and absent.measured == r.measured
    _, results = rules.evaluate('idle', idle_data(), Options())
    assert rules.exit_status([x for x in results if x.rule == RULE]) == 0


def test_xarm_msgs_missing_is_info_not_available():
    r = judged({'topics': {TOPIC: ['/xarm/driver']}, 'topic': None,
                'unavailable': 'xarm_msgs not installed', 'messages': []})
    assert r.verdict == INFO and r.measured == 'not available (xarm_msgs not installed)'


def test_a_publisher_that_sent_nothing_is_not_judged():
    r = judged({'topics': {TOPIC: ['/xarm/driver']}, 'topic': TOPIC, 'unavailable': None,
                'messages': []})
    assert r.verdict == NOT_JUDGED and f'/xarm/driver on {TOPIC}' in r.measured and r.advice


def test_two_topics_of_the_type_are_named_and_neither_is_judged():
    r = judged({'topics': {TOPIC: ['/xarm/driver'], '/xarm/robot_states': ['/standin']},
                'topic': None, 'unavailable': None, 'messages': []})
    assert r.verdict == NOT_JUDGED
    assert r.measured == (f'2 publishers of {TYPE}: /standin on /xarm/robot_states; '
                          f'/xarm/driver on {TOPIC}; none judged')
    assert 'stop every publisher of that type' in r.advice


def test_two_publishers_on_one_topic_are_named_and_not_judged():
    r = judged({'topics': {TOPIC: ['/a', '/b']}, 'topic': None, 'unavailable': None,
                'messages': []})
    assert r.verdict == NOT_JUDGED and f'/a on {TOPIC}; /b on {TOPIC}' in r.measured


class _Graph:
    """A node's graph queries, for the capture's topic lookup."""

    def __init__(self, topics):
        self.topics = topics        # {topic: (type, [publisher node names])}

    def get_topic_names_and_types(self):
        return [(t, [kind]) for t, (kind, _) in self.topics.items()]

    def get_publishers_info_by_topic(self, topic):
        return [SimpleNamespace(node_namespace='/', node_name=n.lstrip('/'))
                for n in self.topics[topic][1]]


def _poll(topics):
    data, subscribed = {}, []
    reader = hw_check._RobotStates(_Graph(topics), data,
                                   lambda kind, topic, *rest: subscribed.append(topic))
    reader.poll(0.0)
    return data['robot_states'], subscribed


def test_the_capture_finds_the_topic_by_type_whatever_its_name():
    record, subscribed = _poll({TOPIC: (TYPE, ['driver']), '/joint_states':
                                ('sensor_msgs/msg/JointState', ['jsb'])})
    assert record['topics'] == {TOPIC: ['/driver']}
    if record['unavailable'] is None:        # xarm_msgs installed: subscribed to that topic
        assert subscribed == [TOPIC] and record['topic'] == TOPIC
    else:
        assert subscribed == [] and record['unavailable'] == 'xarm_msgs not installed'


def test_the_capture_reads_nothing_when_several_publish_and_ignores_subscriber_only_topics():
    record, subscribed = _poll({TOPIC: (TYPE, ['driver']), '/xarm/robot_states':
                                (TYPE, ['standin']), '/listener': (TYPE, [])})
    assert record['topics'] == {TOPIC: ['/driver'], '/xarm/robot_states': ['/standin']}
    assert subscribed == [] and record['topic'] is None and record['messages'] == []
    record, subscribed = _poll({'/listener': (TYPE, [])})
    assert record['topics'] == {} and subscribed == []


def test_two_publishers_of_one_name_both_count():
    record, subscribed = _poll({TOPIC: (TYPE, ['driver', 'driver'])})
    assert record['topics'] == {TOPIC: ['/driver', '/driver']} and subscribed == []


def test_axis_bits_are_six_characters_with_joint_1_first():
    assert rules.axis_bits(63) == '111111'
    assert rules.axis_bits(0) == '000000'
    assert rules.axis_bits(1) == '100000'          # bit 0 is joint 1
    assert rules.axis_bits(0b100000) == '000001'   # bit 5 is joint 6
    assert rules.axis_bits(0b000110) == '011000'
    assert rules.axis_bits(-1) == '111111'         # int16 sign bits are not axes
    assert rules.axis_bits(64 + 3) == '110000'     # nothing beyond six axes


def test_the_bit_strings_are_shown_in_the_stated_order():
    r = judged(capture(brake=0b000011, able=0b100001))
    assert 'mt_brake 110000, mt_able 100001' in r.measured
    assert '6 bits, joint 1 first' in r.measured


def test_measures_carry_the_counts_and_last_values_for_the_report():
    measures, _ = rules.evaluate('idle', idle_data(robot_states=capture(err=3, mode=0)),
                                 Options())
    am = measures['arm_state']
    assert am['messages'] == 50 and am['bad_err'] == 1 and am['bad_mode'] == 1
    assert am['bad_state'] == 0 and am['bad_messages'] == 1
    assert am['last']['err'] == 3 and am['last']['mt_brake'] == '111111'


def test_only_the_idle_stage_judges_it():
    for stage in ('motion', 'teleop'):
        data = idle_data(robot_states=capture(err=3))
        _, results = rules.evaluate(stage, data, Options())
        assert RULE not in {r.rule for r in results}


def test_a_search_with_no_publisher_says_how_long_it_looked():
    r = judged({'topics': {}, 'topic': None, 'unavailable': None, 'messages': [],
                'searched_s': 4.04})
    assert r.verdict == INFO
    assert r.measured == f'not available: no publisher of {TYPE} after 4.0 s'


def test_the_poll_records_how_long_the_type_was_looked_for():
    record, subscribed = _poll({'/joint_states': ('sensor_msgs/msg/JointState', ['jsb'])})
    assert record['topics'] == {} and record['searched_s'] == 0.0 and subscribed == []


class _LateGraph:
    """A graph on a fake clock: the arm state publisher appears ``late_s`` after the start."""

    def __init__(self, late_s):
        self.now = 0.0
        self.late_s = late_s

    def spin(self):
        self.now += 0.05

    def _topics(self):
        topics = {'/joint_states': ('sensor_msgs/msg/JointState', ['jsb'])}
        if self.late_s is not None and self.now >= self.late_s:
            topics[TOPIC] = (TYPE, ['driver'])
        return topics

    def get_topic_names_and_types(self):
        return [(t, [kind]) for t, (kind, _) in self._topics().items()]

    def get_publishers_info_by_topic(self, topic):
        names = self._topics().get(topic, (None, []))[1]
        return [SimpleNamespace(node_namespace='/', node_name=n) for n in names]


def _discover(graph, arm_state=True):
    return hw_check._discover(graph, graph.spin, '/joint_states', arm_state,
                              clock=lambda: graph.now)


def test_discovery_waits_for_an_arm_state_publisher_that_appears_late():
    graph = _LateGraph(late_s=3.0)
    waited = _discover(graph)
    assert 3.0 <= waited < 3.1
    record, subscribed = {}, []
    reader = hw_check._RobotStates(graph, record, lambda kind, topic, *r: subscribed.append(topic))
    reader.poll(waited)
    assert record['robot_states']['topics'] == {TOPIC: ['/driver']}


def test_discovery_gives_up_at_its_limit_and_does_not_wait_outside_the_idle_stage():
    assert hw_check.DISCOVERY_MAX_S <= _discover(_LateGraph(late_s=None)) \
        < hw_check.DISCOVERY_MAX_S + 0.1
    assert hw_check.DISCOVERY_MIN_S <= _discover(_LateGraph(late_s=None), arm_state=False) \
        < hw_check.DISCOVERY_MIN_S + 0.1
    early = _discover(_LateGraph(late_s=0.5))
    assert hw_check.DISCOVERY_MIN_S <= early < hw_check.DISCOVERY_MIN_S + 0.1
