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
"""Another node's parameters are read in three requests, however many have no value."""

from ranger_xarm_teleop_record import parameters as pr

DOUBLE, STRING = 3, 4       # rcl_interfaces/msg/ParameterType


class Asker:
    """A node with ``count`` parameters, every third one without a value; counts requests."""

    def __init__(self, count):
        self.names = [f'p{i}' for i in range(count)]
        self.types = [pr.NOT_SET if i % 3 == 0 else DOUBLE for i in range(count)]
        self.requests = []

    def read(self, read, now=0.0):
        assert read.due(now)
        assert read.begin(now) is None
        self.requests.append(pr.NAMES)
        names = read.listed(self.names, now)
        self.requests.append(pr.TYPES)
        having = read.typed([self.types[self.names.index(n)] for n in names], now)
        self.requests.append(pr.VALUES)
        # get_parameters answers nothing at all when one name has no value.
        values = ([] if any(self.types[self.names.index(n)] == pr.NOT_SET for n in having)
                  else [float(n[1:]) for n in having])
        return read.valued(values)


def test_three_requests_whatever_the_number_of_unset_parameters():
    # MoveIt Servo: 135 parameters, 48 without a value. Name by name that was about 150
    # requests in a few tens of milliseconds.
    node = Asker(135)
    read = pr.ParameterRead()
    assert node.read(read)
    assert node.requests == [pr.NAMES, pr.TYPES, pr.VALUES]
    assert read.done and not read.due(1000.0)
    assert read.unset == [f'p{i}' for i in range(0, 135, 3)]
    assert read.values == {f'p{i}': float(i) for i in range(135) if i % 3}
    assert len(read.values) + len(read.unset) == 135


def test_no_unset_parameter_is_ever_asked_for_its_value():
    read = pr.ParameterRead()
    read.begin(0.0)
    names = read.listed(['a', 'b', 'c', 'd'], 0.0)
    assert names == ['a', 'b', 'c', 'd']
    assert read.typed([DOUBLE, pr.NOT_SET, STRING, pr.NOT_SET], 0.0) == ['a', 'c']
    assert read.unset == ['b', 'd']
    assert read.valued([1.5, 'x'])
    assert read.values == {'a': 1.5, 'c': 'x'}


def test_a_node_that_has_not_declared_its_parameters_yet_is_asked_again():
    read = pr.ParameterRead()
    read.begin(0.0)
    assert read.listed(['use_sim_time', 'start_type_description_service'], 0.0) is None
    assert not read.done and read.due(0.0)            # at the next tick, not after a timeout
    assert read.begin(1.0) is None                    # an answer came: not a silent step
    assert read.restarts == 0


def test_a_step_without_an_answer_starts_the_read_again_and_is_named():
    read = pr.ParameterRead(timeout_s=3.0)
    read.begin(0.0)
    assert not read.due(2.9)                          # the names request is out
    assert read.due(3.0)
    assert read.begin(3.0) == pr.NAMES                # the silent step, for the log line
    read.listed(['a', 'b', 'c'], 3.5)
    assert not read.due(6.0) and read.due(6.5)        # the clock runs from each step
    assert read.begin(6.5) == pr.TYPES
    read.listed(['a', 'b', 'c'], 7.0)
    read.typed([DOUBLE, DOUBLE, DOUBLE], 7.5)
    assert not read.due(10.0) and read.due(10.5) and read.begin(10.5) == pr.VALUES
    assert read.restarts == 3 and not read.done


def test_an_answer_that_does_not_fit_starts_the_read_again():
    read = pr.ParameterRead()
    read.begin(0.0)
    read.listed(['a', 'b', 'c'], 0.0)
    assert read.typed([DOUBLE, DOUBLE], 0.0) is None  # a name went away in between
    assert not read.done and read.due(0.0)
    read.begin(1.0)
    read.listed(['a', 'b', 'c'], 1.0)
    read.typed([DOUBLE, DOUBLE, DOUBLE], 1.0)
    assert not read.valued([])                        # one value was unset in between
    assert not read.done and read.due(1.0) and not read.values


def test_an_answer_to_a_step_that_is_not_out_is_not_used():
    read = pr.ParameterRead()
    assert read.listed(['a', 'b', 'c'], 0.0) is None  # nothing was asked
    read.begin(0.0)
    assert read.typed([DOUBLE], 0.0) is None          # the names are out, not the types
    read.begin(0.0)
    read.listed(['a', 'b', 'c'], 0.0)
    assert not read.valued([1.0, 2.0, 3.0])           # the types are out, not the values
    assert not read.done


def test_a_failed_request_starts_the_read_again_at_the_next_tick():
    read = pr.ParameterRead()
    read.begin(0.0)
    read.failed()
    assert read.due(0.0) and read.begin(0.5) is None


def test_a_node_whose_parameters_all_lack_a_value_is_read_in_two_requests():
    read = pr.ParameterRead()
    read.begin(0.0)
    read.listed(['a', 'b', 'c'], 0.0)
    assert read.typed([pr.NOT_SET] * 3, 0.0) == []
    assert read.done and read.values == {} and read.unset == ['a', 'b', 'c']
