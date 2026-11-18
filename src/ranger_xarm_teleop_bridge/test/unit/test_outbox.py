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
"""INV-4: nothing is sent late and nothing queues without bound (no ROS)."""

from ranger_xarm_teleop_bridge.outbox import Outbox


class Clock:

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_haptics_are_sent_in_order_with_their_age():
    c = Clock()
    box = Outbox(clock=c)
    box.put_haptic({'id': 1}, 0.1, 0.75)
    c.t += 0.2
    box.put_haptic({'id': 2}, 0.0, 0.75)
    out = box.take()
    assert [p['id'] for p, _ in out] == [1, 2]
    assert abs(out[0][1] - 0.3) < 1e-9 and abs(out[1][1]) < 1e-9
    assert box.take() == [] and len(box) == 0


def test_a_haptic_older_than_its_maximum_age_is_dropped():
    c = Clock()
    box = Outbox(clock=c)
    box.put_haptic({'id': 1}, 0.5, 0.75)
    c.t += 0.3
    assert box.take() == [] and box.dropped_old_haptic == 1


def test_the_link_down_for_a_minute_keeps_the_outbox_bounded():
    c = Clock()
    box = Outbox(haptic_max=8, clock=c)
    for i in range(600):
        box.put_haptic({'id': i}, 0.0, 0.75)
        box.set_status({'seq': i}, 0.0)
        c.t += 0.1
        assert len(box) <= 9
    assert box.dropped_overflow == 592
    c.t += 1.0
    assert box.take() == []                          # everything too old by now


def test_status_keeps_only_the_newest_and_ages_out():
    c = Clock()
    box = Outbox(status_max_age_s=0.4, clock=c)
    box.set_status({'seq': 1}, 0.0)
    box.set_status({'seq': 2}, 0.1)
    out = box.take()
    assert [p for p, _ in out] == [{'seq': 2}] and box.replaced_status == 1
    assert abs(out[0][1] - 0.1) < 1e-9
    box.set_status({'seq': 3}, 0.0)
    c.t += 0.41
    assert box.take() == [] and box.dropped_old_status == 1


def test_expire_drops_without_sending():
    c = Clock()
    box = Outbox(clock=c)
    box.put_haptic({'id': 1}, 0.0, 0.75)
    box.put_haptic({'id': 2}, 0.0, 2.0)
    box.set_status({'seq': 1}, 0.0)
    c.t += 1.0
    box.expire()
    assert len(box) == 1 and [p['id'] for p, _ in box.take()] == [2]
