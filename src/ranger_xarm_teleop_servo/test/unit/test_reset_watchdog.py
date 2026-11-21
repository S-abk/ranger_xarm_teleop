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
"""The reset watchdog (reset_watchdog.py): it cancels only the announced goal, until it is over."""

import pytest

from ranger_xarm_teleop_servo.reset_watchdog import HEARTBEAT_TIMEOUT_S, Watchdog
from ranger_xarm_teleop_servo.session import ARM_JOINTS

GOAL = bytes(range(1, 17))
OTHER = bytes(range(2, 18))


class Rig:

    def __init__(self):
        self.w = Watchdog(ARM_JOINTS)
        self.now = 50.0
        self.seq = 0
        self.q = [0.1] * 6

    def step(self, beat=True, goal_out=True, goal=GOAL, dt=0.02):
        self.now += dt
        if beat:
            self.seq += 1
            self.w.on_heartbeat(self.now, self.seq, goal_out, goal)
        self.w.on_joints(self.now, dict(zip(ARM_JOINTS, self.q)))
        self.w.tick(self.now)

    def run(self, seconds, **kw):
        for _ in range(int(round(seconds / 0.02))):
            self.step(**kw)


def test_a_late_heartbeat_while_the_goal_runs_cancels_that_goal_only():
    rig = Rig()
    rig.run(1.0)
    rig.w.on_status(rig.now, GOAL, 2)
    rig.run(HEARTBEAT_TIMEOUT_S - 0.03, beat=False)
    assert not rig.w.stopping
    rig.run(0.06, beat=False)
    assert rig.w.stopping and rig.w.cancel_due(rig.now) == GOAL
    assert rig.w.stops == 1 and any('cancelling it' in line for line in rig.w.take_log())


def test_no_goal_out_or_a_finished_goal_is_left_alone():
    rig = Rig()
    rig.run(0.5, goal_out=False)
    rig.run(1.0, beat=False)
    assert not rig.w.stopping
    rig = Rig()
    rig.run(0.5)
    rig.w.on_status(rig.now, GOAL, 4)               # succeeded
    rig.run(1.0, beat=False)
    assert not rig.w.stopping


def test_a_goal_whose_status_never_came_is_unfinished():
    rig = Rig()
    rig.run(0.5)
    rig.run(0.3, beat=False)
    assert rig.w.stopping and rig.w.cancel_due(rig.now) == GOAL


def test_a_goal_out_without_an_id_is_ignored_never_the_zero_id():
    rig = Rig()
    rig.run(0.5, goal=bytes(16))
    rig.run(1.0, beat=False)
    assert not rig.w.stopping and rig.w.cancel_due(rig.now) is None


def test_it_keeps_cancelling_until_the_goal_ended_and_the_joints_rest():
    rig = Rig()
    rig.run(0.5)
    rig.w.on_status(rig.now, GOAL, 2)
    rig.run(0.3, beat=False)
    for _ in range(6):                              # every cancel answer is lost
        assert rig.w.cancel_due(rig.now) == GOAL
        rig.w.cancel_sent(rig.now)
        rig.run(0.5, beat=False)
        rig.q[0] += 0.001
    assert rig.w.stopping and rig.w.stop.sent_count == 6
    rig.w.cancel_sent(rig.now)
    rig.w.cancel_answered(rig.now, 0, 1)
    rig.w.on_status(rig.now, OTHER, 2)              # another goal of the server: not ours
    rig.w.on_status(rig.now, GOAL, 5)
    assert rig.w.cancel_due(rig.now + 1.0) is None
    rig.run(0.5, beat=False)
    assert rig.w.stopping                           # joints not yet seen at rest for 1 s
    rig.run(1.0, beat=False)
    assert not rig.w.stopping
    assert "7 cancel request(s) for this goal's own id, 6 unanswered" in rig.w.last_report
    assert 'are still now' in rig.w.last_report


def test_heartbeats_that_come_back_neither_undo_nor_restart_a_stop():
    rig = Rig()
    rig.run(0.5)
    rig.run(0.3, beat=False)
    assert rig.w.stopping
    rig.run(0.2)                                    # the reset node is back, same goal out
    assert rig.w.stopping and rig.w.cancel_due(rig.now) == GOAL
    rig.w.cancel_sent(rig.now)
    rig.w.cancel_answered(rig.now, 0, 1)
    rig.w.on_status(rig.now, GOAL, 5)
    rig.run(1.5, beat=False)
    assert not rig.w.stopping and rig.w.stops == 1
    rig.run(1.0, beat=False)
    assert not rig.w.stopping and rig.w.stops == 1  # that goal is never watched again
    rig.run(0.5, goal=OTHER)                        # a new goal is
    rig.run(0.3, beat=False)
    assert rig.w.stopping and rig.w.cancel_due(rig.now) == OTHER and rig.w.stops == 2


def test_an_unknown_goal_ends_after_the_settle_time():
    rig = Rig()
    rig.run(0.5)
    rig.run(0.3, beat=False)
    for _ in range(10):
        if rig.w.cancel_due(rig.now) is not None:
            rig.w.cancel_sent(rig.now)
            rig.w.cancel_answered(rig.now, 2, 0)
        rig.run(0.5, beat=False)
    assert not rig.w.stopping
    assert 'knew no goal with this id' in rig.w.last_report


def test_the_timeout_must_be_positive():
    with pytest.raises(ValueError):
        Watchdog(ARM_JOINTS, heartbeat_timeout_s=0.0)
