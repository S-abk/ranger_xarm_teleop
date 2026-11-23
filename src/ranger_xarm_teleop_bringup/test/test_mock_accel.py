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
The joint acceleration limit of the teleop command (G-18) in mock execution.

Over fixtures that exercise each source of acceleration measured before the limit existed
(a release while moving, a fast hand into the speed cap, a step, full reach, a joint bound,
the scale switch, the link keep-out, a G-4 fault, the wrist singularity, and an engagement
with the hand already moving):

* the joint command the teleop node sends (``JointJog``) never changes by more than
  ``a * period`` on any joint, engaged, stopping or engaging again (INV-1, decided on the
  node's command);
* MoveIt Servo's output to the controller stays within twice the limit: a Servo skip can join
  two command steps into one; how often and by how much is printed (reported, not fixed);
* every stop ramp after a release ends within the stop time, and the tool's travel after the
  release stays within the formula ``v q / (2 a)`` plus what was already on its way.

One mock stack for the module, simulation values (10 rad/s^2, 1.9 rad/s), its own domain.
"""

import math
import threading
import time

from control_msgs.msg import JointJog
import numpy as np
import pytest
from ranger_xarm_teleop_check.rules import COMMAND_ROUNDING, SKIP_FACTOR, SKIP_MARGIN
from ranger_xarm_teleop_interfaces.msg import ArmOwner
from ranger_xarm_teleop_servo import joint_accel as ja
from stack_helpers import (assert_stack_nodes, child_environment, MOCK_STACK_NODES, Probe,
                           Process, READY, replay_with_clean_input, start_input)
from trajectory_msgs.msg import JointTrajectory

A = 10.0                    # simulation joint acceleration limit (SIMULATION_ONLY_limits.yaml)
CAP = 1.9                   # simulation joint speed cap
T = 0.02
JOINTS = [f'xarm_joint{i}' for i in range(1, 7)]
JOINT_LIMIT_START = [0.0, -0.6, -0.7, 0.0, -1.2, 0.0]
BEYOND_HARD_STOP = [0.0, -0.6, -0.7, 0.0, 0.05, 0.0]
JOINT5_LOWER = -1.69297         # the URDF's lower bound of joint 5 (as test_mock_dls)
KEEPOUT_MARGIN = 0.025          # the simulation-only link keep-out margin
# fixture: (start posture, engaged seconds expected for a clean replay)
FIXTURES = {
    'clutch_release_moving': (READY, 1.0),
    'clutch_fast': (READY, 1.0),
    'clutch_step': (READY, 4.0),
    'clutch_reach': (READY, 3.0),
    'clutch_joint_limit': (JOINT_LIMIT_START, 2.0),
    'clutch_scale_switch': (READY, 4.0),
    'clutch_keepout_reverse': (READY, 2.0),
    'gate_out_of_view': (READY, 0.5),
    'clutch_singularity': (BEYOND_HARD_STOP, 2.0),
    'clutch_engage_moving': (READY, 1.0),       # the grip closes with the hand moving
}
# Points already on their way at a release: MoveIt Servo's window and the controller (the
# tool lags the target by about 0.11 s in mock execution); plus 3 mm for the measurement.
IN_FLIGHT_S = 0.15
TRAVEL_SLACK = 0.003


class Taps:
    """The teleop node's joint command and MoveIt Servo's output, with arrival times."""

    def __init__(self, probe):
        self.lock = threading.Lock()
        self.commands, self.trajectories = [], []
        probe.node.create_subscription(JointJog, '/servo_node/delta_joint_cmds',
                                       self._command, 500)
        probe.node.create_subscription(JointTrajectory, '/teleop_arm_controller/joint_trajectory',
                                       self._trajectory, 500)

    def _command(self, msg):
        order = [list(msg.joint_names).index(j) for j in JOINTS]
        with self.lock:
            self.commands.append((time.monotonic(), msg.header.stamp.sec
                                  + msg.header.stamp.nanosec * 1e-9,
                                  np.array([msg.velocities[i] for i in order])))

    def _trajectory(self, msg):
        order = [list(msg.joint_names).index(j) for j in JOINTS]
        base = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        points = [(base + p.time_from_start.sec + p.time_from_start.nanosec * 1e-9,
                   [p.positions[i] for i in order]) for p in msg.points]
        with self.lock:
            self.trajectories.append((time.monotonic(), points))

    def since(self, t):
        with self.lock:
            return ([c for c in self.commands if c[0] >= t],
                    [m for m in self.trajectories if m[0] >= t])


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    logs = tmp_path_factory.mktemp('mock_accel')
    env = child_environment(offset=21)
    launch = Process(['ros2', 'launch', 'ranger_xarm_teleop_bringup', 'teleop_mock.launch.py',
                      'rviz:=false'],
                     env, str(logs / 'launch.log'))
    probe = Probe(env)
    taps = Taps(probe)                  # subscribed before the stack publishes
    try:
        probe.wait_for(lambda: probe.owner_value() == ArmOwner.OWNER_MOVEIT, 90.0,
                       'the mux to report MoveIt as owner')
        probe.wait_for(lambda: probe.state() is not None
                       and probe.state().measured_tool_pose_valid, 60.0, 'the teleop state')
        probe.wait_for(lambda: probe.controllers().get('teleop_arm_controller') == 'inactive',
                       60.0, 'the teleop controller to be loaded inactive')
        probe.move_to(READY)
        start_input(probe, env, logs)
        yield probe, env, logs, launch, taps
    finally:
        probe.close()
        launch.stop()
        print(launch.output()[-6000:])


def command_steps(commands):
    """Return the largest per-joint change between consecutive commands (zero before each run)."""
    steps, previous, last_t = [], np.zeros(6), None
    for arrival, _, v in sorted(commands, key=lambda c: c[1]):
        if last_t is not None and arrival - last_t > 0.1:
            previous = np.zeros(6)              # Servo had stopped (zero sent, then paused)
        steps.append(float(np.max(np.abs(v - previous))))
        previous, last_t = v, arrival
    return steps


def servo_accelerations(trajectories):
    """Return MoveIt Servo's point accelerations (max over joints) on its position path."""
    # Servo re-sends each state in later windows; its time then differs by float rounding
    # only. Points closer than 1 ms are one point (the first copy is kept).
    raw = sorted((t, p) for _, pts in trajectories for t, p in pts)
    points = {}
    for t, p in raw:
        if points and t - max(points) < 1e-3:
            continue
        points[t] = p
    times = sorted(points)
    acc = []
    run_t, run_p = [], []
    for t in times + [math.inf]:
        if run_t and (t - run_t[-1] > 0.1 or t == math.inf):
            if len(run_t) >= 2:
                tt, pp = np.array(run_t), np.array(run_p)
                v = np.diff(pp, axis=0) / np.diff(tt)[:, None]
                ve = np.vstack([np.zeros(6), v, np.zeros(6)])
                le = np.concatenate([[T], np.diff(tt), [T]])
                a = np.diff(ve, axis=0) / ((le[1:] + le[:-1]) / 2.0)[:, None]
                acc.extend(np.max(np.abs(a), axis=1).tolist())
            run_t, run_p = [], []
        if t != math.inf:
            run_t.append(t)
            run_p.append(points[t])
    return acc


@pytest.mark.parametrize('fixture', sorted(FIXTURES))
def test_the_command_stays_within_the_limit_and_stops_within_the_stop_time(stack, fixture):
    probe, env, logs, _, taps = stack
    posture, engaged_s = FIXTURES[fixture]

    def ready():
        probe.move_to(posture, seconds=3.0)
    start, _ = replay_with_clean_input(probe, env, str(logs / f'replay_{fixture}.log'), fixture,
                                       expected_engaged_s=engaged_s, before=ready)
    time.sleep(1.0)
    commands, trajectories = taps.since(start)
    timed = probe.timed_states_since(start)
    assert any(s.clutch_engaged for _, s in timed), f'{fixture}: never engaged'
    assert commands, f'{fixture}: no joint command seen'
    # INV-1 on the node's command.
    steps = command_steps(commands)
    assert max(steps) <= A * T * (1 + 1e-6), (fixture, max(steps) / T)
    assert max(float(np.max(np.abs(c[2]))) for c in commands) <= CAP + 1e-9
    # MoveIt Servo's output: within hw_check's skip tolerance; reported.
    servo = servo_accelerations(trajectories)
    over = [a for a in servo if a > A * (1 + 1e-3)]
    print(f'{fixture}: command peak {max(steps) / T:.2f} rad/s^2; Servo output peak '
          f'{max(servo):.2f}, {len(over)} of {len(servo)} points above the limit'
          + (f' (largest {max(over):.2f})' if over else ''))
    # hw_check's stage-4 rule (G-18): a point FAILs only above 2 x 1.05 x the limit.
    bound = SKIP_FACTOR * SKIP_MARGIN * A * (1 + COMMAND_ROUNDING)
    assert max(servo) <= bound, (fixture, max(servo), bound)
    # INV-5: the bounds keep their meaning under the limit.
    engaged = [s for _, s in timed if s.clutch_engaged]
    if fixture == 'clutch_joint_limit':
        q5 = [q[4] for q in probe.joints_since(start)]
        assert min(q5) - JOINT5_LOWER >= 0.15 - 0.002, min(q5) - JOINT5_LOWER
    if fixture == 'clutch_keepout_reverse':
        assert min(s.keepout_clearance_commanded for s in engaged) >= KEEPOUT_MARGIN - 1e-3
    # Every stop: ramp length and tool travel against the formula.
    ordered = sorted(commands, key=lambda c: c[1])
    for i in range(1, len(timed)):
        if not (timed[i - 1][1].clutch_engaged and not timed[i][1].clutch_engaged):
            continue
        release = timed[i][1]
        stamp = release.header.stamp.sec + release.header.stamp.nanosec * 1e-9
        k = next((j for j, c in enumerate(ordered) if abs(c[1] - stamp) < 1e-6), None)
        if k is None or k == 0:
            continue                        # a release with nothing commanded before it
        before = ordered[k - 1][2]
        q_speed = float(np.max(np.abs(before)))
        ramp = 0
        for c in ordered[k:]:
            ramp += 1
            if not np.any(c[2]):
                break
        assert (ramp - 1) * T <= ja.stop_time(before, A) + T + 1e-9, (fixture, ramp)
        assert (ramp - 1) * T <= ja.STOP_TIME_LIMIT_S + 1e-9, (fixture, ramp)
        v = float(timed[i - 1][1].commanded_linear_speed)
        p0 = timed[i - 1][1].measured_tool_pose.position
        rest = [s for _, s in timed[i:] if s.measured_tool_pose_valid][:int(1.0 / T)]
        path, last = 0.0, (p0.x, p0.y, p0.z)
        for s in rest:
            p = s.measured_tool_pose.position
            path += math.dist(last, (p.x, p.y, p.z))
            last = (p.x, p.y, p.z)
        bound = ja.stop_travel_bound(v, q_speed, A) + v * IN_FLIGHT_S + TRAVEL_SLACK
        print(f'{fixture}: release at {v:.3f} m/s, fastest joint {q_speed:.2f} rad/s: ramp '
              f'{(ramp - 1) * T:.2f} s (formula {ja.stop_time(before, A):.2f} s), tool travel '
              f'{path * 1000:.1f} mm (formula {ja.stop_travel_bound(v, q_speed, A) * 1000:.1f} '
              f'mm + {v * IN_FLIGHT_S * 1000:.1f} mm in flight)')
        assert path <= bound, (fixture, path, bound)
    probe.disarm()


def test_log_has_no_errors(stack):
    output = stack[3].output()
    assert 'MoveIt Servo verified' in output
    assert 'Traceback' not in output


def test_no_other_stack_or_test_shares_this_domain(stack):
    """Isolation: the domain holds this stack's nodes and the probe, nothing else."""
    assert_stack_nodes(stack[0], stack[3], MOCK_STACK_NODES)
