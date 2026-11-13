#!/usr/bin/env python3
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
Record a live teleop session and list every MoveIt Servo halt in it.

    tools/record_session.py record NAME [--duration SECONDS]
    tools/record_session.py halts recordings/NAME.jsonl [--json] [--urdf FILE]

``record`` needs a sourced workspace and a running teleop stack. It writes
``recordings/NAME.jsonl`` (ignored by git) until Ctrl-C: the teleop state, the tool
target, the controller pose, the joint states, MoveIt Servo's status, the arm owner and
the robot description. It only listens: it calls no service and publishes nothing.

Start it BEFORE arming and stop it AFTER releasing the grip. A ROS tool that starts or
stops while the clutch is held can stall the input and release the clutch, so ``record``
refuses to start while the clutch is engaged (exit status 2), and after Ctrl-C it keeps
recording until the clutch is released.

``halts`` needs no ROS. Per halt it prints MoveIt Servo's status, the hold reason, the
joint angles, the singularity measure (the Jacobian condition number MoveIt Servo
thresholds, computed from the recorded joint angles) and its trend over the second
before, the nearest joint limit, the singular configuration the posture resembles and
what the operator was commanding; then counts by reason and the distribution of the
measure over the engaged time.
"""

import argparse
import json
import os
import signal
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'ranger_xarm_teleop_servo'))
import _pkgpath  # noqa: F401,E402  (sys.path for the bridge package)

from ranger_xarm_teleop_servo import session  # noqa: E402

START_WAIT_S = 10.0


def record(name: str, directory: str, duration: float, servo_node: str) -> int:
    """Record the session topics until asked to stop and the clutch is released."""
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.msg import ServoStatus
    from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from rclpy.qos import ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    path = os.path.join(directory, name if name.endswith('.jsonl') else name + '.jsonl')
    if os.path.exists(path):
        print(f'REFUSED: {path} exists')
        return 2
    watch = session.ClutchWatch()
    stop = {'asked': False, 'told': False}
    rows = []                   # rows received before the file is opened

    # Our own handler: the recording must outlive Ctrl-C until the clutch is released.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def ask_stop(*_):
        stop['asked'] = True

    signal.signal(signal.SIGINT, ask_stop)
    signal.signal(signal.SIGTERM, ask_stop)
    node = rclpy.create_node('teleop_session_recorder')
    out = None

    def write(row) -> None:
        if row is None:
            return
        row = {'t': time.monotonic(), **row}
        if out is None:
            rows.append(row)
        else:
            out.write(json.dumps(row) + '\n')

    def on_state(msg) -> None:
        watch.update(time.monotonic(), msg.clutch_engaged)
        write(session.state_row(msg))

    code = 0
    try:
        # Only the state first: nothing else is created until the clutch is known released.
        node.create_subscription(TeleopState, '/teleop/state', on_state, 50)
        deadline = time.monotonic() + START_WAIT_S
        while watch.seen_at is None and time.monotonic() < deadline and not stop['asked']:
            rclpy.spin_once(node, timeout_sec=0.05)
        allowed, why = watch.may_start(time.monotonic())
        if not allowed:
            print(f'REFUSED, nothing recorded: {why}')
            return 2
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(
            String, '/robot_description',
            lambda msg: write(session.description_row(msg.data)), latched)
        node.create_subscription(
            PoseStamped, '/teleop/target_pose',
            lambda msg: write(session.pose_row('target', msg)), 50)
        node.create_subscription(
            PoseStamped, '/teleop/input/right/pose',
            lambda msg: write(session.pose_row('input', msg)), qos_profile_sensor_data)
        node.create_subscription(
            JointState, '/joint_states', lambda msg: write(session.joints_row(msg)), 200)
        node.create_subscription(
            ServoStatus, servo_node.rstrip('/') + '/status',
            lambda msg: write(session.servo_row(msg)), 50)
        node.create_subscription(
            ArmOwner, '/teleop/mux/owner', lambda msg: write(session.owner_row(msg)), 10)
        os.makedirs(directory, exist_ok=True)
        out = open(path, 'x', encoding='utf-8')
        for row in rows:
            out.write(json.dumps(row) + '\n')
        print(f'recording to {path}; Ctrl-C to stop (after releasing the grip)', flush=True)
        end = time.monotonic() + duration if duration > 0 else None
        flushed = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if now - flushed > 1.0:
                out.flush()
                flushed = now
            if end is not None and now >= end:
                stop['asked'] = True
            if stop['asked']:
                if watch.may_stop(now):
                    break
                if not stop['told']:
                    stop['told'] = True
                    print('the clutch is engaged: still recording, will stop when the '
                          'grip is released', flush=True)
    finally:
        if out is not None:
            out.close()
            print(f'wrote {path}')
        node.destroy_node()
        rclpy.try_shutdown()
    return code


def halts(path: str, as_json: bool, urdf_path: str, lower: float, hard_stop: float) -> int:
    """Print the halt table of a recording."""
    from ranger_xarm_teleop_servo.halts import analyze, format_report, SessionError
    with open(path, encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    urdf = None
    if urdf_path:
        with open(urdf_path, encoding='utf-8') as handle:
            urdf = handle.read()
    try:
        result = analyze(records, urdf, lower, hard_stop)
    except SessionError as err:
        print(f'cannot analyse {path}: {err}')
        return 2
    print(json.dumps(result, indent=2) if as_json else format_report(result))
    return 0


def main() -> int:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    rec = sub.add_parser('record')
    rec.add_argument('name')
    rec.add_argument('--dir', default=os.path.join(REPO, 'recordings'),
                     help='where to write (default: recordings/, ignored by git)')
    rec.add_argument('--duration', type=float, default=0.0,
                     help='stop after this many seconds (default: until Ctrl-C)')
    rec.add_argument('--servo-node', default='/servo_node')
    rep = sub.add_parser('halts')
    rep.add_argument('recording')
    rep.add_argument('--json', action='store_true')
    rep.add_argument('--urdf', default='', help='use this URDF instead of the recorded one')
    rep.add_argument('--lower', type=float, default=17.0)
    rep.add_argument('--hard-stop', type=float, default=30.0)
    args = parser.parse_args()
    if args.command == 'record':
        return record(args.name, args.dir, args.duration, args.servo_node)
    return halts(args.recording, args.json, args.urdf, args.lower, args.hard_stop)


if __name__ == '__main__':
    sys.exit(main())
