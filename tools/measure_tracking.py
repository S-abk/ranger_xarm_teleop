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
Record a teleop run and report tracking error, delay and stopping distance.

    tools/measure_tracking.py record run.jsonl [--duration 12] [--servo-node /servo_node]
    tools/measure_tracking.py report run.jsonl [--json]

``record`` needs a sourced ROS workspace; it stamps every received message with this
process's monotonic clock. It records the controller and headset poses, the target, the
teleop state (with the solver's fields), the joint states (positions, velocities) and the
node's joint command. ``report`` needs no ROS; recordings from before the joint-space
solver still work (their new figures read n/a). Write run files outside the
repository (they are measurements, not fixtures).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'src', 'ranger_xarm_teleop_servo'))
import _pkgpath  # noqa: F401,E402  (sys.path for the bridge package)


def record(path: str, duration: float, servo_node: str) -> None:
    """Record the teleop topics to ``path`` for ``duration`` seconds."""
    from control_msgs.msg import JointJog
    from geometry_msgs.msg import PoseStamped
    from ranger_xarm_teleop_interfaces.msg import TeleopState
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    from ranger_xarm_teleop_servo import session

    rclpy.init()
    node = rclpy.create_node('teleop_tracking_recorder')
    out = open(path, 'x', encoding='utf-8')

    def write(row) -> None:
        if row is not None:
            out.write(json.dumps({'t': time.monotonic(), **row}) + '\n')

    node.create_subscription(
        PoseStamped, '/teleop/input/right/pose',
        lambda msg: write(session.pose_row('input', msg)), qos_profile_sensor_data)
    node.create_subscription(
        PoseStamped, '/teleop/input/head/pose',
        lambda msg: write(session.pose_row('head', msg)), qos_profile_sensor_data)
    node.create_subscription(
        PoseStamped, '/teleop/target_pose', lambda msg: write(session.pose_row('target', msg)), 50)
    node.create_subscription(
        TeleopState, '/teleop/state', lambda msg: write(session.state_row(msg)), 50)
    node.create_subscription(
        JointState, '/joint_states', lambda msg: write(session.joints_row(msg)), 200)
    node.create_subscription(
        JointJog, servo_node.rstrip('/') + '/delta_joint_cmds',
        lambda msg: write(session.command_row(msg)), 50)
    end = time.monotonic() + duration
    try:
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        node.destroy_node()
        rclpy.try_shutdown()


def report(path: str, as_json: bool) -> None:
    """Print the figures of a recorded run."""
    from ranger_xarm_teleop_servo.tracking import analyze, format_report
    with open(path, encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    result = analyze(records)
    print(json.dumps(result, indent=2) if as_json else format_report(result))


def main() -> None:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    rec = sub.add_parser('record')
    rec.add_argument('out')
    rec.add_argument('--duration', type=float, default=12.0)
    rec.add_argument('--servo-node', default='/servo_node')
    rep = sub.add_parser('report')
    rep.add_argument('run')
    rep.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.command == 'record':
        record(args.out, args.duration, args.servo_node)
    else:
        report(args.run, args.json)


if __name__ == '__main__':
    main()
