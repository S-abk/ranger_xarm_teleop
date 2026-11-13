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

    tools/measure_tracking.py record run.jsonl [--duration 12]
    tools/measure_tracking.py report run.jsonl [--json]

``record`` needs a sourced ROS workspace; it stamps every received message with this
process's monotonic clock. ``report`` needs no ROS. Write run files outside the
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

ARM_JOINTS = [f'xarm_joint{i}' for i in range(1, 7)]


def record(path: str, duration: float) -> None:
    """Record the teleop topics to ``path`` for ``duration`` seconds."""
    from geometry_msgs.msg import PoseStamped
    from ranger_xarm_teleop_interfaces.msg import TeleopState
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node('teleop_tracking_recorder')
    out = open(path, 'x', encoding='utf-8')

    def write(kind: str, **fields) -> None:
        out.write(json.dumps({'t': time.monotonic(), 'k': kind, **fields}) + '\n')

    def stamp(msg) -> int:
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def pose(p):
        return ([p.position.x, p.position.y, p.position.z],
                [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])

    def on_input(msg) -> None:
        write('input', p=pose(msg.pose)[0])

    def on_target(msg) -> None:
        p, q = pose(msg.pose)
        write('target', stamp=stamp(msg), p=p, q=q)

    def on_state(msg) -> None:
        write('state', stamp=stamp(msg), engaged=msg.clutch_engaged, hold=msg.hold_reason,
              limit=msg.active_limit, owner=msg.owner, servo=msg.servo_status,
              age=msg.input_age,
              tool=list(pose(msg.measured_tool_pose)) if msg.measured_tool_pose_valid else None)

    def on_joints(msg) -> None:
        index = {name: i for i, name in enumerate(msg.name)}
        if all(j in index for j in ARM_JOINTS):
            write('joints', q=[msg.position[index[j]] for j in ARM_JOINTS])

    node.create_subscription(PoseStamped, '/teleop/input/right/pose', on_input,
                             qos_profile_sensor_data)
    node.create_subscription(PoseStamped, '/teleop/target_pose', on_target, 50)
    node.create_subscription(TeleopState, '/teleop/state', on_state, 50)
    node.create_subscription(JointState, '/joint_states', on_joints, 50)
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
    rep = sub.add_parser('report')
    rep.add_argument('run')
    rep.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.command == 'record':
        record(args.out, args.duration)
    else:
        report(args.run, args.json)


if __name__ == '__main__':
    main()
