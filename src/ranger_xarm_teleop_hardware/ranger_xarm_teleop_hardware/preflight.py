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
The hardware teleop launch's preflight: read the running bring-up once, refuse or go on.

It reads the controller states (one ``list_controllers`` read, asked again after a timeout),
the robot description, the publishers of /clock and one arm joint state, and prints either
``ATTACH PREFLIGHT OK`` or one line per problem (``attach.judge``) and exits non-zero. The
launch starts the teleop stack only after it exits zero. It changes nothing: no switch, no
load, no parameter.
"""

import argparse
import sys
import time

from controller_manager_msgs.srv import ListControllers
from ranger_xarm_teleop_hardware.attach import ARM_JOINTS, check_stage, judge
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def read_controllers(node, client, wait_s: float, reads: int = 3):
    """Return {name: state}, or None when the controller manager did not answer."""
    print(f'preflight: waiting up to {wait_s:g} s for /controller_manager/list_controllers',
          flush=True)
    if not client.wait_for_service(timeout_sec=wait_s):
        return None
    for attempt in range(reads):
        future = client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
        if future.done() and future.result() is not None:
            return {c.name: c.state for c in future.result().controller}
        print(f'preflight: list_controllers gave no answer (read {attempt + 1} of {reads}); '
              'asking again', flush=True)
    return None


def main(argv=None) -> None:
    """Run the preflight; exit 0 to attach, 1 to refuse."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--stage', required=True)
    parser.add_argument('--wait', type=float, default=15.0,
                        help='seconds to wait for the controller manager and the topics')
    args, ros_args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    try:
        check_stage(args.stage)
    except ValueError as err:
        print(f'ATTACH REFUSED: {err}', flush=True)
        sys.exit(1)
    rclpy.init(args=ros_args)
    node = rclpy.create_node('teleop_attach_preflight')
    client = node.create_client(ListControllers, '/controller_manager/list_controllers')
    seen = {'description': None, 'joints': None}

    def on_description(msg):
        seen['description'] = msg.data

    def on_joints(msg):
        if seen['joints'] is None and all(j in msg.name for j in ARM_JOINTS):
            seen['joints'] = list(msg.name)

    node.create_subscription(String, '/robot_description', on_description, LATCHED)
    node.create_subscription(JointState, '/joint_states', on_joints, 50)
    try:
        controllers = read_controllers(node, client, args.wait)
        deadline = time.monotonic() + (args.wait if controllers is not None else 3.0)
        while time.monotonic() < deadline and (seen['description'] is None
                                               or seen['joints'] is None):
            rclpy.spin_once(node, timeout_sec=0.1)
        clock = len(node.get_publishers_info_by_topic('/clock'))
        problems = judge(args.stage, controllers, seen['description'], clock, seen['joints'])
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    if problems:
        for line in problems:
            print(f'ATTACH REFUSED: {line}', flush=True)
        sys.exit(1)
    states = ', '.join(f'{k} {v}' for k, v in sorted(controllers.items()))
    print(f'ATTACH PREFLIGHT OK ({args.stage} stage): {states}', flush=True)


if __name__ == '__main__':
    main()
