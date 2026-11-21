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
A stand-in for the command mux's report and the teleop node, beside ``fake_controller.py``.

    fake_teleop_side.py MUX TELEOP

MUX is ``mux`` (the mux's owner report, several times a second) or ``none``. TELEOP is

    none      no ``/teleop/disarm`` service and no ``/teleop/state``
    disarmed  the service, and a state that says "not armed", 20 times a second
    arms      as ``disarmed``, and "armed" once the arm's first joint has moved by 0.02 rad
              from where it was first seen (the motion has begun)

"Armed" is reported as the teleop node does: with teleop as the owner, in the state and in
the mux's report. Nothing here moves the arm; ``fake_controller.py`` does.
"""

import sys

from ranger_xarm_teleop_interfaces.msg import ArmOwner, TeleopState
import rclpy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


def main() -> int:
    """Serve until killed."""
    mux, teleop = sys.argv[1], sys.argv[2]
    rclpy.init()
    node = rclpy.create_node('fake_teleop_side')
    seen = {'first': None, 'armed': False}
    owner = node.create_publisher(ArmOwner, '/teleop/mux/owner', 10) if mux == 'mux' else None
    state = None
    if teleop != 'none':
        state = node.create_publisher(TeleopState, '/teleop/state', 10)

        def on_disarm(request, response):
            print('disarm requested', flush=True)
            response.success, response.message = True, 'was not armed'
            return response
        node.create_service(Trigger, '/teleop/disarm', on_disarm)

    def on_joints(msg) -> None:
        if not msg.position:
            return
        if seen['first'] is None:
            seen['first'] = msg.position[0]
        if teleop == 'arms' and not seen['armed'] and abs(msg.position[0] - seen['first']) > 0.02:
            seen['armed'] = True
            print('armed', flush=True)
    node.create_subscription(JointState, '/joint_states', on_joints, 10)
    ticks = [0]

    def tick() -> None:
        ticks[0] += 1
        owns = ArmOwner.OWNER_TELEOP if seen['armed'] else ArmOwner.OWNER_MOVEIT
        if owner is not None and ticks[0] % 5 == 0:
            owner.publish(ArmOwner(owner=owns))
        if state is not None and ticks[0] % 2 == 0:
            message = TeleopState(armed=seen['armed'], owner=owns)
            message.header.stamp = node.get_clock().now().to_msg()
            state.publish(message)

    node.create_timer(0.02, tick)
    print(f'ready: mux {mux}, teleop {teleop}', flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
