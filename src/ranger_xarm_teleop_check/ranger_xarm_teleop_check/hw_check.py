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
hw_check: read-only checks for one stage of the staged hardware procedure.

Run it as ``ros2 run ranger_xarm_teleop_check hw_check <stage> [options]``.

The tool only subscribes. It creates no publisher, service, client or action, and makes no
parameter call; its node has no parameter services. Graph introspection is used to learn who
publishes a topic. This file collects samples; the judging is in ``rules`` (pure).
"""

import argparse
import datetime
import os
import re
import sys
import time
from typing import List, Optional

from ranger_xarm_teleop_check import report as rep
from ranger_xarm_teleop_check import rules, stats, sysinfo
from ranger_xarm_teleop_check.rules import Options
from ranger_xarm_teleop_servo.joint_accel import ACCELERATION_NAME, IMMEDIATE_STOP_LOG

NODE_NAME = 'hw_check'
TELEOP_NODE_NAME = 'teleop_servo'     # the node whose /rosout summaries are read
VERSION_PACKAGES = ('controller_manager', 'joint_trajectory_controller', 'moveit_servo')
# Discovery: wait at least this long, and up to the longer one for the joint states publisher
# and, in the idle stage, a publisher of the arm driver's state type (s).
DISCOVERY_MIN_S = 2.0
DISCOVERY_MAX_S = 4.0
# Queue depths: joint states 50 (no message lost at 150 Hz), teleop state 50, /rosout 1000.
JOINT_STATES_DEPTH = 50
TELEOP_STATE_DEPTH = 50
ROSOUT_DEPTH = 1000
# MoveIt Servo's output topic, about 50 Hz: no message lost over a stall of one second.
COMMAND_DEPTH = 50
# The vendor driver's state topic (about 5 Hz): no message lost over a stall of ten seconds.
ROBOT_STATES_DEPTH = 50
CAMERA_DEFAULT_QOS = {'reliability': 'best_effort', 'depth': 10}


def _servo_latency():
    """Return MoveIt Servo's max_expected_latency (s) from the installed servo.yaml, or None."""
    try:
        from ament_index_python.packages import get_package_share_directory
        import os
        path = os.path.join(get_package_share_directory('ranger_xarm_teleop_bringup'),
                            'config', 'servo.yaml')
    except Exception:  # noqa: B902 - not installed: the stop rule says it is not judged
        return None
    return sysinfo.servo_latency(path)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    ap = argparse.ArgumentParser(
        prog='hw_check', description='Read-only check of one stage of the hardware procedure.')
    ap.add_argument('stage', choices=rules.STAGES)
    ap.add_argument('--seconds', type=float, default=None,
                    help='measuring time (default: idle 30, motion 20, teleop 20, cameras 30)')
    ap.add_argument('--limits',
                    help='limits YAML (tool_pose_timeout_s, max_lead_translation_m, and for '
                    'the teleop stage joint_acceleration_limit_rad_s2, the limit the Servo '
                    'output is judged against)')
    ap.add_argument('--profile', help='deployment profile (nominal rates, camera streams)')
    ap.add_argument('--data-dir', default=rep.DEFAULT_DATA_DIR,
                    help='data directory; reports go to <dir>/hardware_checks')
    ap.add_argument('--joint-limits',
                    help='MoveIt joint_limits.yaml (max_acceleration): the limit of the motion '
                    "stage's peak joint acceleration; shown for reference in the teleop stage")
    ap.add_argument('--jog-topic', default=rules.DEFAULT_JOG_TOPIC,
                    help="teleop stage: the teleop node's joint-jog commands to MoveIt Servo, "
                    "read to tell Servo's skips (default: %(default)s)")
    ap.add_argument('--command-topic', default=rules.DEFAULT_COMMAND_TOPIC,
                    help='teleop stage: the controller trajectory topic MoveIt Servo publishes '
                    'on, read for the acceleration and stop-time rules (default: %(default)s)')
    ap.add_argument('--expect-fingerprint', help='sha256 the robot description must have')
    ap.add_argument('--workspace', default=os.environ.get('RANGER_XARM_WS'),
                    help='workspace holding src/xarm_ros2 (default: $RANGER_XARM_WS)')
    ap.add_argument('--controllers', default=','.join(rules.DEFAULT_CONTROLLERS),
                    help='controllers whose /<name>/controller_state to look for')
    ap.add_argument('--expect-active', default='', help='comma list: must be active')
    ap.add_argument('--expect-inactive', default='', help='comma list: must not be active')
    return ap


def _names(text: str) -> tuple:
    return tuple(n.strip() for n in (text or '').split(',') if n.strip())


def resolve_options(args) -> tuple:
    """Return (Options, problem): the options read from the files, or None and why not."""
    opts = Options(controllers=_names(args.controllers),
                   expect_active=_names(args.expect_active),
                   expect_inactive=_names(args.expect_inactive),
                   expect_fingerprint=args.expect_fingerprint,
                   command_topic=args.command_topic, jog_topic=args.jog_topic)
    if args.limits:
        data, problem = sysinfo.load_yaml_file(args.limits)
        if problem:
            return None, problem
        opts.limits_given = True
        opts.tool_pose_timeout_s = sysinfo.limit_parameter(data, 'tool_pose_timeout_s')
        opts.max_lead_translation_m = sysinfo.limit_parameter(data, 'max_lead_translation_m')
        opts.joint_acceleration_limit = sysinfo.limit_parameter(data, ACCELERATION_NAME)
    opts.servo_latency_s = _servo_latency()
    if args.joint_limits:
        data, problem = sysinfo.load_yaml_file(args.joint_limits)
        if problem:
            return None, problem
        opts.joint_limits_given = True
        opts.joint_acceleration_limits = sysinfo.joint_acceleration_limits(data)
    profile = None
    if args.profile:
        from ranger_xarm_teleop_record.profile import load_profile
        try:
            profile = load_profile(args.profile)
        except (OSError, ValueError) as exc:
            return None, f'cannot use profile {args.profile}: {exc}'
        rate = (profile['streams'].get('joint_states') or {}).get('rate_hz')
        if rate:
            opts.nominal_hz = float(rate)
        opts.streams = [dict(spec, name=name) for name, spec in profile['streams'].items()
                        if spec.get('kind') == 'image']
    if args.stage == 'cameras':
        if profile is None:
            return None, 'pass --profile <deployment profile>'
        if not opts.streams:
            return None, f'profile {args.profile} has no image streams to check'
    return opts, None


# ---- collection (the only part that touches ROS) ----------------------------------------

def _qos(reliability: str, durability: str, depth: int):
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=depth,
        reliability=(ReliabilityPolicy.RELIABLE if reliability == 'reliable'
                     else ReliabilityPolicy.BEST_EFFORT),
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if durability == 'transient_local'
                    else DurabilityPolicy.VOLATILE))


def _publisher_profiles(node, topic: str) -> list:
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
    out = []
    for info in node.get_publishers_info_by_topic(topic):
        q = info.qos_profile
        out.append(('reliable' if q.reliability == ReliabilityPolicy.RELIABLE
                    else 'best_effort',
                    'transient_local' if q.durability == DurabilityPolicy.TRANSIENT_LOCAL
                    else 'volatile'))
    return out


def _publisher_names(node, topic: str) -> List[str]:
    return sorted({(i.node_namespace.rstrip('/') + '/' + i.node_name)
                   for i in node.get_publishers_info_by_topic(topic)})


def collect(stage: str, opts: Options, seconds: float) -> dict:
    """Subscribe, spin for ``seconds`` and return the samples (see ``rules.evaluate``)."""
    import rclpy
    from rclpy.clock import Clock, ClockType
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    from rclpy.parameter import Parameter

    rclpy.init(args=sys.argv)
    # No parameter services (INV-3). rclpy also starts a get_type_description service unless
    # this local override says otherwise: it is a service, so it is switched off.
    # No /rosout publisher either: the tool prints; its only publisher left is rclpy's own
    # /parameter_events, which nothing here publishes on.
    node = Node(NODE_NAME, start_parameter_services=False, enable_rosout=False,
                parameter_overrides=[Parameter('start_type_description_service',
                                               Parameter.Type.BOOL, False)])
    system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    data = {'arm': [], 'gripper': [], 'js_publishers': [], 'clock_publishers': [],
            'activity': None, 'controller_state': {}, 'description': None, 'states': [],
            'targets': [], 'command': [], 'command_velocities': [], 'jog': [],
            'jog_arrival': [], 'command_capture': None, 'clutch': [],
            'tracking': [],
            'warnings': [], 'robot_states': None,
            'rosout': {'overruns': 0, 'read_failures': 0, 'immediate_stops': 0,
                       'teleop_summaries': [],
                       'overrun_examples': [],
                       'read_examples': []},
            'cameras': {}, 'qos': {}}
    try:
        # Discovery first: the QoS of a subscription must match what is published.
        probe = '/joint_states' if stage != 'cameras' else opts.streams[0]['topic']
        t0 = time.monotonic()
        _discover(node, lambda: executor.spin_once(timeout_sec=0.05), probe, stage == 'idle')
        clock_pubs = set(_publisher_names(node, '/clock'))

        def subscribe(msg_type, topic, callback, depth, reliability, durability):
            rel, dur = sysinfo.choose_qos(_publisher_profiles(node, topic), reliability,
                                          durability)
            data['qos'][topic] = f'{rel}, {dur}, depth {depth}'
            return node.create_subscription(msg_type, topic, callback, _qos(rel, dur, depth))

        if stage == 'cameras':
            _subscribe_cameras(node, opts, data)
        else:
            _subscribe_stack(stage, opts, data, subscribe, system_clock)
        if stage == 'idle':
            # Before the capture window: the graph lookup, the first import of xarm_msgs and
            # its typesupport (about 0.1 s idle, several times that under load) and the new
            # subscription would otherwise hold up the joint-state callbacks inside the window,
            # and so the largest-gap figure that tool_pose_timeout_s is chosen from.
            _RobotStates(node, data, subscribe).poll(time.monotonic() - t0)
        began = time.monotonic()
        deadline = began + seconds
        while True:
            now = time.monotonic()
            if now >= deadline and (stage == 'cameras' or data['activity'] is not None
                                    or now - began >= rules.ACTIVITY_WAIT_S):
                break
            executor.spin_once(timeout_sec=0.02)
        data['command_capture'] = {'began': began, 'ended': time.monotonic()}
        data['clock_publishers'] = sorted(clock_pubs | set(_publisher_names(node, '/clock')))
        data['js_publishers'] = _publisher_names(node, '/joint_states')
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return data


def _stamp(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _stamp_ns(header) -> int:
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def _subscribe_stack(stage, opts, data, subscribe, system_clock):
    from control_msgs.msg import JointJog, JointTrajectoryControllerState
    from controller_manager_msgs.msg import ControllerManagerActivity
    from geometry_msgs.msg import PoseStamped
    from rcl_interfaces.msg import Log
    from ranger_xarm_teleop_interfaces.msg import TeleopState
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory

    def on_joint_states(msg):
        mono = time.monotonic()
        now = system_clock.now().nanoseconds * 1e-9
        kind, pos, vel, effort_ok = stats.classify_joint_state(
            msg.name, msg.position, msg.velocity, msg.effort)
        if kind == 'arm':
            data['arm'].append((mono, now, _stamp(msg.header), pos, vel, effort_ok))
        elif kind == 'gripper':
            data['gripper'].append((mono, _stamp(msg.header)))

    subscribe(JointState, '/joint_states', on_joint_states, JOINT_STATES_DEPTH,
              'reliable', 'volatile')

    def on_activity(msg):
        data['activity'] = {c.name: c.state.label for c in msg.controllers}

    # controller_manager publishes ~/activity latched; its QoS was not found in the installed
    # headers (no source), so reliable + transient_local depth 1 is the default, narrowed to
    # what the publisher offers (choose_qos).
    subscribe(ControllerManagerActivity, '/controller_manager/activity', on_activity, 1,
              'reliable', 'transient_local')

    for name in opts.controllers:
        data['controller_state'][name] = 0

        def on_state(_msg, name=name):
            data['controller_state'][name] += 1

        subscribe(JointTrajectoryControllerState, f'/{name}/controller_state', on_state, 10,
                  'reliable', 'volatile')

    def on_description(msg):
        data['description'] = sysinfo.description_info(msg.data)

    subscribe(String, '/robot_description', on_description, 1, 'reliable', 'transient_local')

    def on_teleop_state(msg):
        if not data['states'] and msg.armed:
            print('WARNING: ' + rules.ARMED_WARNING, flush=True)
            data['warnings'].append(rules.ARMED_WARNING)
        p = msg.measured_tool_pose.position
        data['states'].append((time.monotonic(), _stamp(msg.header), bool(msg.armed),
                               bool(msg.measured_tool_pose_valid), (p.x, p.y, p.z)))
        data['clutch'].append((_stamp_ns(msg.header), bool(msg.clutch_engaged)))
        data['tracking'].append((bool(msg.clutch_engaged), int(msg.active_limit),
                                 bool(msg.keepout_bound), float(msg.keepout_clearance_commanded),
                                 float(msg.keepout_margin), str(msg.keepout_link),
                                 str(msg.keepout_obstacle), float(msg.lead_translation),
                                 float(msg.lead_rotation)))

    subscribe(TeleopState, '/teleop/state', on_teleop_state, TELEOP_STATE_DEPTH,
              'reliable', 'volatile')

    if stage == 'teleop':
        def on_target(msg):
            p = msg.pose.position
            data['targets'].append((_stamp(msg.header), (p.x, p.y, p.z)))

        subscribe(PoseStamped, '/teleop/target_pose', on_target, 50, 'reliable', 'volatile')

        def on_command(msg):
            # Positions and times only: the acceleration is the difference of the segment
            # velocities between the points (stats.command_points).
            points = [(pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9,
                       tuple(pt.positions)) for pt in msg.points]
            data['command'].append((time.monotonic(), _stamp_ns(msg.header),
                                    tuple(msg.joint_names), points))
            # Beside it, not in it: the velocities of each point (empty when it has none).
            data['command_velocities'].append(tuple(tuple(pt.velocities) for pt in msg.points))

        subscribe(JointTrajectory, opts.command_topic, on_command, COMMAND_DEPTH, 'reliable',
                  'volatile')

        def on_jog(msg):
            data['jog'].append((tuple(msg.joint_names), tuple(msg.velocities)))
            data['jog_arrival'].append(time.monotonic())    # beside it: when it arrived

        subscribe(JointJog, opts.jog_topic, on_jog, COMMAND_DEPTH, 'reliable', 'volatile')

    overrun = re.compile('overrun', re.IGNORECASE)

    def on_rosout(msg):
        r = data['rosout']
        if 'controller_manager' in msg.name and overrun.search(msg.msg):
            r['overruns'] += 1
            if len(r['overrun_examples']) < 5:
                r['overrun_examples'].append(f'{msg.name}: {msg.msg}')
        if IMMEDIATE_STOP_LOG in msg.msg:     # the teleop node's stop without a ramp
            r['immediate_stops'] += 1
        if TELEOP_NODE_NAME in msg.name.split('.') and (
                'link keep-out by self pair' in msg.msg
                or 'teleop cycle computation' in msg.msg):
            r['teleop_summaries'].append(msg.msg)     # parsed in rules.keepout_step_measures
        if 'Read() returns' in msg.msg:
            r['read_failures'] += 1
            if len(r['read_examples']) < 5:
                r['read_examples'].append(f'{msg.name}: {msg.msg}')

    # /rosout is published transient_local with a 10 s lifespan; a volatile subscriber does
    # not replay what happened before the run.
    subscribe(Log, '/rosout', on_rosout, ROSOUT_DEPTH, 'reliable', 'volatile')


def _arm_state_topics(node) -> dict:
    """Return {topic: [publisher names]} for every topic of the arm state type with a publisher."""
    topics = {}
    for topic, types in node.get_topic_names_and_types():
        if rules.ARM_STATE_TYPE in types:
            # Every publisher counts, two of one name too (not _publisher_names' set).
            publishers = sorted(i.node_namespace.rstrip('/') + '/' + i.node_name
                                for i in node.get_publishers_info_by_topic(topic))
            if publishers:          # a topic with subscribers only is not a source
                topics[topic] = publishers
    return topics


def _discover(node, spin, probe: str, arm_state: bool, clock=time.monotonic) -> float:
    """
    Wait for the publishers the subscriptions are matched to; return the seconds waited.

    At least DISCOVERY_MIN_S, then until ``probe`` has a publisher and, with ``arm_state``,
    a publisher of the arm driver's state type has appeared too (it is looked up only once,
    after this); at most DISCOVERY_MAX_S.
    """
    t0 = clock()
    while True:
        spin()
        waited = clock() - t0
        have = bool(node.get_publishers_info_by_topic(probe)) and (
            not arm_state or bool(_arm_state_topics(node)))
        if waited >= DISCOVERY_MAX_S or (waited >= DISCOVERY_MIN_S and have):
            return waited


class _RobotStates:
    """
    The arm driver's state topic (xarm_msgs/msg/RobotMsg), read-only, idle stage.

    Found by its message type, not by a name: the driver publishes it in its hardware namespace
    (on the real driver /xarm_xarm/robot_states). Looked up once, after discovery (which in the
    idle stage also waits for a publisher of that type, up to DISCOVERY_MAX_S) and before the
    capture window opens, and subscribed if exactly one publisher of that type is on the graph
    then (the driver starts before the check); with several, they are named and none is read.
    With none, the record keeps how long it was looked for.
    Nothing about it runs inside the window, so it cannot delay the joint-state callbacks the
    gap rules time. Never a service client: the topic is the only thing asked of the driver.
    ``xarm_msgs`` is imported only then, so a machine without it says so instead of failing.
    """

    def __init__(self, node, data, subscribe):
        self._node = node
        self._subscribe = subscribe
        self._done = False
        self._record = data['robot_states'] = {'topics': {}, 'topic': None, 'unavailable': None,
                                               'messages': [], 'searched_s': None}

    def poll(self, searched_s: float):
        """
        Subscribe when one publisher of the type has appeared; record every one seen.

        ``searched_s`` (the seconds since discovery began) is recorded on every call, found or
        not.
        """
        if self._done:
            return
        self._record['searched_s'] = searched_s
        topics = _arm_state_topics(self._node)
        if not topics:
            return
        self._record['topics'] = topics
        self._done = True
        if sum(len(p) for p in topics.values()) > 1:
            print(f'several publishers of {rules.ARM_STATE_TYPE} ({topics}): none read',
                  file=sys.stderr, flush=True)
            return
        topic = next(iter(topics))
        try:
            from xarm_msgs.msg import RobotMsg
        except ImportError as exc:
            print(f'xarm_msgs cannot be imported ({exc}): {topic} not read',
                  file=sys.stderr, flush=True)
            self._record['unavailable'] = 'xarm_msgs not installed'
            return
        self._record['topic'] = topic
        messages = self._record['messages']

        def on_state(msg):
            messages.append((time.monotonic(), int(msg.err), int(msg.warn), int(msg.state),
                             int(msg.mode), int(msg.mt_brake), int(msg.mt_able)))

        self._subscribe(RobotMsg, topic, on_state, ROBOT_STATES_DEPTH, 'reliable', 'volatile')


def _subscribe_cameras(node, opts, data):
    from sensor_msgs.msg import Image

    for spec in opts.streams:
        rec = {'mono': [], 'stamps': [], 'first': None}
        data['cameras'][spec['name']] = rec
        q = dict(CAMERA_DEFAULT_QOS, **(spec.get('qos') or {}))
        data['qos'][spec['topic']] = f"{q['reliability']}, volatile, depth {q['depth']}"

        def on_image(msg, rec=rec):
            rec['mono'].append(time.monotonic())
            rec['stamps'].append(_stamp(msg.header))
            if rec['first'] is None:
                rec['first'] = (msg.width, msg.height, msg.encoding)

        node.create_subscription(Image, spec['topic'], on_image,
                                 _qos(q['reliability'], 'volatile', int(q['depth'])))


# ---- main -------------------------------------------------------------------------------

def _versions_block(data: dict) -> dict:
    gate = data.get('gate') or {}
    return {'packages': data.get('versions', {}),
            'xarm_ros2': {k: gate.get(k) for k in ('head', 'describe', 'evidence', 'verdict')},
            'ros_distro': os.environ.get('ROS_DISTRO')}


def _gather_versions() -> dict:
    from ament_index_python.packages import get_package_share_directory
    return sysinfo.package_versions(VERSION_PACKAGES, get_package_share_directory)


def main(argv: Optional[List[str]] = None) -> int:
    """Run one stage; return the exit status (0 all PASS, 1 any FAIL, 2 not judged)."""
    print(rep.BANNER, flush=True)
    from rclpy.utilities import remove_ros_args
    raw = list(sys.argv if argv is None else ['hw_check'] + list(argv))
    args = build_parser().parse_args(remove_ros_args(raw)[1:])
    opts, problem = resolve_options(args)
    if problem:
        print(f'refused: {problem}')
        return 2
    problem = rep.report_dir_problem(args.data_dir)
    if problem:
        print(f'refused: {problem}')
        return 2
    seconds = args.seconds if args.seconds is not None else rules.DEFAULT_SECONDS[args.stage]
    if not seconds > 0:
        print('refused: --seconds must be positive')
        return 2
    start = datetime.datetime.now().astimezone()
    data = collect(args.stage, opts, seconds)
    end = datetime.datetime.now().astimezone()
    if args.stage != 'cameras':
        data['machine'] = sysinfo.machine_info()
        data['versions'] = _gather_versions()
        data['gate'] = sysinfo.driver_gate(args.workspace)
    measures, results = rules.evaluate(args.stage, data, opts)
    measures['subscription_qos'] = data['qos']
    desc = data.get('description')
    if desc is not None:
        measures['description'] = desc
        mode = {None: 'UNKNOWN', True: 'MOCK/SIMULATOR', False: 'HARDWARE'}[desc['simulator']]
        print(f'backend: {mode}', flush=True)
    status = rules.exit_status(results)
    print(rep.format_results(results), flush=True)
    path = rep.report_path(args.data_dir, args.stage, start)
    document = rep.build_report(args.stage, start, end, measures, results,
                                _versions_block(data), raw, data['warnings'],
                                rep.code_commit(), status)
    try:
        rep.write_report(path, document)
    except OSError as exc:
        print(f'cannot write the report {path}: {exc}', file=sys.stderr)
        return 2
    print(f'report: {path}')
    return status


if __name__ == '__main__':
    sys.exit(main())
