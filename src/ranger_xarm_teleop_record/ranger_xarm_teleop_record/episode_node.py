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
Episode node: episode control from the left controller, driving a running rosbag2 recorder.

It starts with the stack, before arming, next to ``ros2 bag record --start-paused`` (see
``record.launch.py``). Every subscription, publisher and service client is created here, at
start-up; an episode event is only a service call to the recorder, which already exists, so
nothing joins or leaves the ROS graph during a session (INV-1). The teleop node does not
know this node exists.

START resumes the recorder; STOP pauses it and splits the bag, so each episode is one closed
MCAP file. The sidecar (episodes/NNNN.json) holds what makes the episode interpretable by
itself: latched data (description, static transforms, camera intrinsics) arrive while the
recorder is paused and never reach the bag, so they are captured here.
"""

import copy
import math
import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters, ListParameters
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data, QoSProfile,
                       ReliabilityPolicy)
from ranger_xarm_teleop_interfaces.msg import EpisodeState, InputStatus, TeleopState
from ranger_xarm_teleop_servo.rotations import Pose
from rosbag2_interfaces.msg import WriteSplitEvent
from rosbag2_interfaces.srv import Pause, Resume, SplitBagfile
from sensor_msgs.msg import CameraInfo, JointState, Joy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from . import metadata as md
from .episode import buttons_from_joy, Control, Event, State
from .extrinsics import Frames
from .linkstats import link_quality
from .profile import load_checks, load_profile, recorded
from .provenance import git_info, sha256_file, sha256_text

# Joy as the teleop node reads it (contract: best effort): a lost message is not a release.
JOY_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                     depth=5, durability=DurabilityPolicy.VOLATILE)
RELIABLE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST, depth=50)
LATCHED_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
TF_STATIC_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                           history=HistoryPolicy.KEEP_LAST, depth=100,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)
STATE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                       depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

FAULT_HOLDS = {3, 4} | set(range(6, 21))      # TeleopState.HOLD_*: not 0, 1, 2 (disarm), 5 (grip)
HOLD_NAMES = {v: k[len('HOLD_'):] for k, v in vars(TeleopState).items()
              if k.startswith('HOLD_') and isinstance(v, int)}
SUBSCRIBED_KINDS = ('teleop_state', 'joint_states', 'camera_info', 'input_status', 'joy')
RATE_WINDOW_S = 2.0
PUBLISH_PERIOD_S = 0.2
CLOCK_PAIR_PERIOD_S = 1.0


def _secs(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _param_value(value):
    t = value.type
    return {ParameterType.PARAMETER_BOOL: lambda: value.bool_value,
            ParameterType.PARAMETER_INTEGER: lambda: value.integer_value,
            ParameterType.PARAMETER_DOUBLE: lambda: value.double_value,
            ParameterType.PARAMETER_STRING: lambda: value.string_value,
            ParameterType.PARAMETER_BYTE_ARRAY: lambda: list(value.byte_array_value),
            ParameterType.PARAMETER_BOOL_ARRAY: lambda: list(value.bool_array_value),
            ParameterType.PARAMETER_INTEGER_ARRAY: lambda: list(value.integer_array_value),
            ParameterType.PARAMETER_DOUBLE_ARRAY: lambda: list(value.double_array_value),
            ParameterType.PARAMETER_STRING_ARRAY: lambda: list(value.string_array_value),
            }.get(t, lambda: None)()


class StreamWatch:
    """Arrival stamps of one subscribed stream: rate for readiness, counts for an episode."""

    def __init__(self, name: str, spec: dict):
        """Watch the profile stream ``name``."""
        self.name, self.spec = name, spec
        self.recent: List[float] = []          # header stamps, last RATE_WINDOW_S
        self.last_arrival: Optional[float] = None
        self.episode: Optional[List[float]] = None

    def add(self, stamp: float, arrival: float) -> None:
        """Note one message."""
        self.last_arrival = arrival
        self.recent.append(stamp)
        cut = stamp - RATE_WINDOW_S
        while self.recent and (self.recent[0] < cut or self.recent[0] > stamp + 1.0):
            self.recent.pop(0)
        if self.episode is not None:
            self.episode.append(stamp)

    def problem(self, now: float) -> Optional[str]:
        """Return why this stream is not usable now, or None."""
        if self.last_arrival is None:
            return f'{self.name}: no message on {self.spec["topic"]}'
        if now - self.last_arrival > 1.0:
            return f'{self.name}: silent for {now - self.last_arrival:.1f} s'
        rate = self.spec.get('rate_hz')
        if rate is None or len(self.recent) < 2:
            return None if rate is None else f'{self.name}: too few messages to judge the rate'
        span = self.recent[-1] - self.recent[0]
        measured = (len(self.recent) - 1) / span if span > 0 else 0.0
        if measured < 0.9 * float(rate):
            return f'{self.name}: {measured:.1f} Hz, below its nominal {float(rate):g} Hz'
        return None


class EpisodeNode(Node):
    """Episode control, sidecar and recorder calls."""

    def __init__(self):
        """Create every endpoint now; nothing is created later."""
        super().__init__('episode_node')
        self.declare_parameter('profile', 'mock')
        self.declare_parameter('session_dir', '')
        self.declare_parameter('recorder', '/teleop_episode_recorder')
        self.declare_parameter('limits_file', '')
        self.declare_parameter('teleop_node', '/teleop_servo')
        self.declare_parameter('servo_node', '/servo_node')
        self.declare_parameter('mux_node', '/arm_mux')
        self.declare_parameter('tool_frame', '')
        self.declare_parameter('teleop_repo', '')
        self.declare_parameter('robot_workspace', '')
        self.declare_parameter('discard_hold_s', 1.0)
        self.declare_parameter('state_timeout_s', 0.5)
        self.declare_parameter('checks_file', '')
        self.declare_parameter('record_extra', '')
        gp = self.get_parameter
        self.profile = load_profile(str(gp('profile').value))
        for name in [n for n in str(gp('record_extra').value).split(',') if n]:
            self.profile['streams'][name]['record'] = True     # unknown name: KeyError, stop
        self.checks = load_checks(str(gp('checks_file').value) or None)
        self.session_dir = os.path.abspath(os.path.expanduser(str(gp('session_dir').value)))
        if not gp('session_dir').value:
            raise RuntimeError('session_dir is required (record.launch.py creates it)')
        problem = md.data_dir_problem(self.session_dir)
        if problem:
            raise RuntimeError(problem)
        os.makedirs(os.path.join(self.session_dir, 'episodes'), exist_ok=True)
        self.base_frame = self.profile['frames']['base']
        self.tool_frame = str(gp('tool_frame').value) or self.profile['frames']['tool']
        self.state_timeout = float(gp('state_timeout_s').value)
        self.margin = float(self.checks['streams']['count_window_margin_s'])
        self.data_dir = os.path.dirname(self.session_dir)
        self.control = Control(discard_hold_s=float(gp('discard_hold_s').value))

        self.streams = recorded(self.profile)
        self.watch: Dict[str, StreamWatch] = {}
        self.latest_info: Dict[str, CameraInfo] = {}
        self.state: Optional[TeleopState] = None
        self.state_arrival: Optional[float] = None
        self.urdf: Optional[str] = None
        self.static: Dict[str, tuple] = {}
        self.split_files: List[str] = []
        self.episode: Optional[dict] = None
        self.ep_log: Dict[str, list] = {}
        self.index = self._last_index()
        self.kept = self.discarded = 0
        self.event = (Event.NONE, '', None)
        self.event_count = 0
        self.disk_rate: Optional[float] = None
        self._disk_sample = None
        self._last_pair = 0.0
        self._not_ready: List[str] = ['starting']

        for name, spec in self.streams.items():
            if spec['kind'] not in SUBSCRIBED_KINDS:
                continue
            self.watch[name] = StreamWatch(name, spec)
            self._subscribe(name, spec)
        self.create_subscription(String, '/robot_description', self.on_description, LATCHED_QOS)
        self.create_subscription(TFMessage, '/tf_static', self.on_tf_static, TF_STATIC_QOS)
        rec = str(gp('recorder').value).rstrip('/')
        self.recorder = rec
        self.pause_cli = self.create_client(Pause, f'{rec}/pause')
        self.resume_cli = self.create_client(Resume, f'{rec}/resume')
        self.split_cli = self.create_client(SplitBagfile, f'{rec}/split_bagfile')
        self.create_subscription(WriteSplitEvent, '/events/write_split', self.on_split,
                                 RELIABLE_QOS)
        self.param_clients = {}
        for key in ('teleop_node', 'servo_node', 'mux_node'):
            target = str(gp(key).value).rstrip('/')
            self.param_clients[key] = (
                target,
                self.create_client(ListParameters, f'{target}/list_parameters'),
                self.create_client(GetParameters, f'{target}/get_parameters'))
        self.parameters: Dict[str, object] = {}     # key -> request time, or 'done'
        self.state_pub = self.create_publisher(EpisodeState, '/teleop/episode/state', STATE_QOS)
        self.session = self._session_record(gp)
        self._write_session()
        self.create_timer(PUBLISH_PERIOD_S, self.on_timer)
        self.create_timer(1.0, self.fetch_parameters)
        self.get_logger().info(
            f'episode node ready: profile {self.profile["name"]}, session '
            f'{os.path.basename(self.session_dir)}, recorder {rec}; X start/stop, '
            f'Y keep, Y held {self.control.discard_hold_s:g} s discard, left trigger mark')

    # ---------------------------------------------------------------- subscriptions
    def _subscribe(self, name: str, spec: dict) -> None:
        kind = spec['kind']
        if kind == 'teleop_state':
            self.create_subscription(TeleopState, spec['topic'], self.on_state, RELIABLE_QOS)
        elif kind == 'joint_states':
            self.create_subscription(JointState, spec['topic'],
                                     lambda m, n=name: self._note(n, m.header.stamp),
                                     qos_profile_sensor_data)
        elif kind == 'camera_info':
            self.create_subscription(CameraInfo, spec['topic'],
                                     lambda m, n=name: self.on_info(n, m),
                                     qos_profile_sensor_data)
        elif kind == 'input_status':
            self.create_subscription(InputStatus, spec['topic'],
                                     lambda m, n=name: self.on_status(n, m), RELIABLE_QOS)
        elif kind == 'joy':
            self.create_subscription(Joy, spec['topic'], lambda m, n=name: self.on_joy(n, m),
                                     JOY_QOS)

    def _note(self, name: str, stamp) -> None:
        self.watch[name].add(_secs(stamp), time.monotonic())

    def on_info(self, name: str, msg: CameraInfo) -> None:
        """Count a camera_info (one per image) and keep the latest intrinsics."""
        self._note(name, msg.header.stamp)
        self.latest_info[name] = msg

    def on_status(self, name: str, msg: InputStatus) -> None:
        """Count input status and keep what the link numbers need."""
        self._note(name, msg.header.stamp)
        if self.episode is not None:
            self.ep_log['status'].append((msg.seq, msg.xr_time_ms, _secs(msg.header.stamp),
                                          msg.session_id))

    def on_description(self, msg: String) -> None:
        """Keep the robot description; write its copy and fingerprint once per change."""
        if msg.data == self.urdf:
            return
        self.urdf = msg.data
        path = os.path.join(self.session_dir, 'robot_description.urdf')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(msg.data)
        self.session['robot'].update({'description_sha256': sha256_text(msg.data),
                                      'description_file': 'robot_description.urdf'})
        self._write_session()

    def on_tf_static(self, msg: TFMessage) -> None:
        """Keep every static transform seen (calibration, optical frames)."""
        for t in msg.transforms:
            r = t.transform.rotation
            v = t.transform.translation
            self.static[t.child_frame_id] = (t.header.frame_id, t.child_frame_id,
                                             Pose((v.x, v.y, v.z), (r.x, r.y, r.z, r.w)))

    def on_split(self, msg: WriteSplitEvent) -> None:
        """Track the bag file the next episode goes to."""
        self.split_files.append(msg.opened_file)

    def on_state(self, msg: TeleopState) -> None:
        """Teleop state: readiness, auto-stop on disarm, live flags, episode logs."""
        prev = self.state
        self.state, self.state_arrival = msg, time.monotonic()
        name = next((n for n, w in self.watch.items() if w.spec['kind'] == 'teleop_state'),
                    None)
        if name:
            self._note(name, msg.header.stamp)
        if self.episode is None or self.control.state != State.RECORDING:
            return
        t = _secs(msg.header.stamp)
        log = self.ep_log
        if msg.armed:
            log['input_age'].append(msg.input_age)
        log['scales'].add(round(float(msg.translation_scale), 6))
        cal = (msg.calibrated, round(msg.calibration_heading, 9), msg.calibration_session_id,
               msg.calibration_reset_count)
        if cal != log['calibration']:
            log['calibration_changed'] = True
        if prev is not None and prev.clutch_engaged and not msg.clutch_engaged \
                and msg.hold_reason in FAULT_HOLDS:
            self._flag('FAULT_RELEASE', HOLD_NAMES.get(msg.hold_reason, str(msg.hold_reason)),
                       t)
        if prev is not None and prev.armed and not msg.armed:
            self._flag('DISARMED', 'teleop disarmed during the episode', t)
            self.on_events(self.control.auto_stop('disarmed'))

    def on_joy(self, name: str, msg: Joy) -> None:
        """Left-controller presses."""
        self._note(name, msg.header.stamp)
        b = buttons_from_joy(msg.buttons, msg.axes)
        if b is None:
            return
        now = time.monotonic()
        problems = self.start_problems(now) if self.control.state == State.IDLE else \
            self.ready_problems(now)
        self.on_events(self.control.joy(b, now, problems))

    # ---------------------------------------------------------------- readiness
    def ready_problems(self, now: float) -> List[str]:
        """Return why recording is not possible (recorder, required streams, labels, disk)."""
        out = []
        if not (self.resume_cli.service_is_ready() and self.pause_cli.service_is_ready()
                and self.split_cli.service_is_ready()):
            out.append(f'recorder {self.recorder} not reachable')
        for name, spec in self.streams.items():
            if not spec.get('required') or spec.get('gate') or name == 'episode_state':
                continue
            if name in self.watch:
                p = self.watch[name].problem(now)
                if p:
                    out.append(p)
            elif self.count_publishers(spec['topic']) == 0:
                out.append(f'{name}: no publisher on {spec["topic"]}')
            elif spec['kind'] == 'image' and spec.get('info') in self.watch:
                p = self.watch[spec['info']].problem(now)
                if p:
                    out.append(f'{name} (by its camera_info) {p}')
        if self.urdf is None:
            out.append('no robot description yet')
        if 'teleop' not in self.session['limits']['parameters']:
            out.append('teleop parameters (limits in force) not read yet')
        if self.session['robot'].get('semantic_sha256') is None:
            out.append('semantic description (from MoveIt Servo) not read yet')
        free = self.disk_free_gb()
        if free < float(self.profile['disk']['min_free_gb_to_start']):
            out.append(f'disk: {free:.1f} GB free, below '
                       f'{self.profile["disk"]["min_free_gb_to_start"]} GB')
        return out

    def start_problems(self, now: float) -> List[str]:
        """Return why START is refused now."""
        out = self.ready_problems(now)
        labels, problem = md.read_labels(self.data_dir)
        if problem:
            out.append(problem)
        s = self.state
        if s is None or not s.armed:
            out.append('teleop is not armed')
        elif s.owner != 2:                                   # ArmOwner.OWNER_TELEOP
            out.append('teleop does not own the arm yet')
        return out

    def disk_free_gb(self) -> float:
        """Return the free space where the session is written."""
        return shutil.disk_usage(self.session_dir).free / 1e9

    # ---------------------------------------------------------------- events
    def on_events(self, outcomes) -> None:
        """Carry out the state machine's events."""
        for o in outcomes:
            self.event = (o.event, o.detail, self.get_clock().now().to_msg())
            self.event_count += 1
            if o.event == Event.START:
                self.begin()
            elif o.event in (Event.STOP, Event.AUTO_STOP):
                self.end(o.detail if o.event == Event.AUTO_STOP else 'operator')
            elif o.event in (Event.KEEP, Event.DISCARD):
                self.resolve('kept' if o.event == Event.KEEP else 'discarded')
            elif o.event == Event.MARK and self.episode is not None:
                self.episode['episode']['marks'].append(self._now())
            log = self.get_logger().warning if o.event == Event.REFUSED else \
                self.get_logger().info
            log(f'episode {self.index}: {o.event.name}' + (f' ({o.detail})' if o.detail else ''))
            self.publish_state()

    def _now(self) -> dict:
        return {'ros': self.get_clock().now().nanoseconds * 1e-9, 'wall': time.time(),
                'mono': time.monotonic()}

    def _flag(self, flag: str, reason: str, t: Optional[float]) -> None:
        if self.episode is not None:
            self.episode['episode']['flags'].append({'flag': flag, 'reason': reason, 'time': t})

    def begin(self) -> None:
        """START: capture the sidecar's start block, then resume the recorder."""
        self.index += 1
        labels, _ = md.read_labels(self.data_dir)
        s = self.state
        bag_file = self._current_file()
        self.episode = {
            'format': md.FORMAT,
            'episode': {'index': self.index, 'status': 'recording',
                        'status_history': [], 'file': bag_file, 'labels': labels,
                        'start': self._now(), 'stop': None, 'stop_reason': None,
                        'marks': [], 'flags': []},
            'session': {'name': os.path.basename(self.session_dir)},
            'calibration': {'calibrated': bool(s.calibrated),
                            'heading': float(s.calibration_heading),
                            'session_id': int(s.calibration_session_id),
                            'reset_count': int(s.calibration_reset_count),
                            'frame': self._calibration_transform(),
                            'changed_during_episode': False},
            'scale': {'at_start': float(s.translation_scale), 'values_used': []},
            'cameras': self._cameras(),
            'base': self._base(),
        }
        for key in ('software', 'robot', 'limits', 'frames', 'time', 'recorder', 'profile'):
            self.episode[key] = copy.deepcopy(self.session[key])
        self.ep_log = {'input_age': [], 'status': [], 'scales': set(),
                       'calibration': (s.calibrated, round(s.calibration_heading, 9),
                                       s.calibration_session_id, s.calibration_reset_count),
                       'calibration_changed': False, 'pairs': []}
        self._status('recording')
        self._write_episode()
        future = self.resume_cli.call_async(Resume.Request())
        future.add_done_callback(self._resumed)

    def _resumed(self, future) -> None:
        ok = future.exception() is None and future.result() is not None
        if not ok:
            self.episode['episode']['status'] = 'failed'
            self._flag('RECORDER_ERROR', f'resume failed: {future.exception()}', None)
            self._write_episode()
            self.episode = None
            self.index -= 1
            self.on_events(self.control.done(False, str(future.exception())))
            return
        self.episode['episode']['start']['acknowledged'] = self._now()
        for w in self.watch.values():
            w.episode = [] if w.spec.get('count') else None
        self._last_pair = 0.0
        self._pair()
        self.control.done(True)
        self._write_episode()
        self.publish_state()

    def end(self, reason: str) -> None:
        """STOP or AUTO_STOP: pause the recorder, then split, then finish the sidecar."""
        if self.episode is None:
            self.control.done(True)
            return
        self._pair()
        self.episode['episode']['stop'] = self._now()
        self.episode['episode']['stop_reason'] = reason
        self._status('pending')
        future = self.pause_cli.call_async(Pause.Request())
        future.add_done_callback(self._paused)

    def _paused(self, future) -> None:
        if future.exception() is not None:
            self._flag('RECORDER_ERROR', f'pause failed: {future.exception()}', None)
        self.split_cli.call_async(SplitBagfile.Request()).add_done_callback(self._split)

    def _split(self, future) -> None:
        if future.exception() is not None:
            self._flag('RECORDER_ERROR', f'split failed: {future.exception()}', None)
        self._finish()
        self.control.done(True)
        self.publish_state()

    def _finish(self) -> None:
        ep = self.episode
        e = ep['episode']
        start = e['start'].get('acknowledged', e['start'])
        stop = e['stop']
        counts = {}
        for name, w in self.watch.items():
            if w.episode is None:
                continue
            clock = w.spec.get('clock')
            a, b = ((start['wall'], stop['wall']) if clock == 'host_wall'
                    else (start['ros'], stop['ros']))
            a, b = a + self.margin, b - self.margin
            stamps = np.asarray(w.episode, dtype=float)
            counts[name] = {'window': [a, b], 'clock': clock,
                            'count': int(np.count_nonzero((stamps >= a) & (stamps <= b)))}
            w.episode = None
        ep['counts'] = counts
        log = self.ep_log
        st = log['status']
        ep['link'] = link_quality(
            log['input_age'], [s[0] for s in st], [s[1] for s in st], [s[2] for s in st],
            [s[3] for s in st], float(self.checks['flags']['reconnect_gap_s']))
        ep['scale']['values_used'] = sorted(log['scales']) or [ep['scale']['at_start']]
        ep['calibration']['changed_during_episode'] = bool(log['calibration_changed'])
        ep['time'] = dict(ep['time'], clock_pairs=log['pairs'])
        pairs = log['pairs']
        if len(pairs) >= 2 and pairs[-1][1] > pairs[0][1]:
            rtf = (pairs[-1][0] - pairs[0][0]) / (pairs[-1][1] - pairs[0][1])
            ep['rtf'] = rtf
            low = self.profile.get('low_rtf')
            if low is not None and rtf < float(low):
                self._flag('LOW_RTF', f'real-time factor {rtf:.2f} below {low}', None)
        self._write_episode()

    def resolve(self, status: str) -> None:
        """KEEP or DISCARD: mark the episode; nothing is deleted."""
        if self.episode is None:
            return
        self._status(status)
        self._write_episode()
        if status == 'kept':
            self.kept += 1
        else:
            self.discarded += 1
        self.episode = None

    def _status(self, status: str) -> None:
        e = self.episode['episode']
        e['status'] = status
        e['status_history'].append(dict(self._now(), status=status))

    # ---------------------------------------------------------------- periodic
    def on_timer(self) -> None:
        """Readiness, auto-stops, the Y hold, clock pairs, disk, the state message."""
        now = time.monotonic()
        problems = self.ready_problems(now)
        self.control.set_ready(not problems, '; '.join(problems))
        self._not_ready = problems
        self.on_events(self.control.tick(now))
        if self.control.state == State.RECORDING and self.episode is not None \
                and not self.control.busy:
            if self.state_arrival is None or now - self.state_arrival > self.state_timeout:
                self._flag('TELEOP_STATE_LOST', 'no teleop state for '
                           f'{self.state_timeout:g} s', None)
                self.on_events(self.control.auto_stop('teleop state lost'))
            elif self.disk_free_gb() < float(self.profile['disk']['min_free_gb_to_continue']):
                self._flag('DISK_LOW', f'{self.disk_free_gb():.1f} GB free', None)
                self.on_events(self.control.auto_stop('disk low'))
            elif now - self._last_pair >= CLOCK_PAIR_PERIOD_S:
                self._pair()
        self._disk()
        self.publish_state()

    def _pair(self) -> None:
        if self.ep_log.get('pairs') is None:
            return
        self._last_pair = time.monotonic()
        n = self._now()
        self.ep_log['pairs'].append([n['ros'], n['wall'], n['mono']])

    def _disk(self) -> None:
        bag = os.path.join(self.session_dir, 'bag')
        if self.control.state != State.RECORDING or not os.path.isdir(bag):
            self._disk_sample = None
            return
        size = sum(e.stat().st_size for e in os.scandir(bag) if e.is_file())
        t = time.monotonic()
        if self._disk_sample is not None and t - self._disk_sample[0] >= 2.0:
            rate = (size - self._disk_sample[1]) / (t - self._disk_sample[0])
            if rate > 0:
                self.disk_rate = rate
            self._disk_sample = (t, size)
        elif self._disk_sample is None:
            self._disk_sample = (t, size)

    def publish_state(self) -> None:
        """Publish the episode state (contract: on change and at 5 Hz)."""
        m = EpisodeState()
        now = self.get_clock().now()
        m.header.stamp = now.to_msg()
        m.wall_time = time.time()
        m.state = int(self.control.state)
        m.session = os.path.basename(self.session_dir)
        m.episode_index = self.index
        if self.episode is not None:
            m.elapsed = now.nanoseconds * 1e-9 - self.episode['episode']['start']['ros']
            m.flags = sorted({f['flag'] for f in self.episode['episode']['flags']})
        m.kept_count, m.discarded_count = self.kept, self.discarded
        ev, detail, stamp = self.event
        m.last_event = int(ev)
        if stamp is not None:
            m.last_event_stamp = stamp
        m.event_count = self.event_count
        m.detail = detail
        if self.control.state == State.NOT_READY or ev == Event.NONE:
            m.detail = '; '.join(self._not_ready) or detail
        m.discard_armed = self.control.discard_armed
        m.disk_free_gb = self.disk_free_gb()
        m.disk_minutes_left = (m.disk_free_gb * 1e9 / self.disk_rate / 60.0
                               if self.disk_rate else math.nan)
        self.state_pub.publish(m)

    # ---------------------------------------------------------------- metadata
    def _last_index(self) -> int:
        last = 0
        for path in md.iter_episodes(self.session_dir):
            try:
                last = max(last, int(os.path.basename(path)[:-5]))
            except ValueError:
                continue
        return last

    def _current_file(self) -> str:
        bag = os.path.join(self.session_dir, 'bag')
        if self.split_files:
            path = self.split_files[-1]
        else:
            path = os.path.join(bag, 'bag_0.mcap')
        try:
            return md.relative_inside(self.session_dir, path)
        except ValueError:
            return os.path.basename(path)

    def _calibration_transform(self) -> Optional[dict]:
        for parent, child, pose in self.static.values():
            if parent == self.base_frame and child == 'quest_world':
                return {'parent': parent, 'child': child, 'xyz': list(pose.p),
                        'xyzw': list(pose.q)}
        return None

    def _cameras(self) -> dict:
        out = {}
        frames = None
        if self.urdf is not None:
            try:
                frames = Frames(self.urdf, self.static.values())
            except Exception as exc:  # noqa: B902 - a bad description is reported, not fatal
                self.get_logger().error(f'robot description not parsed: {exc}')
        for name, spec in self.streams.items():
            if spec['kind'] != 'image' or not spec.get('info'):
                continue
            camera = spec.get('camera') or name
            info = self.latest_info.get(spec['info'])
            rec = {'stream': name, 'topic': spec['topic'], 'intrinsics': None,
                   'extrinsics': None}
            if info is not None:
                rec['intrinsics'] = {
                    'width': info.width, 'height': info.height, 'k': list(info.k),
                    'd': list(info.d), 'r': list(info.r), 'p': list(info.p),
                    'distortion_model': info.distortion_model,
                    'frame_id': info.header.frame_id}
                if frames is not None:
                    rec['extrinsics'] = frames.extrinsics(info.header.frame_id,
                                                          self.base_frame, self.tool_frame)
            out[camera] = rec
        return out

    def _base(self) -> dict:
        present = [n for n, s in self.streams.items()
                   if s.get('base') and self.count_publishers(s['topic']) > 0]
        mode = ('parked: no base topics' if not present
                else 'base topics present: see the checker for motion')
        return {'mode': mode, 'topics_present': present}

    def _session_record(self, gp) -> dict:
        here = os.path.realpath(__file__)
        repo = str(gp('teleop_repo').value) or os.environ.get('TELEOP_CHECKOUT') or here
        ws = str(gp('robot_workspace').value) or os.environ.get('RANGER_XARM_WS') or ''
        if not ws:
            try:
                from ament_index_python.packages import get_package_share_directory
                ws = get_package_share_directory('ranger_xarm_description')
            except Exception:  # noqa: B902 - reported as an unknown commit
                ws = ''
        limits = str(gp('limits_file').value)
        limits_rec = {'file': None, 'sha256': None, 'parameters': {}}
        if limits and os.path.isfile(limits):
            shutil.copyfile(limits, os.path.join(self.session_dir, 'limits.yaml'))
            limits_rec.update({'file': 'limits.yaml', 'source_name': os.path.basename(limits),
                               'sha256': sha256_file(limits)})
        streams = self.streams
        sim = bool(self.profile['use_sim_time'])
        return {
            'format': md.SESSION_FORMAT,
            'name': os.path.basename(self.session_dir),
            'started_wall': time.time(),
            'software': {'teleop_repo': git_info(repo), 'robot_workspace': git_info(ws)},
            'robot': {'description_sha256': None, 'description_file': None,
                      'semantic_sha256': None, 'semantic_file': None},
            'limits': limits_rec,
            'frames': {'base': self.base_frame, 'tool': self.tool_frame,
                       'input': 'quest_world',
                       'calibration': f'{self.base_frame} -> quest_world (/tf_static)'},
            'time': {
                'use_sim_time': sim,
                'recording_clock': ('ROS time of the host: simulation time (/clock)' if sim
                                    else 'ROS time of the host: its system clock'),
                'streams': {n: s['clock'] for n, s in streams.items()},
                'headset': ('InputStatus.xr_time_ms: the headset XR frame time in ms on the '
                            "headset's own clock, raw; the bridge stamps input with the host "
                            'time at reception'),
                'clock_pairs': [],
            },
            'recorder': {'node': self.get_parameter('recorder').value,
                         'storage': self.profile['recorder']['storage'],
                         'storage_preset': self.profile['recorder']['storage_preset'],
                         'topics': sorted(s['topic'] for s in streams.values()),
                         'one_file_per_episode': True},
            'profile': self.profile,
        }

    def fetch_parameters(self) -> None:
        """Read the teleop, Servo and mux parameters once each (limits in force)."""
        now = time.monotonic()
        for key, (target, list_cli, get_cli) in self.param_clients.items():
            started = self.parameters.get(key)
            if started == 'done' or not list_cli.service_is_ready():
                continue
            if started is not None and now - started < 3.0:
                continue                      # a response can be lost before discovery settles
            self.parameters[key] = now
            req = ListParameters.Request()
            list_cli.call_async(req).add_done_callback(
                lambda f, k=key, c=get_cli: self._listed(k, c, f))

    def _listed(self, key, get_cli, future) -> None:
        if future.exception() is not None or future.result() is None:
            self.parameters.pop(key, None)
            return
        names = list(future.result().result.names)
        if len(names) <= 2:                  # only use_sim_time and the like: not set up yet
            self.parameters.pop(key, None)
            return
        # A parameter declared without a value makes a whole get_parameters reply empty, so
        # read in chunks and fall back to one name at a time.
        chunks = [names[i:i + 10] for i in range(0, len(names), 10)]
        self._read_chunks(key, get_cli, chunks, {}, [])

    def _read_chunks(self, key, get_cli, chunks, values, unset) -> None:
        if not chunks:
            self._got(key, values, unset)
            return
        chunk, rest = chunks[0], chunks[1:]
        req = GetParameters.Request()
        req.names = chunk
        get_cli.call_async(req).add_done_callback(
            lambda f: self._chunk_done(key, get_cli, chunk, rest, values, unset, f))

    def _chunk_done(self, key, get_cli, chunk, rest, values, unset, future) -> None:
        if future.exception() is not None or future.result() is None:
            self.parameters.pop(key, None)            # retried from the timer
            return
        got = list(future.result().values)
        if len(got) == len(chunk):
            values.update({n: _param_value(v) for n, v in zip(chunk, got)})
        elif len(chunk) > 1:
            rest = [[n] for n in chunk] + rest
        else:
            unset.append(chunk[0])
        self._read_chunks(key, get_cli, rest, values, unset)

    def _got(self, key, values, unset) -> None:
        if key == 'servo_node':
            semantic = values.pop('robot_description_semantic', None)
            values.pop('robot_description', None)
            if semantic:
                with open(os.path.join(self.session_dir, 'robot_description.srdf'), 'w',
                          encoding='utf-8') as handle:
                    handle.write(semantic)
                self.session['robot'].update({'semantic_sha256': sha256_text(semantic),
                                              'semantic_file': 'robot_description.srdf'})
        self.parameters[key] = 'done'
        name = {'teleop_node': 'teleop', 'servo_node': 'servo', 'mux_node': 'mux'}[key]
        self.session['limits']['parameters'][name] = values
        # Declared without a value: the node's built-in default applies, which the recorded
        # commit of this repo fixes.
        self.session['limits']['parameters'][name + '_unset'] = sorted(unset)
        if key == 'teleop_node' and values.get('tool_frame'):
            self.tool_frame = values['tool_frame']
            self.session['frames']['tool'] = self.tool_frame
        self._write_session()

    def _write_session(self) -> None:
        md.write_json_atomic(os.path.join(self.session_dir, 'session.json'), self.session)

    def _write_episode(self) -> None:
        md.write_json_atomic(md.episode_path(self.session_dir, self.index), self.episode)

    def shutdown(self) -> None:
        """Mark a running or unreviewed episode; neither becomes kept."""
        if self.episode is None:
            return
        status = self.episode['episode']['status']
        if status == 'recording':
            self.episode['episode']['stop'] = self._now()
            self.episode['episode']['stop_reason'] = 'shutdown'
            self._status('interrupted')
            self._write_episode()
        elif status == 'pending':
            self._flag('UNREVIEWED', 'shut down before keep or discard', None)
            self._write_episode()


def main(args=None):
    """Run the episode node."""
    rclpy.init(args=args)
    node = EpisodeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
