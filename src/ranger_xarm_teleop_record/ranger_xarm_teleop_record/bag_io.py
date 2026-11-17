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
Read one recorded episode into stream tables (ROS: rosbag2_py and message types).

Reading a bag joins no ROS graph. Every message is taken as recorded: its header stamp and
the recorder's receive time, in the recorder's order. Image pixels are not held in memory:
``Episode.images[name](row)`` reads one frame, sequentially (rows asked in increasing order
cost one pass).
"""

import os
from typing import Dict, List

import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

from . import metadata as md
from .streams import Episode, Stream

KIND_FIELDS = {
    'teleop_state': (
        ('armed', bool), ('motion_enabled', bool), ('clutch_engaged', bool),
        ('calibrated', bool), ('measured_tool_pose_valid', bool), ('damped', bool),
        ('keepout_bound', bool), ('gripper_enabled', bool), ('gripper_active', bool),
        ('gripper_measured_valid', bool), ('owner', int), ('mux_status', int),
        ('hold_reason', int), ('active_limit', int), ('servo_status', int),
        ('gripper_hold', int), ('gripper_request', int), ('gripper_reference_source', int),
        ('calibration_session_id', int), ('calibration_reset_count', int),
        ('input_seq', int), ('calibration_heading', float), ('input_age', float),
        ('damping', float), ('lead_translation', float), ('lead_rotation', float),
        ('translation_scale', float), ('gripper_command', float),
        ('gripper_measured', float)),
    'gripper_command': (('position', float), ('request', int), ('reference_source', int)),
    'input_status': (('session_id', int), ('seq', int), ('reset_count', int),
                     ('xr_time_ms', float), ('client_time_ms', float)),
    'image': (('width', int), ('height', int), ('step', int)),
    'camera_info': (('width', int), ('height', int)),
    'owner': (('owner', int), ('status', int)),
    'episode_state': (('state', int), ('wall_time', float)),
}


def _secs(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _pose(p) -> List[float]:
    return [p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w]


def _twist(t) -> List[float]:
    return [t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z]


def _reader(path: str, topics=None):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    if topics:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return reader


class _Rows:
    """Accumulates one stream's rows."""

    def __init__(self, name, spec):
        self.name, self.spec = name, spec
        self.stamp, self.recv, self.frame_id = [], [], []
        self.fields: Dict[str, list] = {}
        self.labels: Dict[str, list] = {}
        self.joint_names: List[str] = []

    def put(self, key, value):
        self.fields.setdefault(key, []).append(value)

    def joints(self, names, *arrays):
        if not self.joint_names:
            self.joint_names = list(names)
        index = {n: i for i, n in enumerate(names)}
        out = []
        for arr in arrays:
            row = [float(arr[index[n]]) if n in index and index[n] < len(arr) else np.nan
                   for n in self.joint_names]
            out.append(row)
        return out


def _add(rows: _Rows, msg, recv: float) -> None:
    kind = rows.spec['kind']
    header = getattr(msg, 'header', None)
    if header is not None:
        rows.stamp.append(_secs(header.stamp))
        rows.frame_id.append(header.frame_id)
    else:
        rows.stamp.append(recv)
        rows.frame_id.append('')
    rows.recv.append(recv)
    for key, cast in KIND_FIELDS.get(kind, ()):
        rows.put(key, cast(getattr(msg, key)))
    if kind == 'teleop_state':
        rows.put('measured_tool_pose', _pose(msg.measured_tool_pose))
        rows.put('gripper_measured_stamp', _secs(msg.gripper_measured_stamp))
    elif kind == 'pose':
        rows.put('pose', _pose(msg.pose))
    elif kind == 'joint_command':
        (vel,) = rows.joints(msg.joint_names, msg.velocities)
        rows.put('velocities', vel)
    elif kind == 'joint_states':
        pos, vel, eff = rows.joints(msg.name, msg.position, msg.velocity, msg.effort)
        rows.put('position', pos)
        rows.put('velocity', vel)
        rows.put('effort', eff)
    elif kind == 'image':
        rows.labels.setdefault('encoding', []).append(msg.encoding)
    elif kind == 'camera_info':
        rows.put('k', list(msg.k))
        rows.labels.setdefault('distortion_model', []).append(msg.distortion_model)
    elif kind == 'input_status':
        rows.put('right_present', bool(msg.right.pose_present))
        rows.put('left_present', bool(msg.left.pose_present))
    elif kind == 'joy':
        rows.put('axes', list(msg.axes))
        rows.put('buttons', list(msg.buttons))
    elif kind == 'odometry':
        rows.put('pose', _pose(msg.pose.pose))
        rows.put('twist', _twist(msg.twist.twist))
    elif kind == 'twist':
        rows.put('twist', _twist(msg))


def _image_array(msg) -> np.ndarray:
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    h, w, enc = msg.height, msg.width, msg.encoding
    if enc in ('rgb8', 'bgr8'):
        img = data.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
        return img[:, :, ::-1].copy() if enc == 'bgr8' else img.copy()
    if enc in ('rgba8', 'bgra8'):
        img = data.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4)[:, :, :3]
        return img[:, :, ::-1].copy() if enc == 'bgra8' else img.copy()
    if enc == 'mono8':
        return data.reshape(h, msg.step)[:, :w].copy()
    if enc in ('16UC1', 'mono16'):
        return data.view(np.uint16).reshape(h, msg.step // 2)[:, :w].copy()
    if enc == '32FC1':
        return data.view(np.float32).reshape(h, msg.step // 4)[:, :w].copy()
    raise ValueError(f'image encoding {enc} not supported by the export')


class _ImageLoader:
    """Read frames of one image topic by row, sequentially."""

    def __init__(self, path, topic, type_name):
        self.path, self.topic = path, topic
        self.cls = get_message(type_name)
        self.reader, self.row = None, -1

    def __call__(self, row: int) -> np.ndarray:
        if self.reader is None or row <= self.row:
            self.reader, self.row = _reader(self.path, [self.topic]), -1
        while self.reader.has_next():
            topic, data, _ = self.reader.read_next()
            if topic != self.topic:
                continue
            self.row += 1
            if self.row == row:
                return _image_array(deserialize_message(data, self.cls))
        raise IndexError(f'{self.topic}: no row {row}')


def read_episode(session_dir: str, sidecar_path: str) -> Episode:
    """Return the stream tables of one episode of a session (paths relative to it)."""
    sidecar = md.read_json(sidecar_path)
    profile = sidecar.get('profile') or {}
    bag = os.path.join(session_dir, sidecar['episode']['file'])
    by_topic = {s['topic']: (n, s) for n, s in (profile.get('streams') or {}).items()
                if s.get('record', True)}
    reader = _reader(bag)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    classes, rows = {}, {}
    images = {}
    while reader.has_next():
        topic, data, recv_ns = reader.read_next()
        if topic not in by_topic:
            continue
        name, spec = by_topic[topic]
        if topic not in classes:
            classes[topic] = get_message(types[topic])
            rows[name] = _Rows(name, spec)
        msg = deserialize_message(data, classes[topic])
        _add(rows[name], msg, recv_ns * 1e-9)
    streams = {}
    for name, r in rows.items():
        spec = r.spec
        streams[name] = Stream(
            name=name, topic=spec['topic'], kind=spec['kind'], stamp=np.asarray(r.stamp),
            recv=np.asarray(r.recv),
            fields={k: np.asarray(v) for k, v in r.fields.items()},
            frame_id=r.frame_id, joint_names=r.joint_names, labels=r.labels,
            has_header=spec.get('clock') != 'none')
        if spec['kind'] == 'image':
            images[name] = _ImageLoader(bag, spec['topic'], types[spec['topic']])
    return Episode(streams=streams, sidecar=sidecar, profile=profile, images=images)
