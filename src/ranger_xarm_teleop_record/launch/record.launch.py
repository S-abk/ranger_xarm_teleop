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
Demonstration recording: the rosbag2 recorder and the episode node, for one session.

Start it with the stack, before arming (``record:=true`` on ``teleop_mock.launch.py`` or
``teleop_sim.launch.py`` includes it). The deployment profile names every topic, the clock
and the data directory; a session directory ``<data_dir>/<UTC time>-<suffix>`` is created
here. The recorder starts paused; the episode node resumes, pauses and splits it, one MCAP
file per episode. Nothing joins or leaves the ROS graph while the session runs.

Before the first episode, write the task label and operator into the data directory with
the ``episode_labels`` tool of ranger_xarm_teleop_record (it joins no ROS graph).
"""

import os
import secrets
import shlex
import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ranger_xarm_teleop_record import metadata as md
from ranger_xarm_teleop_record.profile import load_profile, recorded


def session_name() -> str:
    """Return a new session name: UTC time and a random suffix (no host, no user)."""
    return time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '-' + secrets.token_hex(2)


def recorder_command(profile: dict, bag_dir: str, extra=()) -> list:
    """Return the rosbag2 command line for a profile (paused, one file per episode)."""
    streams = recorded(profile)
    for name in extra:
        if name not in profile['streams']:
            raise RuntimeError(f'record_extra: no stream {name} in profile {profile["name"]}')
        streams[name] = profile['streams'][name]
    rec = profile['recorder']
    command = ['ros2', 'bag', 'record', '-o', bag_dir,
               '-s', rec['storage'], '--storage-preset-profile', rec['storage_preset'],
               '--max-cache-size', str(int(rec['max_cache_size_mb']) * 1024 * 1024),
               '--start-paused', '--node-name', rec['node']]
    if profile['use_sim_time']:
        command.append('--use-sim-time')
    return command + sorted({s['topic'] for s in streams.values()})


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    profile = load_profile(arg('profile'))
    data_dir = os.path.abspath(os.path.expanduser(arg('data_dir') or profile['data_dir']))
    problem = md.data_dir_problem(data_dir)
    if problem:
        raise RuntimeError(problem)
    session = os.path.join(data_dir, arg('session') or session_name())
    if os.path.exists(os.path.join(session, 'bag')):
        raise RuntimeError(f'{session} already holds a recording; choose another session')
    os.makedirs(session, exist_ok=True)
    extra = [s for s in arg('record_extra').split(',') if s]
    command = recorder_command(profile, os.path.join(session, 'bag'), extra)
    log = os.path.join(session, 'recorder.log')
    recorder = ExecuteProcess(
        cmd=['bash', '-c', 'exec ' + ' '.join(shlex.quote(c) for c in command)
             + ' > ' + shlex.quote(log) + ' 2>&1'],
        name='episode_recorder', output='screen', sigterm_timeout='10', sigkill_timeout='10')
    node = Node(
        package='ranger_xarm_teleop_record', executable='episode_node', output='screen',
        parameters=[{'profile': arg('profile'), 'session_dir': session,
                     'recorder': '/' + profile['recorder']['node'],
                     'limits_file': arg('limits_file'),
                     'stability_mass_file': arg('stability_mass_file'),
                     'tool_frame': arg('tool_frame'),
                     'use_sim_time': bool(profile['use_sim_time']),
                     'record_extra': ','.join(extra)}])
    return [recorder, node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('profile', default_value='mock',
                              description='Deployment profile: a name (mock, gazebo) or a path.'),
        DeclareLaunchArgument('data_dir', default_value='',
                              description="Where sessions go (default: the profile's; never "
                                          'inside a git checkout).'),
        DeclareLaunchArgument('session', default_value='',
                              description='Session directory name (default: UTC time + suffix).'),
        DeclareLaunchArgument('record_extra', default_value='',
                              description='Comma list of profile streams to record although '
                                          'the profile does not (for example depth_gantry).'),
        DeclareLaunchArgument('limits_file', default_value='',
                              description='The limits file in force, copied into the session.'),
        DeclareLaunchArgument('stability_mass_file', default_value='',
                              description="The stability monitor's mass file, copied into the "
                                          'session and fingerprinted in the sidecar (optional).'),
        DeclareLaunchArgument('tool_frame', default_value=''),
        OpaqueFunction(function=launch_setup),
    ])
