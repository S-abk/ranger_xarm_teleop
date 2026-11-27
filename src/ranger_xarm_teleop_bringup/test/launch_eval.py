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
Evaluate a launch file as ``ros2 launch`` visits it, without starting anything.

Every Node this stack's own launch files start is recorded with its package, executable, name,
arguments, remappings and parameters; every include that is not skipped by its condition is
recorded with its file and launch arguments. Nodes that the robot workspace's launch files start
(move_group, fake_execution's nodes) are not recorded: only the arguments this stack passes to
those launch files are, so a change in the workspace does not change the result.

The result is a recipe, not the expanded text: a ``Command`` (xacro) is recorded as its command
line and never run, a semantic description by the file it equals, and installation paths as
``<share:package>``, so it is the same on every machine. Temporary file names (the controller
parameter files teleop_core writes) and the record session folder are normalised.

    python3 launch_eval.py --write FIXTURE.jsonl   # (re)write the reference (test_launch_args.py)
"""

import hashlib
import json
import os
import re
import sys
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import ExecuteLocal, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import Command
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

OWN_PREFIX = 'ranger_xarm_teleop'
SRDF_NAMES = ('ranger_xarm.srdf', 'ranger_xarm_mobile_base.srdf',
              'ranger_xarm_mobile_base_wheels.srdf')
SHARE_RE = re.compile(r"/[^\s'\"]*?/share/([A-Za-z0-9_]+)(?=[/\s'\"]|$)")
TMP_YAML_RE = re.compile(re.escape(tempfile.gettempdir())
                         + r'/(teleop_[a-z_]+?)_[A-Za-z0-9_]{8}\.yaml')
SESSION_RE = re.compile(r'<data_dir>/\d{8}-\d{6}-[0-9a-f]{4}')

# The cases the reference holds: (case name, launch file, arguments). record_data_dir is set by
# evaluate() (the record launch creates its session folder when evaluated).
CASES = [
    ('sim', 'teleop_sim.launch.py', []),
    ('sim_all', 'teleop_sim.launch.py', ['reset:=true', 'record:=true', 'stability:=true']),
    ('mock', 'teleop_mock.launch.py', []),
    ('mock_all', 'teleop_mock.launch.py', ['reset:=true', 'record:=true', 'stability:=true']),
    ('mock_base', 'teleop_mock.launch.py', ['base:=true']),
    ('mock_base_all', 'teleop_mock.launch.py',
     ['base:=true', 'reset:=true', 'record:=true', 'stability:=true']),
]
# Also in the reference, compared with the changes for a driven base taken out
# (test_launch_args.py).
BASE_CASES = [
    ('sim_base', 'teleop_sim.launch.py', ['base:=true']),
    ('sim_base_all', 'teleop_sim.launch.py',
     ['base:=true', 'reset:=true', 'record:=true', 'stability:=true']),
]


def _srdf_texts():
    config = os.path.join(get_package_share_directory('ranger_xarm_moveit_config'), 'config')
    out = {}
    for name in SRDF_NAMES:
        path = os.path.join(config, name)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as handle:
                out[handle.read()] = name
    return out


def _subs(context, value):
    if value is None or isinstance(value, str):
        return value
    return perform_substitutions(context, normalize_to_list_of_substitutions(value))


def _environment(context, description):
    """Return a process's env and additional_env, evaluated ({} for none)."""
    out = {}
    for kind in ('env', 'additional_env'):
        pairs = getattr(description, kind) or []
        if pairs:
            out[kind] = {_subs(context, k): _subs(context, v) for k, v in pairs}
    return out


def _share_package(path):
    match = SHARE_RE.search(path or '')
    return match.group(1) if match else ''


class _Recipe:
    """Normalise evaluated values into recipes."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.srdfs = _srdf_texts()

    def __call__(self, value):
        if isinstance(value, str):
            if value in self.srdfs:
                return {'srdf_file': self.srdfs[value]}
            if TMP_YAML_RE.fullmatch(value) and os.path.exists(value):
                with open(value, encoding='utf-8') as handle:     # its content, not its name
                    return {'temporary_file': TMP_YAML_RE.sub(r'<tmp>/\1_<tmp>.yaml', value),
                            'content': self(handle.read())}
            value = value.replace(self.data_dir, '<data_dir>')
            value = SESSION_RE.sub('<data_dir>/<session>', value)
            value = TMP_YAML_RE.sub(r'<tmp>/\1_<tmp>.yaml', value)
            value = SHARE_RE.sub(lambda m: f'<share:{m.group(1)}>', value)
            if len(value) > 2000:
                return {'sha256': hashlib.sha256(value.encode()).hexdigest(), 'len': len(value)}
            return value
        if isinstance(value, (list, tuple)):
            return [self(v) for v in value]
        if isinstance(value, dict):
            return {k: self(v) for k, v in value.items()}
        return value


def evaluate(launch_file, args, data_dir):
    """Return {'nodes': [...], 'includes': [...]} of ``launch_file`` with ``args`` (a:=b)."""
    recipe = _Recipe(data_dir)
    argv = list(args) + [f'record_data_dir:={data_dir}']
    context = LaunchContext(argv=argv)
    for item in argv:
        name, value = item.split(':=', 1)
        context.launch_configurations[name] = value
    nodes, includes = [], []
    original = Command.perform
    Command.perform = lambda self, ctx: 'COMMAND ' + perform_substitutions(ctx, self.command)

    def visit(entity, origin):
        if isinstance(entity, LaunchDescription):
            for child in entity.entities:
                visit(child, origin)
            return
        condition = getattr(entity, 'condition', None)
        if condition is not None and not condition.evaluate(context):
            return
        if isinstance(entity, TimerAction):
            for child in entity.actions:
                visit(child, origin)
            return
        if isinstance(entity, Node):
            if origin.startswith(OWN_PREFIX):
                params = evaluate_parameters(context, entity._Node__parameters or [])
                nodes.append(recipe({
                    'package': _subs(context, entity.node_package),
                    'executable': _subs(context, entity.node_executable),
                    'name': _subs(context, entity._Node__node_name),
                    'namespace': _subs(context, entity._Node__node_namespace),
                    'arguments': [_subs(context, a) for a in (entity._Node__arguments or [])],
                    'ros_arguments': [_subs(context, a)
                                      for a in (entity._Node__ros_arguments or [])],
                    'environment': _environment(context, entity.process_description),
                    'remappings': [[_subs(context, a), _subs(context, b)]
                                   for a, b in (entity._Node__remappings or [])],
                    'parameters': [p if isinstance(p, dict) else {'file': str(p)}
                                   for p in params],
                    'origin': origin}))
            return
        if isinstance(entity, ExecuteLocal):        # a process other than a Node: never run
            if origin.startswith(OWN_PREFIX):
                nodes.append(recipe({'process': [_subs(context, c)
                                                 for c in entity.process_description.cmd],
                                     'environment': _environment(context,
                                                                 entity.process_description),
                                     'origin': origin}))
            return
        if isinstance(entity, IncludeLaunchDescription):
            arguments = [[_subs(context, k), _subs(context, v)]
                         for k, v in entity.launch_arguments]
            children = entity.execute(context) or []
            location = entity.launch_description_source.location
            includes.append(recipe({'file': location, 'launch_arguments': arguments,
                                    'from': origin}))
            for child in children:
                visit(child, _share_package(location))
            return
        for child in entity.visit(context) or []:
            visit(child, origin)

    try:
        source = AnyLaunchDescriptionSource(launch_file)
        visit(source.get_launch_description(context), _share_package(launch_file))
    finally:
        Command.perform = original
    return {'nodes': nodes, 'includes': includes}


def launch_path(name):
    """Return the installed path of one of this package's launch files."""
    return os.path.join(get_package_share_directory('ranger_xarm_teleop_bringup'), 'launch',
                        name)


def reference_lines():
    """Evaluate every case; return the reference lines (JSONL)."""
    lines = []
    with tempfile.TemporaryDirectory(prefix='launch_eval_') as data_dir:
        for case, name, args in CASES + BASE_CASES:
            result = evaluate(launch_path(name), args, data_dir)
            lines.append(json.dumps({'case': case, 'launch': name, 'args': args, **result},
                                    sort_keys=True))
    return lines


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--write':
        raise SystemExit(__doc__)
    with open(sys.argv[2], 'w', encoding='utf-8', newline='\n') as out:
        out.write('\n'.join(reference_lines()) + '\n')
