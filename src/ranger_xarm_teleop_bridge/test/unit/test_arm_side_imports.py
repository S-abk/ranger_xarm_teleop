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
No node that runs beside the arm needs the bridge's WebSocket library.

The arm-side nodes read the headset's topics by name; they take the names from ``topics``,
not from ``bridge_node`` (which imports ``websockets``). A machine without that library must
still run them. Checked on the sources: every module each node's entry point imports, through
this repository's packages, is read, and no import of ``websockets`` may be among them.
"""

import ast
import os

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PACKAGES = sorted(name for name in os.listdir(SRC) if name.startswith('ranger_xarm_teleop_')
                  and os.path.isdir(os.path.join(SRC, name, name)))
# The headset side: the bridge itself and its replay of recordings, which use the relay.
HEADSET_SIDE = {'ranger_xarm_teleop_bridge.bridge_node', 'ranger_xarm_teleop_bridge.replay_node'}
FORBIDDEN = 'websockets'


def module_path(module: str):
    """Return the source file of a module of this repository, or None for any other."""
    parts = module.split('.')
    if parts[0] not in PACKAGES:
        return None
    base = os.path.join(SRC, parts[0], *parts)
    for path in (base + '.py', os.path.join(base, '__init__.py')):
        if os.path.isfile(path):
            return path
    return None


def imports_of(path: str, module: str) -> set:
    """
    Return the modules a source file imports (``from a import b`` gives a and a.b).

    ``module`` is the file's own module name, against which relative imports resolve.
    """
    with open(path, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    package = module.split('.') if path.endswith('__init__.py') else module.split('.')[:-1]
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            source = '.'.join(base + ([node.module] if node.module else []))
            if source:
                found.add(source)
                found.update(f'{source}.{alias.name}' for alias in node.names)
    return found


def closure(module: str) -> set:
    """Return every module imported from ``module``, following this repository's packages."""
    seen, todo, outside = set(), [module], set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        parts = name.split('.')
        for i in range(1, len(parts)):          # a package's __init__ runs first
            todo.append('.'.join(parts[:i]))
        path = module_path(name)
        if path is None:
            if parts[0] not in PACKAGES:
                outside.add(name)
            continue
        todo.extend(imports_of(path, name))
    return outside


def entry_points() -> list:
    """Return (package, module) of every console script of every package."""
    out = []
    for package in PACKAGES:
        setup = os.path.join(SRC, package, 'setup.py')
        if not os.path.isfile(setup):
            continue
        with open(setup, encoding='utf-8') as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and ' = ' in node.value and ':' in node.value:
                target = node.value.split(' = ', 1)[1].split(':', 1)[0].strip()
                if target.startswith(package + '.'):
                    out.append((package, target))
    return out


ARM_SIDE = sorted({module for _, module in entry_points() if module not in HEADSET_SIDE})


def test_the_entry_points_are_found():
    names = set(ARM_SIDE)
    assert 'ranger_xarm_teleop_servo.teleop_node' in names
    assert 'ranger_xarm_teleop_servo.reset_node' in names
    assert 'ranger_xarm_teleop_bridge.feedback_node' in names
    assert len(names) >= 10, names


@pytest.mark.parametrize('module', ARM_SIDE)
def test_an_arm_side_node_does_not_import_the_websocket_library(module):
    outside = closure(module)
    assert not {name for name in outside if name.split('.')[0] == FORBIDDEN}, module


def test_relative_imports_are_followed(tmp_path):
    source = tmp_path / 'm.py'
    source.write_text('from . import a\nfrom .b import c\nfrom ..d import e\n')
    expected = {'pkg.sub', 'pkg.sub.a', 'pkg.sub.b', 'pkg.sub.b.c', 'pkg.d', 'pkg.d.e'}
    assert imports_of(str(source), 'pkg.sub.m') == expected
    init = tmp_path / '__init__.py'
    init.write_text('from .x import y\n')
    assert imports_of(str(init), 'pkg.sub') == {'pkg.sub.x', 'pkg.sub.x.y'}


def test_the_bridge_itself_is_seen_to_import_it():
    """The check reads what it should: the bridge node's closure does contain it."""
    for module in sorted(HEADSET_SIDE):     # replay_node through bridge_node
        assert FORBIDDEN in {name.split('.')[0] for name in closure(module)}, module
