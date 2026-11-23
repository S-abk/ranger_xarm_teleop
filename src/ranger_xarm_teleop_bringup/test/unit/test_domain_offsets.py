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
Every stack test module has a domain offset of its own (stack_helpers.child_environment).

The stack tests run on several workers, one module per worker: two modules with one offset
can run at once on one ROS domain and see each other's nodes. The offsets are read from the
module sources: literal ``offset=`` arguments and module-level ``*OFFSET`` constants.
"""

import ast
import os

import pytest

TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Modules that compute environments but start nothing on a domain.
STARTS_NOTHING = {'test_stack_helpers.py'}


def offsets_of(source: str) -> set:
    """Return the domain offsets a test module's source names."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (kw.arg == 'offset' and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, int)):
                    found.add(kw.value.value)
            name = getattr(node.func, 'id', getattr(node.func, 'attr', ''))
            if (name == 'child_environment' and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, int)):
                found.add(node.args[0].value)       # child_environment(7)
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, int):
            if any(isinstance(t, ast.Name) and t.id.endswith('OFFSET') for t in node.targets):
                found.add(node.value.value)
    return found


def makes_environments(source: str) -> bool:
    """Return whether a test module builds a child environment (a ROS domain of its own)."""
    return 'child_environment(' in source or 'import environment' in source \
        or ' environment,' in source or 'environment(tmp' in source


def stack_modules() -> dict:
    """Return {module file: offsets} for every test module that builds an environment."""
    out = {}
    for name in sorted(os.listdir(TEST_DIR)):
        if not (name.startswith('test_') and name.endswith('.py')) or name in STARTS_NOTHING:
            continue
        with open(os.path.join(TEST_DIR, name), encoding='utf-8') as handle:
            source = handle.read()
        if makes_environments(source):
            out[name] = offsets_of(source)
    return out


def test_every_stack_test_module_names_its_offset():
    missing = [name for name, offsets in stack_modules().items() if not offsets]
    assert not missing, f'modules that build an environment without a literal offset: {missing}'


def test_no_two_stack_test_modules_share_a_domain_offset():
    modules = stack_modules()
    assert len(modules) >= 24, sorted(modules)          # the reader still finds them all
    owners = {}
    for name, offsets in modules.items():
        for offset in offsets:
            owners.setdefault(offset, []).append(name)
    shared = {offset: names for offset, names in owners.items() if len(names) > 1}
    assert not shared, f'domain offsets used by more than one module: {shared}'


def test_offsets_stay_in_the_band():
    for name, offsets in stack_modules().items():
        assert all(1 <= offset < 90 for offset in offsets), (name, offsets)


@pytest.mark.parametrize('source, expected', [
    ('env = child_environment(offset=7)', {7}),
    ('OFFSET = 19\ndef f(offset=OFFSET):\n    return child_environment(offset=offset)', {19}),
    ('def g():\n    return environment(tmp, offset=OFFSET)', set()),
    ('X = 3\nenv = child_environment(offset=4)  # and 5\nenv = child_environment(offset=5)',
     {4, 5}),
    ('env = stack_helpers.child_environment(7)', {7}),
])
def test_the_reader_finds_literal_offsets_and_offset_constants(source, expected):
    assert offsets_of(source) == expected
