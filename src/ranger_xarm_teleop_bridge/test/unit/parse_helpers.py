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
"""Shared helpers for the parsing tests: valid frames and the fixture generator."""

import copy
import importlib.util
import os

FIXTURE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fixtures'))


def load_make_fixtures():
    """Import test/fixtures/make_fixtures.py by path."""
    path = os.path.join(FIXTURE_DIR, 'make_fixtures.py')
    spec = importlib.util.spec_from_file_location('make_fixtures', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def button(pressed=False, value=0.0):
    """Return a wire-format button."""
    return {'p': pressed, 't': pressed, 'v': value}


def hand(position=(0.25, 0.95, -0.45), orientation=(0.0, 0.0, 0.0, 1.0), emulated=False,
         buttons=None, axes=None):
    """Return a wire-format controller entry."""
    return {
        'position': list(position),
        'orientation': list(orientation),
        'emulated_position': emulated,
        'buttons': [button() for _ in range(7)] if buttons is None else buttons,
        'axes': [0.0, 0.0, 0.0, 0.0] if axes is None else axes,
    }


def valid_frame():
    """Return a fresh valid wire-format frame with both hands and a viewer."""
    return copy.deepcopy({
        'type': 'xr_frame',
        't_client': 1000.4,
        't_xr': 1000.0,
        'seq': 5,
        'reset_count': 0,
        'controllers': {
            'left': hand(position=(-0.25, 0.95, -0.45)),
            'right': hand(),
        },
        'viewer': {'position': [0.0, 1.6, 0.0], 'orientation': [0.0, 0.0, 0.0, 1.0],
                   'emulated_position': False},
    })
