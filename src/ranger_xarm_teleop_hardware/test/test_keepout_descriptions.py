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
The link keep-out model on the real robot descriptions (both stages), not a made-up robot.

- INV-2 of the finer cover: no pair kept without it is left out with it, at margins 0.025 to
  0.05, and no left-out pair would refuse the hardware start.
- A pair the model holds is covered finer, and nothing is held at work up to 0.045; at 0.05
  a pair is (the hardware start is refused there, naming the margin that frees it).
- At 0.04 no pair is inside the margin at the ready posture (the stage 4 posture), where the
  sphere covers held joints 1 and 2 before.
- The pruned search finds what a brute-force search finds; a generous bound on its time.
"""

import os
import subprocess
import sys
import time

from ament_index_python.packages import get_package_share_directory
import numpy as np
import pytest
from ranger_xarm_teleop_hardware.attach import description_args, tool_frame
from ranger_xarm_teleop_servo import keepout as ko

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                'ranger_xarm_teleop_servo', 'test', 'unit'))
from keepout_brute import assert_search_is_exhaustive  # noqa: E402, I100

BASE = 'xarm_link_base'
READY = np.array([0.0, -0.6, -0.7, 0.0, 1.3, 0.0])       # the stack tests' ready posture
MARGINS = (0.025, 0.03, 0.04, 0.05)
RESOLVE = ko.make_resolver(get_package_share_directory)


@pytest.fixture(scope='module', params=['execution', 'check'])
def stage(request):
    xacro = os.path.join(get_package_share_directory('ranger_xarm_description'), 'urdf',
                         'ranger_xarm.urdf.xacro')
    urdf = subprocess.run(['xacro', xacro] + list(description_args(request.param)),
                          capture_output=True, text=True, check=True).stdout
    return request.param, urdf, tool_frame(request.param)


def _models(stage, margin):
    _, urdf, tip = stage
    return (ko.build(urdf, RESOLVE, BASE, tip, margin),
            ko.build(urdf, RESOLVE, BASE, tip, margin, fine_cell_ratio=None))


@pytest.mark.parametrize('margin', MARGINS)
def test_the_finer_cover_leaves_no_kept_pair_out_and_refuses_only_above_0_045(stage, margin):
    model, before = _models(stage, margin)
    assert set(before.pairs) <= set(model.pairs)
    assert not any(ko.kept_by_smaller_margin(item) for item in model.excluded)
    assert model.refined and model.refined_by                 # a pair the model held
    if margin <= 0.045:
        assert model.held == []
    else:
        assert model.held
        assert all(item.free_up_to < margin for item in model.held)


def test_at_40_mm_no_pair_is_inside_the_margin_at_the_ready_posture(stage):
    _, urdf, tip = stage
    model = ko.build(urdf, RESOLVE, BASE, tip, 0.04, near_samples=0)
    extra = {name: 0.0 for name in model.extra_joints}
    assert model.pair_minima(READY, extra).min() > 0.04
    before = ko.build(urdf, RESOLVE, BASE, tip, 0.04, near_samples=0, fine_cell_ratio=None)
    assert before.pair_minima(READY, extra).min() < 0.04    # what held the arm


def test_the_search_is_exhaustive_and_takes_milliseconds(stage):
    _, urdf, tip = stage
    model = ko.build(urdf, RESOLVE, BASE, tip, 0.04, near_samples=0)
    extra = {name: 0.0 for name in model.extra_joints}
    rng = np.random.default_rng(1)
    postures = [READY + rng.normal(0.0, 0.4, 6) for _ in range(12)]
    for q in postures:
        assert_search_is_exhaustive(model, q, extra, (0.04, model.influence))
    took = []
    for q in [READY + rng.normal(0.0, 0.05, 6) for _ in range(200)]:
        started = time.perf_counter()
        model.evaluate(q, extra)
        took.append(time.perf_counter() - started)
    # Generous (about 1 ms here): a gross slow-down fails, the machine's load does not.
    assert np.median(took) < 0.010
