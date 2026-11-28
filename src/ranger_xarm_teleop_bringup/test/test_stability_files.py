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
The stability monitor's shipped files against the simulator's wheeled description (INV-5).

The simulation-only mass file corrects no mass and bounds every body; the template is refused;
the simulation settings name links of the description; the core launch file names no
simulation-only file (the wrappers pass them).
"""

import os
import subprocess
import xml.etree.ElementTree as ElementTree

from ament_index_python.packages import get_package_share_directory
import pytest
from ranger_xarm_teleop_servo import stability_mass as sm
from ranger_xarm_teleop_servo.stability_model import RobotModel
import yaml

SHARE = get_package_share_directory('ranger_xarm_teleop_bringup')


def config(name):
    with open(os.path.join(SHARE, 'config', name), encoding='utf-8') as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope='module')
def wheeled_urdf():
    xacro = os.path.join(get_package_share_directory('ranger_xarm_description'), 'urdf',
                         'ranger_xarm.urdf.xacro')
    args = ['xarm_ros2_control_plugin:=gz_ros2_control/GazeboSimSystem',
            'xarm_load_gazebo_plugin:=true', 'xarm_ros2_control_params:=/dev/null',
            'add_gripper:=true', 'use_wheels:=true', 'fix_base_to_world:=false',
            'gz_sensors:=true', 'wheels_command_interface:=velocity']
    return subprocess.run(['xacro', xacro] + args, capture_output=True, text=True,
                          check=True).stdout


@pytest.fixture(scope='module')
def wheeled(wheeled_urdf):
    params = config('stability_SIMULATION_ONLY.yaml')['/**']['ros__parameters']
    return RobotModel(wheeled_urdf, params['base_frame'], params['wheel_links'])


def test_the_simulation_mass_file_corrects_no_mass_and_bounds_every_body(wheeled, wheeled_urdf):
    mm = sm.parse(config('stability_mass_SIMULATION_ONLY.yaml'), wheeled, simulation=True)
    for b, original in zip(mm.bodies, wheeled.bodies):
        assert b.body.mass == original.mass and (b.body.com == original.com).all()
        assert b.mass.plus > 0 and b.mass.minus > 0 and b.com_bound > 0
    assert len(mm.bodies) == len(wheeled.bodies)
    assert mm.stiffness.value == 20000.0 and mm.stiffness.low == pytest.approx(8000.0)
    # The total is the description's own: every link's inertial mass, read here.
    links = ElementTree.fromstring(wheeled_urdf).findall('link')
    total = sum(float(link.find('inertial/mass').get('value')) for link in links
                if link.find('inertial/mass') is not None)
    assert sm.summary(mm)['mass_kg'] == pytest.approx(total, abs=0.01)


def test_the_simulation_mass_file_is_refused_outside_simulation(wheeled):
    with pytest.raises(sm.MassFileError, match='simulation_only'):
        sm.parse(config('stability_mass_SIMULATION_ONLY.yaml'), wheeled, simulation=False)


def test_the_template_is_refused(wheeled):
    with pytest.raises(sm.MassFileError, match='template'):
        sm.parse(config('stability_mass_TEMPLATE.yaml'), wheeled, simulation=True)


def test_the_core_launch_file_names_no_simulation_only_stability_file():
    with open(os.path.join(SHARE, 'launch', 'teleop_core.launch.py'), encoding='utf-8') as f:
        core = f.read()
    assert "'stability', default_value='false'" in core
    assert 'stability_SIMULATION_ONLY' not in core and 'stability_mass_SIMULATION' not in core
    for wrapper in ('teleop_mock.launch.py', 'teleop_sim.launch.py'):
        with open(os.path.join(SHARE, 'launch', wrapper), encoding='utf-8') as f:
            text = f.read()
        assert 'stability_SIMULATION_ONLY.yaml' in text
        assert 'stability_mass_SIMULATION_ONLY.yaml' in text
