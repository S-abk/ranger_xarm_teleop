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
"""Package setup for ranger_xarm_teleop_record."""

from glob import glob

from setuptools import find_packages, setup

package_name = 'ranger_xarm_teleop_record'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'test.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/profiles', glob('profiles/*.yaml')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shuaib Olanrewaju',
    maintainer_email='solanrewaju2020@fau.edu',
    description='Demonstration episodes: episode control, recorder, sidecar, checker, export.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'episode_node = ranger_xarm_teleop_record.episode_node:main',
            'episode_labels = ranger_xarm_teleop_record.cli:labels_main',
            'episode_check = ranger_xarm_teleop_record.cli:check_main',
            'episode_export = ranger_xarm_teleop_record.cli:export_main',
            'episode_purge = ranger_xarm_teleop_record.cli:purge_main',
            'record_commission = ranger_xarm_teleop_record.commission:main',
        ],
    },
)
