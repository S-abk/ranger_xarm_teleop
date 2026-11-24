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
Deployment profiles and checker tolerances (no ROS).

A profile names every topic, its kind, whether it is required, its nominal rate, the clock
its stamps are on, and the set-up's data directory and time sources. Nothing else in this
package names a topic of a particular simulator or machine. The profile in force is copied
into every session and episode, so an episode is judged by the profile it was recorded
under, not by the repo's current files.
"""

import copy
import os
from typing import Dict, List, Optional

from ranger_xarm_teleop_record.streams import OPTIONAL_FIELDS
import yaml

KINDS = ('teleop_state', 'pose', 'gripper_command', 'joint_command', 'joint_states', 'image',
         'camera_info', 'input_status', 'joy', 'episode_state', 'odometry', 'twist', 'owner',
         'twist_stamped', 'base_state', 'other', 'stability_state')
CLOCKS = ('robot', 'host_wall', 'none')
GATES = (None, 'engaged', 'gripper_active')
# Streams every profile must define: the action timeline, the action, the robot state and
# the episode state.
CORE = {'state': 'teleop_state', 'target': 'pose', 'joint_states': 'joint_states',
        'joint_command': 'joint_command', 'episode_state': 'episode_state'}


class ProfileError(ValueError):
    """A profile or tolerance file that cannot be used."""


def package_dir() -> str:
    """Return this package's source or install share root holding profiles/ and config/."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.dirname(here), here):
        if os.path.isdir(os.path.join(candidate, 'profiles')):
            return candidate
    try:                                       # installed: share/<package>
        from ament_index_python.packages import get_package_share_directory  # noqa: I001
        return get_package_share_directory('ranger_xarm_teleop_record')
    except Exception:                          # pragma: no cover - only without the source tree
        return os.path.dirname(here)


def resolve(name_or_path: str, folder: str = 'profiles') -> str:
    """Return the file for a profile name (``gazebo``) or a path."""
    if os.path.sep in name_or_path or name_or_path.endswith('.yaml'):
        return os.path.abspath(os.path.expanduser(name_or_path))
    return os.path.join(package_dir(), folder, f'{name_or_path}.yaml')


def load_yaml(path: str) -> dict:
    """Read one YAML mapping."""
    with open(path, encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ProfileError(f'{path}: not a mapping')
    return data


def validate(profile: dict) -> dict:
    """Return a normalised copy of ``profile``; raise ProfileError naming every problem."""
    p = copy.deepcopy(profile)
    problems: List[str] = []
    for key in ('name', 'use_sim_time', 'data_dir', 'frames', 'recorder', 'disk', 'streams'):
        if key not in p:
            problems.append(f'missing {key}')
    streams = p.get('streams') or {}
    if not isinstance(streams, dict) or not streams:
        problems.append('streams: empty')
        streams = {}
    topics: Dict[str, str] = {}
    for name, s in streams.items():
        if not isinstance(s, dict):
            problems.append(f'{name}: not a mapping')
            continue
        s.setdefault('record', True)
        s.setdefault('required', False)
        s.setdefault('gate', None)
        s.setdefault('count', False)
        s.setdefault('base', False)
        for key in ('topic', 'kind', 'type', 'clock'):
            if key not in s:
                problems.append(f'{name}: missing {key}')
        if s.get('kind') not in KINDS:
            problems.append(f'{name}: unknown kind {s.get("kind")!r}')
        if s.get('clock') not in CLOCKS:
            problems.append(f'{name}: unknown clock {s.get("clock")!r}')
        if s.get('gate') not in GATES:
            problems.append(f'{name}: unknown gate {s.get("gate")!r}')
        rate = s.get('rate_hz')
        if rate is not None and not (isinstance(rate, (int, float)) and rate > 0):
            problems.append(f'{name}: rate_hz must be positive or null')
        if s.get('required') and not s.get('record'):
            problems.append(f'{name}: required but not recorded')
        if s.get('kind') == 'image' and 'image' not in s:
            problems.append(f'{name}: an image stream needs image: {{width, height, encoding}}')
        fields = s.setdefault('required_fields', [])
        if not isinstance(fields, list):
            problems.append(f'{name}: required_fields must be a list')
        else:
            known = OPTIONAL_FIELDS.get(s.get('kind'), ())
            for f in fields:
                if f not in known:
                    problems.append(f'{name}: required field {f!r} is not one of {list(known)} '
                                    f'(kind {s.get("kind")})')
        if s.get('info') is not None and s['info'] not in streams:
            problems.append(f'{name}: info stream {s["info"]!r} not in the profile')
        topic = s.get('topic')
        if topic in topics:
            problems.append(f'{name}: topic {topic} also used by {topics[topic]}')
        topics[topic] = name
    for name, kind in CORE.items():
        if name not in streams:
            problems.append(f'core stream {name} missing')
        elif streams[name].get('kind') != kind:
            problems.append(f'core stream {name} must be of kind {kind}')
        elif not streams[name].get('required'):
            problems.append(f'core stream {name} must be required')
    if streams.get('state', {}).get('rate_hz') is None:
        problems.append('state: the action timeline needs a nominal rate')
    if p.get('low_rtf') is not None and not 0.0 < float(p['low_rtf']) <= 1.0:
        problems.append('low_rtf must be in (0, 1] or null')
    overrides = p.setdefault('checks', {}) or {}
    p['checks'] = overrides
    if not isinstance(overrides, dict) or not all(isinstance(v, dict)
                                                  for v in overrides.values()):
        problems.append('checks: must map a section of checks.yaml to {key: value}')
        overrides = {}
    provisional = p.setdefault('provisional', {}) or {}
    p['provisional'] = provisional
    if not isinstance(provisional, dict):
        problems.append('provisional: must map section.key to why the value is provisional')
    else:
        for dotted in provisional:
            section, _, key = str(dotted).partition('.')
            if key not in (overrides.get(section) or {}):
                problems.append(f'provisional: {dotted} is not a tolerance this profile sets')
    if problems:
        raise ProfileError('; '.join(problems))
    return p


def load_profile(name_or_path: str) -> dict:
    """Load and validate a profile by name or path."""
    path = resolve(name_or_path)
    return validate(load_yaml(path))


def load_checks(path: Optional[str] = None) -> dict:
    """Load the checker tolerances (the package default when ``path`` is None)."""
    return load_yaml(path or os.path.join(package_dir(), 'config', 'checks.yaml'))


# What a profile may override: the fitted consistency tolerances only. Everything else
# (fail flags, gaps, offsets, the lag range) stays the tolerance file's, so a profile carried
# in a sidecar cannot switch a failure off.
PROFILE_TOLERANCES = {
    'consistency': ('arm_rms_m', 'arm_p99_m', 'arm_rot_p99_deg', 'gripper_abs_rad'),
}


def effective_checks(checks: dict, profile: dict, source: str) -> dict:
    """
    Return the tolerances in force for ``profile``: ``checks`` with the profile's overrides.

    A deployment profile may override the fitted consistency tolerances (PROFILE_TOLERANCES)
    in its ``checks`` block and mark some of them ``provisional`` (set by hand before enough
    data to fit them). ``source`` says which copy of the profile the overrides came from; it
    is reported with every check under ``_tolerances``. An override of anything else is an
    error, so neither a typing mistake nor a profile carried in a sidecar can change another
    tolerance or switch a failure off.
    """
    out = copy.deepcopy({k: v for k, v in checks.items() if k != '_tolerances'})
    overrides = profile.get('checks') or {}
    for section, values in overrides.items():
        if section not in PROFILE_TOLERANCES or not isinstance(out.get(section), dict):
            raise ProfileError(f'profile {profile.get("name")}: checks.{section} cannot be '
                               f'set by a profile (only {sorted(PROFILE_TOLERANCES)})')
        for key, value in values.items():
            if key not in PROFILE_TOLERANCES[section] or key not in out[section]:
                raise ProfileError(f'profile {profile.get("name")}: checks.{section}.{key} '
                                   'cannot be set by a profile (only '
                                   f'{list(PROFILE_TOLERANCES[section])})')
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ProfileError(f'profile {profile.get("name")}: checks.{section}.{key} '
                                   'must be a positive number')
            out[section][key] = value
    out['_tolerances'] = {'profile': profile.get('name'), 'source': source,
                          'overrides': copy.deepcopy(overrides),
                          'provisional': dict(profile.get('provisional') or {})}
    return out


def package_profile(name: str) -> Optional[dict]:
    """Return the package's profile called ``name``, or None when there is none."""
    path = resolve(name) if name and os.path.sep not in name else None
    if path is None or not os.path.isfile(path):
        return None
    return load_profile(path)


def recorded(profile: dict) -> Dict[str, dict]:
    """Return the streams the recorder records, by name."""
    return {k: s for k, s in profile['streams'].items() if s.get('record', True)}


def required(profile: dict) -> Dict[str, dict]:
    """Return the required streams, by name."""
    return {k: s for k, s in profile['streams'].items() if s.get('required')}


def period(stream: dict) -> Optional[float]:
    """Return a stream's nominal period, or None for an event stream."""
    rate = stream.get('rate_hz')
    return None if rate is None else 1.0 / float(rate)
