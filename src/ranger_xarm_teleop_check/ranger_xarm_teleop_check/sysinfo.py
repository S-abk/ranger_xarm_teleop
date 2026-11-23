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
What the check reads outside ROS: machine state, files, the robot description, the driver.

No ROS imports; every outside effect (git, sysfs paths, package lookup) is a parameter so the
tests can fake it.
"""

import glob
import math
import os
import subprocess
from typing import Callable, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ElementTree

from ranger_xarm_teleop_record.provenance import sha256_text
from ranger_xarm_teleop_servo.simcheck import hardware_plugins, SIMULATOR_PLUGINS
import yaml

# The xarm_ros2 commit "[fix] Modify the data source of joint_states and use feedback data"
# (2025-03-19): from it on, ros2_control reads the joint state from the arm every control
# cycle. Before it, the state was stair-stepped at 5 Hz under report_type normal.
DRIVER_FIX_COMMIT = 'cc44f9f'
DRIVER_SOURCE = os.path.join('xarm_controller', 'src', 'hardware',
                             'uf_robot_system_hardware.cpp')
DRIVER_MARKERS = ('20250318', 'get_joint_states(')


# ---- machine ----------------------------------------------------------------------------

def read_governors(root: str = '/sys/devices/system/cpu') -> Dict[str, str]:
    """Return {cpu name: scaling governor} for every cpu that has one (empty: none)."""
    out = {}
    pattern = os.path.join(root, 'cpu[0-9]*', 'cpufreq', 'scaling_governor')
    for path in sorted(glob.glob(pattern)):
        cpu = os.path.basename(os.path.dirname(os.path.dirname(path)))
        try:
            with open(path, encoding='utf-8') as handle:
                out[cpu] = handle.read().strip()
        except OSError:
            continue
    return out


def read_loadavg(path: str = '/proc/loadavg') -> Optional[Tuple[float, float, float]]:
    """Return the 1, 5 and 15 minute load averages, or None when unreadable."""
    try:
        with open(path, encoding='utf-8') as handle:
            parts = handle.read().split()
        return float(parts[0]), float(parts[1]), float(parts[2])
    except (OSError, ValueError, IndexError):
        return None


def machine_info(sys_root: str = '/sys/devices/system/cpu',
                 loadavg_path: str = '/proc/loadavg') -> dict:
    """Return the governors (by cpu), their set, the load averages and the core count."""
    governors = read_governors(sys_root)
    return {'governors': governors, 'governor_set': sorted(set(governors.values())),
            'loadavg': read_loadavg(loadavg_path), 'cores': os.cpu_count()}


# ---- package versions -------------------------------------------------------------------

def package_version(package_xml_text: str) -> Optional[str]:
    """Return the <version> of a package.xml text, or None."""
    try:
        node = ElementTree.fromstring(package_xml_text).find('version')
    except ElementTree.ParseError:
        return None
    return (node.text or '').strip() or None if node is not None else None


def package_versions(names, share_dir: Callable[[str], str]) -> Dict[str, Optional[str]]:
    """
    Return {package: version or None} read from each package.xml.

    ``share_dir(name)`` returns the package's share directory or raises when it is not
    installed (the node passes ament_index's lookup); a missing package gives None.
    """
    out = {}
    for name in names:
        try:
            with open(os.path.join(share_dir(name), 'package.xml'), encoding='utf-8') as fh:
                out[name] = package_version(fh.read())
        except Exception:  # noqa: BLE001 - any lookup failure means "not installed"
            out[name] = None
    return out


# ---- robot description ------------------------------------------------------------------

def description_info(text: str) -> dict:
    """
    Return what the robot description says that the procedure needs.

    Keys: sha256, plugins, simulator (True when every plugin is a simulator plugin, None
    when there is no ros2_control block), gripper (a joint xarm_drive_joint exists),
    velocity_control (the distinct values of the hardware param, lower-cased; empty when it
    is absent), error (None, or why the XML could not be read).
    """
    out = {'sha256': sha256_text(text), 'plugins': [], 'simulator': None, 'gripper': False,
           'velocity_control': [], 'error': None}
    try:
        out['plugins'] = hardware_plugins(text)
        root = ElementTree.fromstring(text)
    except Exception as exc:  # noqa: BLE001 - NotASimulator wraps the XML error
        out['error'] = str(exc)
        return out
    if out['plugins']:
        out['simulator'] = all(p in SIMULATOR_PLUGINS for p in out['plugins'])
    out['gripper'] = any(j.get('name') == 'xarm_drive_joint' for j in root.iter('joint'))
    values = []
    for block in root.iter('ros2_control'):
        for param in block.findall("./hardware/param[@name='velocity_control']"):
            value = (param.text or '').strip().lower()
            if value not in values:
                values.append(value)
    out['velocity_control'] = values
    return out


# ---- input files ------------------------------------------------------------------------

def _walk(node, key: str):
    """Yield every value stored under ``key`` at any depth of a nested mapping."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, key)


def load_yaml_file(path: str):
    """Return (data, problem): the parsed YAML of ``path``, or None and why not."""
    try:
        with open(os.path.expanduser(path), encoding='utf-8') as handle:
            return yaml.safe_load(handle), None
    except (OSError, yaml.YAMLError) as exc:
        return None, f'cannot read {path}: {exc}'


def is_number(value) -> bool:
    """Return True for a finite int or float (not a bool, not NaN)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def limit_parameter(data, key: str):
    """
    Return the value of ``key`` under any ``ros__parameters`` block at any nesting, or None.

    The first match in file order wins.
    """
    for block in _walk(data, 'ros__parameters'):
        for value in _walk({'ros__parameters': block}, key):
            return value
    return None


def joint_acceleration_limits(data) -> Dict[str, Optional[float]]:
    """
    Return {joint: max_acceleration or None} from a MoveIt joint_limits.yaml mapping.

    A joint gets a value only when ``has_acceleration_limits`` is true and
    ``max_acceleration`` is a number; otherwise None.
    """
    out: Dict[str, Optional[float]] = {}
    for block in _walk(data, 'joint_limits'):
        if not isinstance(block, dict):
            continue
        for joint, spec in block.items():
            if not isinstance(spec, dict):
                continue
            ok = spec.get('has_acceleration_limits') is True
            value = spec.get('max_acceleration')
            out[str(joint)] = float(value) if ok and is_number(value) else None
        break
    return out


# ---- the xarm_ros2 driver gate ----------------------------------------------------------

Runner = Callable[[List[str]], Tuple[int, str]]


def run_git(args: List[str]) -> Tuple[int, str]:
    """Run ``git`` with ``args``; return (exit code, stdout). 127 when git cannot run."""
    try:
        done = subprocess.run(['git', *args], capture_output=True, text=True, timeout=15.0,
                              check=False)
    except (OSError, subprocess.SubprocessError):
        return 127, ''
    return done.returncode, done.stdout


def driver_gate(workspace: Optional[str], runner: Runner = run_git) -> dict:
    """
    Check that the installed xarm_ros2 driver has the 2025-03-18 joint-state change.

    Returns {verdict ('PASS', 'FAIL', 'NOT JUDGED'), evidence, head, describe, reason}.
    Order: git ancestry of DRIVER_FIX_COMMIT; when git cannot answer (not a repository, the
    commit unknown: a shallow clone or a rebased fork) the source markers. A commit known
    but not an ancestor is a FAIL.
    """
    out = {'verdict': 'NOT JUDGED', 'evidence': None, 'head': None, 'describe': None,
           'reason': ''}
    if not workspace:
        out['reason'] = 'workspace not given: pass --workspace'
        return out
    repo = os.path.join(os.path.expanduser(workspace), 'src', 'xarm_ros2')
    if not os.path.isdir(repo):
        out['reason'] = f'no xarm_ros2 at {repo}: pass the right --workspace'
        return out
    rc, head = runner(['-C', repo, 'rev-parse', 'HEAD'])
    if rc == 0:
        out['head'] = head.strip()
        rc2, desc = runner(['-C', repo, 'describe', '--always'])
        out['describe'] = desc.strip() if rc2 == 0 else None
        rc3, _ = runner(['-C', repo, 'cat-file', '-e', f'{DRIVER_FIX_COMMIT}^{{commit}}'])
        if rc3 == 0:
            rc4, _ = runner(['-C', repo, 'merge-base', '--is-ancestor', DRIVER_FIX_COMMIT,
                             'HEAD'])
            if rc4 == 0:
                out.update(verdict='PASS', evidence='git: ' + DRIVER_FIX_COMMIT
                           + ' is an ancestor of HEAD')
                return out
            if rc4 == 1:
                out.update(verdict='FAIL', evidence='git: ' + DRIVER_FIX_COMMIT
                           + ' is not an ancestor of HEAD')
                return out
    source = os.path.join(repo, DRIVER_SOURCE)
    try:
        with open(source, encoding='utf-8', errors='replace') as handle:
            text = handle.read()
    except OSError as exc:
        out['reason'] = f'cannot read {source}: {exc}'
        return out
    found = [m for m in DRIVER_MARKERS if m in text]
    if len(found) == len(DRIVER_MARKERS):
        out.update(verdict='PASS', evidence='source markers 20250318 and get_joint_states( '
                   'found (git could not place ' + DRIVER_FIX_COMMIT + ')')
    else:
        out.update(verdict='FAIL', evidence='git could not place ' + DRIVER_FIX_COMMIT
                   + '; source markers missing: ' + ', '.join(
                       m for m in DRIVER_MARKERS if m not in found))
    return out


# ---- subscription QoS -------------------------------------------------------------------

def choose_qos(publisher_profiles, default_reliability: str, default_durability: str):
    """
    Return (reliability, durability) for a subscription that must match what is published.

    ``publisher_profiles`` is a list of (reliability, durability) strings of the topic's
    publishers ('reliable' or 'best_effort'; 'volatile' or 'transient_local'). A subscriber
    cannot be stricter than a publisher, so it takes the weakest of the publishers; with no
    publisher visible the defaults stand.
    """
    if not publisher_profiles:
        return default_reliability, default_durability
    reliability = ('best_effort' if any(r == 'best_effort' for r, _ in publisher_profiles)
                   else default_reliability)
    durability = ('volatile' if any(d == 'volatile' for _, d in publisher_profiles)
                  else default_durability)
    return reliability, durability


def servo_latency(path: str) -> Optional[float]:
    """Return ``max_expected_latency`` (s) of a MoveIt Servo yaml file, or None."""
    data, problem = load_yaml_file(path)
    if problem or not isinstance(data, dict):
        return None
    value = data.get('max_expected_latency')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else None
