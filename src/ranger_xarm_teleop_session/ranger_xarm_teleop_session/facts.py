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
Facts about this machine for the pre-flight: the GPU, adb, git, disk. No ROS imports.

Reads only, except ``adb_facts(run_reverse=True)``, which runs exactly
``adb reverse tcp:8443 tcp:8443`` once, and only when a device is listed and the mapping is
absent (the human's decision, 2026-11-21). Every command here runs without the session's mark
in its environment: an adb server it starts is never taken for a process of the session.
"""

import os
import shutil
import subprocess
from typing import Dict, List, Optional

from .procs import MARK_VAR

EGL_VENDOR_NVIDIA = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'   # sim.launch.py's criterion
ADB_PORT = 8443
ADB_REVERSE = ['adb', 'reverse', f'tcp:{ADB_PORT}', f'tcp:{ADB_PORT}']
# Set by automated runs (the stack tests): no adb command at all, so a test never reaches a
# headset that happens to be plugged in, nor starts an adb server.
NO_ADB_VAR = 'TELEOP_SESSION_NO_ADB'


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    env.pop(MARK_VAR, None)
    return env


def _run(argv: List[str], timeout: float = 10.0):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=_env(),
                              stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError) as err:
        return None, str(err)
    return done.returncode, (done.stdout or '') + (done.stderr or '')


def nvidia_found() -> bool:
    """Return True when an NVIDIA card is listed and its EGL vendor file is installed."""
    if not os.path.exists(EGL_VENDOR_NVIDIA) or shutil.which('nvidia-smi') is None:
        return False
    code, out = _run(['nvidia-smi', '-L'])
    return code == 0 and 'GPU' in out


def parse_adb_devices(text: str) -> List[str]:
    """Return the serials ``adb devices`` lists as ready (state ``device``)."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == 'device' and not line.startswith('List of'):
            out.append(parts[0])
    return out


def adb_facts(run_reverse: bool, port: int = ADB_PORT) -> dict:
    """
    Return what ``preflight.adb_item`` reads.

    With ``run_reverse``, run ``adb reverse tcp:8443 tcp:8443`` once when a device is listed
    and ``tcp:8443`` is not reversed yet; never any other adb command, never retried.
    """
    if os.environ.get(NO_ADB_VAR):
        return {'adb': None}
    if shutil.which('adb') is None:
        return {'adb': False}
    code, out = _run(['adb', 'devices'])
    devices = parse_adb_devices(out) if code == 0 else []
    reverse: List[str] = []
    if devices:
        code, out = _run(['adb', 'reverse', '--list'])
        reverse = out.splitlines() if code == 0 else []
    ran: Optional[dict] = None
    mapped = any(f'tcp:{ADB_PORT}' in line for line in reverse)
    if run_reverse and devices and port == ADB_PORT and not mapped:
        code, out = _run(ADB_REVERSE)
        ran = {'command': ' '.join(ADB_REVERSE), 'code': code, 'output': out}
        code, out = _run(['adb', 'reverse', '--list'])
        reverse = out.splitlines() if code == 0 else []
    return {'adb': True, 'devices': devices, 'reverse': reverse, 'ran': ran}


def free_gb(path: str) -> Optional[float]:
    """Return the free space (GB) of the nearest existing directory at or above ``path``."""
    p = os.path.abspath(os.path.expanduser(path))
    while p and not os.path.isdir(p):
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent
    try:
        return shutil.disk_usage(p).free / 1e9
    except OSError:
        return None


def robot_workspace() -> str:
    """Return where the episode node looks for the robot workspace's commit."""
    ws = os.environ.get('RANGER_XARM_WS')
    if ws:
        return ws
    for prefix in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep):
        share = os.path.join(prefix, 'share', 'ranger_xarm_description')
        if prefix and os.path.isdir(share):
            return share
    return ''


def feedback_js(checkout: str) -> str:
    """Return the path of the headset page's feedback.js this checkout serves."""
    return os.path.join(checkout, 'third_party', 'vr-teleop-kit', 'src', 'vr_teleop_kit',
                        'relay', 'web', 'feedback.js')
