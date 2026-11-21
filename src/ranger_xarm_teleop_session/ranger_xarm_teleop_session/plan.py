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
What a session starts, in which order, and how it stops. No ROS imports.

The launcher adds no launch file of its own: it runs the manual sequence's commands (README,
"Run the headset input bridge" and the teleop section) as separate processes, with the same
arguments. ``build_plan`` returns them for a profile, a role and the options; nothing here
starts anything.

Profiles. ``PROFILES`` lists the deployments the launcher brings up. Both are simulation
profiles; the stack's own launch files keep every refusal (the simulation-only limits file
outside simulation, a missing required value). A hardware profile is not built: there the
human's bring-up runs first and the launcher would only check it (report, "hardware"). A
profile whose deployment profile is not marked ``simulation: true`` is refused here, so the
reset (simulation only) can never be switched on for hardware by this launcher.
"""

from dataclasses import dataclass, field
import os
import re
import secrets
import time
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8443
PROBE_NODE = 'teleop_session_probe'
LIMITS_FILE = os.path.join('config', 'SIMULATION_ONLY_limits.yaml')   # in the bringup share
SIM_CHECK_SECONDS = 30.0

# Long-running steps stop in this order: the input first (the arm then holds on stale input,
# G-3), then the teleop stack with the recorder (the episode node marks an episode in progress
# interrupted), the simulator, and the relay last.
STOP_ORDER = ('probe', 'sim_check', 'bridge', 'stack', 'simulator', 'relay')
ROLES = ('stack', 'relay')
GPU_CHOICES = ('auto', 'nvidia', 'default')


@dataclass(frozen=True)
class Profile:
    """A deployment the launcher brings up."""

    name: str
    record_profile: str                        # the deployment profile (record package)
    stack: Tuple[str, str]                     # (package, launch file) of the teleop stack
    simulator: Optional[Tuple[str, str]] = None  # started first, when the launcher owns it
    rviz: bool = False                         # the stack launch takes ``rviz``


PROFILES: Dict[str, Profile] = {
    'mock': Profile('mock', 'mock', ('ranger_xarm_teleop_bringup', 'teleop_mock.launch.py'),
                    rviz=True),
    'gazebo': Profile('gazebo', 'gazebo',
                      ('ranger_xarm_teleop_bringup', 'teleop_sim.launch.py'),
                      simulator=('ranger_xarm_gazebo', 'sim.launch.py')),
}


@dataclass
class Options:
    """What the operator chose (defaults suit a person, not an automated run)."""

    profile: str = 'mock'
    role: str = 'stack'
    record: bool = True
    feedback: bool = True
    reset: bool = True
    headless: bool = False
    cameras: bool = True
    gpu: str = 'auto'
    rviz: bool = True
    relay_url: Optional[str] = None            # a relay elsewhere: none is started here
    listen: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    data_dir: Optional[str] = None
    sim_check_seconds: float = SIM_CHECK_SECONDS
    explicit: Tuple[str, ...] = field(default_factory=tuple)   # options typed by the operator


@dataclass(frozen=True)
class Step:
    """One process of the session."""

    name: str                  # one of STOP_ORDER
    argv: Tuple[str, ...]
    runs_to_end: bool = False  # a check that exits by itself (the simulator check, the probe)

    @property
    def stop_rank(self) -> int:
        """Return the position of this step in the stop order."""
        return STOP_ORDER.index(self.name)


class PlanError(ValueError):
    """The options cannot make a session; the message says why and what to do."""


def session_name(now: Optional[float] = None, token: Optional[str] = None) -> str:
    """Return a session name: UTC time and a random suffix, as record.launch.py makes it."""
    stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime(time.time() if now is None else now))
    return stamp + '-' + (token if token is not None else secrets.token_hex(2))


def is_loopback(host: str) -> bool:
    """Return True for an address only this machine can reach."""
    return host in ('localhost', '::1') or host.startswith('127.')


def listen_warning(opts: 'Options') -> Optional[str]:
    """Return the line said when the relay started here listens beyond this machine, or None."""
    if (opts.role == 'relay' or not opts.relay_url) and not is_loopback(opts.listen):
        return (f'the relay listens on {opts.listen}: it has no authentication, so anything on '
                'that network can connect to it')
    return None


def local_relay_url(port: int, listen: str = DEFAULT_HOST) -> str:
    """Return the bridge's URL for a relay on this machine, listening on ``listen``."""
    host = '127.0.0.1' if listen in ('0.0.0.0', '::') else listen
    return f'ws://{host}:{port}/ws'


def relay_url_problem(url: str) -> Optional[str]:
    """Return why ``url`` is not a relay WebSocket URL, or None."""
    if not re.fullmatch(r'ws://[^/\s:]+(:\d+)?/ws', url or ''):
        return (f'--relay {url!r}: expected ws://HOST:PORT/ws (the URL the relay role '
                'prints on the operator PC)')
    return None


def gpu_choice(requested: str, nvidia_found: bool, renders: bool) -> Tuple[Optional[str], str]:
    """
    Return the simulator's gpu argument (None: its default) and a sentence for the operator.

    ``renders``: the window or the cameras are on (gz renders nothing otherwise).
    """
    if requested not in GPU_CHOICES:
        raise PlanError(f'--gpu {requested!r}: one of {", ".join(GPU_CHOICES)}')
    if requested == 'nvidia':
        return 'nvidia', ('rendering on the NVIDIA card (gpu:=nvidia, as asked)' if nvidia_found
                          else 'gpu:=nvidia asked for, but no NVIDIA card was found: the '
                               'simulator launch will refuse')
    if requested == 'default':
        return None, 'rendering on the default device (--gpu default, as asked)'
    if not renders:
        return None, 'headless with the cameras off: the simulator renders nothing'
    if nvidia_found:
        return 'nvidia', 'NVIDIA card found: the simulator renders on it (gpu:=nvidia)'
    return None, ('no NVIDIA card found: the simulator renders on the default device; the '
                  'simulator check says whether that is fast enough')


def camera_streams(record_profile: dict) -> List[str]:
    """Return the required image streams of a deployment profile."""
    return sorted(n for n, s in (record_profile.get('streams') or {}).items()
                  if s.get('required') and s.get('kind') == 'image')


def check_options(opts: Options, record_profile: Optional[dict]) -> List[str]:
    """
    Raise PlanError when the options cannot make a session; return warnings otherwise.

    ``record_profile``: the deployment profile (stack role), None in the relay role.
    """
    if opts.role not in ROLES:
        raise PlanError(f'--role {opts.role!r}: one of {", ".join(ROLES)}')
    if not 0 < int(opts.port) < 65536:
        raise PlanError(f'--port {opts.port}: not a port')
    if ':' in opts.listen:
        raise PlanError(f'--listen {opts.listen}: an IPv4 address or a host name (an IPv6 '
                        "address would not fit the bridge's relay_url)")
    warnings = []
    if opts.role == 'relay':
        if opts.relay_url:
            raise PlanError('--relay names a relay elsewhere; the relay role starts one here')
        stack_only = [o for o in opts.explicit if o not in ('role', 'listen', 'port')]
        if stack_only:
            warnings.append('ignored in the relay role: ' + ', '.join(
                '--' + o.replace('_', '-') for o in stack_only))
        return warnings
    if opts.profile not in PROFILES:
        raise PlanError(f'--profile {opts.profile!r}: one of {", ".join(sorted(PROFILES))}. '
                        'A hardware profile is not part of this launcher yet: the hardware '
                        'bring-up is started by hand, and the teleop stack with it')
    if record_profile is None or record_profile.get('simulation') is not True:
        raise PlanError(f'profile {opts.profile}: its deployment profile is not marked '
                        '"simulation: true". This launcher starts simulation profiles only; '
                        'it never passes the simulation-only limits file or the reset to a '
                        'hardware deployment')
    if opts.relay_url:
        problem = relay_url_problem(opts.relay_url)
        if problem:
            raise PlanError(problem)
        if 'listen' in opts.explicit or 'port' in opts.explicit:
            raise PlanError('--listen and --port are for a relay started here; with --relay '
                            'the relay runs elsewhere')
    profile = PROFILES[opts.profile]
    if profile.simulator is None:
        for name in ('headless', 'cameras', 'gpu'):
            if name in opts.explicit:
                warnings.append(f'--{name.replace("_", "-")} applies to a simulator; '
                                f'profile {opts.profile} has none')
    elif opts.record and not opts.cameras:
        cams = camera_streams(record_profile)
        if cams:
            raise PlanError(f'recording with the cameras off: profile {opts.profile} '
                            f'requires {", ".join(cams)}, so the recorder would never become '
                            'ready. Turn the cameras on, or recording off (--no-record)')
    if not profile.rviz and 'rviz' in opts.explicit:
        warnings.append(f'--no-rviz applies to the mock profile; {opts.profile} starts no RViz')
    return warnings


def relay_step(opts: Options, checkout: str) -> Step:
    """Return the relay step (tools/relay.sh, as by hand)."""
    argv = [os.path.join(checkout, 'tools', 'relay.sh')]
    if opts.listen != DEFAULT_HOST:
        argv += ['--host', opts.listen]
    if int(opts.port) != DEFAULT_PORT:
        argv += ['--port', str(int(opts.port))]
    return Step('relay', tuple(argv))


def build_plan(opts: Options, *, checkout: str, bringup_share: str = '', session: str = '',
               log_dir: str = '', gpu: Optional[str] = None) -> List[Step]:
    """
    Return the steps of a session, in start order.

    ``checkout``: the code checkout (tools/relay.sh). ``bringup_share``: the bringup package's
    share directory (the simulation-only limits file for the simulator check). ``session``:
    the session name, given to the recorder so the launcher knows its directory. ``gpu``: the
    simulator's gpu argument from ``gpu_choice``.
    """
    if opts.role == 'relay':
        return [relay_step(opts, checkout)]
    profile = PROFILES[opts.profile]
    steps = []
    if not opts.relay_url:
        steps.append(relay_step(opts, checkout))
    if profile.simulator is not None:
        argv = ['ros2', 'launch', *profile.simulator]
        if opts.headless:
            argv.append('headless:=true')
        if opts.cameras:
            argv.append('sensors:=true')
        if gpu:
            argv.append(f'gpu:={gpu}')
        steps.append(Step('simulator', tuple(argv)))
        steps.append(Step('sim_check', (
            'ros2', 'run', 'ranger_xarm_teleop_record', 'record_commission', '--simulator',
            '--seconds', f'{float(opts.sim_check_seconds):g}',
            '--limits', os.path.join(bringup_share, LIMITS_FILE),
            '--out', os.path.join(log_dir, 'sim_check.json')), runs_to_end=True))
    argv = ['ros2', 'launch', *profile.stack]
    if opts.record:
        argv.append('record:=true')
    if opts.reset:
        argv.append('reset:=true')
    if not opts.feedback:
        argv.append('feedback:=false')
    if profile.rviz and not opts.rviz:
        argv.append('rviz:=false')
    if opts.record:
        argv.append(f'record_session:={session}')
        if opts.data_dir:
            argv.append(f'record_data_dir:={opts.data_dir}')
    steps.append(Step('stack', tuple(argv)))
    argv = ['ros2', 'launch', 'ranger_xarm_teleop_bridge', 'bridge.launch.py']
    url = opts.relay_url or local_relay_url(int(opts.port), opts.listen)
    if url != local_relay_url(DEFAULT_PORT):
        argv.append(f'relay_url:={url}')
    if not opts.feedback:
        argv.append('feedback:=false')
    steps.append(Step('bridge', tuple(argv)))
    steps.append(Step('probe', probe_argv(opts, log_dir), runs_to_end=True))
    return steps


def probe_argv(opts: Options, log_dir: str, timeout_s: float = 120.0) -> Tuple[str, ...]:
    """Return the probe's command line."""
    argv = ['ros2', 'run', 'ranger_xarm_teleop_session', 'session_probe',
            '--profile', PROFILES[opts.profile].record_profile,
            '--out', os.path.join(log_dir, 'probe.json'), '--timeout', f'{timeout_s:g}']
    for name in ('record', 'feedback', 'reset'):
        if getattr(opts, name):
            argv.append('--' + name)
    return tuple(argv)


def stop_sequence(steps: Sequence[Step]) -> List[Step]:
    """Return the steps in stop order."""
    return sorted(steps, key=lambda s: s.stop_rank)


def args_of(argv: Sequence[str]) -> Dict[str, str]:
    """Return the ``name:=value`` arguments of a launch command line."""
    return dict(a.split(':=', 1) for a in argv if ':=' in a)
