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
The pre-flight: each item judged ok, missing or unknown, with what to do. No ROS imports.

Every function takes plain values collected elsewhere (the probe's JSON, the log parsers in
``logwatch``, local reads) and returns an ``Item``. Nothing here reads the system or acts on
it. An item that cannot be judged is UNKNOWN, never OK (INV-4). A part that is switched off is
OFF; the plan line is INFO. The session is ready when no item is MISSING; UNKNOWN items are
listed for the operator to check.

Labels and the operator's name are never part of any item's text (INV-7): the labels item says
only whether both are set.
"""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence

OK, MISSING, UNKNOWN, OFF, INFO = 'ok', 'missing', 'unknown', 'off', 'info'
SHOWN = {OK: 'ok', MISSING: 'MISSING', UNKNOWN: 'UNKNOWN', OFF: 'off', INFO: 'info'}

GAP_MARGIN = 0.5            # the task's criterion: p99 joint-state gap <= half the timeout
MIN_RTF = 0.6               # and the real-time factor at least this
MIN_STATUS_HZ = 2.0         # the operator feedback node publishes at 5 Hz
MIN_RESET_HZ = 2.0          # the reset node's state and its watchdog's sign of life
OWNER_MOVEIT = 1            # ArmOwner.OWNER_MOVEIT
OWNER_NAMES = {0: 'none', 1: 'MoveIt', 2: 'teleop', 3: 'switching'}
EPISODE_NOT_READY, EPISODE_IDLE = 0, 1
RENDER_HINT = ('render the simulator on the discrete GPU: README "Simulator rendering", '
               "gpu:=nvidia (the launcher's --gpu auto or --gpu nvidia)")
ADB_COMMAND = 'adb reverse tcp:8443 tcp:8443'
LABELS_TOOL = 'the episode_labels tool of ranger_xarm_teleop_record'


@dataclass(frozen=True)
class Item:
    """One line of the pre-flight."""

    key: str
    title: str
    status: str
    detail: str = ''
    todo: str = ''

    def line(self) -> str:
        """Return the console text of this item."""
        text = f'[{SHOWN[self.status]:>7}] {self.title}'
        if self.detail:
            text += f': {self.detail}'
        if self.todo and self.status in (MISSING, UNKNOWN):
            text += f'\n          -> {self.todo}'
        return text


def missing(items: Iterable[Item]) -> List[Item]:
    """Return the items that keep the session from being ready."""
    return [i for i in items if i.status == MISSING]


def unknown(items: Iterable[Item]) -> List[Item]:
    """Return the items the operator has to check."""
    return [i for i in items if i.status == UNKNOWN]


def render(items: Sequence[Item]) -> str:
    """Return the table for the console."""
    return '\n'.join(i.line() for i in items)


def headset_streams(record_profile: dict) -> List[str]:
    """Return the profile's streams that only the headset's input feeds (joy, input status)."""
    return sorted(n for n, s in (record_profile.get('streams') or {}).items()
                  if s.get('kind') in ('joy', 'input_status'))


# ------------------------------------------------------------------ before the stack
def plan_item(profile: str, role: str, parts: Dict[str, bool], sim: Optional[dict]) -> Item:
    """Return the profile, the role and which optional parts are on (always shown)."""
    on = ', '.join(f'{k} {"on" if v else "off"}' for k, v in parts.items())
    text = f'profile {profile}, role {role}' + (f'; {on}' if on else '')
    if sim:
        text += (f'; simulator {"headless" if sim["headless"] else "window"}, cameras '
                 f'{"on" if sim["cameras"] else "off"}')
    return Item('plan', 'Session', INFO, text)


def gpu_item(simulator: bool, gpu: Optional[str], sentence: str, nvidia_found: bool,
             renders: bool) -> Item:
    """Return where the simulator renders."""
    title = 'Simulator GPU'
    if not simulator:
        return Item('gpu', title, OFF, 'no simulator in this profile')
    if renders and nvidia_found and gpu != 'nvidia':
        return Item('gpu', title, MISSING, sentence,
                    'an NVIDIA card is here and the window or cameras are on: use it '
                    '(--gpu auto or --gpu nvidia); on the integrated GPU the simulator ran at '
                    '0.36-0.72 of real time')
    if gpu == 'nvidia' and not nvidia_found:
        return Item('gpu', title, MISSING, sentence,
                    'install the NVIDIA driver, or start without --gpu nvidia')
    return Item('gpu', title, OK if gpu == 'nvidia' or not renders else INFO, sentence)


def _ms(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'unknown'
    return f'{v:.0f} ms'


def sim_check_item(report: Optional[dict], exit_code: Optional[int], output_tail: str = '',
                   enabled: bool = True) -> Item:
    """
    Return the simulator's speed and joint-state delivery, from record_commission --simulator.

    OK when the 99th-percentile gap between joint states is at most half of
    tool_pose_timeout_s and the simulator runs at 0.6 of real time or more (the task's two
    criteria). Gaps over the timeout itself are shown in the detail.
    """
    title = 'Simulator speed and joint-state delivery'
    if not enabled:
        return Item('sim_check', title, OFF, 'no simulator in this profile')
    if report is None:
        lines = output_tail.strip().splitlines()
        why = lines[-1] if lines else 'no report'
        return Item('sim_check', title, UNKNOWN,
                    f'the check gave no report (exit {exit_code}): {why}',
                    'run it by hand with the simulator up and before the teleop stack '
                    '(README "Simulator rendering")')
    limit_ms = report.get('tool_pose_timeout_ms')
    p99 = (report.get('gap_ms') or {}).get('p99')
    rtf = report.get('real_time_factor')
    n = report.get('joint_states') or 0
    seconds = report.get('seconds', '?')
    if n < 2 or p99 is None or limit_ms is None:
        return Item('sim_check', title, MISSING, f'{n} joint states in {seconds} s',
                    'is the simulator running with its controllers? See its log')
    margin = GAP_MARGIN * limit_ms
    detail = (f'p99 gap {_ms(p99)} (at most {_ms(margin)}), real-time factor '
              f'{"unknown" if rtf is None else f"{rtf:.2f}"} (at least {MIN_RTF}), '
              f'{report.get("gaps_over_limit") or 0} gaps over tool_pose_timeout_s '
              f'({_ms(limit_ms)}) in {seconds} s; window and cameras as this session uses them')
    if p99 > margin or (rtf is not None and rtf < MIN_RTF):
        return Item('sim_check', title, MISSING, detail, RENDER_HINT)
    if rtf is None:
        return Item('sim_check', title, UNKNOWN, detail + '; no /clock, so no real-time factor',
                    'is the simulator publishing /clock?')
    return Item('sim_check', title, OK, detail)


def disk_item(free_gb: Optional[float], min_gb: float, where: str, record: bool) -> Item:
    """Return the free space where the session is written."""
    title = 'Disk space'
    if not record:
        return Item('disk', title, OFF, 'recording off')
    if free_gb is None:
        return Item('disk', title, UNKNOWN, f'could not read the free space of {where}',
                    'check that the data directory exists and is readable')
    if free_gb < min_gb:
        return Item('disk', title, MISSING,
                    f'{free_gb:.1f} GB free in {where}; the recorder needs {min_gb:g} GB to '
                    'start', 'free space there, or choose another --data-dir')
    return Item('disk', title, OK, f'{free_gb:.0f} GB free in {where} (needs {min_gb:g} GB)')


def labels_item(present: bool, problem: Optional[str], record: bool) -> Item:
    """Return whether the task label and operator are set (their values are never shown)."""
    title = 'Task label and operator'
    if not record:
        return Item('labels', title, OFF, 'recording off')
    if present:
        return Item('labels', title, OK, 'both set (kept in the data directory only)')
    # read_labels' problems name the file and the absent key, never a value.
    return Item('labels', title, MISSING, problem or 'not set',
                f'type them at the start prompt (or with {LABELS_TOOL}); START is refused '
                'without them')


def commits_item(teleop: Optional[dict], workspace: Optional[dict]) -> Item:
    """Return the commits of the checkout and the robot workspace, and whether either is dirty."""
    parts, unknown_ = [], []
    for name, info in (('teleop checkout', teleop), ('robot workspace', workspace)):
        info = info or {}
        commit = info.get('commit')
        if not commit:
            unknown_.append(name)
            parts.append(f'{name}: commit unknown')
            continue
        dirty = info.get('dirty')
        if dirty is None:
            state = 'dirty or not: unknown'
        elif dirty:
            state = f'dirty, {len(info.get("changed") or [])} changed files'
        else:
            state = 'clean'
        parts.append(f'{name} {commit[:8]} ({state})')
    if unknown_:
        return Item('commits', 'Commits', UNKNOWN, '; '.join(parts),
                    'the episode node refuses START without both commits: is the '
                    + ' / '.join(unknown_) + ' a git checkout (RANGER_XARM_WS)?')
    return Item('commits', 'Commits', OK, '; '.join(parts))


def adb_item(facts: Optional[dict], relay_here: bool) -> Item:
    """
    Return ``adb reverse`` on the machine running the relay.

    ``facts``: {'adb': installed, 'devices': [serials], 'reverse': [lines of
    adb reverse --list], 'ran': None or {'command', 'code', 'output'}} (the start step's run).
    """
    title = 'adb reverse (headset USB link)'
    if not relay_here:
        return Item('adb', title, UNKNOWN, 'the relay runs on another machine',
                    'checked by the relay role on the operator PC')
    if not facts or not facts.get('adb'):
        return Item('adb', title, UNKNOWN, 'adb is not installed here',
                    f'install adb, then {ADB_COMMAND}')
    ran = facts.get('ran')
    ran_text = ''
    if ran:
        out = (ran.get('output') or '').strip()
        ran_text = f'ran "{ran["command"]}": exit {ran["code"]}' + (f', {out}' if out else '')
    if not facts.get('devices'):
        return Item('adb', title, MISSING, 'no headset listed by adb devices',
                    'plug the headset in by USB and allow USB debugging on it, then '
                    + ADB_COMMAND)
    if any('tcp:8443' in line for line in facts.get('reverse') or []):
        return Item('adb', title, OK,
                    'tcp:8443 reversed to the headset' + (f' ({ran_text})' if ran_text else ''))
    return Item('adb', title, MISSING, ran_text or 'tcp:8443 is not reversed',
                f'run {ADB_COMMAND} (after every USB re-plug)')


# ------------------------------------------------------------------ the stack, from the probe
def streams_item(probe: Optional[dict]) -> Item:
    """Return the profile's required, ungated, robot-side streams at their rates."""
    title = 'Required streams at their rates'
    if probe is None:
        return Item('streams', title, UNKNOWN, 'the probe gave no report',
                    'see the probe log in the session log directory')
    streams = probe.get('streams') or {}
    if not streams:
        return Item('streams', title, UNKNOWN, 'the probe judged no stream', 'see the probe log')
    bad = [s['problem'] for _, s in sorted(streams.items()) if s.get('problem')]
    if bad:
        return Item('streams', title, MISSING, '; '.join(bad),
                    'see the stack log for the node that publishes it')
    return Item('streams', title, OK, ', '.join(sorted(streams)))


def owner_item(probe: Optional[dict]) -> Item:
    """Return whether the stack came up disarmed, with MoveIt owning the arm (INV-1)."""
    title = 'Teleop disarmed, MoveIt owns the arm'
    teleop = (probe or {}).get('teleop') or {}
    if not teleop.get('seen'):
        return Item('owner', title, UNKNOWN, 'no /teleop/state seen',
                    'see the stack log: is the teleop node running?')
    armed, owner = bool(teleop.get('armed')), teleop.get('owner')
    if armed or owner != OWNER_MOVEIT:
        return Item('owner', title, MISSING,
                    f'armed {armed}, owner {OWNER_NAMES.get(owner, owner)}',
                    'the stack must come up disarmed with MoveIt as owner: see the stack log')
    return Item('owner', title, OK, 'disarmed, owner MoveIt')


def recorder_item(probe: Optional[dict], record: bool, headset_streams: Sequence[str]) -> Item:
    """Return whether the recorder is ready, or waits only for the headset's input."""
    title = 'Recorder'
    if not record:
        return Item('recorder', title, OFF, 'recording off')
    ep = (probe or {}).get('episode') or {}
    if not ep.get('seen'):
        return Item('recorder', title, MISSING if probe else UNKNOWN,
                    'no /teleop/episode/state seen',
                    'see the stack log for the episode node and the recorder')
    if ep.get('state') == EPISODE_IDLE:
        return Item('recorder', title, OK, 'ready (X starts an episode once armed)')
    reasons = [r.strip() for r in (ep.get('detail') or '').split(';') if r.strip()]
    if ep.get('state') == EPISODE_NOT_READY:
        others = [r for r in reasons if not any(r.startswith(f'{s}:') for s in headset_streams)]
        if reasons and not others:
            return Item('recorder', title, OK,
                        'ready except for the headset input it waits for (the panel shows '
                        'the recording state once input arrives)')
        return Item('recorder', title, MISSING, '; '.join(others) or 'not ready',
                    'the episode node says what is missing (above)')
    return Item('recorder', title, MISSING, f'state {ep.get("state")} at start (expected ready)',
                'see the stack log')


def _rate_item(key: str, title: str, probe: Optional[dict], enabled: bool,
               topics: Sequence[str], min_hz: float, todo: str) -> Item:
    if not enabled:
        return Item(key, title, OFF, 'switched off')
    if probe is None:
        return Item(key, title, UNKNOWN, 'the probe gave no report', todo)
    rates = probe.get('rates') or {}
    parts, bad = [], False
    for topic in topics:
        hz = rates.get(topic)
        parts.append(f'{topic} ' + ('none' if not hz else f'{hz:.1f} Hz'))
        bad = bad or not hz or hz < min_hz
    return Item(key, title, MISSING if bad else OK, ', '.join(parts), todo if bad else '')


def feedback_item(probe: Optional[dict], enabled: bool) -> Item:
    """Return whether the operator feedback node publishes the panel status."""
    return _rate_item('feedback', 'Operator feedback node', probe, enabled,
                      ['/teleop/operator/status'], MIN_STATUS_HZ,
                      'see the stack log for feedback_node')


def reset_item(probe: Optional[dict], enabled: bool) -> Item:
    """
    Return whether the reset node and its watchdog are alive.

    From /teleop/reset/state alone (its rate, ``enabled`` and ``watchdog_alive``): the probe
    never subscribes to the two liveness topics between the reset node and its watchdog.
    """
    title = 'Reset node and watchdog (hold-to-run, simulation only)'
    todo = 'see the stack log for reset_node and reset_watchdog_node'
    item = _rate_item('reset', title, probe, enabled, ['/teleop/reset/state'], MIN_RESET_HZ,
                      todo)
    if item.status != OK:
        return item
    reset = (probe or {}).get('reset') or {}
    if reset.get('enabled') is not True:
        return Item('reset', title, MISSING,
                    item.detail + '; the reset node reports itself off (it refuses unless the '
                    'robot description shows simulator hardware only)', todo)
    if reset.get('watchdog_alive') is not True:
        return Item('reset', title, MISSING,
                    item.detail + '; the reset node has no fresh sign of life from its '
                    'watchdog', todo)
    return Item('reset', title, OK, item.detail + '; enabled, watchdog alive')


def pairs_item(result: Optional[str], count: int, why: str) -> Item:
    """Return the teleop node's start-up check of MoveIt Servo's rigid self-collision pairs."""
    title = "MoveIt Servo's rigid self-collision pairs"
    if result == 'none':
        return Item('pairs', title, OK, 'no enabled pair without an arm joint between its links')
    if result == 'some':
        return Item('pairs', title, MISSING,
                    f'{count} enabled pairs no arm motion can change: Servo slows every joint '
                    'while one is near (a closed gripper)',
                    'disable them in the SRDF move_group reads; the teleop log lists them as '
                    'SRDF lines (README)')
    if result == 'not_checked':
        return Item('pairs', title, UNKNOWN, why, 'see the teleop node log')
    return Item('pairs', title, UNKNOWN, 'the teleop node has not reported the check',
                'see the teleop node log ("self-collision pair")')


# ------------------------------------------------------------------ the headset, from the logs
def headset_item(bridge: Optional[dict], relay: Optional[dict], role: str) -> Item:
    """
    Return whether the headset is connected and its input arrives.

    Stack role: the bridge's statistics line ({'lines', 'connected', 'rate', 'url'}). Relay
    role: the relay's log ({'peers': {peer: since}}).
    """
    title = 'Headset connected and sending input'
    todo = ('put on the headset, open http://localhost:8443 in its browser and press Enter VR '
            f'(USB link: {ADB_COMMAND})')
    if role == 'relay':
        peers = (relay or {}).get('peers') or {}
        if peers:
            since = sorted(peers.values())[0]
            return Item('headset', title, OK,
                        f'a page sends headset frames to this relay (since {since or "?"})')
        return Item('headset', title, MISSING, 'no page sends headset frames to this relay', todo)
    if not bridge or not bridge.get('lines'):
        return Item('headset', title, UNKNOWN,
                    "the bridge's first report is not in yet (every 5 s)",
                    'wait a few seconds; see the bridge log')
    if not bridge.get('connected'):
        return Item('headset', title, MISSING,
                    f'the bridge is not connected to the relay {bridge.get("url") or ""}'.strip(),
                    'is the relay running? (on the operator PC in the Wi-Fi set-up)')
    rate = bridge.get('rate') or 0.0
    if rate <= 0.0:
        return Item('headset', title, MISSING, 'connected to the relay, no headset frames', todo)
    return Item('headset', title, OK, f'{rate:.0f} frames/s reach the bridge')


def version_item(expected: Optional[str], relay_started_here: bool,
                 relay: Optional[dict]) -> Item:
    """
    Return whether the headset page is at the client version this checkout serves.

    Nothing on the wire carries the version. The page opens its WebSocket once, at load, and
    never reconnects; the relay serves the page with no-store. Headset frames reaching a relay
    this launcher started therefore come from a page loaded from it after it started.
    """
    title = 'Headset page version'
    look = f'client {expected}' if expected else 'the client version'
    if not relay_started_here:
        return Item('version', title, UNKNOWN, 'this launcher did not start the relay',
                    f'the panel footer must say "{look}"; reload the page if not')
    if not expected:
        return Item('version', title, UNKNOWN, 'feedback.js declares no FEEDBACK_VERSION',
                    'the panel footer shows the version the page runs')
    if (relay or {}).get('peers'):
        return Item('version', title, OK,
                    f'{look}: the page sending frames was loaded from this relay after it '
                    'started')
    return Item('version', title, UNKNOWN, 'no page sends frames to this relay yet',
                f'open or reload http://localhost:8443; the panel footer must say "{look}"')
