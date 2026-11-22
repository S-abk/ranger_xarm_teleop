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
The rules of each check stage: thresholds, verdicts and the one sentence that says what to do.

Pure: ``evaluate`` takes the samples the node collected (plain lists, see ``evaluate``) and the
resolved options, and returns the measured values and the verdicts. No ROS imports.

These thresholds are pass rules for a check, not robot limits; the safety gate owns the limits.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

from . import stats
from .sysinfo import is_number

PASS = 'PASS'
FAIL = 'FAIL'
NOT_JUDGED = 'NOT JUDGED'
INFO = 'INFO'

STAGES = ('idle', 'motion', 'teleop', 'cameras')

# ---- thresholds -------------------------------------------------------------------------

# Nominal /joint_states rate of the six-joint message when the profile gives none (Hz).
NOMINAL_JOINT_STATES_HZ = 150.0
# The measured rate must reach this share of nominal (a loaded host loses a few messages).
RATE_FRACTION = 0.95
# The largest gap in /joint_states must stay below this share of tool_pose_timeout_s, so that
# the teleop node's stale-pose hold cannot trigger on a gap the driver makes in normal running.
GAP_TIMEOUT_FRACTION = 1.0 / 3.0
# The suggested minimum tool_pose_timeout_s is this many times the largest gap.
TIMEOUT_GAP_FACTOR = 3.0
# How long to wait for /controller_manager/activity before saying it is not available (s).
ACTIVITY_WAIT_S = 5.0
# Header stamp against local arrival: the median offset must stay below this (s).
STAMP_OFFSET_LIMIT = 0.05
# The teleop node cycle (50 Hz, 20 ms): 99th percentile period, maximum period, longest gap (s).
TELEOP_P99_LIMIT = 0.030
TELEOP_MAX_LIMIT = 0.060
TELEOP_GAP_LIMIT = 0.100
# Stale repeats while the arm moves: fewer than this many in a row (messages) ...
STALE_RUN_LIMIT = 3
# ... and at least this share of the messages while moving must carry new joint values.
VALUE_CHANGE_FRACTION = 0.9
# Wrist velocity sign changes in any one second: at most this many.
OSCILLATION_LIMIT = 4
# Cameras: rate share of nominal, and the share of frames after a gap of more than 2 periods.
CAMERA_RATE_FRACTION = 0.95
CAMERA_GAP_FRACTION = 0.01
# A sample is "moving" above this joint speed (rad/s), measured between distinct samples; a
# moving span must last at least this long (s) to be judged.
MOVING_SPEED = 0.01
MIN_MOVING_SPAN = 0.5
# Wrist velocity sign changes only count above this speed (rad/s), within this window (s).
OSCILLATION_SPEED = 0.02
OSCILLATION_WINDOW = 1.0
# A /teleop/state and a /teleop/target_pose stamp match within this many seconds (they share a
# stamp when a target is published in the cycle).
MATCH_TOLERANCE = 0.010
# The lead is clamped at max_lead_translation_m (G-14), so it sits at the limit whenever the hand
# outruns the arm: only a lead beyond the limit by more than this means the clamp did not hold.
LEAD_TOLERANCE_M = 0.002
# A matched cycle within this of the limit counts as one where the clamp acted.
LEAD_AT_LIMIT_M = 0.001
# Wrist joints looked at for oscillation (index into the arm joints, 0-based).
WRIST_JOINTS = (3, 5)

DEFAULT_SECONDS = {'idle': 30.0, 'motion': 20.0, 'teleop': 20.0, 'cameras': 30.0}
DEFAULT_CONTROLLERS = ('xarm6_traj_controller', 'teleop_arm_controller')

ARMED_WARNING = ('the teleop node is armed: joining the graph can stall the stack for up to a '
                 'second; run this tool only while disarmed')


@dataclass
class Result:
    """One rule's outcome."""

    rule: str
    verdict: str
    measured: str
    threshold: str
    advice: str = ''

    def line(self) -> str:
        """Return the result as one line; FAIL and NOT JUDGED end with what to do."""
        text = f'{self.verdict}  {self.rule}: {self.measured} (rule: {self.threshold})'
        if self.verdict in (FAIL, NOT_JUDGED) and self.advice:
            text += f' -- to do: {self.advice}'
        return text

    def as_dict(self) -> dict:
        """Return the JSON form: rule, verdict, measured, threshold and reason."""
        return {'rule': self.rule, 'verdict': self.verdict, 'measured': self.measured,
                'threshold': self.threshold, 'reason': self.advice}


@dataclass
class Options:
    """What the run was asked for, already read from the files (see hw_check)."""

    nominal_hz: float = NOMINAL_JOINT_STATES_HZ
    limits_given: bool = False
    tool_pose_timeout_s: Any = None
    max_lead_translation_m: Any = None
    joint_limits_given: bool = False
    joint_acceleration_limits: Dict[str, Optional[float]] = field(default_factory=dict)
    expect_fingerprint: Optional[str] = None
    expect_active: Tuple[str, ...] = ()
    expect_inactive: Tuple[str, ...] = ()
    controllers: Tuple[str, ...] = DEFAULT_CONTROLLERS
    streams: List[dict] = field(default_factory=list)


def exit_status(results: List[Result]) -> int:
    """Return 1 when any rule FAILs, else 2 when any is NOT JUDGED, else 0 (INFO is ignored)."""
    verdicts = [r.verdict for r in results]
    if FAIL in verdicts:
        return 1
    if NOT_JUDGED in verdicts:
        return 2
    return 0


def _fmt(value, spec='.3f', unit='') -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return 'n/a'
    return format(value, spec) + unit


def _hz(value) -> str:
    return _fmt(value, '.1f', ' Hz')


# ---- joint states: the common measures --------------------------------------------------

def joint_state_measures(data: dict, nominal: float) -> dict:
    """Return the period distribution of the arm messages (arrival and stamp) and more."""
    arm = data.get('arm', [])
    gripper = data.get('gripper', [])
    mono = [s[0] for s in arm]
    stamps = [s[2] for s in arm]
    return {
        'arm_messages': len(arm),
        'gripper_messages': len(gripper),
        'gripper_rate_hz': stats.period_distribution([g[0] for g in gripper], None)['rate_hz'],
        'nominal_hz': nominal,
        'arrival': stats.period_distribution(mono, 1.0 / nominal),
        'stamp': stats.period_distribution(stamps, 1.0 / nominal),
        'stamp_offset_s': stats.stamp_offset([s[1] for s in arm], stamps),
        'publishers': sorted(data.get('js_publishers', [])),
    }


def rule_rate(m: dict) -> Result:
    """Judge the six-joint /joint_states rate against RATE_FRACTION of nominal."""
    name = 'joint_states rate'
    nominal = m['nominal_hz']
    threshold = (f'>= {RATE_FRACTION:g} x {nominal:g} Hz on the worse of arrival and header '
                 'stamps')
    a, s = m['arrival']['rate_hz'], m['stamp']['rate_hz']
    measured = f"arrival {_hz(a)}, stamps {_hz(s)}, {m['arm_messages']} messages"
    if a is None or s is None:
        return Result(name, NOT_JUDGED, measured, threshold,
                      'no /joint_states message with all six arm joints arrived: start the '
                      'driver and check the topic')
    if min(a, s) >= RATE_FRACTION * nominal:
        return Result(name, PASS, measured, threshold)
    return Result(name, FAIL, measured, threshold,
                  'the controller manager is not holding its rate: see the procedure, stage '
                  "2, 'if 150 Hz is not held'")


def rule_gap(m: dict, opts: Options) -> Result:
    """Largest /joint_states gap against tool_pose_timeout_s / 3 (arrival and stamps)."""
    name = 'joint_states largest gap'
    a, s = m['arrival']['largest_gap'], m['stamp']['largest_gap']
    if a is None or s is None:
        return Result(name, NOT_JUDGED, 'no gaps measured',
                      f'< tool_pose_timeout_s x {GAP_TIMEOUT_FRACTION:.3f}',
                      'no six-joint /joint_states arrived: start the driver and check the topic')
    gap = max(a, s)
    measured = f'arrival {_fmt(a)} s, stamps {_fmt(s)} s'
    if not opts.limits_given:
        return Result(name, NOT_JUDGED, measured + f'; suggested minimum timeout '
                      f'{_fmt(TIMEOUT_GAP_FACTOR * gap)} s',
                      f'largest gap < tool_pose_timeout_s / {1 / GAP_TIMEOUT_FRACTION:g}',
                      'pass --limits <your limits file> to judge it')
    timeout = opts.tool_pose_timeout_s
    threshold = f'largest gap < tool_pose_timeout_s / {1 / GAP_TIMEOUT_FRACTION:g}'
    if not is_number(timeout) or timeout <= 0:
        return Result(name, NOT_JUDGED, measured, threshold,
                      'set tool_pose_timeout_s in your limits file first')
    bound = timeout * GAP_TIMEOUT_FRACTION
    threshold += f' = {bound:.3f} s'
    if gap < bound:
        return Result(name, PASS, measured, threshold)
    return Result(name, FAIL, measured, threshold,
                  'the largest gap is too long for tool_pose_timeout_s: raise the timeout to '
                  f'at least {_fmt(TIMEOUT_GAP_FACTOR * gap)} s or fix the cause of the gap')


def info_effort(data: dict) -> Result:
    """Whether the arm messages carry a finite effort array (informational)."""
    arm = data.get('arm', [])
    name = 'joint_states effort'
    if not arm:
        return Result(name, INFO, 'no arm message', 'informational (expected absent)')
    ok = sum(1 for s in arm if s[5])
    state = 'present' if ok == len(arm) else 'absent' if ok == 0 else 'present in some'
    return Result(name, INFO, f'{state} ({ok} of {len(arm)} messages)',
                  'informational (expected absent on this driver)')


# ---- controllers, description, clock, machine ------------------------------------------

def rules_controllers(data: dict, opts: Options) -> List[Result]:
    """Report controller states; judge --expect-active / --expect-inactive."""
    activity = data.get('activity')
    cstate = data.get('controller_state', {})
    results = []
    if activity is None:
        shown = ('not available (no /controller_manager/activity: older controller_manager?)')
    else:
        shown = ', '.join(f'{n}: {label}' for n, label in sorted(activity.items())) or 'none'
    publishing = [n for n in opts.controllers if cstate.get(n, 0) > 0]
    results.append(Result('controllers', INFO, f'{shown}; publishing controller_state: '
                          + (', '.join(publishing) or 'none'), 'informational'))
    for name in opts.expect_active:
        if activity is not None:
            label = activity.get(name)
            ok = label == 'active'
            measured = f'{name} is {label or "not loaded"}'
        else:
            label = None
            ok = cstate.get(name, 0) > 0
            measured = (f'{name} publishes controller_state' if ok
                        else f'{name} does not publish controller_state')
        results.append(Result(f'controller {name} active', PASS if ok else FAIL, measured,
                              'expected active', f'activate {name} before this stage'))
    for name in opts.expect_inactive:
        if activity is not None:
            label = activity.get(name)
            ok = label != 'active'
            measured = f'{name} is {label or "not loaded"}'
        else:
            ok = cstate.get(name, 0) == 0
            measured = (f'{name} does not publish controller_state' if ok
                        else f'{name} publishes controller_state')
        results.append(Result(f'controller {name} inactive', PASS if ok else FAIL, measured,
                              'expected not active',
                              f'deactivate {name} before this stage'))
    return results


def rules_description(info: Optional[dict], opts: Options) -> List[Result]:
    """Judge position control and the fingerprint; report the plugins prominently."""
    if info is None:
        text = 'no /robot_description received'
        advice = 'start the robot bring-up (robot_state_publisher) before this check'
        out = [Result('robot description', NOT_JUDGED, text, 'received within the run', advice),
               Result('velocity_control', NOT_JUDGED, text, 'false (position control)', advice)]
        if opts.expect_fingerprint:
            out.append(Result('description fingerprint', NOT_JUDGED, text,
                              f'sha256 = {opts.expect_fingerprint}', advice))
        return out
    out = []
    if info['simulator'] is None:
        mode = 'unknown (no ros2_control hardware found)'
    else:
        mode = 'mock/simulator' if info['simulator'] else 'hardware'
    out.append(Result(
        'description backend', INFO,
        f"{mode.upper()}: plugins {', '.join(info['plugins']) or 'none'}; gripper "
        f"{'present' if info['gripper'] else 'absent'}; sha256 {info['sha256']}",
        'informational'))
    values = info['velocity_control']
    name = 'velocity_control'
    threshold = 'false (position control)'
    if info['error']:
        out.append(Result(name, NOT_JUDGED, info['error'], threshold,
                          'the robot description is not readable XML: restart the bring-up'))
    elif not values:
        out.append(Result(name, NOT_JUDGED, 'param velocity_control not found', threshold,
                          'the description has no velocity_control hardware param: check the '
                          'bring-up arguments'))
    elif all(v in ('false', '0') for v in values):
        out.append(Result(name, PASS, ', '.join(values), threshold))
    else:
        out.append(Result(name, FAIL, ', '.join(values), threshold,
                          'the bring-up runs velocity control; the procedure requires position '
                          'control (xarm_velocity_control:=false)'))
    if opts.expect_fingerprint:
        want = opts.expect_fingerprint.strip().lower()
        out.append(Result('description fingerprint',
                          PASS if info['sha256'] == want else FAIL, info['sha256'],
                          f'sha256 = {want}',
                          'the robot description changed since the run that gave this '
                          'fingerprint'))
    return out


def rule_clock(data: dict, m: dict) -> List[Result]:
    """No /clock publisher; header stamps within STAMP_OFFSET_LIMIT of the local clock."""
    pubs = data.get('clock_publishers', [])
    out = [Result('no /clock publisher', FAIL if pubs else PASS,
                  ', '.join(pubs) if pubs else 'none', 'no publisher of /clock',
                  'something publishes /clock: a simulator is running or use_sim_time is set '
                  'somewhere')]
    off = m['stamp_offset_s']
    threshold = f'|median(local time - header stamp)| < {STAMP_OFFSET_LIMIT} s'
    if off is None:
        out.append(Result('stamp offset', NOT_JUDGED, 'no six-joint message', threshold,
                          'no six-joint /joint_states arrived: start the driver'))
    elif abs(off) < STAMP_OFFSET_LIMIT:
        out.append(Result('stamp offset', PASS, f'{off:+.4f} s', threshold))
    else:
        out.append(Result('stamp offset', FAIL, f'{off:+.4f} s', threshold,
                          'header stamps are not on the local clock: check that no node uses '
                          'simulated time and that the clocks agree'))
    return out


def rule_governor(machine: dict) -> Result:
    """Every cpu on the performance governor."""
    name = 'cpu governor'
    threshold = 'every cpu on performance'
    govs = machine.get('governors') or {}
    if not govs:
        return Result(name, NOT_JUDGED, 'no cpufreq governor files', threshold,
                      'this machine does not expose the governor: set it by hand and note it')
    shown = ', '.join(machine.get('governor_set') or sorted(set(govs.values())))
    if all(v == 'performance' for v in govs.values()):
        return Result(name, PASS, f'{shown} on {len(govs)} cpus', threshold)
    return Result(name, FAIL, f'{shown} on {len(govs)} cpus', threshold,
                  'set the CPU governor to performance (see the procedure)')


def teleop_cycle(data: dict) -> dict:
    """Return the period distribution of the /teleop/state header stamps."""
    return stats.period_distribution([s[1] for s in data.get('states', [])], 0.02)


def rules_teleop_cycle(data: dict) -> List[Result]:
    """Judge the teleop node cycle: p99, maximum and longest gap of the state stamps."""
    d = teleop_cycle(data)
    names = ('teleop cycle p99', 'teleop cycle max', 'teleop cycle gap')
    limits = (TELEOP_P99_LIMIT, TELEOP_MAX_LIMIT, TELEOP_GAP_LIMIT)
    ops = ('<', '<', '<')
    if d['count'] < 2:
        return [Result(n, NOT_JUDGED, f"{d['count']} /teleop/state messages",
                       f'{op} {lim:g} s', 'the teleop stack is not running')
                for n, lim, op in zip(names, limits, ops)]
    values = (d['p99'], d['max'], d['largest_gap'])
    detail = (f"min {_fmt(d['min'])} p50 {_fmt(d['p50'])} p90 {_fmt(d['p90'])} "
              f"p99 {_fmt(d['p99'])} max {_fmt(d['max'])} s over {d['count']} messages")
    out = []
    for n, v, lim in zip(names, values, limits):
        out.append(Result(n, PASS if v < lim else FAIL, f'{_fmt(v)} s ({detail})',
                          f'< {lim:g} s',
                          'the teleop node is not keeping its 50 Hz cycle: check the load of '
                          'the machine and the governor'))
    return out


def rule_overruns(data: dict) -> Result:
    """No controller_manager overruns and no driver Read() failures in /rosout."""
    r = data.get('rosout', {})
    overruns, reads = r.get('overruns', 0), r.get('read_failures', 0)
    examples = (r.get('overrun_examples', []) + r.get('read_examples', []))[:5]
    measured = f'{overruns} overruns, {reads} read failures'
    if examples:
        measured += '; e.g. ' + ' | '.join(e[:120] for e in examples)
    return Result('overruns and read failures', PASS if not overruns and not reads else FAIL,
                  measured, 'zero of both during the run',
                  'the controller manager or the driver reported trouble: stop and read the '
                  'log before moving the arm')


def versions_info(versions: dict) -> Result:
    """Report the installed versions of the controller stack (informational)."""
    shown = ', '.join(f'{k} {v or "not found"}' for k, v in sorted(versions.items()))
    return Result('package versions', INFO, shown, 'informational')


def rule_driver_gate(gate: dict) -> Result:
    """Judge that xarm_ros2 contains the 2025-03-18 joint-state change."""
    name = 'xarm_ros2 joint-state change'
    threshold = 'commit cc44f9f is an ancestor of the installed xarm_ros2 HEAD'
    head = (gate.get('head') or 'unknown')[:12]
    measured = f"HEAD {head} ({gate.get('describe') or 'no describe'})"
    if gate.get('evidence'):
        measured += '; ' + gate['evidence']
    if gate['verdict'] == PASS:
        return Result(name, PASS, measured, threshold)
    if gate['verdict'] == FAIL:
        return Result(name, FAIL, measured, threshold,
                      'the installed xarm_ros2 does not contain the 2025-03-18 joint-state '
                      'change (cc44f9f): stop; update the driver before stage 2')
    return Result(name, NOT_JUDGED, gate.get('reason') or measured, threshold,
                  'pass --workspace <the workspace holding src/xarm_ros2>')


# ---- motion -----------------------------------------------------------------------------

def motion_measures(data: dict) -> dict:
    """Return stale repeats, velocity source, peak accelerations and oscillation counts."""
    arm = data.get('arm', [])
    t = [s[2] for s in arm]
    pos = [s[3] for s in arm]
    vel, source = stats.joint_velocities(t, pos, [s[4] for s in arm])
    peaks = stats.peak_acceleration(t, vel)
    osc = {stats.ARM_JOINTS[j]: stats.sign_changes_per_window(
        t, vel[:, j], OSCILLATION_SPEED, OSCILLATION_WINDOW)
           for j in WRIST_JOINTS} if len(arm) else {}
    return {'stale': stats.stale_analysis(t, pos, MOVING_SPEED, MIN_MOVING_SPAN),
            'velocity_source': source,
            'peak_acceleration': dict(zip(stats.ARM_JOINTS, peaks)),
            'wrist_sign_changes_per_s': osc}


def rule_stale(mm: dict) -> Result:
    """No stale repeats while moving; joint values change with the messages."""
    s = mm['stale']
    name = 'stale repeats while moving'
    threshold = (f'longest stale run < {STALE_RUN_LIMIT} messages and value-change rate >= '
                 f'{VALUE_CHANGE_FRACTION:g} x message rate while moving')
    if not s['judged']:
        return Result(name, NOT_JUDGED, 'no motion seen', threshold,
                      'no motion seen: run this during the planned motion')
    measured = (f"{s['stale_count']} stale, longest {s['longest_stale_messages']} messages "
                f"({_fmt(s['longest_stale_seconds'])} s); values change at "
                f"{_hz(s['value_change_rate_hz'])} against {_hz(s['message_rate_hz'])} "
                f"messages while moving ({_fmt(s['moving_seconds'], '.1f')} s moving)")
    ok = (s['longest_stale_messages'] < STALE_RUN_LIMIT
          and s['value_change_fraction'] >= VALUE_CHANGE_FRACTION)
    if ok:
        return Result(name, PASS, measured, threshold)
    return Result(name, FAIL, measured, threshold,
                  'joint states repeat while the arm moves: the driver is not reading the arm '
                  'every cycle; stop (procedure stage 3)')


def rule_acceleration(mm: dict, opts: Options) -> Result:
    """Peak joint acceleration against the MoveIt joint limits file."""
    name = 'peak joint acceleration'
    peaks = mm['peak_acceleration']
    shown = ', '.join(f'{j[-1]}: {_fmt(p, ".2f")}' for j, p in peaks.items())
    measured = f"{shown} rad/s^2 (velocity from {mm['velocity_source']})"
    if not opts.joint_limits_given:
        return Result(name, NOT_JUDGED, measured, 'peak <= max_acceleration of every joint',
                      'pass --joint-limits <MoveIt joint_limits.yaml> to judge it')
    limits = opts.joint_acceleration_limits
    over, judged = [], 0
    for joint, peak in peaks.items():
        limit = limits.get(joint)
        if limit is None or peak is None:
            continue
        judged += 1
        if peak > limit:
            over.append(f'{joint} {peak:.2f} > {limit:g}')
    threshold = 'peak <= max_acceleration of every joint with has_acceleration_limits'
    if not judged:
        return Result(name, NOT_JUDGED, measured, threshold,
                      'no joint has acceleration limits in that file or no motion was seen')
    if over:
        return Result(name, FAIL, measured + '; over: ' + ', '.join(over), threshold,
                      'a joint exceeded its acceleration limit: reduce the planned motion '
                      'speed or acceleration scaling')
    return Result(name, PASS, measured, threshold)


def rule_oscillation(mm: dict) -> Result:
    """Wrist velocity sign changes: at most OSCILLATION_LIMIT in any second."""
    name = 'wrist oscillation'
    threshold = f'at most {OSCILLATION_LIMIT} sign changes per second on joint 4 and joint 6'
    osc = mm['wrist_sign_changes_per_s']
    if not osc:
        return Result(name, NOT_JUDGED, 'no six-joint message', threshold,
                      'no six-joint /joint_states arrived: start the driver')
    measured = ', '.join(f'{j}: {n}/s' for j, n in osc.items())
    if all(n <= OSCILLATION_LIMIT for n in osc.values()):
        return Result(name, PASS, measured, threshold)
    return Result(name, FAIL, measured, threshold,
                  'the wrist oscillates: release the grip / stop the motion; this is the '
                  "solver's limit cycle near the wrist singularity")


# ---- teleop -----------------------------------------------------------------------------

def teleop_measures(data: dict) -> dict:
    """Return the lead statistics and the lag estimate from /teleop/state and the target."""
    states = [s for s in data.get('states', []) if s[3]]
    targets = data.get('targets', [])
    st = [s[1] for s in states]
    m = [s[4] for s in states]
    tt = [x[0] for x in targets]
    g = [x[1] for x in targets]
    return {'lead': stats.lead_stats(st, m, tt, g, MATCH_TOLERANCE),
            'lag_s': stats.lag_estimate(st, m, tt, g) if states and targets else None}


def rules_lead(tm: dict, opts: Options) -> List[Result]:
    """
    Lead of the target over the measured tool pose, against the G-14 clamp; and the lag.

    The clamp holds the lead at max_lead_translation_m whenever the hand outruns the arm, so
    the rule is that the largest lead stays within the limit (plus LEAD_TOLERANCE_M); how often
    the clamp acted is reported beside it, and the lag estimate is what the limit is chosen
    from (lead limit >= speed cap x lag).
    """
    lead = tm['lead']
    name = 'target lead (largest, translation)'
    shown = (f"p50 {_fmt(lead['p50'], '.4f')} p99 {_fmt(lead['p99'], '.4f')} max "
             f"{_fmt(lead['max'], '.4f')} m over {lead['count']} matched cycles")
    threshold = f'max <= max_lead_translation_m + {LEAD_TOLERANCE_M:g} m'
    out = []
    if lead['count'] == 0:
        out.append(Result(name, NOT_JUDGED, 'no target matched a valid measured pose',
                          threshold, 'run this while the clutch is engaged and '
                          '/teleop/target_pose is published'))
    elif not opts.limits_given:
        out.append(Result(name, NOT_JUDGED, shown, threshold,
                          'pass --limits <your limits file> to judge it'))
    elif not is_number(opts.max_lead_translation_m):
        out.append(Result(name, NOT_JUDGED, shown, threshold,
                          'set max_lead_translation_m in your limits file first'))
    else:
        bound = float(opts.max_lead_translation_m)
        out.append(Result(name, PASS if lead['max'] <= bound + LEAD_TOLERANCE_M else FAIL,
                          shown, f'max <= max_lead_translation_m = {bound:g} m + '
                          f'{LEAD_TOLERANCE_M:g} m',
                          'the target led the arm beyond the lead limit: the clamp did not '
                          'hold; stop and report it'))
        values = lead.get('values') or []
        share = sum(1 for v in values if v >= bound - LEAD_AT_LIMIT_M) / len(values)
        out.append(Result('target lead at the limit', INFO,
                          f'{share * 100.0:.0f} % of matched cycles (the G-14 clamp acted)',
                          'informational: often at the limit means the hand outruns the '
                          'speed cap; raise neither alone (lead >= cap x lag)'))
    out.append(Result('target lag', INFO,
                      f"{_fmt(tm['lag_s'])} s (cross-correlation of target and measured x, y, "
                      'z; positive: the arm is later)', 'informational'))
    return out


# ---- cameras ----------------------------------------------------------------------------

def camera_measures(data: dict, streams: List[dict]) -> dict:
    """Return the rate and gap statistics of every image stream of the profile."""
    out = {}
    for spec in streams:
        rec = data.get('cameras', {}).get(spec['name'], {'mono': [], 'stamps': [], 'first': None})
        nominal = spec.get('rate_hz')
        period = 1.0 / nominal if nominal else None
        a = stats.period_distribution(rec['mono'], period)
        s = stats.period_distribution(rec['stamps'], period)
        out[spec['name']] = {'topic': spec['topic'], 'frames': len(rec['mono']),
                             'arrival': a, 'stamp': s, 'first_frame': rec['first']}
    return out


def rules_cameras(cm: dict, streams: List[dict]) -> List[Result]:
    """Per stream: rate, gaps of more than two periods, size and encoding."""
    out = []
    for spec in streams:
        n, rec = spec['name'], cm[spec['name']]
        nominal = spec.get('rate_hz')
        a, s = rec['arrival'], rec['stamp']
        if rec['frames'] < 2:
            for rule, thr in ((f'{n} rate', f'>= {CAMERA_RATE_FRACTION:g} x rate_hz'),
                              (f'{n} gaps', f'< {CAMERA_GAP_FRACTION:.0%} of frames after '
                               'a gap of more than 2 periods'),
                              (f'{n} format', 'size and encoding as the profile')):
                out.append(Result(rule, NOT_JUDGED, f"{rec['frames']} frames on {rec['topic']}",
                                  thr, f"no images on {rec['topic']}: start the camera driver "
                                  'and check the topic name'))
            continue
        if nominal:
            thr = f'>= {CAMERA_RATE_FRACTION:g} x {nominal:g} Hz on the worse of arrival and ' \
                  'stamps'
            measured = f"arrival {_hz(a['rate_hz'])}, stamps {_hz(s['rate_hz'])}"
            ok = (a['rate_hz'] is not None and s['rate_hz'] is not None
                  and min(a['rate_hz'], s['rate_hz']) >= CAMERA_RATE_FRACTION * nominal)
            out.append(Result(f'{n} rate', PASS if ok else FAIL, measured, thr,
                              f'the camera does not deliver its {nominal:g} Hz: check the '
                              'USB port, the cable and the stream profile'))
            frames = rec['frames']
            fa, fs = a['over_2x'] or 0, s['over_2x'] or 0
            share = max(fa, fs) / frames
            thr = f'< {CAMERA_GAP_FRACTION:.0%} of frames after a gap of more than 2 periods'
            measured = (f'arrival {fa} ({fa / frames:.2%}), stamps {fs} ({fs / frames:.2%}) of '
                        f'{frames} frames')
            out.append(Result(f'{n} gaps', PASS if share < CAMERA_GAP_FRACTION else FAIL,
                              measured, thr, 'the camera stalls: check the USB port, the '
                              'cable and the host load'))
        want = spec.get('image') or {}
        first = rec['first_frame']
        thr = (f"{want.get('width')}x{want.get('height')} {want.get('encoding')}"
               if want else 'no image block in the profile')
        measured = f'{first[0]}x{first[1]} {first[2]}' if first else 'no frame'
        if not want:
            out.append(Result(f'{n} format', INFO, measured, thr))
        else:
            ok = bool(first) and (first[0], first[1], first[2]) == (
                want.get('width'), want.get('height'), want.get('encoding'))
            out.append(Result(f'{n} format', PASS if ok else FAIL, measured, thr,
                              'the stream size or encoding differs from the profile: set the '
                              'camera driver profile or correct the deployment profile'))
    return out


# ---- the stages -------------------------------------------------------------------------

def evaluate(stage: str, data: dict, opts: Options) -> Tuple[dict, List[Result]]:
    """
    Return (measures, results) of one stage.

    ``data`` holds what the node collected: ``arm`` (tuples of arrival monotonic s, local
    system time s, header stamp s, six positions, six velocities or None, effort_ok),
    ``gripper`` ((arrival, stamp) of messages without the six arm joints), ``js_publishers``,
    ``clock_publishers``, ``activity`` (None or {controller: state label}),
    ``controller_state`` ({name: message count}), ``description`` (description_info or None),
    ``states`` ((arrival, header stamp s, armed, measured pose valid, (x, y, z))), ``targets``
    ((stamp s, (x, y, z))), ``rosout`` (counts and examples), ``machine``, ``versions``,
    ``gate`` (driver_gate), ``cameras`` ({stream: {mono, stamps, first}}).
    """
    measures: Dict[str, Any] = {}
    results: List[Result] = []
    if stage == 'cameras':
        measures['cameras'] = camera_measures(data, opts.streams)
        results += rules_cameras(measures['cameras'], opts.streams)
        return measures, results
    m = joint_state_measures(data, opts.nominal_hz)
    measures['joint_states'] = m
    results += [rule_rate(m), rule_gap(m, opts), info_effort(data)]
    results += rules_controllers(data, opts)
    results += rules_description(data.get('description'), opts)
    results += rule_clock(data, m)
    machine = data.get('machine', {})
    measures['machine'] = machine
    results.append(rule_governor(machine))
    measures['teleop_cycle'] = teleop_cycle(data)
    results += rules_teleop_cycle(data)
    measures['rosout'] = data.get('rosout', {})
    results.append(rule_overruns(data))
    measures['versions'] = data.get('versions', {})
    results.append(versions_info(data.get('versions', {})))
    measures['driver_gate'] = data.get('gate', {})
    results.append(rule_driver_gate(data.get('gate') or {'verdict': NOT_JUDGED,
                                                         'reason': 'no gate result'}))
    if stage in ('motion', 'teleop'):
        mm = motion_measures(data)
        measures['motion'] = mm
        results += [rule_stale(mm), rule_acceleration(mm, opts), rule_oscillation(mm)]
    if stage == 'teleop':
        tm = teleop_measures(data)
        measures['teleop'] = tm
        results += rules_lead(tm, opts)
    return measures, results
