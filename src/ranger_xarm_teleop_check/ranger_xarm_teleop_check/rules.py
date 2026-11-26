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

import numpy as np

from ranger_xarm_teleop_servo.joint_accel import ACCELERATION_NAME, IMMEDIATE_STOP_LOG
from ranger_xarm_teleop_servo.joint_accel import STOP_TIME_LIMIT_S

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

# MoveIt Servo's output, the teleop command (G-18): the controller's trajectory topic, and the
# command period (s).
DEFAULT_COMMAND_TOPIC = '/teleop_arm_controller/joint_trajectory'
# The teleop node's joint-jog commands to MoveIt Servo, read to tell Servo's skips.
DEFAULT_JOG_TOPIC = '/servo_node/delta_joint_cmds'
COMMAND_PERIOD_S = 0.02
# A Servo skip can join two consecutive command steps into one, at most 2 x the limit; Servo's
# point timing moves that slightly either way. A point FAILs only above SKIP_FACTOR x
# SKIP_MARGIN x the limit. The first real reports decide whether to tighten it.
SKIP_FACTOR = 2.0
SKIP_MARGIN = 1.05
# Points above this many times the limit that no detected skip explains are reported (INFO).
EXPLAINED_ABOVE = 1.1
# Relative slack for floating-point rounding in the comparisons with the limit.
COMMAND_ROUNDING = 1e-6

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
    joint_acceleration_limit: Any = None
    # MoveIt Servo's max_expected_latency (s): how far ahead of its cycle it stamps each point,
    # from the servo.yaml the teleop launch loads; None when it could not be read.
    servo_latency_s: Any = None
    command_topic: str = DEFAULT_COMMAND_TOPIC
    jog_topic: str = DEFAULT_JOG_TOPIC
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


# ---- teleop: is the arm held back from following the hand? ----------------------------------

KEEPOUT_HOLD_INFO = 'link keep-out shaping the command (INFO)'
ROTATION_LEAD_INFO = 'target lead (rotation, INFO)'


def tracking_measures(data: dict) -> dict:
    """
    Return how often the link keep-out shaped the joint command, and the rotation lead.

    Over the engaged cycles of /teleop/state: those where the keep-out shaped the joint
    command (``keepout_bound``), with the clearance at the solver's reference against the
    margin and the nearest pair at the measured joints. In strokes through open space it
    should not act: on the first hardware stage 4 run (a pair at a 40 mm margin at the working
    posture) it removed every motion of joints 1 and 2 for the whole run, and the lead rules
    passed, because the gate absorbs the lead the keep-out blocks (G-14). Also the target's
    rotation lead (target vs measured, rad): the solver traded orientation for the translation
    the keep-out removed, and the tool pivoted about the wrist.
    """
    rows = [r for r in data.get('tracking', []) if r[0]]
    bound = [r for r in rows if r[2]]
    room = sorted(r[3] - r[4] for r in bound if math.isfinite(r[3]) and math.isfinite(r[4]))
    pairs: Dict[str, int] = {}
    for r in bound:
        if r[5]:
            key = f'{r[5]} - {r[6]}'
            pairs[key] = pairs.get(key, 0) + 1
    rotation = sorted(r[8] for r in rows if math.isfinite(r[8]))
    return {'engaged': len(rows), 'bound': len(bound), 'pairs': pairs,
            'room_p50_m': _quantile(room, 0.5), 'room_min_m': room[0] if room else None,
            'rotation_p50': _quantile(rotation, 0.5), 'rotation_p99': _quantile(rotation, 0.99),
            'rotation_max': rotation[-1] if rotation else None}


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return values[min(len(values) - 1, int(q * len(values)))]


def info_keepout_hold(km: dict) -> Result:
    """
    INFO: the share of engaged cycles in which the link keep-out shaped the joint command.

    Not judged yet: how often it acts in normal strokes at a working margin is not measured on
    the arm; the threshold comes from that data.
    """
    threshold = ('informational: in strokes through open space the keep-out should not act; '
                 'often, with the clearance at the margin, means the posture holds the arm '
                 'back; the threshold waits for stage 4 data')
    if km['engaged'] == 0:
        return Result(KEEPOUT_HOLD_INFO, INFO, 'no engaged cycle', threshold)
    share = km['bound'] / km['engaged']
    shown = f"{share * 100.0:.0f} % of {km['engaged']} engaged cycles ({km['bound']})"
    if km['bound']:
        if km['room_p50_m'] is not None:
            shown += (f"; clearance at the command's reference minus the margin: median "
                      f"{km['room_p50_m'] * 1000.0:.1f} mm, least "
                      f"{km['room_min_m'] * 1000.0:.1f} mm")
        ranked = sorted(km['pairs'].items(), key=lambda kv: -kv[1])
        names = ', '.join(f'{k} ({v})' for k, v in ranked)
        shown += f"; nearest pair at the measured joints: {names or 'not reported'}"
    return Result(KEEPOUT_HOLD_INFO, INFO, shown, threshold)


def info_rotation_lead(km: dict) -> Result:
    """INFO: the target's rotation lead over the measured tool pose while engaged."""
    threshold = ('informational: the G-14 clamp bounds it; often near its limit while the hand '
                 'turns slowly means the tool is not following the orientation')
    if km['rotation_max'] is None:
        return Result(ROTATION_LEAD_INFO, INFO, 'no engaged cycle', threshold)
    shown = ' '.join(f'{name} {math.degrees(km[key]):.1f}' for name, key in
                     (('p50', 'rotation_p50'), ('p99', 'rotation_p99'),
                      ('max', 'rotation_max'))) + f" deg over {km['engaged']} engaged cycles"
    return Result(ROTATION_LEAD_INFO, INFO, shown, threshold)


# ---- teleop: the command MoveIt Servo sends to the controller ----------------------------

def _command_limit(opts: Options) -> Tuple[Optional[float], str]:
    """Return (the acceleration limit, '') or (None, what to do about it)."""
    if not opts.limits_given:
        return None, 'pass --limits <your limits file> to judge it'
    value = opts.joint_acceleration_limit
    if not is_number(value) or value <= 0:
        return None, f'set {ACCELERATION_NAME} in your limits file first'
    return float(value), ''


def command_measures(data: dict, limit: Optional[float], latency: float = 0.0) -> dict:
    """
    Return the peaks, skips, spacing and stops of Servo's output (``data['command']``).

    ``limit`` is the acceleration limit (rad/s^2) or None: the counts above it are then None.
    Nothing here is a verdict.
    """
    messages = data.get('command', [])
    path = stats.command_points([m[1:] for m in messages])
    t, p = path['t'], path['p']
    capture = data.get('command_capture') or {}
    open_start, open_end = stats.capture_edges(
        messages[0][0] if messages else None, messages[-1][0] if messages else None,
        capture.get('began'), capture.get('ended'))
    an = stats.command_analysis(t, p, open_start, open_end)
    acc = np.abs(an['acc'])
    point_peak = np.full(len(t), np.nan)
    judged = np.isfinite(acc).all(axis=1)
    if judged.any():
        point_peak[judged] = acc[judged].max(axis=1)
    peaks = {}
    for j, name in enumerate(stats.ARM_JOINTS):
        col = acc[:, j][np.isfinite(acc[:, j])]
        peaks[name] = float(col.max()) if len(col) else None
    speed = an['speed']
    moved = bool(np.isfinite(speed).any() and np.nanmax(speed) > stats.SERVO_REST_SPEED)
    spacing = an['spacing']
    origin = path['origin_ns']
    releases, engages = stats.clutch_events(data.get('clutch', []))
    stops = []
    if origin is not None:
        rel = [(r - origin) * 1e-9 for r in releases]
        eng = [(e - origin) * 1e-9 for e in engages]
        stops = stats.stop_durations(t, an, rel, eng, open_end, latency)
    out = {'topic_messages': len(messages), 'skipped_messages': path['skipped'],
           'points': int(len(t)), 'runs': len(an['runs']), 'moved': moved,
           'open_start': open_start, 'open_end': open_end,
           'peak_acceleration': peaks,
           'peak_overall': (max(v for v in peaks.values() if v is not None)
                            if any(v is not None for v in peaks.values()) else None),
           'spacing_ms': {'p50': _ms(stats.percentile(spacing, 50.0)),
                          'p99': _ms(stats.percentile(spacing, 99.0)),
                          'max': _ms(max(spacing) if spacing else None)},
           'releases_seen': len(releases), 'stops': stops,
           'above_1x': None, 'above_2x': None}
    # Servo's skips, from its steps matched to the teleop node's commands (stats.servo_skips).
    jog = stats.jog_commands(data.get('jog', []))
    out['jog_messages'] = len(data.get('jog', []))
    step_limit = limit * COMMAND_PERIOD_S if limit is not None else math.inf
    skips = (stats.servo_skips(p, an['runs'], jog, COMMAND_PERIOD_S, step_limit)
             if jog else None)
    out['skips'] = ({k: v for k, v in skips.items() if k != 'skip_points'}
                    if skips is not None else None)
    out['judged_points'] = int(judged.sum())
    out['above_explained_limit'] = None
    out['unexplained'] = None
    if limit is not None:
        finite = point_peak[np.isfinite(point_peak)]
        out['above_1x'] = int((finite > limit * (1.0 + COMMAND_ROUNDING)).sum())
        out['above_2x'] = int((finite > SKIP_FACTOR * SKIP_MARGIN * limit
                               * (1.0 + COMMAND_ROUNDING)).sum())
        high = [i for i in range(len(t)) if np.isfinite(point_peak[i])
                and point_peak[i] > EXPLAINED_ABOVE * limit * (1.0 + COMMAND_ROUNDING)]
        out['above_explained_limit'] = len(high)
        if skips is not None:
            explained = set(skips['skip_points'])
            out['unexplained'] = sum(1 for i in high if i not in explained)
    return out


def _ms(seconds) -> Optional[float]:
    return None if seconds is None else float(seconds) * 1e3


JOG_RULE = 'commanded joint acceleration (teleop joint command)'


def jog_measures(data: dict, limit: Optional[float]) -> dict:
    """
    Return the step of the teleop node's own joint command, cycle to cycle (G-18).

    The node limits each change of its command to ``limit`` x the command period on every
    joint (ranger_xarm_teleop_servo.joint_accel). A step is ``max |v_k - v_k-1| / period`` over
    the arm joints, between consecutive JointJog commands. A step to all zeros above the limit
    is the node's immediate stop (teleop lost the arm, or Servo halted), which it announces on
    /rosout; the announcements are counted from ``data['rosout']['immediate_stops']``.
    """
    jog = stats.jog_commands(data.get('jog', []))
    steps, to_zero = [], []
    for before, after in zip(jog, jog[1:]):
        steps.append(float(np.max(np.abs(after - before))) / COMMAND_PERIOD_S)
        to_zero.append(not np.any(after))
    out = {'messages': len(data.get('jog', [])), 'commands': len(jog), 'steps': len(steps),
           'moved': any(bool(np.any(v)) for v in jog),
           'peak': max(steps) if steps else None, 'above': None, 'above_to_zero': None,
           'announced': int((data.get('rosout') or {}).get('immediate_stops', 0))}
    if limit is not None:
        bound = limit * (1.0 + COMMAND_ROUNDING)
        out['above'] = sum(1 for s, z in zip(steps, to_zero) if s > bound and not z)
        out['above_to_zero'] = sum(1 for s, z in zip(steps, to_zero) if s > bound and z)
    return out


def rule_jog_acceleration(jm: dict, opts: Options) -> Result:
    """Judge the teleop node's own command: every step within the limit, where G-18 acts."""
    name = JOG_RULE
    limit, todo = _command_limit(opts)
    threshold = (f'every step of the teleop command <= {ACCELERATION_NAME}'
                 + (f' = {limit:g} rad/s^2' if limit is not None else ' (limit not known)')
                 + f', as max over the arm joints of |change| / {COMMAND_PERIOD_S:g} s between '
                 'consecutive commands; a step to zero above it only as an immediate stop the '
                 f"node announced ('{IMMEDIATE_STOP_LOG}' on /rosout: teleop lost the arm, or "
                 'MoveIt Servo halted)')
    measured = (f"peak {_fmt(jm['peak'], '.3f')} rad/s^2 over {jm['steps']} steps of "
                f"{jm['commands']} commands on {opts.jog_topic}")
    if jm['messages'] != jm['commands']:
        measured += f"; {jm['messages'] - jm['commands']} messages without all six arm joints"
    if limit is None:
        return Result(name, NOT_JUDGED, measured, threshold, todo)
    measured += (f"; {jm['above']} steps above the limit, {jm['above_to_zero']} steps to zero "
                 f"above it, {jm['announced']} immediate stops announced")
    if not jm['moved']:
        return Result(name, NOT_JUDGED, measured, threshold,
                      f'no teleop joint command with motion on {opts.jog_topic}: engage the '
                      'clutch and move during the run (is --jog-topic the topic MoveIt Servo '
                      'reads?)')
    if jm['above']:
        return Result(name, FAIL, measured, threshold,
                      'a stop: (procedure stage 4): the teleop node changed its joint command '
                      'faster than the limit it enforces; release the grip, keep the report '
                      'and report it')
    if jm['above_to_zero'] > jm['announced']:
        return Result(name, FAIL, measured, threshold,
                      'a stop: (procedure stage 4): the joint command went to zero at once '
                      'more often than the node announced an immediate stop; keep the report '
                      'and report it')
    return Result(name, PASS, measured, threshold)


def rule_command_acceleration(cm: dict, opts: Options) -> Result:
    """Servo's output: every point's acceleration within 2 x 1.05 x the limit (a stop above)."""
    name = 'commanded joint acceleration (MoveIt Servo output)'
    bound = SKIP_FACTOR * SKIP_MARGIN
    skip = ('a Servo skip can join two consecutive command steps into one, so a single step '
            f'is accepted up to {SKIP_FACTOR:g} x the limit, with {SKIP_MARGIN:g} for '
            "Servo's point timing")
    limit, todo = _command_limit(opts)
    threshold = f'every point <= {SKIP_FACTOR:g} x {SKIP_MARGIN:g} x {ACCELERATION_NAME}'
    if limit is not None:
        threshold += f' = {bound * limit:g} rad/s^2 (limit {limit:g}'
    else:
        threshold += ' (limit not known'
    threshold += f'; {skip})'
    peaks = cm['peak_acceleration']
    shown = ', '.join(f'{j[-1]}: {_fmt(v, ".2f")}' for j, v in peaks.items())
    sp = cm['spacing_ms']
    measured = (f"peak {shown} rad/s^2 over {cm['points']} points from "
                f"{cm['topic_messages']} messages on {opts.command_topic}")
    if (limit is not None and cm['above_1x'] is not None
            and cm['peak_overall'] is not None):
        top = cm['peak_overall']
        measured += (f"; {cm['above_1x']} points above 1 x limit (joined by a skip), largest "
                     f"{_fmt(top, '.2f')} rad/s^2 = {_fmt(top / limit, '.2f')} x limit")
    measured += (f"; point spacing p50 {_fmt(sp['p50'], '.1f')} p99 {_fmt(sp['p99'], '.1f')} "
                 f"max {_fmt(sp['max'], '.1f')} ms")
    if cm['open_start'] or cm['open_end']:
        measured += ('; the capture cut the stream at its '
                     + ' and '.join(n for n, f in (('start', cm['open_start']),
                                                   ('end', cm['open_end'])) if f)
                     + ' (no rest assumed there)')
    if cm['skipped_messages']:
        measured += f"; {cm['skipped_messages']} messages without all six arm joints skipped"
    if limit is None:
        return Result(name, NOT_JUDGED, measured, threshold, todo)
    if not cm['moved']:
        return Result(name, NOT_JUDGED, measured, threshold,
                      f'no Servo output with motion on {opts.command_topic}: engage the clutch '
                      'and move during the run (is --command-topic the controller input '
                      'topic?)')
    if cm['above_2x']:
        return Result(name, FAIL, measured + f"; {cm['above_2x']} points above "
                      f'{bound:g} x limit', threshold,
                      'a stop: (procedure stage 4): the joint command accelerates faster than '
                      f'{bound:g} x the limit; release the grip, keep the report and '
                      'report it')
    return Result(name, PASS, measured, threshold)


def info_unexplained_acceleration(cm: dict, opts: Options) -> Result:
    """INFO: Servo-output points above 1.1 x the limit that no detected skip explains."""
    name = 'Servo output above 1.1 x limit without a skip (INFO)'
    threshold = (f'informational: the share of points above {EXPLAINED_ABOVE:g} x '
                 f'{ACCELERATION_NAME} that no detected Servo skip explains (a skip: a step of '
                 "Servo's matches a teleop command more than one ahead of the last, on "
                 f'{opts.jog_topic}; it explains a point only if the commands it jumped each '
                 'change within the limit), and the number of skips; the first real reports '
                 'decide whether the stop rule is tightened')
    limit, _ = _command_limit(opts)
    sk = cm.get('skips')
    if limit is None or cm.get('above_explained_limit') is None:
        return Result(name, INFO, 'not judged: the acceleration limit is not known', threshold)
    high, total = cm['above_explained_limit'], cm.get('judged_points') or 0
    if sk is None:
        return Result(name, INFO,
                      f'{high} of {total} points above {EXPLAINED_ABOVE:g} x limit; skips not '
                      f'detectable: no teleop command on {opts.jog_topic} '
                      f"({cm.get('jog_messages', 0)} messages)", threshold)
    unexplained = cm['unexplained']
    share = f'{100.0 * unexplained / high:.1f} %' if high else 'none above'
    return Result(name, INFO,
                  f'{unexplained} of {high} points above {EXPLAINED_ABOVE:g} x limit ({share}) '
                  f'without a skip that explains them, of {total} judged points; '
                  f"{sk['skips']} skips ({sk['skipped_commands']} commands skipped), "
                  f"{sk['reused']} reused, {sk['unmatched']} steps matching no command, over "
                  f"{sk['steps']} steps and {cm.get('jog_messages', 0)} commands", threshold)


def rule_stop_time(cm: dict, latency=None) -> Result:
    """After each release, Servo's output comes to rest within the stop time (plus a cycle)."""
    name = 'stop time after a release'
    bound = STOP_TIME_LIMIT_S + COMMAND_PERIOD_S
    threshold = (f'each stop <= {STOP_TIME_LIMIT_S:g} s + {COMMAND_PERIOD_S:g} s command period '
                 f'= {bound:g} s (the extra period: MoveIt Servo takes up the first ramp command '
                 'up to one cycle after the release); measured from the release to the end of '
                 "motion in Servo's output, the release moved onto Servo's point times by its "
                 f'max_expected_latency ({_fmt(latency, ".3f", " s")})')
    if not isinstance(latency, (int, float)):
        return Result(name, NOT_JUDGED, 'MoveIt Servo max_expected_latency not known',
                      threshold, "the stop cannot be placed on Servo's point times: check that "
                      'the bringup package and its config/servo.yaml are installed')
    stops = cm['stops']
    unfinished = [s for s in stops if s['unfinished']]
    durations = [s['duration_s'] for s in stops]
    measured = (f"{len(stops)} releases with motion of {cm['releases_seen']} seen"
                + (f', longest {max(durations):.3f} s; stops '
                   + ', '.join(f'{d:.3f} s' for d in durations) if stops else ''))
    if unfinished:
        measured += (f'; {len(unfinished)} stops where Servo output had not come to rest '
                     'when the stream ended (their time so far is above)')
    over = [d for d in durations if d > bound * (1.0 + COMMAND_ROUNDING)]
    if over:
        return Result(name, FAIL, measured, threshold,
                      'a stop: (procedure stage 4): the arm took longer than the stop time to '
                      'stop after a release; release the grip, keep the report and report it')
    if not [s for s in stops if not s['unfinished']]:
        if unfinished:
            return Result(name, NOT_JUDGED, measured, threshold,
                          'the run ended before Servo came to rest: release the grip and keep '
                          'recording for a second more (--seconds)')
        return Result(name, NOT_JUDGED, measured, threshold,
                      'no release with motion was seen: engage the clutch, move, then release '
                      'during the run (and pass --command-topic if it is not the default)')
    return Result(name, PASS, measured, threshold)


def joint_state_acceleration(data: dict) -> dict:
    """Return the peak acceleration per joint of the joint states on a 20 ms grid."""
    arm = data.get('arm', [])
    peaks = stats.resampled_peak_acceleration([s[2] for s in arm], [s[3] for s in arm])
    return {'peak_acceleration': dict(zip(stats.ARM_JOINTS, peaks)),
            'window_s': stats.RESAMPLE_S}


def info_joint_state_acceleration(jm: dict, opts: Options) -> Result:
    """Report the 20 ms joint-state acceleration as INFO (encoder noise is unknown)."""
    name = 'joint-state acceleration (20 ms, INFO)'
    peaks = jm['peak_acceleration']
    shown = ', '.join(f'{j[-1]}: {_fmt(v, ".2f")}' for j, v in peaks.items())
    if opts.joint_limits_given:
        values = sorted({v for v in opts.joint_acceleration_limits.values() if v is not None})
        moveit = ', '.join(f'{v:g}' for v in values) or 'none set'
    else:
        moveit = 'not given (--joint-limits)'
    return Result(name, INFO, f'{shown} rad/s^2',
                  'informational: differentiated joint states, 20 ms window; on the real arm '
                  'encoder noise adds an unknown amount; '
                  f"MoveIt's max_acceleration {moveit} for reference")


# ---- the arm's own state (vendor driver, xarm_msgs/msg/RobotMsg) -----------------------

ARM_STATE_RULE = 'arm state (xarm_msgs/RobotMsg)'
# Found by type: the driver publishes it in its hardware namespace (on the real driver
# /xarm_xarm/robot_states), so no fixed name is assumed.
ARM_STATE_TYPE = 'xarm_msgs/msg/RobotMsg'
ARM_STATE_THRESHOLD = 'err 0, mode 1 (servo), state <= 2 (1 running, 2 ready) in every message'
ARM_STATE_MODE = 1
ARM_STATE_MAX_STATE = 2
ARM_AXES = 6
ARM_ERR_ADVICE = "stop: clear the error in the vendor's software and restart from stage 2"
ARM_MODE_ADVICE = 'set servo mode (mode 1) as the procedure says'
ARM_STATE_ADVICE = 'the arm is not ready (stopped, paused or configuration changed)'


def axis_bits(value) -> str:
    """Return the low six bits of a per-axis flag as a string, joint 1 (bit 0) first."""
    return format(int(value) & ((1 << ARM_AXES) - 1), f'0{ARM_AXES}b')[::-1]


def arm_state_measures(data: dict) -> dict:
    """
    Return what the arm driver's state topic (xarm_msgs/msg/RobotMsg) showed.

    ``data['robot_states']`` is None or {'topics': {topic: [publisher names]}, 'topic': the
    topic read or None, 'unavailable': reason or None, 'messages': [(arrival s, err, warn,
    state, mode, mt_brake, mt_able)], 'searched_s': how long the type was looked for, or
    None}. ``topics`` holds every topic of that type with a
    publisher; more than one publisher in all is ambiguous, and nothing is read then.
    """
    raw = data.get('robot_states') or {}
    topics = {t: sorted(p) for t, p in sorted((raw.get('topics') or {}).items()) if p}
    messages = list(raw.get('messages') or [])
    out = {'topics': topics, 'topic': raw.get('topic'),
           'publishers': [f'{name} on {topic}' for topic, names in topics.items()
                          for name in names],
           'unavailable': raw.get('unavailable'), 'searched_s': raw.get('searched_s'),
           'messages': len(messages),
           'rate_hz': stats.period_distribution([m[0] for m in messages], None)['rate_hz'],
           'bad_err': sum(1 for m in messages if m[1] != 0),
           'bad_mode': sum(1 for m in messages if m[4] != ARM_STATE_MODE),
           'bad_state': sum(1 for m in messages if m[3] > ARM_STATE_MAX_STATE),
           'bad_messages': sum(1 for m in messages if _breaks_arm_state(m)), 'last': None}
    if messages:
        _, err, warn, state, mode, brake, able = messages[-1]
        out['last'] = {'err': err, 'warn': warn, 'state': state, 'mode': mode,
                       'mt_brake': axis_bits(brake), 'mt_able': axis_bits(able)}
    return out


def _breaks_arm_state(message) -> bool:
    return (message[1] != 0 or message[4] != ARM_STATE_MODE
            or message[3] > ARM_STATE_MAX_STATE)


def rule_arm_state(am: dict) -> Result:
    """Judge the vendor driver's err, mode and state in every message; INFO when absent."""
    name, threshold = ARM_STATE_RULE, ARM_STATE_THRESHOLD
    if len(am['publishers']) > 1:
        return Result(name, NOT_JUDGED,
                      f"{len(am['publishers'])} publishers of {ARM_STATE_TYPE}: "
                      f"{'; '.join(am['publishers'])}; none judged", threshold,
                      'which one is the arm is not known: stop every publisher of that type '
                      "but the arm's driver, and run the check again")
    if am['unavailable']:
        return Result(name, INFO, f"not available ({am['unavailable']})", threshold)
    if not am['publishers'] and not am['messages']:
        after = '' if am['searched_s'] is None else f" after {am['searched_s']:.1f} s"
        return Result(name, INFO, f'not available: no publisher of {ARM_STATE_TYPE}{after}',
                      threshold)
    if not am['messages']:
        return Result(name, NOT_JUDGED,
                      f"no message arrived from {', '.join(am['publishers'])}", threshold,
                      f'{ARM_STATE_TYPE} has a publisher but sent nothing during the run: '
                      'check that the vendor driver is connected to the arm')
    last = am['last']
    n = am['messages']
    measured = (f"{n} messages on {am['topic']} at {_hz(am['rate_hz'])}; last err "
                f"{last['err']}, warn "
                f"{last['warn']}, state {last['state']}, mode {last['mode']}; mt_brake "
                f"{last['mt_brake']}, mt_able {last['mt_able']} (6 bits, joint 1 first; "
                f"mt_brake 1 = brake on, mt_able 1 = servo enabled); {am['bad_messages']} of "
                f'{n} messages broke the rule')
    if not am['bad_messages']:
        return Result(name, PASS, measured, threshold)
    wrong, advice = [], []
    if am['bad_err']:
        wrong.append(f"err != 0 in {am['bad_err']}")
        advice.append(ARM_ERR_ADVICE + f" (last err {last['err']})")
    if am['bad_mode']:
        wrong.append(f"mode != {ARM_STATE_MODE} in {am['bad_mode']}")
        advice.append(ARM_MODE_ADVICE + f" (last mode {last['mode']})")
    if am['bad_state']:
        wrong.append(f"state > {ARM_STATE_MAX_STATE} in {am['bad_state']}")
        advice.append(ARM_STATE_ADVICE + f" (last state {last['state']})")
    return Result(name, FAIL, f"{', '.join(wrong)} of {n} messages; {measured}", threshold,
                  '; '.join(advice))


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


# ---- teleop: where velocity reversals arise, the reported velocity's own noise, step times ----

REVERSALS_INFO = 'velocity reversals: teleop command, Servo output, joint states (INFO)'
STILL_NOISE_INFO = 'reported joint velocity with the arm still (INFO)'
KEEPOUT_STEP_INFO = 'teleop step times, link keep-out by self pair (INFO)'
# The teleop node logs its summary this often while commanding (CYCLE_REPORT_S in teleop_node).
TELEOP_SUMMARY_PERIOD_S = 60


def _count(n: int, one: str, many: str) -> str:
    return f'{n} {one if n == 1 else many}'


def _joint_label(j: int) -> str:
    return f'joint {j + 1}'


def reversal_measures(data: dict, mm: dict) -> dict:
    """
    Return the reversal counts of the command, MoveIt Servo's output and the joint states.

    The counting is the wrist oscillation rule's (``stats.sign_changes_per_window``, |v| above
    OSCILLATION_SPEED, most in any OSCILLATION_WINDOW s) on joints 4 and 6, applied to the
    teleop node's joint-jog velocities (at their arrival) and to Servo's output (the velocity
    of its point, or the central difference of its positions); ``mm`` is ``motion_measures``,
    whose joint-state counts are shown for comparison.
    """
    jog_t, jog_v = stats.jog_series(data.get('jog', []), data.get('jog_arrival', []))
    series = stats.command_velocity_series(data.get('command', []),
                                           data.get('command_velocities', []))

    def counts(t, v):
        return {stats.ARM_JOINTS[j]: stats.sign_changes_per_window(
            t, v[:, j], OSCILLATION_SPEED, OSCILLATION_WINDOW) for j in WRIST_JOINTS}

    osc = mm.get('wrist_sign_changes_per_s') or {}
    return {'jog_messages': len(data.get('jog', [])), 'jog_usable': int(len(jog_t)),
            'command_messages': len(data.get('command', [])),
            'command_source': series['source'],
            'command': counts(jog_t, jog_v) if len(jog_t) else {},
            'servo': counts(series['t'], series['v']) if series['source'] else {},
            'joint_states': dict(osc),
            'joint_states_source': mm.get('velocity_source')}


def info_reversals_command_servo(rm: dict, opts: Options) -> Result:
    """INFO: velocity reversals per second window on joints 4 and 6, in three places."""
    threshold = (f'informational: sign flips of the velocity above {OSCILLATION_SPEED:g} rad/s, '
                 f'most in any {OSCILLATION_WINDOW:g} s window, counted as the wrist '
                 f'oscillation rule counts them (limit {OSCILLATION_LIMIT} there, on '
                 '/joint_states only); shows where reversals arise')

    def count(source, j):
        value = rm[source].get(stats.ARM_JOINTS[j])
        return 'n/a' if value is None else str(value)

    shown = '; '.join(
        f'{_joint_label(j)}: command {count("command", j)}, Servo output {count("servo", j)}, '
        f'joint states {count("joint_states", j)}' for j in WRIST_JOINTS)
    notes = []
    if not rm['jog_messages']:
        notes.append(f'command: no message on {opts.jog_topic}')
    elif not rm['jog_usable']:
        notes.append(f"command: {rm['jog_messages']} messages on {opts.jog_topic}, none with "
                     'all six arm joints and a time')
    else:
        notes.append(f"command: joint-jog velocities of {rm['jog_usable']} messages on "
                     f'{opts.jog_topic}, at arrival')
    if not rm['command_messages']:
        notes.append(f'Servo output: no message on {opts.command_topic}')
    elif rm['command_source'] == 'velocities':
        notes.append(f"Servo output: velocities of the point of {rm['command_messages']} "
                     f'messages on {opts.command_topic}')
    elif rm['command_source'] == 'positions':
        notes.append(f'Servo output: no velocities in the points, central difference of the '
                     f"positions of {rm['command_messages']} messages on {opts.command_topic} "
                     'over their stamps')
    else:
        notes.append(f"Servo output: {rm['command_messages']} messages on "
                     f'{opts.command_topic}, none with all six arm joints')
    if not rm['joint_states']:
        notes.append('joint states: no six-joint message')
    else:
        notes.append(f"joint states: velocity from {rm['joint_states_source']}")
    return Result(REVERSALS_INFO, INFO, f'{shown} (most sign changes in any '
                  f"{OSCILLATION_WINDOW:g} s window; {'; '.join(notes)})", threshold)


def still_noise_measures(data: dict) -> dict:
    """
    Return the joint-state velocity noise of joints 4 and 6 while the arm is still.

    Still is ``stats.still_windows``: no nonzero joint-jog command (any joint of any message
    on the jog topic) for ``STILL_SPAN_S`` before and during, and the joint moving less than
    ``STILL_MOVE_RAD`` over each such span. Velocity is as the driver reports it; with it
    missing from any message nothing further is measured. Times are arrival times, the clock
    the jog messages are timed on too.
    """
    arm = data.get('arm', [])
    out = {'messages': len(arm), 'reported': bool(arm) and all(s[4] is not None for s in arm),
           'jog_messages': len(data.get('jog', [])), 'joints': {}}
    if not out['reported']:
        return out
    t = np.asarray([s[0] for s in arm], dtype=float)
    pos = np.asarray([s[3] for s in arm], dtype=float)
    vel = np.asarray([s[4] for s in arm], dtype=float)
    jog = data.get('jog', [])
    arrivals = data.get('jog_arrival', [])
    commanded = ([a for (_names, v), a in zip(jog, arrivals) if any(x != 0.0 for x in v)]
                 if len(arrivals) == len(jog) else [])
    for j in WRIST_JOINTS:
        windows = stats.still_windows(t, pos[:, j], commanded)
        out['joints'][stats.ARM_JOINTS[j]] = stats.still_noise(
            t, vel[:, j], windows, OSCILLATION_SPEED, OSCILLATION_WINDOW)
    return out


def info_still_velocity_noise(sm: dict) -> Result:
    """INFO: RMS, peak and sign changes of the reported velocity over the arm's still time."""
    threshold = (f'informational: still = no nonzero joint-jog command for {stats.STILL_SPAN_S:g} '
                 f's before and during, and the joint position within {stats.STILL_MOVE_RAD:g} '
                 f'rad over each {stats.STILL_SPAN_S:g} s span, in stretches of at least '
                 f'{stats.STILL_MIN_S:g} s; shows how much of a reversal count the reported '
                 'velocity alone can make')
    if not sm['messages']:
        return Result(STILL_NOISE_INFO, INFO, 'no six-joint message', threshold)
    if not sm['reported']:
        return Result(STILL_NOISE_INFO, INFO,
                      'the driver reports no velocity in /joint_states (or not in every '
                      'message): nothing measured', threshold)
    parts = []
    for name, s in sm['joints'].items():
        label = _joint_label(stats.ARM_JOINTS.index(name))
        if not s['windows']:
            parts.append(f'{label}: no still window of at least '
                         f'{stats.STILL_MIN_S:g} s')
            continue
        parts.append(f"{label}: still {s['seconds']:.1f} s in "
                     f"{_count(s['windows'], 'window', 'windows')}, RMS {s['rms']:.4f} rad/s, "
                     f"peak {s['peak']:.4f} rad/s, most "
                     f'sign changes above {OSCILLATION_SPEED:g} rad/s in any '
                     f"{OSCILLATION_WINDOW:g} s window {s['sign_changes']}")
    jog = (f"{sm['jog_messages']} joint-jog messages" if sm['jog_messages']
           else 'no joint-jog message')
    return Result(STILL_NOISE_INFO, INFO, '; '.join(parts) + f' ({jog} in the run)', threshold)


def keepout_step_measures(data: dict) -> dict:
    """Return the teleop node's logged step-time summaries, parsed (stats.parse_teleop_summary)."""
    parsed = [stats.parse_teleop_summary(text)
              for text in (data.get('rosout') or {}).get('teleop_summaries', [])]
    halves = {}
    for key in ('active', 'inactive'):
        rows = [p['pair'][key] for p in parsed if p['pair']]
        medians = [r['median_ms'] for r in rows if r['median_ms'] is not None]
        p99s = [r['p99_ms'] for r in rows if r['p99_ms'] is not None]
        longest = [r['longest_ms'] for r in rows if r['longest_ms'] is not None]
        halves[key] = {'summaries': len(rows), 'steps': sum(r['steps'] for r in rows),
                       'median_of_medians_ms': stats.median(medians),
                       'max_p99_ms': max(p99s) if p99s else None,
                       'longest_ms': max(longest) if longest else None}
    cycles = [p['cycles'] for p in parsed if p['cycles']]
    p99s = [c['p99_ms'] for c in cycles if c['p99_ms'] is not None]
    longest = [c['longest_ms'] for c in cycles if c['longest_ms'] is not None]
    medians = [c['median_ms'] for c in cycles if c['median_ms'] is not None]
    return {'summaries': len(parsed), 'pair': halves,
            'cycles': {'summaries': len(cycles), 'cycles': sum(c['cycles'] for c in cycles),
                       'median_of_medians_ms': stats.median(medians),
                       'max_p99_ms': max(p99s) if p99s else None,
                       'longest_ms': max(longest) if longest else None,
                       'over': sum(c['over'] or 0 for c in cycles)}}


def info_keepout_step_time(km: dict) -> Result:
    """INFO: step times of the link keep-out split by self pair active, and the cycle times."""
    threshold = ('informational: "self pair active" means a pair of the arm\'s own links '
                 'on the sphere covers (three or more joints apart) gave a keep-out row; '
                 'totals over every summary of the run; the median is the median of the '
                 'per-summary medians, p99 and longest the largest seen')
    pair, cycles = km['pair'], km['cycles']
    none = ('no keep-out summary from the teleop node in this run (it logs one every '
            f'{TELEOP_SUMMARY_PERIOD_S} s while commanding)')
    if not pair['active']['summaries'] and not cycles['summaries']:
        return Result(KEEPOUT_STEP_INFO, INFO, none, threshold)

    def half(label, h):
        if not h['steps']:
            return f'{label} 0 steps'
        return (f"{label} {h['steps']} steps, median {_fmt(h['median_of_medians_ms'], '.2f')} "
                f"ms, largest p99 {_fmt(h['max_p99_ms'], '.2f')} ms, longest "
                f"{_fmt(h['longest_ms'], '.2f')} ms")

    parts = []
    if pair['active']['summaries']:
        parts.append('link keep-out by self pair '
                     f"({_count(pair['active']['summaries'], 'summary', 'summaries')}): "
                     + half('active', pair['active']) + '; '
                     + half('not active', pair['inactive']))
    else:
        parts.append(none)
    if cycles['summaries']:
        parts.append('teleop cycle computation '
                     f"({_count(cycles['summaries'], 'summary', 'summaries')}): "
                     f"{cycles['cycles']} cycles, median "
                     f"{_fmt(cycles['median_of_medians_ms'], '.2f')} ms, largest p99 "
                     f"{_fmt(cycles['max_p99_ms'], '.2f')} ms, longest "
                     f"{_fmt(cycles['longest_ms'], '.2f')} ms, {cycles['over']} over the period")
    else:
        parts.append('no cycle-computation summary')
    return Result(KEEPOUT_STEP_INFO, INFO, '; '.join(parts), threshold)


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
    ((stamp s, (x, y, z))), ``command`` (teleop: (arrival, header stamp ns, joint names,
    [(time_from_start s, positions)]) of the Servo output), ``command_capture``
    ({'began', 'ended'} on the arrival clock), ``clutch`` ((header stamp ns, clutch_engaged)
    of /teleop/state), ``rosout`` (counts and examples), ``machine``, ``versions``,
    ``gate`` (driver_gate), ``robot_states`` (idle: None or {topics, topic, unavailable,
    messages of (arrival s, err, warn, state, mode, mt_brake, mt_able), searched_s}), ``cameras``
    ({stream: {mono, stamps, first}}), ``tracking`` ((clutch_engaged, active_limit,
    keepout_bound, keepout_clearance_commanded, keepout_margin, keepout_link,
    keepout_obstacle, lead_translation, lead_rotation) of /teleop/state); teleop only, beside
    ``command`` and ``jog``: ``command_velocities`` (per command message, the velocities of
    each point), ``jog_arrival`` (arrival monotonic s of each ``jog`` message), and in
    ``rosout`` ``teleop_summaries`` (the teleop node's step-time log messages).
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
    if stage == 'idle':
        measures['arm_state'] = arm_state_measures(data)
        results.append(rule_arm_state(measures['arm_state']))
    if stage in ('motion', 'teleop'):
        mm = motion_measures(data)
        measures['motion'] = mm
        if stage == 'motion':
            results += [rule_stale(mm), rule_acceleration(mm, opts), rule_oscillation(mm)]
        else:
            jm = joint_state_acceleration(data)
            measures['joint_state_acceleration_20ms'] = jm
            results += [rule_stale(mm), info_joint_state_acceleration(jm, opts),
                        rule_oscillation(mm)]
    if stage == 'teleop':
        tm = teleop_measures(data)
        measures['teleop'] = tm
        results += rules_lead(tm, opts)
        latency = opts.servo_latency_s
        cm = command_measures(data, _command_limit(opts)[0],
                              latency if isinstance(latency, (int, float)) else 0.0)
        measures['servo_command'] = cm
        jm = jog_measures(data, _command_limit(opts)[0])
        measures['teleop_command'] = jm
        results += [rule_jog_acceleration(jm, opts), rule_command_acceleration(cm, opts),
                    info_unexplained_acceleration(cm, opts), rule_stop_time(cm, latency)]
        km = tracking_measures(data)
        measures['tracking'] = km
        results += [info_keepout_hold(km), info_rotation_lead(km)]
        rm = reversal_measures(data, mm)
        measures['reversals'] = rm
        sm = still_noise_measures(data)
        measures['still_velocity_noise'] = sm
        ks = keepout_step_measures(data)
        measures['keepout_step_time'] = ks
        results += [info_reversals_command_servo(rm, opts), info_still_velocity_noise(sm),
                    info_keepout_step_time(ks)]
    return measures, results
