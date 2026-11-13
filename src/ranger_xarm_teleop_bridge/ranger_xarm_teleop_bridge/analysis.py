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
Statistics over a JSONL recording of relay text, for the head-turn and occlusion tests.

Pure: no ROS imports. Everything is computed on the raw WebXR values (spread and angles
are invariant under the axis conversion). Positions in metres (reported in mm), angles
in degrees, times in seconds relative to the first record unless stated.

The per-hand statistics use every source the headset reported for that hand, including
one that looks like a tracked hand (which the bridge publishes as not present): the
``sources`` section says what each source was and warns when it is not a controller.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ranger_xarm_teleop_bridge.frame_parse import (HandSample, Ignored, is_controller, Malformed,
                                                   parse_text, ParsedFrame, ViewerSample)
from ranger_xarm_teleop_bridge.jsonl_io import Record

PASS_POSITION_MM = 10.0   # head-turn pass line (human-approved): max deviation from the mean
PASS_ANGLE_DEG = 2.0


@dataclass(frozen=True)
class Sample:
    """One parsed frame with its receive time."""

    t_recv: float
    frame: ParsedFrame


# ---- small quaternion helpers (xyzw) ----

def _angle_between(q1: Sequence[float], q2: Sequence[float]) -> float:
    """Return the rotation angle in degrees between two unit xyzw quaternions."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _mean_quaternion(quats: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Return the hemisphere-aligned normalised average of unit xyzw quaternions."""
    ref = quats[0]
    acc = [0.0, 0.0, 0.0, 0.0]
    for q in quats:
        sign = -1.0 if sum(a * b for a, b in zip(q, ref)) < 0 else 1.0
        for i in range(4):
            acc[i] += sign * q[i]
    norm = math.sqrt(sum(a * a for a in acc))
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (acc[0] / norm, acc[1] / norm, acc[2] / norm, acc[3] / norm)


def _qmul(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _qconj(q: Sequence[float]) -> Tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def _rotate(q: Sequence[float], v: Sequence[float]) -> Tuple[float, float, float]:
    """Rotate vector ``v`` by unit xyzw quaternion ``q``."""
    out = _qmul(_qmul(q, (v[0], v[1], v[2], 0.0)), _qconj(q))
    return (out[0], out[1], out[2])


def _in_viewer_frame(hand: HandSample,
                     viewer: ViewerSample) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return the hand pose expressed in the headset frame (position and xyzw orientation)."""
    inv = _qconj(viewer.orientation)
    delta = tuple(hand.position[i] - viewer.position[i] for i in range(3))
    return _rotate(inv, delta), _qmul(inv, hand.orientation)


def _spread(positions: Sequence[Sequence[float]],
            orientations: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Return the position and orientation spread of a set of poses."""
    n = len(positions)
    mean = [sum(p[i] for p in positions) / n for i in range(3)]
    std = [math.sqrt(sum((p[i] - mean[i]) ** 2 for p in positions) / n) for i in range(3)]
    p2p = [max(p[i] for p in positions) - min(p[i] for p in positions) for i in range(3)]
    max_dev = max(math.dist(p, mean) for p in positions)
    q_mean = _mean_quaternion(orientations)
    angles = [_angle_between(q, q_mean) for q in orientations]
    return {
        'tracked_frames': n,
        'mean_position_m': mean,
        'std_mm': [s * 1000.0 for s in std],
        'peak_to_peak_mm': [p * 1000.0 for p in p2p],
        'max_deviation_mm': max_dev * 1000.0,
        'mean_orientation_xyzw': list(q_mean),
        'max_angle_deg': max(angles),
        'rms_angle_deg': math.sqrt(sum(a * a for a in angles) / n),
    }


def _viewer_yaw_deg(q: Sequence[float]) -> float:
    """Return the heading of the viewer's forward axis (-z) about WebXR +y, in degrees."""
    x, y, z, w = q
    # forward = R @ (0, 0, -1)
    fx = -(2 * (x * z + w * y))
    fz = -(1 - 2 * (x * x + y * y))
    return math.degrees(math.atan2(-fx, -fz))


def _unwrap_deg(angles: List[float]) -> List[float]:
    out: List[float] = []
    offset = 0.0
    for i, a in enumerate(angles):
        if i > 0:
            d = a - angles[i - 1]
            if d > 180:
                offset -= 360
            elif d < -180:
                offset += 360
        out.append(a + offset)
    return out


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return float('nan')
    k = (len(sorted_values) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def _intervals(times: List[float], flags: List[bool], end_time: float) -> List[Dict[str, float]]:
    """Return [{'start','end','duration'}] for every run of consecutive True flags."""
    out: List[Dict[str, float]] = []
    start: Optional[float] = None
    for t, f in zip(times, flags):
        if f and start is None:
            start = t
        elif not f and start is not None:
            out.append({'start': start, 'end': t, 'duration': t - start})
            start = None
    if start is not None:
        out.append({'start': start, 'end': end_time, 'duration': end_time - start})
    return out


def _sources(samples: List[Sample]) -> Dict[str, Any]:
    """Summarise, per side, every distinct kind of input source the headset reported."""
    out: Dict[str, Any] = {'left': [], 'right': [], 'warnings': []}
    for side in ('left', 'right'):
        kinds: Dict[Tuple[Any, ...], int] = {}
        for s in samples:
            info = getattr(s.frame, f'{side}_source')
            if info is None:
                continue
            key = (info.profiles, info.button_count, info.axis_count, info.client_hand_tracked)
            kinds[key] = kinds.get(key, 0) + 1
        ranked = sorted(kinds.items(), key=lambda kv: -kv[1])
        for (profiles, buttons, axes, flagged), count in ranked:
            controller = (not flagged and axes >= 4
                          and not any('hand' in p.lower() for p in profiles))
            out[side].append({'profiles': list(profiles), 'buttons': buttons, 'axes': axes,
                              'client_hand_tracked': flagged, 'frames': count,
                              'controller': controller})
            if not controller:
                why = ('flagged hand_tracked by the client' if flagged
                       else f'{buttons} buttons, {axes} axes, profiles {list(profiles)}')
                out['warnings'].append(f'{side}: {count} frames look like a tracked hand ({why}); '
                                       f'the bridge publishes them as not present')
    return out


def _reset_part(frame: ParsedFrame, name: str) -> Optional[float]:
    """Return one reported reset movement (distance_m or angle_deg); None if not reported."""
    return None if frame.reset_event is None else getattr(frame.reset_event, name)


def _gamepad_activity(hand_samples: List[Optional[HandSample]]) -> Dict[str, Any]:
    """Per button index: pressed/touched frames and max value; per axis: min and max."""
    buttons: List[Dict[str, Any]] = []
    axes: List[Dict[str, Any]] = []
    for h in hand_samples:
        if h is None:
            continue
        for i, b in enumerate(h.buttons):
            while len(buttons) <= i:
                buttons.append({'index': len(buttons), 'pressed_frames': 0, 'touched_frames': 0,
                                'max_value': 0.0})
            buttons[i]['pressed_frames'] += int(b.pressed)
            buttons[i]['touched_frames'] += int(b.touched)
            buttons[i]['max_value'] = max(buttons[i]['max_value'], b.value)
        for i, a in enumerate(h.axes):
            while len(axes) <= i:
                axes.append({'index': len(axes), 'min': a, 'max': a, 'nonzero_frames': 0})
            axes[i]['min'] = min(axes[i]['min'], a)
            axes[i]['max'] = max(axes[i]['max'], a)
            axes[i]['nonzero_frames'] += int(abs(a) > 0.01)
    return {'buttons': buttons, 'axes': axes}


# ---- the analysis ----

def load_samples(records: Iterable[Record]) -> Tuple[List[Sample], Dict[str, Any]]:
    """Parse records; return the frames and the counts of everything that was not a frame."""
    samples: List[Sample] = []
    counts: Dict[str, Any] = {'records': 0, 'frames': 0, 'malformed': 0, 'raw_lines': 0,
                              'ignored': 0, 'ignored_types': {}, 'malformed_reasons': []}
    for rec in records:
        counts['records'] += 1
        if rec.raw is not None:
            counts['raw_lines'] += 1
        result = parse_text(rec.text())
        if isinstance(result, ParsedFrame):
            samples.append(Sample(rec.t_recv, result))
            counts['frames'] += 1
        elif isinstance(result, Ignored):
            counts['ignored'] += 1
            counts['ignored_types'][result.type] = counts['ignored_types'].get(result.type, 0) + 1
        elif isinstance(result, Malformed):
            counts['malformed'] += 1
            if len(counts['malformed_reasons']) < 20:
                counts['malformed_reasons'].append(result.reason)
    return samples, counts


def analyze(records: Iterable[Record], hand: str = 'right',
            t_from: Optional[float] = None, t_to: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyse a recording for one hand; return a JSON-serialisable report.

    ``t_from``/``t_to`` select a window in seconds relative to the first record (used to
    pick the part of a recording where the controller was physically still).
    """
    if hand not in ('left', 'right'):
        raise ValueError(f'hand must be left or right, got {hand!r}')
    all_samples, counts = load_samples(records)
    report: Dict[str, Any] = {'hand': hand, 'counts': counts,
                              'window': {'from': t_from, 'to': t_to}}
    if not all_samples:
        report['error'] = 'no valid frames'
        return report
    t0 = all_samples[0].t_recv
    samples = [s for s in all_samples
               if (t_from is None or s.t_recv - t0 >= t_from)
               and (t_to is None or s.t_recv - t0 <= t_to)]
    if not samples:
        report['error'] = 'no frames in the window'
        return report
    times = [s.t_recv - t0 for s in samples]
    end_time = times[-1]
    report['duration_s'] = end_time - times[0]
    report['frames_in_window'] = len(samples)
    span = end_time - times[0]
    report['mean_rate_hz'] = (len(samples) - 1) / span if span > 0 else 0.0

    # Gaps between consecutive frames, by receive time and by headset frame time.
    gaps_recv = [(times[i] - times[i - 1], times[i]) for i in range(1, len(samples))]
    gaps_xr = [((samples[i].frame.xr_time_ms - samples[i - 1].frame.xr_time_ms) / 1000.0,
                times[i]) for i in range(1, len(samples))]
    longest_recv = max(gaps_recv, default=(0.0, 0.0))
    longest_xr = max(gaps_xr, default=(0.0, 0.0))
    report['longest_gap'] = {'t_recv_s': longest_recv[0], 't_recv_at': longest_recv[1],
                             't_xr_s': longest_xr[0], 't_xr_at': longest_xr[1]}
    seq_jumps = [(samples[i - 1].frame.seq, samples[i].frame.seq, times[i])
                 for i in range(1, len(samples))
                 if samples[i].frame.seq != samples[i - 1].frame.seq + 1]
    report['seq'] = {
        'first': samples[0].frame.seq, 'last': samples[-1].frame.seq,
        'missing': sum(max(0, b - a - 1) for a, b, _ in seq_jumps),
        'jumps': [{'from': a, 'to': b, 'at': t} for a, b, t in seq_jumps[:50]],
    }

    # Reference-space resets.
    # The movement comes from the frame's reset_event when the headset reported one.
    resets = [{'at': times[i], 'seq': samples[i].frame.seq,
               'from': samples[i - 1].frame.reset_count, 'to': samples[i].frame.reset_count,
               'distance_m': _reset_part(samples[i].frame, 'distance_m'),
               'angle_deg': _reset_part(samples[i].frame, 'angle_deg')}
              for i in range(1, len(samples))
              if samples[i].frame.reset_count != samples[i - 1].frame.reset_count]
    report['resets'] = {'count': len(resets), 'events': resets,
                        'final_reset_count': samples[-1].frame.reset_count}

    # XR session changes (Start Teleop, page reload): quest_world was redefined.
    sessions = [{'at': times[i], 'seq': samples[i].frame.seq,
                 'from': samples[i - 1].frame.session_id, 'to': samples[i].frame.session_id}
                for i in range(1, len(samples))
                if samples[i].frame.session_id != samples[i - 1].frame.session_id]
    report['sessions'] = {'count': len(sessions), 'events': sessions,
                          'first_session_id': samples[0].frame.session_id,
                          'final_session_id': samples[-1].frame.session_id}

    # Presence and emulation intervals for the hand.
    hand_samples = [getattr(s.frame, hand) for s in samples]
    absent = [h is None for h in hand_samples]
    emulated = [h is not None and h.emulated_position for h in hand_samples]
    absent_intervals = _intervals(times, absent, end_time)
    emulated_intervals = _intervals(times, emulated, end_time)
    report['absent_intervals'] = absent_intervals
    report['emulated_intervals'] = emulated_intervals
    report['present_frames'] = sum(1 for a in absent if not a)

    # Pose jump at re-acquisition: after every interval in which the pose was not tracked
    # (absent or emulated), compare the last tracked sample before it with the first
    # tracked sample after it. Emulated samples never serve as reference.
    tracked_flags = [h is not None and not h.emulated_position for h in hand_samples]
    untracked_intervals = _intervals(times, [not t for t in tracked_flags], end_time)
    jumps = []
    for iv in untracked_intervals:
        before = next((hand_samples[i] for i in range(len(samples) - 1, -1, -1)
                       if times[i] < iv['start'] and tracked_flags[i]), None)
        after = next((hand_samples[i] for i in range(len(samples))
                      if times[i] >= iv['end'] and tracked_flags[i]), None)
        if before is not None and after is not None:
            dist_mm = math.dist(before.position, after.position) * 1000.0
            angle = _angle_between(before.orientation, after.orientation)
            jumps.append({'untracked_from': iv['start'], 'reacquired_at': iv['end'],
                          'position_jump_mm': dist_mm, 'angle_jump_deg': angle,
                          'pass': dist_mm < PASS_POSITION_MM and angle < PASS_ANGLE_DEG})
    report['untracked_intervals'] = untracked_intervals
    report['reacquisition_jumps'] = jumps

    # Still-controller spread over tracked (present, not emulated) frames, in the room
    # (quest_world) and, for comparison, relative to the headset. A room-fixed pose has a
    # small room spread and a headset spread that grows with the head motion; a pose that
    # follows the head shows the opposite.
    # The PASS/FAIL verdict against the still-controller line only makes sense over a window
    # the caller chose as the still part (``t_from``/``t_to``); over a whole recording the hand
    # usually moves at some point, so without a window the numbers are reported and
    # ``pass_position``/``pass_angle`` stay None.
    windowed = t_from is not None or t_to is not None
    tracked = [h for h, flag in zip(hand_samples, tracked_flags) if flag]
    if tracked:
        report['spread'] = _spread([h.position for h in tracked],
                                   [h.orientation for h in tracked])
        report['spread']['windowed'] = windowed
        report['spread']['pass_position'] = (
            report['spread']['max_deviation_mm'] < PASS_POSITION_MM if windowed else None)
        report['spread']['pass_angle'] = (
            report['spread']['max_angle_deg'] < PASS_ANGLE_DEG if windowed else None)
        in_viewer = [_in_viewer_frame(h, s.frame.viewer)
                     for h, flag, s in zip(hand_samples, tracked_flags, samples)
                     if flag and s.frame.viewer is not None]
        report['spread_in_headset'] = (_spread([p for p, _ in in_viewer],
                                               [q for _, q in in_viewer])
                                       if in_viewer else None)
    else:
        report['spread'] = None
        report['spread_in_headset'] = None

    # What the headset said about each input source, and the gamepad activity per index.
    report['sources'] = _sources(samples)
    report['gamepad'] = _gamepad_activity(hand_samples)
    report['bridge_not_present_frames'] = sum(
        1 for h in hand_samples if h is not None and not is_controller(h))

    # Viewer motion (did the head actually turn?).
    viewers = [s.frame.viewer for s in samples if s.frame.viewer is not None]
    if viewers:
        yaws = _unwrap_deg([_viewer_yaw_deg(v.orientation) for v in viewers])
        pos = [v.position for v in viewers]
        report['viewer'] = {
            'frames': len(viewers),
            'yaw_span_deg': max(yaws) - min(yaws),
            'position_range_m': [max(p[i] for p in pos) - min(p[i] for p in pos)
                                 for i in range(3)],
            'emulated_frames': sum(1 for v in viewers if v.emulated_position),
        }
    else:
        report['viewer'] = None

    # Transport timing: t_recv (unix ms) minus t_client (headset ms). The minimum is the
    # headset-to-ROS clock offset estimate; the rest is jitter above it.
    deltas = sorted(s.t_recv * 1000.0 - s.frame.client_time_ms for s in samples)
    offset = deltas[0]
    jitter = [d - offset for d in deltas]
    report['timing'] = {
        'clock_offset_ms': offset,
        'jitter_ms': {'p50': _percentile(jitter, 0.5), 'p90': _percentile(jitter, 0.9),
                      'p99': _percentile(jitter, 0.99), 'max': jitter[-1],
                      'mean': sum(jitter) / len(jitter)},
        'xr_minus_client_ms_mean': (sum(s.frame.xr_time_ms - s.frame.client_time_ms
                                        for s in samples) / len(samples)),
    }
    return report


def _verdict(passed: Optional[bool], line: str) -> str:
    """Bracketed PASS/FAIL against ``line``, or the no-window note when there is no verdict."""
    if passed is None:
        return '[no verdict without a window]'
    return f"[{'PASS' if passed else 'FAIL'} {line}]"


def format_report(report: Dict[str, Any]) -> str:
    """Render a report as a readable text table."""
    lines: List[str] = []
    c = report['counts']
    w = report['window']
    lines.append(f"hand: {report['hand']}   window: {w['from']} .. {w['to']} s")
    lines.append(f"records {c['records']}  frames {c['frames']}  malformed {c['malformed']}  "
                 f"raw lines {c['raw_lines']}  ignored {c['ignored']} "
                 f"{c['ignored_types'] or ''}")
    if 'error' in report:
        lines.append(f"error: {report['error']}")
        return '\n'.join(lines)
    lines.append(f"duration {report['duration_s']:.2f} s  "
                 f"frames in window {report['frames_in_window']}  "
                 f"mean rate {report['mean_rate_hz']:.1f} Hz")
    g = report['longest_gap']
    lines.append(f"longest gap: {g['t_recv_s'] * 1000:.1f} ms by t_recv "
                 f"(at {g['t_recv_at']:.2f} s), {g['t_xr_s'] * 1000:.1f} ms by t_xr "
                 f"(at {g['t_xr_at']:.2f} s)")
    s = report['seq']
    lines.append(f"seq {s['first']}..{s['last']}  missing {s['missing']}  jumps {len(s['jumps'])}")
    r = report['resets']
    lines.append(f"reference-space resets: {r['count']}  "
                 f"final reset_count {r['final_reset_count']}")
    for ev in r['events']:
        moved = ('movement not reported' if ev['distance_m'] is None
                 else f"moved {ev['distance_m']:.3f} m, rotated {ev['angle_deg']:.2f} deg")
        lines.append(f"  reset at {ev['at']:.2f} s (seq {ev['seq']}): {ev['from']} -> {ev['to']}, "
                     f'{moved}')
    se = report['sessions']
    lines.append(f"XR session changes: {se['count']}  "
                 f"session_id {se['first_session_id']} .. {se['final_session_id']}")
    for ev in se['events']:
        lines.append(f"  new session at {ev['at']:.2f} s (seq {ev['seq']}): "
                     f"{ev['from']} -> {ev['to']}")
    for side in ('left', 'right'):
        for src in report['sources'][side]:
            kind = 'controller' if src['controller'] else 'HAND-LIKE'
            lines.append(f"source {side}: {src['frames']} frames, {src['buttons']} buttons, "
                         f"{src['axes']} axes, profiles {src['profiles']}"
                         f"{' (client: hand_tracked)' if src['client_hand_tracked'] else ''}"
                         f' -> {kind}')
    for warning in report['sources']['warnings']:
        lines.append(f'WARNING {warning}')
    lines.append(f"present frames {report['present_frames']}  "
                 f"absent intervals {len(report['absent_intervals'])}  "
                 f"emulated intervals {len(report['emulated_intervals'])}  "
                 f"not present for the bridge (hand-like) {report['bridge_not_present_frames']}")
    for iv in report['absent_intervals']:
        lines.append(f"  absent   {iv['start']:.2f} .. {iv['end']:.2f} s "
                     f"({iv['duration']:.2f} s)")
    for iv in report['emulated_intervals']:
        lines.append(f"  emulated {iv['start']:.2f} .. {iv['end']:.2f} s "
                     f"({iv['duration']:.2f} s)")
    for j in report['reacquisition_jumps']:
        verdict = 'PASS' if j['pass'] else 'FAIL'
        lines.append(f"  re-acquired at {j['reacquired_at']:.2f} s "
                     f"(untracked since {j['untracked_from']:.2f} s): "
                     f"position jump {j['position_jump_mm']:.1f} mm, "
                     f"angle jump {j['angle_jump_deg']:.2f} deg  "
                     f'[{verdict} < {PASS_POSITION_MM:g} mm / {PASS_ANGLE_DEG:g} deg]')
    sp = report['spread']
    if sp:
        if sp['windowed']:
            lines.append(f"spread over {sp['tracked_frames']} tracked frames in the window "
                         '(still-controller verdict):')
        else:
            lines.append(f"spread over {sp['tracked_frames']} tracked frames "
                         '(no verdict without a window: give the still part with --from/--to):')
        std = sp['std_mm']
        p2p = sp['peak_to_peak_mm']
        lines.append(f'  position std (mm) x/y/z: {std[0]:.2f} / {std[1]:.2f} / {std[2]:.2f}   '
                     f'peak-to-peak (mm): {p2p[0]:.2f} / {p2p[1]:.2f} / {p2p[2]:.2f}')
        lines.append(f"  max deviation from mean: {sp['max_deviation_mm']:.2f} mm  "
                     f'{_verdict(sp["pass_position"], f"< {PASS_POSITION_MM:g} mm")}')
        sh = report['spread_in_headset']
        if sh:
            lines.append(f"  relative to the headset: max deviation {sh['max_deviation_mm']:.2f} "
                         f"mm, std (mm) x/y/z: {sh['std_mm'][0]:.2f} / {sh['std_mm'][1]:.2f} / "
                         f"{sh['std_mm'][2]:.2f}, max angle {sh['max_angle_deg']:.2f} deg "
                         '(room-fixed pose: room spread small, headset spread follows the head)')
        lines.append(f"  orientation: max {sp['max_angle_deg']:.3f} deg, "
                     f"rms {sp['rms_angle_deg']:.3f} deg from mean  "
                     f'{_verdict(sp["pass_angle"], f"< {PASS_ANGLE_DEG:g} deg")}')
    else:
        lines.append('spread: no tracked frames for this hand')
    v = report['viewer']
    if v:
        pr = v['position_range_m']
        lines.append(f"viewer: {v['frames']} frames, yaw span {v['yaw_span_deg']:.1f} deg, "
                     f'position range (m) {pr[0]:.2f} / {pr[1]:.2f} / {pr[2]:.2f}, '
                     f"emulated frames {v['emulated_frames']}")
    t = report['timing']
    j = t['jitter_ms']
    lines.append(f"timing: clock offset (min t_recv - t_client) {t['clock_offset_ms']:.1f} ms; "
                 f"jitter above it p50 {j['p50']:.2f}  p90 {j['p90']:.2f}  p99 {j['p99']:.2f}  "
                 f"max {j['max']:.2f} ms (mean {j['mean']:.2f}); "
                 f"t_xr - t_client mean {t['xr_minus_client_ms_mean']:.2f} ms")
    gp = report['gamepad']
    if gp['buttons'] or gp['axes']:
        lines.append('gamepad activity (index: pressed frames / touched frames / max value):')
        lines.append('  buttons ' + '  '.join(
            f"{b['index']}: {b['pressed_frames']}/{b['touched_frames']}/{b['max_value']:.2f}"
            for b in gp['buttons']))
        if gp['axes']:
            lines.append('  axes    ' + '  '.join(
                f"{a['index']}: {a['min']:.2f}..{a['max']:.2f} ({a['nonzero_frames']} nonzero)"
                for a in gp['axes']))
    return '\n'.join(lines)
