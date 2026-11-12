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
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ranger_xarm_teleop_bridge.frame_parse import Ignored, Malformed, parse_text, ParsedFrame
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
    resets = [{'at': times[i], 'seq': samples[i].frame.seq,
               'from': samples[i - 1].frame.reset_count, 'to': samples[i].frame.reset_count}
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

    # Still-controller spread over tracked (present, not emulated) frames.
    tracked = [h for h, flag in zip(hand_samples, tracked_flags) if flag]
    if tracked:
        n = len(tracked)
        mean = [sum(h.position[i] for h in tracked) / n for i in range(3)]
        std = [math.sqrt(sum((h.position[i] - mean[i]) ** 2 for h in tracked) / n)
               for i in range(3)]
        p2p = [max(h.position[i] for h in tracked) - min(h.position[i] for h in tracked)
               for i in range(3)]
        max_dev = max(math.dist(h.position, mean) for h in tracked)
        q_mean = _mean_quaternion([h.orientation for h in tracked])
        angles = [_angle_between(h.orientation, q_mean) for h in tracked]
        report['spread'] = {
            'tracked_frames': n,
            'mean_position_m': mean,
            'std_mm': [s * 1000.0 for s in std],
            'peak_to_peak_mm': [p * 1000.0 for p in p2p],
            'max_deviation_mm': max_dev * 1000.0,
            'mean_orientation_xyzw': list(q_mean),
            'max_angle_deg': max(angles),
            'rms_angle_deg': math.sqrt(sum(a * a for a in angles) / n),
            'pass_position': max_dev * 1000.0 < PASS_POSITION_MM,
            'pass_angle': max(angles) < PASS_ANGLE_DEG,
        }
    else:
        report['spread'] = None

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
        lines.append(f"  reset at {ev['at']:.2f} s (seq {ev['seq']}): {ev['from']} -> {ev['to']}")
    se = report['sessions']
    lines.append(f"XR session changes: {se['count']}  "
                 f"session_id {se['first_session_id']} .. {se['final_session_id']}")
    for ev in se['events']:
        lines.append(f"  new session at {ev['at']:.2f} s (seq {ev['seq']}): "
                     f"{ev['from']} -> {ev['to']}")
    lines.append(f"present frames {report['present_frames']}  "
                 f"absent intervals {len(report['absent_intervals'])}  "
                 f"emulated intervals {len(report['emulated_intervals'])}")
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
        lines.append(f"spread over {sp['tracked_frames']} tracked frames (still controller):")
        std = sp['std_mm']
        p2p = sp['peak_to_peak_mm']
        lines.append(f'  position std (mm) x/y/z: {std[0]:.2f} / {std[1]:.2f} / {std[2]:.2f}   '
                     f'peak-to-peak (mm): {p2p[0]:.2f} / {p2p[1]:.2f} / {p2p[2]:.2f}')
        lines.append(f"  max deviation from mean: {sp['max_deviation_mm']:.2f} mm  "
                     f"[{'PASS' if sp['pass_position'] else 'FAIL'} < {PASS_POSITION_MM:g} mm]")
        lines.append(f"  orientation: max {sp['max_angle_deg']:.3f} deg, "
                     f"rms {sp['rms_angle_deg']:.3f} deg from mean  "
                     f"[{'PASS' if sp['pass_angle'] else 'FAIL'} < {PASS_ANGLE_DEG:g} deg]")
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
    return '\n'.join(lines)
