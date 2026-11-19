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
Check that the simulator on this machine is fast and steady enough for teleoperation (no ROS).

The teleop gate releases the clutch when the newest joint state is older than
``tool_pose_timeout_s`` (0.1 s by default), by its arrival on the wall clock. A simulator that
runs well below real time delivers clock and joint states in bursts, and the gaps between
bursts reach that limit: the clutch then releases on "Arm position late". On a machine with two
GPUs this happens when the simulator renders its window and cameras on the integrated one.

``assess`` takes the wall-clock arrivals of /joint_states and (ROS time, wall time) pairs from
/clock, measured with the simulator running as it will be used, and says whether the arrival
gaps leave a margin to the limit.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

GAP_MARGIN = 0.5     # warn when the p99 arrival gap exceeds this fraction of the limit
MIN_RTF = 0.6        # warn when the ROS clock runs slower than this fraction of real time
GPU_HINT = ('if this machine has a second (discrete) GPU, render the simulator on it '
            '(README: "Simulator rendering")')


def _percentile(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return math.nan
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def parameter_from(data: Any, name: str, default: float) -> float:
    """
    Return parameter ``name`` from a loaded ROS parameter file, else ``default``.

    Namespaces and node names may nest above ``ros__parameters`` (as in bringup's
    ``load_limits``); a file that is not a mapping, or does not set it, gives ``default``.
    """
    found: List[Any] = []

    def collect(section: Any) -> None:
        if not isinstance(section, dict):
            return
        for key, value in section.items():
            if key == 'ros__parameters' and isinstance(value, dict):
                if name in value:
                    found.append(value[name])
            else:
                collect(value)

    collect(data)
    return float(found[-1]) if found else float(default)


def real_time_factor(clock_pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Return ROS time over wall time elapsed, first to last pair; None if unknown."""
    pairs = [(r, w) for r, w in clock_pairs if math.isfinite(r) and math.isfinite(w) and r > 0.0]
    if len(pairs) < 2:
        return None
    (r0, w0), (r1, w1) = pairs[0], pairs[-1]
    if w1 - w0 < 1.0 or r1 < r0:
        return None
    return (r1 - r0) / (w1 - w0)


def assess(joint_arrivals: Sequence[float], clock_pairs: Sequence[Tuple[float, float]],
           tool_pose_timeout_s: float, seconds: float) -> Dict:
    """
    Return the report: arrival gaps of /joint_states, the real-time factor and warnings.

    ``joint_arrivals``: monotonic arrival times (s) of /joint_states. ``clock_pairs``: (ROS
    time, monotonic time) of /clock messages; empty without a simulated clock. ``ok`` is
    False when anything would make the gate release on late arm data during a session.
    """
    arrivals = sorted(joint_arrivals)
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    limit = float(tool_pose_timeout_s)
    rtf = real_time_factor(clock_pairs)
    over = sum(1 for g in gaps if g > limit)
    p99 = _percentile(gaps, 99.0)
    warnings: List[str] = []
    if len(arrivals) < 2:
        warnings.append(f'fewer than two /joint_states in {seconds:.0f} s: is the simulator '
                        'running, and are its controllers up?')
    else:
        if over:
            warnings.append(
                f'{over} gaps between joint states were longer than tool_pose_timeout_s '
                f'({limit * 1000:.0f} ms): while engaged, the clutch would have been released '
                f'up to that often ("Arm position late"); {GPU_HINT}')
        if p99 > GAP_MARGIN * limit:
            warnings.append(
                f'p99 gap between joint states {p99 * 1000:.0f} ms is over '
                f'{GAP_MARGIN:.0%} of tool_pose_timeout_s ({limit * 1000:.0f} ms): little '
                f'margin; {GPU_HINT}')
    if rtf is not None and rtf < MIN_RTF:
        warnings.append(f'the simulator runs at {rtf:.2f} of real time (below {MIN_RTF}): '
                        f'arm data comes in bursts; {GPU_HINT}')

    def ms(v: float) -> Optional[float]:
        return None if not math.isfinite(v) else round(1000.0 * v, 1)

    return {
        'seconds': round(float(seconds), 1),
        'joint_states': len(arrivals),
        'joint_state_rate_hz': (round((len(arrivals) - 1) / (arrivals[-1] - arrivals[0]), 1)
                                if len(arrivals) > 1 and arrivals[-1] > arrivals[0] else None),
        'gap_ms': {'median': ms(_percentile(gaps, 50.0)), 'p99': ms(p99),
                   'max': ms(max(gaps)) if gaps else None},
        'gaps_over_limit': over,
        'tool_pose_timeout_ms': round(limit * 1000.0, 1),
        'real_time_factor': None if rtf is None else round(rtf, 3),
        'simulated_clock': bool(clock_pairs),
        'warnings': warnings,
        'ok': not warnings,
    }
