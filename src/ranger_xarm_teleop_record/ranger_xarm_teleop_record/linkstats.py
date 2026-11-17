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
"""Link quality of the headset input over one episode (no ROS)."""

from typing import Sequence

import numpy as np


def _stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'n': 0, 'median': None, 'p99': None, 'max': None}
    return {'n': int(v.size), 'median': float(np.median(v)),
            'p99': float(np.percentile(v, 99)), 'max': float(v.max())}


def link_quality(input_age: Sequence[float], status_seq: Sequence[int],
                 xr_time_ms: Sequence[float], status_stamp: Sequence[float],
                 session_id: Sequence[int], reconnect_gap_s: float) -> dict:
    """
    Return the link numbers of one episode.

    ``input_age`` is TeleopState.input_age over the armed cycles (s); the status arrays are
    InputStatus in receive order, ``status_stamp`` on the bridge host's clock (s).
    Sequence gaps count jumps of more than one; ``seq_backwards`` counts repeats or steps
    back. XR frame-time jitter is the spread of the frame-time steps around their median.
    A reconnect is a status gap longer than ``reconnect_gap_s`` or a change of session id.
    """
    seq = np.asarray(status_seq, dtype=np.int64)
    dseq = np.diff(seq) if seq.size > 1 else np.zeros(0, dtype=np.int64)
    xr = np.asarray(xr_time_ms, dtype=float)
    dxr = np.diff(xr) if xr.size > 1 else np.zeros(0)
    dxr = dxr[np.isfinite(dxr)]
    jitter = {'median_period_ms': None, 'std_ms': None, 'p99_ms': None}
    if dxr.size:
        med = float(np.median(dxr))
        dev = np.abs(dxr - med)
        jitter = {'median_period_ms': med, 'std_ms': float(np.std(dxr)),
                  'p99_ms': float(np.percentile(dev, 99))}
    st = np.asarray(status_stamp, dtype=float)
    gaps = np.diff(st) if st.size > 1 else np.zeros(0)
    sid = np.asarray(session_id, dtype=np.int64)
    session_changes = int(np.count_nonzero(np.diff(sid))) if sid.size > 1 else 0
    long_gaps = int(np.count_nonzero(gaps > reconnect_gap_s))
    return {
        'input_age_s': _stats(input_age),
        'status_messages': int(seq.size),
        'seq_gaps': int(np.count_nonzero(dseq > 1)),
        'seq_missing': int(np.sum(dseq[dseq > 1] - 1)) if dseq.size else 0,
        'seq_backwards': int(np.count_nonzero(dseq <= 0)),
        'xr_frame_time': jitter,
        'largest_status_gap_s': float(gaps.max()) if gaps.size else None,
        'reconnects': long_gaps + session_changes,
        'session_changes': session_changes,
    }
