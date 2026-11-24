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
Causal filters of the stability monitor's inputs, for irregular sample times (numpy, no ROS).

* ``LowPass``: first order, cutoff in Hz; its lag is about 1 / (2 pi fc) (16 ms at 10 Hz).
* ``Derivative``: the backward difference of a low-passed signal, low-passed again at the same
  cutoff (the gyro's derivative, the arm's joint speeds and accelerations).
* ``Cascade``: two low-passes in series, so that a signal used beside a ``Derivative`` carries
  the same lag (about 2 / (2 pi fc): 32 ms at 10 Hz). The IMU's specific force, its rate and the
  rate's derivative are combined in one sum; terms filtered with different lags would not
  cancel where they should (measured: a 40 ms halt read 0.18 m too low with unequal lags).
* ``ComplementaryTilt``: the direction of gravity in the body, from the gyro (fast) corrected
  toward the accelerometer (slow, time constant tau). It never reads an orientation published
  by the IMU.
* ``JointAcceleration``: low-passed joint values with their first and second derivatives.

A sample older than the last one, or at the same stamp, is ignored (``accept`` says so).
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np


class LowPass:
    """First-order low-pass on irregular samples."""

    def __init__(self, cutoff_hz: float):
        if not cutoff_hz > 0.0:
            raise ValueError('the cutoff must be above 0 Hz')
        self.tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        self.value: Optional[np.ndarray] = None
        self.stamp: Optional[float] = None

    def update(self, stamp: float, x) -> Optional[np.ndarray]:
        x = np.asarray(x, dtype=float)
        if self.value is None:
            self.value, self.stamp = x.copy(), stamp
            return self.value
        dt = stamp - self.stamp
        if not dt > 0.0:
            return None
        a = dt / (self.tau + dt)
        self.value = self.value + a * (x - self.value)
        self.stamp = stamp
        return self.value


class Cascade:
    """Two first-order low-passes in series: the same lag as ``Derivative``'s path."""

    def __init__(self, cutoff_hz: float):
        self.first = LowPass(cutoff_hz)
        self.second = LowPass(cutoff_hz)
        self.value: Optional[np.ndarray] = None

    def update(self, stamp: float, x) -> Optional[np.ndarray]:
        """Take a sample; return the filtered value (None for an old stamp)."""
        y = self.first.update(stamp, x)
        if y is None:
            return None
        self.value = self.second.update(stamp, y)
        return self.value


class Derivative:
    """Backward difference of a low-passed signal, low-passed again."""

    def __init__(self, cutoff_hz: float):
        self.signal = LowPass(cutoff_hz)
        self.rate = LowPass(cutoff_hz)
        self._prev: Optional[np.ndarray] = None
        self._prev_stamp: Optional[float] = None
        self.value: Optional[np.ndarray] = None

    def update(self, stamp: float, x) -> Optional[np.ndarray]:
        y = self.signal.update(stamp, x)
        if y is None:
            return None
        if self._prev is not None:
            d = (y - self._prev) / (stamp - self._prev_stamp)
            self.value = self.rate.update(stamp, d)
        self._prev, self._prev_stamp = y.copy(), stamp
        return self.value


class ComplementaryTilt:
    """The unit direction of gravity in the body (base axes) from a gyro and an accelerometer."""

    def __init__(self, tau_s: float):
        if not tau_s > 0.0:
            raise ValueError('tau must be above 0 s')
        self.tau = tau_s
        self.down: Optional[np.ndarray] = None
        self.stamp: Optional[float] = None

    def update(self, stamp: float, specific_force, rate) -> Optional[np.ndarray]:
        f = np.asarray(specific_force, dtype=float)
        norm = float(np.linalg.norm(f))
        if norm < 1e-6:
            return None
        measured = -f / norm
        if self.down is None:
            self.down, self.stamp = measured, stamp
            return self.down
        dt = stamp - self.stamp
        if not dt > 0.0:
            return None
        # A vector fixed in the world, seen from a body turning at ``rate``: d/dt = -rate x v.
        down = self.down - np.cross(np.asarray(rate, dtype=float), self.down) * dt
        k = dt / (self.tau + dt)
        down = (1.0 - k) * down + k * measured
        self.down = down / np.linalg.norm(down)
        self.stamp = stamp
        return self.down


class JointAcceleration:
    """Filtered joint values, speeds and accelerations, keyed by joint name."""

    def __init__(self, cutoff_hz: float, names: Sequence[str]):
        self.names = list(names)
        self.position = LowPass(cutoff_hz)
        self.speed = Derivative(cutoff_hz)
        self.acceleration = Derivative(cutoff_hz)
        self.stamp: Optional[float] = None

    def accept(self, stamp: float) -> bool:
        return self.stamp is None or stamp > self.stamp

    def update(self, stamp: float, values: Dict[str, float]) -> bool:
        if not self.accept(stamp):
            return False
        x = np.array([values[n] for n in self.names], dtype=float)
        self.position.update(stamp, x)
        v = self.speed.update(stamp, x)
        if v is not None:
            self.acceleration.update(stamp, v)
        self.stamp = stamp
        return True

    def state(self):
        """Return (q, qdot, qddot) as dicts; speeds and accelerations are 0 until known."""
        q = self.position.value
        if q is None:
            return None
        v = self.speed.value if self.speed.value is not None else np.zeros_like(q)
        a = (self.acceleration.value if self.acceleration.value is not None
             else np.zeros_like(q))
        return (dict(zip(self.names, q)), dict(zip(self.names, v)), dict(zip(self.names, a)))
