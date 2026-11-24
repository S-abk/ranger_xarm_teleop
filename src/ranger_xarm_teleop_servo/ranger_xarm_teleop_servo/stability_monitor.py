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
The stability monitor's state, from its inputs to one published figure per cycle (no ROS).

Information only (contract, "Stability monitor"; INV-1): nothing here commands, holds or
refuses anything. The node feeds the inputs and publishes what ``cycle`` returns.

* **Never a number it cannot stand behind (INV-4).** The figure is unavailable, with the
  reason, when the description or the mass file is missing or refused, when a joint the figure
  needs has no fresh value, when odometry is missing or stale, when a configured tilt source
  is missing or stale ("no tilt", naming the launch argument that bridges the simulated one),
  when no tilt source is configured and the base is not at rest, or when any value is not
  finite. No number comes from a built-in default or a stale input.
* **Tilt source** (``tilt_configured``):
  - configured and fresh: the IMU rigid-body transfer (``stability.imu_transfer``), tilt
    MEASURED;
  - not configured: a figure only while the base is at rest by odometry, LEVEL_ASSUMED; its
    pessimistic figure takes the floor tilted toward each edge by the assumed-slope bound,
    amplified by the lean on the springs.
* **Freshness** is judged on arrival (monotonic seconds, like the base's rest rule); the
  filters run on the messages' own stamps.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ranger_xarm_teleop_servo import stability as st
from ranger_xarm_teleop_servo import stability_mass as sm
from ranger_xarm_teleop_servo.base_rest import AT_REST, RestParams, RestWatch
from ranger_xarm_teleop_servo.stability_filters import (Cascade, ComplementaryTilt, Derivative,
                                                        JointAcceleration)
from ranger_xarm_teleop_servo.stability_model import DescriptionError, RobotModel

# Reasons (the message's REASON_* constants carry the same values).
OK = 0
NO_DESCRIPTION = 1
DESCRIPTION_UNUSABLE = 2
NO_MASS_FILE = 3
MASS_FILE_REFUSED = 4
NO_JOINT_STATES = 5
JOINT_STATES_STALE = 6
NO_ODOMETRY = 7
ODOMETRY_STALE = 8
NO_TILT = 9
MOVING_WITHOUT_TILT = 10
NOT_FINITE = 11
NOT_SIMULATION = 12
REASON_NAMES = ('OK', 'NO_DESCRIPTION', 'DESCRIPTION_UNUSABLE', 'NO_MASS_FILE',
                'MASS_FILE_REFUSED', 'NO_JOINT_STATES', 'JOINT_STATES_STALE', 'NO_ODOMETRY',
                'ODOMETRY_STALE', 'NO_TILT', 'MOVING_WITHOUT_TILT', 'NOT_FINITE',
                'NOT_SIMULATION')

TILT_NONE, TILT_MEASURED, TILT_LEVEL_ASSUMED = 0, 1, 2

NOTE = 'does not see a force on the tool'

# The stop figure's memory of the present figure never spans more than this (a body with no
# spring stiffness left would otherwise remember for ever).
MAX_RING_MEMORY_S = 3.0


@dataclass(frozen=True)
class Config:
    """The monitor's settings; every one is required (the node has no defaults for them)."""

    base_frame: str
    wheel_links: Tuple[str, ...]
    tilt_configured: bool
    cutoff_hz: float                        # low-pass on the IMU and the joints
    tilt_tau_s: float                       # complementary filter
    joint_timeout_s: float
    odometry_timeout_s: float
    imu_timeout_s: float
    stop_decel_m_s2: float                  # the stop the figure "if stopped now" assumes
    stop_turn_decel_rad_s2: float
    assumed_slope_deg: Optional[float]      # required when no tilt source is configured
    tilt_hint: str = ''                     # appended to "no tilt" (the launch argument)
    rest: RestParams = field(default_factory=RestParams)

    def check(self) -> None:
        for name in ('cutoff_hz', 'tilt_tau_s', 'joint_timeout_s', 'odometry_timeout_s',
                     'imu_timeout_s', 'stop_decel_m_s2', 'stop_turn_decel_rad_s2'):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0):
                raise ValueError(f'stability {name} must be finite and above zero')
        if not self.tilt_configured:
            v = self.assumed_slope_deg
            if v is None or not (math.isfinite(v) and 0.0 <= v < 45.0):
                raise ValueError('stability assumed_slope_deg is required (0 to 45) when no '
                                 'tilt source is configured')
        if len(self.wheel_links) < 3:
            raise ValueError('stability wheel_links: at least three wheel links are required')
        self.rest.check()


@dataclass(frozen=True)
class Result:
    """One cycle's state (the message's fields)."""

    available: bool
    reason: int
    reason_text: str
    tilt_source: int = TILT_NONE
    margin: float = math.nan
    edge: int = 0
    margin_pessimistic: float = math.nan
    edge_pessimistic: int = 0
    stop_margin: float = math.nan
    stop_margin_pessimistic: float = math.nan
    stop_edge: int = 0
    stop_decel: float = math.nan
    energy_reserve: float = math.nan
    energy_edge: int = 0
    cop: Tuple[float, float] = (math.nan, math.nan)
    roll_deg: float = math.nan
    pitch_deg: float = math.nan
    speed: float = math.nan
    turn_rate: float = math.nan
    at_rest: bool = False
    fingerprint: str = ''
    note: str = NOTE


def unavailable(reason: int, text: str, fingerprint: str = '', **kw) -> Result:
    return Result(False, reason, text, fingerprint=fingerprint, **kw)


class Monitor:
    """The stability monitor's state."""

    def __init__(self, config: Config, simulation: bool):
        config.check()
        self.config = config
        self.simulation = simulation
        self.model: Optional[RobotModel] = None
        self.model_error: Optional[str] = None
        self.mass: Optional[sm.MassModel] = None
        self.mass_error: Optional[str] = None
        self._mass_data = None
        self._mass_given = False
        self.joints: Dict[str, Tuple[float, float]] = {}     # name -> (value, arrival)
        self.arm: Optional[JointAcceleration] = None
        self.rest = RestWatch(config.rest)
        self.odometry: Optional[Tuple[float, float, float, float]] = None  # vx, vy, wz, arrival
        self.f_filter = Cascade(config.cutoff_hz)
        self.w_filter = Cascade(config.cutoff_hz)
        self.alpha = Derivative(config.cutoff_hz)
        self.tilt = ComplementaryTilt(config.tilt_tau_s)
        self.imu_arrival: Optional[float] = None
        self.imu_rotation: Optional[np.ndarray] = None       # IMU axes -> base axes
        self.imu_position: Optional[np.ndarray] = None       # in the base frame
        self.imu_error: Optional[str] = None
        self._derived = None
        self._period = math.inf
        self._recent = deque()                               # (now, margin, edge, pess, edge)

    # ---------------------------------------------------------------- inputs
    def set_description(self, urdf: str) -> None:
        try:
            self.model = RobotModel(urdf, self.config.base_frame, self.config.wheel_links)
            self.model_error = None
        except DescriptionError as err:
            self.model, self.model_error = None, str(err)
        self._apply_mass()

    def set_mass_data(self, data, error: Optional[str] = None) -> None:
        """Take a loaded mass file (``yaml.safe_load`` output), or the error reading it."""
        self._mass_given = True
        self._mass_data = data
        self.mass_error = error
        self._apply_mass()

    def _apply_mass(self) -> None:
        self.mass, self._derived = None, None
        self._recent.clear()
        if self.model is None or not self._mass_given or self._mass_data is None:
            return
        try:
            self.mass = sm.parse(self._mass_data, self.model, self.simulation)
            self.mass_error = None
        except sm.MassFileError as err:
            self.mass_error = str(err)
            return
        required = self.model.required_joints()
        moving = [n for n in required if not self._is_steering(n)]
        self.arm = JointAcceleration(self.config.cutoff_hz, moving)
        self._derived = self._derive()

    def _is_steering(self, joint: str) -> bool:
        """Return whether ``joint`` lies on the path from the base to a wheel."""
        wheel_joints = set()
        for w in self.model.wheel_links:
            link = w
            while link in self.model._child_joint:
                jn = self.model._child_joint[link]
                wheel_joints.add(jn)
                link = self.model.joints[jn].parent
        return joint in wheel_joints

    def _derive(self):
        """Return what depends only on the model: the sprung bodies, stiffness, damping."""
        model, mass = self.model, self.mass
        unsprung = set(model.unsprung_links())
        sprung = np.array([b.body.link not in unsprung for b in mass.bodies])
        k_roll, k_pitch = model.suspension_stiffness()
        scale = {}
        for label, k_bound, c_bound in (('nominal', 'value', 'value'), ('low', 'low', 'low')):
            if mass.stiffness is None:
                scale[label] = (1.0, 1.0)
                continue
            k_nom = mass.stiffness.value
            kf = getattr(mass.stiffness, k_bound) / k_nom if k_nom > 0 else 1.0
            cf = (getattr(mass.damping, c_bound) / mass.damping.value
                  if mass.damping is not None and mass.damping.value > 0 else 1.0)
            scale[label] = (kf, cf)
        return {'sprung': sprung, 'k_roll': k_roll, 'k_pitch': k_pitch, 'scale': scale}

    def on_joint_states(self, arrival: float, stamp: float, names: Sequence[str],
                        positions: Sequence[float], source_is_arm: bool = True) -> None:
        for n, p in zip(names, positions):
            self.joints[n] = (float(p), arrival)
        if self.arm is not None and source_is_arm:
            if all(n in self.joints for n in self.arm.names):
                self.arm.update(stamp, {n: self.joints[n][0] for n in self.arm.names})

    def on_odometry(self, arrival: float, vx: float, vy: float, wz: float) -> None:
        self.rest.add(arrival, vx, vy, wz)
        self.odometry = (vx, vy, wz, arrival)

    def set_imu_frame(self, rotation: Optional[np.ndarray], position: Optional[np.ndarray],
                      error: Optional[str] = None) -> None:
        """Take the IMU's pose in the base frame (the static transform), or why it is unknown."""
        self.imu_rotation, self.imu_position, self.imu_error = rotation, position, error

    def on_imu(self, arrival: float, stamp: float, accel, rate) -> None:
        """Take one IMU sample: linear acceleration and angular rate (IMU axes) only."""
        if self.imu_rotation is None:
            return
        f = self.imu_rotation @ np.asarray(accel, dtype=float)
        w = self.imu_rotation @ np.asarray(rate, dtype=float)
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(w))):
            return
        if self.f_filter.update(stamp, f) is None:
            return
        self.w_filter.update(stamp, w)
        self.alpha.update(stamp, w)
        self.tilt.update(stamp, f, w)
        self.imu_arrival = arrival

    # ---------------------------------------------------------------- the cycle
    def cycle(self, now: float) -> Result:
        """Return the state at ``now`` (monotonic seconds)."""
        c = self.config
        fp = self.mass.fingerprint if self.mass is not None else ''
        if self.model is None:
            if self.model_error is not None:
                return unavailable(DESCRIPTION_UNUSABLE, self.model_error)
            return unavailable(NO_DESCRIPTION, 'no robot description received')
        if self.mass is None:
            if self.mass_error is not None:
                return unavailable(MASS_FILE_REFUSED, f'mass file refused: {self.mass_error}')
            return unavailable(NO_MASS_FILE, 'no mass file given')
        required = self.model.required_joints()
        missing = [n for n in required if n not in self.joints]
        if missing:
            return unavailable(NO_JOINT_STATES, f'no value for joint(s) {missing}', fp)
        stale = [n for n in required if now - self.joints[n][1] > c.joint_timeout_s]
        if stale:
            return unavailable(JOINT_STATES_STALE,
                               f'joint value(s) older than {c.joint_timeout_s:g} s: {stale}', fp)
        if self.odometry is None:
            return unavailable(NO_ODOMETRY, 'no odometry received', fp)
        vx, vy, wz, arrival = self.odometry
        if now - arrival > c.odometry_timeout_s:
            return unavailable(ODOMETRY_STALE,
                               f'odometry older than {c.odometry_timeout_s:g} s', fp)
        at_rest = self.rest.state(now) == AT_REST
        motion = {'speed': math.hypot(vx, vy), 'turn_rate': wz, 'at_rest': at_rest}
        if c.tilt_configured:
            if self.imu_rotation is None:
                return unavailable(NO_TILT, f'no tilt: the IMU frame is unknown '
                                   f'({self.imu_error or "no transform"}){c.tilt_hint}', fp,
                                   **motion)
            if self.imu_arrival is None:
                return unavailable(NO_TILT, f'no tilt: no IMU sample received{c.tilt_hint}', fp,
                                   **motion)
            if now - self.imu_arrival > c.imu_timeout_s:
                return unavailable(NO_TILT, f'no tilt: the IMU is older than '
                                   f'{c.imu_timeout_s:g} s{c.tilt_hint}', fp, **motion)
        elif not at_rest:
            return unavailable(MOVING_WITHOUT_TILT, 'moving without a tilt source: a figure '
                               'only at rest (level floor assumed)', fp, **motion)
        try:
            return self._figures(now, vx, vy, wz, at_rest, fp)
        except (ValueError, FloatingPointError) as err:
            return unavailable(NOT_FINITE, f'no figure: {err}', fp, **motion)

    def _arm_accelerations(self, q: Mapping[str, float], qd: Mapping[str, float],
                           qdd: Mapping[str, float], posed) -> np.ndarray:
        h = 0.01
        plus = {n: q[n] + qd[n] * h + 0.5 * qdd[n] * h * h for n in q}
        minus = {n: q[n] - qd[n] * h + 0.5 * qdd[n] * h * h for n in q}
        bodies = self.mass.nominal_bodies
        steer = {n: v for n, (v, _) in self.joints.items()}
        pp = self.model.pose({**steer, **plus}, bodies).positions
        pm = self.model.pose({**steer, **minus}, bodies).positions
        return (pp - 2.0 * posed.positions + pm) / (h * h)

    def _figures(self, now, vx, vy, wz, at_rest, fp) -> Result:
        c, d = self.config, self._derived
        values = {n: v for n, (v, _) in self.joints.items()}
        arm_state = self.arm.state() if self.arm is not None else None
        if arm_state is not None:
            values.update(arm_state[0])
        posed = self.model.pose(values, self.mass.nominal_bodies)
        arrays = sm.as_arrays(self.mass)
        edges = st.support_polygon(list(posed.contacts.values()))
        a_arm = np.zeros_like(posed.positions)
        if arm_state is not None:
            a_arm = self._arm_accelerations(arm_state[0], arm_state[1], arm_state[2], posed)
        sprung = d['sprung']
        ms = float(arrays['mass'][sprung].sum())
        hs = float((arrays['mass'][sprung] * (posed.positions[sprung, 2] - posed.plane_z)).sum()
                   / ms) if ms > 0 else 0.0
        velocity = np.array([vx, vy])
        if c.tilt_configured:
            f = self.f_filter.value
            omega = self.w_filter.value
            alpha = self.alpha.value if self.alpha.value is not None else np.zeros(3)
            w = st.imu_transfer(posed.positions, self.imu_position, f, omega, alpha) - a_arm
            hdot = st.angular_momentum_rate(posed.inertias, omega, alpha)
            g_body = st.G * self.tilt.down
            tilt_source = TILT_MEASURED
        else:
            g_body = np.array([0.0, 0.0, -st.G])
            w = st.gravity_vectors(len(posed.masses), g_body) - a_arm
            hdot = None
            tilt_source = TILT_LEVEL_ASSUMED
        if not st.finite(w, g_body, posed.positions):
            raise ValueError('an input is not finite')
        nominal = st.figure(posed.positions, posed.masses, w, posed.plane_z, edges, hdot)
        if c.tilt_configured:
            pess, pess_edge = st.pessimistic_margin(
                posed.positions, posed.masses, arrays['mass_low'], arrays['mass_high'],
                arrays['com_bound'], w, posed.plane_z, edges, hdot)
        else:
            pess, pess_edge = self._level_pessimistic(posed, arrays, edges, ms, hs)
        # If the base stopped now: the measured tilt, the stop deceleration, the suspension.
        stop_nom = self._stop(posed, arrays, edges, g_body, velocity, wz, ms, hs, 'nominal')
        stop_pess = self._stop(posed, arrays, edges, g_body, velocity, wz, ms, hs, 'low',
                               pessimistic=True)
        # A stop does not undo what is already under way: a body still rocking (from a bump, a
        # start, an earlier stop) swings back to its recent extremes, damped. The figure through
        # the stop is never above the present figure's minimum over the last period of the
        # body's slowest mode (at the softest springs).
        memory = self.ring_period(posed, ms, hs)
        self._recent.append((now, nominal.margin, nominal.edge, pess, pess_edge))
        while self._recent and self._recent[0][0] < now - memory:
            self._recent.popleft()
        low_nom = min(self._recent, key=lambda r: r[1])
        low_pess = min(self._recent, key=lambda r: r[3])
        if low_nom[1] < stop_nom[0]:
            stop_nom = (low_nom[1], low_nom[2])
        if low_pess[3] < stop_pess[0]:
            stop_pess = (low_pess[3], low_pess[4])
        shift = st.combined_com_bound(arrays['mass'], arrays['com_bound'])
        reserve, reserve_edge = st.energy_reserve(posed.positions, posed.masses, posed.inertias,
                                                  posed.plane_z, edges, g_body, velocity, wz,
                                                  com_shift=shift)
        down = g_body / np.linalg.norm(g_body)
        roll = math.degrees(math.atan2(-down[1], -down[2]))
        pitch = math.degrees(math.atan2(down[0], math.hypot(down[1], down[2])))
        values_out = (nominal.margin, pess, stop_nom[0], stop_pess[0], reserve)
        if not st.finite(values_out, nominal.cop):
            raise ValueError('a figure is not finite')
        return Result(
            available=True, reason=OK, reason_text='', tilt_source=tilt_source,
            margin=nominal.margin, edge=nominal.edge.code,
            margin_pessimistic=min(pess, nominal.margin), edge_pessimistic=pess_edge.code,
            stop_margin=stop_nom[0], stop_margin_pessimistic=min(stop_pess[0], stop_nom[0]),
            stop_edge=stop_nom[1].code, stop_decel=c.stop_decel_m_s2,
            energy_reserve=reserve, energy_edge=reserve_edge.code,
            cop=(float(nominal.cop[0]), float(nominal.cop[1])), roll_deg=roll, pitch_deg=pitch,
            speed=float(np.hypot(vx, vy)), turn_rate=float(wz), at_rest=at_rest, fingerprint=fp)

    def _factors(self, posed, ms, hs, label) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Return the suspension's (amplification, overshoot) along x (pitch) and y (roll).

        A stop moves the load toward the direction of motion by the quasi-static shift times
        the amplification, then the body rings: first past it by the overshoot, then back
        past the resting position by about the same overshoot (measured in stage 1: the loads'
        minimum after a stop is often on the opposite edge).
        """
        d = self._derived
        kf, cf = d['scale'][label]
        out = []
        for k_axis, axis in ((d['k_pitch'], 0), (d['k_roll'], 1)):
            k = k_axis * kf
            amp = st.lean_amplification(k, ms, hs)
            zeta = self._zeta(posed, axis, k, cf, ms, hs)
            out.append((amp, st.overshoot(zeta)))
        return out[0], out[1]

    def _zeta(self, posed, axis, k, cf, ms, hs) -> float:
        """
        Return the damping ratio of the body's pitch or roll mode.

        ``axis`` 0: pitch (motion along x); 1: roll (motion along y). The ratio is
        C / (2 sqrt((K - m g h) I)); the mode's period is kept in ``_period``.
        """
        if math.isinf(k):
            return 1.0
        springs = list(self.model.springs.values())
        tf = self.model.link_transforms({})
        coords = np.array([tf[self.model.joints[s.joint].child][:3, 3] for s in springs])
        damp = np.array([s.damping for s in springs]) * cf
        centre = coords[:, axis].mean()
        c_mode = float((damp * (coords[:, axis] - centre) ** 2).sum())
        sprung = self._derived['sprung']
        p = posed.positions[sprung]
        m = posed.masses[sprung]
        about = 1 - axis                    # pitch turns about y, roll about x
        inertia = float((m * ((p[:, axis] - centre) ** 2 + (p[:, 2] - posed.plane_z) ** 2)).sum()
                        + posed.inertias[sprung][:, about, about].sum())
        k_eff = k - ms * st.G * hs
        if k_eff <= 0.0 or inertia <= 0.0:
            self._period = math.inf
            return 0.0
        self._period = 2.0 * math.pi / math.sqrt(k_eff / inertia)
        return c_mode / (2.0 * math.sqrt(k_eff * inertia))

    def ring_period(self, posed, ms, hs) -> float:
        """Return the longest period of the body's roll and pitch modes at the softest springs."""
        d = self._derived
        if math.isinf(d['k_roll']):
            return 0.0
        kf, cf = d['scale']['low']
        periods = []
        for k_axis, axis in ((d['k_pitch'], 0), (d['k_roll'], 1)):
            self._zeta(posed, axis, k_axis * kf, cf, ms, hs)
            periods.append(self._period)
        return min(max(periods), MAX_RING_MEMORY_S)

    def _stop(self, posed, arrays, edges, g_body, velocity, wz, ms, hs, label,
              pessimistic=False):
        """
        Return the smallest figure through a stop from the present motion, and its edge.

        Two phases: the load thrown toward the motion (amplification x (1 + overshoot)), and
        the swing back past rest (amplification x overshoot, the other way).
        """
        (ax, ox), (ay, oy) = self._factors(posed, ms, hs, label)
        best = None
        for fx, fy in ((ax * (1.0 + ox), ay * (1.0 + oy)), (-ax * ox, -ay * oy)):
            w = st.stop_vectors(posed.positions, g_body, velocity, wz,
                                self.config.stop_decel_m_s2, self.config.stop_turn_decel_rad_s2,
                                fx, fy)
            if pessimistic:
                m, e = st.pessimistic_margin(posed.positions, posed.masses, arrays['mass_low'],
                                             arrays['mass_high'], arrays['com_bound'], w,
                                             posed.plane_z, edges)
            else:
                fig = st.figure(posed.positions, posed.masses, w, posed.plane_z, edges)
                m, e = fig.margin, fig.edge
            if best is None or m < best[0]:
                best = (m, e)
        return best

    def _level_pessimistic(self, posed, arrays, edges, ms, hs):
        """Return the smallest figure, the floor tilted toward each edge by the assumed bound."""
        d = self._derived
        kf, _ = d['scale']['low']
        best = None
        for e in edges:
            out = np.array([-e.inward[0], -e.inward[1]])
            k_axis = d['k_pitch'] if abs(out[0]) >= abs(out[1]) else d['k_roll']
            amp = st.lean_amplification(k_axis * kf, ms, hs)
            theta = math.radians(self.config.assumed_slope_deg) * amp
            g = st.G * np.array([math.sin(theta) * out[0], math.sin(theta) * out[1],
                                 -math.cos(theta)])
            w = st.gravity_vectors(len(posed.masses), g)
            m, edge = st.pessimistic_margin(posed.positions, posed.masses, arrays['mass_low'],
                                            arrays['mass_high'], arrays['com_bound'], w,
                                            posed.plane_z, [e])
            if best is None or m < best[0]:
                best = (m, edge)
        return best


def reason_name(code: int) -> str:
    return REASON_NAMES[code] if 0 <= code < len(REASON_NAMES) else str(code)
