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

# A gap longer than this between two samples of the IMU or of the joints restarts their filters,
# which then settle again before a figure is published: differences across a gap are not rates.
GAP_S = 0.1

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
        """Start with nothing received."""
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
        self.arm_arrival: Optional[float] = None             # the arm filter's last sample
        self.rest = RestWatch(config.rest)
        self.odometry: Optional[Tuple[float, float, float, float]] = None  # vx, vy, wz, arrival
        self._start_imu()
        self.imu_restarts = 0
        self.imu_arrival: Optional[float] = None
        self.imu_stamp: Optional[float] = None
        self.imu_rotation: Optional[np.ndarray] = None       # IMU axes -> base axes
        self.imu_position: Optional[np.ndarray] = None       # in the base frame
        self.imu_error: Optional[str] = None
        self._derived = None
        self._period = math.inf
        self._recent = deque()                               # (now, margin, edge, pess, edge)
        # The filters need this much of their input before their output counts (five time
        # constants of one stage: the first sample's step has died down to under 1 %).
        self.settle_s = 5.0 / (2.0 * math.pi * config.cutoff_hz)

    def _start_imu(self) -> None:
        c = self.config
        self.f_filter = Cascade(c.cutoff_hz)
        self.w_filter = Cascade(c.cutoff_hz)
        self.alpha = Derivative(c.cutoff_hz)
        self.tilt = ComplementaryTilt(c.tilt_tau_s)
        self.imu_first_stamp: Optional[float] = None

    # ---------------------------------------------------------------- inputs
    def set_description(self, urdf: str) -> None:
        """Take the robot description."""
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
        self.mass, self._derived, self.arm, self.arm_arrival = None, None, None, None
        self._recent.clear()
        if self.model is None or not self._mass_given or self._mass_data is None:
            return
        try:
            self.mass = sm.parse(self._mass_data, self.model, self.simulation)
            self.mass_error = None
        except sm.MassFileError as err:
            self.mass_error = str(err)
            return
        moving = [n for n in self.model.required_joints()
                  if n not in self.model.wheel_path_joints]
        self.arm = JointAcceleration(self.config.cutoff_hz, moving, gap_s=GAP_S)
        self._derived = self._derive()

    def _derive(self):
        """Return what depends only on the model: the sprung bodies and the springs' corners."""
        model, mass = self.model, self.mass
        unsprung = set(model.unsprung_links())
        sprung = np.array([b.body.link not in unsprung for b in mass.bodies])
        k_roll, k_pitch = model.suspension_stiffness()      # at the description's stiffness
        corners = {'nominal': [(1.0, 1.0)], 'stiff_low': [(1.0, 1.0)], 'all': [(1.0, 1.0)]}
        if mass.stiffness is not None:
            k_desc = next(iter(model.springs.values())).stiffness
            c_desc = next(iter(model.springs.values())).damping

            def kf(v):
                return v / k_desc

            def cf(v):
                return v / c_desc if c_desc > 0 else 1.0
            k, c = mass.stiffness, mass.damping
            corners = {
                'nominal': [(kf(k.value), cf(c.value))],
                'stiff_low': [(kf(k.low), cf(c.value))],
                # A softer spring leans more, a stiffer one rings with less damping; less damping
                # always rings more. The worst stop: the least damping, at five stiffnesses from
                # the lower end to the upper end (the worst can lie between them; sampled).
                'all': [(kf(kk), cf(c.low)) for kk in (
                    k.low, (k.low + k.value) / 2, k.value, (k.value + k.high) / 2, k.high)],
            }
        tf = model.link_transforms({})
        springs = list(model.springs.values())
        coords = np.array([tf[model.joints[sp.joint].child][:3, 3]
                           for sp in springs]).reshape(-1, 3)
        damping = np.array([sp.damping for sp in springs])
        return {'sprung': sprung, 'k_roll': k_roll, 'k_pitch': k_pitch, 'corners': corners,
                'spring_xyz': coords, 'spring_damping': damping}

    def on_joint_states(self, arrival: float, stamp: float, names: Sequence[str],
                        positions: Sequence[float], source_is_arm: bool = True) -> None:
        """Take joint values (both the arm's and the steering's sources)."""
        given = {}
        for n, p in zip(names, positions):
            self.joints[n] = (float(p), arrival)
            given[n] = float(p)
        # The filter takes only a message that carries every one of its joints, so a message of
        # other joints never refreshes it with stored values.
        if self.arm is not None and self.arm.names and all(n in given for n in self.arm.names):
            if self.arm.update(stamp, {n: given[n] for n in self.arm.names}):
                self.arm_arrival = arrival

    def on_odometry(self, arrival: float, vx: float, vy: float, wz: float) -> None:
        """Take one odometry sample (body twist)."""
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
        old = self.imu_stamp is not None and not stamp > self.imu_stamp
        if float(np.linalg.norm(f)) < 1e-6 or old:
            return                      # no direction of gravity in it, or not newer
        if self.imu_stamp is not None and stamp - self.imu_stamp > GAP_S:
            # The gyro's rate is not known across a gap: every IMU filter starts again, the
            # tilt from the accelerometer, and settles again.
            self._start_imu()
            self.imu_restarts += 1
        self.f_filter.update(stamp, f)
        self.w_filter.update(stamp, w)
        self.alpha.update(stamp, w)
        self.tilt.update(stamp, f, w)
        if self.imu_first_stamp is None:
            self.imu_first_stamp = stamp
        self.imu_stamp = stamp
        self.imu_arrival = arrival

    def _imu_settled(self) -> bool:
        # The rates settle in a few of their time constants; the tilt filter, started from the
        # accelerometer alone, in its own time constant.
        return (self.tilt.down is not None and self.alpha.value is not None
                and self.imu_stamp - self.imu_first_stamp >= max(self.settle_s,
                                                                 self.config.tilt_tau_s))

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
        if self.arm.names and (self.arm_arrival is None
                               or now - self.arm_arrival > c.joint_timeout_s):
            stale += [f'{len(self.arm.names)} filtered joint(s) (no newer, finite sample)']
        if stale:
            return unavailable(JOINT_STATES_STALE,
                               f'joint value(s) older than {c.joint_timeout_s:g} s: {stale}', fp)
        if self.arm.names and not self.arm.settled(self.settle_s):
            return unavailable(NO_JOINT_STATES, "the joints' accelerations are still settling",
                               fp)
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
            if not self._imu_settled():
                return unavailable(NO_TILT, 'no tilt: the IMU filters are still settling', fp,
                                   **motion)
        elif not at_rest:
            return unavailable(MOVING_WITHOUT_TILT, 'moving without a tilt source: a figure '
                               'only at rest (level floor assumed)', fp, **motion)
        try:
            return self._figures(now, vx, vy, wz, at_rest, fp)
        except (ValueError, FloatingPointError) as err:
            return unavailable(NOT_FINITE, f'no figure: {err}', fp, **motion)

    def _arm_accelerations(self, values: Mapping[str, float], qd: Mapping[str, float],
                           qdd: Mapping[str, float], posed) -> np.ndarray:
        """Return each body's acceleration from the arm's own motion (base frame)."""
        h = 0.01
        plus = dict(values)
        minus = dict(values)
        for n in qd:
            plus[n] = values[n] + qd[n] * h + 0.5 * qdd[n] * h * h
            minus[n] = values[n] - qd[n] * h + 0.5 * qdd[n] * h * h
        bodies = self.mass.nominal_bodies
        pp = self.model.pose(plus, bodies).positions
        pm = self.model.pose(minus, bodies).positions
        return (pp - 2.0 * posed.positions + pm) / (h * h)

    def _sprung(self, posed, masses):
        """Return the sprung mass and the height of its centre of mass above the contacts."""
        sprung = self._derived['sprung']
        ms = float(masses[sprung].sum())
        if ms <= 0.0:
            return 0.0, 0.0
        hs = float((masses[sprung] * (posed.positions[sprung, 2] - posed.plane_z)).sum() / ms)
        return ms, hs

    def _figures(self, now, vx, vy, wz, at_rest, fp) -> Result:
        c = self.config
        values = {n: v for n, (v, _) in self.joints.items()}     # measured: no filter lag
        posed = self.model.pose(values, self.mass.nominal_bodies)
        arrays = sm.as_arrays(self.mass)
        edges = st.support_polygon(list(posed.contacts.values()))
        a_arm = np.zeros_like(posed.positions)
        arm_state = self.arm.state()
        if self.arm.names and arm_state is not None:
            a_arm = self._arm_accelerations(values, arm_state[1], arm_state[2], posed)
        nominal_sprung = self._sprung(posed, arrays['mass'])
        worst_sprung = (self._sprung(posed, arrays['mass_high'])[0],
                        nominal_sprung[1] + float(arrays['com_bound'].max(initial=0.0)))
        velocity = np.array([vx, vy])
        if c.tilt_configured:
            f = self.f_filter.value
            omega = self.w_filter.value
            alpha = self.alpha.value
            w = st.imu_transfer(posed.positions, self.imu_position, f, omega, alpha) - a_arm
            hdot = st.angular_momentum_rate(posed.inertias, omega, alpha)
            g_body = st.G * self.tilt.down
            tilt_source = TILT_MEASURED
        else:
            lean = self._static_lean(posed, arrays['mass'], 'nominal', nominal_sprung)
            g_body = st.G * np.array([math.sin(lean[0]), math.sin(lean[1]),
                                      -math.sqrt(max(0.0, 1.0 - math.sin(lean[0]) ** 2
                                                     - math.sin(lean[1]) ** 2))])
            w = st.gravity_vectors(len(posed.masses), g_body) - a_arm
            hdot = None
            tilt_source = TILT_LEVEL_ASSUMED
        if not st.finite(w, g_body, posed.positions):
            raise ValueError('an input is not finite')
        nominal = st.figure(posed.positions, posed.masses, w, posed.plane_z, edges, hdot)
        if c.tilt_configured:
            pess = st.pessimistic_margin(
                posed.positions, posed.masses, arrays['mass_low'], arrays['mass_high'],
                arrays['com_bound'], w, posed.plane_z, edges, hdot)
        else:
            pess = self._level_pessimistic(posed, arrays, edges, a_arm, worst_sprung)
        pess = min(pess, (nominal.margin, nominal.edge), key=lambda x: x[0])
        # If the base stopped now: the measured tilt, the stop deceleration, the suspension.
        stop_nom = self._stop(posed, arrays, edges, g_body, velocity, wz, nominal_sprung,
                              'nominal')
        stop_pess = self._stop(posed, arrays, edges, g_body, velocity, wz, worst_sprung, 'all',
                               pessimistic=True)
        # A stop does not undo what is already under way: a body still rocking (from a bump, a
        # start, an earlier stop) swings back to its recent extremes, damped. The figure through
        # the stop is never above the present figure's minimum over the last period of the
        # body's slowest mode (at the softest springs).
        memory = self.ring_period(posed, worst_sprung)
        self._recent.append((now, nominal.margin, nominal.edge, pess[0], pess[1]))
        while self._recent and self._recent[0][0] < now - memory:
            self._recent.popleft()
        low_nom = min(self._recent, key=lambda r: r[1])
        low_pess = min(self._recent, key=lambda r: r[3])
        stop_nom = min(stop_nom, (low_nom[1], low_nom[2]), key=lambda x: x[0])
        stop_pess = min(stop_pess, (low_pess[3], low_pess[4]), stop_nom, key=lambda x: x[0])
        reserve, reserve_edge = self._energy_reserve(posed, arrays, edges, g_body, velocity, wz)
        down = g_body / np.linalg.norm(g_body)
        roll = math.degrees(math.atan2(-down[1], -down[2]))
        pitch = math.degrees(math.atan2(down[0], math.hypot(down[1], down[2])))
        values_out = (nominal.margin, pess[0], stop_nom[0], stop_pess[0], reserve)
        if not st.finite(values_out, nominal.cop):
            raise ValueError('a figure is not finite')
        return Result(
            available=True, reason=OK, reason_text='', tilt_source=tilt_source,
            margin=nominal.margin, edge=nominal.edge.code,
            margin_pessimistic=pess[0], edge_pessimistic=pess[1].code,
            stop_margin=stop_nom[0], stop_margin_pessimistic=stop_pess[0],
            stop_edge=stop_nom[1].code, stop_decel=c.stop_decel_m_s2,
            energy_reserve=reserve, energy_edge=reserve_edge.code,
            cop=(float(nominal.cop[0]), float(nominal.cop[1])), roll_deg=roll, pitch_deg=pitch,
            speed=float(np.hypot(vx, vy)), turn_rate=float(wz), at_rest=at_rest, fingerprint=fp)

    def _energy_reserve(self, posed, arrays, edges, g_body, velocity, wz):
        """
        Return the smallest energy reserve (J) over the edges the motion heads toward.

        The pessimistic mass model throughout: the centre of mass's foot at the worst of the
        mass and centre-of-mass bounds for each edge (geometry: gravity along the body), its
        height raised by the largest bound, the energy to tip at the lightest total mass and
        the kinetic energy at the heaviest.
        """
        down_body = st.gravity_vectors(len(posed.masses), np.array([0.0, 0.0, -st.G]))
        heaviest, lightest = float(arrays['mass_high'].sum()), float(arrays['mass_low'].sum())
        c = (posed.masses[:, None] * posed.positions).sum(axis=0) / posed.masses.sum()
        height = c[2] - posed.plane_z + float(arrays['com_bound'].max(initial=0.0))
        yaw = st.yaw_inertia(posed.positions, posed.masses, posed.inertias)
        ke = st.kinetic_energy(heaviest, velocity, wz, yaw * heaviest / posed.masses.sum())
        best = None
        for e in st.edges_toward(edges, velocity):
            d0, _ = st.pessimistic_margin(posed.positions, posed.masses, arrays['mass_low'],
                                          arrays['mass_high'], arrays['com_bound'], down_body,
                                          posed.plane_z, [e])
            foot = e.point + d0 * e.inward                  # on the edge's normal, d0 inside
            tip = st.energy_to_tip(foot, height, lightest, e, st.tilt_toward(g_body, e.outward))
            if best is None or tip - ke < best[0]:
                best = (tip - ke, e)
        return best

    def _modes(self, posed, sprung_mh, kf, cf, masses=None):
        """Return [(amplification, overshoot, period)] for pitch (x) and roll (y)."""
        d = self._derived
        ms, hs = sprung_mh
        out = []
        for k_axis, axis in ((d['k_pitch'], 0), (d['k_roll'], 1)):
            k = k_axis * kf
            amp = st.lean_amplification(k, ms, hs)
            zeta = self._zeta(posed, axis, k, cf, ms, hs, masses)
            out.append((amp, st.overshoot(zeta), self._period))
        return out

    def _zeta(self, posed, axis, k, cf, ms, hs, masses=None) -> float:
        """
        Return the damping ratio of the body's pitch or roll mode.

        ``axis`` 0: pitch (motion along x); 1: roll (motion along y). The ratio is
        C / (2 sqrt((K - m g h) I)); the mode's period is kept in ``_period``. ``masses``
        (default nominal) scales the bodies and their own inertias.
        """
        if math.isinf(k):
            self._period = 0.0
            return 1.0
        coords = self._derived['spring_xyz']
        damp = self._derived['spring_damping'] * cf
        centre = coords[:, axis].mean()
        c_mode = float((damp * (coords[:, axis] - centre) ** 2).sum())
        sprung = self._derived['sprung']
        p = posed.positions[sprung]
        scale = (np.ones(len(posed.masses)) if masses is None
                 else np.asarray(masses) / posed.masses)[sprung]
        m = posed.masses[sprung] * scale
        about = 1 - axis                    # pitch turns about y, roll about x
        inertia = float((m * ((p[:, axis] - centre) ** 2 + (p[:, 2] - posed.plane_z) ** 2)).sum()
                        + (posed.inertias[sprung][:, about, about] * scale).sum())
        k_eff = k - ms * st.G * hs
        if k_eff <= 0.0 or inertia <= 0.0:
            self._period = math.inf
            return 0.0
        self._period = 2.0 * math.pi / math.sqrt(k_eff / inertia)
        return c_mode / (2.0 * math.sqrt(k_eff * inertia))

    def ring_period(self, posed, sprung_mh) -> float:
        """Return the longest period of the body's roll and pitch modes at the softest springs."""
        kf, cf = self._derived['corners']['stiff_low'][0]
        periods = [m[2] for m in self._modes(posed, sprung_mh, kf, cf,
                                             sm.as_arrays(self.mass)['mass_high'])]
        return min(max(periods), MAX_RING_MEMORY_S)

    def _stop(self, posed, arrays, edges, g_body, velocity, wz, sprung_mh, corners,
              pessimistic=False):
        """
        Return the smallest figure through a stop from the present motion, and its edge.

        Two phases: the load thrown toward the motion (amplification x (1 + overshoot)), and
        the swing back past rest (amplification x overshoot, the other way). Pessimistic: the
        worst over every corner of the springs' bounds and over the mass bounds.
        """
        best = None
        masses = arrays['mass_high'] if pessimistic else None
        for kf, cf in self._derived['corners'][corners]:
            (ax, ox, _), (ay, oy, _) = self._modes(posed, sprung_mh, kf, cf, masses)
            for fx, fy in ((ax * (1.0 + ox), ay * (1.0 + oy)), (-ax * ox, -ay * oy)):
                w = st.stop_vectors(posed.positions, g_body, velocity, wz,
                                    self.config.stop_decel_m_s2,
                                    self.config.stop_turn_decel_rad_s2, fx, fy)
                if pessimistic:
                    m, e = st.pessimistic_margin(posed.positions, posed.masses,
                                                 arrays['mass_low'], arrays['mass_high'],
                                                 arrays['com_bound'], w, posed.plane_z, edges)
                else:
                    fig = st.figure(posed.positions, posed.masses, w, posed.plane_z, edges)
                    m, e = fig.margin, fig.edge
                if best is None or m < best[0]:
                    best = (m, e)
        return best

    def _static_lean(self, posed, masses, corners, sprung_mh,
                     shift=(0.0, 0.0)) -> Tuple[float, float]:
        """
        Return the body's static lean on its springs (rad, toward +x and +y) on a level floor.

        The sprung centre of mass off the springs' centre leans the body toward it until the
        springs' moment balances: angle = m g offset / (K - m g h). Rigid: none. ``shift``
        moves the sprung centre of mass (x, y) first (a bound).
        """
        d = self._derived
        if not self.model.springs or math.isinf(d['k_roll']):
            return 0.0, 0.0
        kf, _ = d['corners'][corners][0]
        ms, hs = sprung_mh
        sprung = d['sprung']
        coords = d['spring_xyz']
        c = (masses[sprung][:, None] * posed.positions[sprung]).sum(axis=0) / masses[sprung].sum()
        c = c + np.array([shift[0], shift[1], 0.0])
        out = []
        for k_axis, axis in ((d['k_pitch'], 0), (d['k_roll'], 1)):
            k_eff = k_axis * kf - ms * st.G * hs
            if k_eff <= 0.0:
                raise ValueError('the springs cannot hold the body up (no stiffness left)')
            out.append(ms * st.G * (c[axis] - coords[:, axis].mean()) / k_eff)
        return out[0], out[1]

    def _level_pessimistic(self, posed, arrays, edges, a_arm, worst_sprung):
        """
        Return the smallest figure on an assumed slope toward each edge, and its edge.

        The floor is tilted toward each edge by the assumed bound, amplified by the lean on the
        softest springs, plus the static lean from the centre of mass (softest springs, heaviest
        sprung mass); the arm's own acceleration counts; the mass bounds as everywhere.
        """
        d = self._derived
        kf, _ = d['corners']['stiff_low'][0]
        ms, hs = worst_sprung
        bound = st.combined_com_bound(arrays['mass'], arrays['com_bound'])
        best = None
        for e in edges:
            out = np.array([-e.inward[0], -e.inward[1]])
            # The centre of mass toward this edge by its bound leans the body further that way.
            lean = self._static_lean(posed, arrays['mass_high'], 'stiff_low', worst_sprung,
                                     shift=bound * out)
            k_axis = d['k_pitch'] if abs(out[0]) >= abs(out[1]) else d['k_roll']
            amp = st.lean_amplification(k_axis * kf, ms, hs)
            theta = math.radians(self.config.assumed_slope_deg) * amp
            gx = math.sin(theta) * out[0] + math.sin(lean[0])
            gy = math.sin(theta) * out[1] + math.sin(lean[1])
            g = st.G * np.array([gx, gy, -math.sqrt(max(0.0, 1.0 - gx * gx - gy * gy))])
            w = st.gravity_vectors(len(posed.masses), g) - a_arm
            m, edge = st.pessimistic_margin(posed.positions, posed.masses, arrays['mass_low'],
                                            arrays['mass_high'], arrays['com_bound'], w,
                                            posed.plane_z, [e])
            if best is None or m < best[0]:
                best = (m, edge)
        return best


def reason_name(code: int) -> str:
    """Return the name of a reason code."""
    return REASON_NAMES[code] if 0 <= code < len(REASON_NAMES) else str(code)
