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
"""The monitor's state: never a figure it cannot stand behind (INV-4), the filters."""

import copy
import inspect
import math

import numpy as np
import pytest
from ranger_xarm_teleop_servo import stability as st
from ranger_xarm_teleop_servo import stability_mass as smass
from ranger_xarm_teleop_servo import stability_monitor as sm
from ranger_xarm_teleop_servo.stability_filters import (Cascade, ComplementaryTilt, Derivative,
                                                        JointAcceleration, LowPass)
from stability_urdf import make_urdf, MASS_FILE, WHEELS

G = st.G
JOINTS = ['arm_joint1', 'arm_joint2']
STEER = [f'{c}_steer_joint' for c in ('front_left', 'front_right', 'rear_left', 'rear_right')]
IMU_POS = np.array([-0.3, 0.0, 1.1])
WARM = 3.5                  # the tilt filter settles in its tau (3 s) after the IMU starts


def config(tilt=True, **kw):
    values = {'base_frame': 'base_link', 'wheel_links': WHEELS, 'tilt_configured': tilt,
              'cutoff_hz': 15.0, 'tilt_tau_s': 3.0, 'joint_timeout_s': 0.5,
              'odometry_timeout_s': 0.5, 'imu_timeout_s': 0.5, 'stop_decel_m_s2': 1.0,
              'stop_turn_decel_rad_s2': 3.5, 'assumed_slope_deg': None if tilt else 5.0,
              'tilt_hint': ' (start the simulator with sensors:=true)'}
    values.update(kw)
    return sm.Config(**values)


def feed(mon, t0, seconds, q1=math.pi / 2, v=(0.0, 0.0, 0.0), f=(0.0, 0.0, G), rate=(0, 0, 0),
         joints=True, odom=True, imu=True, dt=0.01):
    """Feed every input at 100 Hz from t0 for ``seconds``; return the end time."""
    t = t0
    while t < t0 + seconds - 1e-9:
        if joints:
            mon.on_joint_states(t, t, JOINTS, [q1, 0.0])
            mon.on_joint_states(t, t, STEER, [0.0] * 4, source_is_arm=False)
        if odom:
            mon.on_odometry(t, *v)
        if imu:
            mon.on_imu(t, t, f, rate)
        t += dt
    return t


def ready(tilt=True, **kw):
    mon = sm.Monitor(config(tilt, **kw), simulation=True)
    mon.set_description(make_urdf())
    mon.set_mass_data(copy.deepcopy(MASS_FILE))
    if tilt:
        mon.set_imu_frame(np.eye(3), IMU_POS)
    return mon


# ------------------------------------------------------------------------------- filters
def test_a_low_pass_lags_by_its_time_constant_and_ignores_an_old_stamp():
    lp = LowPass(10.0)
    lp.update(0.0, [0.0])
    tau = 1.0 / (2 * math.pi * 10.0)
    t, y = 0.0, None
    while t < tau - 1e-9:
        t += 0.0005
        y = lp.update(t, [1.0])
    assert y[0] == pytest.approx(1 - math.exp(-1), abs=0.01)
    assert lp.update(t - 0.1, [5.0]) is None


def test_a_derivative_of_a_ramp_is_its_slope_and_a_cascade_lags_like_it():
    d, c = Derivative(15.0), Cascade(15.0)
    for i in range(400):
        t = i * 0.005
        v = d.update(t, [2.0 * t])
        c.update(t, [2.0 * t])
    assert v[0] == pytest.approx(2.0, rel=1e-3)
    tau = 1.0 / (2 * math.pi * 15.0)
    assert 2.0 * 1.995 - c.value[0] == pytest.approx(2.0 * 2 * tau, rel=0.05)


def test_the_complementary_tilt_follows_the_gyro_and_settles_on_the_accelerometer():
    tilt = ComplementaryTilt(3.0)
    tilt.update(0.0, [0.0, 0.0, G], [0.0, 0.0, 0.0])
    # Roll at 0.5 rad/s (about +x) for 0.2 s with the accelerometer still reading level: the
    # gyro wins, and the roll reads +0.1 rad.
    t = 0.0
    for _ in range(100):
        t += 0.002
        down = tilt.update(t, [0.0, 0.0, G], [0.5, 0.0, 0.0])
    assert math.degrees(math.atan2(-down[1], -down[2])) == pytest.approx(
        math.degrees(0.1), abs=0.2)
    # Then held at a 10 degree slope: the accelerometer pulls it there with tau.
    th = math.radians(10.0)
    f = [0.0, -G * math.sin(th), G * math.cos(th)]
    for _ in range(10000):
        t += 0.002
        down = tilt.update(t, f, [0.0, 0.0, 0.0])
    assert down == pytest.approx(-np.array(f) / G, abs=1e-3)


def test_joint_accelerations_of_a_parabola():
    ja = JointAcceleration(15.0, ['a'])
    for i in range(1000):
        t = i * 0.005
        ja.update(t, {'a': 1.5 * t * t})
    q, v, a = ja.state()
    assert a['a'] == pytest.approx(3.0, rel=0.01)
    assert not ja.update(1.0, {'a': 0.0})                 # older than the last sample


# --------------------------------------------------------------------- INV-4: unavailable
def test_nothing_received_is_no_description():
    mon = sm.Monitor(config(), simulation=True)
    r = mon.cycle(0.0)
    assert not r.available and r.reason == sm.NO_DESCRIPTION and math.isnan(r.margin)


def test_a_description_without_wheels_is_unusable_and_says_why():
    mon = sm.Monitor(config(), simulation=True)
    mon.set_description(make_urdf().replace('front_left_wheel"', 'front_left_tyre"'))
    r = mon.cycle(0.0)
    assert r.reason == sm.DESCRIPTION_UNUSABLE and 'front_left_wheel' in r.reason_text


def test_no_mass_file_and_a_refused_one():
    mon = sm.Monitor(config(), simulation=True)
    mon.set_description(make_urdf())
    assert mon.cycle(0.0).reason == sm.NO_MASS_FILE
    data = copy.deepcopy(MASS_FILE)
    data['template'] = True
    mon.set_mass_data(data)
    r = mon.cycle(0.0)
    assert r.reason == sm.MASS_FILE_REFUSED and 'template' in r.reason_text
    mon.set_mass_data(None, error='cannot read /nowhere.yaml')
    assert mon.cycle(0.0).reason == sm.MASS_FILE_REFUSED
    mon = sm.Monitor(config(), simulation=False)
    mon.set_description(make_urdf())
    mon.set_mass_data(copy.deepcopy(MASS_FILE))          # simulation_only, not a simulation
    assert mon.cycle(0.0).reason == sm.MASS_FILE_REFUSED


def test_a_missing_joint_and_a_stale_one():
    mon = ready()
    t = feed(mon, 0.0, 1.0, joints=False)
    r = mon.cycle(t)
    assert r.reason == sm.NO_JOINT_STATES and 'arm_joint1' in r.reason_text
    assert r.fingerprint                                  # the model in force is known
    mon.on_joint_states(t, t, JOINTS, [0.0, 0.0])
    r = mon.cycle(t)
    assert r.reason == sm.NO_JOINT_STATES and 'steer' in r.reason_text
    t = feed(mon, t, WARM)
    assert mon.cycle(t).available
    t2 = feed(mon, t, 0.6, joints=False)
    r = mon.cycle(t2)
    assert r.reason == sm.JOINT_STATES_STALE


def test_no_odometry_and_stale_odometry():
    mon = ready()
    t = feed(mon, 0.0, 1.0, odom=False)
    assert mon.cycle(t).reason == sm.NO_ODOMETRY
    t = feed(mon, t, 1.0)
    t = feed(mon, t, 0.6, odom=False)
    assert mon.cycle(t).reason == sm.ODOMETRY_STALE


def test_a_configured_tilt_source_missing_or_stale_is_no_tilt_naming_the_launch_argument():
    mon = sm.Monitor(config(), simulation=True)
    mon.set_description(make_urdf())
    mon.set_mass_data(copy.deepcopy(MASS_FILE))
    mon.set_imu_frame(None, None, error='no transform from base_link to imu_link')
    t = feed(mon, 0.0, WARM)
    r = mon.cycle(t)
    assert r.reason == sm.NO_TILT and 'imu_link' in r.reason_text
    mon = ready()
    t = feed(mon, 0.0, 1.0, imu=False)
    r = mon.cycle(t)
    assert r.reason == sm.NO_TILT and 'sensors:=true' in r.reason_text
    t = feed(mon, t, WARM)
    assert mon.cycle(t).available
    t = feed(mon, t, 0.6, imu=False)
    r = mon.cycle(t)
    assert r.reason == sm.NO_TILT and 'older' in r.reason_text and 'sensors:=true' in r.reason_text


def test_a_non_finite_input_never_becomes_a_figure():
    mon = ready()
    t = feed(mon, 0.0, WARM)
    mon.on_joint_states(t, t + 0.001, JOINTS, [float('nan'), 0.0])
    r = mon.cycle(t + 0.002)
    assert not r.available and math.isnan(r.margin)
    # An IMU sample that is not finite is dropped, not filtered in.
    mon = ready()
    t = feed(mon, 0.0, WARM)
    before = mon.f_filter.value.copy()
    mon.on_imu(t, t + 0.001, [float('nan'), 0.0, G], [0.0, 0.0, 0.0])
    assert mon.f_filter.value == pytest.approx(before)


def test_without_a_tilt_source_a_figure_only_at_rest_marked_level_floor_assumed():
    mon = ready(tilt=False)
    t = feed(mon, 0.0, 1.0, v=(0.2, 0.0, 0.0), imu=False)
    r = mon.cycle(t)
    assert r.reason == sm.MOVING_WITHOUT_TILT and not r.available
    t = feed(mon, t, 1.0, imu=False)
    r = mon.cycle(t)
    assert r.available and r.tilt_source == sm.TILT_LEVEL_ASSUMED and r.at_rest
    # The pessimistic figure takes the assumed slope (5 deg, lean-amplified) toward the edge.
    assert r.margin - r.margin_pessimistic > 0.3 * math.tan(math.radians(5.0))


def test_the_imu_is_read_for_acceleration_and_rate_only():
    params = list(inspect.signature(sm.Monitor.on_imu).parameters)
    assert params == ['self', 'arrival', 'stamp', 'accel', 'rate']


# ----------------------------------------------------------------------------- the figures
def test_at_rest_on_a_level_floor_the_figure_is_the_static_distance():
    mon = ready()
    t = feed(mon, 0.0, WARM)
    r = mon.cycle(t)
    assert r.available and r.tilt_source == sm.TILT_MEASURED and r.reason == sm.OK
    posed = mon.model.pose({'arm_joint1': math.pi / 2}, mon.mass.nominal_bodies)
    c = (posed.masses[:, None] * posed.positions).sum(axis=0) / posed.masses.sum()
    assert r.margin == pytest.approx(0.2 - c[1], abs=1e-4)
    assert st.EDGE_NAMES[r.edge - 1] == 'left'
    assert r.margin_pessimistic < r.margin
    assert r.stop_margin == pytest.approx(r.margin, abs=1e-4)      # not moving
    assert r.stop_decel == 1.0 and r.note == sm.NOTE and len(r.fingerprint) == 64
    assert r.roll_deg == pytest.approx(0.0, abs=1e-6) and r.energy_reserve > 0


def test_a_sideways_acceleration_measured_by_the_imu_lowers_the_figure():
    mon = ready()
    t = feed(mon, 0.0, WARM)
    still = mon.cycle(t).margin
    # Accelerating right (-y) at 1 m/s2: the accelerometer reads f = a - g.
    t = feed(mon, t, 1.0, f=(0.0, -1.0, G), v=(0.0, -0.3, 0.0))
    r = mon.cycle(t)
    assert r.margin < still - 0.03 and st.EDGE_NAMES[r.edge - 1] == 'left'


def test_the_stop_figure_reads_lower_while_moving_and_carries_its_deceleration():
    mon = ready(stop_decel_m_s2=2.0)
    t = feed(mon, 0.0, WARM)
    still = mon.cycle(t)
    t = feed(mon, t, 1.0, v=(0.0, 0.25, 0.0))
    r = mon.cycle(t)
    assert r.stop_decel == 2.0
    assert r.stop_margin < still.margin - 0.05
    assert r.stop_margin_pessimistic <= r.stop_margin
    assert r.energy_reserve < still.energy_reserve


def test_a_rocking_body_holds_the_stop_figure_down_for_one_period():
    mon = ready()
    t = feed(mon, 0.0, WARM)
    still = mon.cycle(t)
    t = feed(mon, t, 0.15, f=(0.0, -3.0, G))
    dip = mon.cycle(t)
    t = feed(mon, t, 0.3)
    after = mon.cycle(t)
    assert after.margin > dip.margin + 0.05               # the present figure recovered
    assert after.stop_margin <= dip.margin + 1e-9          # the stop figure remembers
    t = feed(mon, t, 12.0)                                # the tilt filter's tau is 3 s
    assert mon.cycle(t).stop_margin == pytest.approx(still.stop_margin, abs=1e-3)


def test_a_first_imu_sample_of_zero_neither_crashes_nor_counts():
    mon = ready()
    t = feed(mon, 0.0, 1.0, imu=False)
    mon.on_imu(t, t, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    r = mon.cycle(t)
    assert r.reason == sm.NO_TILT and not r.available


def test_no_figure_until_the_filters_have_settled():
    mon = ready()
    t = feed(mon, 0.0, 1.0, imu=False)
    mon.on_imu(t, t, [0.0, 0.0, G], [0.0, 0.0, 0.0])
    r = mon.cycle(t)
    assert r.reason == sm.NO_TILT and 'settling' in r.reason_text
    mon = ready()
    mon.on_joint_states(0.0, 0.0, JOINTS, [0.0, 0.0])
    mon.on_joint_states(0.0, 0.0, STEER, [0.0] * 4, source_is_arm=False)
    t = feed(mon, 0.0, 1.0, joints=False)
    r = mon.cycle(t - 0.5)
    assert r.reason in (sm.NO_JOINT_STATES, sm.JOINT_STATES_STALE) and not r.available


def test_joint_values_whose_stamps_go_backwards_are_stale_not_a_figure():
    mon = ready()
    t = feed(mon, 0.0, WARM)
    assert mon.cycle(t).available
    # Fresh arrivals, old stamps: the filter takes none of them.
    for i in range(70):
        mon.on_joint_states(t + i * 0.01, 0.5, JOINTS, [0.3, 0.0])
        mon.on_joint_states(t + i * 0.01, t + i * 0.01, STEER, [0.0] * 4, source_is_arm=False)
        mon.on_odometry(t + i * 0.01, 0.0, 0.0, 0.0)
        mon.on_imu(t + i * 0.01, t + i * 0.01, [0.0, 0.0, G], [0.0, 0.0, 0.0])
    r = mon.cycle(t + 0.69)
    assert r.reason == sm.JOINT_STATES_STALE and 'filtered' in r.reason_text


def test_a_nan_joint_sample_is_not_a_figure_and_does_not_stick():
    mon = ready()
    t = feed(mon, 0.0, WARM)
    mon.on_joint_states(t, t + 0.001, JOINTS, [float('nan'), 0.0])
    assert not mon.cycle(t + 0.002).available
    t = feed(mon, t + 0.01, 1.0)
    assert mon.cycle(t).available


def test_the_mass_files_stiffness_value_is_used():
    """A file giving 8 kN/m gives the figures of a description whose springs are 8 kN/m."""
    by_file = ready()
    data = copy.deepcopy(MASS_FILE)
    data['suspension']['stiffness'] = {'value': 8000.0, 'plus': 0.0, 'minus': 0.0}
    data['suspension']['damping'] = {'value': 500.0, 'plus': 0.0, 'minus': 0.0}
    by_file.set_mass_data(data)
    by_description = ready()
    by_description.set_description(make_urdf().replace('20000.0', '8000.0'))
    plain = copy.deepcopy(MASS_FILE)
    plain.pop('suspension')
    by_description.set_mass_data(plain)
    stiff = ready()
    results = []
    for mon in (by_file, by_description, stiff):
        t = feed(mon, 0.0, WARM, v=(0.0, 0.25, 0.0))
        results.append(mon.cycle(t))
    assert results[0].stop_margin == pytest.approx(results[1].stop_margin, abs=1e-9)
    assert abs(results[0].stop_margin - results[2].stop_margin) > 1e-3


def test_the_pessimistic_stop_is_the_worst_corner_of_the_springs():
    mon = ready()
    t = feed(mon, 0.0, WARM, v=(0.0, 0.25, 0.0))
    r = mon.cycle(t)
    posed = mon.model.pose({n: v for n, (v, _) in mon.joints.items()}, mon.mass.nominal_bodies)
    edges = st.support_polygon(list(posed.contacts.values()))
    arrays = smass.as_arrays(mon.mass)
    sprung = mon._sprung(posed, arrays['mass'])
    k, c = mon.mass.stiffness, mon.mass.damping
    grid = [(kk / 20000.0, cc / 500.0) for kk in (k.low, k.value, k.high)
            for cc in (c.low, c.value, c.high)]                     # every corner, 3 x 3
    for kf, cf in grid:
        mon._derived['corners']['probe'] = [(kf, cf)]
        m, _ = mon._stop(posed, arrays, edges, st.G * mon.tilt.down, np.array([0.0, 0.25]),
                         0.0, sprung, 'probe')
        assert r.stop_margin_pessimistic <= m + 1e-9


def test_damping_ratio_and_period_of_the_roll_mode():
    mon = ready()
    t = feed(mon, 0.0, WARM)
    mon.cycle(t)
    posed = mon.model.pose({n: v for n, (v, _) in mon.joints.items()}, mon.mass.nominal_bodies)
    ms, hs = mon._sprung(posed, posed.masses)
    k = 4 * 20000.0 * 0.2 ** 2
    sprung = mon._derived['sprung']
    p, m = posed.positions[sprung], posed.masses[sprung]
    inertia = ((m * (p[:, 1] ** 2 + (p[:, 2] + 0.3) ** 2)).sum()
               + posed.inertias[sprung][:, 0, 0].sum())
    zeta = mon._zeta(posed, 1, k, 1.0, ms, hs)
    k_eff = k - ms * G * hs
    assert zeta == pytest.approx(4 * 500.0 * 0.2 ** 2 / (2 * math.sqrt(k_eff * inertia)))
    assert mon._period == pytest.approx(2 * math.pi / math.sqrt(k_eff / inertia))


def test_an_imu_mounted_turned_gives_the_same_figure():
    aligned = ready()
    turned = ready()
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # IMU x = base y
    turned.set_imu_frame(rot, IMU_POS)
    t = feed(aligned, 0.0, WARM, f=(0.3, -0.8, G))
    feed(turned, 0.0, WARM, f=tuple(rot.T @ np.array([0.3, -0.8, G])))
    assert turned.cycle(t).margin == pytest.approx(aligned.cycle(t).margin, abs=1e-9)


def test_a_steady_turn_measured_by_an_imu_off_the_axis():
    mon = ready()
    omega = 1.0
    a_imu = -omega ** 2 * np.array([IMU_POS[0], IMU_POS[1], 0.0])
    f = tuple(a_imu + np.array([0.0, 0.0, G]))
    t = feed(mon, 0.0, WARM, f=f, rate=(0.0, 0.0, omega), v=(0.0, 0.0, omega))
    r = mon.cycle(t)
    posed = mon.model.pose({n: v for n, (v, _) in mon.joints.items()}, mon.mass.nominal_bodies)
    g = np.array([0.0, 0.0, -G])
    r_xy = np.column_stack([posed.positions[:, 0], posed.positions[:, 1],
                            np.zeros(len(posed.masses))])
    w = g[None, :] + omega ** 2 * r_xy
    hdot = st.angular_momentum_rate(posed.inertias, np.array([0.0, 0.0, omega]), np.zeros(3))
    edges = st.support_polygon(list(posed.contacts.values()))
    direct = st.figure(posed.positions, posed.masses, w, posed.plane_z, edges, hdot)
    assert r.margin == pytest.approx(direct.margin, abs=1e-4)


def test_without_a_tilt_source_the_static_lean_toward_the_centre_of_mass_counts():
    rigid_model = ready(tilt=False)
    t = feed(rigid_model, 0.0, 1.0, imu=False)
    r = rigid_model.cycle(t)
    posed = rigid_model.model.pose({'arm_joint1': math.pi / 2}, rigid_model.mass.nominal_bodies)
    c = (posed.masses[:, None] * posed.positions).sum(axis=0) / posed.masses.sum()
    assert r.available and r.margin < 0.2 - c[1] - 1e-4      # leaning toward the arm's side


def test_after_an_imu_outage_the_filters_start_again_and_settle_before_a_figure():
    mon = ready()
    t = feed(mon, 0.0, WARM)
    assert mon.cycle(t).available
    t = feed(mon, t, 0.6, imu=False)                      # a stall: 0.6 s without the IMU
    assert mon.cycle(t).reason == sm.NO_TILT
    t = feed(mon, t, 0.1, rate=(0.0, 0.0, 0.2))
    r = mon.cycle(t)
    assert r.reason == sm.NO_TILT and 'settling' in r.reason_text
    assert mon.imu_restarts == 1
    t = feed(mon, t, WARM, rate=(0.0, 0.0, 0.2))
    assert mon.cycle(t).available


def test_a_short_gap_in_the_joints_restarts_their_filter():
    mon = ready()
    t = feed(mon, 0.0, WARM)
    assert mon.cycle(t).available
    t = feed(mon, t, 0.3, joints=False)                   # under the 0.5 s timeout
    t = feed(mon, t, 0.02)
    r = mon.cycle(t)
    assert not r.available and 'settling' in r.reason_text and mon.arm.restarts == 1
    t = feed(mon, t, 0.5)
    assert mon.cycle(t).available


def test_the_energy_reserve_is_taken_on_the_pessimistic_mass_model():
    mon = ready()
    t = feed(mon, 0.0, WARM, v=(0.0, 0.25, 0.0))
    r = mon.cycle(t)
    posed = mon.model.pose({n: v for n, (v, _) in mon.joints.items()}, mon.mass.nominal_bodies)
    edges = st.support_polygon(list(posed.contacts.values()))
    nominal, _ = st.energy_reserve(posed.positions, posed.masses, posed.inertias,
                                   posed.plane_z, edges, st.G * mon.tilt.down,
                                   np.array([0.0, 0.25]), 0.0)
    assert r.energy_reserve < nominal - 0.5


def test_without_a_tilt_source_the_centre_of_mass_bound_lowers_the_pessimistic_figure():
    figures = []
    for com in (0.0, 0.03):
        mon = ready(tilt=False)
        data = copy.deepcopy(MASS_FILE)
        data['default_uncertainty']['com_m'] = com
        data['bodies']['chassis']['com']['bound'] = com
        mon.set_mass_data(data)
        t = feed(mon, 0.0, 1.0, imu=False)
        figures.append(mon.cycle(t).margin_pessimistic)
    assert figures[1] < figures[0] - 0.03                 # the bound itself, and the lean it adds
