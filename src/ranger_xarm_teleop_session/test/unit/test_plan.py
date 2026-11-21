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
"""The launcher starts exactly the manual sequence's commands, with the same arguments (INV-2)."""

import os

import pytest
from ranger_xarm_teleop_session import plan as pl
import yaml

PROFILES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                            'ranger_xarm_teleop_record', 'profiles')
CHECKOUT = '/co'

# The manual sequence (README "Run the headset input bridge", "Simulator rendering" and the
# teleop section; record report section 7; reset report section 7), one command per terminal.
# record_session and record_data_dir are left out: by hand the recorder makes the session name
# itself (the same form) and the profile's data directory is the default.
MANUAL = {
    'gazebo': ['/co/tools/relay.sh',
               'ros2 launch ranger_xarm_gazebo sim.launch.py sensors:=true gpu:=nvidia',
               'ros2 launch ranger_xarm_teleop_bringup teleop_sim.launch.py reset:=true '
               'record:=true',
               'ros2 launch ranger_xarm_teleop_bridge bridge.launch.py'],
    'gazebo_headless': ['/co/tools/relay.sh',
                        'ros2 launch ranger_xarm_gazebo sim.launch.py headless:=true '
                        'sensors:=true gpu:=nvidia',
                        'ros2 launch ranger_xarm_teleop_bringup teleop_sim.launch.py '
                        'reset:=true record:=true',
                        'ros2 launch ranger_xarm_teleop_bridge bridge.launch.py'],
    'mock': ['/co/tools/relay.sh',
             'ros2 launch ranger_xarm_teleop_bringup teleop_mock.launch.py reset:=true '
             'record:=true',
             'ros2 launch ranger_xarm_teleop_bridge bridge.launch.py'],
}
SESSION_ONLY = ('record_session', 'record_data_dir')


def record_profile(name):
    with open(os.path.join(PROFILES_DIR, f'{name}.yaml'), encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def long_running(steps):
    return [s for s in steps if not s.runs_to_end]


def same_command(step, manual):
    words = manual.split()
    assert [a for a in step.argv if ':=' not in a] == [w for w in words if ':=' not in w]
    got = {k: v for k, v in pl.args_of(step.argv).items() if k not in SESSION_ONLY}
    assert got == pl.args_of(words), f'{step.name}: {step.argv} vs {manual}'


@pytest.mark.parametrize('case,profile,headless', [('gazebo', 'gazebo', False),
                                                   ('gazebo_headless', 'gazebo', True),
                                                   ('mock', 'mock', False)])
def test_the_long_running_steps_are_the_manual_commands(case, profile, headless):
    opts = pl.Options(profile=profile, headless=headless)
    steps = pl.build_plan(opts, checkout=CHECKOUT, bringup_share='/share', session='S',
                          log_dir='/logs', gpu='nvidia' if profile == 'gazebo' else None)
    procs = long_running(steps)
    assert len(procs) == len(MANUAL[case])
    for step, manual in zip(procs, MANUAL[case]):
        same_command(step, manual)


def test_the_two_checks_are_the_documented_ones():
    steps = pl.build_plan(pl.Options(profile='gazebo'), checkout=CHECKOUT,
                          bringup_share='/share', session='S', log_dir='/logs', gpu='nvidia')
    check = next(s for s in steps if s.name == 'sim_check')
    assert check.argv[:5] == ('ros2', 'run', 'ranger_xarm_teleop_record', 'record_commission',
                              '--simulator')
    assert '/share/config/SIMULATION_ONLY_limits.yaml' in check.argv
    probe = steps[-1]
    assert probe.name == 'probe' and probe.runs_to_end
    assert {'--record', '--feedback', '--reset'} <= set(probe.argv)


def test_start_order_and_stop_order():
    steps = pl.build_plan(pl.Options(profile='gazebo'), checkout=CHECKOUT,
                          bringup_share='/share', session='S', log_dir='/logs', gpu='nvidia')
    assert [s.name for s in steps] == ['relay', 'simulator', 'sim_check', 'stack', 'bridge',
                                       'probe']
    # The input first (the arm holds on stale input), the relay last.
    assert [s.name for s in pl.stop_sequence(long_running(steps))] == [
        'bridge', 'stack', 'simulator', 'relay']


def test_the_recorder_gets_the_session_name_and_the_data_directory():
    opts = pl.Options(profile='mock', data_dir='/data')
    stack = next(s for s in pl.build_plan(opts, checkout=CHECKOUT, session='20261121-1-ab')
                 if s.name == 'stack')
    assert pl.args_of(stack.argv)['record_session'] == '20261121-1-ab'
    assert pl.args_of(stack.argv)['record_data_dir'] == '/data'


def test_parts_switched_off_are_passed_as_off():
    opts = pl.Options(profile='mock', record=False, feedback=False, reset=False, rviz=False)
    steps = {s.name: s for s in pl.build_plan(opts, checkout=CHECKOUT, session='S')}
    args = pl.args_of(steps['stack'].argv)
    assert args == {'feedback': 'false', 'rviz': 'false'}       # record and reset default off
    assert pl.args_of(steps['bridge'].argv) == {'feedback': 'false'}
    assert not {'--record', '--feedback', '--reset'} & set(steps['probe'].argv)


def test_a_relay_elsewhere_starts_no_relay_and_tells_the_bridge():
    opts = pl.Options(profile='mock', relay_url='ws://10.0.0.5:8443/ws')
    steps = {s.name: s for s in pl.build_plan(opts, checkout=CHECKOUT, session='S')}
    assert 'relay' not in steps
    assert pl.args_of(steps['bridge'].argv) == {'relay_url': 'ws://10.0.0.5:8443/ws'}


def test_the_relay_listens_on_this_machine_unless_asked():
    assert pl.relay_step(pl.Options(), CHECKOUT).argv == ('/co/tools/relay.sh',)
    asked = pl.relay_step(pl.Options(listen='192.168.1.4', port=9000), CHECKOUT).argv
    assert asked == ('/co/tools/relay.sh', '--host', '192.168.1.4', '--port', '9000')
    assert pl.Options().listen == '127.0.0.1'


def test_listening_beyond_this_machine_is_said_in_one_line():
    assert pl.listen_warning(pl.Options()) is None
    assert pl.listen_warning(pl.Options(role='relay', listen='127.0.0.1')) is None
    line = pl.listen_warning(pl.Options(role='relay', listen='192.168.1.4'))
    assert 'no authentication' in line and 'anything on that network can connect' in line
    assert '\n' not in line
    assert pl.listen_warning(pl.Options(listen='0.0.0.0')) is not None     # stack role, here
    assert pl.listen_warning(pl.Options(listen='0.0.0.0', relay_url='ws://x:1/ws')) is None


def test_the_bridge_reaches_a_relay_listening_on_another_address():
    steps = {s.name: s for s in pl.build_plan(pl.Options(listen='10.1.2.3'), checkout=CHECKOUT,
                                              session='S')}
    assert pl.args_of(steps['bridge'].argv) == {'relay_url': 'ws://10.1.2.3:8443/ws'}
    steps = {s.name: s for s in pl.build_plan(pl.Options(listen='0.0.0.0'), checkout=CHECKOUT,
                                              session='S')}
    assert pl.args_of(steps['bridge'].argv) == {}           # 127.0.0.1 reaches 0.0.0.0


def test_an_ipv6_listen_address_is_refused():
    with pytest.raises(pl.PlanError, match='IPv4 address or a host name'):
        pl.check_options(pl.Options(role='relay', listen='::1'), None)


def test_the_relay_role_starts_the_relay_only():
    steps = pl.build_plan(pl.Options(role='relay'), checkout=CHECKOUT)
    assert [s.name for s in steps] == ['relay']


def test_simulation_profiles_are_accepted():
    for name in pl.PROFILES:
        assert pl.check_options(pl.Options(profile=name), record_profile(name)) == []


def test_a_profile_not_marked_simulation_is_refused():
    hardware = dict(record_profile('gazebo'), simulation=False)
    with pytest.raises(pl.PlanError, match='simulation profiles only'):
        pl.check_options(pl.Options(profile='gazebo'), hardware)
    with pytest.raises(pl.PlanError, match='simulation profiles only'):
        pl.check_options(pl.Options(profile='gazebo'), None)


def test_an_unknown_profile_is_refused_and_names_the_hardware_path():
    with pytest.raises(pl.PlanError, match='hardware bring-up is started by hand'):
        pl.check_options(pl.Options(profile='hardware'), {'simulation': False})


def test_recording_without_the_cameras_the_profile_requires_is_refused_before_the_start():
    opts = pl.Options(profile='gazebo', cameras=False)
    with pytest.raises(pl.PlanError, match='camera_gantry, camera_wrist'):
        pl.check_options(opts, record_profile('gazebo'))
    opts = pl.Options(profile='gazebo', cameras=False, record=False)
    assert pl.check_options(opts, record_profile('gazebo')) == []


def test_options_that_do_not_apply_are_noted():
    opts = pl.Options(profile='mock', headless=True, explicit=('headless',))
    assert any('headless' in w for w in pl.check_options(opts, record_profile('mock')))
    opts = pl.Options(role='relay', record=False, explicit=('record',))
    assert any('--record' in w for w in pl.check_options(opts, None))


def test_a_bad_relay_url_and_relay_options_with_it_are_refused():
    with pytest.raises(pl.PlanError, match='ws://HOST:PORT/ws'):
        pl.check_options(pl.Options(relay_url='http://x'), record_profile('mock'))
    with pytest.raises(pl.PlanError, match='--listen and --port'):
        pl.check_options(pl.Options(relay_url='ws://x:8443/ws', explicit=('listen',)),
                         record_profile('mock'))
    with pytest.raises(pl.PlanError, match='starts one here'):
        pl.check_options(pl.Options(role='relay', relay_url='ws://x:8443/ws'), None)


def test_gpu_choice():
    assert pl.gpu_choice('auto', True, True)[0] == 'nvidia'
    gpu, text = pl.gpu_choice('auto', False, True)
    assert gpu is None and 'no NVIDIA card found' in text
    assert pl.gpu_choice('auto', True, False)[0] is None          # nothing renders
    assert pl.gpu_choice('default', True, True)[0] is None
    assert 'will refuse' in pl.gpu_choice('nvidia', False, True)[1]
    with pytest.raises(pl.PlanError):
        pl.gpu_choice('intel', True, True)


def test_no_label_or_name_reaches_any_command_line():
    # Labels are typed at a prompt and written to the data directory only: the plan takes
    # no label, so no command line can carry one.
    fields = set(pl.Options.__dataclass_fields__)
    assert not {'task', 'operator', 'label', 'labels'} & fields


def test_session_names_have_the_recorders_form():
    assert pl.session_name(0.0, 'abcd') == '19700101-000000-abcd'
