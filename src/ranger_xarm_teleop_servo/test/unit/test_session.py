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
"""INV-2, INV-3: when a recording may start and stop, when a recovery may move; the rows."""

from types import SimpleNamespace as NS

from ranger_xarm_teleop_servo import session


def test_recording_starts_only_on_fresh_evidence_of_a_released_clutch():
    watch = session.ClutchWatch()
    allowed, why = watch.may_start(10.0)
    assert not allowed and '/teleop/state' in why
    watch.update(10.0, True)
    allowed, why = watch.may_start(10.1)
    assert not allowed and 'clutch is engaged' in why
    watch.update(10.2, False)
    assert watch.may_start(10.3) == (True, '')
    assert not watch.may_start(10.2 + session.STATE_FRESH_S + 0.01)[0]      # evidence too old


def test_recording_never_stops_while_the_clutch_is_engaged():
    watch = session.ClutchWatch()
    assert watch.may_stop(0.0)                  # no teleop node: nothing to disturb
    watch.update(1.0, True)
    assert not watch.may_stop(1.5)
    watch.update(2.0, True)
    assert not watch.may_stop(2.9)
    watch.update(3.0, False)
    assert watch.may_stop(3.1)
    watch.update(4.0, True)
    assert watch.may_stop(4.0 + session.STATE_FRESH_S + 0.01)   # teleop node gone silent


def test_recovery_moves_only_when_moveit_owns_the_arm_and_teleop_is_disarmed():
    assert session.may_recover(session.OWNER_MOVEIT, False) == (True, '')
    assert session.may_recover(session.OWNER_MOVEIT, None) == (True, '')    # no teleop node
    for owner, name in ((session.OWNER_TELEOP, 'TELEOP'), (session.OWNER_NONE, 'NONE'),
                        (session.OWNER_SWITCHING, 'SWITCHING'), (None, 'not reported')):
        for armed in (True, False, None):
            allowed, why = session.may_recover(owner, armed)
            assert not allowed and name in why
    allowed, why = session.may_recover(session.OWNER_MOVEIT, True)
    assert not allowed and 'armed' in why


def test_rejection_reason_names_the_owner_and_the_way_out():
    text = session.rejection_reason(session.OWNER_TELEOP)
    assert 'teleop owns the arm' in text and 'disarm first' in text
    assert '/teleop/disarm' in text and 'recover_arm.py' in text
    assert 'handed over' in session.rejection_reason(session.OWNER_SWITCHING)
    assert 'no arm controller is active' in session.rejection_reason(session.OWNER_NONE)
    assert '--controller' in session.rejection_reason(session.OWNER_MOVEIT)
    assert 'no owner' in session.rejection_reason(None)


def header(sec=5, nanosec=7):
    return NS(stamp=NS(sec=sec, nanosec=nanosec), frame_id='base')


def pose(x=1.0):
    return NS(position=NS(x=x, y=2.0, z=3.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))


def test_rows():
    state = NS(header=header(), motion_enabled=True, armed=True, clutch_engaged=True, owner=2,
               mux_status=0, hold_reason=0, active_limit=4, servo_status=1, calibrated=True,
               input_age=0.01, input_seq=42, measured_tool_pose_valid=True,
               measured_tool_pose=pose(), conditioning=0.25, damped=True, damping=0.3,
               solver_error_translation=0.004, solver_error_rotation=0.02,
               lead_translation=0.03, lead_rotation=0.1, commanded_linear_speed=0.2,
               commanded_angular_speed=0.5)
    row = session.state_row(state)
    assert row == {'k': 'state', 'stamp': 5_000_000_007, 'motion': True, 'armed': True,
                   'engaged': True, 'owner': 2, 'mux': 0, 'hold': 0, 'limit': 4, 'servo': 1,
                   'calibrated': True, 'age': 0.01, 'seq': 42,
                   'tool': [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]],
                   'conditioning': 0.25, 'damped': True, 'damping': 0.3,
                   'solver_error_translation': 0.004, 'solver_error_rotation': 0.02,
                   'lead_translation': 0.03, 'lead_rotation': 0.1,
                   'commanded_linear_speed': 0.2, 'commanded_angular_speed': 0.5}
    state.measured_tool_pose_valid = False
    assert session.state_row(state)['tool'] is None
    assert session.pose_row('target', NS(header=header(), pose=pose(4.0))) == {
        'k': 'target', 'stamp': 5_000_000_007, 'frame': 'base', 'p': [4.0, 2.0, 3.0],
        'q': [0.0, 0.0, 0.0, 1.0]}
    assert session.pose_row('head', NS(header=header(), pose=pose(0.5))) == {
        'k': 'head', 'stamp': 5_000_000_007, 'frame': 'base', 'p': [0.5, 2.0, 3.0],
        'q': [0.0, 0.0, 0.0, 1.0]}
    assert session.command_row(NS(header=header(), joint_names=['a', 'b'],
                                  velocities=[0.5, -1.0], duration=0.0)) == {
        'k': 'command', 'stamp': 5_000_000_007, 'names': ['a', 'b'], 'v': [0.5, -1.0]}
    assert session.command_row(NS(header=header(), joint_names=[], velocities=[])) == {
        'k': 'command', 'stamp': 5_000_000_007, 'names': [], 'v': []}
    assert session.servo_row(NS(code=2, message='stop')) == {'k': 'servo', 'code': 2,
                                                             'msg': 'stop'}
    assert session.owner_row(NS(owner=2, status=0, detail='')) == {
        'k': 'owner', 'owner': 2, 'status': 0, 'detail': ''}
    assert session.description_row('<robot/>') == {'k': 'description', 'urdf': '<robot/>'}


def test_joint_row_picks_the_arm_joints_in_order():
    names = ['wheel', *reversed(session.ARM_JOINTS)]
    msg = NS(header=header(), name=names, position=[9.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
             velocity=[0.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    row = session.joints_row(msg)
    assert row['q'] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] and row['v'][0] == 0.1
    msg.velocity = []
    assert session.joints_row(msg)['v'] is None
    assert session.joints_row(NS(header=header(), name=['wheel'], position=[0.0],
                                 velocity=[])) is None
