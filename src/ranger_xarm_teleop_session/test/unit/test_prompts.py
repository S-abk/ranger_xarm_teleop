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
The operator's prompts: the labels and "continue anyway?".

The label prompt is the one place the launcher handles the label values: they go into the data
directory's labels.json and nowhere else (INV-7).
"""

import builtins
import json
import os
import secrets
import sys

import pytest
from ranger_xarm_teleop_session import cli
from ranger_xarm_teleop_session import logwatch as lw
from ranger_xarm_teleop_session import plan as pl
from ranger_xarm_teleop_session import preflight as pf
from ranger_xarm_teleop_session import supervisor as sv

RECORD = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ranger_xarm_teleop_record')
sys.path.insert(0, os.path.abspath(RECORD))


def supervisor(tmp_path, answers=(), policy='ask'):
    console = sv.Console(out=open(os.devnull, 'w'), log_path=str(tmp_path / 'launcher.log'))
    replies = list(answers)
    return sv.Supervisor(pl.Options(), console, checkout=str(tmp_path), mark='m',
                         state_file=str(tmp_path / 'state.json'), session='s',
                         log_dir=str(tmp_path), policy=policy,
                         ask=lambda _q: replies.pop(0))


def feed(monkeypatch, answers):
    replies = list(answers)
    asked = []

    def fake_input(question=''):
        asked.append(question)
        return replies.pop(0)
    monkeypatch.setattr(builtins, 'input', fake_input)
    return asked


def test_typed_labels_go_into_the_data_directory_only(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    task, operator = f'task-{secrets.token_hex(4)}', f'operator-{secrets.token_hex(4)}'
    asked = feed(monkeypatch, [task, operator])
    sup = supervisor(tmp_path)
    cli.ask_labels(sup, str(data))
    assert asked == ['Task label: ', 'Operator: ']
    with open(data / 'labels.json', encoding='utf-8') as handle:
        assert json.load(handle) == {'task': task, 'operator': operator}
    log = (tmp_path / 'launcher.log').read_text()
    assert 'labels written to the data directory' in log
    assert task not in log and operator not in log


def test_labels_already_set_are_kept_on_enter_and_replaced_on_new(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    feed(monkeypatch, ['first-task', 'first-operator'])
    cli.ask_labels(supervisor(tmp_path), str(data))
    asked = feed(monkeypatch, [''])
    cli.ask_labels(supervisor(tmp_path), str(data))
    assert len(asked) == 1 and 'Enter keeps them' in asked[0]
    assert json.loads((data / 'labels.json').read_text())['task'] == 'first-task'
    feed(monkeypatch, ['new', 'second-task', 'second-operator'])
    cli.ask_labels(supervisor(tmp_path), str(data))
    assert json.loads((data / 'labels.json').read_text())['task'] == 'second-task'


def test_one_empty_label_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    feed(monkeypatch, ['a-task', ''])
    cli.ask_labels(supervisor(tmp_path), str(data))
    assert not (data / 'labels.json').exists()
    assert 'both are needed' in (tmp_path / 'launcher.log').read_text()


MISSING_ITEM = pf.Item('disk', 'Disk space', pf.MISSING, '5 GB free', 'free space')


def test_continue_anyway_defaults_to_no(tmp_path):
    with pytest.raises(sv.Refused, match='by the operator'):
        supervisor(tmp_path, ['']).gate('at stack ready', [MISSING_ITEM])
    with pytest.raises(sv.Refused):
        supervisor(tmp_path, ['n']).gate('at stack ready', [MISSING_ITEM])
    supervisor(tmp_path, ['y']).gate('at stack ready', [MISSING_ITEM])       # goes on
    supervisor(tmp_path, []).gate('at stack ready', [])                      # nothing asked


def test_yes_goes_on_and_strict_stops(tmp_path):
    supervisor(tmp_path, policy='yes').gate('at stack ready', [MISSING_ITEM])
    with pytest.raises(sv.Refused, match='missing at at stack ready'):
        supervisor(tmp_path, policy='strict').gate('at stack ready', [MISSING_ITEM])


def test_a_closed_rviz_is_said_and_not_fatal():
    s = lw.LaunchState()
    s.feed('[INFO] [rviz2-6]: process started with pid [16]')
    s.feed('[INFO] [rviz2-6]: process has finished cleanly [pid 16]')
    assert s.ended_wrongly() == [] and s.viewers_ended() == ['rviz2-6']
    s.feed('[INFO] [rviz2-7]: process started with pid [17]')
    s.feed("[ERROR] [rviz2-7]: process has died [pid 17, exit code -11, cmd 'rviz2']")
    assert s.ended_wrongly() == [] and s.viewers_ended() == ['rviz2-6', 'rviz2-7']
