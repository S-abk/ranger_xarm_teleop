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
"""The optional stability key of the sidecar: absent without a mass file (no ROS)."""

import os

from ranger_xarm_teleop_record import metadata, provenance
import synth


def test_without_a_mass_file_there_is_no_record_and_nothing_is_copied(tmp_path):
    assert provenance.stability_record('', str(tmp_path)) is None
    assert os.listdir(tmp_path) == []


def test_with_a_mass_file_the_record_holds_its_hash_and_the_session_gets_a_copy(tmp_path):
    source = tmp_path / 'in' / 'my_mass.yaml'
    source.parent.mkdir()
    source.write_text('mass: 12.5\n')
    session = tmp_path / 'session'
    session.mkdir()
    record = provenance.stability_record(str(source), str(session))
    assert record == {'mass_file': 'stability_mass.yaml',
                      'mass_file_sha256': provenance.sha256_file(str(source)),
                      'source_name': 'my_mass.yaml'}
    assert (session / 'stability_mass.yaml').read_text() == 'mass: 12.5\n'
    assert record['mass_file_sha256'] == provenance.sha256_text('mass: 12.5\n')
    assert str(tmp_path) not in repr(record)             # no directory of the machine


def test_an_unreadable_mass_file_is_reported_not_swallowed(tmp_path):
    record = provenance.stability_record(str(tmp_path / 'absent.yaml'), str(tmp_path))
    assert record['mass_file'] is None and record['mass_file_sha256'] is None
    assert 'absent.yaml' in record['error']
    assert not (tmp_path / 'stability_mass.yaml').exists()


def test_the_key_is_optional_it_is_not_required_and_its_presence_changes_nothing():
    assert not any(k.startswith('stability') for k in metadata.REQUIRED)
    ep = synth.build()
    assert 'stability' not in ep.sidecar
    assert metadata.missing(ep.sidecar) == []
    ep.sidecar['stability'] = {'mass_file': 'stability_mass.yaml',
                               'mass_file_sha256': 'f' * 64}
    assert metadata.missing(ep.sidecar) == []
