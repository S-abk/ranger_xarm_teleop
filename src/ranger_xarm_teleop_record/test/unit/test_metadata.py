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
"""Session layout helpers, labels, data-directory refusal, link numbers, provenance (no ROS)."""

import os
import subprocess

import pytest
from ranger_xarm_teleop_record import metadata as md
from ranger_xarm_teleop_record.linkstats import link_quality
from ranger_xarm_teleop_record.provenance import git_info, sha256_text


def git(path, *args):
    subprocess.run(['git', '-C', str(path), *args], check=True, capture_output=True)


def test_labels_round_trip_and_missing_parts(tmp_path):
    labels, problem = md.read_labels(str(tmp_path))
    assert labels is None and 'no labels file' in problem
    md.write_labels(str(tmp_path), ' test-task ', 'test-operator')
    expected = {'task': 'test-task', 'operator': 'test-operator'}
    assert md.read_labels(str(tmp_path)) == (expected, None)
    md.write_json_atomic(md.labels_path(str(tmp_path)), {'task': 'x', 'operator': ''})
    labels, problem = md.read_labels(str(tmp_path))
    assert labels is None and 'operator' in problem


def test_data_dir_refused_inside_the_code_checkout_even_where_ignored(tmp_path):
    code = tmp_path / 'code'
    code.mkdir()
    git(code, 'init', '-q')
    (code / '.gitignore').write_text('recordings/\n')
    assert md.data_dir_problem(str(code / 'recordings' / 'x'), code_root=str(code))
    assert md.data_dir_problem(str(tmp_path / 'outside'), code_root=str(code)) is None


def test_data_dir_in_another_work_tree_only_where_it_is_ignored(tmp_path):
    other = tmp_path / 'tooling'
    other.mkdir()
    git(other, 'init', '-q')
    (other / '.gitignore').write_text('/logs/\n')
    assert md.data_dir_problem(str(other / 'logs' / 'data'), code_root='/nonexistent') is None
    assert 'not ignored' in md.data_dir_problem(str(other / 'notes' / 'data'),
                                                code_root='/nonexistent')


def test_relative_inside_refuses_paths_outside_the_session(tmp_path):
    assert md.relative_inside(str(tmp_path), str(tmp_path / 'bag' / 'bag_0.mcap')) == \
        'bag/bag_0.mcap'
    with pytest.raises(ValueError):
        md.relative_inside(str(tmp_path / 's'), str(tmp_path / 'elsewhere'))


def test_missing_lists_required_keys_and_camera_keys():
    profile = {'streams': {'cam': {'kind': 'image', 'required': True, 'record': True,
                                   'camera': 'c1'}}}
    out = md.missing({'profile': profile})
    assert 'software.teleop_repo.commit' in out
    assert 'cameras.c1.intrinsics.k' in out and 'cameras.c1.extrinsics.in_mount' in out


def test_link_quality_counts_gaps_jitter_and_reconnects():
    seq = [1, 2, 3, 5, 6, 6, 7]
    xr = [0.0, 11.1, 22.2, 44.4, 55.5, 55.5, 66.6]
    stamps = [0.0, 0.011, 0.022, 0.044, 1.6, 1.6, 1.611]
    sid = [7, 7, 7, 7, 7, 8, 8]
    q = link_quality([0.005, 0.006, 0.2], seq, xr, stamps, sid, reconnect_gap_s=1.0)
    assert q['seq_gaps'] == 1 and q['seq_missing'] == 1 and q['seq_backwards'] == 1
    assert q['reconnects'] == 2 and q['session_changes'] == 1
    assert q['input_age_s']['max'] == pytest.approx(0.2)
    assert q['xr_frame_time']['median_period_ms'] == pytest.approx(11.1)


def test_git_info_names_no_path_and_sees_dirty_trees(tmp_path):
    repo = tmp_path / 'some_repo'
    repo.mkdir()
    git(repo, 'init', '-q')
    git(repo, '-c', 'user.email=t@example.com', '-c', 'user.name=t', 'commit', '-q',
        '--allow-empty', '-m', 'x')
    (repo / 'f.txt').write_text('a')
    git(repo, 'add', 'f.txt')
    info = git_info(str(repo / 'f.txt'))
    assert info['name'] == 'some_repo' and len(info['commit']) == 40 and info['dirty'] is True
    assert not any(os.sep in str(v) and str(v).startswith(os.sep) for v in info.values())
    assert git_info(str(tmp_path / 'none'))['commit'] is None
    assert sha256_text('a') == sha256_text('a') != sha256_text('b')
