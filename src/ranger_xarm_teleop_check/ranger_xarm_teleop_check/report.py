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
The check report: where it goes, what it holds, how it prints.

Pure; the only effects are the git calls of ``code_commit`` and the file ``write_report``
writes. The data-directory refusal is the recorder's own ``data_dir_problem``.
"""

import datetime
import math
import os
import subprocess
from typing import List, Optional

from ranger_xarm_teleop_record.metadata import data_dir_problem, write_json_atomic

from .rules import Result

# The first line the tool prints, before anything else.
BANNER = ('read-only: this tool only subscribes; it creates no publisher, client, action or '
          'parameter call')
DEFAULT_DATA_DIR = '~/teleop_data'
REPORT_FOLDER = 'hardware_checks'


def report_dir(data_dir: str) -> str:
    """Return the folder that holds the reports of a data directory."""
    return os.path.join(os.path.abspath(os.path.expanduser(data_dir)), REPORT_FOLDER)


def report_dir_problem(data_dir: str, code_root: Optional[str] = None) -> Optional[str]:
    """Return why reports may not be written under ``data_dir``, or None."""
    return data_dir_problem(report_dir(data_dir), code_root)


def report_path(data_dir: str, stage: str, when: datetime.datetime) -> str:
    """Return ``<data dir>/hardware_checks/<YYYYmmdd-HHMMSS>_<stage>.json``."""
    return os.path.join(report_dir(data_dir), f"{when.strftime('%Y%m%d-%H%M%S')}_{stage}.json")


def code_commit(here: Optional[str] = None) -> str:
    """Return the commit of the checkout this file is in, or 'unknown'."""
    folder = os.path.dirname(os.path.realpath(here or __file__))
    try:
        done = subprocess.run(['git', '-C', folder, 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=10.0, check=False)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    out = done.stdout.strip()
    return out if done.returncode == 0 and out else 'unknown'


def host_clock(now: Optional[datetime.datetime] = None) -> dict:
    """Return the host's clock reading: local time with its offset from UTC, and the epoch."""
    local = (now or datetime.datetime.now().astimezone()).astimezone()
    return {'local_time': local.isoformat(timespec='milliseconds'),
            'utc_offset': local.strftime('%z'), 'epoch_s': local.timestamp()}


def sanitize(value):
    """Return ``value`` with NaN and infinity replaced by None (JSON has neither)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def build_report(stage: str, start: datetime.datetime, end: datetime.datetime,
                 measures: dict, results: List[Result], versions: dict, argv: List[str],
                 warnings: List[str], commit: str, exit_code: int) -> dict:
    """Return the report: stage, times, clock, measures, rules, versions, commit, argv."""
    return sanitize({
        'stage': stage,
        'start_time': start.isoformat(timespec='milliseconds'),
        'end_time': end.isoformat(timespec='milliseconds'),
        'host_clock': host_clock(end),
        'measures': measures,
        'rules': [r.as_dict() for r in results],
        'exit_status': exit_code,
        'warnings': warnings,
        'versions': versions,
        'code_commit': commit,
        'argv': list(argv),
    })


def write_report(path: str, report: dict) -> str:
    """Write the report atomically and return its path."""
    write_json_atomic(path, report)
    return path


def format_results(results: List[Result]) -> str:
    """Return the printed result lines, one rule each (advice on a second line)."""
    return '\n'.join(r.line() for r in results)
