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
JSONL recording format and I/O (no ROS, INV-6).

One JSON object per line: ``{"t_recv": <unix seconds>, "msg": <decoded message>}``
when the received text was strict JSON, else ``{"t_recv": ..., "raw": "<text>"}``.
Feeding ``Record.text()`` to ``frame_parse.parse_text`` reproduces what the
bridge saw.
"""

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Iterator, List, Optional

from .frame_parse import loads_strict


@dataclass(frozen=True)
class Record:
    """One recorded relay message; ``raw`` is set instead of ``msg`` for non-JSON text."""

    t_recv: float
    msg: Any = None
    raw: Optional[str] = None

    def text(self) -> str:
        """Return the relay text to feed to the parser."""
        if self.raw is not None:
            return self.raw
        return json.dumps(self.msg, separators=(',', ':'))


def _dump_line(obj: dict) -> str:
    """Serialise one record line, falling back to ASCII escapes if UTF-8 cannot hold it."""
    line = json.dumps(obj, ensure_ascii=False, allow_nan=False)
    try:
        line.encode('utf-8')
    except UnicodeEncodeError:
        line = json.dumps(obj, ensure_ascii=True, allow_nan=False)
    return line


class RecordWriter:
    """Write relay messages to a JSONL file, flushing every line."""

    def __init__(self, path: str) -> None:
        self._path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(parent, exist_ok=True)
        self._file = open(self._path, 'w', encoding='utf-8', newline='\n')

    def __enter__(self) -> 'RecordWriter':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write_text(self, text: str, t_recv: Optional[float] = None) -> None:
        """Append one received text message; ``t_recv`` defaults to the current time."""
        stamp = time.time() if t_recv is None else float(t_recv)
        try:
            line = _dump_line({'t_recv': stamp, 'msg': loads_strict(text)})
        except (ValueError, RecursionError):
            # Not strict JSON, or not representable as strict JSON (e.g. 1e999).
            line = _dump_line({'t_recv': stamp, 'raw': text})
        self._file.write(line + '\n')
        self._file.flush()

    def close(self) -> None:
        """Close the file; safe to call twice."""
        if not self._file.closed:
            self._file.close()


def _parse_line(line: str, where: str) -> Record:
    try:
        obj = loads_strict(line)
    except (ValueError, RecursionError) as err:
        raise ValueError(f'{where}: not valid JSON: {err}') from err
    if not isinstance(obj, dict):
        raise ValueError(f'{where}: expected a JSON object')
    t_recv = obj.get('t_recv')
    if not isinstance(t_recv, (int, float)) or isinstance(t_recv, bool):
        raise ValueError(f'{where}: t_recv: expected a number')
    if 'raw' in obj:
        if not isinstance(obj['raw'], str):
            raise ValueError(f'{where}: raw: expected a string')
        return Record(t_recv=float(t_recv), raw=obj['raw'])
    if 'msg' not in obj:
        raise ValueError(f'{where}: expected a "msg" or "raw" key')
    return Record(t_recv=float(t_recv), msg=obj['msg'])


def iter_records(path: str) -> Iterator[Record]:
    """Yield the records of a JSONL file; raise ``ValueError('<path>:<line>: ...')``."""
    path = os.fspath(path)
    with open(path, encoding='utf-8') as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield _parse_line(line, f'{path}:{lineno}')


def read_records(path: str) -> List[Record]:
    """Return all records of a JSONL file."""
    return list(iter_records(path))
