#!/usr/bin/env node
// Copyright 2026 Shuaib Olanrewaju
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
'use strict';
// Replays relay messages through the headset feedback logic (feedback.js) and prints what the
// panel would say. Usage: node panel_replay.js --session <xr session id> < stream.jsonl
//
// Input, one JSON object per line, in time order (t_ms is the page time, performance.now(), at
// which the message reached the page; it is also "now" for the echo age):
//   {"t_ms": 1000, "msg": {...}}   a relay message (teleop_status or teleop_haptic)
//   {"query_ms": 1500}             print the panel as it would be drawn at that time
//   {"tick_ms": 1500}              print the vibration pulses a frame at that time would start
// Output, one JSON line per query:
//   {"t_ms": 1500, "stale": false, "lines": ["text", ...]}
//   {"t_ms": 1500, "pulses": [{"hand": ..., "intensity": ..., "duration": ...}]}
const fs = require('node:fs');
const path = require('node:path');

const FB = require(path.join(__dirname, '..', '..', '..', '..', 'third_party', 'vr-teleop-kit',
  'src', 'vr_teleop_kit', 'relay', 'web', 'feedback.js'));

function main(argv) {
  const i = argv.indexOf('--session');
  const session = i >= 0 ? Number(argv[i + 1]) : NaN;
  if (!Number.isFinite(session)) {
    process.stderr.write('usage: panel_replay.js --session <xr session id> < stream.jsonl\n');
    return 2;
  }
  const status = FB.createStatus();
  const haptics = FB.createHaptics();
  const out = [];
  const lines = fs.readFileSync(0, 'utf8').split('\n');
  for (let n = 0; n < lines.length; n++) {
    if (lines[n].trim() === '') continue;
    let rec;
    try {
      rec = JSON.parse(lines[n]);
    } catch (e) {
      process.stderr.write(`line ${n + 1}: not JSON: ${e.message}\n`);
      return 1;
    }
    if (rec.msg) {
      const msg = rec.msg;
      if (msg.type === 'teleop_status') status.offer(msg, rec.t_ms, session);
      else if (msg.type === 'teleop_haptic') haptics.offer(msg, rec.t_ms, session, true);
    } else if (typeof rec.query_ms === 'number') {
      const res = FB.panelLines(status.state, rec.query_ms);
      out.push({ t_ms: rec.query_ms, stale: res.stale, lines: res.lines.map((l) => l.text) });
    } else if (typeof rec.tick_ms === 'number') {
      out.push({ t_ms: rec.tick_ms, pulses: haptics.tick(rec.tick_ms) });
    } else {
      process.stderr.write(`line ${n + 1}: unknown record\n`);
      return 1;
    }
  }
  process.stdout.write(out.map((o) => JSON.stringify(o) + '\n').join(''));
  return 0;
}

process.exitCode = main(process.argv.slice(2));
