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
'use strict';
// Tests of the reset line of the headset panel (feedback.js). Run with `node --test`.
const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');

const FB = require(path.join(__dirname, '..', '..', '..', '..', 'third_party', 'vr-teleop-kit',
  'src', 'vr_teleop_kit', 'relay', 'web', 'feedback.js'));

const SESSION = 1234;
const NOW = 100000;
const echo = { session_id: SESSION, seq: 1, t_client: NOW - 25, held_ms: 5 };

const base = () => ({
  type: 'teleop_status', v: 1, run: 1, seq: 1, age_ms: 10, stale_ms: 1000, period_ms: 200, echo,
  teleop: {
    armed: false, clutch: false, calibrated: true, motion_enabled: true, scale: 0.5, hold: '',
    fault: '', fault_age_s: null, limit: '', gripper: '',
  },
  teleop_age_s: 0.1,
  episode: {
    state: 'IDLE', index: 7, elapsed_s: 0, kept: 5, discarded: 1, discard_armed: false,
    last_event: '', last_event_text: '', last_event_age_s: null, refusal: '', refusal_age_s: null,
    missed: 0, disk_minutes_left: null,
  },
  episode_age_s: 0.1,
  sim_text: '',
  panel: { anchor: 'left_controller', offset: [0, 0.1, 0], width_m: 0.16, visible_default: true },
});

const lines = (msg, at = NOW) => {
  const s = FB.createStatus();
  assert.strictEqual(s.offer(msg, NOW, SESSION), 'accepted');
  return FB.panelLines(s.state, at).lines;
};
const reset = (over) => Object.assign({
  valid: true, state: 'MOVING', reason: 'NONE', text: 'Reset: moving to ready; release to stop',
  event_age_s: 1.0,
}, over);
const withReset = (r) => Object.assign(base(), { reset: r });

test('reset: without a reset section nothing is drawn for it', () => {
  const plain = lines(base());
  assert.ok(!plain.some((l) => l.text.startsWith('Reset')), plain);
  // The same lines as a status whose reset section is absent in any other way.
  assert.deepStrictEqual(lines(Object.assign(base(), { reset: undefined })), plain);
  assert.deepStrictEqual(lines(Object.assign(base(), { reset: null })), plain);
});

test('reset: a valid state is one line with its text, after the other lines', () => {
  const plain = lines(base());
  const got = lines(withReset(reset()));
  assert.deepStrictEqual(got.slice(0, plain.length), plain);
  assert.deepStrictEqual(got.slice(plain.length).map((l) => l.text),
    ['Reset: moving to ready; release to stop']);
  assert.strictEqual(got[got.length - 1].tone, 'text');
});

test('reset: tone is bad for a refusal and a stop by a fault, not for a release; ok on arrival', () => {
  const tone = (r) => lines(withReset(reset(r))).pop().tone;
  assert.strictEqual(tone({ state: 'REFUSED', reason: 'ARMED', text: 'Reset refused: disarm first (B) (2 s ago)' }), 'bad');
  assert.strictEqual(tone({ state: 'STOPPED', reason: 'INPUT_STALE', text: 'Reset stopped: headset input late or lost (2 s ago)' }), 'bad');
  assert.strictEqual(tone({ state: 'STOPPED', reason: 'RELEASED', text: 'Reset stopped: released. Pull back again to go on (2 s ago)' }), 'text');
  assert.strictEqual(tone({ state: 'ARRIVED', reason: 'REACHED', text: 'Reset: at ready. Arm with B (2 s ago)' }), 'ok');
  assert.strictEqual(tone({ state: 'WAITING', reason: 'NONE', text: 'Reset: keep holding…' }), 'text');
});

test('reset: a section that is not valid says no data, never its old text', () => {
  const got = lines(withReset({ valid: false, state: null, reason: null, text: '', event_age_s: null }));
  const last = got[got.length - 1];
  assert.strictEqual(last.text, 'Reset: no data');
  assert.strictEqual(last.tone, 'warn');
  assert.strictEqual(got.filter((l) => l.text.startsWith('Reset')).length, 1);
});

test('reset: an empty text draws nothing', () => {
  const plain = lines(base());
  assert.deepStrictEqual(lines(withReset(reset({ state: 'IDLE', text: '' }))), plain);
  assert.deepStrictEqual(lines(withReset(reset({ state: 'REFUSED', reason: 'ARMED', text: '' }))), plain);
});

test('reset: the longest text is wrapped whole, never cut', () => {
  const text = 'Reset refused: arm in collision; nothing moved. Disarm, use the retreat tool (12 s ago)';
  const got = lines(withReset(reset({ state: 'REFUSED', reason: 'START_IN_COLLISION', text })))
    .filter((l) => l.tone === 'bad').map((l) => l.text);
  assert.strictEqual(got.join(' '), text);
});

test('reset: a stale status shows NO DATA only, whatever it carried', () => {
  const s = FB.createStatus();
  s.offer(withReset(reset()), NOW, SESSION);
  const res = FB.panelLines(s.state, NOW + 5000);
  assert.deepStrictEqual(res.lines.map((l) => l.text), ['NO DATA']);
});

test('reset: the picture changes when the reset line does', () => {
  const key = (r) => {
    const s = FB.createStatus();
    s.offer(withReset(r), NOW, SESSION);
    return FB.panelLines(s.state, NOW).key;
  };
  assert.notStrictEqual(key(reset()), key(reset({ text: 'Reset: stopping…' })));
  const s = FB.createStatus();
  s.offer(base(), NOW, SESSION);
  assert.notStrictEqual(FB.panelLines(s.state, NOW).key, key(reset()));
});
