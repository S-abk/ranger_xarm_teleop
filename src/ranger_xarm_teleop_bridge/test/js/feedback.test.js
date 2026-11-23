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
// Tests of the headset feedback logic (third_party/.../relay/web/feedback.js). Run with
// `node --test`; no dependencies. Works on node 18 and later.
const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');

const FB = require(path.join(__dirname, '..', '..', '..', '..', 'third_party', 'vr-teleop-kit',
  'src', 'vr_teleop_kit', 'relay', 'web', 'feedback.js'));

const SESSION = 1234;
const NOW = 100000;

// An echo whose downlink is `down` ms at time NOW.
const echo = (down, session = SESSION, at = NOW) => ({
  session_id: session, seq: 1, t_client: at - down - 5, held_ms: 5,
});

const haptic = (over = {}, at = NOW) => Object.assign({
  type: 'teleop_haptic', v: 1, run: 1, id: 1, hand: 'right', source: 'episode', event: 'START',
  priority: 1, pulses: [[0.8, 100, 50], [0.4, 100, 0]], age_ms: 10, max_age_ms: 300, missed: 0,
  echo: echo(20, SESSION, at),
}, over);

const status = (over = {}, at = NOW) => Object.assign({
  type: 'teleop_status', v: 1, run: 1, seq: 1, age_ms: 10, stale_ms: 1000, period_ms: 200,
  echo: echo(20, SESSION, at),
  teleop: {
    armed: true, clutch: false, calibrated: true, motion_enabled: true, scale: 0.5, hold: '',
    fault: '', fault_age_s: null, limit: '', gripper: '',
  },
  teleop_age_s: 0.1,
  episode: {
    state: 'RECORDING', index: 7, elapsed_s: 42, kept: 5, discarded: 1, discard_armed: false,
    last_event: '', last_event_text: '', last_event_age_s: null, refusal: '', refusal_age_s: null,
    missed: 0, disk_minutes_left: null,
  },
  episode_age_s: 0.1,
  panel: { anchor: 'left_controller', offset: [0, 0.1, 0], width_m: 0.16, visible_default: true },
}, over);

const texts = (res) => res.lines.map((l) => l.text);

// ---- age rule ----

test('age: echo from another session or none is unknown', () => {
  assert.strictEqual(FB.downlinkMs(echo(20, SESSION + 1), NOW, SESSION), null);
  assert.strictEqual(FB.downlinkMs(null, NOW, SESSION), null);
  assert.strictEqual(FB.totalAgeMs(haptic({ echo: null }), NOW, SESSION), null);
  assert.strictEqual(FB.totalAgeMs(haptic({ age_ms: null }), NOW, SESSION), null);
});

test('age: total is the reported age plus the downlink, clamped at zero', () => {
  assert.strictEqual(FB.downlinkMs(echo(20), NOW, SESSION), 20);
  assert.strictEqual(FB.totalAgeMs(haptic({ age_ms: 15 }), NOW, SESSION), 35);
  const future = { session_id: SESSION, seq: 1, t_client: NOW + 50, held_ms: 0 };
  assert.strictEqual(FB.downlinkMs(future, NOW, SESSION), 0);
});

test('haptic: dropped on echo session mismatch, played when ok, dropped when too old', () => {
  const h = FB.createHaptics();
  assert.strictEqual(h.offer(haptic({ echo: echo(20, SESSION + 1) }), NOW, SESSION, true),
    'age unknown');
  assert.strictEqual(h.offer(haptic({ echo: null, id: 2 }), NOW, SESSION, true), 'age unknown');
  assert.strictEqual(h.offer(haptic({ age_ms: 290, id: 3 }), NOW, SESSION, true), 'too old');
  assert.strictEqual(h.offer(haptic({ id: 4 }), NOW, SESSION, false), 'not in XR');
  assert.deepStrictEqual(h.tick(NOW), []);
  assert.strictEqual(h.offer(haptic({ id: 5 }), NOW, SESSION, true), 'played');
  assert.strictEqual(h.tick(NOW).length, 1);
});

test('haptic: a message at exactly max age is played, just over is dropped', () => {
  const h = FB.createHaptics();
  assert.strictEqual(h.offer(haptic({ age_ms: 280, max_age_ms: 300 }), NOW, SESSION, true),
    'played');
  assert.strictEqual(h.offer(haptic({ id: 2, age_ms: 281, max_age_ms: 300 }), NOW, SESSION, true),
    'too old');
});

test('haptic: (run, id) de-duplication, resend, new run', () => {
  const h = FB.createHaptics();
  assert.strictEqual(h.offer(haptic(), NOW, SESSION, true), 'played');
  assert.strictEqual(h.offer(haptic({}, NOW + 1), NOW + 1, SESSION, true), 'duplicate');
  // Resent after the pattern finished: still not played again.
  assert.strictEqual(h.offer(haptic({}, NOW + 5000), NOW + 5000, SESSION, true), 'duplicate');
  // Same id in a new run is a new message.
  assert.strictEqual(h.offer(haptic({ run: 2 }, NOW + 6000), NOW + 6000, SESSION, true), 'played');
  assert.strictEqual(h.offer(haptic({ run: 2, id: 2 }, NOW + 7000), NOW + 7000, SESSION, true), 'played');
});

test('haptic: seen memory is bounded', () => {
  const h = FB.createHaptics();
  for (let i = 0; i < 2000; i++) h.offer(haptic({ id: i }, NOW + i * 20), NOW + i * 20, SESSION, true);
  // The oldest id was forgotten, the newest is remembered.
  assert.strictEqual(h.offer(haptic({ id: 1999 }, NOW + 40000), NOW + 40000, SESSION, true), 'duplicate');
  assert.strictEqual(h.offer(haptic({ id: 0 }, NOW + 40001), NOW + 40001, SESSION, true), 'played');
});

test('haptic: malformed messages are dropped', () => {
  const h = FB.createHaptics();
  assert.strictEqual(h.offer(haptic({ hand: 'both' }), NOW, SESSION, true), 'bad hand');
  assert.strictEqual(h.offer(haptic({ id: 'x' }), NOW, SESSION, true), 'bad id');
  assert.strictEqual(h.offer(haptic({ id: 2, pulses: [] }), NOW, SESSION, true), 'bad pulses');
  assert.strictEqual(h.offer(haptic({ id: 3, pulses: [[1, -5, 0]] }), NOW, SESSION, true),
    'bad pulses');
  assert.strictEqual(h.offer(haptic({ id: 4, pulses: [[1, 1e6, 0]] }), NOW, SESSION, true),
    'bad pulses');
  assert.strictEqual(h.offer(haptic({ id: 5, pulses: 'x' }), NOW, SESSION, true), 'bad pulses');
});

// ---- player timing ----

test('player: pulses start at the right frames with the remaining on time', () => {
  const h = FB.createHaptics();
  h.offer(haptic({ pulses: [[0.8, 100, 50], [0.4, 100, 0]] }), NOW, SESSION, true);
  assert.deepStrictEqual(h.tick(NOW + 5), [{ hand: 'right', intensity: 0.8, duration: 95 }]);
  assert.deepStrictEqual(h.tick(NOW + 22), [], 'one call per pulse');
  assert.deepStrictEqual(h.tick(NOW + 120), [], 'off time');
  assert.deepStrictEqual(h.tick(NOW + 150), [{ hand: 'right', intensity: 0.4, duration: 100 }]);
  assert.deepStrictEqual(h.tick(NOW + 160), []);
  assert.deepStrictEqual(h.tick(NOW + 400), []);
});

test('player: a pulse whose window passed entirely is skipped', () => {
  const h = FB.createHaptics();
  h.offer(haptic({ pulses: [[0.8, 100, 50], [0.5, 100, 50], [0.3, 100, 0]] }), NOW, SESSION, true);
  // First frame comes late: pulse 0 and 1 on-times are over (pulse 1 ends at +250), pulse 2 is on.
  assert.deepStrictEqual(h.tick(NOW + 320), [{ hand: 'left'.replace('left', 'right'),
    intensity: 0.3, duration: 80 }]);
  // A frame in the off time of the only remaining pulse plays nothing.
  const g = FB.createHaptics();
  g.offer(haptic({ pulses: [[0.8, 100, 50]] }), NOW, SESSION, true);
  assert.deepStrictEqual(g.tick(NOW + 120), []);
});

test('player: intensity 0 is a pause', () => {
  const h = FB.createHaptics();
  h.offer(haptic({ pulses: [[0, 100, 0], [0.7, 100, 0]] }), NOW, SESSION, true);
  assert.deepStrictEqual(h.tick(NOW + 10), []);
  assert.deepStrictEqual(h.tick(NOW + 110), [{ hand: 'right', intensity: 0.7, duration: 90 }]);
});

test('player: intensity is clamped to 0..1', () => {
  const h = FB.createHaptics();
  h.offer(haptic({ pulses: [[7, 100, 0]] }), NOW, SESSION, true);
  assert.strictEqual(h.tick(NOW)[0].intensity, 1);
});

test('player: priority replaces when higher or equal, drops another source when lower', () => {
  const h = FB.createHaptics();
  h.offer(haptic({ id: 1, priority: 5, pulses: [[0.5, 200, 0]] }), NOW, SESSION, true);
  assert.strictEqual(h.offer(haptic({ id: 2, priority: 1, source: 'limit' }, NOW + 10), NOW + 10,
    SESSION, true), 'lower priority');
  assert.strictEqual(h.offer(haptic({ id: 3, priority: 5, pulses: [[0.9, 50, 0]] }, NOW + 20),
    NOW + 20, SESSION, true), 'played');
  assert.deepStrictEqual(h.tick(NOW + 30), [{ hand: 'right', intensity: 0.9, duration: 40 }]);
  assert.strictEqual(h.offer(haptic({ id: 4, priority: 9, pulses: [[0.2, 50, 0]] }, NOW + 40),
    NOW + 40, SESSION, true), 'played');
  // After the pattern ended a lower priority is accepted again.
  assert.strictEqual(h.offer(haptic({ id: 5, priority: 0 }, NOW + 5000), NOW + 5000, SESSION, true), 'played');
});

test('player: a newer request of the same source replaces a playing one of higher priority', () => {
  // An alarm (AUTO_STOP) is playing; the operator's Y (KEEP) right after it is still felt.
  const h = FB.createHaptics();
  h.offer(haptic({ id: 1, priority: 3, event: 'AUTO_STOP', pulses: [[1.0, 2000, 0]] }), NOW,
    SESSION, true);
  h.tick(NOW);
  assert.strictEqual(h.offer(haptic({ id: 2, priority: 2, event: 'KEEP', pulses: [[0.7, 500, 0]] },
    NOW + 300), NOW + 300, SESSION, true), 'played');
  assert.deepStrictEqual(h.tick(NOW + 310), [{ hand: 'right', intensity: 0.7, duration: 490 }]);
});

test('player: hands are independent', () => {
  const h = FB.createHaptics();
  h.offer(haptic({ id: 1, hand: 'left', priority: 9 }), NOW, SESSION, true);
  assert.strictEqual(h.offer(haptic({ id: 2, hand: 'right', priority: 0 }), NOW, SESSION, true),
    'played');
  assert.deepStrictEqual(h.tick(NOW).map((p) => p.hand).sort(), ['left', 'right']);
});

test('player: clear stops a playing pattern', () => {
  const h = FB.createHaptics();
  h.offer(haptic(), NOW, SESSION, true);
  h.clear();
  assert.deepStrictEqual(h.tick(NOW + 10), []);
});

// ---- status acceptance and staleness ----

test('status: accepted, old seq ignored, new run accepted', () => {
  const s = FB.createStatus();
  assert.strictEqual(s.offer(status({ seq: 5 }), NOW, SESSION), 'accepted');
  assert.strictEqual(s.offer(status({ seq: 5 }, NOW + 1), NOW + 1, SESSION), 'old seq');
  assert.strictEqual(s.offer(status({ seq: 4 }, NOW + 2), NOW + 2, SESSION), 'old seq');
  assert.strictEqual(s.offer(status({ seq: 6 }, NOW + 3), NOW + 3, SESSION), 'accepted');
  assert.strictEqual(s.offer(status({ run: 2, seq: 1 }, NOW + 4), NOW + 4, SESSION), 'accepted');
  assert.strictEqual(s.state.run, 2);
  assert.strictEqual(s.state.seq, 1);
});

test('status: unknown or too old age is rejected', () => {
  const s = FB.createStatus();
  assert.strictEqual(s.offer(status({ echo: null }), NOW, SESSION), 'age unknown');
  assert.strictEqual(s.offer(status({ echo: echo(20, SESSION + 1) }), NOW, SESSION),
    'age unknown');
  assert.strictEqual(s.offer(status({ age_ms: 990 }), NOW, SESSION), 'too old');
  assert.strictEqual(s.state.msg, null);
  assert.strictEqual(FB.panelLines(s.state, NOW).stale, true);
});

test('status: records receipt time and age at receipt', () => {
  const s = FB.createStatus();
  s.offer(status({ age_ms: 30, echo: echo(20) }), NOW, SESSION);
  assert.strictEqual(s.state.receivedAt, NOW);
  assert.strictEqual(s.state.ageAtReceipt, 50);
});

test('staleness: never accepted is stale', () => {
  const s = FB.createStatus();
  const res = FB.panelLines(s.state, NOW);
  assert.strictEqual(res.stale, true);
  assert.deepStrictEqual(texts(res), ['NO DATA']);
});

test('staleness: stale after stale_ms; nothing of the old state is shown', () => {
  const s = FB.createStatus();
  const msg = status({
    teleop: {
      armed: true, clutch: true, calibrated: true, motion_enabled: true, scale: 0.5,
      hold: 'holding for the old reason', fault: 'old fault words', fault_age_s: 3,
      limit: 'old limit', gripper: 'old gripper',
    },
  });
  s.offer(msg, NOW, SESSION);   // ageAtReceipt = 30
  assert.strictEqual(FB.panelLines(s.state, NOW + 100).stale, false);
  // stale_ms 1000: stale when now - receivedAt + 30 > 1000, i.e. after 970 ms.
  assert.strictEqual(FB.panelLines(s.state, NOW + 970).stale, false);
  const res = FB.panelLines(s.state, NOW + 971);
  assert.strictEqual(res.stale, true);
  const all = texts(res).join('|');
  assert.ok(all.includes('NO DATA'));
  for (const old of ['REC', 'old', 'kept', 'ARMED', 'scale', 'ep 7']) {
    assert.ok(!all.includes(old), `stale panel must not show ${old}`);
  }
  assert.ok(res.lines[0].big);
  // A fresh status brings it back.
  s.offer(status({ seq: 2 }, NOW + 2000), NOW + 2000, SESSION);
  assert.strictEqual(FB.panelLines(s.state, NOW + 2100).stale, false);
});

test('staleness: a status that was already old when it arrived goes stale sooner', () => {
  const s = FB.createStatus();
  s.offer(status({ age_ms: 600, echo: echo(300) }), NOW, SESSION);   // 900 ms old on arrival
  assert.strictEqual(FB.panelLines(s.state, NOW + 90).stale, false);
  assert.strictEqual(FB.panelLines(s.state, NOW + 101).stale, true);
});

// ---- panel lines ----

const accepted = (over, at = NOW) => {
  const s = FB.createStatus();
  assert.strictEqual(s.offer(status(over), at, SESSION), 'accepted');
  return s.state;
};
const episode = (over) => Object.assign({}, status().episode, over);
const teleop = (over) => Object.assign({}, status().teleop, over);

test('lines: recording', () => {
  const res = FB.panelLines(accepted(), NOW);
  assert.strictEqual(res.stale, false);
  assert.deepStrictEqual(texts(res).slice(0, 2), ['● REC  ep 7  0:42', 'kept 5  discarded 1']);
  assert.strictEqual(res.lines[0].tone, 'rec');
  assert.deepStrictEqual(texts(res).slice(2),
    ['ARMED  clutch off  calibrated', 'scale 0.50x']);
});

test('lines: the recording clock counts up locally between statuses', () => {
  const st = accepted({ episode: episode({ elapsed_s: 42.5 }) });
  assert.ok(texts(FB.panelLines(st, NOW + 20))[0].endsWith('0:42'));
  assert.ok(texts(FB.panelLines(st, NOW + 600))[0].endsWith('0:43'));
  const long = accepted({ episode: episode({ elapsed_s: 3725 }) });
  assert.ok(texts(FB.panelLines(long, NOW))[0].endsWith('62:05'));
});

test('lines: idle names the next episode', () => {
  const res = FB.panelLines(accepted({ episode: episode({ state: 'IDLE', index: 7 }) }), NOW);
  assert.strictEqual(texts(res)[0], 'IDLE  X starts ep 8');
});

test('lines: review, with and without discard armed', () => {
  const rev = FB.panelLines(accepted({ episode: episode({ state: 'REVIEW' }) }), NOW);
  assert.deepStrictEqual(texts(rev).slice(0, 2), ['REVIEW ep 7', 'Y keep, hold Y discard']);
  assert.deepStrictEqual(rev.lines.slice(0, 2).map((l) => l.size), ['key', 'small']);
  assert.ok(!texts(rev).some((t) => t.includes('DISCARDING')));
  const arm = FB.panelLines(
    accepted({ episode: episode({ state: 'REVIEW', discard_armed: true }) }), NOW);
  assert.strictEqual(texts(arm)[0], 'REVIEW ep 7: DISCARDING\u2026');
  assert.strictEqual(arm.lines[0].tone, 'bad');
});

test('lines: not ready', () => {
  const res = FB.panelLines(accepted({ episode: episode({ state: 'NOT_READY' }) }), NOW);
  assert.strictEqual(texts(res)[0], 'NOT READY');
});

test('lines: episode section null says no data and keeps the teleop section', () => {
  const res = FB.panelLines(accepted({ episode: null }), NOW);
  assert.strictEqual(texts(res)[0], 'Recording: no data');
  assert.ok(!texts(res).some((t) => t.startsWith('kept')));
  assert.ok(texts(res).includes('scale 0.50x'));
});

test('lines: teleop section null says no data and keeps the episode section', () => {
  const res = FB.panelLines(accepted({ teleop: null }), NOW);
  assert.ok(texts(res).includes('Teleop: no data'));
  assert.ok(!texts(res).some((t) => t.startsWith('scale')));
  assert.ok(texts(res)[0].includes('REC'));
});

test('lines: teleop flags, limit, gripper', () => {
  const res = FB.panelLines(accepted({
    teleop: teleop({ armed: false, clutch: true, calibrated: false, scale: 1, limit: 'wrist',
      gripper: 'open' }),
  }), NOW);
  assert.ok(texts(res).includes('disarmed  CLUTCH  NOT CALIBRATED'));
  assert.ok(texts(res).includes('Limit: wrist'));
  assert.ok(texts(res).includes('scale 1.00x'));
  assert.ok(texts(res).includes('gripper: open'));
});

test('lines: the key values are the large ones, the rest small', () => {
  const sizeOf = (res, start) => res.lines.find((l) => l.text.startsWith(start)).size;
  const res = FB.panelLines(accepted({
    teleop: teleop({ hold: 'Arm position late: wait', limit: 'keep-out zone', gripper: 'open' }),
    episode: episode({ refusal: 'arm first (B)', refusal_age_s: 2, missed: 1 }),
  }), NOW);
  assert.strictEqual(sizeOf(res, '● REC'), 'key');
  assert.strictEqual(sizeOf(res, 'Arm position late'), 'key');
  assert.strictEqual(sizeOf(res, 'Limit: keep-out zone'), 'key');
  for (const start of ['kept', 'ARMED', 'scale', 'gripper', 'Refused', 'missed']) {
    assert.strictEqual(sizeOf(res, start), 'small', start);
  }
  const fault = FB.panelLines(accepted({ teleop: teleop({ fault: 'Arm position late: wait', fault_age_s: 1 }) }), NOW);
  assert.strictEqual(sizeOf(fault, 'Arm position late'), 'key');
  for (const ep of [episode({ state: 'IDLE' }), episode({ state: 'REVIEW' }), null]) {
    assert.strictEqual(FB.panelLines(accepted({ episode: ep }), NOW).lines[0].size, 'key');
  }
  assert.strictEqual(sizeOf(FB.panelLines(accepted({ teleop: null }), NOW), 'Teleop: no data'), 'key');
});

test('lines: no key line is wider than a key line holds (no squeezed key text)', () => {
  const eps = [episode({}), episode({ state: 'IDLE', index: 99 }), episode({ state: 'REVIEW', index: 999 }),
    episode({ state: 'REVIEW', index: 999, discard_armed: true }), episode({ state: 'NOT_READY' }), null];
  // every limit word of operator_status.py LIMIT_TEXT
  const limits = ['reach range', 'rotation range', 'keep-out zone', 'near robot body', 'arm controller slowing',
    'arm lagging hand', 'speed', 'joint speed', 'joint at its end'];
  for (const ep of eps) {
    for (const limit of limits) {
      const res = FB.panelLines(accepted({ episode: ep, teleop: teleop({ limit }) }), NOW);
      for (const l of res.lines.filter((x) => x.size === 'key')) {
        assert.ok(l.text.length <= 26, `key line too wide: ${l.text}`);
      }
    }
  }
});

test('lines: a slow simulator is said in one small line after the hold, only while it is sent', () => {
  const slow = 'Simulator slow: 0.4x real time';
  const res = FB.panelLines(accepted({ sim_text: slow, teleop: teleop({ hold: 'Arm position late: wait' }) }), NOW);
  const i = texts(res).indexOf(slow);
  assert.ok(i > 0, texts(res));
  assert.strictEqual(texts(res)[i - 1], 'Arm position late: wait');
  assert.deepStrictEqual([res.lines[i].size, res.lines[i].tone], ['small', 'warn']);
  assert.ok(texts(FB.panelLines(accepted({ sim_text: slow, teleop: null }), NOW)).includes(slow));
  for (const none of [{}, { sim_text: '' }, { sim_text: 7 }]) {
    assert.ok(!texts(FB.panelLines(accepted(none), NOW)).some((t) => t.startsWith('Simulator')));
  }
  // Stale: NO DATA only, never the old line.
  const st = accepted({ sim_text: slow });
  assert.deepStrictEqual(texts(FB.panelLines(st, NOW + 5000)), ['NO DATA']);
});

test('lines: the longest hold text, with its age, is never cut', () => {
  // HOLD_SERVO_HALTED, the longest text of the hold table (operator_status.py), 81 characters.
  const text = 'Arm halted near a collision or joint limit: disarm (B), then use the retreat tool';
  for (const over of [{ hold: text }, { fault: text, fault_age_s: 9.5 }, { fault: text, fault_age_s: 599 }]) {
    const lines = FB.panelLines(accepted({ teleop: teleop(over) }), NOW).lines.filter((l) => l.size === 'key');
    const shown = lines.slice(1).map((l) => l.text).join(' ');   // lines[0]: the recording state
    assert.ok(shown.startsWith(text), shown);
    assert.ok(!shown.includes('\u2026'), shown);
  }
});

test('lines: the picture key carries the size of each line (a size change redraws)', () => {
  const key = FB.panelLines(accepted(), NOW).key.split('\n');
  assert.ok(key[0].startsWith('key|rec|● REC'), key[0]);
  assert.ok(key[1].startsWith('small|text|kept'), key[1]);
});

test('layout: key lines larger and bold, everything above the footer, even at the worst case', () => {
  const typical = FB.panelLayout(FB.panelLines(accepted({ teleop: teleop({ hold: 'Arm position late: wait' }) }), NOW));
  const key = typical.filter((l) => l.bold), small = typical.filter((l) => !l.bold);
  assert.ok(key.length >= 2 && small.length >= 2);
  assert.ok(key.every((l) => l.px === FB.KEY_PX) && small.every((l) => l.px === FB.SMALL_PX));
  for (let i = 1; i < typical.length; i++) assert.ok(typical[i].y > typical[i - 1].y);
  const long = 'word '.repeat(40);
  const worst = FB.panelLayout(FB.panelLines(accepted({
    teleop: teleop({ fault: long, fault_age_s: 1, limit: 'keep-out zone', gripper: 'closing 40 %' }),
    episode: episode({ refusal: long, refusal_age_s: 1, missed: 2 }),
    sim_text: 'Simulator slow: 0.4x real time',
  }), NOW));
  assert.strictEqual(worst.length, 14);
  assert.ok(Math.min(...worst.map((l) => l.px)) >= 16, 'worst case still readable');
  const last = worst[worst.length - 1];
  const lastStep = last.px * (last.bold ? 1.2 : 1.25);
  assert.ok(last.y + lastStep <= FB.PANEL_H - FB.FOOTER_PX + 1, `${last.y} + ${lastStep}`);
  const kw = worst.filter((l) => l.bold), sw = worst.filter((l) => !l.bold);
  assert.ok(Math.min(...kw.map((l) => l.px)) > Math.max(...sw.map((l) => l.px)));
  const stale = FB.panelLayout({ stale: true, lines: [{ text: 'NO DATA', tone: 'bad', big: true }] });
  assert.deepStrictEqual([stale.length, stale[0].px, stale[0].align], [1, 72, 'center']);
});

test('lines: hold text, and a fault whose age grows locally', () => {
  const hold = FB.panelLines(accepted({ teleop: teleop({ hold: 'waiting for the arm' }) }), NOW);
  assert.ok(texts(hold).includes('waiting for the arm'));
  const st = accepted({ teleop: teleop({ fault: 'joint 3 stalled', fault_age_s: 3.5 }) });
  assert.ok(texts(FB.panelLines(st, NOW)).includes('joint 3 stalled (3 s ago)'));
  assert.ok(texts(FB.panelLines(st, NOW + 400)).includes('joint 3 stalled (3 s ago)'));
  assert.ok(texts(FB.panelLines(st, NOW + 500)).includes('joint 3 stalled (4 s ago)'));
  const nul = accepted({ teleop: teleop({ fault: 'bad thing', fault_age_s: null }) });
  assert.ok(texts(FB.panelLines(nul, NOW)).includes('bad thing'));
});

test('lines: refusal for 30 s, with a growing age', () => {
  const st = accepted({ episode: episode({ refusal: 'not calibrated', refusal_age_s: 4.5 }) });
  assert.ok(texts(FB.panelLines(st, NOW)).includes('Refused 4 s ago: not calibrated'));
  assert.ok(texts(FB.panelLines(st, NOW + 400)).includes('Refused 4 s ago: not calibrated'));
  assert.ok(texts(FB.panelLines(st, NOW + 500)).includes('Refused 5 s ago: not calibrated'));
  const old = accepted({ episode: episode({ refusal: 'not calibrated', refusal_age_s: 30 }) });
  assert.ok(!texts(FB.panelLines(old, NOW)).some((t) => t.includes('Refused')));
  const unknown = accepted({ episode: episode({ refusal: 'x', refusal_age_s: null }) });
  assert.ok(!texts(FB.panelLines(unknown, NOW)).some((t) => t.includes('Refused')));
});

test('lines: an automatic stop is shown with its text, whole', () => {
  const st = accepted({ episode: episode({
    state: 'REVIEW', last_event: 'AUTO_STOP',
    last_event_text: 'Episode 12 stopped by itself: teleop state lost', last_event_age_s: 2,
  }) });
  const shown = texts(FB.panelLines(st, NOW));
  const i = shown.findIndex((t) => t.startsWith('Episode 12 stopped by itself:'));
  assert.ok(i >= 0, shown);
  assert.strictEqual(shown[i] + ' ' + shown[i + 1], 'Episode 12 stopped by itself: teleop state lost');
  const keep = accepted({ episode: episode({
    last_event: 'KEEP', last_event_text: 'kept', last_event_age_s: 2 }) });
  assert.ok(!texts(FB.panelLines(keep, NOW)).some((t) => t.includes('Stopped')));
  const late = accepted({ episode: episode({
    last_event: 'AUTO_STOP', last_event_text: 'x', last_event_age_s: 60 }) });
  assert.ok(!texts(FB.panelLines(late, NOW)).some((t) => t.includes('Stopped')));
});

test('lines: missed events', () => {
  const none = FB.panelLines(accepted(), NOW);
  assert.ok(!texts(none).some((t) => t.includes('missed')));
  const res = FB.panelLines(accepted({ episode: episode({ missed: 3 }) }), NOW);
  assert.ok(texts(res).includes('missed 3 events'));
});

test('lines: unknown keys are ignored', () => {
  const msg = status({ future_field: { a: 1 } });
  msg.episode.new_thing = 'x';
  const s = FB.createStatus();
  assert.strictEqual(s.offer(msg, NOW, SESSION), 'accepted');
  assert.strictEqual(FB.panelLines(s.state, NOW).stale, false);
});

test('lines: the worst case is bounded (at most 14 lines of at most 40 characters)', () => {
  const long = 'word '.repeat(40);
  const st = accepted({
    episode: episode({ state: 'REVIEW', discard_armed: true, refusal: long, refusal_age_s: 1,
      missed: 9 }),
    teleop: teleop({ fault: long, fault_age_s: 1, limit: 'wrist', gripper: 'closed' }),
    sim_text: 'Simulator slow: 0.4x real time',
  });
  const res = FB.panelLines(st, NOW);
  assert.ok(res.lines.length <= 14, `got ${res.lines.length} lines`);
  for (const l of res.lines.filter((x) => x.size === 'key')) {
    assert.ok(l.text.length <= 26, `key line too long: ${l.text}`);
  }
  for (const l of res.lines) {
    assert.ok(l.text.length <= 40, `line too long: ${l.text}`);
  }
  assert.ok(texts(res).some((t) => t.endsWith('…')));
});

test('lines: the key is stable while nothing changes, and changes when something does', () => {
  const st = accepted({ teleop: teleop({ fault: 'oops', fault_age_s: 3 }) });
  const a = FB.panelLines(st, NOW + 10).key;
  assert.strictEqual(FB.panelLines(st, NOW + 20).key, a);
  assert.strictEqual(FB.panelLines(st, NOW + 300).key, a);
  assert.notStrictEqual(FB.panelLines(st, NOW + 1500).key, a, 'ages advance');
  const s = FB.createStatus();
  s.offer(status({ seq: 1 }), NOW, SESSION);
  const k1 = FB.panelLines(s.state, NOW + 10).key;
  s.offer(status({ seq: 2 }, NOW + 200), NOW + 200, SESSION);   // same content, next message
  assert.strictEqual(FB.panelLines(s.state, NOW + 210).key, k1);
  s.offer(status({ seq: 3, teleop: teleop({ armed: false }) }, NOW + 400), NOW + 400, SESSION);
  assert.notStrictEqual(FB.panelLines(s.state, NOW + 410).key, k1);
});

// ---- word wrap ----

test('wrap: fits, wraps, ellipsis, long words', () => {
  assert.deepStrictEqual(FB.wrapText('short', 10, 2), ['short']);
  assert.deepStrictEqual(FB.wrapText('aaa bbb ccc ddd', 7, 2), ['aaa bbb', 'ccc ddd']);
  assert.deepStrictEqual(FB.wrapText('aaa bbb ccc ddd eee', 7, 2), ['aaa bbb', 'ccc dd…']);
  const w = FB.wrapText('x'.repeat(100), 10, 2);
  assert.strictEqual(w.length, 2);
  assert.ok(w[1].endsWith('…'));
  assert.ok(w.every((l) => l.length <= 10));
  assert.deepStrictEqual(FB.wrapText('   ', 10, 2), []);
});

// ---- toggle ----

test('toggle: one edge per press; a button already held at the start is not a press', () => {
  const e = FB.createEdge();
  assert.strictEqual(e.update(true), false, 'first reading only seeds');
  assert.strictEqual(e.update(true), false);
  assert.strictEqual(e.update(false), false);
  assert.strictEqual(e.update(true), true);
  assert.strictEqual(e.update(true), false);
  assert.strictEqual(e.update(false), false);
  assert.strictEqual(e.update(true), true);
});

test('toggle: a controller that disappears and returns re-seeds', () => {
  const e = FB.createEdge();
  e.update(false);
  assert.strictEqual(e.update(null), false);
  assert.strictEqual(e.update(true), false);
  assert.strictEqual(e.update(false), false);
  assert.strictEqual(e.update(true), true);
});

test('visibility: stored choice wins over the default', () => {
  assert.strictEqual(FB.resolveVisible(null, true), true);
  assert.strictEqual(FB.resolveVisible(null, false), false);
  assert.strictEqual(FB.resolveVisible('0', true), false);
  assert.strictEqual(FB.resolveVisible('1', false), true);
});

// ---- placement ----

const IDENT = [0, 0, 0, 1];
const col = (m, c) => m.slice(c * 4, c * 4 + 3);
const near = (a, b, eps = 1e-9) => a.every((v, i) => Math.abs(v - b[i]) < eps);

test('placement: controller anchor puts the panel above the grip, facing the head', () => {
  const cfg = FB.panelConfig({ anchor: 'left_controller', offset: [0, 0.1, 0], width_m: 0.16 });
  const head = { position: [0, 1.6, 0], orientation: IDENT };
  const grip = [-0.2, 1.1, -0.4];
  const m = FB.panelModel(cfg, head, grip);
  assert.ok(near(m.slice(12, 15), [-0.2, 1.2, -0.4]));
  assert.ok(Math.abs(Math.hypot(...col(m, 0)) - 0.16) < 1e-9, 'width');
  assert.ok(Math.abs(Math.hypot(...col(m, 1)) - 0.16 * 320 / 512) < 1e-9, 'height');
  // z points from the panel toward the head; x is to the head's right (+x here).
  const z = col(m, 2);
  const toHead = [0.2, 0.4, 0.4].map((v) => v / Math.hypot(0.2, 0.4, 0.4));
  assert.ok(near(z, toHead, 1e-6));
  assert.ok(col(m, 0)[0] > 0);
  assert.ok(Math.abs(col(m, 0)[1]) < 1e-9, 'world up kept: x stays horizontal');
});

test('placement: the facing-frame offset moves toward the viewer and to their right', () => {
  const cfg = FB.panelConfig({ anchor: 'left_controller', offset: [0.1, 0, 0.2] });
  const head = { position: [0, 1.5, 2], orientation: IDENT };   // head on +z looking at -z
  const m = FB.panelModel(cfg, head, [0, 1, 0]);
  assert.ok(near(m.slice(12, 15), [0.1, 1, 0.2], 1e-9));
});

test('placement: no controller falls back to the head-locked default', () => {
  const cfg = FB.panelConfig({ anchor: 'left_controller', offset: [0, 0.1, 0] });
  const head = { position: [1, 1.6, 2], orientation: IDENT };
  const m = FB.panelModel(cfg, head, null);
  assert.ok(near(m.slice(12, 15), [1 - 0.22, 1.6 - 0.15, 2 - 0.6]));
});

test('placement: head anchor follows the head orientation', () => {
  const cfg = FB.panelConfig({ anchor: 'head', offset: [-0.22, -0.15, -0.6] });
  const s = Math.SQRT1_2;
  const head = { position: [0, 1.6, 0], orientation: [0, s, 0, s] };   // 90 degrees about y
  const m = FB.panelModel(cfg, head, [0, 1, 0]);
  // -z of the head (forward) now points at -x: the offset (-0.22, -0.15, -0.6) -> (-0.6, ., 0.22).
  assert.ok(near(m.slice(12, 15), [-0.6, 1.45, 0.22], 1e-9));
});

test('config: defaults, limits and bad values', () => {
  const d = FB.panelConfig(null);
  assert.strictEqual(d.anchor, 'left_controller');
  assert.deepStrictEqual(d.offset, [0, 0.1, 0]);
  assert.strictEqual(d.visibleDefault, true);
  assert.deepStrictEqual(FB.panelConfig({ anchor: 'head' }).offset, [-0.22, -0.15, -0.6]);
  assert.deepStrictEqual(FB.panelConfig({ offset: [1, 'x', 3] }).offset, [0, 0.1, 0]);
  assert.strictEqual(FB.panelConfig({ width_m: 50 }).width, 0.6);
  assert.strictEqual(FB.panelConfig({ visible_default: false }).visibleDefault, false);
});

test('version is exported', () => {
  assert.strictEqual(FB.FEEDBACK_VERSION, 'feedback-7');
});

test('lines: a section older than the staleness limit shows no data, never its old state', () => {
  // The feedback node built the status from a teleop state 0.9 s old; 0.2 s later it is 1.13 s.
  const st = accepted({ teleop_age_s: 0.9, episode_age_s: 0.1, episode: episode({}) });
  assert.ok(texts(FB.panelLines(st, NOW)).includes('ARMED  clutch off  calibrated'));
  const later = texts(FB.panelLines(st, NOW + 200));
  assert.ok(later.includes('Teleop: no data'), later);
  assert.ok(!later.some((t) => t.startsWith('ARMED')), later);
  assert.ok(!later.includes('Recording: no data'), later);
  // The same for the episode section, and an unknown age is never shown as fresh.
  const ep = accepted({ teleop_age_s: 0.1, episode_age_s: 0.95, episode: episode({ index: 7 }) });
  const epLater = texts(FB.panelLines(ep, NOW + 100));
  assert.ok(epLater.includes('Recording: no data'), epLater);
  assert.ok(!epLater.some((t) => t.includes('ep 7')), epLater);
  const unknown = accepted({ teleop_age_s: null, episode_age_s: null, episode: episode({}) });
  const u = texts(FB.panelLines(unknown, NOW));
  assert.ok(u.includes('Teleop: no data') && u.includes('Recording: no data'), u);
});

// ---- the base line (driving the base with the left grip): only with a base section ----

const baseSection = (over) => Object.assign({
  valid: true, state: 'DRIVING', reason: 'NONE',
  text: 'Base: driving, release left grip to stop (stick up = robot front)', event_age_s: 1.0,
}, over);
const baseLines = (section, over = {}) =>
  FB.panelLines(accepted(Object.assign({ base: section }, over)), NOW).lines;

test('base: without a base section nothing is drawn for it', () => {
  const plain = FB.panelLines(accepted({}), NOW).lines;
  assert.ok(!plain.some((l) => l.text.startsWith('Base')), plain);
  assert.deepStrictEqual(FB.panelLines(accepted({ base: undefined }), NOW).lines, plain);
  assert.deepStrictEqual(FB.panelLines(accepted({ base: null }), NOW).lines, plain);
});

test('base: a valid state is lines with its text, after the others', () => {
  const plain = FB.panelLines(accepted({}), NOW).lines;
  const got = baseLines(baseSection());
  assert.deepStrictEqual(got.slice(0, plain.length), plain);
  assert.strictEqual(got.slice(plain.length).map((l) => l.text).join(' '),
    'Base: driving, release left grip to stop (stick up = robot front)');
  assert.strictEqual(got[got.length - 1].tone, 'text');
});

test('base: the line follows the reset line when both are present', () => {
  const reset = { valid: true, state: 'MOVING', reason: 'NONE', text: 'Reset: moving to ready; release to stop', event_age_s: 1 };
  const got = texts({ lines: baseLines(baseSection(), { reset }) });
  assert.ok(got.indexOf('Reset: moving to ready; release to stop') < got.findIndex((t) => t.startsWith('Base')), got);
});

test('base: tone is bad for a refusal and a stop by a fault, not for a release', () => {
  const tone = (s) => baseLines(baseSection(s)).pop().tone;
  assert.strictEqual(tone({ state: 'REFUSED', reason: 'ODOM_MISSING', text: 'Base refused: base odometry missing (2 s ago)' }), 'bad');
  assert.strictEqual(tone({ state: 'STOPPED', reason: 'INPUT_STALE', text: 'Base stopped: headset input late or lost (2 s ago)' }), 'bad');
  assert.strictEqual(tone({ state: 'STOPPED', reason: 'RELEASED', text: 'Base stopped: released (2 s ago)' }), 'text');
  assert.strictEqual(tone({ state: 'STOPPING', reason: 'RELEASED', text: 'Base: stopping…' }), 'text');
});

test('base: a section that is not valid says no data, never its old text', () => {
  const got = baseLines({ valid: false, state: null, reason: null, text: '', event_age_s: null });
  const last = got[got.length - 1];
  assert.strictEqual(last.text, 'Base: no data');
  assert.strictEqual(last.tone, 'warn');
  assert.strictEqual(got.filter((l) => l.text.startsWith('Base')).length, 1);
});

test('base: an empty text draws nothing', () => {
  const plain = FB.panelLines(accepted({}), NOW).lines;
  assert.deepStrictEqual(baseLines(baseSection({ state: 'IDLE', text: '' })), plain);
  assert.deepStrictEqual(baseLines(baseSection({ state: 'REFUSED', reason: 'OFF', text: '' })), plain);
});

test('base: the longest texts are wrapped whole, never cut', () => {
  const longest = [
    'Base: driving, release left grip to stop (stick up = robot front) (slowed: tool speed)',
    'Base: driving; turn ignored, centre the left stick to turn (slowed: tool speed)',
    'Base refused: another node publishes the base command: /a_long_publisher_name_x… (29 s ago)',
    'Base refused: release the right grip (arm clutch) first (29 s ago)',
  ];
  for (const text of longest) {
    const got = baseLines(baseSection({ state: 'REFUSED', reason: 'OTHER_PUBLISHER', text }))
      .filter((l) => l.tone === 'bad').map((l) => l.text);
    assert.strictEqual(got.join(' '), text);
  }
});

test('base: a stale status shows NO DATA only, whatever it carried', () => {
  const res = FB.panelLines(accepted({ base: baseSection() }), NOW + 5000);
  assert.deepStrictEqual(texts(res), ['NO DATA']);
});

test('base: the picture changes when the base line does', () => {
  const key = (s) => FB.panelLines(accepted({ base: s }), NOW).key;
  assert.notStrictEqual(key(baseSection()), key(baseSection({ text: 'Base: stopping…' })));
  assert.notStrictEqual(FB.panelLines(accepted({}), NOW).key, key(baseSection()));
});

test('base: a base vibration on the right hand does not disturb a left one of the same source', () => {
  const h = FB.createHaptics();
  const base = (over) => haptic(Object.assign({ source: 'base', event: 'REFUSED', hand: 'left', priority: 3 }, over));
  assert.strictEqual(h.offer(base({ id: 1 }), NOW, SESSION, true), 'played');
  assert.strictEqual(h.offer(base({ id: 2, hand: 'right', event: 'AT_REST', priority: 2,
    pulses: [[0.5, 60, 0]] }), NOW, SESSION, true), 'played');
  assert.deepStrictEqual(h.tick(NOW).map((p) => p.hand).sort(), ['left', 'right']);
});
