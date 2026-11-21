// Operator feedback on the headset: pure logic, no DOM and no GL (added by ranger_xarm_teleop,
// not part of upstream vr-teleop-kit; see third_party/vr-teleop-kit/VENDORED.md).
//
// Receives two relay message types from the ROS side:
//   teleop_haptic  one vibration pattern for one hand
//   teleop_status  the state to show on a small panel
// and decides what the page does with them. client.js only calls these functions and draws.
//
// No clock sync. Each message carries `echo`, the t_client (this page's performance.now()) of the
// newest xr_frame the sender had received and how long the sender held it (held_ms). The time
// since then on this page, plus the age the sender reports, is an upper bound on how old the
// message is. An echo from another XR session, or none, means the age is unknown: too old.
//
// Every function takes the current time (ms, performance.now()) as an argument.
(function (root) {
  'use strict';

  const FEEDBACK_VERSION = 'feedback-6';
  const SEEN_MAX = 512;               // (run, id) pairs remembered for de-duplication
  const MAX_PULSES = 32;
  const MAX_PATTERN_MS = 10000;
  // Two text sizes: the key values (recording state, current hold or fault, active limit) larger
  // and bold, the rest smaller. Characters per line at each size on the 480 px text width.
  const KEY_PX = 34, SMALL_PX = 22, SMALL_MIN_PX = 16;
  const KEY_WRAP = 26, SMALL_WRAP = 40;
  const HOLD_LINES = 4;                       // the longest hold text with its age, never cut
  const KEY_LINE = 1.2, SMALL_LINE = 1.25;   // line height, in font sizes
  const TOP_PX = 10, FOOTER_PX = 26;          // above the first line; below the last (version)
  const TEXT_W = 480;                         // text squeezed to this width if wider
  const RECENT_S = 30;                // how long a refusal / automatic stop stays on the panel
  const DEFAULT_STALE_MS = 1000;
  const PANEL_W = 512, PANEL_H = 320; // panel canvas size in pixels
  const HEAD_DEFAULT_OFFSET = [-0.22, -0.15, -0.6];
  const CONTROLLER_DEFAULT_OFFSET = [0, 0.10, 0];
  const TOGGLE_BUTTON = 3;            // thumbstick click on the Quest controller gamepad

  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

  // ---- Age ----

  // Time since the echoed xr_frame was sent back by the other side, on this page's clock.
  // null (unknown) unless the echo is complete and belongs to the current XR session.
  function downlinkMs(echo, nowMs, sessionId) {
    if (!echo || !isNum(echo.t_client) || !isNum(echo.held_ms)) return null;
    if (echo.session_id !== sessionId) return null;
    return Math.max(0, nowMs - (echo.t_client + echo.held_ms));
  }

  // Age of a message when it arrives: the age it reports plus the downlink. null when unknown.
  function totalAgeMs(msg, nowMs, sessionId) {
    if (!msg || !isNum(msg.age_ms)) return null;
    const d = downlinkMs(msg.echo, nowMs, sessionId);
    return d === null ? null : msg.age_ms + d;
  }

  // ---- Vibration ----

  // [[intensity 0..1, on_ms, off_ms], ...] -> [{begin, endOn, end, intensity}] from startMs, or null.
  function layoutPulses(pulses, startMs) {
    if (!Array.isArray(pulses) || pulses.length === 0 || pulses.length > MAX_PULSES) return null;
    const out = [];
    let t = startMs;
    for (const p of pulses) {
      if (!Array.isArray(p) || !isNum(p[0]) || !isNum(p[1]) || !isNum(p[2])) return null;
      if (p[1] < 0 || p[2] < 0) return null;
      out.push({ begin: t, endOn: t + p[1], end: t + p[1] + p[2], intensity: Math.min(1, Math.max(0, p[0])) });
      t += p[1] + p[2];
    }
    return t - startMs > MAX_PATTERN_MS ? null : out;
  }

  // One pattern per hand at a time; driven by tick() from the XR frame loop.
  function createHaptics() {
    const seen = new Set();
    const order = [];
    const players = { left: null, right: null };

    // Returns 'played' or the reason the message was dropped.
    function offer(msg, nowMs, sessionId, inXr) {
      if (!inXr) return 'not in XR';
      if (!msg || (msg.hand !== 'left' && msg.hand !== 'right')) return 'bad hand';
      if (!isNum(msg.run) || !isNum(msg.id)) return 'bad id';
      const age = totalAgeMs(msg, nowMs, sessionId);
      if (age === null) return 'age unknown';
      if (!isNum(msg.max_age_ms) || age > msg.max_age_ms) return 'too old';
      const key = msg.run + ':' + msg.id;
      if (seen.has(key)) return 'duplicate';
      seen.add(key);
      order.push(key);
      if (order.length > SEEN_MAX) seen.delete(order.shift());
      const pulses = layoutPulses(msg.pulses, nowMs);
      if (!pulses) return 'bad pulses';
      const priority = isNum(msg.priority) ? msg.priority : 0;
      const source = typeof msg.source === 'string' ? msg.source : '';
      const cur = players[msg.hand];
      // A newer request of the same source always replaces (the operator's latest press wins);
      // another source's lower priority does not cut a playing pattern.
      if (cur && nowMs < cur.pulses[cur.pulses.length - 1].end && priority < cur.priority
          && source !== cur.source) {
        return 'lower priority';
      }
      players[msg.hand] = { priority, source, pulses, next: 0 };
      return 'played';
    }

    // Pulses to start at this frame: [{hand, intensity, duration}] (duration = on time left, ms).
    function tick(nowMs) {
      const out = [];
      for (const hand of ['left', 'right']) {
        const p = players[hand];
        if (!p) continue;
        while (p.next < p.pulses.length) {
          const q = p.pulses[p.next];
          if (nowMs >= q.endOn) { p.next++; continue; }   // its on-time passed: skip
          if (nowMs < q.begin) break;                     // not yet
          p.next++;
          if (q.intensity > 0) out.push({ hand, intensity: q.intensity, duration: q.endOn - nowMs });
          break;
        }
        if (nowMs >= p.pulses[p.pulses.length - 1].end) players[hand] = null;
      }
      return out;
    }

    function clear() { players.left = null; players.right = null; }
    return { offer, tick, clear };
  }

  // ---- Status ----

  // Keeps the newest accepted teleop_status and when it arrived.
  function createStatus() {
    const st = { msg: null, run: null, seq: null, receivedAt: 0, ageAtReceipt: 0 };

    // Returns 'accepted' or the reason it was ignored.
    function offer(msg, nowMs, sessionId) {
      if (!msg || !isNum(msg.run) || !isNum(msg.seq)) return 'bad id';
      const age = totalAgeMs(msg, nowMs, sessionId);
      if (age === null) return 'age unknown';
      const staleMs = isNum(msg.stale_ms) ? msg.stale_ms : DEFAULT_STALE_MS;
      if (age > staleMs) return 'too old';
      if (st.msg && msg.run === st.run && msg.seq <= st.seq) return 'old seq';
      st.msg = msg; st.run = msg.run; st.seq = msg.seq;
      st.receivedAt = nowMs; st.ageAtReceipt = age;
      return 'accepted';
    }

    function reset() { st.msg = null; st.run = null; st.seq = null; }
    return { state: st, offer, reset };
  }

  const isStale = (st, nowMs) => {
    if (!st || !st.msg) return true;
    const staleMs = isNum(st.msg.stale_ms) ? st.msg.stale_ms : DEFAULT_STALE_MS;
    return nowMs - st.receivedAt + st.ageAtReceipt > staleMs;
  };

  // ---- Panel text ----

  // Wrap on spaces to lines of at most maxChars; over maxLines lines, the rest is cut with an ellipsis.
  function wrapText(text, maxChars, maxLines) {
    const all = [];
    let cur = '';
    for (let w of String(text).split(/\s+/).filter((x) => x.length > 0)) {
      if (cur && cur.length + 1 + w.length <= maxChars) { cur += ' ' + w; continue; }
      if (cur) all.push(cur);
      while (w.length > maxChars) { all.push(w.slice(0, maxChars)); w = w.slice(maxChars); }
      cur = w;
    }
    if (cur) all.push(cur);
    if (all.length <= maxLines) return all;
    const lines = all.slice(0, maxLines);
    const last = lines[maxLines - 1];
    lines[maxLines - 1] = (last.length >= maxChars ? last.slice(0, maxChars - 1) : last) + '\u2026';
    return lines;
  }

  const clock = (s) => {
    const n = Math.max(0, Math.floor(s));
    return Math.floor(n / 60) + ':' + String(n % 60).padStart(2, '0');
  };

  // Lines for the panel from the tracker state: {stale, lines: [{text, tone, size, big?}], key}.
  // tone: 'rec' | 'ok' | 'warn' | 'bad' | 'dim' | 'text'. size: 'key' (recording state, current
  // hold or fault, active limit) | 'small' (the rest). `key` changes exactly when the picture does.
  // When stale, nothing of the old state is in the result (never show old state as current).
  function panelLines(st, nowMs) {
    if (isStale(st, nowMs)) {
      return { stale: true, lines: [{ text: 'NO DATA', tone: 'bad', big: true }], key: 'stale' };
    }
    const msg = st.msg;
    const since = Math.max(0, nowMs - st.receivedAt + st.ageAtReceipt) / 1000;  // s since built
    const ageOf = (a) => (isNum(a) ? Math.floor(a + since) : null);
    // A section is shown only while its source (its age when the status was built, plus the
    // status's own age) is within the staleness limit: no section is older than stale_ms.
    const staleS = (isNum(msg.stale_ms) ? msg.stale_ms : DEFAULT_STALE_MS) / 1000;
    const fresh = (a) => isNum(a) && a + since <= staleS;   // an unknown age is not fresh
    const lines = [];
    const add = (text, tone, size) => lines.push({ text, tone, size: size || 'small' });
    const addKey = (text, tone) => add(text, tone, 'key');
    const addWrapped = (text, tone, size, maxLines = 2) =>
      wrapText(text, size === 'key' ? KEY_WRAP : SMALL_WRAP, maxLines).forEach((t) => add(t, tone, size));

    const ep = msg.episode && fresh(msg.episode_age_s) ? msg.episode : null;
    if (!ep) {
      addKey('Recording: no data', 'warn');
    } else {
      if (ep.state === 'RECORDING') {
        addKey('● REC  ep ' + ep.index + '  ' + clock((isNum(ep.elapsed_s) ? ep.elapsed_s : 0) + since), 'rec');
      } else if (ep.state === 'IDLE') {
        addKey('IDLE  X starts ep ' + (ep.index + 1), 'text');
      } else if (ep.state === 'REVIEW') {
        if (ep.discard_armed) addKey('REVIEW ep ' + ep.index + ': DISCARDING\u2026', 'bad');
        else { addKey('REVIEW ep ' + ep.index, 'warn'); add('Y keep, hold Y discard', 'warn'); }
      } else if (ep.state === 'NOT_READY') {
        addKey('NOT READY', 'warn');
      } else {
        addKey('Recording: ' + String(ep.state), 'warn');
      }
      add('kept ' + ep.kept + '  discarded ' + ep.discarded, 'text');
    }

    const t = msg.teleop && fresh(msg.teleop_age_s) ? msg.teleop : null;
    // Information only: the simulator runs much slower than real time (arm data comes late).
    const sim = typeof msg.sim_text === 'string' && msg.sim_text ? msg.sim_text : '';
    if (!t) {
      addKey('Teleop: no data', 'warn');
      if (sim) add(sim, 'warn');
    } else {
      add((t.armed ? 'ARMED' : 'disarmed') + '  ' + (t.clutch ? 'CLUTCH' : 'clutch off') + '  '
        + (t.calibrated ? 'calibrated' : 'NOT CALIBRATED'), t.calibrated ? (t.armed ? 'ok' : 'dim') : 'bad');
      if (t.fault) {
        const a = ageOf(t.fault_age_s);
        addWrapped(t.fault + (a === null ? '' : ' (' + a + ' s ago)'), 'bad', 'key', HOLD_LINES);
      } else if (t.hold) {
        addWrapped(t.hold, 'warn', 'key', HOLD_LINES);
      }
      if (sim) add(sim, 'warn');
      if (t.limit) addWrapped('Limit: ' + t.limit, 'warn', 'key');
      add('scale ' + (isNum(t.scale) ? t.scale.toFixed(2) : '?') + 'x', 'text');
      if (t.gripper) add('gripper: ' + t.gripper, 'text');
    }

    if (ep) {
      const refAge = ep.refusal ? ageOf(ep.refusal_age_s) : null;
      const stopAge = ep.last_event === 'AUTO_STOP' ? ageOf(ep.last_event_age_s) : null;
      const showRef = refAge !== null && refAge < RECENT_S;
      const showStop = stopAge !== null && stopAge < RECENT_S;
      if (showRef && (!showStop || refAge <= stopAge)) {
        addWrapped('Refused ' + refAge + ' s ago: ' + ep.refusal, 'bad');
      } else if (showStop) {
        addWrapped(ep.last_event_text || 'Stopped by itself: no reason given', 'bad');
      }
      if (ep.missed > 0) add('missed ' + ep.missed + ' events', 'warn');
    }

    // The reset line (a headset-started planned motion): only when the status has a reset section,
    // which the sender adds once a reset state has been seen. Its text carries its own age.
    const rs = msg.reset;
    if (rs && typeof rs === 'object') {
      if (!rs.valid) {
        add('Reset: no data', 'warn');
      } else if (typeof rs.text === 'string' && rs.text) {
        const bad = rs.state === 'REFUSED' || (rs.state === 'STOPPED' && rs.reason !== 'RELEASED');
        addWrapped(rs.text, bad ? 'bad' : (rs.state === 'ARRIVED' ? 'ok' : 'text'), 'small', 3);
      }
    }
    return { stale: false, lines, key: lines.map((l) => l.size + '|' + l.tone + '|' + l.text).join('\n') };
  }

  // Where and how each line is drawn on the canvas (pure; computed only when the picture changes):
  // [{text, tone, px, bold, x, y, align, baseline, maxWidth}]. Lines run top-down from TOP_PX. When
  // they would reach the footer (the version text), the small lines shrink first, down to
  // SMALL_MIN_PX; only then do the key lines shrink too.
  function panelLayout(res) {
    if (res.stale) {
      return res.lines.map((l) => ({ text: l.text, tone: l.tone, px: 72, bold: true, x: PANEL_W / 2,
        y: PANEL_H / 2, align: 'center', baseline: 'middle', maxWidth: TEXT_W }));
    }
    const isKey = (l) => l.size === 'key';
    const room = PANEL_H - TOP_PX - FOOTER_PX;
    const keyH = res.lines.filter(isKey).length * KEY_PX * KEY_LINE;
    const nSmall = res.lines.filter((l) => !isKey(l)).length;
    let smallPx = SMALL_PX, keyPx = KEY_PX;
    if (keyH + nSmall * SMALL_PX * SMALL_LINE > room) {
      smallPx = nSmall ? Math.max(SMALL_MIN_PX, (room - keyH) / (nSmall * SMALL_LINE)) : SMALL_PX;
      const smallH = nSmall * smallPx * SMALL_LINE;
      if (keyH + smallH > room) keyPx = Math.max(smallPx, KEY_PX * (room - smallH) / keyH);
    }
    let y = TOP_PX;
    return res.lines.map((l) => {
      const px = isKey(l) ? keyPx : smallPx;
      const out = { text: l.text, tone: l.tone, px: Math.floor(px), bold: isKey(l), x: 16,
        y: Math.round(y), align: 'left', baseline: 'top', maxWidth: TEXT_W };
      y += px * (isKey(l) ? KEY_LINE : SMALL_LINE);
      return out;
    });
  }

  // ---- Panel placement ----

  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm3 = (a) => {
    const n = Math.hypot(a[0], a[1], a[2]);
    return n < 1e-6 ? null : [a[0] / n, a[1] / n, a[2] / n];
  };
  const rotate = (q, v) => {   // rotate v by unit quaternion q = [x, y, z, w]
    const [x, y, z, w] = q;
    const t = cross3([x, y, z], v).map((c) => 2 * c);
    const u = cross3([x, y, z], t);
    return [v[0] + w * t[0] + u[0], v[1] + w * t[1] + u[1], v[2] + w * t[2] + u[2]];
  };

  const validOffset = (o) => Array.isArray(o) && o.length === 3 && o.every(isNum);

  // Panel settings from the status message with defaults and limits applied.
  function panelConfig(panel) {
    const p = panel || {};
    const anchor = p.anchor === 'head' ? 'head' : 'left_controller';
    const def = anchor === 'head' ? HEAD_DEFAULT_OFFSET : CONTROLLER_DEFAULT_OFFSET;
    const width = isNum(p.width_m) ? Math.min(0.6, Math.max(0.05, p.width_m)) : 0.16;
    return {
      anchor, offset: validOffset(p.offset) ? p.offset : def, width,
      visibleDefault: p.visible_default !== false,
    };
  }

  // Model matrix (column-major, 16 numbers) of the panel quad (a unit quad in XY scaled to the panel
  // size), from the head pose {position, orientation [x,y,z,w]} and the left grip position (or null).
  // Anchor left_controller: grip + offset in a facing frame (x right as seen from the head, y world up,
  // z toward the head); without a grip it falls back to the head-locked default. The quad faces the head
  // with world up kept; for the head anchor it takes the head's orientation.
  function panelModel(cfg, head, grip) {
    const w = cfg.width, h = cfg.width * PANEL_H / PANEL_W;
    let centre, x, y, z;
    if (cfg.anchor === 'left_controller' && grip) {
      const flat = norm3([head.position[0] - grip[0], 0, head.position[2] - grip[2]]) || [0, 0, 1];
      const fx = cross3([0, 1, 0], flat);
      const o = cfg.offset;
      centre = [0, 1, 2].map((i) => grip[i] + fx[i] * o[0] + (i === 1 ? o[1] : 0) + flat[i] * o[2]);
      z = norm3([head.position[0] - centre[0], head.position[1] - centre[1], head.position[2] - centre[2]]) || flat;
      x = norm3(cross3([0, 1, 0], z));
      if (!x) { x = fx; z = flat; }
      y = cross3(z, x);
    } else {
      const off = cfg.anchor === 'head' ? cfg.offset : HEAD_DEFAULT_OFFSET;
      const r = rotate(head.orientation, off);
      centre = [head.position[0] + r[0], head.position[1] + r[1], head.position[2] + r[2]];
      x = rotate(head.orientation, [1, 0, 0]);
      y = rotate(head.orientation, [0, 1, 0]);
      z = rotate(head.orientation, [0, 0, 1]);
    }
    return [x[0] * w, x[1] * w, x[2] * w, 0, y[0] * h, y[1] * h, y[2] * h, 0, z[0], z[1], z[2], 0,
      centre[0], centre[1], centre[2], 1];
  }

  // ---- Visibility toggle ----

  // Rising edge of a button. update(true|false) -> true once per press; update(null) (no gamepad)
  // forgets the previous state, and the first reading after that only seeds it, so a button already
  // held when the controller appears is not a press.
  function createEdge() {
    let prev = null;
    return {
      update(pressed) {
        if (pressed === null || pressed === undefined) { prev = null; return false; }
        const edge = prev === false && pressed === true;
        prev = pressed;
        return edge;
      },
    };
  }

  // Stored preference ('1' | '0' | null) over the panel's default.
  const resolveVisible = (stored, visibleDefault) => (stored === '1' ? true : stored === '0' ? false : visibleDefault);

  const api = {
    FEEDBACK_VERSION, PANEL_W, PANEL_H, TOGGLE_BUTTON, KEY_PX, SMALL_PX, TOP_PX, FOOTER_PX,
    downlinkMs, totalAgeMs, layoutPulses, createHaptics, createStatus, isStale,
    wrapText, panelLines, panelLayout, panelConfig, panelModel, createEdge, resolveVisible,
  };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TeleopFeedback = api;
})(typeof self !== 'undefined' ? self : this);
