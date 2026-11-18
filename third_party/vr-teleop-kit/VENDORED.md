# Vendored: vr-teleop-kit

- Upstream: https://github.com/oscardvs/vr-teleop-kit
- Commit: `80589ee`
- Licence: Apache-2.0 (`LICENSE`, verbatim from upstream)

## Included

Verbatim from upstream, except `client.js`, `index.html` and `server.py` (patched, see below):

- `LICENSE`
- `src/vr_teleop_kit/__init__.py`
- `src/vr_teleop_kit/relay/__init__.py`
- `src/vr_teleop_kit/relay/server.py` (patched)
- `src/vr_teleop_kit/relay/web/index.html` (patched)
- `src/vr_teleop_kit/relay/web/client.js` (patched)

Added by this repo (not upstream): `requirements-relay.txt`, `VENDORED.md`, `src/vr_teleop_kit/relay/web/feedback.js`.

Not included: everything else upstream, in particular `ik`, `lerobot`, `data`,
`core`, `examples`, `tools`, `pyproject.toml`, `requirements.lock.txt`, and the
data viewer `relay/web/data.html` and `data.js`. `pyproject.toml` is left out
on purpose so colcon does not try to build this directory (`third_party/COLCON_IGNORE`).

When the relay is started as `python -m vr_teleop_kit.relay.server`, the
relative import `from ..data import api` in `server.py` fails. `server.py`
catches this and logs a warning; the relay itself is unaffected.

## Patch: client.js

`src/vr_teleop_kit/relay/web/client.js` changed as below (`diff -u` of upstream
against this copy; file header paths omitted). The change keeps the poses
room-fixed and raw, and makes absent, emulated and hand-tracked inputs, reference-space
resets and new XR sessions visible to the bridge, as required by the interface contract.
Only a source that is a real controller is streamed as a pose; the bridge applies the same
rule again on its side.

Reason for each change:

1. File header comment: marks the file as modified and points here.
2. `xrSeq`, `xrResetCount`, `xrSessionId` declarations and `newXrSessionId()`: state for the sequence number, the reset count and the session id (a random 32-bit number from `crypto.getRandomValues`).
3-4. `requestSession(..., { optionalFeatures: ["local-floor", "hand-tracking"] })` at both sites, `reset` listener and `newXrSessionId()` after each `requestReferenceSpace` (teleop and calibration sessions): with the hand-tracking feature granted, tracked hands are `XRHand` sources (`src.hand` set) instead of controller-like sources; count reference-space resets (the room frame jumped) and keep the movement the headset reports with each reset (`ev.transform`); draw a new session id every time an XR session starts (the room frame was redefined). The id is also written to the page log.
5. Controller rule: a source is streamed as a controller pose only when it has a gamepad with at least 4 axes (thumbstick) and a controller profile (`oculus-touch*`, `meta-quest-touch*`, `generic-trigger*`) and no `hand` profile. Anything else (an `XRHand`, no `gripSpace`, or a hand presented as a controller-like source: the Quest 3 does this with 10 buttons and 0 axes when the controllers are put down) is sent as `{hand_tracked: true, profiles, buttons: <count>, axes: <count>}`, so the bridge treats it as not present and the analysis can show what it was. The `targetRaySpace` fallback is gone: only `gripSpace` is used.
6. `position: rawPos`, `emulated_position` and `profiles`: stream the raw grip position (no wrist-pivot offset), flag emulated poses, and send the source's WebXR input profiles.
7. Viewer `emulated_position`: flag an emulated headset position.
8. Unconditional `xr_frame` send with `t_xr`, `seq`, `reset_count`, `session_id`: frames go out even with no controller, so a missing controller is visible as absence. The bridge rejects frames without these fields as "client not patched".
   A reference-space reset adds `reset_event: {count, transform}` to the next frame only (`transform` is `{position, orientation}` in WebXR axes, or null when the headset does not report it); frames without a pending reset do not carry the key.
9. Operator feedback: a block before `onXRFrame`, one branch in `ws.onmessage` and one call in `onXRFrame`. The page receives `teleop_haptic` (vibration patterns) and `teleop_status` (a small panel near the left controller) from the robot side. The decisions (age against the sender's echo of this page's `xr_frame`, de-duplication, priority, text, placement) are in `feedback.js`; `client.js` passes messages in, calls the player each XR frame, toggles the panel with the left thumbstick click (kept in `localStorage`), draws it as one textured quad per eye (own 2D canvas, redrawn only when its text changes) and shows the feedback version in the page log, next to the status pill and in the panel corner. The two message types skip the RTT readout (their echo is not a round trip). Until the first accepted `teleop_status` nothing is drawn, and `xr_frame` is unchanged. The header comment (reason 1) names this.

```diff
@@ -1,3 +1,4 @@
+// Modified from oscardvs/vr-teleop-kit commit 80589ee for ranger_xarm_teleop. Changes: raw grip position (no wrist-pivot offset), emulated_position per pose, t_xr frame time, seq counter, reset_count, reset_event (how far a reference-space reset moved the frame, sent once on the next frame), session_id drawn per XR session, hand-tracking requested so hands are XRHand sources, profiles sent, only real controllers (thumbstick axes + controller profile) streamed as poses, frame sent even with no controller, operator feedback (teleop_haptic vibration, teleop_status panel; logic in feedback.js). See third_party/vr-teleop-kit/VENDORED.md.
 (() => {
   const $ = (id) => document.getElementById(id);
   // Live tick rate of get_action() in the teleop process. Populated from
@@ -252,6 +253,20 @@
   const HAPTIC_MIN_INTENSITY = 0.05;  // dead-zone: don't bother below this
 
   let sent = 0, recv = 0;
+  // xr_frame sequence number, count of XRReferenceSpace "reset" events and the id of the
+  // current XR session (random 32-bit number drawn at every session start), sent with every frame.
+  let xrSeq = 0, xrResetCount = 0, xrSessionId = 0;
+  const newXrSessionId = () => { xrSessionId = crypto.getRandomValues(new Uint32Array(1))[0]; return xrSessionId; };
+  // Latest reset, with how far the headset says the frame moved (ev.transform, null if not reported),
+  // in WebXR axes; the next xr_frame carries it once as reset_event.
+  let xrResetPending = null;
+  const onXrReset = (ev) => {
+    xrResetCount++;
+    const t = ev.transform;
+    xrResetPending = { count: xrResetCount, transform: t ? { position: [t.position.x, t.position.y, t.position.z], orientation: [t.orientation.x, t.orientation.y, t.orientation.z, t.orientation.w] } : null };
+    const moved = xrResetPending.transform;
+    append(`reference space reset (#${xrResetCount}), ` + (moved ? `moved ${JSON.stringify(moved.position)} rotated ${JSON.stringify(moved.orientation)}` : "movement not reported"));
+  };
   const append = (msg) => {
     const line = document.createElement("div");
     line.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
@@ -313,6 +328,8 @@
     recvEl.textContent = recv;
     try {
       const data = JSON.parse(e.data);
+      // Operator feedback (feedback.js). Their echo is not a round trip: keep them out of the RTT readout.
+      if (data.type === "teleop_haptic" || data.type === "teleop_status") { onFeedbackMessage(data); return; }
       const t = data.echo && data.echo.t_client;
       if (typeof t === "number") {
         const rtt = performance.now() - t;
@@ -876,10 +893,12 @@
     try {
       gl = canvas.getContext("webgl2", { xrCompatible: true });
       if (!gl) throw new Error("WebGL2 unavailable");
-      xrSession = await navigator.xr.requestSession(xrMode, { optionalFeatures: ["local-floor"] });
+      xrSession = await navigator.xr.requestSession(xrMode, { optionalFeatures: ["local-floor", "hand-tracking"] });
       xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
       xrRefSpace = await xrSession.requestReferenceSpace("local-floor")
         .catch(() => xrSession.requestReferenceSpace("local"));
+      xrRefSpace.addEventListener("reset", onXrReset);
+      append(`xr session id ${newXrSessionId()}`);
       xrSession.addEventListener("end", () => {
         xrSession = null; xrRefSpace = null;
         videoGL = null; cameraAnchor = null;
@@ -1000,10 +1019,12 @@
       try {
         gl = canvas.getContext("webgl2", { xrCompatible: true });
         if (!gl) throw new Error("WebGL2 unavailable");
-        xrSession = await navigator.xr.requestSession(xrMode, { optionalFeatures: ["local-floor"] });
+        xrSession = await navigator.xr.requestSession(xrMode, { optionalFeatures: ["local-floor", "hand-tracking"] });
         xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
         xrRefSpace = await xrSession.requestReferenceSpace("local-floor")
           .catch(() => xrSession.requestReferenceSpace("local"));
+        xrRefSpace.addEventListener("reset", onXrReset);
+        append(`xr session id ${newXrSessionId()}`);
 
         calibMode = true;
         calibrating = false;
@@ -1348,6 +1369,150 @@
     gl.disable(gl.BLEND);
   };
 
+  // ---- Operator feedback (feedback.js; added by ranger_xarm_teleop) ----
+  // teleop_haptic: vibration patterns for the controllers. teleop_status: a small panel near the
+  // left controller. All decisions (age, de-duplication, priority, text) are in feedback.js;
+  // this is only the glue and the drawing. Nothing here changes the xr_frame stream, and until
+  // the first accepted teleop_status nothing is drawn.
+  const FB = window.TeleopFeedback || null;
+  const fbVersion = FB ? FB.FEEDBACK_VERSION : "feedback.js missing";
+  append(`client ${fbVersion}`);
+  if ($("client-version")) $("client-version").textContent = `client ${fbVersion}`;
+  const FB_VISIBLE_KEY = "vrteleop:feedback_panel_visible_v1";
+  const fbHaptics = FB && FB.createHaptics();
+  const fbStatus = FB && FB.createStatus();
+  const fbEdge = FB && FB.createEdge();
+  let fbCfg = FB && FB.panelConfig(null);
+  let fbAccepted = false;        // a status was accepted on this page load
+  let fbStored = null;           // '1' | '0' | null: the operator's choice of visibility
+  try { fbStored = localStorage.getItem(FB_VISIBLE_KEY); } catch (e) { append(`feedback: localStorage read failed: ${e.message}`); }
+  let fbSession = 0, fbLines = null, fbCanvas = null, fbLogAt = -1e9;
+  // At most one log line per 5 s, so a bad 5 Hz stream cannot flood the page log.
+  const fbLog = (msg) => {
+    const now = performance.now();
+    if (now - fbLogAt < 5000) return;
+    fbLogAt = now; append(msg);
+  };
+
+  const onFeedbackMessage = (data) => {
+    if (!FB) return;
+    try {
+      const now = performance.now();
+      if (data.type === "teleop_haptic") {
+        fbHaptics.offer(data, now, xrSessionId, !!xrSession && !calibMode);
+      } else if (fbStatus.offer(data, now, xrSessionId) === "accepted") {
+        fbAccepted = true;
+        fbCfg = FB.panelConfig(data.panel);
+      }
+    } catch (e) { fbLog(`feedback message failed: ${e.message}`); }
+  };
+
+  const FB_TONES = { rec: "#ff3b3b", ok: "#5be37a", warn: "#ffc94d", bad: "#ff7a5c", dim: "#a0a8b8", text: "#f4f6fa" };
+  const _drawFeedbackCanvas = (res) => {
+    if (!fbCanvas) { fbCanvas = document.createElement("canvas"); fbCanvas.width = FB.PANEL_W; fbCanvas.height = FB.PANEL_H; }
+    const c = fbCanvas.getContext("2d");
+    c.clearRect(0, 0, FB.PANEL_W, FB.PANEL_H);
+    c.fillStyle = "#10131a";
+    c.beginPath();
+    if (c.roundRect) c.roundRect(0, 0, FB.PANEL_W, FB.PANEL_H, 24); else c.rect(0, 0, FB.PANEL_W, FB.PANEL_H);
+    c.fill();
+    c.lineWidth = 4; c.strokeStyle = res.stale ? FB_TONES.bad : "#2a2e38"; c.stroke();
+    c.textBaseline = "top"; c.textAlign = "left";
+    res.lines.forEach((l, i) => {
+      c.fillStyle = FB_TONES[l.tone] || FB_TONES.text;
+      if (l.big) { c.font = "bold 72px sans-serif"; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(l.text, FB.PANEL_W / 2, FB.PANEL_H / 2, 480); c.textAlign = "left"; c.textBaseline = "top"; }
+      else { c.font = "26px sans-serif"; c.fillText(l.text, 16, 8 + i * 29, 480); }  // 480: squeezed if wider
+    });
+    c.font = "14px sans-serif"; c.fillStyle = FB_TONES.dim; c.textAlign = "right";
+    c.fillText(fbVersion, FB.PANEL_W - 14, FB.PANEL_H - 20);
+  };
+
+  // Textured unit quad with the video quad's program and buffers (see _drawVideoQuad).
+  const _drawTexQuad = (VP, modelMat, tex) => {
+    const { prog, vbo, ibo, aPos, aTex, uMvp, uTex } = videoGL;
+    gl.useProgram(prog);
+    gl.uniformMatrix4fv(uMvp, false, mat4Mul(VP, modelMat));
+    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
+    gl.enableVertexAttribArray(aPos);
+    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
+    gl.enableVertexAttribArray(aTex);
+    gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 16, 8);
+    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
+    gl.activeTexture(gl.TEXTURE0);
+    gl.bindTexture(gl.TEXTURE_2D, tex);
+    gl.uniform1i(uTex, 0);
+    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
+  };
+
+  // Per XR frame, after the haptic loop: vibration, visibility toggle, panel. Never throws.
+  const onFeedbackFrame = (frame, layer, now) => {
+    try {
+      if (fbSession !== xrSessionId) { fbSession = xrSessionId; fbHaptics.clear(); fbEdge.update(null); }
+      let left = null;
+      for (const src of xrSession.inputSources) if (src.handedness === "left") left = src;
+      const plays = fbHaptics.tick(now);
+      if (plays.length && !calibMode) {
+        for (const src of xrSession.inputSources) {
+          const ha = src.gamepad && src.gamepad.hapticActuators;
+          if (!ha || !ha[0]) continue;
+          for (const p of plays) {
+            if (p.hand !== src.handedness) continue;
+            const r = ha[0].pulse(p.intensity, p.duration);
+            if (r && r.catch) r.catch((e) => fbLog(`feedback haptic failed: ${e.message}`));
+          }
+        }
+      }
+      const btn = left && left.gamepad && left.gamepad.buttons[FB.TOGGLE_BUTTON];
+      const pressedEdge = fbEdge.update(btn ? !!btn.pressed : null);
+      if (!fbAccepted || calibMode) return;
+      if (pressedEdge) {
+        fbStored = FB.resolveVisible(fbStored, fbCfg.visibleDefault) ? "0" : "1";
+        try { localStorage.setItem(FB_VISIBLE_KEY, fbStored); } catch (e) { fbLog(`feedback: localStorage write failed: ${e.message}`); }
+      }
+      if (!FB.resolveVisible(fbStored, fbCfg.visibleDefault)) return;
+      const vpose = frame.getViewerPose(xrRefSpace);
+      if (!vpose) return;
+      fbLines = FB.panelLines(fbStatus.state, now);   // cheap; every frame, so old state never outlives its staleness
+      if (!videoGL) _initVideoGL();
+      if (!videoGL.fbTex) {
+        videoGL.fbTex = gl.createTexture(); videoGL.fbKey = null;
+        gl.bindTexture(gl.TEXTURE_2D, videoGL.fbTex);
+        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
+        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
+        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
+        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
+      }
+      if (videoGL.fbKey !== fbLines.key) {
+        _drawFeedbackCanvas(fbLines);
+        gl.bindTexture(gl.TEXTURE_2D, videoGL.fbTex);
+        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
+        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
+        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, fbCanvas);
+        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
+        videoGL.fbKey = fbLines.key;
+      }
+      let grip = null;
+      if (left && left.gripSpace && !left.hand) {
+        const gp = frame.getPose(left.gripSpace, xrRefSpace);
+        if (gp) grip = xyz(gp.transform.position);
+      }
+      const head = { position: xyz(vpose.transform.position), orientation: xyzw(vpose.transform.orientation) };
+      const model = FB.panelModel(fbCfg, head, grip);
+      gl.enable(gl.BLEND);
+      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // the texture is premultiplied
+      gl.disable(gl.DEPTH_TEST);
+      for (const view of vpose.views) {
+        const vpRect = layer.getViewport(view);
+        gl.viewport(vpRect.x, vpRect.y, vpRect.width, vpRect.height);
+        _drawTexQuad(mat4Mul(view.projectionMatrix, view.transform.inverse.matrix), model, videoGL.fbTex);
+      }
+      gl.disable(gl.BLEND);
+    } catch (e) {
+      gl.disable(gl.BLEND);
+      fbLog(`feedback frame failed: ${e.message}`);
+    }
+  };
+
   function onXRFrame(_time, frame) {
     if (!xrSession) return;
     xrSession.requestAnimationFrame(onXRFrame);
@@ -1411,11 +1576,23 @@
     for (const src of xrSession.inputSources) {
       if (calibMode || calibrating) calibHandednessSeen.add(src.handedness || "(unset)");
       if (src.handedness !== "left" && src.handedness !== "right") continue;
-      const space = src.gripSpace || src.targetRaySpace;
+      // A tracked hand is not a controller; the bridge treats it as not present. Without the
+      // "hand-tracking" feature the Quest 3 presents hands as controller-like sources (10 buttons,
+      // 0 axes, src.hand null), so a source counts as a controller only when it has thumbstick
+      // axes and a controller profile.
+      const gp = src.gamepad;
+      const profiles = Array.isArray(src.profiles) ? Array.from(src.profiles) : [];
+      const isController = !!gp && gp.axes.length >= 4 && profiles.length > 0
+        && !profiles.some((p) => /hand/i.test(p))
+        && profiles.some((p) => /^(oculus-touch|meta-quest-touch|generic-trigger)/.test(p));
+      if (src.hand || !src.gripSpace || !isController) {
+        controllers[src.handedness] = { hand_tracked: true, profiles, buttons: gp ? gp.buttons.length : 0, axes: gp ? gp.axes.length : 0 };
+        continue;
+      }
+      const space = src.gripSpace;
       const pose = frame.getPose(space, xrRefSpace);
       if (!pose) continue;
       const t = pose.transform;
-      const gp = src.gamepad;
       const orient = xyzw(t.orientation);
       const rawPos = xyz(t.position);
       // Shift the reported readout point in the controller's local frame so
@@ -1434,10 +1611,12 @@
         calibBuffers[src.handedness].push({ p: rawPos, R: quatToMat3(orient) });
       }
       controllers[src.handedness] = {
-        position: posOut,
+        position: rawPos,
+        emulated_position: !!pose.emulatedPosition,
         orientation: orient,
         buttons: gp ? gp.buttons.map((b) => ({ p: b.pressed, t: b.touched, v: b.value })) : [],
         axes:    gp ? Array.from(gp.axes) : [],
+        profiles,
       };
     }
 
@@ -1461,6 +1640,9 @@
       }
     }
 
+    // Operator feedback: vibration, panel toggle and panel (see onFeedbackFrame).
+    if (FB) onFeedbackFrame(frame, layer, performance.now());
+
     // ---- Calibration trigger / capture lifecycle (in-VR) ----
     if (calibMode) {
       // Quest controller buttons[1] is "grip" (analog 0..1 .value).
@@ -1503,11 +1685,12 @@
       viewer = {
         position: xyz(vt.position),
         orientation: xyzw(vt.orientation),
+        emulated_position: !!vp.emulatedPosition,
       };
     }
 
-    if (Object.keys(controllers).length > 0) {
-      wsSend({ type: "xr_frame", t_client: performance.now(), controllers, viewer });
-    }
+    const xrFrame = { type: "xr_frame", t_client: performance.now(), t_xr: _time, seq: xrSeq++, reset_count: xrResetCount, session_id: xrSessionId, controllers, viewer };
+    if (xrResetPending) { xrFrame.reset_event = xrResetPending; xrResetPending = null; }
+    wsSend(xrFrame);
   }
 })();
```

## Patch: index.html

`src/vr_teleop_kit/relay/web/index.html` loads `feedback.js` before `client.js`, bumps the cache-busting `v=` of `client.js` (37 to 38) and adds an element that shows the client version, so the operator can see that the reloaded page runs the new code.

```diff
@@ -245,6 +245,7 @@
     <div class="status-strip">
       <span class="pill" id="conn-pill">connecting…</span>
       <span class="pill" id="calib-pill">checking calibration…</span>
+      <span class="pill" id="client-version"></span>
     </div>
     <h1>VR Teleop</h1>
     <p class="sub">Drive your robot arm from your headset.</p>
@@ -522,6 +523,7 @@
   <canvas id="xr-canvas"></canvas>
 
   <!-- bump v= to force the Quest browser to refetch the JS after a change -->
-  <script src="/static/client.js?v=37"></script>
+  <script src="/static/feedback.js?v=1"></script>
+  <script src="/static/client.js?v=38"></script>
 </body>
 </html>
```

## Patch: server.py

`src/vr_teleop_kit/relay/server.py` rebroadcasts `teleop_haptic` and `teleop_status` to the other clients. Types not in `RELAY_TYPES` are not relayed, so without this the headset would never see them.

```diff
@@ -66,6 +66,8 @@
     # terminal, so this is the only clean way to end it — killing the
     # process would lose the episode in progress.
     "record_control",
+    # Robot -> headset operator feedback: vibration patterns, status panel (web/feedback.js).
+    "teleop_haptic", "teleop_status",
 }
 
 
```

## Added: feedback.js

`src/vr_teleop_kit/relay/web/feedback.js` is new (not in upstream): the pure logic of the operator feedback (no DOM, no GL), also loaded by node for the tests in `src/ranger_xarm_teleop_bridge/test/js/`.

## Running

```
tools/relay_setup.sh    # creates .venv (git-ignored) and installs requirements-relay.txt
tools/relay.sh          # serves http://127.0.0.1:8443 (WebSocket at /ws)
adb reverse tcp:8443 tcp:8443   # then open http://localhost:8443 in the Quest browser
```
