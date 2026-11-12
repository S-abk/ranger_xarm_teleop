# Vendored: vr-teleop-kit

- Upstream: https://github.com/oscardvs/vr-teleop-kit
- Commit: `80589ee`
- Licence: Apache-2.0 (`LICENSE`, verbatim from upstream)

## Included

Verbatim from upstream, except `client.js` (patched, see below):

- `LICENSE`
- `src/vr_teleop_kit/__init__.py`
- `src/vr_teleop_kit/relay/__init__.py`
- `src/vr_teleop_kit/relay/server.py`
- `src/vr_teleop_kit/relay/web/index.html`
- `src/vr_teleop_kit/relay/web/client.js` (patched)

Added by this repo (not upstream): `requirements-relay.txt`, `VENDORED.md`.

Not included: everything else upstream, in particular `ik`, `lerobot`, `data`,
`core`, `examples`, `tools`, `pyproject.toml`, `requirements.lock.txt`, and the
data viewer `relay/web/data.html` and `data.js`. `pyproject.toml` is left out
on purpose so colcon does not try to build this directory (`third_party/COLCON_IGNORE`).

When the relay is started as `python -m vr_teleop_kit.relay.server`, the
relative import `from ..data import api` in `server.py` fails. `server.py`
catches this and logs a warning; the relay itself is unaffected.

## Patch

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

```diff
@@ -1,3 +1,4 @@
+// Modified from oscardvs/vr-teleop-kit commit 80589ee for ranger_xarm_teleop. Changes: raw grip position (no wrist-pivot offset), emulated_position per pose, t_xr frame time, seq counter, reset_count, reset_event (how far a reference-space reset moved the frame, sent once on the next frame), session_id drawn per XR session, hand-tracking requested so hands are XRHand sources, profiles sent, only real controllers (thumbstick axes + controller profile) streamed as poses, frame sent even with no controller. See third_party/vr-teleop-kit/VENDORED.md.
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
@@ -876,10 +891,12 @@
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
@@ -1000,10 +1017,12 @@
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
@@ -1411,11 +1430,23 @@
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
@@ -1434,10 +1465,12 @@
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
 
@@ -1503,11 +1536,12 @@
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

## Running

```
tools/relay_setup.sh    # creates .venv (git-ignored) and installs requirements-relay.txt
tools/relay.sh          # serves http://127.0.0.1:8443 (WebSocket at /ws)
adb reverse tcp:8443 tcp:8443   # then open http://localhost:8443 in the Quest browser
```
