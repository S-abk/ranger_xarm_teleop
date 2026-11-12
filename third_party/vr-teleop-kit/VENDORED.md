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

Reason for each change:

1. File header comment: marks the file as modified and points here.
2. `xrSeq`, `xrResetCount`, `xrSessionId` declarations and `newXrSessionId()`: state for the sequence number, the reset count and the session id (a random 32-bit number from `crypto.getRandomValues`).
3-4. `reset` listener and `newXrSessionId()` after each `requestReferenceSpace` site (teleop and calibration sessions): count reference-space resets (the room frame jumped) and draw a new session id every time an XR session starts (the room frame was redefined). The id is also written to the page log.
5. Hand-tracked inputs: send `{hand_tracked: true}` for that hand instead of a pose, so the bridge treats it as not present and flags it as hand tracked.
6. `position: rawPos` and `emulated_position`: stream the raw grip position (no wrist-pivot offset) and flag emulated poses.
7. Viewer `emulated_position`: flag an emulated headset position.
8. Unconditional `xr_frame` send with `t_xr`, `seq`, `reset_count`, `session_id`: frames go out even with no controller, so a missing controller is visible as absence. The bridge rejects frames without these fields as "client not patched".

```diff
@@ -1,3 +1,4 @@
+// Modified from oscardvs/vr-teleop-kit commit 80589ee for ranger_xarm_teleop. Changes: raw grip position (no wrist-pivot offset), emulated_position per pose, t_xr frame time, seq counter, reset_count, session_id drawn per XR session, hand-tracked sources not streamed, frame sent even with no controller. See third_party/vr-teleop-kit/VENDORED.md.
 (() => {
   const $ = (id) => document.getElementById(id);
   // Live tick rate of get_action() in the teleop process. Populated from
@@ -252,6 +253,10 @@
   const HAPTIC_MIN_INTENSITY = 0.05;  // dead-zone: don't bother below this
 
   let sent = 0, recv = 0;
+  // xr_frame sequence number, count of XRReferenceSpace "reset" events and the id of the
+  // current XR session (random 32-bit number drawn at every session start), sent with every frame.
+  let xrSeq = 0, xrResetCount = 0, xrSessionId = 0;
+  const newXrSessionId = () => { xrSessionId = crypto.getRandomValues(new Uint32Array(1))[0]; return xrSessionId; };
   const append = (msg) => {
     const line = document.createElement("div");
     line.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
@@ -880,6 +885,8 @@
       xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
       xrRefSpace = await xrSession.requestReferenceSpace("local-floor")
         .catch(() => xrSession.requestReferenceSpace("local"));
+      xrRefSpace.addEventListener("reset", () => { xrResetCount++; append(`reference space reset (#${xrResetCount})`); });
+      append(`xr session id ${newXrSessionId()}`);
       xrSession.addEventListener("end", () => {
         xrSession = null; xrRefSpace = null;
         videoGL = null; cameraAnchor = null;
@@ -1004,6 +1011,8 @@
         xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
         xrRefSpace = await xrSession.requestReferenceSpace("local-floor")
           .catch(() => xrSession.requestReferenceSpace("local"));
+        xrRefSpace.addEventListener("reset", () => { xrResetCount++; append(`reference space reset (#${xrResetCount})`); });
+        append(`xr session id ${newXrSessionId()}`);
 
         calibMode = true;
         calibrating = false;
@@ -1411,6 +1420,8 @@
     for (const src of xrSession.inputSources) {
       if (calibMode || calibrating) calibHandednessSeen.add(src.handedness || "(unset)");
       if (src.handedness !== "left" && src.handedness !== "right") continue;
+      // A tracked hand is not a controller; the bridge treats it as not present.
+      if (src.hand || !src.gripSpace) { controllers[src.handedness] = { hand_tracked: true }; continue; }
       const space = src.gripSpace || src.targetRaySpace;
       const pose = frame.getPose(space, xrRefSpace);
       if (!pose) continue;
@@ -1434,7 +1445,8 @@
         calibBuffers[src.handedness].push({ p: rawPos, R: quatToMat3(orient) });
       }
       controllers[src.handedness] = {
-        position: posOut,
+        position: rawPos,
+        emulated_position: !!pose.emulatedPosition,
         orientation: orient,
         buttons: gp ? gp.buttons.map((b) => ({ p: b.pressed, t: b.touched, v: b.value })) : [],
         axes:    gp ? Array.from(gp.axes) : [],
@@ -1503,11 +1515,10 @@
       viewer = {
         position: xyz(vt.position),
         orientation: xyzw(vt.orientation),
+        emulated_position: !!vp.emulatedPosition,
       };
     }
 
-    if (Object.keys(controllers).length > 0) {
-      wsSend({ type: "xr_frame", t_client: performance.now(), controllers, viewer });
-    }
+    wsSend({ type: "xr_frame", t_client: performance.now(), t_xr: _time, seq: xrSeq++, reset_count: xrResetCount, session_id: xrSessionId, controllers, viewer });
   }
 })();
```

## Running

```
tools/relay_setup.sh    # creates .venv (git-ignored) and installs requirements-relay.txt
tools/relay.sh          # serves http://127.0.0.1:8443 (WebSocket at /ws)
adb reverse tcp:8443 tcp:8443   # then open http://localhost:8443 in the Quest browser
```
