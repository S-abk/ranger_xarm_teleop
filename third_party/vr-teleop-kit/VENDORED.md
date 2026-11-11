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
room-fixed and raw, and makes absent, emulated and hand-tracked inputs visible
to the bridge, as required by the interface contract.

Reason for each change:

1. File header comment: marks the file as modified and points here.
2. `xrSeq`, `xrResetCount` declarations: state for the sequence number and reset count.
3-4. `reset` listener after each `requestReferenceSpace` site: count reference-space resets (the room frame jumped).
5. Hand-tracked inputs: send `{hand_tracked: true}` for that hand instead of a pose, so the bridge treats it as not present.
6. `position: rawPos` and `emulated_position`: stream the raw grip position (no wrist-pivot offset) and flag emulated poses.
7. Viewer `emulated_position`: flag an emulated headset position.
8. Unconditional `xr_frame` send with `t_xr`, `seq`, `reset_count`: frames go out even with no controller, so a missing controller is visible as absence.

```diff
@@ -1,3 +1,4 @@
+// Modified from oscardvs/vr-teleop-kit commit 80589ee for ranger_xarm_teleop. Changes: raw grip position (no wrist-pivot offset), emulated_position per pose, t_xr frame time, seq counter, reset_count, hand-tracked sources not streamed, frame sent even with no controller. See third_party/vr-teleop-kit/VENDORED.md.
 (() => {
   const $ = (id) => document.getElementById(id);
   // Live tick rate of get_action() in the teleop process. Populated from
@@ -252,6 +253,8 @@
   const HAPTIC_MIN_INTENSITY = 0.05;  // dead-zone: don't bother below this
 
   let sent = 0, recv = 0;
+  // xr_frame sequence number and count of XRReferenceSpace "reset" events, sent with every frame.
+  let xrSeq = 0, xrResetCount = 0;
   const append = (msg) => {
     const line = document.createElement("div");
     line.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
@@ -880,6 +883,7 @@
       xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
       xrRefSpace = await xrSession.requestReferenceSpace("local-floor")
         .catch(() => xrSession.requestReferenceSpace("local"));
+      xrRefSpace.addEventListener("reset", () => { xrResetCount++; append(`reference space reset (#${xrResetCount})`); });
       xrSession.addEventListener("end", () => {
         xrSession = null; xrRefSpace = null;
         videoGL = null; cameraAnchor = null;
@@ -1004,6 +1008,7 @@
         xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
         xrRefSpace = await xrSession.requestReferenceSpace("local-floor")
           .catch(() => xrSession.requestReferenceSpace("local"));
+        xrRefSpace.addEventListener("reset", () => { xrResetCount++; append(`reference space reset (#${xrResetCount})`); });
 
         calibMode = true;
         calibrating = false;
@@ -1411,6 +1416,8 @@
     for (const src of xrSession.inputSources) {
       if (calibMode || calibrating) calibHandednessSeen.add(src.handedness || "(unset)");
       if (src.handedness !== "left" && src.handedness !== "right") continue;
+      // A tracked hand is not a controller; the bridge treats it as not present.
+      if (src.hand || !src.gripSpace) { controllers[src.handedness] = { hand_tracked: true }; continue; }
       const space = src.gripSpace || src.targetRaySpace;
       const pose = frame.getPose(space, xrRefSpace);
       if (!pose) continue;
@@ -1434,7 +1441,8 @@
         calibBuffers[src.handedness].push({ p: rawPos, R: quatToMat3(orient) });
       }
       controllers[src.handedness] = {
-        position: posOut,
+        position: rawPos,
+        emulated_position: !!pose.emulatedPosition,
         orientation: orient,
         buttons: gp ? gp.buttons.map((b) => ({ p: b.pressed, t: b.touched, v: b.value })) : [],
         axes:    gp ? Array.from(gp.axes) : [],
@@ -1503,11 +1511,10 @@
       viewer = {
         position: xyz(vt.position),
         orientation: xyzw(vt.orientation),
+        emulated_position: !!vp.emulatedPosition,
       };
     }
 
-    if (Object.keys(controllers).length > 0) {
-      wsSend({ type: "xr_frame", t_client: performance.now(), controllers, viewer });
-    }
+    wsSend({ type: "xr_frame", t_client: performance.now(), t_xr: _time, seq: xrSeq++, reset_count: xrResetCount, controllers, viewer });
   }
 })();
```

## Running

```
tools/relay_setup.sh    # creates .venv (git-ignored) and installs requirements-relay.txt
tools/relay.sh          # serves http://127.0.0.1:8443 (WebSocket at /ws)
adb reverse tcp:8443 tcp:8443   # then open http://localhost:8443 in the Quest browser
```
