# ranger_xarm_teleop

VR teleoperation and demonstration recording for the Ranger Mini V3 + xArm 6, driven from a
Meta Quest 3 through a WebXR page. One code path for Gazebo, Isaac Sim and hardware.
Depends on the robot packages in `ranger_xarm_ws` (built separately).

## Layout

| Path | What |
|---|---|
| `src/ranger_xarm_teleop_interfaces` | `InputStatus` / `HandStatus` messages |
| `src/ranger_xarm_teleop_bridge` | Bridge node (relay WebSocket → `/teleop/input/*`), JSONL replay node, pure parsing/conversion/analysis modules and their tests |
| `third_party/vr-teleop-kit` | Vendored relay and WebXR web client (upstream commit `80589ee`, Apache-2.0, minimal patch listed in `VENDORED.md`) |
| `tools/` | No-ROS CLIs: `relay_setup.sh`, `relay.sh`, `record_relay.py`, `fake_quest.py`, `analyze_input.py`, `clip_recording.py` |
| `recordings/` | Local headset recordings (git-ignored) |

## Build and test

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
colcon test && colcon test-result --verbose
# pure unit tests, no ROS needed:
python3 -m pytest -q $(find src -type d -path '*/test/unit')
```

The bridge uses the apt `python3-websockets` (10.x API) and `python3-numpy`; nothing is pip-installed
outside the relay's own virtual environment.

## Run the headset input bridge

```bash
tools/relay_setup.sh                 # once: creates .venv with the relay's dependencies
tools/relay.sh                       # terminal 1: relay on 127.0.0.1:8443
adb reverse tcp:8443 tcp:8443        # after every USB re-plug; Quest browser → http://localhost:8443
source install/setup.bash
ros2 launch ranger_xarm_teleop_bridge bridge.launch.py record_path:=recordings/session.jsonl   # terminal 2
```

Without a headset, `tools/fake_quest.py <recording.jsonl>` plays a recording (or a test fixture)
into the relay as if it were the Quest. `ros2 run ranger_xarm_teleop_bridge replay_node --ros-args -p file:=<recording.jsonl>`
republishes a recording directly through the same bridge code path (no relay needed).

## Topics (interface contract, input side)

| Topic | Type | QoS |
|---|---|---|
| `/teleop/input/left/pose`, `/teleop/input/right/pose` | `geometry_msgs/PoseStamped` in `quest_world` (raw grip pose) | best effort, depth 1 |
| `/teleop/input/joy` | `sensor_msgs/Joy` | best effort, depth 1 |
| `/teleop/input/status` | `ranger_xarm_teleop_interfaces/InputStatus` | reliable, depth 10 |

All messages built from one headset frame carry the identical `header.stamp` (ROS time at
publication). A pose is published only for a controller whose pose the headset reported; a
hand-tracked input source counts as not present and is flagged `hand_tracked` in the status.
The headset pose is never published as a controller pose. An emulated position is published
with `emulated_position` set in the status.

**Hands are never controllers.** When the Touch controllers are put down, the Quest 3 switches
to hand tracking and, unless the session has the `hand-tracking` feature, presents each hand as a
controller-like input source (10 buttons, 0 axes, no `XRHand`). The web client requests the
feature and, independently, streams a source as a controller only if it has at least 4 gamepad
axes and a controller profile; the bridge applies the same rule again (`ParsedFrame.controller`):
a source with fewer than 4 axes, or with a `hand` profile, is published as not present with
`hand_tracked` set, whatever the client said. A real Touch controller reports 12 buttons and
4 axes. `tools/analyze_input.py` prints every source's button and axis counts and profiles and
warns when one looks like a hand.

`InputStatus.session_id` is a random 32-bit number the web client draws each time an XR
session starts (Start Teleop, page reload). Together with `reset_count` it identifies one
stable `quest_world` frame: when either changes, earlier poses and the calibration no longer
apply (safety gate G-4, G-13). Frames without `session_id`, `seq`, `t_xr` or `reset_count`
come from an unpatched client and are rejected as malformed.

### Frames

`quest_world` is the WebXR `local-floor` space of the current XR session (see `session_id`
above) converted once, in the bridge, to ROS axes:
`x_ros = -z_xr`, `y_ros = -x_xr`, `z_ros = y_xr`, applied to the position and to the rotation
(`q_ros = (-qz, -qx, qy, qw)` for an xyzw quaternion). No other node swaps or negates axes.

### Joy index order

Buttons (int 0/1), left block then right block:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| | L trigger | L grip | L thumbstick press | L X | L Y | R trigger | R grip | R thumbstick press | R A | R B |

Axes (float), left block then right block:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| | L trigger value | L grip value | L stick x | L stick y | R trigger value | R grip value | R stick x | R stick y |

Values are the raw WebXR `xr-standard` gamepad values (stick y is positive when pushed back).
A missing hand, or a source that is not a real controller, gives zeros for its block.

Verification status against a real Quest 3 Touch Plus recording (`recordings/occlusion.jsonl`,
12 buttons, 4 axes, controller held but not operated): index 1 (grip) is consistent with the
analog squeeze value (partial values up to 0.23 while touched); every other index is
**unverified** (nothing was pressed, no stick motion). The headset also reports capacitive
touch on indices 0, 4, 5, 7, 8, 9, 10 and analog values on 7..10 (thumb rest / proximity
sensors, not mapped). A button-by-button recording is needed to confirm 0, 3, 4, 5 and
`axes[2]`, `axes[3]`; the `gamepad activity` table of `tools/analyze_input.py` shows per-index
pressed/touched counts and value ranges for that purpose.

## Recordings

The bridge's `record_path` parameter and `tools/record_relay.py` write the raw relay text, one
JSON object per line: `{"t_recv": <unix s>, "msg": {...}}` (or `"raw"` for non-JSON text).
`tools/analyze_input.py <file> --hand right [--from s --to s]` reports the still-controller spread
in the room and, for comparison, relative to the headset (a room-fixed pose has a small room
spread and a headset spread that follows the head), the longest frame gap, every absent or
emulated interval, re-acquisition jumps, reference-space resets, XR session changes, the input
sources (button/axis counts, profiles, hand-like warning), per-index gamepad activity, viewer
yaw span and the transport timing (clock offset and jitter).
`tools/clip_recording.py <in> <out> --from s --to s` cuts a clip verbatim, e.g. to promote it
to a test fixture (`test/fixtures/clip_*.jsonl` are such clips from real recordings).
