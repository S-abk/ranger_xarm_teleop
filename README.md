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
| `tools/` | No-ROS CLIs: `relay_setup.sh`, `relay.sh`, `record_relay.py`, `fake_quest.py`, `analyze_input.py` |
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
hand-tracked input source counts as not present. The headset pose is never published as a
controller pose. An emulated position is published with `emulated_position` set in the status.

### Frames

`quest_world` is the WebXR `local-floor` space converted once, in the bridge, to ROS axes:
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
A missing hand gives zeros for its block.

## Recordings

The bridge's `record_path` parameter and `tools/record_relay.py` write the raw relay text, one
JSON object per line: `{"t_recv": <unix s>, "msg": {...}}` (or `"raw"` for non-JSON text).
`tools/analyze_input.py <file> --hand right [--from s --to s]` reports the still-controller spread
(pass line 10 mm / 2°), the longest frame gap, every absent or emulated interval, re-acquisition
jumps, reference-space resets, viewer yaw span and the transport timing (clock offset and jitter).
