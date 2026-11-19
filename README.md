# ranger_xarm_teleop

VR teleoperation and demonstration recording for the Ranger Mini V3 + xArm 6, driven from a
Meta Quest 3 through a WebXR page. One code path for Gazebo, Isaac Sim and hardware.
Depends on the robot packages in `ranger_xarm_ws` (built separately).

## Layout

| Path | What |
|---|---|
| `src/ranger_xarm_teleop_interfaces` | `InputStatus` / `HandStatus`, `TeleopState`, `ArmOwner`, `GripperCommand`, `EpisodeState` messages, `SetArmOwner` service |
| `src/ranger_xarm_teleop_bridge` | Bridge node (relay WebSocket → `/teleop/input/*`), JSONL replay node, pure parsing/conversion/analysis modules and their tests |
| `src/ranger_xarm_teleop_servo` | Teleop node (clutch-relative mapping, safety gate, damped least-squares solver, MoveIt Servo joint-jog commands), command mux node, gripper relay node, and the pure modules behind them with their tests |
| `src/ranger_xarm_teleop_bringup` | MoveIt Servo configuration, the teleop trajectory controller, launch files for mock execution and a running simulator (`record:=true` adds the recorder), simulation-only limit values |
| `src/ranger_xarm_teleop_record` | Demonstration episodes: episode node (left controller, rosbag2 recorder control, sidecar), deployment profiles, the pure checker, consistency check and fixed-rate export, and session tools (see Recording demonstrations) |
| `third_party/vr-teleop-kit` | Vendored relay and WebXR web client (upstream commit `80589ee`, Apache-2.0, minimal patch listed in `VENDORED.md`) |
| `tools/` | No-ROS CLIs: `relay_setup.sh`, `relay.sh`, `record_relay.py`, `fake_quest.py`, `analyze_input.py`, `clip_recording.py`. With ROS: `move_arm.py`, `recover_arm.py`, `retreat_arm.py`, `compare_keepout_moveit.py`, `keepout_workspace.py` (sourced workspace only), `replay_measured.sh`, `measure_tracking.py` (`report` needs no ROS), `record_session.py` (`halts` needs no ROS). No ROS: `workspace_map.py`, `servo_rigid_pairs.py` |
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
outside the relay's own virtual environment. The teleop packages need the robot packages
(`ranger_xarm_ws`), `moveit_servo` and `ros2_control` sourced. `colcon test` starts a mock
execution stack (no simulator); the Gazebo tests are opt-in:
`TELEOP_TEST_GAZEBO=1 python3 -m pytest src/ranger_xarm_teleop_bringup/test/test_gazebo_stack.py`.

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
| `/teleop/input/head/pose` | `geometry_msgs/PoseStamped` in `quest_world` (headset pose) | best effort, depth 1 |
| `/teleop/input/joy` | `sensor_msgs/Joy` | best effort, depth 1 |
| `/teleop/input/status` | `ranger_xarm_teleop_interfaces/InputStatus` | reliable, depth 10 |

All messages built from one headset frame carry the identical `header.stamp` (ROS time at
publication). A pose is published only for a controller whose pose the headset reported; a
hand-tracked input source counts as not present and is flagged `hand_tracked` in the status.
The headset pose is never published as a controller pose; it goes only to
`/teleop/input/head/pose`, for every frame that has a headset pose, with the same stamp and the
same axis conversion, and is used only to judge whether a controller is in view and for
operator displays, never as an arm command. An emulated position is published
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
apply (safety gate G-4, G-13). When the headset resets the reference space, the client sends
`reset_event: {count, transform}` once on the next frame, where `transform` is how far the headset
says the frame moved (`{position, orientation}` in WebXR axes) or `null` when it did not say; the
bridge logs every change of `reset_count` with that distance and angle, and
`tools/analyze_input.py` lists them per reset. Frames without `session_id`, `seq`, `t_xr` or `reset_count`
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

Values are the raw WebXR `xr-standard` gamepad values, not flipped. A missing hand, or a
source that is not a real controller (a tracked hand), gives zeros for its block, except the
trigger value: it is NaN whenever there is no trigger value (no controller, or a controller
that sends none), never 0.0, which would read as released.

**Verified on a Quest 3 Touch Plus** (`recordings/button_clip.jsonl`, right controller, each
control operated once in the order trigger, grip, thumbstick click, A, B, then the four stick
directions; nine 0.1 s cuts of it are the fixture `test/fixtures/clip_controller_buttons.jsonl`,
checked by `test/unit/test_real_clips.py`):

| Control | XR source | Joy (right block) | Measured |
|---|---|---|---|
| trigger | `buttons[0]` | button 5, axis 4 | pressed, analog value 1.0 |
| grip | `buttons[1]` | button 6, axis 5 | pressed, analog value 1.0 |
| thumbstick click | `buttons[3]` | button 7 | pressed |
| A (X on the left) | `buttons[4]` | button 8 | pressed |
| B (Y on the left) | `buttons[5]` | button 9 | pressed |
| stick x | `axes[2]` | axis 6 | **left is negative** (−0.98), right positive (+0.90) |
| stick y | `axes[3]` | axis 7 | **up (forward, away from the user) is negative** (−1.00 in a lone stick-up recording), down (back) positive (+0.98) |

The stick signs match the Gamepad API standard-mapping convention (negative left, negative
up); the y sign was settled by a separate recording of a single stick-up push
(`recordings/stick-up.jsonl`, fixture `clip_controller_stick_up.jsonl`), because in the
button-by-button clips the back push came before the forward push. `axes[0]`, `axes[1]`
(touchpad) stay 0.0; `buttons[2]` and `[6]` never report anything.
**`buttons[7]`..`[11]` are capacitive touch sensors** (thumb rest and proximity): they report
`touched` (indices 7–10 on most frames, 11 on a few) and analog values up to 1.0, and never
report `pressed`. They are not mapped. **The left controller** (`recordings/button_clip_left.jsonl`,
fixture `clip_controller_buttons_left.jsonl`, same procedure) uses the same XR indices for its
Joy block (buttons 0–4, axes 0–3) with the same signs; it reports 13 buttons instead of 12, the
extra `buttons[12]` being always silent.

## Recordings

The bridge's `record_path` parameter and `tools/record_relay.py` write the raw relay text, one
JSON object per line: `{"t_recv": <unix s>, "msg": {...}}` (or `"raw"` for non-JSON text).
`tools/analyze_input.py <file> --hand right [--from s --to s]` reports the controller's pose
spread in the room and, for comparison, relative to the headset (a room-fixed pose has a small
room spread and a headset spread that follows the head), the longest frame gap, every absent or
emulated interval, re-acquisition jumps, reference-space resets, XR session changes, the input
sources (button/axis counts, profiles, hand-like warning), per-index gamepad activity, viewer
yaw span and the transport timing (clock offset and jitter). The still-controller PASS/FAIL
verdict (10 mm / 2°) is printed only when a window is given with `--from`/`--to`, i.e. when
the caller has chosen the part where the controller was still; without one the numbers are
printed with `[no verdict without a window]`, because over a whole recording the hand usually
moves at some point.
`tools/clip_recording.py <in> <out> --from s --to s` cuts a clip verbatim, e.g. to promote it
to a test fixture (`test/fixtures/clip_*.jsonl` are such clips from real recordings;
`clip_controller_buttons*.jsonl` are nine short cuts of one recording joined together).

## Teleoperating the arm

The teleop node turns the right controller's motion into a tool-frame target
(`/teleop/target_pose`, in `xarm_link_base`, 50 Hz), solves it into joint velocities itself
(damped least squares, below) and sends those to MoveIt Servo as joint-jog commands
(`/servo_node/delta_joint_cmds`, same stamp). MoveIt Servo keeps its collision monitor and
joint-limit checks and feeds the teleop trajectory controller. Nothing here addresses hardware:
the launch files below start mock execution or attach to a running simulator.

```bash
# mock execution (ros2_control mock components, move_group, RViz), system clock
# (rviz:=false starts it without RViz):
ros2 launch ranger_xarm_teleop_bringup teleop_mock.launch.py
# or a simulator that is already running, simulation time:
ros2 launch ranger_xarm_gazebo sim.launch.py headless:=true          # terminal 1
ros2 launch ranger_xarm_teleop_bringup teleop_sim.launch.py          # terminal 2 (starts move_group too)
# then the input: the live bridge (see above) or a recording
ros2 run ranger_xarm_teleop_bridge replay_node --ros-args -p file:=<recording.jsonl> \
    -p wait_for_subscriber_s:=15.0 -p start_delay_s:=3.0
```

**Simulator rendering (machines with two GPUs).** The gate releases the clutch when the newest
joint state is older than `tool_pose_timeout_s` (0.1 s). A simulator well below real time
delivers clock and joint states in bursts, and the gaps between bursts reach that limit: the
clutch then releases on "Arm position late". On a laptop with an integrated and a discrete GPU,
gz renders its window **and its cameras** (`sensors:=true`, also with `headless:=true`) on the
default device, often the integrated one, which also draws the desktop. Measured on one such
machine with both cameras on: 0.36-0.72 of real time on the integrated GPU, with up to 7 such
releases in 4 minutes; 0.87-0.92 and 0-1 per run on the discrete GPU. Render it on the discrete GPU
(NVIDIA PRIME render offload; the NVIDIA driver provides the files):

```bash
# terminal 1, for the simulator only:
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
       __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
ros2 launch ranger_xarm_gazebo sim.launch.py sensors:=true          # window on, or headless:=true
# check the machine with the simulator up, before the teleop stack (refuses while one runs):
ros2 run ranger_xarm_teleop_record record_commission --simulator \
    --limits $(ros2 pkg prefix ranger_xarm_teleop_bringup)/share/ranger_xarm_teleop_bringup/config/SIMULATION_ONLY_limits.yaml
```

With the window on, `nvidia-smi` then lists `gz sim server` and `gz sim gui`; headless it lists
nothing in either case (the server loads `libnvidia-glcore` only with these settings). The check
prints the joint-state gaps and the real-time factor, and warns (exit code 4) when any gap
exceeds `tool_pose_timeout_s`, the p99 gap exceeds half of it, or the simulator runs below 0.6
of real time. During a session the headset panel says "Simulator slow: 0.4x real time" whenever
the real-time factor is below 0.6 (it does not see the gaps themselves).

**Operating it** (right controller):

0. **Start from the ready posture.** A simulator spawns the arm with every joint at 0, where
   the wrist is aligned (joints 4 and 6 share an axis). Move it to ready first: in simulation
   `tools/move_arm.py 0 -0.6 -0.7 0 1.3 0` (it asks the command mux for MoveIt's controller),
   on a robot through MoveIt.
1. **Calibrate (A).** Point the controller along the direction that should mean "arm forward"
   and press A. The horizontal projection of the pointing axis becomes `xarm_link_base` +x; the
   result is published as the static transform `xarm_link_base` -> `quest_world` and is valid only
   for the XR session and reference-space reset count it was made under. After a recentre, a
   tracking recovery or a page reload, calibrate again.
2. **Arm (B).** Toggles arming. Arming asks the command mux for the arm; it is granted only with
   the arm stopped and no trajectory executing. Disarming (B again, or
   `ros2 service call /teleop/disarm std_srvs/srv/Trigger`) gives the arm back to MoveIt.
   Arming is refused while the wrist is near alignment: the solver's orientation conditioning
   at the measured joints below `min_arming_conditioning` (default 0.2; 1.09 at the ready
   posture, 0 at the spawn pose; 0.2 refuses within about 0.32 rad of the wrist singularity
   of the ready posture's family). The log then says `arm refused: ...; move the arm to the
   ready posture first`. Only the start is checked: once armed the arm may go anywhere.
3. **Clutch (grip).** While the grip is held the tool follows the hand, relative to where both
   were when the grip closed. Release holds. Any gate condition (below) also releases, and the
   grip then has to be released and pressed again.
4. **Scale (thumbstick click).** Switches between the fine and the coarse translation scale,
   while armed; every arming starts at fine. Hand translation is multiplied by the scale,
   rotation never. A click during a stroke does not move the target: the mapping restarts at
   that point and only the motion after it is scaled anew. `translation_scale` in
   `/teleop/state` shows the scale in use (or the one the next engagement uses) and the log
   says `translation scale: coarse (2.00)`.
5. **Gripper (trigger).** While armed, pressing the trigger past 0.6 closes the gripper and
   letting it fall below 0.25 opens it, clutch held or not (see Gripper below). After arming,
   and after any interruption of the input, the trigger acts only once it crosses one of the
   two thresholds again: a trigger still held at that moment does nothing until it is released
   (or pressed) again. `gripper_hold` in `/teleop/state` says `AWAIT_TRIGGER` meanwhile.

`ros2 topic echo /teleop/state` shows why the arm is not moving (`hold_reason`), which bound is
shaping the target or the joint command (`active_limit`), the owner of the arm, the calibration
in use, the measured input age, the measured tool pose, and the solver: `conditioning`,
`damped`, `solver_error_*`, `lead_*` and the commanded tool speed (next section).

### Mapping

Clutch-relative in a fixed frame: `scale * (p_now - p_clutch)` and `q_now * inverse(q_clutch)`
in `quest_world`, rotated into `xarm_link_base` by the calibration heading, applied to the tool
pose measured at engagement. `scale` is `translation_scale_fine` or
`translation_scale_coarse` (both default 1, so the switch does nothing until a coarse value is
set); each must be positive and at most the required `max_translation_scale`, or the node
refuses to start. The raw-input check (G-5) stays on the unscaled hand; the step check of the
final target, the offset limit, the lead limit and the speed caps all act on the scaled
target, so at scale 2 a hand faster than half the per-cycle limit is rejected, and hand motion
beyond the speed cap is absorbed by the lead limit as at scale 1. How the controller is held
when the grip closes does not change
the mapping. `wrist_pivot_offset` (controller frame, default zero) moves the tracked point from
the grip origin to the operator's wrist pivot. `tool_frame` (default `xarm_link_tcp`) is the
frame the target is for. The solver works on the chain from `base_frame` to `servo_tip_frame`
in `/robot_description`, plus the fixed transform from the tip to the tool frame taken from
`/tf_static` only: a tool frame that is not rigidly attached to the tip is refused (no
target). The measured tool pose is the forward kinematics of the newest `/joint_states` on
that chain, not a transform-tree lookup.

### Joint commands, singularities and the lead limit

Each cycle the solver turns the target into joint velocities:

- **Units.** The arm Jacobian is divided by the tool speed caps (linear rows by the linear
  cap, angular rows by the angular cap) and multiplied by the joint velocity limits. Its
  singular values are then the share of the speed cap the tool reaches in that direction with
  the joints at their limits, in the same units for translation and rotation. `conditioning`
  in `/teleop/state` is the smallest of them (>= 1 unrestricted, 0 singular).
- **Position first.** Orientation is solved in what the position leaves free. For the xArm 6
  the tool position stays well conditioned through the wrist and shoulder regions, so near
  them the accuracy given up is orientation.
- **Damping**, per level, grows as that level's smallest singular value falls below 0.5
  (`damping_threshold`), up to `max_damping` 0.5 for position and `orientation_max_damping`
  1.0 for orientation. The larger orientation value keeps joints 4 and 6 from turning fast
  against each other near the wrist singularity for a small tool rotation, at the cost of
  some orientation accuracy there. The arm never halts for a singularity; `damped` says when
  accuracy is reduced and `solver_error_*` by how much.
- **Speed caps per level.** Each of the two levels' commands is first limited to its own
  speed cap, which can change the tool's direction of motion a little when both are beyond
  it (`active_limit` 6).
- **Full reach.** Past the reach the arm stretches into the position singularity and comes
  to rest there. When the hand turns back, the arm follows after about 0.8 s: the target
  first comes back through its lead, and at the singularity the way back is the direction
  the arm has lost.
- **Limits.** A joint slows toward a position limit from 0.35 rad and stops 0.15 rad short
  (`joint_limit_zone`, `joint_limit_margin`); the rest of the motion is solved again without
  it. One factor then scales the whole command, so its direction is kept, to stay within the
  joint velocity limits (the smaller of the URDF's and MoveIt's `joint_limits.yaml`, x 0.95)
  and the tool speed caps.
- **Reference.** The error is taken against MoveIt Servo's own commanded state (its output on
  the teleop controller's command topic), seeded from the measured joints at engagement. If
  that output stops following the commands for 0.1 s, the clutch releases.
- **Lead limit (G-14).** The target may not lead the measured tool pose by more than
  `max_lead_translation_m` / `max_lead_rotation_deg` (required, no default). The excess is
  absorbed into the mapping, so after a hand motion faster than the arm the arm does not catch
  up once the hand stops; it follows the hand again from where it is.
- **Stop.** The cycle that releases the clutch sends a zero joint command.
- **Stuck near a collision.** MoveIt Servo slows every joint command by one factor that
  depends only on the distance to the nearest collision (down to 0.001 at contact), so near
  contact no direction leads away. When Servo reports collision deceleration (`servo_status`
  4) and 0.5 s of commanded motion (at least 20 mm or 0.2 rad) moves the tool less than 1 mm
  and 1 deg and less than 2 % of what was commanded, the clutch releases with `hold_reason`
  19 (`SERVO_HALTED`) and the log says `stuck near a collision`. An arm that has only come to
  rest close to the body, with the hand still and a few millimetres of error left, is not
  released: moving the hand away still moves it. MoveIt cannot plan from a stuck state
  either (below).

### Safety gate

| Condition | Result |
|---|---|
| `motion_enabled` false (the default), not armed, or teleop does not own the arm | no target |
| grip released | no target from that cycle on |
| no input for 100 ms, measured from local arrival on a monotonic clock | clutch released |
| controller pose absent or emulated; headset pose absent or emulated; controller outside the view cone; reset count or session id changed; sequence or headset time went backwards (judged on every headset frame, not only the newest one at each 50 Hz cycle) | clutch released |
| raw controller motion between two headset frames faster than the per-cycle limit allows (checked before filtering), or a step of the final target beyond the per-cycle limit | clutch released, never slowed into a smaller jump |
| no calibration for the current session id and reset count | no target |
| measured tool pose older than 100 ms (by joint-state arrival and by its stamp), or MoveIt Servo not confirmed running in joint-jog mode, or its collision checking, `command_in_type` (`speed_units`), joint limit margins (narrower than the solver's) or command timeout (at most the input timeout) no longer as required (re-read every second), or its output no longer following the commands, or a singularity status (never caused by joint jog) | no target; clutch released if engaged |
| no kinematic chain or joint state, or the tool at MoveIt Servo's commanded state more than two lead limits from the measured one | no target (`hold_reason` 20); clutch released if engaged |
| target ahead of the measured tool pose by more than the lead limit | clamped onto the limit, the excess absorbed (`active_limit` 5) |
| target beyond the offset limit from the engagement pose | clamped onto the limit |
| (view cone) | the cone's axis is the headset's forward axis, tilted down by `view_cone_tilt_deg` (default 0) about the headset's left-right axis; its half-angle is the required `view_cone_half_angle_deg` |
| tool point or wrist point (`wrist_frame`) beyond a keep-out plane (`keep_out_planes`, flat list `nx ny nz d` per plane, allowed side `n . x >= d`, in `xarm_link_base`) | pushed back onto the plane; if that breaks another bound, the last accepted target is kept |
| singularity, joint position limit, joint velocity limit, tool speed cap | the solver damps, holds the joint or scales the command (`active_limit` 6-8); never a halt |
| an arm link or the tool nearer the robot's fixed structure than `link_keepout_margin_m` plus the slowing zone (see below) | motion toward it is removed, motion along or away from it kept (`active_limit` 9, `keepout_bound`); the target's lead into the blocked direction is absorbed |
| self-collision, scene collision; a joint inside MoveIt Servo's own margin (a backstop) | MoveIt Servo slows (`active_limit` 4, `servo_status`); when it halts, the clutch is released (`hold_reason` 19) so the target cannot run on ahead of a standing arm |

**The limit values have no defaults.** Maximum offset, view cone, per-cycle limits, speed
caps, lead limits, the largest translation scale and the link keep-out margin are required
parameters, and so are the gripper's range and speed when gripper teleop is enabled; the
teleop node exits if one is missing. For simulation,
`config/SIMULATION_ONLY_limits.yaml` supplies values. That file marks itself simulation-only:
the node then starts only if every ros2_control hardware plugin in the robot description is a
simulator, and `teleop_core.launch.py` refuses the file unless called with `simulation:=true`,
which only `teleop_mock.launch.py` and `teleop_sim.launch.py` do. An extra parameter file
(`teleop_params_file`) may not set a limit value, `motion_enabled`, `gripper_enabled`, the
gripper's joint, range or speed, or the simulation-only flag. A robot needs its own limits
file, written by whoever is responsible for it, and its own launch path.

Clocks: MoveIt Servo, the mux, the teleop node and the controllers use simulation time under a
simulator (their stamps must agree with the joint states); the bridge and the replay node stay
on the system clock. Input age, the target cycle and every timeout use the monotonic clock of
the machine, so 100 ms is 100 ms of real time whatever the simulator does.

### Link keep-out

The teleop node keeps every arm link, and the tool, `link_keepout_margin_m` away from the
robot's own fixed structure (G-15), so pulling the tool back toward the base no longer ends in
MoveIt Servo's collision halt. The model is built from `/robot_description` when it arrives
(`ranger_xarm_teleop_servo/keepout.py`, numpy only):

- **What is kept apart.** Fixed: every link joined to the arm base through fixed joints (base,
  mounts, sensors, anything bolted on). Moving: every link downstream of the first arm joint
  (arm, gripper, cameras, any tool a project attaches). Every moving link against every fixed
  one, and the links moved by the last arm joint against the arm links three or more joints up.
  Left out, and logged at start-up: a moving link attached directly to a fixed one, and a pair
  that is within the margin in every sampled posture. The service `/teleop/keepout_info`
  (`std_srvs/Trigger`) lists the model, the pairs left out and why.
- **The model never reports more clearance than the geometry has.** Moving links are covered
  by spheres (25 mm cells) that contain every triangle; fixed boxes, cylinders and spheres are
  used exactly. Only STL collision meshes are supported; other formats refuse the start.
- **In the solver.** Within 60 mm of the margin, a pair's clearance `d` may fall at most at
  `(d - margin) / 0.2 s`: 0.3 m/s at 60 mm, nothing at the margin, so at 0.25 m/s motion toward
  the structure slows from about 50 mm and stops at the margin; motion along or away from it
  keeps its speed. Where the command breaks that, it is replaced by the nearest
  command that does not (in tool motion; motion across the hand's direction costs more than
  slowing along it).
- **Checks.** A later `/robot_description`, or MoveIt Servo's robot description, with other
  geometry disables teleop until the node is restarted. `TeleopState` reports the clearance,
  its pair and whether the keep-out shaped the command.

### Command mux

Two trajectory controllers claim the arm joints: MoveIt's, and `teleop_arm_controller`, which
MoveIt Servo feeds. ros2_control lets only one be active; an inactive one rejects action goals
and ignores its command topic. `arm_mux_node` reports the owner on `/teleop/mux/owner` and
switches with one strict `switch_controller` call, only when every arm joint has been still for
0.3 s (judged from an unbroken run of joint-state samples), no trajectory goal is active and
the clutch is released. A refused or failed switch leaves the owner as it was and says why.

With gripper teleop, the gripper has the same owner as the arm: MoveIt's gripper controller
(`moveit_gripper_controller`; `xarm_xarm_gripper_traj_controller` in Gazebo) and
`teleop_gripper_controller` are swapped in the same strict switch, and a goal on MoveIt's
gripper controller refuses the switch. The gripper must be stopped too, judged by net travel:
over the same 0.3 s its joint may span at most `gripper_still_band` (0.01 rad). (Judged like
the arm, from sample-to-sample speed, Gazebo read the driven joint as moving at about
0.16 rad/s again and again, and the switch back to MoveIt was refused.) `ArmOwner.gripper` says
the gripper is included. Controllers split between owners (or both of a pair active) are
reported as no owner; the mux then refuses teleop and only switches back to MoveIt. The robot
workspace's mock execution has no gripper controller, so `teleop_mock.launch.py` starts a
stand-in for MoveIt's (`xarm_gripper_traj_controller`, `config/mock_moveit_gripper_controller.yaml`);
move_group cannot command it in mock execution.

### Gripper

Off unless `gripper_enabled` is true (default false; the mock and simulator launch files set
it). Off, the node creates no gripper publisher and does not read the trigger, so a robot that
carries a fixed tool in the gripper can never have it opened by teleop.

- **Mapping: two states with hysteresis.** The request becomes close when the trigger
  (`/teleop/input/joy` axis 4) rises through `gripper_close_threshold` (0.6) and open when it
  falls through `gripper_open_threshold` (0.25). The command then moves to
  `gripper_closed_position` or `gripper_open_position` at `gripper_max_speed`; these three
  are required (no default) and must lie within the driven joint's URDF limits, or the node
  refuses to start. The squeeze is the same on every clutch stroke, and a re-grip that relaxes
  the index finger into the band does not open the gripper.
- **The recorded action.** `/teleop/gripper/command` (`GripperCommand`: joint, position in its
  units, request, where the first command came from) is published in every cycle where the
  gripper is commanded, with the cycle's stamp: the same as `/teleop/target_pose` and
  `/teleop/state`. `/teleop/state` carries the command in force in every cycle
  (`gripper_command`, also while frozen) and, separately, the measured opening
  (`gripper_measured`, from `/joint_states`, with that message's stamp).
- **When it is commanded.** The trigger is a button, not a pose: the gripper depends on the
  trigger and on what concerns the gripper. It is commanded while armed, teleop owns the
  gripper (see Command mux), the input is fresh (a frame within the input timeout, by
  arrival) with no session or sequence fault in any frame since the last cycle, the newest
  frame has a trigger value in [0, 1] (a lost or put-down controller has none), and the joint
  states are recent enough (`tool_pose_timeout_s`) to check the finger keep-out. Otherwise
  nothing is sent and the gripper holds its last command (`gripper_hold`: `NOT_ARMED`,
  `NOT_OWNER`, `INPUT`, `JOINTS_STALE`). Pose-only conditions (controller out of view,
  emulated position, a raw input jump, the headset pose, a reference-space reset, no
  calibration) and arm-side conditions (solver, MoveIt Servo, tool pose, a rejected target
  step) hold the arm but not the gripper, clutch held or not.
- **Arming never moves it.** The first command is the opening the gripper is already holding:
  teleop's own last command after an interruption, else what the controller holds after the
  owner switch (the previous owner's last command, reported by the gripper relay on
  `/teleop/gripper/held`), else, after `gripper_reference_timeout_s` (0.5 s) without one, the
  measured opening. `gripper_reference_source` says which.
- **Measured or assumed.** In Gazebo the opening is measured by the physics engine. In mock
  execution it is the command echoed back. On the hardware driver it is partly assumed: the
  gripper action proxy publishes the commanded target, so a grasp stalled on an object reads
  as closed.
- **Link keep-out.** The model places the fingers at their measured joint values. A step of the
  gripper command that would bring a finger nearer the fixed structure than the margin, and
  nearer than at the current command, is not taken (`gripper_hold` `KEEP_OUT`); the request
  stays, so the gripper continues by itself once the arm has moved away.
- **Backends.** `gripper_relay_node` carries the command to the teleop gripper controller
  (`config/teleop_gripper_controller.yaml`, a trajectory controller with no tolerances and
  no command timeout, so a finger blocked on an object keeps its squeeze) and reports the
  opening that controller holds. The hardware adapter (a `GripperCommand` action client) is
  not built; the relay refuses `mode:=action`.

#### A closed gripper and MoveIt Servo's speed

MoveIt Servo slows every joint while any self-collision pair its collision matrix leaves
enabled is within `self_collision_proximity_threshold` (10 mm in `config/servo.yaml`), in
every direction of motion. Pairs inside the gripper, or between a finger and the link the
gripper is mounted on, keep their distance however the arm moves, but a closed gripper brings
some within 10 mm: with the robot workspace's description a finger is 7.8 mm from the gripper
base when closed (14.5 mm open), and while such pairs were enabled the arm ran at about a fifth
of the commanded speed. The robot workspace's SRDF now disables them.

Such pairs are derived from the description, with no link names (`rigid_pairs.py`): both
links have collision geometry, are moved by the arm, have no joint of MoveIt Servo's group
between them, and are left enabled by the SRDF. The teleop node logs them at start-up as
`<disable_collisions>` lines; `tools/servo_rigid_pairs.py --urdf FILE --srdf FILE` prints the
same. They belong in the SRDF move_group reads: MoveIt Servo takes move_group's collision
matrix (it asks move_group for the planning scene when it starts), so an SRDF given to Servo
alone has no effect. For move_group, disabling them changes nothing for arm plans; for the
gripper's own motion it gives up the check of those pairs; with the robot workspace's
description, sampled every 0.01 rad over the gripper's range, none comes closer than 7.8 mm.
The mock and Gazebo stack tests that compare the tool speed with the gripper open and closed,
and the mock test that drives the closed gripper into the robot's body, skip, naming the
pairs, while any is left enabled.

### Measuring a run

```bash
ros2 service call /teleop/disarm std_srvs/srv/Trigger           # MoveIt must own the arm, or the goal is rejected
tools/move_arm.py [--controller NAME]      # a posture with room to move (simulation only)
tools/replay_measured.sh src/ranger_xarm_teleop_servo/test/fixtures/clutch_circle.jsonl /tmp/run.jsonl
tools/measure_tracking.py report /tmp/run.jsonl
```

reports, per engagement, the distance of the measured tip from the commanded path, the
following error and lag, the delay from controller motion to joint motion, and how far the tip
travels after the last target. The fixtures `clutch_*.jsonl` and `gate_*.jsonl` are synthetic
(`test/fixtures/make_fixtures.py`); `clip_*.jsonl` are cuts of real headset recordings.

### Diagnosing MoveIt Servo halts

With joint-jog commands MoveIt Servo's singularity stop no longer applies; it still halts for a
collision or a joint inside its own margin. The tools below also compute the condition number
of the arm Jacobian that its singularity stop used (`ranger_xarm_teleop_servo/kinematics.py`,
numpy only), for comparing recordings made with pose commands.

```bash
tools/record_session.py record NAME          # start BEFORE arming; Ctrl-C AFTER releasing the grip
tools/record_relay.py --out recordings/NAME_relay.jsonl  # live sessions: the raw headset frames too
tools/record_session.py halts recordings/NAME.jsonl     # no ROS needed
tools/workspace_map.py recordings/NAME.jsonl            # or a URDF file; no ROS needed
```

`record` writes `recordings/NAME.jsonl` (git-ignored): teleop state, tool target, controller
and headset poses, buttons (`/teleop/input/joy`), input status, the joint command, joint
states, MoveIt Servo's status, the arm owner and the robot description; each state row carries
the translation scale (`scale`). It only listens. It refuses to start while the clutch is
engaged, and after Ctrl-C it keeps recording
until the clutch is released (see the timing note below). `halts` prints, per halt, MoveIt
Servo's status, the hold reason, the joint angles, the condition number and its trend over the
second before, the nearest joint limit, the singularity the posture resembles (wrist, elbow,
shoulder) and what was being commanded; then counts by reason, every clutch release by its
hold reason, the share of the engaged time at each translation scale, and the distribution of
the condition number over the engaged time (labelled as the old rule's when the recording comes
from the joint-space solver). `workspace_map.py` reports the condition number along
tool motions from a posture: how far the tool gets in each direction before each threshold.

After a collision halt MoveIt Servo refuses motion in every direction: the arm has to be moved
out by MoveIt.

```bash
tools/recover_arm.py          # disarm, wait until MoveIt owns the arm, plan to the ready posture
```

When the arm is in contact, MoveIt refuses to plan from that start state
(`start state in collision`) and `recover_arm.py` moves nothing. Then:

```bash
tools/retreat_arm.py          # disarm, then back out along the recorded joint path, slowly
```

The teleop node records the measured joint path while teleop is armed (the last 60 s,
`/teleop/joint_path`). `retreat_arm.py` walks it backwards from the arm's posture to the first
point that is clear by the keep-out margin and valid for move_group, within 0.15 m of tool path
and 0.5 rad per joint, and sends that stretch to MoveIt's controller at 0.1 rad/s and 0.03 m/s
at most, without planning. It moves nothing when there is no path, the arm has moved since,
the path goes deeper into contact on the way back or no point within the bounds is clear, and
says why and what to do. Simulation only, like `move_arm.py`.

(simulation only, like `tools/move_arm.py`, which prints why a goal was rejected). Without
`--controller`, `move_arm.py` asks the command mux which controller is MoveIt's (failing that,
it takes the one arm trajectory controller other than teleop's); it never sends to teleop's
controller, and when it finds none it lists the loaded controllers with their state and
joints. If teleop re-engages while MoveIt Servo is halted for a collision, the log says so:
Servo publishes nothing then, and teleop cannot move the arm out of the contact.

For a live session, also run `tools/record_relay.py` from the start. It is one more client of
the relay (no ROS), and its file replays the session exactly through the bridge
(`replay_node`), which the ROS-side recording alone cannot.

The teleop node's log helps tell a busy process from missing input:
- `slow cycle: ...` (throttled), with the longest step of the cycle (logging has its own
  step) and the parts of the joint command: reference, solve and the two publish calls;
- once a minute, the count of slow cycles and of cycles that started late;
- `frame seq N incomplete: ...` when a frame that releases the clutch lacks a part (no pose,
  no joy, or the status reporting no pose), with how long the input callbacks had paused.

Timing note: when a new ROS participant joins the graph (a command-line tool, a node being
started), DDS endpoint matching can delay messages or block a publish call for 0.1 to 0.3 s.
Under a held clutch that is a release on stale data (`hold_reason` 6 or 17), by design. The
teleop node logs `late cycle: ...` when its own cycle was delayed. Start a replay with
`wait_for_subscriber_s` and a few seconds of `start_delay_s`, as above.

## Recording demonstrations

Package `ranger_xarm_teleop_record`. A session records demonstration episodes: the action
(`/teleop/target_pose` and `/teleop/gripper/command`, the joint command alongside), the robot
state (joint states, measured tool pose and gripper opening in `/teleop/state`),
the colour images of the configured cameras, the raw headset input and the base topics when
they exist. Commanded and measured values stay separate fields with their original stamps.

**What runs.** `record:=true` on `teleop_mock.launch.py` or `teleop_sim.launch.py` (or
`ranger_xarm_teleop_record record.launch.py` on its own, before arming) starts a rosbag2 recorder (MCAP, zstd chunk
compression: lossless) paused, and the episode node. The episode node resumes, pauses and
splits the recorder, so each episode is one closed MCAP file. Both start with the stack;
an episode event is only a service call between them, so nothing joins or leaves the ROS
graph during a session and the teleop node knows nothing about recording (with
`record:=false` nothing of it starts).

**Deployment profile.** `profiles/<name>.yaml` names every topic, its kind, whether it is
required, its nominal rate, the clock its stamps are on, the data directory, the recorder
settings and the disk thresholds (`mock` and `gazebo` here, both simulation only). The profile
is copied into every session and episode; the checker's tolerances (`config/checks.yaml`) are
relative to it. A deployment passes its own profile with `record_profile:=<path>`. A profile
may also:

- override the fitted consistency tolerances of `config/checks.yaml` (arm RMS, p99 and
  rotation, gripper) in a `checks` block, and mark some of them `provisional` (set by hand
  before there was data to fit them). Nothing else can be overridden by a profile, so a
  profile cannot switch a failure off. The `gazebo` profile sets the
  arm rotation limit to 1.5 deg this way. The checking tools apply the package's current
  profile of the episode's profile name, so a changed tolerance also judges older episodes;
  every check report names the source and lists the provisional values.
- list `required_fields` of a stream: optional fields (joint `position`, `velocity`,
  `effort`; odometry `pose`, `twist`) whose absence fails an episode (and, for a required
  joint-state stream, refuses START while the newest message lacks the field). An
  absent field (no finite value in the episode) is otherwise only reported, in the sidecar and
  the check report. Gazebo and mock execution publish joint effort as NaN (their joints declare
  no effort state interface), so neither profile requires it.

**Left controller** (the right one is teleop's; the left stick and grip are reserved for the
base):

| Press | When | Effect |
|---|---|---|
| X | idle, teleop armed and owning the arm | start an episode |
| X | recording | stop it (review) |
| Y, released within 1 s | review | keep it |
| Y held 1 s | review | discard it: marked, nothing deleted |
| trigger past 0.6 (re-arms below 0.25) | recording | a time mark inside the episode |

`/teleop/episode/state` (`EpisodeState`) says the state, the episode, elapsed time, flags so
far, the last event and why a press was refused (a required stream missing or below its
rate, no labels, teleop not armed, disk low, recorder busy), the free disk in minutes, and a
ROS-time/wall-time pair. Disarming stops a running episode by itself (`DISARMED`); a gate
fault does not stop it but flags it (`FAULT_RELEASE` with the hold reason). Joy is read as
levels with the teleop node's QoS: one lost message neither loses nor doubles a press.

```bash
ros2 run ranger_xarm_teleop_record episode_labels --profile gazebo   # task label, operator: typed, kept in the data dir
ros2 launch ranger_xarm_teleop_bringup teleop_sim.launch.py record:=true   # before arming
ros2 topic echo /teleop/episode/state --field detail
ros2 run ranger_xarm_teleop_record episode_check ~/teleop_data/<session>          # no ROS graph
ros2 run ranger_xarm_teleop_record episode_export ~/teleop_data/<session> 3       # fixed rate
ros2 run ranger_xarm_teleop_record episode_export ~/teleop_data/<session> 3 --tick-stream camera_wrist
ros2 run ranger_xarm_teleop_record episode_fit ~/teleop_data/<s1> ~/teleop_data/<s2> --out fit.yaml
ros2 run ranger_xarm_teleop_record episode_purge ~/teleop_data                    # dry run
ros2 run ranger_xarm_teleop_record record_commission --profile gazebo             # new machine
ros2 run ranger_xarm_teleop_record record_commission --simulator   # simulator up, before teleop
```

**Data directory** (default `~/teleop_data`; refused inside this repo's checkout, and inside
any other git tree unless ignored there). Every path inside a session is relative, so a
session can be moved and checked or exported elsewhere:

```
<data_dir>/labels.json                 task label and operator in force
<data_dir>/<UTC time>-<suffix>/
  session.json                         commits and dirty flags (this repo, robot workspace), description
                                       and SRDF fingerprints, limits file, teleop/Servo/mux parameters,
                                       frames, clocks, recorder settings, the profile
  robot_description.urdf/.srdf, limits.yaml
  bag/                                 one MCAP file per episode
  episodes/NNNN.json                   the sidecar: status, file, labels, start/stop (ROS, wall,
                                       monotonic), marks, flags, calibration and what it is bound to,
                                       scales, camera intrinsics and extrinsics, base mode, counts,
                                       link quality, clock pairs, real-time factor, absent fields,
                                       + the session block
  checks/NNNN.json, exports/NNNN/, recorder.log
```

Latched data (description, static transforms, the first camera_info) reach a paused recorder
and are dropped, so the sidecar holds them; an episode is interpretable from its MCAP file and
its sidecar alone.

**Checker** (`episode_check`, pure, never changes data). Per stream: rate and largest gap in
the recording clock (gated streams only inside their spans: the target while the clutch is
engaged), the clock domain (one-host rule: robot-side streams on one clock; wall-stamped
input mapped through the clock pairs), the offset to the action stream, and the counts the
episode node saw against what was recorded (a shortfall is `DATA_LOST`). Flags: fault
releases, limits bound for more than a share of the engaged cycles, disarm, low real-time
factor, base motion, link quality (input age, sequence gaps, XR frame-time jitter,
reconnects). A consistency check stands in for a replay: the measured tool pose must follow
the recorded target, and the measured opening the gripper command, within a fitted lag and a
tolerance, over the cycles where teleop commanded (damped or limited cycles are reported
separately). An episode in which teleop never commanded the arm fails with one reason,
`NO_ACTION: no teleop arm action in the episode`. An episode missing any sidecar key is
refused. Tolerances belong to the checker, not the recording. So every report records what
judged it: the tolerance values in force and their source, and the checking code's commit,
dirty state and a digest of its source (`checked_by`), so a verdict can be reproduced.

**Refitting** (`episode_fit`). Proposes the consistency tolerances from kept episodes of one
profile: in recording order, the worst value of the first 70 % times 1.5, rounded up (0.5 mm,
0.1 deg, 0.005 rad); the other 30 % are held out and judged with the proposal. Fewer than five
eligible episodes give no proposal. It prints the proposal (and writes it with `--out`); the
profile is edited by hand.

**Export** (`episode_export`): a fixed rate (30 Hz default), causal hold: every field is the
newest sample not later than the tick, with its age (never negative); nothing is interpolated.
A fixed grid beats against camera frames that are not exactly 1/rate apart and skips or
repeats some; `--tick-stream <camera>` puts the ticks on that camera's frame stamps instead,
so each of its frames is used exactly once (`export.json` `ticks` says which grid, and
`images` counts the frames used, unused and repeated per camera). `clutch_engaged`
false means hold: no new target; `action_valid` is false there, and `gripper_action_valid` is
false in cycles where teleop held no gripper command. `data.npz` + `export.json` + lossless PNG
images (32-bit depth as `.npy`): numpy and PIL only, nothing tied to a training framework.


## Operator feedback (vibration and the headset panel)

The operator, in the headset, knows the system's state without a screen.

**What runs.** The operator feedback node (`ranger_xarm_teleop_bridge feedback_node`, node
`operator_feedback`) starts with the stack (`feedback:=true`, the default of
`teleop_core.launch.py`), before arming. It only observes `/teleop/state` (every message) and
`/teleop/episode/state`, and publishes:

| Topic | Type | What |
|---|---|---|
| `/teleop/haptic` | `Haptic` | one vibration request per episode event (left controller) |
| `/teleop/operator/status` | `OperatorStatus` | the panel's content, 5 Hz: plain words for the hold, the limit and the last refusal, the last fault and the most severe recent limit latched, how fresh each source is |

Both are recorded (optional streams of the profiles): what the operator felt and saw. The
bridge (`bridge.launch.py feedback:=true`, the default) subscribes to these two topics only and
sends them to the page over its relay connection (`teleop_haptic`, `teleop_status`). It never
waits for a send while reading headset frames, drops what is older than its maximum age
instead of sending it late, and queues nothing while the link is down. With `feedback:=false`
nothing is subscribed or sent. If the feedback node stops, teleop goes on and the panel shows
NO DATA.

**Vibration** (left controller; distinct by count and length; the two you must not miss are
long and strong):

| Event | Pattern |
|---|---|
| mark (left trigger) | one tick, 40 ms |
| start | one buzz, 200 ms |
| stop | two short, 120 ms each |
| keep | one long, 500 ms |
| discard | three quick, 80 ms each |
| refused | two long and strong, 350 ms each |
| stopped by itself | an alarm: five short and strong, a pause, five again (2.4 s) |

An event is vibrated once (by the episode node's event counter): a repeated state message, a
reconnect or a lost message neither doubles nor invents one. A request older than 0.75 s
anywhere on the way is dropped, never played late.

**Panel.** A small panel about 10 cm above the left controller, turned to face you (or
head-locked at the lower left: `panel_anchor: head` in the parameter file below). It shows the recording state with the
episode and elapsed time, kept and discarded counts, armed, clutch, calibration, the
translation scale and active limit, the gripper, the hold or latched fault in plain words, and
the last refusal or self-stop. The recording state, the current hold or fault and the active
limit are large and bold; the rest is smaller. When the ROS clock runs below 0.6 of real time
(a slow simulator; `sim_slow_below`, `sim_slow_clear` in the parameter file) it adds
"Simulator slow: 0.4x real time", measured by the bridge from the statuses' ROS stamps against
their wall times. The left thumbstick click hides and shows it (remembered by the
page). It has no controls. When its newest status is older than 1 s it shows NO DATA instead.

**Tuning.** Patterns, priorities, rates and the panel's place and size are parameters in
`ranger_xarm_teleop_bridge/config/operator_feedback.yaml` (`feedback_params_file:=` on both
launches). The page shows its version (`client feedback-N`) in its log, next to the status pill
and in the panel corner: after an update, reload the page fully in the headset browser.

**Cost on the headset.** Record the same motion with the panel hidden, shown, and with
`feedback:=false`, then compare the link statistics of the sidecars:

```bash
tools/compare_link.py ~/teleop_data/<session> --group hidden=1-3 --group shown=4-6 --group off=7-9
```
