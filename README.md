# ranger_xarm_teleop

VR teleoperation and demonstration recording for the Ranger Mini V3 + xArm 6, driven from a
Meta Quest 3 through a WebXR page. One code path for Gazebo, Isaac Sim and hardware.
Depends on the robot packages in `ranger_xarm_ws` (built separately).

## Layout

| Path | What |
|---|---|
| `src/ranger_xarm_teleop_interfaces` | `InputStatus` / `HandStatus`, `TeleopState`, `ArmOwner` messages, `SetArmOwner` service |
| `src/ranger_xarm_teleop_bridge` | Bridge node (relay WebSocket → `/teleop/input/*`), JSONL replay node, pure parsing/conversion/analysis modules and their tests |
| `src/ranger_xarm_teleop_servo` | Teleop node (clutch-relative mapping, safety gate, damped least-squares solver, MoveIt Servo joint-jog commands), command mux node, and the pure modules behind them with their tests |
| `src/ranger_xarm_teleop_bringup` | MoveIt Servo configuration, the teleop trajectory controller, launch files for mock execution and a running simulator, simulation-only limit values |
| `third_party/vr-teleop-kit` | Vendored relay and WebXR web client (upstream commit `80589ee`, Apache-2.0, minimal patch listed in `VENDORED.md`) |
| `tools/` | No-ROS CLIs: `relay_setup.sh`, `relay.sh`, `record_relay.py`, `fake_quest.py`, `analyze_input.py`, `clip_recording.py`. With ROS: `move_arm.py`, `recover_arm.py`, `replay_measured.sh`, `measure_tracking.py` (`report` needs no ROS), `record_session.py` (`halts` needs no ROS). No ROS: `workspace_map.py` |
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
source that is not a real controller, gives zeros for its block.

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
# mock execution (ros2_control mock components, move_group, RViz), system clock:
ros2 launch ranger_xarm_teleop_bringup teleop_mock.launch.py
# or a simulator that is already running, simulation time:
ros2 launch ranger_xarm_gazebo sim.launch.py headless:=true          # terminal 1
ros2 launch ranger_xarm_teleop_bringup teleop_sim.launch.py          # terminal 2 (starts move_group too)
# then the input: the live bridge (see above) or a recording
ros2 run ranger_xarm_teleop_bridge replay_node --ros-args -p file:=<recording.jsonl> \
    -p wait_for_subscriber_s:=15.0 -p start_delay_s:=3.0
```

**Operating it** (right controller):

1. **Calibrate (A).** Point the controller along the direction that should mean "arm forward"
   and press A. The horizontal projection of the pointing axis becomes `xarm_link_base` +x; the
   result is published as the static transform `xarm_link_base` -> `quest_world` and is valid only
   for the XR session and reference-space reset count it was made under. After a recentre, a
   tracking recovery or a page reload, calibrate again.
2. **Arm (B).** Toggles arming. Arming asks the command mux for the arm; it is granted only with
   the arm stopped and no trajectory executing. Disarming (B again, or
   `ros2 service call /teleop/disarm std_srvs/srv/Trigger`) gives the arm back to MoveIt.
3. **Clutch (grip).** While the grip is held the tool follows the hand, relative to where both
   were when the grip closed. Release holds. Any gate condition (below) also releases, and the
   grip then has to be released and pressed again.

`ros2 topic echo /teleop/state` shows why the arm is not moving (`hold_reason`), which bound is
shaping the target or the joint command (`active_limit`), the owner of the arm, the calibration
in use, the measured input age, the measured tool pose, and the solver: `conditioning`,
`damped`, `solver_error_*`, `lead_*` and the commanded tool speed (next section).

### Mapping

Clutch-relative in a fixed frame: `p_now - p_clutch` and `q_now * inverse(q_clutch)` in
`quest_world`, rotated into `xarm_link_base` by the calibration heading, applied to the tool
pose measured at engagement. How the controller is held when the grip closes does not change
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
  (`damping_threshold`, `max_damping` 0.5). The arm never halts for a singularity; `damped`
  says when accuracy is reduced and `solver_error_*` by how much.
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
| self-collision, scene collision; a joint inside MoveIt Servo's own margin (a backstop) | MoveIt Servo slows (`active_limit` 4, `servo_status`); when it halts, the clutch is released (`hold_reason` 19) so the target cannot run on ahead of a standing arm |

**The limit values have no defaults.** Maximum offset, view cone, per-cycle limits, speed
caps and lead limits are required parameters; the teleop node exits if one is missing. For simulation,
`config/SIMULATION_ONLY_limits.yaml` supplies values. That file marks itself simulation-only:
the node then starts only if every ros2_control hardware plugin in the robot description is a
simulator, and `teleop_core.launch.py` refuses the file unless called with `simulation:=true`,
which only `teleop_mock.launch.py` and `teleop_sim.launch.py` do. An extra parameter file
(`teleop_params_file`) may not set a limit value, `motion_enabled` or the simulation-only flag. A robot needs its own limits
file, written by whoever is responsible for it, and its own launch path.

Clocks: MoveIt Servo, the mux, the teleop node and the controllers use simulation time under a
simulator (their stamps must agree with the joint states); the bridge and the replay node stay
on the system clock. Input age, the target cycle and every timeout use the monotonic clock of
the machine, so 100 ms is 100 ms of real time whatever the simulator does.

### Command mux

Two trajectory controllers claim the arm joints: MoveIt's, and `teleop_arm_controller`, which
MoveIt Servo feeds. ros2_control lets only one be active; an inactive one rejects action goals
and ignores its command topic. `arm_mux_node` reports the owner on `/teleop/mux/owner` and
switches with one strict `switch_controller` call, only when every arm joint has been still for
0.3 s (judged from an unbroken run of joint-state samples), no trajectory goal is active and
the clutch is released. A refused or failed switch leaves the owner as it was and says why.

### Measuring a run

```bash
ros2 service call /teleop/disarm std_srvs/srv/Trigger           # MoveIt must own the arm, or the goal is rejected
tools/move_arm.py [--controller xarm_xarm6_traj_controller]      # a posture with room to move (simulation only)
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
tools/record_session.py halts recordings/NAME.jsonl     # no ROS needed
tools/workspace_map.py recordings/NAME.jsonl            # or a URDF file; no ROS needed
```

`record` writes `recordings/NAME.jsonl` (git-ignored): teleop state, tool target, controller
pose, joint states, MoveIt Servo's status, the arm owner and the robot description. It only
listens. It refuses to start while the clutch is engaged, and after Ctrl-C it keeps recording
until the clutch is released (see the timing note below). `halts` prints, per halt, MoveIt
Servo's status, the hold reason, the joint angles, the condition number and its trend over the
second before, the nearest joint limit, the singularity the posture resembles (wrist, elbow,
shoulder) and what was being commanded; then counts by reason and the distribution of the
condition number over the engaged time. `workspace_map.py` reports the condition number along
tool motions from a posture: how far the tool gets in each direction before each threshold.

After a collision halt MoveIt Servo refuses motion in every direction: the arm has to be moved
out by MoveIt.

```bash
tools/recover_arm.py          # disarm, wait until MoveIt owns the arm, plan to the ready posture
```

When the arm is in contact, MoveIt refuses to plan from that start state
(`start state in collision`) and `recover_arm.py` moves nothing.

(simulation only, like `tools/move_arm.py`, which now prints why a goal was rejected).

Timing note: when a new ROS participant joins the graph (a command-line tool, a node being
started), DDS endpoint matching can delay messages or block a publish call for 0.1 to 0.3 s.
Under a held clutch that is a release on stale data (`hold_reason` 6 or 17), by design. The
teleop node logs `late cycle: ...` when its own cycle was delayed. Start a replay with
`wait_for_subscriber_s` and a few seconds of `start_delay_s`, as above.

