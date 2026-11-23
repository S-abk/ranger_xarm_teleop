# Hardware procedure: the first sessions on the real arm

For one person, standing behind the robot with the arm's and the base's emergency stops in
reach (by foot or the back of a hand), both controllers strapped to the wrists, the headset
cabled to the robot computer. The general robot with a bare gripper, nothing held. The base
stays off (`enable_base:=false`); base driving is not built.

The ladder has five stages. Each page says what must be true before you start, what to run,
what you should see, when to stop and how to back out. **Never take a step whose check has not
passed.** Every check below is either a run of the read-only check tool (`hw_check`) or a
behaviour the stack tests show on the hardware-shaped mock (named in brackets).

Rules for every stage:

- **Everything joins the ROS graph before arming.** A process that joins can stall the stack
  for a third of a second, up to one second. Start the bring-up, the teleop launch, the bridge
  and any `hw_check` run first; arm only then. Never start a tool while armed.
- **After any emergency stop:** if teleop was armed, press B to disarm before you release the
  stop. Then shut down the teleop launch, the bridge and the bring-up, release the stop, and
  start again from stage 2's checks. See "After an emergency stop" at the end for why.
- **Never move the arm by hand while the bring-up is running.**
- The session launcher (`teleop_session`) is not used on hardware; it starts simulation
  profiles only. Every command here is typed by you, in its own terminal.
- Robot address, camera serials and your limits file are yours: they are arguments or files
  outside any checkout, never committed. Below they are written `<robot_ip>`, `<serial>` and
  `<limits>` (the path of your limits file), and `<planes>` (the path of your keep-out planes
  file, stage 4's precondition), and `<robot_ws>`: the path of the robot workspace's folder,
  the one holding `src/xarm_ros2`. Type the path itself: a shell variable set in another
  terminal is empty in a new one.

**Building the teleop checkout** on the robot computer (once, and after every update): in a
terminal where the robot workspace's `install/setup.bash` is sourced, so that the teleop
packages build against the robot's packages, install the system packages the teleop packages
declare, then build:

```bash
rosdep install --from-paths src --ignore-src -y   # e.g. python3-websockets for the bridge
colcon build --symlink-install
```

Every terminal: source the robot workspace's `install/setup.bash`, then the teleop
checkout's `install/setup.bash`, and run the commands from the teleop checkout. The relay
needs `tools/relay_setup.sh` once per machine, which needs `python3-venv`
(`sudo apt install python3-venv`).

---

## Values you supply

The hardware path refuses to start until every value below is in your limits file. Copy the
template outside any checkout, fill in every `PENDING`, delete its first line (the marker):

```bash
cp install/ranger_xarm_teleop_bringup/share/ranger_xarm_teleop_bringup/config/hardware_limits_TEMPLATE.yaml ~/robot_site/limits.yaml
```

The template is refused, and so is a copy with the marker line, a `PENDING`, a missing value or
one that is not a positive number, and the simulation-only file (`test_refusals.py`,
`test_hardware_limits.py`). The suggestions below are for a **first session** only; they are
suggestions, not defaults, and appear nowhere the hardware path reads them.

| Value | What it bounds | First-session suggestion |
|---|---|---|
| `linear_speed_cap_m_s` | tool linear speed (joint command) | 20 % of the simulation value: 0.05 m/s |
| `angular_speed_cap_rad_s` | tool angular speed | 20 % of the simulation value: 0.2 rad/s |
| `joint_speed_cap_rad_s` | every arm joint's commanded speed (G-18) | choose it first, well under the arm's rated joint speed (see below) |
| `joint_acceleration_limit_rad_s2` | every arm joint's commanded acceleration, also the stop after a release, a gate fault or a disarm (G-18) | then this: large enough that the joint speed cap divided by it is at most 0.2 s, and well under the arm's rated joint acceleration (see below) |
| `max_lead_translation_m` | target ahead of the measured tool | at least cap x lag x 1.2. Lag is unknown until stage 4 (simulation: 0.11 to 0.17 s), so start at 0.015 m and recompute from stage 4's `target lag` |
| `max_lead_rotation_deg` | the same in rotation | angular cap x lag x 1.2 in degrees; start at 3 |
| `max_offset_translation_m` | target from the pose at clutch engagement | 0.10 m (simulation 0.60) |
| `max_offset_rotation_deg` | the same in rotation | 20 (simulation 60) |
| `max_translation_scale` | largest scale the operator can pick | 1.0. Below 1.0, also set `translation_scale_fine` and `translation_scale_coarse` (their default is 1.0), or the file is refused |
| `step_translation_limit_m` | target step per 20 ms cycle (larger releases the clutch) | 0.020 to 0.025 (1.0 to 1.25 m/s of hand speed; see below) |
| `step_rotation_limit_deg` | the same in rotation | a third of the simulation value: 3 |
| `view_cone_half_angle_deg` | controller angle off the headset's forward axis | 50 (simulation 65; narrower is stricter) |
| `link_keepout_margin_m` | arm links from the robot's fixed structure | 0.025 (simulation 0.025); see below |
| `tool_pose_timeout_s` | age of the measured tool pose | from stage 2's `joint_states largest gap`: at least 3 x that gap and at most 0.1 s. If the gap needs more than 0.1 s, stop |

The speed cap and the lead limit bound each other: the arm cannot go faster than lead / lag.
Raise them together, deliberately, and only after a stage has passed at the lower values.

**The step limit is a glitch detector, not a speed limit.** It releases the clutch when the
hand's target moves more than this in one 20 ms cycle, which catches a tracking jump. The
arm's own speed is bounded by the speed caps, the lead limit and the joint acceleration limit
(G-18), not by it. At 0.010 (0.5 m/s of hand speed) ordinary hand motion tripped it in the
first real session (0.51 to 0.72 m/s logged), so start at 0.020 to 0.025.

**The keep-out margin drops a pair it can never keep.** A pair of an arm link and a fixed
link that is closer than the margin in every posture is left out of the keep-out altogether,
not kept clear. On the hardware path the teleop node then refuses to start and names the pair
and its clearance range; in simulation it logs a WARN line. Choose a margin below the largest
clearance the refusal names. If that clearance is below the smallest margin you can accept
(MoveIt Servo's self-collision threshold and move_group's planning padding, plus the overshoot
measured on this arm), stop: the robot description's geometry needs a look first.

**A kept pair can still hold the arm back.** The model covers links with spheres, which read a
pair closer than its surfaces are. Where a pair stays within that error of the margin across
its whole range, as `xarm_link2` against the arm base does, the model covers both sides
finer (15 mm cells; the start-up log names them). That pair then reads 47 to 54 mm over joint
1 and joint 2 within 1.6 rad, where the true surfaces are about 70 mm apart (31 to 48 mm with
the coarser cells, which at a margin of 0.04 held joints 1 and 2 still at the stage 4 posture
and left the tool pivoting about the wrist on the first stage 4 run). If a kept pair is still
inside the margin over more than a quarter of the joint range the solver allows, the hardware
path refuses the start, naming the pair and the largest margin that frees it: on this arm 0.05
is refused that way (free up to about 48 mm), 0.045 is not. At the stage 4 posture and on
10 cm strokes from it, no pair is inside a margin up to about 0.042 in open space; pulling the
tool back over the base brings link 2 within the margin of the structure behind the arm,
which is real structure and is meant to hold it. 0.025 is the value that has worked on the
arm. The node logs at every start the largest margin no pair is inside around the posture it
starts in. Whatever margin you choose, check it in the preview at the stage 4 posture (3b')
before arming.

**The joint speed cap and the joint acceleration limit (G-18).** The arm's streamed servo mode
(the firmware's mode 1, which the ros2_control driver uses) executes each point it receives at
the fastest speed and acceleration, with no buffer and no smoothing of its own: the smoothness
of the command is all there is. So teleop limits its joint command itself. Every cycle the
change of all joints together is scaled so that no joint changes faster than the acceleration
limit, and no joint is commanded faster than the joint speed cap. A release, a gate fault or a
disarm ramps the command to zero at the limit, keeping the tool's direction.

- **Choose the joint speed cap first**, well under the arm's rated joint speed (180 degrees/s,
  3.14 rad/s, in the vendor's manual).
- **Then the acceleration limit:** large enough that the stop from the cap takes at most 0.2 s
  (cap / limit <= 0.2 s; the stack refuses to start otherwise, with one line naming the value to
  change), and well under the arm's rated joint acceleration (the vendor's manual gives a range
  up to 1145 degrees/s^2, 20 rad/s^2).
- **What it costs:**
  - a stop after a release adds at most v x q / (2 a) of tool travel to the points already on
    their way, where v is the tool speed, q the fastest joint's speed and a the limit;
  - starts are softer, so the arm lags the hand a little longer at the start of a stroke, and
    the lead clamp acts more often;
  - recompute the lead limits from stage 4's `target lag` as above.
- MoveIt's `joint_limits.yaml` value is for planned motion (stage 3's check uses it); it is not
  this limit.

---

## Stage 1: on the laptop

**Before you start:** the teleop checkout at the commit you will run on the robot; the robot
workspace built.

**Commands**

1. Build and run the quick tests (the `unit` line of the project's checks).
2. Rehearse stages 2 to 4 on the hardware-shaped mock: it has the real bring-up's controllers,
   rates and gripper action, on mock hardware.

   ```bash
   ros2 launch ranger_xarm_teleop_hardware hwmock.launch.py stage:=check        # terminal 1
   ros2 launch ranger_xarm_teleop_hardware teleop_hardware.launch.py stage:=check limits_file:=<limits>   # terminal 2
   ros2 run ranger_xarm_teleop_check hw_check idle --limits <limits>              # terminal 3
   ```

   Then the same with `stage:=execution` (and `motion_enabled:=true` on the teleop launch).
   The mock refuses to start if a controller manager is already answering on the domain, so
   it can never run beside a real bring-up.
3. On the robot computer, apply the workspace patch that adds the teleop controller to the
   real bring-up (P1: `teleop_arm_controller` in `real_ros2_controllers.yaml`, spawned
   inactive by `real_execution.launch.py`), and rebuild the workspace there.

4. Also on the robot computer, re-enable the collision check of `xarm_link2` and `xarm_link4` in
   the robot workspace's SRDF (P3). The SRDF marks the pair "Never" in collision, but within
   the joint limits the two links collide (joint 3 above about 8.6 degrees with joint 4 near
   130 to 186 degrees, modulo 360); while the pair is disabled, neither MoveIt nor MoveIt
   Servo checks it, and the teleop keep-out does not cover arm-against-arm pairs. The other
   four disabled arm pairs keep at least 11 mm apart over the whole joint range.

**What you should see:** the mock's first log line says `HARDWARE-SHAPED MOCK`; the teleop
launch prints `ATTACH PREFLIGHT OK`; the check tool's first line says it only subscribes.

**Stop if:** any of it differs. Nothing here touches the robot.

---

## Stage 2: on the robot, nothing moving

**Before you start**

- Stage 1 done; your limits file filled in (the attach refuses without it, even with motion
  disabled); your keep-out planes file written (stage 4's precondition), so that this stage's
  preview can try each plane.
- **At the robot computer's own desktop.** Open every terminal of this stage and the next ones
  in the robot computer's graphical session (at its screen, or a remote desktop), not over a
  plain SSH login: the bring-up starts RViz, which needs a display.
- The teleop checkout built on the robot computer with the robot workspace sourced (Building
  the teleop checkout, above), and the relay's environment made once with
  `tools/relay_setup.sh`, which needs `python3-venv` (`sudo apt install python3-venv`).
- **Network:** a direct Ethernet cable from the robot computer to the arm's control box, no
  switch in between.
- **Driver version (gate).** The installed xarm_ros2 must contain the 2025-03-18 change that
  makes the ros2_control interface read joint state from the arm every control cycle; before
  it, the state was stair-stepped at 5 Hz under `report_type:=normal`. Check:

  ```bash
  git -C <robot_ws>/src/xarm_ros2 merge-base --is-ancestor cc44f9f HEAD && echo contains
  ```

  If git does not know the commit (a shallow clone or a rebased fork), check the source
  instead: `grep -n 20250318 <robot_ws>/src/xarm_ros2/xarm_controller/src/hardware/uf_robot_system_hardware.cpp`
  must print a line. `hw_check idle --limits <limits> --workspace <robot_ws>` judges the same rule.
  **If neither shows it, stop:** update the driver first.
- CPU governor at `performance` on the robot computer (the idle check reports it), for
  example `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`.
- **In the vendor's software, confirm and write down** (next to this stage's report):
  - the firmware version (2.3.0 or later is wanted: it reports fault details and the safety
    boundary's status);
  - the mounting direction;
  - the tool payload and its centre of mass (wrong values give false collision alarms);
  - collision detection on, sensitivity 3 or more;
  - collision rebound off;
  - self-collision detection on;
  - the safety boundary set;
  - reduced mode set.

  None of these screens a streamed command before the arm executes it: they act on the arm's
  own measurements, after the fact. **The safety boundary and reduced mode are not a backstop
  for teleop:** in streaming mode (mode 1) nothing shows that they act, and in the first session
  on the arm the boundary did not even stop a planned motion that crossed it (stage 3, 3d). The
  stack's keep-out planes (`keep_out_planes`) are the backstop for where the tool and the wrist
  may go (stage 4's precondition). Set the boundary and reduced mode anyway.
- Emergency stops in reach; nobody else near the arm.

**Commands**

```bash
ros2 launch ranger_xarm_moveit_config real_hardware_check.launch.py robot_ip:=<robot_ip>   # terminal 1
ros2 launch ranger_xarm_teleop_hardware teleop_hardware.launch.py stage:=check limits_file:=<limits> teleop_params_file:=<planes>   # terminal 2
tools/relay.sh                                                                             # terminal 3
adb reverse tcp:8443 tcp:8443
ros2 launch ranger_xarm_teleop_bridge bridge.launch.py record_path:=recordings/hw_stage2.jsonl   # terminal 4
ros2 run ranger_xarm_teleop_check hw_check idle --limits <limits> --workspace <robot_ws> --expect-active joint_state_broadcaster   # terminal 5
```

Note the arm's firmware version from terminal 1's start-up log (the tool cannot ask the arm
for it); write it next to the check report.

**What you should see**

- Terminal 2: `ATTACH PREFLIGHT OK (check stage)` and `motion disabled`. The stack starts
  nothing in the bring-up and changes none of its controllers [`test_hwmock_check_stage`:
  the attach leaves the bring-up as it found it].
- The check tool: `PASS` on `joint_states rate`, `joint_states largest gap`,
  `velocity_control` (position control), `no /clock publisher`, `stamp offset`, the three
  `teleop cycle` rules, `overruns and read failures`, `xarm_ros2 joint-state change` and
  `cpu governor`, and `arm state (xarm_msgs/RobotMsg)`: err 0, mode 1 (servo), state 2 or lower
  in every message, with the brake and enable flags shown. The check finds the driver's state
  topic by its type (on the real driver `/xarm_xarm/robot_states`) and names it; `INFO not
  available` where nothing publishes that type, and `NOT JUDGED`, naming each, where more than
  one publisher does. `joint_states effort` is `INFO absent`: this driver does
  not copy effort into the state interface. It writes one report file under `~/teleop_data/hardware_checks/`.
- In the bring-up's RViz, add a PoseStamped display on `/teleop/preview/target_pose`. Point
  the right controller where "arm forward" should be and press A (calibrate). Hold the grip
  and move your hand forward slowly: the preview pose moves forward from the tool (in this
  stage the flange, `xarm_link_eef`: the check stage's description has no gripper), and stops
  when you release [`test_hwmock_check_stage`: the preview follows a stroke to a keep-out
  plane and is never a command]. The preview is shown only while motion is disabled; why it is
  not shown is in `ros2 topic echo --field preview_reason /teleop/state`.
- **What would act on the preview.** The panel's limit line ends in "(preview)" and says what
  would hold the arm back if it were armed: "keep-out zone" or "reach range" while a plane or
  the offset limit holds the preview; "near robot body" or "joint at its end" when, from the
  posture the arm is in, the solver would remove the motion toward that target (the link
  keep-out, a joint position limit). Nothing is commanded. Terminal 2 logs each change; for the
  link keep-out it names the nearest pair and its clearance:
  `preview: LINK_KEEPOUT would act (nearest pair ..., clearance ... mm, margin ... mm)`
  [`test_hwmock_preview_limits`: at a margin of 0.04 the preview at the stage 4 posture reports
  the link keep-out and names the pair; at 0.025 nothing]. The solver is checked from the
  posture the arm is in now, not along the way to the target. "near robot body (preview)" or
  "joint at its end (preview)" while your hand moves through open space means the arm would
  not follow from this posture: note the pair terminal 2 names. This is not yet the stage 4
  posture; the check that counts for stage 4 is 3b'.
- **Each keep-out plane, in the preview** (stage 4 needs it). The preview never goes farther
  from the tool than `max_offset_translation_m` (0.10 m at the first-session values), and the
  arm does not move in this stage, so for this trial only: stop terminal 2 and start it again
  with a copy of your limits file whose `max_offset_translation_m` is larger than the distance
  from the tool to the farthest plane (motion is disabled: the preview commands nothing). Hold
  the grip and move your hand slowly past each plane in turn: the preview pose stops at the
  plane, or a little before it when the wrist point reaches it first, and stays there while
  your hand goes on [`test_hwmock_check_stage`: the preview stops at a keep-out plane;
  `test_motion_disabled`: the preview is bounded by every plane, at the tool point and the
  wrist point]. Then stop terminal 2 and start it again with your own limits file before you
  go on.
- Press B: the panel shows "Motion disabled", and terminal 2 logs `arm refused: teleop motion
  is disabled ...` [the arming refusal test].

**Stop if**

- the driver gate, `velocity_control`, the clock rules or `stale`/`read failures` fail;
- **150 Hz is not held** (`joint_states rate` fails, or the period distribution in the report
  shows periods above 2 x 6.7 ms, or controller-manager overrun messages are counted): set
  the governor to performance, close everything else on the robot computer, check that the
  cable goes straight to the control box, and run the check again. If it still fails, stop
  and keep the report: the controller manager reads the arm over the network every cycle, and
  a cycle that overruns delays every joint state;
- the preview moves in a direction other than your hand's (calibrate again; if it is still
  wrong, stop);
- a preview crosses a keep-out plane, or stops short of it with the offset limit and the
  wrist point both clear of it: the plane is not where you meant it. Correct the planes file
  and try that plane again before stage 3;
- anything moves at all.

Then set `tool_pose_timeout_s` from this run's largest gap (Values you supply) before stage 4.

**How to back out:** Ctrl-C terminal 5, 2, 4, then 1. Nothing was commanded.

---

## Stage 3: MoveIt motion through the owner switch, nothing armed

**Before you start:** stage 2 passed and its report kept; P1 applied and built on the robot
computer; P3 applied and built on the robot computer (stages 3 and 4 need it); the arm in a
posture with room around it; emergency stops in reach. Shut stage 2 down first.

**Commands**

```bash
ros2 launch ranger_xarm_moveit_config real_execution.launch.py robot_ip:=<robot_ip>        # terminal 1 (RViz with motion planning)
ros2 launch ranger_xarm_teleop_hardware teleop_hardware.launch.py stage:=execution motion_enabled:=true limits_file:=<limits> teleop_params_file:=<planes>   # terminal 2
tools/relay.sh; adb reverse tcp:8443 tcp:8443                                              # terminal 3
ros2 launch ranger_xarm_teleop_bridge bridge.launch.py record_path:=recordings/hw_stage3.jsonl   # terminal 4
ros2 run ranger_xarm_teleop_check hw_check idle --limits <limits> --workspace <robot_ws> --expect-active joint_state_broadcaster,xarm6_traj_controller --expect-inactive teleop_arm_controller   # terminal 5
```

`motion_enabled:=true` here is the deliberate step that lets the arm be switched to teleop at
all; with it false the mux refuses every switch [`test_hwmock_execution`: motion disabled never
switches to teleop]. Nobody arms in this stage.

**What you should see before anything moves:** terminal 2 prints `ATTACH PREFLIGHT OK
(execution stage)`; the idle check in terminal 5 gives the same `PASS` results as in stage 2,
plus `controller xarm6_traj_controller active` and `controller teleop_arm_controller inactive`.
**Stop if** any of them fails: nothing has moved yet.

**3a. A small planned motion.** In terminal 5 start

```bash
ros2 run ranger_xarm_teleop_check hw_check motion --seconds 40 --limits <limits> --joint-limits <robot_ws>/src/ranger_xarm_moveit_config/config/joint_limits.yaml   # terminal 5
```

then in RViz plan a small motion (one joint, about 10 degrees, velocity scaling 0.1), look at
the planned path, and execute it.

- You should see: the arm moves along the path RViz showed, about 10 degrees on that one
  joint, slowly, and stops at its goal; when the check ends it gives `PASS` on
  `stale repeats while moving` and `wrist oscillation`.
- **Stop if:** the arm moves anywhere the plan did not show, or does not stop at its goal
  (emergency stop); `stale repeats while moving` fails (the driver is not reading the arm every
  cycle: stop the stage); `wrist oscillation` fails. If `peak joint acceleration` fails (the
  planned motion against MoveIt's `joint_limits.yaml`), keep the report and plan the next
  motion slower.

**3b. To the posture for stage 4**, only after 3a's check has passed. Plan the arm to a
posture with the wrist bent (joint 5 well away from zero; the tests use 0, -0.6, -0.7, 0, 1.3,
0 rad), velocity scaling 0.1, with a new run of 3a's `hw_check motion` command, and execute it.

- You should see: the arm moves along the planned path and stops there; the check passes as in
  3a.
- **Stop if:** as in 3a.

**3b'. The preview at the stage 4 posture.** Stop terminal 2 and start it again without
`motion_enabled:=true` (motion disabled: the preview exists only then, and nothing can be
armed):

```bash
ros2 launch ranger_xarm_teleop_hardware teleop_hardware.launch.py stage:=execution limits_file:=<limits> teleop_params_file:=<planes>   # terminal 2, motion disabled
```

Calibrate (A), hold the grip and move your hand slowly about 10 cm forward and back, left and
right, up and down, releasing between strokes, as in stage 2.

- You should see: the preview follows each stroke; the panel shows no limit line, or only
  "keep-out zone (preview)" or "reach range (preview)" where a plane or the offset limit holds
  the preview [`test_hwmock_preview_limits`].
- **Stop if:** "near robot body (preview)" or "joint at its end (preview)" shows while your hand
  moves through open space. The arm would not follow there: at a margin of 0.04 this is what
  held joints 1 and 2 still in the first stage 4 run. Do not arm. Note the pair terminal 2
  names and its clearance; choose another posture (plan it as in 3b) or a smaller margin
  (Values you supply), and repeat this step. **Stop also if** terminal 2 logs `preview: the
  solver check failed` or keeps logging `not ready`: the check did not run, and no limit line
  then proves nothing.

Then stop terminal 2 and start it again with `motion_enabled:=true`, as at the start of this
stage, before 3c.

**3c. The switch to teleop and back.**

```bash
ros2 service call /teleop/mux/set_owner ranger_xarm_teleop_interfaces/srv/SetArmOwner "{owner: 2}"   # to teleop
ros2 service call /teleop/mux/set_owner ranger_xarm_teleop_interfaces/srv/SetArmOwner "{owner: 1}"   # back to MoveIt
```

- What the arm does at the switch: nothing. The switch is one strict controller switch, made
  only with the arm still and no goal running. The teleop controller starts from the last
  position the outgoing controller commanded and holds it; the arm's driver sends a new
  position only when the command changes [`test_hwmock_execution`: the switch to teleop and
  back moves nothing].
- You should see: `success: true` both times, and no motion at all.
- **Stop if:** anything moves (press the emergency stop), or a switch is refused for a reason
  other than motion.

**3d. The safety boundary: a recorded finding, not a step.** In the first session on the arm,
a planned motion (velocity scaling 0.1) that crossed the boundary set in the vendor's software
by a few centimetres was **not stopped**: the arm went through it. The boundary is therefore
not relied on, and crossing it is no longer part of this ladder. Two set-up checks, written
down next to this stage's report:

1. **The boundary is enabled and saved** in the vendor's software, not only entered on its
   screen, and is still set after the control box is restarted.
2. **Which point the firmware treats as the tool point** for the boundary: the flange, or a
   tool point set in its TCP offset. Checked at the flange, the boundary lets a tool tip cross
   it by the tool's length.

Neither check makes the boundary a backstop for teleop (stage 2); the keep-out planes are
(stage 4's precondition).

**How to back out:** switch back to MoveIt (`{owner: 1}`) if teleop owns the arm; Ctrl-C
terminal 5, 2, 4, then 1.

---

## Stage 4: teleop armed at very low limits

**Before you start:**

- Stage 3 passed; your limits file at the first-session values; the arm at the bent-wrist
  posture from stage 3b (arming is refused near the wrist singularity); emergency stops in
  reach. Same terminals 1 to 4 as stage 3.
- **Keep-out planes set for the working space.** The firmware's safety boundary and reduced
  mode are not a backstop in streaming mode (stage 2, stage 3's 3d); the stack's
  `keep_out_planes` are. Each plane is a half-space in `xarm_link_base` that the tool point and
  the wrist point may not leave: four numbers `nx ny nz d` per plane, allowed where
  `nx*x + ny*y + nz*z >= d` (metres, a unit normal). Put a plane on every side where the arm
  could reach something the planning scene does not hold (the table, the base's top, where
  people stand) in your planes file, and pass it to the teleop launch (terminal 2 of stages 2
  to 4) as `teleop_params_file:=<planes>`.
  - The wrist point is bounded only when `wrist_frame` names it: a frame fixed to the tool
    (on this arm the flange, `xarm_link6`). On the hardware path the teleop node refuses to
    start with keep-out planes and no `wrist_frame`, in one line naming it; its start-up line
    says which points the planes bound.
  - Write every number with a decimal point (`1.0`, not `1`): the list is a list of decimals.
  - The format, with example numbers only:

  ```yaml
  /**:
    ros__parameters:
      keep_out_planes: [0.0, 0.0, 1.0, 0.05,    # stay above z = 0.05 m
                        -1.0, 0.0, 0.0, -0.60]  # stay at x <= 0.60 m
      wrist_frame: xarm_link6                   # the flange: kept inside every plane too
  ```

- **Each plane tried in stage 2's preview and passed** (stage 2, "Each keep-out plane, in the
  preview"), with the same planes file.
- **The stage 4 posture inside every plane**, tool point and flange on the allowed side;
  otherwise the first grip pushes the target back onto the plane.
- **3b' passed at this posture with the `link_keepout_margin_m` you arm with:** no "near robot
  body" or "joint at its end" in the preview through open space.

**Commands:** before arming, start the check (everything joins before arming):

```bash
ros2 run ranger_xarm_teleop_check hw_check teleop --seconds 120 --limits <limits> --joint-limits <robot_ws>/src/ranger_xarm_moveit_config/config/joint_limits.yaml   # terminal 5
```

Then, in the headset: point forward and press A (calibrate), press B (arm). **Arming moves
nothing:** the panel shows armed and the arm switches to teleop standing still (stop if
anything moves). Hold the grip and move your hand **5 cm, slowly**, in free space, then
release. Read terminal 5's result before the next stroke: it prints when its 120 s end, so
disarm (B), let it finish, check it as below, and start a new run before arming again. Then
one more stroke the other way.

**What you should see:** the tool follows your hand at no more than the speed cap, and after you
release it stops within about a centimetre and within 0.2 s (the command ramps to zero at the
acceleration limit; the arm also finishes the points already on their way); the target in RViz;
the check gives `PASS` on `stale repeats while moving`, `target lead (largest, translation)`,
`wrist oscillation`, `commanded joint acceleration (teleop joint command)` (every step of the
teleop node's own joint command within your acceleration limit, where G-18 is enforced; a step
straight to zero only as an immediate stop the node announced: teleop lost the arm, or MoveIt
Servo halted), `commanded joint acceleration (MoveIt Servo output)` (every point of
MoveIt Servo's output within twice your acceleration limit, plus 5 % for Servo's point timing:
one Servo skip can join two command steps) and `stop time after a release` (at most 0.2 s plus
one cycle), reports `joint-state acceleration (20 ms, INFO)` (information only: differentiated
encoder data, with MoveIt's value for reference), `Servo output above 1.1 x limit without a
skip (INFO)` (the share of points above 1.1 times your limit that no detected Servo skip
explains, and how many skips there were; keep these numbers: they decide whether the rule is
tightened), `target lag`, `link keep-out shaping the command (INFO)` (the share of engaged
cycles in which the link keep-out removed motion, how close to the margin, and the pair) and
`target lead (rotation, INFO)`. Keep these two as well: their values in strokes that track set
the threshold of a rule on them. In a stroke through open space the first should read 0 %, and
the second stay near 0 when your hand does not turn
[`test_hwmock_execution`: one short armed stroke with the teleop check;
`test_hwmock_preview_limits`: at a margin of 0.04 the keep-out line reads most engaged cycles,
names the pair, and the rotation lead rises although the stroke asks for no rotation, while
`target lead (largest, translation)` passes; at 0.025 the keep-out line reads 0 % and the
rotation lead stays near 0].

**Stop if** (release the grip; press the emergency stop if it does not stop):

- the arm moves without the grip held, or in a direction other than your hand's;
- the tool does not follow your hand: it turns although your hand does not, or joints stay
  still that the stroke needs. The judged rules can all pass while this happens. Release,
  disarm, keep the report (`link keep-out shaping the command` and `target lead (rotation)` show
  it), and go back to 3b';
- it keeps moving more than about 2 cm after you release;
- the wrist oscillates (the solver's limit cycle near the wrist singularity: release, then
  disarm, and plan the arm away from the singular posture through MoveIt);
- `commanded joint acceleration (teleop joint command)`, `commanded joint acceleration (MoveIt
  Servo output)` or `stop time after a release` fails: keep the report and stop the stage;
- the arm reports a collision error: payload, mounting or sensitivity may be set wrong, but
  uneven point timing can also cause it, because the firmware estimates acceleration from the
  points it receives (the check reports Servo's point spacing). Keep the report and stop.

Afterwards, recompute the lead limits from the measured lag (Values you supply) before more
strokes.

**If the arm leaves teleop by itself** (an arm error or an emergency stop makes the driver
switch every controller off): teleop disarms and stays disarmed (G-17), and the mux deactivates
a teleop controller it did not activate [`test_hwmock_execution`: the three G-17 tests].
Follow "After an emergency stop".

**How to back out:** release the grip, press B (disarm: the arm returns to MoveIt standing
still; stop if anything moves), Ctrl-C terminal 5, 2, 4, then 1.

---

## Stage 5: gripper, then recording

**Gripper teleop on hardware is not built.** The real gripper is a `GripperCommand` action, not
a ros2_control joint, and its reported state is partly assumed; the teleop launch refuses
`gripper_enabled:=true`. The gripper stays with MoveIt (RViz) until the action adapter has been
built and reviewed.

**Recording**, after stage 4 has passed at the values you will record with:

```bash
ros2 launch ranger_xarm_sensors robot.launch.py robot_ip:=<robot_ip> enable_base:=false enable_ouster:=false enable_rplidar:=false gantry_d455_serial_no:=<serial> wrist_d435_serial_no:=<serial> color_profile:=640x480x30 depth_profile:=640x480x30   # terminal 1
ros2 launch ranger_xarm_teleop_hardware teleop_hardware.launch.py stage:=execution motion_enabled:=true record:=true limits_file:=<limits> teleop_params_file:=<planes>   # terminal 2
ros2 run ranger_xarm_teleop_check hw_check cameras --profile install/ranger_xarm_teleop_record/share/ranger_xarm_teleop_record/profiles/hardware.yaml   # terminal 5, recorder up, before arming
```

The bring-up's cameras default to 15 fps; the recording profile expects 30 fps at 640x480,
hence the two profile arguments.

Before arming, also run the idle check again on this bring-up (cameras and recorder add load
to the robot computer): `hw_check idle --limits <limits> --workspace <robot_ws>` in
terminal 5.

**You should see:** `PASS` on every camera's rate, gap and size rules, and the idle check as in
stage 3 (150 Hz held). Then arm and teleoperate exactly as in stage 4: its expectations and
stop conditions apply to every stroke.

**Stop if:** a camera does not hold 30 fps or loses frames (USB bandwidth): keep the report;
do not record with a profile it does not meet. If the idle check no longer holds 150 Hz with
the cameras running, stop: do not arm.

---

## After an emergency stop

1. If teleop was armed, press B to disarm before releasing the stop.
2. Shut down the teleop launch, the bridge and the bring-up.
3. Release the stop. Do not move the arm by hand while anything of the bring-up runs.
4. Start again from stage 2's checks.

Why, from the driver's source (xarm_ros2, `uf_robot_system_hardware.cpp`) and the controller
manager's (ros2_control 4.48.1; the switch path is unchanged in 4.50.0, and the trajectory
controller's start from the last commanded position unchanged from 4.42.2 to 4.43.1, both
read in the source):

- On any arm error, an emergency stop included, the driver deactivates every controller. When
  the arm is ready again it asks, once, for every listed controller to be activated, best
  effort. With both arm controllers loaded (P1), the controller manager refuses that request
  as a whole, because two controllers in it claim the same joints; the joint state
  broadcaster is in the same request and stays off too [`test_hwmock_execution`: the full
  reactivation is refused]. The bring-up then has no joint states and no trajectory
  controller until it is restarted.
- The driver does not put the firmware back into servo mode after an error; until it is,
  nothing is commanded.
- Other users report the arm jumping back toward an old setpoint after control was handed
  back from another mode or after the arm was moved outside the controller (xarm_ros2 issues
  83 and 117). A fresh bring-up starts its controllers from the measured position.
- The arm sags slightly on its brakes after a stop, and the last joint values on ROS may be
  stale, so the joint state the stack holds is not where the arm is: one more reason for the
  restart.

G-17 is the backstop: armed teleop that loses the arm without asking disarms and never asks for
it again. The instruction is still to disarm before releasing the stop.

## A planned motion outlives move_group

If move_group dies during a planned motion, the trajectory controller still holds the whole
trajectory and finishes it [`test_hwmock_execution`: a planned motion when move_group dies].
Keep planned motions short in the first sessions, and be ready to press the emergency stop.
