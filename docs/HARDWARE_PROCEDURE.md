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
  `<limits>` (the path of your limits file).

Every terminal: source the robot workspace's `install/setup.bash`, then the teleop
checkout's `install/setup.bash`, and run the commands from the teleop checkout. The relay
needs `tools/relay_setup.sh` once per machine.

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
| `max_lead_translation_m` | target ahead of the measured tool | at least cap x lag x 1.2. Lag is unknown until stage 4 (simulation: 0.11 to 0.17 s), so start at 0.015 m and recompute from stage 4's `target lag` |
| `max_lead_rotation_deg` | the same in rotation | angular cap x lag x 1.2 in degrees; start at 3 |
| `max_offset_translation_m` | target from the pose at clutch engagement | 0.10 m (simulation 0.60) |
| `max_offset_rotation_deg` | the same in rotation | 20 (simulation 60) |
| `max_translation_scale` | largest scale the operator can pick | 1.0, and do not set a coarse scale above it |
| `step_translation_limit_m` | target step per 20 ms cycle (larger releases the clutch) | a third of the simulation value: 0.010 |
| `step_rotation_limit_deg` | the same in rotation | a third of the simulation value: 3 |
| `view_cone_half_angle_deg` | controller angle off the headset's forward axis | 50 (simulation 65; narrower is stricter) |
| `link_keepout_margin_m` | arm links from the robot's fixed structure | twice the simulation value: 0.05 |
| `tool_pose_timeout_s` | age of the measured tool pose | from stage 2's `joint_states largest gap`: at least 3 x that gap and at most 0.1 s. If the gap needs more than 0.1 s, stop |

The speed cap and the lead limit bound each other: the arm cannot go faster than lead / lag.
Raise them together, deliberately, and only after a stage has passed at the lower values.

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

**What you should see:** the mock's first log line says `HARDWARE-SHAPED MOCK`; the teleop
launch prints `ATTACH PREFLIGHT OK`; the check tool's first line says it only subscribes.

**Stop if:** any of it differs. Nothing here touches the robot.

---

## Stage 2: on the robot, nothing moving

**Before you start**

- Stage 1 done; your limits file filled in (the attach refuses without it, even with motion
  disabled).
- **Network:** a direct Ethernet cable from the robot computer to the arm's control box, no
  switch in between.
- **Driver version (gate).** The installed xarm_ros2 must contain the 2025-03-18 change that
  makes the ros2_control interface read joint state from the arm every control cycle; before
  it, the state was stair-stepped at 5 Hz under `report_type:=normal`. Check:

  ```bash
  git -C "$RANGER_XARM_WS/src/xarm_ros2" merge-base --is-ancestor cc44f9f HEAD && echo contains
  ```

  If git does not know the commit (a shallow clone or a rebased fork), check the source
  instead: `grep -n 20250318 "$RANGER_XARM_WS/src/xarm_ros2/xarm_controller/src/hardware/uf_robot_system_hardware.cpp"`
  must print a line. `hw_check idle --workspace "$RANGER_XARM_WS"` judges the same rule.
  **If neither shows it, stop:** update the driver first.
- CPU governor at `performance` on the robot computer (the idle check reports it), for
  example `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`.
- Emergency stops in reach; nobody else near the arm.

**Commands**

```bash
ros2 launch ranger_xarm_moveit_config real_hardware_check.launch.py robot_ip:=<robot_ip>   # terminal 1
ros2 launch ranger_xarm_teleop_hardware teleop_hardware.launch.py stage:=check limits_file:=<limits>   # terminal 2
tools/relay.sh                                                                             # terminal 3
adb reverse tcp:8443 tcp:8443
ros2 launch ranger_xarm_teleop_bridge bridge.launch.py record_path:=recordings/hw_stage2.jsonl   # terminal 4
ros2 run ranger_xarm_teleop_check hw_check idle --limits <limits> --workspace "$RANGER_XARM_WS" --expect-active joint_state_broadcaster   # terminal 5
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
  `cpu governor`. `joint_states effort` is `INFO absent`: this driver does not copy effort
  into the state interface. It writes one report file under `~/teleop_data/hardware_checks/`.
- In the bring-up's RViz, add a PoseStamped display on `/teleop/preview/target_pose`. Point
  the right controller where "arm forward" should be and press A (calibrate). Hold the grip
  and move your hand forward slowly: the preview pose moves forward from the tool (in this
  stage the flange, `xarm_link_eef`: the check stage's description has no gripper), and stops
  when you release [`test_hwmock_check_stage`: the preview follows a stroke and is never a
  command]. The preview is shown only while motion is disabled; why it is not shown is in
  `ros2 topic echo --field preview_reason /teleop/state`.
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
- anything moves at all.

Then set `tool_pose_timeout_s` from this run's largest gap (Values you supply) before stage 4.

**How to back out:** Ctrl-C terminal 5, 2, 4, then 1. Nothing was commanded.

---

## Stage 3: MoveIt motion through the owner switch, nothing armed

**Before you start:** stage 2 passed and its report kept; P1 applied and built on the robot
computer; the arm in a posture with room around it; emergency stops in reach. Shut stage 2
down first.

**Commands**

```bash
ros2 launch ranger_xarm_moveit_config real_execution.launch.py robot_ip:=<robot_ip>        # terminal 1 (RViz with motion planning)
ros2 launch ranger_xarm_teleop_hardware teleop_hardware.launch.py stage:=execution motion_enabled:=true limits_file:=<limits>   # terminal 2
tools/relay.sh; adb reverse tcp:8443 tcp:8443                                              # terminal 3
ros2 launch ranger_xarm_teleop_bridge bridge.launch.py record_path:=recordings/hw_stage3.jsonl   # terminal 4
ros2 run ranger_xarm_teleop_check hw_check idle --limits <limits> --workspace "$RANGER_XARM_WS" --expect-active joint_state_broadcaster,xarm6_traj_controller --expect-inactive teleop_arm_controller   # terminal 5
```

`motion_enabled:=true` here is the deliberate step that lets the arm be switched to teleop at
all; with it false the mux refuses every switch [`test_hwmock_execution`: motion disabled never
switches to teleop]. Nobody arms in this stage.

**3a. A small planned motion.** In terminal 5 start
`hw_check motion --seconds 40 --joint-limits "$RANGER_XARM_WS/src/ranger_xarm_moveit_config/config/joint_limits.yaml"`,
then in RViz plan a small motion (one joint, about 10 degrees, velocity scaling 0.1) and
execute it. Then plan the arm to a posture with the wrist bent (joint 5 well away from zero;
the tests use 0, -0.6, -0.7, 0, 1.3, 0 rad) for stage 4.

- You should see: the arm follows the plan smoothly and stops at its goal; the check gives
  `PASS` on `stale repeats while moving` and `wrist oscillation`.
- **Stop if:** `stale repeats while moving` fails (the driver is not reading the arm every
  cycle: stop the stage); the arm moves anywhere the plan did not show; `wrist oscillation`
  fails. If `peak joint acceleration` fails, keep the report and plan the next motion slower;
  this is an open item (acceleration limiting is not built).

**3b. The switch to teleop and back.**

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

**How to back out:** switch back to MoveIt (`{owner: 1}`) if teleop owns the arm; Ctrl-C
terminal 5, 2, 4, then 1.

---

## Stage 4: teleop armed at very low limits

**Before you start:** stage 3 passed; your limits file at the first-session values; the arm at
the bent-wrist posture from stage 3a (arming is refused near the wrist singularity); emergency
stops in reach. Same terminals 1 to 4 as stage 3.

**Commands:** before arming, start the check (everything joins before arming):

```bash
ros2 run ranger_xarm_teleop_check hw_check teleop --seconds 120 --limits <limits> --joint-limits "$RANGER_XARM_WS/src/ranger_xarm_moveit_config/config/joint_limits.yaml"   # terminal 5
```

Then, in the headset: point forward and press A (calibrate), press B (arm). Hold the grip and
move your hand **5 cm, slowly**, in free space, then release. Then one more stroke the other
way.

**What you should see:** the tool follows your hand at no more than the speed cap and stops
within about a centimetre after you release (the arm finishes the points it already has); the
target in RViz; the check gives `PASS` on `stale repeats while moving`,
`target lead (largest, translation)` and `wrist oscillation`, and reports `target lag`
[`test_hwmock_execution`: one short armed stroke with the teleop check].

**Stop if** (release the grip; press the emergency stop if it does not stop):

- the arm moves without the grip held, or in a direction other than your hand's;
- it keeps moving more than about 2 cm after you release;
- the wrist oscillates (the solver's limit cycle near the wrist singularity: release, then
  disarm, and plan the arm away from the singular posture through MoveIt);
- `peak joint acceleration` fails: keep the report and stop the stage (open item: acceleration
  bursts; the mock shows them too at 40 % of the simulation speed caps).

Afterwards, recompute the lead limits from the measured lag (Values you supply) before more
strokes.

**If the arm leaves teleop by itself** (an arm error or an emergency stop makes the driver
switch every controller off): teleop disarms and stays disarmed (G-17), and the mux deactivates
a teleop controller it did not activate [`test_hwmock_execution`: the three G-17 tests].
Follow "After an emergency stop".

**How to back out:** release the grip, press B (disarm; the arm returns to MoveIt), Ctrl-C
terminal 5, 2, 4, then 1.

---

## Stage 5: gripper, then recording

**Gripper teleop on hardware is not built.** The real gripper is a `GripperCommand` action, not
a ros2_control joint, and its reported state is partly assumed; the teleop launch refuses
`gripper_enabled:=true`. The gripper stays with MoveIt (RViz) until the action adapter has been
built and reviewed.

**Recording**, after stage 4 has passed at the values you will record with:

```bash
ros2 launch ranger_xarm_sensors robot.launch.py robot_ip:=<robot_ip> enable_base:=false enable_ouster:=false enable_rplidar:=false gantry_d455_serial_no:=<serial> wrist_d435_serial_no:=<serial> color_profile:=640x480x30 depth_profile:=640x480x30   # terminal 1
ros2 launch ranger_xarm_teleop_hardware teleop_hardware.launch.py stage:=execution motion_enabled:=true record:=true limits_file:=<limits>   # terminal 2
ros2 run ranger_xarm_teleop_check hw_check cameras --profile install/ranger_xarm_teleop_record/share/ranger_xarm_teleop_record/profiles/hardware.yaml   # terminal 5, recorder up, before arming
```

The bring-up's cameras default to 15 fps; the recording profile expects 30 fps at 640x480,
hence the two profile arguments.

**You should see:** `PASS` on every camera's rate, gap and size rules.

**Stop if:** a camera does not hold 30 fps or loses frames (USB bandwidth): keep the report;
do not record with a profile it does not meet.

---

## After an emergency stop

1. If teleop was armed, press B to disarm before releasing the stop.
2. Shut down the teleop launch, the bridge and the bring-up.
3. Release the stop. Do not move the arm by hand while anything of the bring-up runs.
4. Start again from stage 2's checks.

Why, from the driver's source (xarm_ros2, `uf_robot_system_hardware.cpp`) and the controller
manager's (ros2_control 4.48.1):

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

G-17 is the backstop: armed teleop that loses the arm without asking disarms and never asks for
it again. The instruction is still to disarm before releasing the stop.

## A planned motion outlives move_group

If move_group dies during a planned motion, the trajectory controller still holds the whole
trajectory and finishes it [`test_hwmock_execution`: a planned motion when move_group dies].
Keep planned motions short in the first sessions, and be ready to press the emergency stop.
