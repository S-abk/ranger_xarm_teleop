# State of the teleop servo node, published every target cycle (50 Hz).
# When a target is published in a cycle, this message carries the same header.stamp.

std_msgs/Header header

# G-1: the motion_enabled parameter (default false). False: no target is ever published.
bool motion_enabled
# The operator asked for teleop (arm button) and teleop owns the arm.
bool armed
# The clutch is engaged: targets are being published.
bool clutch_engaged

# Owner of the arm and the result of the last ownership request (ArmOwner.OWNER_*, STATUS_*).
uint8 owner
uint8 mux_status

# Why no target is published (HOLD_NONE while the clutch is engaged).
uint8 HOLD_NONE = 0
uint8 HOLD_MOTION_DISABLED = 1
uint8 HOLD_NOT_ARMED = 2
uint8 HOLD_NOT_OWNER = 3
uint8 HOLD_NOT_CALIBRATED = 4      # G-13: no calibration for the current session id and reset count
uint8 HOLD_CLUTCH_RELEASED = 5     # G-2
uint8 HOLD_INPUT_STALE = 6         # G-3: no input for longer than the timeout, by local arrival time
uint8 HOLD_POSE_ABSENT = 7         # G-4
uint8 HOLD_POSE_EMULATED = 8       # G-4
uint8 HOLD_OUT_OF_VIEW = 9         # G-4: controller outside the view cone of the headset
uint8 HOLD_HEAD_ABSENT = 10        # G-4: no headset pose to judge the view cone from
uint8 HOLD_HEAD_EMULATED = 11      # G-4
uint8 HOLD_RESET_CHANGED = 12      # G-4: reference-space reset count changed
uint8 HOLD_SESSION_CHANGED = 13    # G-4: XR session id changed
uint8 HOLD_SEQ_BACKWARDS = 14      # G-4: sequence number or headset time went backwards
uint8 HOLD_INPUT_JUMP = 15         # G-5: raw controller motion between two headset frames too fast
uint8 HOLD_STEP_REJECTED = 16      # G-5: target step beyond the per-cycle limit
uint8 HOLD_TOOL_POSE_STALE = 17    # G-11: measured tool pose too old or unavailable
uint8 HOLD_SERVO_NOT_READY = 18    # MoveIt Servo not verified, not in joint-jog mode, or its output stopped
uint8 HOLD_SERVO_HALTED = 19       # MoveIt Servo halted (collision or joint bound), see servo_status
uint8 HOLD_SOLVER = 20             # no kinematic chain or joint state, or commanded and measured disagree
uint8 hold_reason

# Bound that shaped the target or the joint command of this cycle (G-6, G-8, G-14).
uint8 LIMIT_NONE = 0
uint8 LIMIT_OFFSET_TRANSLATION = 1
uint8 LIMIT_OFFSET_ROTATION = 2
uint8 LIMIT_KEEP_OUT = 3
uint8 LIMIT_SERVO = 4              # MoveIt Servo is slowing or halting, see servo_status
uint8 LIMIT_LEAD = 5               # G-14: target clamped to the lead limit, excess absorbed
uint8 LIMIT_TOOL_SPEED = 6         # G-8: joint command scaled to the tool speed caps
uint8 LIMIT_JOINT_VELOCITY = 7     # G-8: joint command scaled to the joint velocity limits
uint8 LIMIT_JOINT_POSITION = 8     # G-8: a joint held at its position bound, the rest re-solved
uint8 LIMIT_LINK_KEEPOUT = 9       # G-15: motion toward the robot's fixed structure removed
uint8 active_limit

# Latest moveit_msgs/ServoStatus code; -1 when none was received.
int8 servo_status

# G-13: the calibration in use and what it was made under.
bool calibrated
uint32 calibration_session_id
uint32 calibration_reset_count
float64 calibration_heading        # rad, rotation about z from quest_world to the arm base

# Age of the newest input frame, seconds, measured on a monotonic clock from its local arrival.
float64 input_age
# Sequence number of the input frame this cycle used.
uint64 input_seq

# G-12: measured tool pose (tool_frame in the arm base frame), separate from the commanded
# target on /teleop/target_pose: forward kinematics of the newest /joint_states.
# Valid only when measured_tool_pose_valid is true.
bool measured_tool_pose_valid
geometry_msgs/Pose measured_tool_pose

# G-8: the damped least-squares solver. Valid while the clutch is engaged, else 0 / false.
# Conditioning: the smaller of the solver's two levels (position, then orientation given
# position) as the smallest singular value of the speed-normalised Jacobian: the share of
# the speed cap the tool reaches in its weakest direction with every joint at its velocity
# limit. >= 1 unrestricted, 0 singular. The same units for translation and rotation.
float64 conditioning
# Damping is in use (reduced accuracy near a singularity), and its normalised value.
bool damped
float64 damping
# Target vs the tool pose of the solver's reference (MoveIt Servo's commanded state):
# accuracy lost to damping and limits, without the lag. m and rad.
float64 solver_error_translation
float64 solver_error_rotation
# G-14: target vs the measured tool pose (includes the lag). m and rad.
float64 lead_translation
float64 lead_rotation
# Tool speed of the joint command sent this cycle, m/s and rad/s (G-8 caps).
float64 commanded_linear_speed
float64 commanded_angular_speed

# Translation scale selected by the operator (right thumbstick click switches between the
# fine and the coarse value; every arming starts at fine). Filled in every cycle, engaged or
# not: while engaged it is the scale in use, otherwise the one the next engagement uses.
# Hand translation is multiplied by it; rotation is never scaled.
float64 translation_scale

# G-15 link keep-out: the arm's links (and the tool) against the robot's own fixed structure,
# from a model built from the robot description at start-up (spheres covering each moving
# link; it never reports more clearance than the collision geometry has).
# Clearance (m): the smallest over all pairs, at the measured joint state (updated at 10 Hz,
# 2 Hz while engaged). Exact within the keep-out's influence distance (the margin + 60 mm);
# larger values are a lower bound. NaN when there is no model or no joint state.
float64 keepout_clearance
# The pair with that clearance: a moving link (arm or tool) and a fixed link (or, for the
# tool, an arm link). Empty when the clearance is not exact.
string keepout_link
string keepout_obstacle
# G-12: the same at the solver's reference (MoveIt Servo's commanded state) while the clutch
# is engaged; NaN otherwise.
float64 keepout_clearance_commanded
# The keep-out shaped this cycle's joint command (also when another limit is reported in
# active_limit).
bool keepout_bound
# The margin in use (m), from the limits.
float64 keepout_margin

# Gripper teleop (right trigger). The command (the recorded action, also on
# /teleop/gripper/command) and the measured opening are separate fields (G-12).
# gripper_enabled: the gripper_enabled parameter (default false). False: no gripper command is
# ever published and the trigger has no effect.
bool gripper_enabled
# The gripper was commanded this cycle (a GripperCommand with this header.stamp was published).
bool gripper_active
# Why the gripper is not commanded, or not moving.
uint8 GRIPPER_HOLD_NONE = 0
uint8 GRIPPER_HOLD_DISABLED = 1
uint8 GRIPPER_HOLD_NOT_ARMED = 2
uint8 GRIPPER_HOLD_NOT_OWNER = 3      # teleop does not own the gripper (the mux swaps it with the arm)
uint8 GRIPPER_HOLD_INPUT = 4          # a gate condition or trip, stale input, or no valid trigger value
uint8 GRIPPER_HOLD_NO_REFERENCE = 5   # waiting for the opening the controller holds
uint8 GRIPPER_HOLD_AWAIT_TRIGGER = 6  # after arming or a freeze: holding until the trigger crosses a threshold
uint8 GRIPPER_HOLD_KEEP_OUT = 7       # G-15: a step toward the robot's fixed structure was not taken
uint8 gripper_hold
# The command in force, every cycle: the last command teleop sent (or the reference it holds)
# while teleop owns the gripper; NaN when teleop holds none (not the owner, or no reference yet).
float64 gripper_command
# GripperCommand.REQUEST_* and SOURCE_* of the command in force.
uint8 gripper_request
uint8 gripper_reference_source
# Measured opening of the driven joint from /joint_states, with that message's stamp. What
# "measured" means depends on the backend: a physics simulator measures it; mock execution
# echoes the command; on the hardware driver it is partly assumed (see the README).
bool gripper_measured_valid
float64 gripper_measured
builtin_interfaces/Time gripper_measured_stamp
