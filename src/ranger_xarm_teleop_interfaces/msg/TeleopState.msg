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
uint8 HOLD_SERVO_NOT_READY = 18    # MoveIt Servo did not confirm it is running in pose mode
uint8 HOLD_SERVO_HALTED = 19       # MoveIt Servo halted (collision or singularity), see servo_status
uint8 hold_reason

# Bound that shaped the target of this cycle (G-6, G-8).
uint8 LIMIT_NONE = 0
uint8 LIMIT_OFFSET_TRANSLATION = 1
uint8 LIMIT_OFFSET_ROTATION = 2
uint8 LIMIT_KEEP_OUT = 3
uint8 LIMIT_SERVO = 4              # MoveIt Servo is slowing or halting, see servo_status
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
# target on /teleop/target_pose. Valid only when measured_tool_pose_valid is true.
bool measured_tool_pose_valid
geometry_msgs/Pose measured_tool_pose
