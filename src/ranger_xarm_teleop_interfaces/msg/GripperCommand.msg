# Gripper command of the teleop node: part of the recorded action, with /teleop/target_pose.
# Published on /teleop/gripper/command in every cycle where gripper teleop is active (enabled,
# armed, teleop owns the gripper, valid input); silent otherwise, and the gripper then holds
# the last command it was given. header.stamp is the cycle's stamp: the same as
# /teleop/target_pose and /teleop/state of that cycle.

std_msgs/Header header

# The gripper's driven joint (followers move with it through <mimic>).
string joint_name
# Commanded position of that joint, in its units (rad for a revolute joint), after the
# hysteresis, the speed limit and the link keep-out.
float64 position

# The operator's request: the trigger's last threshold crossing. NONE after arming or a
# freeze until the trigger crosses a threshold (the command then holds the reference).
uint8 REQUEST_NONE = 0
uint8 REQUEST_OPEN = 1
uint8 REQUEST_CLOSE = 2
uint8 request

# Where the command started when gripper teleop last became active.
uint8 SOURCE_NONE = 0
uint8 SOURCE_HELD = 1       # the opening the gripper's controller held (the previous owner's command)
uint8 SOURCE_OWN = 2        # teleop's own last command (after a freeze)
uint8 SOURCE_MEASURED = 3   # the measured opening (no held reference arrived in time)
uint8 reference_source
