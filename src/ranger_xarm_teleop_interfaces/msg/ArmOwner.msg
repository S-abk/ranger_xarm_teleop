# Who owns the arm, published by the command mux (safety gate G-9).
# Exactly one of two trajectory controllers on the arm joints is active; the owner is the
# source that feeds the active one.

std_msgs/Header header

uint8 OWNER_NONE = 0        # neither controller is active, or the controller manager is unreachable
uint8 OWNER_MOVEIT = 1      # MoveIt's trajectory controller is active
uint8 OWNER_TELEOP = 2      # the teleop (MoveIt Servo) trajectory controller is active
uint8 OWNER_SWITCHING = 3   # a switch was requested and is not confirmed yet
uint8 owner

# Result of the last ownership request. A refused or failed request leaves the owner as it was.
uint8 STATUS_OK = 0
uint8 STATUS_REFUSED_MOVING = 1         # an arm joint was moving
uint8 STATUS_REFUSED_GOAL_ACTIVE = 2    # a trajectory goal was accepted or executing
uint8 STATUS_REFUSED_TELEOP_ACTIVE = 3  # the teleop node reported an engaged clutch
uint8 STATUS_STALE_JOINT_STATE = 4      # no fresh joint state to judge "stopped" from
uint8 STATUS_SWITCH_FAILED = 5          # the controller manager refused or did not confirm the switch
uint8 status

# Human-readable detail of the last request (empty when OK).
string detail
