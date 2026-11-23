# State of base driving from the headset (G-19), published by the base node (contract:
# /teleop/base/state, 10 Hz and on change). Hold-to-drive: the base is commanded only while the
# operator holds the left grip. Information for the operator feedback node (panel and vibration),
# the arm's teleop node does not depend on it, and nothing commands the base from it.

std_msgs/Header header
float64 wall_time               # host wall clock (s) at publication

bool enabled                    # on by configuration and the description is a simulator's

uint8 STATE_OFF = 0             # off by configuration, or not a simulator: the grip does nothing
uint8 STATE_IDLE = 1            # nothing commanded; holding the left grip starts a drive
uint8 STATE_DRIVING = 2         # the grip is held and the base is commanded every cycle
uint8 STATE_STOPPING = 3        # the drive ended (zero sent, publishing stopped); odometry not yet at rest
uint8 STATE_REFUSED = 4         # the last press was refused; reason says why
uint8 STATE_STOPPED = 5         # the last drive was stopped by something other than the release; reason says why
uint8 state

# One motion mode per command, as the base executes it.
uint8 MODE_NONE = 0
uint8 MODE_TRANSLATE = 1        # forward and sideways along the left stick (the robot's own axes)
uint8 MODE_TURN = 2             # turn in place from the right stick's x axis
uint8 mode

# Why (REFUSED, STOPPED; IDLE and STOPPING keep the reason of the drive's end).
uint8 REASON_NONE = 0
uint8 REASON_OFF = 1                    # off by configuration
uint8 REASON_NOT_SIMULATION = 2         # the robot description is not a simulator's
uint8 REASON_RELEASED = 3               # the left grip was released
uint8 REASON_INPUT_STALE = 4            # no fresh headset input (G-3; also a relay disconnect)
uint8 REASON_CONTROLLER_MISSING = 5     # no value for the left controller's controls
uint8 REASON_SESSION_CHANGED = 6        # the XR session id changed
uint8 REASON_SEQ_BACKWARDS = 7          # the input's sequence or time went backwards
uint8 REASON_CLUTCH_ENGAGED = 8         # the arm's clutch is engaged, or its stop ramp still runs
uint8 REASON_RIGHT_GRIP_HELD = 9        # the right grip (the arm's clutch) was held when the drive would start
uint8 REASON_TELEOP_UNKNOWN = 10        # no fresh state of the arm's teleop node
uint8 REASON_ODOM_MISSING = 11          # no fresh odometry: the base's rest cannot be judged
uint8 REASON_RESET_ACTIVE = 12          # a return to the ready posture is under way
uint8 REASON_RESET_UNKNOWN = 13         # a reset node runs and no fresh state of it is at hand
uint8 REASON_OTHER_PUBLISHER = 14       # something else publishes the base command (G-9); text names it
uint8 REASON_TOOL_POSE_UNKNOWN = 15     # the tool speed limit is set and the arm's posture is unknown
uint8 REASON_SHUTDOWN = 16              # the base node is stopping
uint8 reason
string reason_text                      # plain words, for the log and the panel

bool driving                            # a command was published this cycle (the base is commanded)
geometry_msgs/Twist command             # the last command published (zero when none)
bool turn_ignored                       # the right stick asks for a turn while the left stick wins

bool at_rest                            # odometry says the base is at rest
float64 odom_age                        # s since the newest odometry arrived; NaN when none

float64 linear_cap                      # m/s, forward and sideways together
float64 turn_cap                        # rad/s
uint8 LIMIT_NONE = 0
uint8 LIMIT_CAP = 1                     # the command is at a speed cap
uint8 LIMIT_ACCELERATION = 2            # the command is changing at the acceleration limit
uint8 LIMIT_TOOL_SPEED = 3              # the whole command is scaled down by the tool's speed in the room
uint8 active_limit
float64 tool_speed_cap                  # m/s; NaN when unset (no such limit)
float64 tool_speed_scale                # the scale the tool speed limit applies (1: not acting); NaN when unset

# Events, counted so that a lost message cannot hide one.
uint8 EVENT_NONE = 0
uint8 EVENT_START = 1                   # a drive was accepted
uint8 EVENT_REFUSED = 2
uint8 EVENT_STOPPED = 3                 # stopped by a cause other than the release
uint8 EVENT_RELEASED = 4                # stopped by the release (no vibration)
uint8 EVENT_AT_REST = 5                 # the base came to rest after a drive: the arm's clutch may engage
uint32 event_count
uint8 last_event
uint8 last_event_reason
float64 last_event_wall_time            # host wall clock of the newest event; NaN when none
