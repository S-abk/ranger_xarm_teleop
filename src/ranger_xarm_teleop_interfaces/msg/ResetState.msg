# State of a headset-started planned motion (G-16): the return to a named posture, published by
# the reset node (contract: /teleop/reset/state, 10 Hz and on change). Hold-to-run: the arm moves
# along a MoveIt plan only while the operator holds the control. Information for the operator
# feedback node (panel and vibration); nothing commands the arm from it.

std_msgs/Header header
float64 wall_time               # host wall clock (s) at publication

uint8 STATE_IDLE = 0            # nothing going on; the control starts a reset
uint8 STATE_WAITING = 1         # the control is held; motion starts after the dwell
uint8 STATE_REFUSED = 2         # the last request was refused; reason says why
uint8 STATE_PLANNING = 3        # the goal is out; MoveIt is planning
uint8 STATE_MOVING = 4          # the arm is moving along the plan
uint8 STATE_STOPPING = 5        # the goal is being cancelled; waiting for it to end and the joints to rest
uint8 STATE_STOPPED = 6         # the last reset was stopped before the posture; reason says why
uint8 STATE_ARRIVED = 7         # the arm reached the posture
uint8 state

# Why the state is what it is (REFUSED, STOPPING, STOPPED, ARRIVED). Refusals:
uint8 REASON_NONE = 0
uint8 REASON_OFF = 1                    # the feature is off by configuration
uint8 REASON_NOT_SIMULATION = 2         # the robot description is not a simulator's
uint8 REASON_ARMED = 3                  # teleop is armed
uint8 REASON_NOT_MOVEIT_OWNER = 4       # the command mux does not report MoveIt as the owner
uint8 REASON_ARMED_UNKNOWN = 5          # a teleop node runs and no fresh state of it is at hand
uint8 REASON_RECORDING = 6              # an episode is being recorded
uint8 REASON_REVIEW = 7                 # an episode waits to be kept or discarded
uint8 REASON_EPISODE_UNKNOWN = 8        # an episode node runs and no fresh state of it is at hand
uint8 REASON_WATCHDOG_ABSENT = 9        # no fresh sign of life from the watchdog
uint8 REASON_MOVEIT_NOT_READY = 10      # move_group's action and its cancel path are not ready
uint8 REASON_PLAN_FAILED = 11           # MoveIt found no plan to the posture; nothing moved
uint8 REASON_START_IN_COLLISION = 12    # MoveIt considers the posture in collision; nothing moved
uint8 REASON_BUSY = 13                  # the previous goal is still being stopped
uint8 REASON_JOINTS_UNKNOWN = 14        # no fresh joint state of the arm
# Arrival:
uint8 REASON_REACHED = 20               # the plan was executed to the posture
uint8 REASON_ALREADY_THERE = 21         # the arm was already at the posture; nothing was sent
# Stops:
uint8 REASON_RELEASED = 30              # the control was released
uint8 REASON_INPUT_STALE = 31           # no fresh headset input (G-3; also a relay disconnect)
uint8 REASON_CONTROLLER_MISSING = 32    # no value for the control: the right controller is missing
uint8 REASON_SESSION_CHANGED = 33       # the XR session id changed
uint8 REASON_SEQ_BACKWARDS = 34         # the input's sequence or time went backwards
uint8 REASON_TELEOP_ARMED = 35          # teleop was armed during the motion
uint8 REASON_OWNER_CHANGED = 36         # the command mux stopped reporting MoveIt as the owner
uint8 REASON_EPISODE_STARTED = 37       # the episode state left idle during the motion
uint8 REASON_WATCHDOG_LOST = 38         # the watchdog's sign of life stopped during the motion
uint8 REASON_HEARTBEAT_LATE = 39        # the watchdog cancelled the goal: this node's heartbeat was late
uint8 REASON_CANCELLED_ELSEWHERE = 40   # the goal was cancelled by someone else
uint8 REASON_MOVEIT_FAILED = 41         # move_group ended the goal with an error; see moveit_error_code
uint8 REASON_NO_ANSWER = 42             # move_group did not answer the goal or its result in time
uint8 REASON_STATE_UNKNOWN = 43         # a state the motion depends on went stale (owner, teleop, episode)
uint8 reason
string reason_text                      # plain words, for the log

string posture                          # the named posture the control goes to
bool goal_out                           # a goal is out and not yet seen ended
uint8[16] goal_id                       # its id while one is out (zeros otherwise)
int32 moveit_error_code                 # move_group's error code of the last goal (0: none)

# Events, counted so that a lost message cannot hide one (the feedback node gives one vibration
# per event, the newest).
uint8 EVENT_NONE = 0
uint8 EVENT_START = 1                   # motion began (the goal was accepted)
uint8 EVENT_ARRIVED = 2
uint8 EVENT_REFUSED = 3
uint8 EVENT_STOPPED = 4                 # stopped by a cause other than the release
uint8 EVENT_RELEASED = 5                # stopped by the release (no vibration)
uint32 event_count
uint8 last_event
uint8 last_event_reason
float64 last_event_wall_time            # host wall clock of the newest event; NaN when none

# The last stop, measured on the joint states from the stop request: largest joint travel after
# it (rad) and the time until the joints came to rest (s). NaN until a stop has been measured.
float64 stop_distance
float64 stop_time

bool enabled                            # on by configuration and the description is a simulator's
bool watchdog_alive                     # a fresh sign of life from the watchdog
