# How far the robot is from starting to tip, published by the stability monitor (contract:
# "Stability monitor", /teleop/stability/state, every cycle). INFORMATION ONLY: nothing in the
# stack reads it to decide a command, a hold or a refusal, and the monitor has no command path.
# It does not see a force on the tool (the arm caught on something while the base drives).
#
# The figure: the signed distance (m) from the centre of pressure, where the floor's reaction
# acts, to the nearest edge of the wheel contacts' hull, positive inside, negative past it, with
# that edge named. On a level floor at rest it is the distance from the point under the centre
# of mass to that edge. Nominal: the mass model in force; pessimistic: the smallest over the
# mass file's bounds (never above the nominal).

std_msgs/Header header          # stamp of the cycle; frame: the base frame of the figure
float64 wall_time               # host wall clock (s) at publication

bool available                  # false: every figure is NaN and reason says why

uint8 REASON_OK = 0
uint8 REASON_NO_DESCRIPTION = 1         # no robot description received
uint8 REASON_DESCRIPTION_UNUSABLE = 2   # e.g. no wheel contacts in it; text says why
uint8 REASON_NO_MASS_FILE = 3           # no mass file given
uint8 REASON_MASS_FILE_REFUSED = 4      # unreadable or refused; text names the key
uint8 REASON_NO_JOINT_STATES = 5        # no value yet for a joint the figure needs; text names it
uint8 REASON_JOINT_STATES_STALE = 6     # a joint value older than its timeout
uint8 REASON_NO_ODOMETRY = 7
uint8 REASON_ODOMETRY_STALE = 8
uint8 REASON_NO_TILT = 9                # a configured tilt source (IMU) missing or stale
uint8 REASON_MOVING_WITHOUT_TILT = 10   # no tilt source configured: a figure only at rest
uint8 REASON_NOT_FINITE = 11            # an input or a result is not finite
uint8 REASON_NOT_SIMULATION = 12        # simulation only until reviewed for hardware
uint8 reason
string reason_text

uint8 TILT_NONE = 0
uint8 TILT_MEASURED = 1                 # the IMU's acceleration and rate (its orientation is never read)
uint8 TILT_LEVEL_ASSUMED = 2            # no tilt source configured; at rest only
uint8 tilt_source

# Edges by the direction of their outward normal in the base frame.
uint8 EDGE_NONE = 0
uint8 EDGE_FRONT = 1
uint8 EDGE_FRONT_LEFT = 2
uint8 EDGE_LEFT = 3
uint8 EDGE_REAR_LEFT = 4
uint8 EDGE_REAR = 5
uint8 EDGE_REAR_RIGHT = 6
uint8 EDGE_RIGHT = 7
uint8 EDGE_FRONT_RIGHT = 8

float64 margin_m                        # the figure, nominal
uint8 edge
float64 margin_pessimistic_m            # the figure over the mass file's bounds
uint8 edge_pessimistic

# If the base stopped now: the figure through a stop from the present motion at stop_decel
# (and the turn's stop_turn_decel), amplified by the suspension, never above the present
# figure's minimum over the last period of the body's slowest mode.
float64 stop_margin_m
float64 stop_margin_pessimistic_m
uint8 stop_edge
float64 stop_decel_m_s2                 # the deceleration the stop figure assumes
float64 stop_turn_decel_rad_s2

# Energy to roll the robot over the edge the motion heads toward, minus the kinetic energy of
# the present motion (pessimistic centre of mass). Positive: a stop from this motion cannot
# roll the robot over that edge (a sufficient condition only).
float64 energy_reserve_j
uint8 energy_edge

float64 cop_x_m                         # centre of pressure, base frame
float64 cop_y_m
float64 roll_deg                        # tilt of the body (slope + lean), from the tilt filter
float64 pitch_deg
float64 speed_m_s                       # base speed from odometry
float64 turn_rate_rad_s
bool at_rest                            # by odometry (the base's own rest rule)

string model_sha256                     # the mass model in force (description + mass file)
string note                             # "does not see a force on the tool"
