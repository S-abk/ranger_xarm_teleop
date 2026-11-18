# Demonstration episode state, published by the episode node (contract: /teleop/episode/state).
# Sent on every change and at 5 Hz, reliable, transient local, depth 1. Recorded with the data.
# header.stamp: ROS time of the host at publication (simulation time in simulation); wall_time
# is the host's wall clock at the same moment, so every recording carries clock pairs.

std_msgs/Header header
float64 wall_time

uint8 STATE_NOT_READY = 0   # the recorder or a required stream is not there yet; see detail
uint8 STATE_IDLE = 1        # ready: X starts an episode
uint8 STATE_RECORDING = 2   # X stops it; left trigger marks a moment
uint8 STATE_REVIEW = 3      # stopped: Y keeps it, Y held for 1 s discards it (marked, never deleted)
uint8 state

# Session directory name (no path) and the current episode (RECORDING, REVIEW) or the last one.
string session
uint32 episode_index
float64 elapsed             # seconds of the current episode, ROS time
uint32 kept_count
uint32 discarded_count

# Flags raised in the current episode so far (for example FAULT_RELEASE, DISARMED, DISK_LOW).
string[] flags

uint8 EVENT_NONE = 0
uint8 EVENT_START = 1
uint8 EVENT_STOP = 2
uint8 EVENT_KEEP = 3
uint8 EVENT_DISCARD = 4
uint8 EVENT_MARK = 5
uint8 EVENT_REFUSED = 6     # a press was refused; detail says why
uint8 EVENT_AUTO_STOP = 7   # the episode stopped by itself (disarm, lost state, disk, recorder); detail says why
uint8 last_event
builtin_interfaces/Time last_event_stamp
uint32 event_count          # increments with every event, so a missed message cannot hide one
string detail

bool discard_armed          # Y is held in REVIEW: the episode is discarded when the hold reaches 1 s
float64 disk_free_gb
float64 disk_minutes_left   # at the recording's measured data rate; NaN while unknown
