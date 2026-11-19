# A vibration request for one controller (contract: /teleop/haptic), published by the operator
# feedback node and forwarded to the headset by the bridge. On event; recorded with the data
# (what the operator felt). A request is a pattern of pulses; parallel arrays, one entry each.
# A request delivered later than max_age (on wall_time) anywhere on the way is dropped, never
# played late.

std_msgs/Header header      # ROS time of the request
float64 wall_time           # host wall clock (s) of the event this request reports; ages are taken on it

string hand                 # "left" or "right"
string source               # what asks for it: "episode" (later sources add their own names)
string event                # the source's event name, e.g. START, STOP, KEEP, DISCARD, MARK, REFUSED, AUTO_STOP
# A newer request of the same source always replaces the playing one; another source's request
# replaces it only with equal or higher priority (otherwise it is dropped, never queued).
uint8 priority

float32[] intensity         # per pulse, 0..1 (0: a pause)
uint32[] on_ms              # per pulse, vibration time
uint32[] off_ms             # per pulse, silence after it

float64 max_age             # s
uint32 source_count         # the source's event counter (EpisodeState.event_count for "episode")
uint32 missed               # events of the source skipped before this one (lost state messages); never played
