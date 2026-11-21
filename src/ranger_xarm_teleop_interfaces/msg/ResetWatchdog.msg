# The reset watchdog's sign of life for the reset node (contract: /teleop/reset/watchdog, 10 Hz,
# reliable, depth 1). Sent only once the watchdog's cancel path has completed one round trip.
# The reset node sends no goal without a fresh one. Nothing else subscribes to it for control.

std_msgs/Header header
uint32 seq
bool stopping                   # the watchdog is cancelling a goal now (the reset node's heartbeat was late)
uint8[16] goal_id               # the goal it is cancelling or last cancelled (zeros when none)
uint32 stops                    # how many goals it has cancelled since it started
string last_report              # what it saw while cancelling the last one
uint32 heartbeat_seq            # the newest reset heartbeat it heard (0: none): it hears the reset node
