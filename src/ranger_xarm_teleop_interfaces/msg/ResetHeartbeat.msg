# The reset node's sign of life for its watchdog (contract: /teleop/reset/heartbeat, 50 Hz,
# reliable, depth 1). The watchdog cancels the goal named here when these stop arriving while
# that goal is unfinished. Nothing else subscribes to it for control.

std_msgs/Header header
uint32 seq                      # counts up by one per message
bool goal_out                   # a goal is out (from just before it is sent until it is seen ended)
uint8[16] goal_id               # its id (zeros when none)
