# Per-frame validity and timing of the headset input, published once per headset
# frame with the identical header.stamp as the pose and joy messages of that frame.

# ROS time at publication; frame_id is the pose frame (quest_world).
std_msgs/Header header

# Client sequence number, incremented by the web client for every frame it builds.
# Preserved exactly; a gap means frames were lost between headset and ROS.
uint64 seq

# XR frame time from the headset (requestAnimationFrame timestamp, milliseconds,
# same timebase as client_time_ms). Preserved exactly.
float64 xr_time_ms

# performance.now() on the headset when the frame was sent, milliseconds. Preserved exactly.
float64 client_time_ms

# Number of WebXR reference-space "reset" events since the client page loaded.
# A change means the room-fixed frame moved; the safety gate treats it as tracking loss (G-4).
uint32 reset_count

HandStatus left
HandStatus right

# Headset (viewer) pose validity. The viewer pose is never published as a controller pose.
bool viewer_present
bool viewer_emulated_position

# Frames rejected by the bridge as malformed since the node started (INV-7).
uint64 malformed_count

# Relay messages that were not xr_frame messages (camera_list, echo, ...) since start.
uint64 ignored_count

# "relay" for live input, "replay:<file>" when republished from a recording.
string source
