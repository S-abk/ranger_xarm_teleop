# Validity of one controller's pose in the headset frame it was published from.
# Both flags are levels (the full current condition), not edges.

# True when the headset reported a tracked pose for this controller in this frame.
# False: no pose message was published for this hand with this stamp.
# A hand-tracked input source (no controller) also counts as not present.
bool pose_present

# True when the input source for this hand is a tracked hand, not a controller.
# pose_present is then false and no pose is published: a hand is never streamed as a
# controller pose. The flag only tells the operator why the controller is missing.
bool hand_tracked

# True when WebXR flagged the position as emulated (XRPose.emulatedPosition):
# the orientation is tracked but the position is an estimate, not a measurement.
# The pose message is still published; consumers must not follow it (safety gate G-4).
bool emulated_position
