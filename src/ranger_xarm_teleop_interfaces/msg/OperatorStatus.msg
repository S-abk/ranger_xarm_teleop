# What the operator's panel shows (contract: /teleop/operator/status), published by the operator
# feedback node at 5 Hz from /teleop/state and /teleop/episode/state. Information only: nothing
# acts on it. Recorded with the data (what the operator saw). The texts are plain words for the
# operator; the codes beside them are the source messages' constants.
# Ages are seconds on the feedback node's monotonic clock since the newest message of that source
# arrived; NaN when none has arrived. A source older than its staleness limit is not valid, and
# its fields then keep their defaults (the panel shows "no data" for it).

std_msgs/Header header
float64 wall_time                   # host wall clock (s) at publication

# ---- teleop (TeleopState) ----
bool teleop_valid
float64 teleop_age
bool motion_enabled
bool armed
bool clutch_engaged
bool calibrated
float64 translation_scale
uint8 hold_reason                   # TeleopState.HOLD_* now
string hold_text
# The newest fault hold (a hold other than NONE, MOTION_DISABLED, NOT_ARMED, NOT_OWNER,
# NOT_CALIBRATED, CLUTCH_RELEASED), kept even if it lasted one cycle; HOLD_NONE when none in the
# latch time. last_fault_age: s since it last applied (0 while it applies), NaN when none.
uint8 last_fault
string last_fault_text
float64 last_fault_age
# The most severe TeleopState.LIMIT_* seen in the last limit hold time (LIMIT_NONE when none).
uint8 limit
string limit_text
bool gripper_enabled
bool gripper_active
uint8 gripper_hold                  # TeleopState.GRIPPER_HOLD_*
string gripper_text
float64 gripper_closed_fraction     # measured opening as a fraction of the range, 0 open, 1 closed; NaN unknown

# ---- episode (EpisodeState) ----
bool episode_valid
float64 episode_age
uint8 episode_state                 # EpisodeState.STATE_*
uint32 episode_index
float64 elapsed                     # s
uint32 kept_count
uint32 discarded_count
bool discard_armed
uint8 last_event                    # EpisodeState.EVENT_*
string last_event_text
float64 last_event_age              # s since the event (host wall clock); NaN when none
string last_refusal_text            # the newest refusal, plain words; empty when none
float64 last_refusal_age            # s; NaN when none
uint32 events_missed                # events lost between state messages since the node started
float64 disk_minutes_left           # NaN while unknown

# ---- reset (ResetState), appended: a headset-started planned motion (G-16) ----
# reset_valid is False when no reset node runs (the feature is off) or its state is too old;
# the reset fields then keep their defaults and the panel shows no reset line.
bool reset_valid
float64 reset_age
uint8 reset_state                   # ResetState.STATE_*
uint8 reset_reason                  # ResetState.REASON_*
string reset_text                   # the panel line, plain words
float64 reset_event_age             # s since the newest reset event (host wall clock); NaN when none

# ---- base (BaseState), appended: driving the base from the headset (G-19) ----
# base_valid is False when no base node runs (the feature is off) or its state is too old; the
# base fields then keep their defaults and the panel shows no base line.
bool base_valid
float64 base_age
uint8 base_state                    # BaseState.STATE_*
uint8 base_reason                   # BaseState.REASON_*
string base_text                    # the panel line, plain words
float64 base_event_age              # s since the newest base event (host wall clock); NaN when none
