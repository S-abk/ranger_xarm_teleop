#!/bin/bash
# Replay a recording into a running teleop stack and measure the run.
#
#   tools/replay_measured.sh <recording.jsonl> <run.jsonl> [record seconds]
#
# Needs a sourced workspace and a running teleop launch (mock or simulator). The arm
# button in a recording toggles arming, so the stack is disarmed first (/teleop/disarm)
# and the replay starts once MoveIt owns the arm again. Prints the tracking report.
set -e
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
REC=$(readlink -f "$1"); RUN=$2; SECS=${3:-14}
ros2 service call /teleop/disarm std_srvs/srv/Trigger >/dev/null 2>&1 \
  || { echo "no /teleop/disarm: is the teleop launch running?"; exit 2; }
for _ in $(seq 1 20); do
  owner=$(timeout 5 ros2 topic echo --once /teleop/mux/owner --field owner 2>/dev/null | head -1)
  [ "$owner" = 1 ] && break
  sleep 0.5
done
[ "$owner" = 1 ] || { echo "the arm did not return to MoveIt (owner=$owner)"; exit 1; }
python3 "$HERE/measure_tracking.py" record "$RUN" --duration "$SECS" &
recorder=$!
sleep 1
ros2 run ranger_xarm_teleop_bridge replay_node --ros-args -p file:="$REC" -p start_delay_s:=1.5
wait $recorder
python3 "$HERE/measure_tracking.py" report "$RUN"
