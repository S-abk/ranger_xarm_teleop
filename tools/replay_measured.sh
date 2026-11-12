#!/bin/bash
# Replay a recording into a running teleop stack and measure the run.
#
#   tools/replay_measured.sh <recording.jsonl> <run.jsonl> [record seconds]
#
# Needs a sourced workspace and a running teleop launch (mock or simulator). The arm
# button in a recording toggles arming, so the stack must be disarmed before the replay:
# this script refuses to start otherwise. Prints the tracking report at the end.
set -e
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
REC=$(readlink -f "$1"); RUN=$2; SECS=${3:-14}
armed=$(timeout 10 ros2 topic echo --once /teleop/state --field armed 2>/dev/null | head -1)
case "$armed" in
  false|False) ;;
  true|True) echo "teleop is armed; disarm first (arm button, or replay a recording that presses it)"; exit 1;;
  *) echo "no /teleop/state: is the teleop launch running?"; exit 2;;
esac
python3 "$HERE/measure_tracking.py" record "$RUN" --duration "$SECS" &
recorder=$!
sleep 1
ros2 run ranger_xarm_teleop_bridge replay_node --ros-args -p file:="$REC" -p start_delay_s:=1.5
wait $recorder
python3 "$HERE/measure_tracking.py" report "$RUN"
