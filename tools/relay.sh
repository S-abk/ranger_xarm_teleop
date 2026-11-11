#!/usr/bin/env bash
# Run the vendored vr-teleop-kit relay from the relay venv. Extra arguments are
# passed to the relay (default bind 127.0.0.1:8443).
# USB link: adb reverse tcp:8443 tcp:8443, then open http://localhost:8443 in the Quest browser.
# Set up the venv first with tools/relay_setup.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -x .venv/bin/python ]; then
  echo "relay.sh: .venv is missing; run tools/relay_setup.sh first" >&2
  exit 1
fi
PYTHONPATH=third_party/vr-teleop-kit/src exec .venv/bin/python -m vr_teleop_kit.relay.server "$@"
