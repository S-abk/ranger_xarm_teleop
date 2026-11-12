#!/usr/bin/env bash
# Create the relay's virtual environment (.venv at the repo root, ignored by git)
# and install the relay dependencies. Safe to run again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -r third_party/vr-teleop-kit/requirements-relay.txt

cat <<'MSG'

Relay environment ready (.venv).
Run the relay:     tools/relay.sh            (binds 127.0.0.1:8443)
USB to the Quest:  adb reverse tcp:8443 tcp:8443
Quest browser:     http://localhost:8443
MSG
