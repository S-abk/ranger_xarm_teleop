#!/usr/bin/env bash
# Create the relay's virtual environment (.venv at the repo root, ignored by git)
# and install the relay dependencies. Safe to run again.
# A .venv without pip (an earlier `python3 -m venv` that failed half way, for example when
# python3-venv is not installed) is not ready: it is removed and made again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -d .venv ] && [ ! -x .venv/bin/pip ]; then
  echo "relay_setup.sh: .venv is incomplete (no .venv/bin/pip); making it again" >&2
  rm -rf .venv
fi
if [ ! -d .venv ]; then
  if ! python3 -m venv .venv; then
    rm -rf .venv
    echo "relay_setup.sh: could not create .venv. On Ubuntu: sudo apt install python3-venv" >&2
    exit 1
  fi
  if [ ! -x .venv/bin/pip ]; then
    rm -rf .venv
    echo "relay_setup.sh: .venv has no pip. On Ubuntu: sudo apt install python3-venv" >&2
    exit 1
  fi
fi
.venv/bin/pip install -r third_party/vr-teleop-kit/requirements-relay.txt

cat <<'MSG'

Relay environment ready (.venv).
Run the relay:     tools/relay.sh            (binds 127.0.0.1:8443)
USB to the Quest:  adb reverse tcp:8443 tcp:8443
Quest browser:     http://localhost:8443
MSG
