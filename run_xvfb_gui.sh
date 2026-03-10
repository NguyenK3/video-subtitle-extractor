#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run not found. Install it first: sudo apt-get install -y xvfb xauth"
  exit 1
fi

# Auto-activate project virtualenv if it exists.
if [[ -f "$ROOT_DIR/videoEnv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/videoEnv/bin/activate"
fi

XVFB_RESOLUTION="${XVFB_RESOLUTION:-1920x1080x24}"

exec xvfb-run \
  --auto-servernum \
  --server-args="-screen 0 ${XVFB_RESOLUTION} -ac +extension RANDR" \
  python "$ROOT_DIR/gui.py" "$@"
