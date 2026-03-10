#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$ROOT_DIR/.novnc"

if [[ ! -d "$STATE_DIR" ]]; then
  echo "No running noVNC state found."
  exit 0
fi

for name in novnc vnc gui xvfb; do
  pid_file="$STATE_DIR/${name}.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    kill "$pid" 2>/dev/null || true
    rm -f "$pid_file"
  fi
done

echo "Stopped noVNC/Xvfb/GUI processes (if they were running)."
