#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$ROOT_DIR/.novnc"
mkdir -p "$STATE_DIR"
WEB_ROOT="${NOVNC_WEB_ROOT:-$ROOT_DIR/web/novnc}"

NOVNC_PORT="${1:-${NOVNC_PORT:-6080}}"
DISPLAY_NUM="${DISPLAY_NUM:-:99}"
VNC_PORT="${VNC_PORT:-5901}"
XVFB_RESOLUTION="${XVFB_RESOLUTION:-1920x1080x24}"

for cmd in Xvfb x11vnc websockify; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing dependency: $cmd"
    echo "Install first: sudo apt-get install -y xvfb xauth x11vnc novnc websockify"
    exit 1
  fi
done

if [[ -f "$ROOT_DIR/videoEnv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/videoEnv/bin/activate"
fi

if ss -ltn | awk '{print $4}' | grep -q ":${NOVNC_PORT}$"; then
  echo "Port ${NOVNC_PORT} is already in use."
  exit 1
fi

cleanup() {
  set +e
  [[ -f "$STATE_DIR/novnc.pid" ]] && kill "$(cat "$STATE_DIR/novnc.pid")" 2>/dev/null || true
  [[ -f "$STATE_DIR/vnc.pid" ]] && kill "$(cat "$STATE_DIR/vnc.pid")" 2>/dev/null || true
  [[ -f "$STATE_DIR/gui.pid" ]] && kill "$(cat "$STATE_DIR/gui.pid")" 2>/dev/null || true
  [[ -f "$STATE_DIR/xvfb.pid" ]] && kill "$(cat "$STATE_DIR/xvfb.pid")" 2>/dev/null || true
  rm -f "$STATE_DIR"/*.pid
}
trap cleanup EXIT INT TERM

Xvfb "$DISPLAY_NUM" -screen 0 "$XVFB_RESOLUTION" -ac +extension RANDR >"$STATE_DIR/xvfb.log" 2>&1 &
echo $! >"$STATE_DIR/xvfb.pid"
export DISPLAY="$DISPLAY_NUM"

sleep 1

python "$ROOT_DIR/gui.py" >"$STATE_DIR/gui.log" 2>&1 &
echo $! >"$STATE_DIR/gui.pid"

x11vnc -display "$DISPLAY" -rfbport "$VNC_PORT" -localhost -forever -shared -nopw >"$STATE_DIR/x11vnc.log" 2>&1 &
echo $! >"$STATE_DIR/vnc.pid"

if [[ -x "$ROOT_DIR/scripts/prepare_novnc_web.sh" ]]; then
  "$ROOT_DIR/scripts/prepare_novnc_web.sh" >/dev/null
fi

websockify --web="$WEB_ROOT" "$NOVNC_PORT" "localhost:$VNC_PORT" >"$STATE_DIR/novnc.log" 2>&1 &
echo $! >"$STATE_DIR/novnc.pid"

echo "noVNC is running."
echo "Open: http://localhost:${NOVNC_PORT}/vnc.html"
echo "If using Codespaces/remote container, forward port ${NOVNC_PORT} first."
echo "Logs: $STATE_DIR"

wait "$(cat "$STATE_DIR/novnc.pid")"
