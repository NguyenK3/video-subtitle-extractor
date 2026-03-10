#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-6080}"

echo "[1/3] Checking fontconfig fallback for multilingual text..."
fc-match -s ':lang=zh-cn' | head -n 3 || true
fc-match -s ':lang=ja' | head -n 3 || true
fc-match -s ':lang=ko' | head -n 3 || true
fc-match -s ':lang=vi' | head -n 3 || true

echo "[2/3] Checking noVNC custom stylesheet..."
curl -sSf "http://localhost:${PORT}/vnc.html" | grep -q 'custom-fonts.css'
curl -sSf "http://localhost:${PORT}/custom-fonts.css" | grep -q 'font-family: "NotoSans"'

echo "[3/3] Checking font files served by noVNC..."
curl -sI "http://localhost:${PORT}/fonts/NotoSans-Regular.ttf" | grep -q '200 OK'
curl -sI "http://localhost:${PORT}/fonts/NotoSansCJK-Regular.ttc" | grep -q '200 OK'
curl -sI "http://localhost:${PORT}/fonts/DejaVuSans.ttf" | grep -q '200 OK'

# Source Han is optional in some distros; accept either this font or Noto CJK fallback.
if curl -sI "http://localhost:${PORT}/fonts/SourceHanSansSC-Regular.otf" | grep -q '200 OK'; then
  echo "Source Han Sans served successfully."
else
  echo "Source Han Sans file is not served; Noto Sans CJK fallback is active."
fi

echo "noVNC font verification passed on port ${PORT}."
