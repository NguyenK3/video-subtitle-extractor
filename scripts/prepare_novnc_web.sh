#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Prepare the noVNC web root for the VSE Remote Desktop viewer.
#
#  1. Copy the stock noVNC distribution (core/rfb.js, vendor/, etc.)
#  2. Copy and configure multilingual fonts
#  3. Install the VSE custom viewer (vse-viewer.html, vse-connection.js, …)
#  4. Optionally patch the stock vnc.html for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOVNC_SRC="${NOVNC_SRC:-/usr/share/novnc}"
WEB_ROOT="${NOVNC_WEB_ROOT:-$ROOT_DIR/web/novnc}"
VSE_SRC="$ROOT_DIR/web/novnc-src"
FONT_DST="$WEB_ROOT/fonts"

if [[ ! -d "$NOVNC_SRC" ]]; then
  echo "noVNC source directory not found: $NOVNC_SRC"
  echo "Install with: sudo apt-get install -y novnc websockify"
  exit 1
fi

# ── Step 1: Copy stock noVNC distribution ────────────────────────────────────

rm -rf "$WEB_ROOT"
mkdir -p "$WEB_ROOT"

if command -v rsync >/dev/null 2>&1; then
  rsync -a "$NOVNC_SRC"/ "$WEB_ROOT"/
else
  cp -a "$NOVNC_SRC"/. "$WEB_ROOT"/
fi

mkdir -p "$FONT_DST"

copy_first_match() {
  local pattern="$1"
  local target_name="$2"
  local found
  found="$(find /usr/share/fonts -type f -iname "$pattern" | head -n 1 || true)"
  if [[ -n "$found" ]]; then
    cp -f "$found" "$FONT_DST/$target_name"
    echo "Using font: $found -> $FONT_DST/$target_name"
  else
    echo "Font not found for pattern: $pattern"
  fi
}

copy_first_match 'NotoSans-Regular.ttf' 'NotoSans-Regular.ttf'
copy_first_match 'NotoSerif-Regular.ttf' 'NotoSerif-Regular.ttf'
copy_first_match 'DejaVuSans.ttf' 'DejaVuSans.ttf'
copy_first_match 'NotoSansCJK-Regular.ttc' 'NotoSansCJK-Regular.ttc'
copy_first_match 'SourceHanSans-Regular.otf' 'SourceHanSans-Regular.otf'
copy_first_match 'SourceHanSansSC-Regular.otf' 'SourceHanSansSC-Regular.otf'
copy_first_match 'SourceHanSansCN-Regular.otf' 'SourceHanSansCN-Regular.otf'

# ── Step 3: Generate custom-fonts.css ────────────────────────────────────────

cat > "$WEB_ROOT/custom-fonts.css" <<'CSS'
@font-face {
  font-family: "NotoSans";
  src: url("/fonts/NotoSans-Regular.ttf") format("truetype");
  font-display: swap;
}
@font-face {
  font-family: "NotoSerif";
  src: url("/fonts/NotoSerif-Regular.ttf") format("truetype");
  font-display: swap;
}
@font-face {
  font-family: "NotoSansCJK";
  src: url("/fonts/NotoSansCJK-Regular.ttc") format("truetype");
  font-display: swap;
}
@font-face {
  font-family: "SourceHanSans";
  src: url("/fonts/SourceHanSans-Regular.otf") format("opentype"),
       url("/fonts/SourceHanSansSC-Regular.otf") format("opentype"),
       url("/fonts/SourceHanSansCN-Regular.otf") format("opentype");
  font-display: swap;
}
:root, html, body, button, input, select, textarea,
#noVNC-control-bar, #noVNC_status, #noVNC_transition_text,
#noVNC_keyboardinput, .noVNC_message {
  font-family: "NotoSans", "NotoSansCJK", "NotoSerif",
    "DejaVu Sans",
    "SourceHanSans",
    sans-serif !important;
}
CSS

echo "Generated custom-fonts.css"

# ── Step 4: Install VSE viewer files ─────────────────────────────────────────

if [[ -d "$VSE_SRC" ]]; then
  cp -f "$VSE_SRC/vse-viewer.html"   "$WEB_ROOT/vse-viewer.html"
  cp -f "$VSE_SRC/vse-connection.js"  "$WEB_ROOT/vse-connection.js"
  cp -f "$VSE_SRC/vse-viewer.css"     "$WEB_ROOT/vse-viewer.css"
  echo "Installed VSE viewer files from $VSE_SRC"
else
  echo "WARNING: VSE source directory not found: $VSE_SRC"
  echo "The vse-viewer.html page will not be available."
fi

# ── Step 5: Inject font CSS into stock vnc.html (backward compat) ────────────

inject_css_link() {
  local html_file="$1"
  [[ -f "$html_file" ]] || return 0
  if ! grep -q 'custom-fonts.css' "$html_file"; then
    sed -i 's#</head>#    <link rel="stylesheet" href="/custom-fonts.css">\n</head>#' "$html_file"
  fi
}

inject_css_link "$WEB_ROOT/vnc.html"
inject_css_link "$WEB_ROOT/vnc_lite.html"

# ── Done ─────────────────────────────────────────────────────────────────────

echo "Prepared noVNC web root: $WEB_ROOT"
echo "  Default viewer: vse-viewer.html"
echo "  Legacy viewer:  vnc.html"
ls -la "$FONT_DST" 2>/dev/null || true
