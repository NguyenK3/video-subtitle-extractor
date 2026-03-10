#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOVNC_SRC="${NOVNC_SRC:-/usr/share/novnc}"
WEB_ROOT="${NOVNC_WEB_ROOT:-$ROOT_DIR/web/novnc}"
FONT_DST="$WEB_ROOT/fonts"

if [[ ! -d "$NOVNC_SRC" ]]; then
  echo "noVNC source directory not found: $NOVNC_SRC"
  echo "Install with: sudo apt-get install -y novnc websockify"
  exit 1
fi

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

:root,
html,
body,
button,
input,
select,
textarea,
#noVNC-control-bar,
#noVNC_status,
#noVNC_transition_text,
#noVNC_keyboardinput,
.noVNC_message {
  font-family:
    "NotoSans",
    "NotoSansCJK",
    "NotoSerif",
    "DejaVu Sans",
    "SourceHanSans",
    sans-serif !important;
}
CSS

inject_css_link() {
  local html_file="$1"
  [[ -f "$html_file" ]] || return 0
  if grep -q 'custom-fonts.css' "$html_file"; then
    return 0
  fi
  sed -i 's#</head>#    <link rel="stylesheet" href="/custom-fonts.css">\n</head>#' "$html_file"
}

inject_css_link "$WEB_ROOT/vnc.html"
inject_css_link "$WEB_ROOT/vnc_lite.html"

echo "Prepared noVNC web root with multilingual fonts: $WEB_ROOT"
ls -la "$FONT_DST" || true
