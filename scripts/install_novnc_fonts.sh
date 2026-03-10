#!/usr/bin/env bash
set -euo pipefail

# Install multilingual open-source fonts for noVNC remote desktop rendering.
# Works in Linux servers and Docker containers with apt.

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get is required on this script."
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update -y || true
sudo apt-get install -y --no-install-recommends \
  fonts-noto \
  fonts-noto-core \
  fonts-noto-extra \
  fonts-noto-cjk \
  fonts-dejavu-core \
  fonts-dejavu-extra \
  fonts-noto-color-emoji

# Source Han Sans package name may vary by distro mirror.
if apt-cache show fonts-source-han-sans >/dev/null 2>&1; then
  sudo apt-get install -y --no-install-recommends fonts-source-han-sans
fi

if apt-cache show fonts-adobe-source-han-sans-otc >/dev/null 2>&1; then
  sudo apt-get install -y --no-install-recommends fonts-adobe-source-han-sans-otc
fi

sudo fc-cache -f -v >/dev/null 2>&1 || true

# If Source Han Sans is still unavailable from apt, fetch open-source OTF directly.
if ! find /usr/share/fonts -type f \( -iname 'SourceHanSans*.otf' -o -iname 'SourceHanSans*.ttf' \) | grep -q .; then
  echo "Source Han Sans not found in system fonts. Downloading fallback OTF..."
  sudo mkdir -p /usr/share/fonts/opentype/source-han-sans
  if command -v curl >/dev/null 2>&1; then
    sudo curl -L --fail \
      -o /usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf \
      https://raw.githubusercontent.com/adobe-fonts/source-han-sans/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf || true
    sudo curl -L --fail \
      -o /usr/share/fonts/opentype/source-han-sans/SourceHanSans-Regular.otf \
      https://raw.githubusercontent.com/adobe-fonts/source-han-sans/release/OTF/SourceHanSans-Regular.otf || true
  fi
  sudo fc-cache -f -v >/dev/null 2>&1 || true
fi

echo "Font installation completed."
fc-list | grep -Ei 'Noto Sans|Noto Serif|Noto Sans CJK|DejaVu Sans|Source Han Sans' | head -n 20 || true
