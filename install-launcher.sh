#!/usr/bin/env bash
# Add the "VanishCam" shortcut to the app menu (for your user only).
# To remove it: rm ~/.local/share/applications/vanishcam.desktop
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${XDG_DATA_HOME:-$HOME/.local/share}/applications/vanishcam.desktop"

say() { case "${VANISHCAM_LANG:-${LANGUAGE:-${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}}}" in pt*) echo "$1" ;; *) echo "$2" ;; esac; }

mkdir -p "$(dirname "$TARGET")"
cat > "$TARGET" <<DESKTOP
[Desktop Entry]
Type=Application
Name=VanishCam
Comment=Webcam vanishing effects for video calls
Comment[pt_BR]=Efeitos de desaparecer na webcam para videochamadas
Comment[pt]=Efeitos de desaparecer na webcam para videochamadas
Exec="$APP_DIR/vanishcam-launcher.sh"
Icon=$APP_DIR/assets/vanish_icon.png
Terminal=true
Categories=AudioVideo;Video;
Keywords=webcam;teams;camera;
DESKTOP

desktop-file-validate "$TARGET"
update-desktop-database "$(dirname "$TARGET")" 2>/dev/null || true
say "Atalho instalado: $TARGET" "Shortcut installed: $TARGET"
