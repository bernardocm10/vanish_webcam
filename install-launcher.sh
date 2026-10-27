#!/usr/bin/env bash
# Instala o atalho "VanishCam" no menu de aplicativos (só para o seu usuário).
# Para remover: rm ~/.local/share/applications/vanishcam.desktop
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${XDG_DATA_HOME:-$HOME/.local/share}/applications/vanishcam.desktop"

mkdir -p "$(dirname "$TARGET")"
cat > "$TARGET" <<EOF
[Desktop Entry]
Type=Application
Name=VanishCam
Comment=Efeitos de desaparecer na webcam para o Teams
Exec="$APP_DIR/vanishcam-launcher.sh"
Icon=$APP_DIR/assets/vanish_icon.png
Terminal=true
Categories=AudioVideo;Video;
Keywords=webcam;teams;camera;
EOF

desktop-file-validate "$TARGET"
update-desktop-database "$(dirname "$TARGET")" 2>/dev/null || true
echo "Atalho instalado: $TARGET"
