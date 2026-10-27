#!/usr/bin/env bash
# App menu shortcut: make sure the virtual webcam exists, then open VanishCam with its control panel.
# After a reboot the virtual webcam is gone: the system asks for the password in a
# dialog (pkexec) only to load v4l2loopback.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

say() { case "${VANISHCAM_LANG:-${LANGUAGE:-${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}}}" in pt*) echo "$1" ;; *) echo "$2" ;; esac; }

DEVICE=/dev/video10

if [ ! -e "$DEVICE" ]; then
  say "Criando a webcam virtual VanishCam (o sistema vai pedir sua senha)..." \
      "Creating the VanishCam virtual webcam (the system will ask for your password)..."
  pkexec /usr/sbin/modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1 card_label=VanishCam
  for _ in $(seq 50); do
    [ -e "$DEVICE" ] && break
    sleep 0.1
  done
fi

if ! ./teams.sh "$@"; then
  echo
  read -rp "$(say "O VanishCam fechou com erro. Pressione Enter para sair..." \
                  "VanishCam closed with an error. Press Enter to exit...")"
fi
