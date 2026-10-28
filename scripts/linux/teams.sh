#!/usr/bin/env bash
# Start VanishCam sending the video to the "VanishCam" virtual webcam (/dev/video10),
# which you pick as the camera in Teams, and create the "VanishCam Mic" microphone
# (voice + effect sounds). Extra arguments are passed through, e.g. scripts/linux/teams.sh --mode naruto
set -euo pipefail
cd "$(dirname "$0")/../.."

# Messages follow the system language, like the app; VANISHCAM_LANG=pt or en overrides it.
say() { case "${VANISHCAM_LANG:-${LANGUAGE:-${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}}}" in pt*) echo "$1" ;; *) echo "$2" ;; esac; }

DEVICE=/dev/video10

if [ ! -e "$DEVICE" ]; then
  say "Webcam virtual $DEVICE não existe. Rode uma vez (pede senha):" \
      "Virtual webcam $DEVICE does not exist. Run this once (asks for your password):"
  echo '  sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1 card_label="VanishCam"'
  exit 1
fi

exec .venv/bin/python -m vanishcam \
  --virtual-camera --virtual-backend v4l2loopback --virtual-device "$DEVICE" \
  --no-mirror --call-audio "$@"
