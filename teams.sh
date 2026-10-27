#!/usr/bin/env bash
# Abre o VanishCam enviando o vídeo para a webcam virtual "VanishCam" (/dev/video10),
# que você escolhe como câmera no Teams. Argumentos extras são repassados, ex.: ./teams.sh --mode naruto
set -euo pipefail
cd "$(dirname "$0")"

DEVICE=/dev/video10

if [ ! -e "$DEVICE" ]; then
  echo "Webcam virtual $DEVICE não existe. Rode uma vez (pede senha):"
  echo '  sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1 card_label="VanishCam"'
  exit 1
fi

exec .venv/bin/python vanish_cam.py \
  --virtual-camera --virtual-backend v4l2loopback --virtual-device "$DEVICE" \
  --no-mirror "$@"
