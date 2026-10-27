#!/usr/bin/env bash
# Atalho do menu: garante a webcam virtual e abre o VanishCam com o painel de controle.
# Depois de reiniciar o PC a webcam virtual não existe: o sistema pede a senha
# numa janela (pkexec) só para carregar o v4l2loopback.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

DEVICE=/dev/video10

if [ ! -e "$DEVICE" ]; then
  echo "Criando a webcam virtual VanishCam (o sistema vai pedir sua senha)..."
  pkexec /usr/sbin/modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1 card_label=VanishCam
  for _ in $(seq 50); do
    [ -e "$DEVICE" ] && break
    sleep 0.1
  done
fi

if ! ./teams.sh "$@"; then
  echo
  read -rp "O VanishCam fechou com erro. Pressione Enter para sair..."
fi
