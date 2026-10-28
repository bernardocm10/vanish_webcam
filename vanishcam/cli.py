"""Command-line options."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .i18n import LANGUAGES, set_language, t
from .modes import MODES


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError(t("cli.fps_range"))
    return fps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # --lang first: the help text itself must be in the chosen language.
    language = argparse.ArgumentParser(add_help=False)
    language.add_argument("--lang")
    chosen, _ = language.parse_known_args(argv)
    # An invalid value is reported by the full parser below.
    if chosen.lang in LANGUAGES:
        set_language(chosen.lang)

    parser = argparse.ArgumentParser(prog="vanishcam", description=t("cli.description"))
    # None: the webcam used last time (or the first one found).
    parser.add_argument("--camera", type=int, default=None, help=t("cli.camera"))
    # 1280x720 is the native 16:9 of most webcams (through MJPG):
    # video calls must not stretch a 4:3 picture.
    parser.add_argument("--width", type=int, default=1280, help=t("cli.width"))
    parser.add_argument("--height", type=int, default=720, help=t("cli.height"))
    parser.add_argument("--fullscreen", action="store_true", help=t("cli.fullscreen"))
    parser.add_argument("--no-mirror", action="store_true", help=t("cli.no_mirror"))
    parser.add_argument("--fps", type=positive_fps, default=30.0, help=t("cli.fps"))
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help=t("cli.virtual_camera"),
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help=t("cli.virtual_backend"),
    )
    parser.add_argument("--virtual-device", help=t("cli.virtual_device"))
    parser.add_argument("--mode", choices=tuple(MODES), help=t("cli.mode"))
    parser.add_argument("--call-audio", action="store_true", help=t("cli.call_audio"))
    parser.add_argument("--static-background", action="store_true", help=t("cli.static_background"))
    parser.add_argument(
        "--background-image", type=Path, metavar=t("cli.background_metavar"),
        help=t("cli.background_image"),
    )
    parser.add_argument("--video-window", action="store_true", help=t("cli.video_window"))
    parser.add_argument("--gesture-debug", action="store_true", help=t("cli.gesture_debug"))
    parser.add_argument("--lang", choices=LANGUAGES, help=t("cli.lang"))
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)
