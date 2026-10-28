"""The virtual webcam that call apps pick as their camera (OBS, Unity Capture, v4l2loopback)."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from .i18n import t


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = t("virtual.unavailable", error=error)
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[VanishCam] {message}\n[VanishCam] {t('virtual.preview_only')}")
        return None
    print(f"[VanishCam] {t('virtual.active', device=camera.device, width=width, height=height, fps=args.fps)}")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))
