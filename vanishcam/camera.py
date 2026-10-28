"""The webcam: opening it, reopening it when it drops, and the card shown meanwhile."""

from __future__ import annotations

import platform
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .control_panel import AMBER, font_file
from .i18n import t

# Failed reads for this long mean the webcam is gone (one dropped frame is normal);
# then it is reopened every CAMERA_RETRY_SECONDS until it is back.
CAMERA_LOST_SECONDS = 0.5
CAMERA_RETRY_SECONDS = 1.0


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(t("camera.unavailable", index=index))

    # Without MJPG many USB webcams give 1280x720 at only 10 FPS (YUYV) and
    # OpenCV falls back to 640x480.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class CameraReopener:
    """Reopen the webcam in a thread: opening can block for seconds, the video must not."""

    def __init__(self, open_capture: Callable[[], cv2.VideoCapture], retry_seconds: float):
        self.open_capture = open_capture
        self.retry_seconds = retry_seconds
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.result: tuple[cv2.VideoCapture, np.ndarray] | None = None
        self.thread = threading.Thread(target=self._run, name="camera-reopen", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                capture = self.open_capture()
            except (RuntimeError, cv2.error):
                capture = None
            if capture is not None:
                ok, frame = capture.read()
                if ok and frame is not None:
                    with self.lock:
                        self.result = (capture, frame)
                    return
                capture.release()
            self.stopping.wait(self.retry_seconds)

    def take(self) -> tuple[cv2.VideoCapture, np.ndarray] | None:
        with self.lock:
            result, self.result = self.result, None
            return result

    def close(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=5.0)
        reopened = self.take()
        if reopened is not None:
            reopened[0].release()


class CameraFeed:
    """The webcam, reopened in the background when it stops sending images (unplugged, taken by another app).

    read() returns None instead of failing: first for a short grace period (a single
    dropped frame is common), then while CameraReopener retries. The first open stays
    synchronous, so a missing webcam at startup is still reported as an error.
    """

    def __init__(self, open_capture: Callable[[], cv2.VideoCapture],
                 lost_after: float = CAMERA_LOST_SECONDS, retry_seconds: float = CAMERA_RETRY_SECONDS):
        self.open_capture = open_capture
        self.lost_after = lost_after
        self.retry_seconds = retry_seconds
        self.capture = open_capture()
        self.failing_since: float | None = None
        self.lost_since: float | None = None
        self.reopener: CameraReopener | None = None

    @property
    def reconnecting(self) -> bool:
        return self.reopener is not None

    def read(self) -> np.ndarray | None:
        if self.reopener is not None:
            reopened = self.reopener.take()
            if reopened is None:
                return None
            self.capture, frame = reopened
            self.reopener = None
            self.lost_since = None
            return frame
        ok, frame = self.capture.read()
        if ok and frame is not None:
            self.failing_since = None
            return frame
        now = time.perf_counter()
        if self.failing_since is None:
            self.failing_since = now
        elif now - self.failing_since >= self.lost_after:
            self.capture.release()
            self.failing_since = None
            self.lost_since = now
            self.reopener = CameraReopener(self.open_capture, self.retry_seconds)
        return None

    def release(self) -> None:
        if self.reopener is not None:
            self.reopener.close()
            self.reopener = None
        self.capture.release()


def reconnecting_card(base: np.ndarray, title: str, detail: str, lamp_on: bool) -> np.ndarray:
    """The frame sent while the webcam is gone: the last image, blurred and dimmed, with a status line.

    base is already blurred and dimmed (once, when the webcam was lost); the text is
    drawn with the panel's fonts, which also cover accented letters.
    """
    from PIL import Image, ImageDraw, ImageFont

    height, width = base.shape[:2]
    image = Image.fromarray(np.ascontiguousarray(base[..., ::-1]))
    draw = ImageDraw.Draw(image)
    unit = height / 720

    def face(style: str, size: float):
        name = font_file(style)
        return ImageFont.truetype(name, round(size * unit)) if name else ImageFont.load_default(size * unit)

    big, small = face("title", 40), face("mono", 20)
    lamp = round(16 * unit)
    title_width = draw.textlength(title, font=big)
    x = (width - title_width - lamp * 2) / 2
    y = height / 2
    draw.rectangle(
        [x, y - lamp / 2 - 4 * unit, x + lamp, y + lamp / 2 - 4 * unit],
        fill=AMBER if lamp_on else None, outline=AMBER, width=max(1, round(2 * unit)),
    )
    draw.text((x + lamp * 2, y), title, font=big, fill=(244, 242, 237), anchor="lm")
    draw.text((width / 2, y + 44 * unit), detail, font=small, fill=(190, 185, 175), anchor="mm")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def dim_for_card(frame: np.ndarray | None, width: int, height: int) -> np.ndarray:
    if frame is None:
        return np.full((height, width, 3), 26, np.uint8)
    small = cv2.resize(frame, (max(1, width // 8), max(1, height // 8)), interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.convertScaleAbs(blurred, alpha=0.35)
