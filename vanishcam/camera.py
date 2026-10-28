"""The webcam: opening it, reopening it when it drops, and the card shown meanwhile."""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .control_panel import AMBER, font_file
from .i18n import t

# Failed reads for this long mean the webcam is gone (one dropped frame is normal);
# then it is reopened every CAMERA_RETRY_SECONDS until it is back.
CAMERA_LOST_SECONDS = 0.5
CAMERA_RETRY_SECONDS = 1.0
# Virtual cameras are outputs (ours among them): reading one back would loop the video.
VIRTUAL_CAMERAS = ("obs virtual camera", "unity video capture", "vanishcam")


@dataclass(frozen=True)
class CameraInfo:
    index: int  # The number open_camera takes.
    name: str


def list_cameras() -> tuple[CameraInfo, ...]:
    """The real webcams by name, numbered like open_camera; empty when they cannot be listed."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras

        # The same backend open_camera uses, so the numbers match.
        system = platform.system()
        backend = cv2.CAP_DSHOW if system == "Windows" else cv2.CAP_V4L2 if system == "Linux" else cv2.CAP_ANY
        found = enumerate_cameras(backend)
    except Exception:
        return ()
    cameras: dict[int, CameraInfo] = {}
    for info in found:
        if info.index not in cameras and not any(name in info.name.lower() for name in VIRTUAL_CAMERAS):
            cameras[info.index] = CameraInfo(info.index, info.name)
    return tuple(cameras.values())


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
                        # Abandoned meanwhile (another camera was chosen): let the webcam go.
                        if self.stopping.is_set():
                            capture.release()
                        else:
                            self.result = (capture, frame)
                    return
                capture.release()
            self.stopping.wait(self.retry_seconds)

    def take(self) -> tuple[cv2.VideoCapture, np.ndarray] | None:
        with self.lock:
            result, self.result = self.result, None
            return result

    def abandon(self) -> None:
        """Stop without waiting: an open in progress releases the webcam by itself."""
        with self.lock:
            self.stopping.set()
            result, self.result = self.result, None
        if result is not None:
            result[0].release()

    def close(self) -> None:
        self.abandon()
        self.thread.join(timeout=5.0)


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
        # True while a reopen asked for by the user runs: not a lost webcam.
        self.opening = False
        self.frame_size: tuple[int, int] | None = None
        self.fps: float | None = None
        self.last_frame_at: float | None = None

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
            self.opening = False
            self.fps = self.last_frame_at = None
            return self._delivered(frame)
        ok, frame = self.capture.read()
        if ok and frame is not None:
            self.failing_since = None
            return self._delivered(frame)
        now = time.perf_counter()
        if self.failing_since is None:
            self.failing_since = now
        elif now - self.failing_since >= self.lost_after:
            self.capture.release()
            self.failing_since = None
            self.lost_since = now
            self.reopener = CameraReopener(self.open_capture, self.retry_seconds)
        return None

    def _delivered(self, frame: np.ndarray) -> np.ndarray:
        """Track the real resolution and frame rate, shown in the panel."""
        now = time.perf_counter()
        if self.last_frame_at is not None and now > self.last_frame_at:
            rate = 1.0 / (now - self.last_frame_at)
            self.fps = rate if self.fps is None else 0.9 * self.fps + 0.1 * rate
        self.last_frame_at = now
        self.frame_size = (frame.shape[1], frame.shape[0])
        return frame

    def reopen(self, open_capture: Callable[[], cv2.VideoCapture] | None = None) -> None:
        """Let the webcam go and open it (or another one) again in the background."""
        if open_capture is not None:
            self.open_capture = open_capture
        if self.reopener is not None:
            self.reopener.abandon()
        else:
            self.capture.release()
        self.failing_since = None
        self.lost_since = time.perf_counter()
        self.opening = True
        self.reopener = CameraReopener(self.open_capture, self.retry_seconds)

    def open_settings(self) -> bool:
        """The driver's own settings dialog (exposure, focus, white balance); DirectShow only."""
        if self.reopener is not None or platform.system() != "Windows":
            return False
        return bool(self.capture.set(cv2.CAP_PROP_SETTINGS, 1))

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
