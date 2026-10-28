"""The webcam: opening it, reopening it when it drops, and the card shown meanwhile."""

from __future__ import annotations

import platform
import subprocess
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
    path: str | None = None  # The system's device path: finds its driver.


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
            cameras[info.index] = CameraInfo(info.index, info.name, getattr(info, "path", None))
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


# pnputil's exit codes besides success; the rest are reported as numbers.
DRIVER_CANCELLED = 1223  # The Windows permission prompt (UAC) was declined.
DRIVER_REBOOT = 3010


def device_instance_id(path: str | None) -> str | None:
    r"""The Windows device path of a webcam as the instance id pnputil takes.

    \\?\usb#vid_090c&pid_b371&mi_00#6&341aedb9&0&0000#{65e8773d-...}\global
    becomes USB\VID_090C&PID_B371&MI_00\6&341AEDB9&0&0000: bus, hardware id and
    instance, without the interface GUID that follows them.
    """
    if not path:
        return None
    parts = path.removeprefix("\\\\?\\").split("#")
    if len(parts) < 4 or not parts[0]:
        return None
    return "\\".join(parts[:3]).upper()


def restart_driver(instance_id: str) -> str | None:
    """Restart a webcam's driver (Windows, asks for permission); None when it worked, else why not.

    Fixes a webcam stuck on a black image that reopening alone does not bring back.
    """
    if platform.system() != "Windows":
        return t("camera.driver_windows_only")
    # pnputil needs an administrator: Start-Process -Verb RunAs shows the UAC prompt.
    script = (
        "try { $p = Start-Process pnputil -ArgumentList '/restart-device', "
        f"'\"{instance_id}\"' -Verb RunAs -WindowStyle Hidden -Wait -PassThru; exit $p.ExitCode }} "
        f"catch {{ exit {DRIVER_CANCELLED} }}"
    )
    try:
        code = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).returncode
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)
    if code == 0:
        return None
    if code == DRIVER_CANCELLED:
        return t("camera.driver_cancelled")
    if code == DRIVER_REBOOT:
        return t("camera.driver_reboot")
    return t("camera.driver_exit", code=code)


class FrameReader:
    """Read the webcam in its own thread: a slow or stuck webcam must not freeze the window.

    Wraps a cv2.VideoCapture. read() never waits: (True, frame) for an image not handed
    out yet, (True, None) while the next one is on its way, (False, None) when the webcam
    failed or has sent nothing for stall_seconds. Some webcams take a second per image in
    the dark; reading them on the video thread would stall clicks and keys for as long.
    """

    def __init__(self, capture: cv2.VideoCapture, stall_seconds: float = 2.0):
        self.capture = capture
        self.stall_seconds = stall_seconds
        self.lock = threading.Lock()
        self.latest: np.ndarray | None = None
        self.failed = False
        self.last_frame_at = time.monotonic()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera-read", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stopping.is_set():
                ok, frame = self.capture.read()
                with self.lock:
                    if ok and frame is not None:
                        self.latest, self.failed = frame, False
                        self.last_frame_at = time.monotonic()
                    else:
                        self.failed = True
                if not ok:
                    self.stopping.wait(0.01)  # A failing webcam answers at once: do not spin.
        finally:
            # Only this thread touches the capture, so it is released after its last read.
            self.capture.release()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            frame, self.latest = self.latest, None
            if frame is not None:
                return True, frame
            stalled = time.monotonic() - self.last_frame_at > self.stall_seconds
            return not (self.failed or stalled), None

    def set(self, prop: int, value: float) -> bool:
        return self.capture.set(prop, value)

    def release(self, wait: float = 1.0) -> None:
        """Stop reading; waits a little so the webcam is free before it is opened again."""
        self.stopping.set()
        self.thread.join(timeout=wait)


class CameraReopener:
    """Reopen the webcam in a thread: opening can block for seconds, the video must not."""

    def __init__(self, open_capture: Callable[[], cv2.VideoCapture], retry_seconds: float,
                 prepare: Callable[[], None] | None = None):
        self.open_capture = open_capture
        self.retry_seconds = retry_seconds
        # Runs once before the first attempt, still off the video thread (a driver restart).
        self.prepare = prepare
        self.preparing = prepare is not None
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.result: tuple[cv2.VideoCapture, np.ndarray] | None = None
        self.thread = threading.Thread(target=self._run, name="camera-reopen", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        if self.prepare is not None:
            try:
                self.prepare()
            finally:
                self.preparing = False
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

    read() returns None instead of failing: when no new image is ready, for a short grace
    period after a failed read (a single dropped frame is common), then while
    CameraReopener retries. The first open stays synchronous, so a missing webcam at
    startup is still reported as an error. wrap (FrameReader in the app) is applied to
    every capture once it has sent its first image.
    """

    def __init__(self, open_capture: Callable[[], cv2.VideoCapture],
                 lost_after: float = CAMERA_LOST_SECONDS, retry_seconds: float = CAMERA_RETRY_SECONDS,
                 wrap: Callable[[cv2.VideoCapture], cv2.VideoCapture] | None = None):
        self.open_capture = open_capture
        self.lost_after = lost_after
        self.retry_seconds = retry_seconds
        self.wrap = wrap or (lambda capture: capture)
        self.capture = self.wrap(open_capture())
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

    @property
    def restarting(self) -> bool:
        """The driver is being restarted, before the webcam is opened again."""
        return self.reopener is not None and self.reopener.preparing

    def read(self) -> np.ndarray | None:
        if self.reopener is not None:
            reopened = self.reopener.take()
            if reopened is None:
                return None
            capture, frame = reopened
            self.capture = self.wrap(capture)
            self.reopener = None
            self.lost_since = None
            self.opening = False
            self.fps = self.last_frame_at = None
            return self._delivered(frame)
        ok, frame = self.capture.read()
        if ok:
            # Alive; without a frame it is just not ready yet (FrameReader).
            self.failing_since = None
            return self._delivered(frame) if frame is not None else None
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

    def reopen(self, open_capture: Callable[[], cv2.VideoCapture] | None = None,
               prepare: Callable[[], None] | None = None) -> None:
        """Let the webcam go and open it (or another one) again in the background.

        prepare runs first in the background, once the webcam is released (a driver restart).
        """
        if open_capture is not None:
            self.open_capture = open_capture
        if self.reopener is not None:
            self.reopener.abandon()
        else:
            self.capture.release()
        self.failing_since = None
        self.lost_since = time.perf_counter()
        self.opening = True
        self.reopener = CameraReopener(self.open_capture, self.retry_seconds, prepare)

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
