"""VanishCam: the video loop, the active effect, and the keys and panel actions."""

from __future__ import annotations

import argparse
import platform
import queue
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .background import fit_background, load_background_image, replace_background, update_background
from .call_audio import CallSound, VirtualMic, open_virtual_mic
from .camera import (
    CameraFeed, CameraInfo, device_instance_id, dim_for_card, list_cameras, open_camera, reconnecting_card,
    restart_driver,
)
from .cli import parse_args
from .control_panel import ACCENTS, AMBER, FONT_DIR, FONTS, GREEN, MODE_CARDS, ControlPanel, PanelView
from .detectors import AsyncRecognizer, create_recognizer, detection_function, detector_status
from .effects.goku import SOUND_NAME as GOKU_SOUND_NAME, sound_path
from .effects.naruto import load_log
from .i18n import t
from .models import FACE_MODEL, GESTURE_MODEL, HAND_MODEL, SEGMENTER_MODEL, download_models
from .modes import MODES, FrameInfo, Mode
from .segmentation import PersonSegmenter, create_segmenter
from .sound import SOUND_PATH, VanishSound
from .user_data import Settings, UserData
from .virtual_camera import open_virtual_camera, send_virtual_frame

WINDOW_NAME = "VanishCam - Video"

CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
BACKGROUND_UPDATE_SECONDS = 0.5


@dataclass
class AppState:
    """Video loop state; each mode keeps its own effect phase."""

    phase: str = "calibrating"  # calibrating | running
    calibration_started: float = 0.0
    last_gesture_check: float = float("-inf")
    last_background_update: float = float("-inf")
    fullscreen: bool = False


def window_visible(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    fonts = tuple(FONT_DIR / names[0] for names in FONTS.values())
    required = (SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL, SOUND_PATH, *fonts)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(t("selftest.missing", files=", ".join(missing)))
    load_log()
    import pygame.mixer  # The distributed build must contain its MP3 decoder.
    import cv2_enumerate_cameras  # And the native helper that names the webcams in the panel.
    if sys.platform == "win32":
        import sounddevice  # And PortAudio, for the call microphone through VB-CABLE.
    create_segmenter().close()
    for mode in MODES:
        create_recognizer(mode).close()
    print(f"[VanishCam] {t('selftest.ok')}")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    # Closing the terminal (SIGHUP) or a kill (SIGTERM) become a normal exit:
    # the webcam, the windows and the virtual microphone are released.
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop_on_signal)
    with ExitStack() as resources:
        return run_camera(args, resources, args.mode)


def stop_on_signal(signum, frame) -> None:
    raise KeyboardInterrupt


def open_call_mic(args: argparse.Namespace, resources: ExitStack) -> VirtualMic | None:
    if not args.call_audio:
        return None
    try:
        mic = open_virtual_mic()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        # Video matters more than audio: warn and carry on without the virtual microphone.
        print(f"[VanishCam Mic] {t('mic.unavailable', error=error)}")
        return None
    resources.callback(mic.close)
    print(f"[VanishCam Mic] {t('mic.active', name=mic.description)}")
    return mic


def open_sound(mode: str, mic: VirtualMic | None = None):
    if mode == "naruto":
        path = SOUND_PATH
    elif mode == "goku":
        path = sound_path()
        if path is None:
            print(f"[Goku] {t('goku.no_sound', name=GOKU_SOUND_NAME)}")
            return None
        print(f"[Goku] {t('goku.sound', name=path.name)}")
    else:
        return None
    sound = VanishSound(path) if mode == "goku" else VanishSound()
    # With the virtual microphone the effect is heard in the call too.
    return CallSound(sound, mic, path) if mic is not None else sound


class EffectRuntime:
    """Recognizer, sound and effect of one mode: switching mode replaces only this.

    Camera, segmenter, virtual devices and background stay open across switches.
    """

    def __init__(self, name: str, segment: PersonSegmenter, mic: VirtualMic | None):
        download_models(name)
        self.name = name
        self.recognizer = create_recognizer(name)
        self.detector = AsyncRecognizer(detection_function(name, self.recognizer), self.recognizer.close)
        try:
            self.sound = open_sound(name, mic)
        except Exception:
            self.detector.close()
            raise
        self.mode: Mode = MODES[name](segment, self.sound)

    def close(self) -> None:
        try:
            self.detector.close()
        finally:
            if self.sound is not None:
                self.sound.close()


KEY_ACTIONS = {
    ord(" "): ("recapture",),
    ord("r"): ("reappear",),
    ord("v"): ("vanish",),
    ord("f"): ("fullscreen",),
    ord("d"): ("detach",),
    **{ord(str(number)): ("mode", card.key) for number, card in enumerate(MODE_CARDS, start=1)},
}
MESSAGE_SECONDS = 6.0


def camera_state(camera: CameraFeed) -> str:
    if not camera.reconnecting:
        return "live"
    if camera.restarting:
        return "restarting"
    return "opening" if camera.opening else "reconnecting"


def panel_status(
    phase: str, mode: Mode, calibration_remaining: int | None, camera: str = "live",
) -> tuple[str, tuple[int, int, int]]:
    if camera != "live":
        return t(f"status.{camera}"), AMBER
    if phase == "calibrating":
        return t("status.calibrating", seconds=calibration_remaining or 0), AMBER
    if mode.phase in ("vanishing", "vanished", "appearing"):
        return t(f"status.{mode.phase}"), ACCENTS.get(mode.name, GREEN)
    return t("status.ready"), GREEN


def choose_camera(requested: int | None, saved: str | None, cameras: tuple[CameraInfo, ...]) -> int:
    """--camera wins; then the webcam used last time, found by name (numbers shift when one is unplugged)."""
    if requested is not None:
        return requested
    for camera in cameras:
        if camera.name == saved:
            return camera.index
    return cameras[0].index if cameras else 0


def run_camera(args: argparse.Namespace, resources: ExitStack, mode_name: str | None = None) -> int:
    user_data = UserData.default()
    saved = user_data.load()
    mode_name = mode_name or (saved.mode if saved.mode in MODES else "normal")
    cameras = list_cameras()
    camera_index = choose_camera(getattr(args, "camera", None), saved.camera, cameras)
    # The video in its own window: asked for on the command line, or how it was left last time.
    detached = bool(getattr(args, "video_window", False) or saved.detached)

    def camera_name() -> str | None:
        return next((camera.name for camera in cameras if camera.index == camera_index), None)

    chosen: Path | None = None
    background_source: np.ndarray | None = None  # Chosen picture, at its original resolution.
    if getattr(args, "background_image", None) is not None:
        chosen = Path(args.background_image)
        background_source = load_background_image(chosen)
    elif saved.background is not None:
        try:
            background_source = load_background_image(saved.background)
            chosen = saved.background
        except RuntimeError as error:
            print(f"[VanishCam] {error}")
    user_data.save(Settings(mode_name, chosen, camera_name(), detached))
    gallery = tuple(user_data.gallery())

    download_models(mode_name)

    def open_capture() -> cv2.VideoCapture:
        # Reads camera_index when called: reopening after a switch opens the new webcam.
        capture = open_camera(camera_index, args.width, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        return capture

    camera = CameraFeed(open_capture)
    resources.callback(camera.release)
    segmenter = create_segmenter()
    resources.callback(segmenter.close)
    segment = PersonSegmenter(segmenter)
    mic = open_call_mic(args, resources)
    effect = EffectRuntime(mode_name, segment, mic)
    # Closes whichever effect is active at that point, also after a mode switch.
    resources.callback(lambda: effect.close())
    resources.callback(cv2.destroyAllWindows)
    panel = ControlPanel()
    resources.callback(panel.close)

    def apply_fullscreen(enabled: bool) -> None:
        # Detached, the video goes fullscreen in its own window; otherwise the app window shows only the video.
        if detached:
            set_fullscreen(enabled)
        else:
            panel.set_fullscreen(enabled)

    state = AppState(calibration_started=time.perf_counter(), fullscreen=args.fullscreen)

    background: np.ndarray | None = None
    real_background: np.ndarray | None = None  # Last real background: switching back needs no recalibration.
    background_samples: list[np.ndarray] = []
    virtual_camera = None
    virtual_camera_initialized = False
    calibration_remaining = None
    last_gesture_report = float("-inf")
    message, message_until = "", float("-inf")
    # Driver restarts finish off the video thread: their outcome (None or why it failed) comes back here.
    driver_results: queue.SimpleQueue[str | None] = queue.SimpleQueue()
    camera_detail, camera_detail_at = "", float("-inf")
    started = state.calibration_started
    width = height = 0

    def set_detached(enabled: bool) -> None:
        """Move the video to its own window (a second monitor, OBS's window capture) or back beside the panel."""
        nonlocal detached
        if state.fullscreen:
            state.fullscreen = False
            apply_fullscreen(False)
        detached = enabled
        if enabled:
            # GUI_NORMAL: just the video, without the Qt toolbar.
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, width or args.width, height or args.height)
        else:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass  # Already closed by the user.
        panel.set_detached(enabled)
        user_data.save(Settings(effect.name, chosen, camera_name(), detached))

    if detached:
        set_detached(True)
    if state.fullscreen:
        apply_fullscreen(True)
    if background_source is None:
        print(f"[VanishCam] {t('loop.step_out')}")
    else:
        # The picture is the background: no calibration.
        state.phase = "running"
        print(f"[VanishCam] {t('loop.background_file', path=chosen)}")
    print(f"[{effect.mode.title}] {effect.mode.instructions}")
    print(f"[VanishCam] {t('loop.panel_hint', count=len(MODE_CARDS))}")

    def notify(text: str, now: float) -> None:
        nonlocal message, message_until
        print(f"[VanishCam] {text}")
        message, message_until = text, now + MESSAGE_SECONDS

    def start_calibration(now: float) -> None:
        nonlocal calibration_remaining
        state.phase = "calibrating"
        state.calibration_started = now
        background_samples.clear()
        calibration_remaining = None
        effect.mode.reset()
        print(f"[VanishCam] {t('loop.step_out_again')}")

    def switch_mode(name: str) -> None:
        nonlocal effect
        if name == effect.name or name not in MODES:
            return
        previous = effect.name
        # Close first: the old sound owns the shared audio mixer.
        effect.close()
        try:
            effect = EffectRuntime(name, segment, mic)
        except Exception as error:
            effect = EffectRuntime(previous, segment, mic)
            notify(t("loop.mode_failed", name=name, error=error), time.perf_counter())
            return
        user_data.save(Settings(name, chosen, camera_name(), detached))
        print(f"[{effect.mode.title}] {effect.mode.instructions}")

    def reopen_camera(now: float) -> None:
        """Let the webcam go, restart its driver (Windows, after the permission prompt), open it again."""
        device = next((info for info in cameras if info.index == camera_index), None)
        instance = device_instance_id(device.path) if device is not None else None
        if instance is None or platform.system() != "Windows":
            camera.reopen()
            return
        notify(t("camera.driver_restarting"), now)
        camera.reopen(prepare=lambda: driver_results.put(restart_driver(instance)))

    def switch_camera(index: int, now: float) -> None:
        nonlocal camera_index
        if index == camera_index and not camera.reconnecting:
            return
        camera_index = index
        camera.reopen(open_capture)
        user_data.save(Settings(effect.name, chosen, camera_name(), detached))
        notify(t("camera.switched", name=camera_name() or t("panel.camera_number", index=index)), now)

    def choose_background(path: Path | None, now: float) -> None:
        nonlocal chosen, background_source, background
        if path is None:
            chosen, background_source = None, None
            if real_background is None:
                start_calibration(now)
            else:
                background = real_background
                effect.mode.reset()
        else:
            try:
                image = load_background_image(path)
            except RuntimeError as error:
                notify(str(error), now)
                return
            chosen, background_source = path, image
            background = fit_background(image, width, height)
            state.phase = "running"
            background_samples.clear()
            effect.mode.reset()
        user_data.save(Settings(effect.name, chosen, camera_name(), detached))

    def handle(action: tuple, now: float) -> None:
        nonlocal gallery, cameras
        kind = action[0]
        running = state.phase == "running"
        if kind == "mode":
            switch_mode(action[1])
        elif kind == "background":
            choose_background(action[1], now)
        elif kind == "import":
            try:
                load_background_image(action[1])
                imported = user_data.import_image(action[1])
            except (OSError, RuntimeError, ValueError) as error:
                notify(t("loop.image_not_added", error=error), now)
                return
            gallery = tuple(user_data.gallery())
            choose_background(imported, now)
        elif kind == "remove":
            user_data.remove_image(action[1])
            gallery = tuple(user_data.gallery())
            if chosen == action[1]:
                choose_background(None, now)
        elif kind == "error":
            notify(action[1], now)
        elif kind == "recapture" and background_source is None:
            start_calibration(now)
        elif kind == "reappear" and running and effect.mode.manual_keys:
            effect.mode.reappear()
        elif kind == "vanish" and running and effect.mode.manual_keys:
            effect.mode.vanish(now)
        elif kind == "fullscreen":
            state.fullscreen = not state.fullscreen
            apply_fullscreen(state.fullscreen)
        elif kind == "cameras":
            cameras = list_cameras()
        elif kind == "camera":
            switch_camera(action[1], now)
        elif kind == "reopen_camera":
            reopen_camera(now)
        elif kind == "camera_settings" and not camera.open_settings():
            notify(t("camera.settings_unavailable"), now)
        elif kind == "detach":
            set_detached(not detached)

    card_base: np.ndarray | None = None  # Last image sent, dimmed: background of the reconnecting card.
    card: tuple[tuple[int, bool], np.ndarray] | None = None
    last_display: np.ndarray | None = None
    try:
        while True:
            frame = camera.read()
            now = time.perf_counter()
            mode = effect.mode
            if frame is None and not camera.reconnecting:
                # A dropped frame, or the first ones after opening: wait for the next one.
                time.sleep(0.01)
                continue
            if frame is None:
                if card_base is None:
                    if not camera.opening:
                        print(f"[VanishCam] {t('camera.lost')}")
                    card_base = dim_for_card(last_display, width or args.width, height or args.height)
                lost = now - camera.lost_since
                # Redrawn only when the seconds or the blinking lamp change.
                seconds, lamp_on = int(lost), int(lost * 2) % 2 == 0
                title = t({"restarting": "reconnect.restarting_title", "opening": "reconnect.opening_title"}.get(
                    camera_state(camera), "reconnect.title"))
                if card is None or card[0] != (seconds, lamp_on, title):
                    detail = t("reconnect.detail", index=camera_index, seconds=seconds)
                    card = (seconds, lamp_on, title), reconnecting_card(card_base, title, detail, lamp_on)
                display = card[1]
            else:
                if card_base is not None:
                    print(f"[VanishCam] {t('camera.back')}")
                    card_base = card = None
                    if state.phase == "calibrating":
                        # The samples taken before the webcam was lost may be stale.
                        start_calibration(now)
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                if not virtual_camera_initialized:
                    height, width = frame.shape[:2]
                    # The webcam may not support the requested resolution: fit the
                    # window to the real one, without stretching the image.
                    print(f"[VanishCam] {t('camera.resolution', width=width, height=height)}")
                    if detached and (width, height) != (args.width, args.height):
                        cv2.resizeWindow(WINDOW_NAME, width, height)
                    virtual_camera = open_virtual_camera(args, width, height)
                    virtual_camera_initialized = True
                    if virtual_camera is not None:
                        resources.callback(virtual_camera.close)
                    if background_source is not None:
                        background = fit_background(background_source, width, height)
                elif frame.shape[:2] != (height, width):
                    # Reopened at another resolution: keep the one the virtual webcam and background use.
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                elapsed_ms = int((now - started) * 1000)
                camera_frame = frame
                if background_source is not None:
                    # The mask stays cached for this timestamp: the effects reuse it.
                    frame = replace_background(frame, background, segment(frame, elapsed_ms))

                if state.phase == "calibrating":
                    elapsed = now - state.calibration_started
                    remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                    if remaining != calibration_remaining:
                        print(f"[VanishCam] {t('loop.capturing', seconds=remaining)}")
                        calibration_remaining = remaining
                    if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                        background_samples.append(frame.copy())
                    if elapsed >= CALIBRATION_SECONDS:
                        if background_samples:
                            stack = np.stack(background_samples[-18:], axis=0)
                            background = np.median(stack, axis=0).astype(np.uint8)
                        else:
                            background = frame.copy()
                        background_samples.clear()
                        real_background = background
                        panel.set_real_background(background)
                        state.phase = "running"
                        mode.reset()
                        print(f"[VanishCam] {t('loop.background_ready')}")
                    display = frame

                else:
                    if (
                        mode.detecting
                        and now - state.last_gesture_check >= mode.detect_interval
                        and not effect.detector.busy
                    ):
                        state.last_gesture_check = now
                        effect.detector.submit(mode.detection_image(camera_frame), elapsed_ms)
                    detected = effect.detector.take_result()
                    if detected is not None and mode.detecting:
                        mode.on_detection(detected[0], now)
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(f"[Gesture] {detector_status(effect.recognizer, detected[0])}")
                            last_gesture_report = now
                    display = mode.render(FrameInfo(frame, background, now, elapsed_ms))

                    # The background follows light and small movements, even while invisible.
                    if (
                        not args.static_background
                        and background_source is None
                        and mode.phase in ("ready", "vanished")
                        and now - state.last_background_update >= BACKGROUND_UPDATE_SECONDS
                    ):
                        state.last_background_update = now
                        background = update_background(background, frame, segment(frame, elapsed_ms))
                        real_background = background
                        mode.background_changed()

                last_display = display

            if virtual_camera is not None:
                send_virtual_frame(virtual_camera, display)
            if detached:
                cv2.imshow(WINDOW_NAME, display)
            if now - camera_detail_at >= 1.0 and camera.frame_size is not None:
                # Once a second: a steadier number, and the panel is not redrawn every frame.
                camera_detail_at = now
                camera_detail = t("panel.camera_live", width=camera.frame_size[0], height=camera.frame_size[1],
                                  fps=round(camera.fps or 0))
            while not driver_results.empty():
                error = driver_results.get()
                notify(t("camera.driver_restarted") if error is None else t("camera.driver_failed", error=error), now)
            status, status_color = panel_status(state.phase, mode, calibration_remaining, camera_state(camera))
            panel.show(PanelView(
                mode=effect.name, status=status, status_color=status_color, background=chosen,
                gallery=gallery, manual_keys=mode.manual_keys, can_recapture=background_source is None,
                fullscreen=state.fullscreen, message=message if now < message_until else "",
                cameras=cameras, camera_index=camera_index, camera_detail=camera_detail,
                camera_state=camera_state(camera),
                can_adjust=platform.system() == "Windows", detached=detached,
            ), display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 and state.fullscreen:
                # Esc leaves fullscreen first, like everywhere else; Q still quits.
                key = ord("f")
            elif key in (ord("q"), 27) or panel.closed:
                break
            if detached and not window_visible(WINDOW_NAME):
                # Closing the video window brings the video back beside the panel; only the panel quits.
                set_detached(False)
            actions = panel.poll()
            if key in KEY_ACTIONS:
                actions.append(KEY_ACTIONS[key])
            for action in actions:
                handle(action, now)
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
            elif frame is None:
                time.sleep(1 / args.fps)
    finally:
        resources.close()

    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[VanishCam] {t('error', error=error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
