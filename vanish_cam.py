"""Webcam locale: dissolvenza avviata dal gesto con indice e medio a V."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .naruto_effect import (
        EFFECT_SECONDS, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from .vanish_sound import SOUND_PATH, VanishSound
    from .goku_effect import (
        EFFECT_SECONDS as GOKU_EFFECT_SECONDS, SOUND_NAME as GOKU_SOUND_NAME,
        HoldTracker, face_reference, forehead_fingers_pose, make_speed_lines, sound_path,
        render_teleport, teleport_region,
    )
except ImportError:
    from naruto_effect import (
        EFFECT_SECONDS, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from vanish_sound import SOUND_PATH, VanishSound
    from goku_effect import (
        EFFECT_SECONDS as GOKU_EFFECT_SECONDS, SOUND_NAME as GOKU_SOUND_NAME,
        HoldTracker, face_reference, forehead_fingers_pose, make_speed_lines, sound_path,
        render_teleport, teleport_region,
    )


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "Webcam - Anteprima"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = APP_DIR / "hand_landmarker.task"
FACE_MODEL = APP_DIR / "face_landmarker.task"

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
}

CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
VANISH_SECONDS = 3.6
GESTURE_INTERVAL_SECONDS = 0.14
GOKU_INTERVAL_SECONDS = 0.066


@dataclass
class AppState:
    phase: str = "calibrating"
    calibration_started: float = 0.0
    vanish_started: float | None = None
    last_gesture_check: float = 0.0
    fullscreen: bool = False


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("Gli FPS devono essere compresi tra 1 e 120.")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scegli Normal Vanish, Naruto Vanish o Goku per la webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Indice della webcam")
    # 1280x720 e' il 16:9 nativo della maggior parte delle webcam (via MJPG):
    # le videochiamate non devono stirare un 4:3.
    parser.add_argument("--width", type=int, default=1280, help="Larghezza video")
    parser.add_argument("--height", type=int, default=720, help="Altezza video")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Avvia a schermo intero"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Non specchiare la webcam"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="FPS massimi dell'uscita virtuale (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Richiede la webcam virtuale; per default si attiva se disponibile. --no-virtual-camera la disattiva",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help="Driver della webcam virtuale da usare",
    )
    parser.add_argument("--virtual-device", help="Nome del dispositivo virtuale da usare")
    parser.add_argument(
        "--mode", choices=("normal", "naruto", "goku"),
        help="Salta il menu iniziale e avvia direttamente la modalità scelta",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Stampa nel terminale lo stato del riconoscimento (senza scritte nel video)",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def choose_mode() -> str | None:
    """Simple keyboard/mouse menu shown before the clean camera feed."""
    title = "Scegli effetto"
    selected = [None]
    window = np.zeros((540, 700, 3), dtype=np.uint8)
    window[:] = (26, 29, 34)
    cv2.putText(window, "SCEGLI IL TUO VANISH", (68, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (236, 239, 242), 2, cv2.LINE_AA)
    buttons = (
        ("normal", "1  NORMAL VANISH", "Segno a V  /  dissolvenza", 140),
        ("naruto", "2  NARUTO VANISH", "Mani unite  /  tronco e fumo", 260),
        ("goku", "3  GOKU", "Due dita sulla fronte per 1s  /  teletrasporto", 380),
    )
    for mode, heading, detail, top in buttons:
        cv2.rectangle(window, (65, top), (635, top + 94), (48, 54, 62), -1, cv2.LINE_AA)
        cv2.rectangle(window, (65, top), (635, top + 94), (111, 142, 153), 2, cv2.LINE_AA)
        cv2.putText(window, heading, (90, top + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.77, (240, 245, 247), 2, cv2.LINE_AA)
        cv2.putText(window, detail, (90, top + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (179, 195, 201), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP and 65 <= x <= 635:
            for mode, _, _, top in buttons:
                if top <= y <= top + 94:
                    selected[0] = mode

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(title, window)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("1"):
                selected[0] = "normal"
            elif key == ord("2"):
                selected[0] = "naruto"
            elif key == ord("3"):
                selected[0] = "goku"
            elif key in (ord("q"), 27):
                return None
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)


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
        message = (
            "Webcam virtuale non disponibile. Installa OBS Studio (con il driver "
            "OBS Virtual Camera) oppure Unity Capture. Se usi OBS Virtual Camera, "
            "ferma 'Avvia fotocamera virtuale' in OBS. "
            "Avvia da AVVIA_LOCALE.bat per installare pyvirtualcam. "
            f"Dettaglio: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Webcam] {message}\n[Webcam] Proseguo con l'anteprima locale.")
        return None
    print(f"[Webcam] Uscita virtuale attiva: {camera.device} ({width}x{height}, {args.fps:g} FPS massimi)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def download_models(mode: str = "normal") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if mode in ("naruto", "goku") and destination == GESTURE_MODEL:
            continue
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[Vanish Lab] Scarico {destination.name}...")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Vanish-Lab/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = int(received * 100 / total)
                            print(f"\r  {percent:3d}%", end="", flush=True)
            print()
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Non riesco a scaricare i modelli MediaPipe. "
                "Controlla la connessione internet e riprova."
            ) from error


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} non disponibile. Chiudi le altre app oppure prova --camera 1."
        )

    # Senza MJPG molte webcam USB danno 1280x720 solo a 10 FPS (YUYV) e
    # OpenCV ripiega su 640x480.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def create_tasks(mode: str = "normal") -> tuple[
    vision.ImageSegmenter, MultiViewGestureRecognizer | MultiViewHandLandmarker | GokuPoseDetector
]:
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    if mode == "goku":
        for model in (HAND_MODEL, FACE_MODEL):
            if not model.is_file():
                raise RuntimeError(f"Modello non trovato: {model}")
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
        try:
            face = vision.FaceLandmarker.create_from_options(face_options)
            try:
                hands = vision.HandLandmarker.create_from_options(hand_options)
            except Exception:
                face.close()
                raise
        except Exception:
            segmenter.close()
            raise
        return segmenter, GokuPoseDetector(face, hands)
    if mode == "naruto":
        if not HAND_MODEL.is_file():
            raise RuntimeError(f"Modello delle mani non trovato: {HAND_MODEL}")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        task_class = vision.HandLandmarker
        wrapper_class = MultiViewHandLandmarker
    else:
        classifier_options = mp.tasks.components.processors.ClassifierOptions(
            score_threshold=0.55,
            category_allowlist=["Victory"],
        )
        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.5,
            canned_gesture_classifier_options=classifier_options,
        )
        task_class = vision.GestureRecognizer
        wrapper_class = MultiViewGestureRecognizer

    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        primary = task_class.create_from_options(options)
        try:
            alternate = task_class.create_from_options(
                replace(options, running_mode=vision.RunningMode.IMAGE)
            )
        except Exception:
            primary.close()
            raise
    except Exception:
        segmenter.close()
        raise
    return segmenter, wrapper_class(primary, alternate)


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field, 0.0, 1.0), fine_noise


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3 = alpha[:, :, None]
    return (
        person.astype(np.float32) * alpha_3
        + background.astype(np.float32) * (1.0 - alpha_3)
    ).astype(np.uint8)


def apply_dissolve(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    progress: float,
    field: np.ndarray,
    sparkle_noise: np.ndarray,
) -> np.ndarray:
    if progress <= 0.002:
        return alpha_composite(frame, background, mask)
    if progress >= 0.999:
        return background.copy()

    points = cv2.findNonZero((mask > 0.01).astype(np.uint8))
    if points is None:
        return background.copy()
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x - 18), max(0, y - 18)
    right = min(mask.shape[1], x + width + 18)
    bottom = min(mask.shape[0], y + height + 18)
    area = np.s_[top:bottom, left:right]
    mask = mask[area]
    field = field[area]
    sparkle_noise = sparkle_noise[area]
    eased = ease_in_out_cubic(progress)
    survival = smoothstep(eased - 0.055, eased + 0.055, field)
    alpha = mask * survival
    output = alpha_composite(frame[area], background[area], alpha)
    edge = (
        (np.abs(field - eased) < 0.018)
        & (mask > 0.48)
        & (sparkle_noise > 0.54)
    ).astype(np.float32)
    glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=5.0, sigmaY=5.0)
    glow = np.clip(glow * 5.0 + edge, 0.0, 1.0)[:, :, None]
    cyan = np.full_like(output, (255, 228, 92), dtype=np.uint8)
    result = np.clip(
        output.astype(np.float32) * (1.0 - glow * 0.8)
        + cyan.astype(np.float32) * glow * 0.8,
        0,
        255,
    ).astype(np.uint8)
    canvas = background.copy()
    canvas[area] = result
    return canvas


def victory_landmark_score(points: np.ndarray) -> float:
    """Riconosci indice e medio distesi usando distanze 3D, non l'orientamento."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP e punta: indice, medio, anulare, mignolo.
    for index, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    # Di lato le dita ripiegate possono apparire dritte ma molto corte:
    # non pretendere articolazioni ben visibili per anulare e mignolo.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # Le due punte devono essere separate; il pollice puo' restare nascosto.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Qualita' geometrica delle dita distese, non probabilita' del classificatore.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # I landmark vengono restituiti anche quando il classificatore non riconosce
    # "Victory" (ad esempio con il dorso della mano rivolto alla webcam).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # Controlla SEMPRE anche la geometria visibile: la profondita' stimata
    # puo' essere imprecisa quando palmo e dita piegate sono nascosti.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


class MultiViewGestureRecognizer:
    """Riprova la rilevazione su fotogrammi ruotati senza alterare il tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(image, timestamp_ms)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # IMAGE forza una nuova rilevazione; il modello VIDEO resta sul frame
        # originale. Evita timestamp/tracking incoerenti tra rotazioni diverse.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # Il palmo nella posa laterale e' spesso troppo piccolo nel frame
        # completo. I due ritagli inferiori si sovrappongono e coprono entrambe
        # le mani, anche quando l'immagine e' specchiata.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Spread the expensive IMAGE attempts over successive video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(crops[crop], rotation)),
            )
            result = self.alternate.recognize(rotated)
            candidate = classify_gesture_result(result, rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
        return best

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class MultiViewHandLandmarker:
    """Track two hands in video and retry difficult views with the hand model."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.last_hand_count = 0
        self.retry_cursor = 0

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            self.preferred_view = None
            return True
        if timestamp_ms < self.next_retry_ms:
            return False

        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        views = {
            "full": rgb,
            "center": rgb[int(height * 0.18):, int(width * 0.12):int(width * 0.88)],
            "lower_left": rgb[int(height * 0.22):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.22):, int(width * 0.22):],
        }
        attempts = [("full", rotation) for rotation in (1, 3, 2)]
        attempts += [("center", rotation) for rotation in (0, 1, 3)]
        attempts += [("lower_left", 0), ("lower_right", 0)]
        selected = []
        if self.preferred_view in attempts:
            selected.append(self.preferred_view)
        while len(selected) < 2:
            view = attempts[self.retry_cursor % len(attempts)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(views[crop], rotation)),
            )
            result = self.alternate.detect(rotated)
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            if joined_hands_from_landmarks(result.hand_landmarks, rotated.width, rotated.height):
                self.preferred_view = (crop, rotation)
                self.next_retry_ms = timestamp_ms + 90
                return True
        self.preferred_view = None
        self.next_retry_ms = timestamp_ms + 80
        return False

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class GokuPoseDetector:
    """Fronte dal Face Landmarker e dita dal Hand Landmarker sullo stesso frame."""

    FACE_MEMORY_MS = 1000

    def __init__(self, face, hands):
        self.face = face
        self.hands = hands
        # Le dita sulla fronte coprono parte del viso: ricorda l'ultima fronte vista.
        self.last_face = None
        self.last_face_ms = -self.FACE_MEMORY_MS
        self.face_visible = False
        self.last_hand_count = 0

    def recognize_pose(self, image: mp.Image, timestamp_ms: int) -> bool:
        faces = self.face.detect_for_video(image, timestamp_ms)
        self.face_visible = bool(faces.face_landmarks)
        if self.face_visible:
            self.last_face = face_reference(faces.face_landmarks[0], image.width, image.height)
            self.last_face_ms = timestamp_ms
        elif timestamp_ms - self.last_face_ms > self.FACE_MEMORY_MS:
            self.last_face = None
        result = self.hands.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        return any(
            forehead_fingers_pose(hand, self.last_face, image.width, image.height)
            for hand in result.hand_landmarks
        )

    def close(self) -> None:
        try:
            self.face.close()
        finally:
            self.hands.close()


class BackgroundPose:
    """Riconosce la posa Goku in un thread: il video non aspetta i ~30 ms dei modelli."""

    def __init__(self, detector: GokuPoseDetector):
        self.detector = detector
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.pending: tuple[np.ndarray, int] | None = None
        self.result: tuple[bool, int] | None = None
        self.error: Exception | None = None
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="goku-pose", daemon=True)
        self.thread.start()

    @property
    def busy(self) -> bool:
        with self.lock:
            return self.pending is not None

    def submit(self, rgb: np.ndarray, timestamp_ms: int) -> None:
        with self.lock:
            self.pending = (rgb, timestamp_ms)
        self.wake.set()

    def take_result(self) -> tuple[bool, int] | None:
        """Ultimo risultato non ancora letto, oppure None."""
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"Riconoscimento della posa fallito: {self.error}") from self.error
            result, self.result = self.result, None
            return result

    def _run(self) -> None:
        while True:
            self.wake.wait()
            self.wake.clear()
            if self.stopping:
                return
            with self.lock:
                job = self.pending
            if job is None:
                continue
            rgb, timestamp_ms = job
            try:
                posed = self.detector.recognize_pose(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
            except Exception as error:
                with self.lock:
                    self.error = error
                    self.pending = None
                return
            with self.lock:
                self.result = (posed, timestamp_ms)
                self.pending = None

    def close(self) -> None:
        self.stopping = True
        self.wake.set()
        self.thread.join(timeout=2.0)
        self.detector.close()


def recognize_gesture(
    recognizer: MultiViewGestureRecognizer,
    image: mp.Image,
    timestamp_ms: int,
) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


def start_vanish_on_gesture(state: AppState, name: str, score: float, now: float) -> bool:
    """Avvia la dissolvenza al primo riconoscimento affidabile della V."""
    if state.phase == "ready" and name == "Victory" and score >= 0.62:
        state.phase = "vanishing"
        state.vanish_started = now
        return True
    return False


def start_naruto_on_join(state: AppState, joined: bool, now: float) -> bool:
    if state.phase == "ready" and joined:
        state.phase = "naruto_vanishing"
        state.vanish_started = now
        return True
    return False


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    required = (SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL, SOUND_PATH)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Risorse mancanti: {', '.join(missing)}")
    load_log()
    import pygame.mixer  # The distributed build must contain its MP3 decoder.
    for mode in ("normal", "naruto", "goku"):
        segmenter, recognizer = create_tasks(mode)
        try:
            recognizer.close()
        finally:
            segmenter.close()
    print("[Webcam] Self-test completato: modelli, asset e audio presenti.")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    mode = args.mode or choose_mode()
    if mode is None:
        return 0
    with ExitStack() as resources:
        return run_camera(args, resources, mode)


def run_camera(args: argparse.Namespace, resources: ExitStack, mode: str) -> int:
    download_models(mode)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, recognizer = create_tasks(mode)
    resources.callback(segmenter.close)
    if mode == "goku":
        recognizer = BackgroundPose(recognizer)
    resources.callback(recognizer.close)
    log_sprite = load_log() if mode == "naruto" else None
    vanish_sound = VanishSound() if mode == "naruto" else None
    if mode == "goku":
        teleport_sound = sound_path()
        if teleport_sound is not None:
            vanish_sound = VanishSound(teleport_sound)
            print(f"[Goku] Suono: {teleport_sound.name}")
        else:
            print(f"[Goku] Nessun suono: aggiungi {GOKU_SOUND_NAME} in assets/ per attivarlo.")
    if vanish_sound is not None:
        resources.callback(vanish_sound.close)
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    state = AppState(
        calibration_started=time.perf_counter(), fullscreen=args.fullscreen
    )
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    background_samples: list[np.ndarray] = []
    last_mask: np.ndarray | None = None
    last_mask_at = float("-inf")
    dissolve_field: np.ndarray | None = None
    sparkle_noise: np.ndarray | None = None
    log_anchor: tuple[int, int, int] | None = None
    log_cache: dict[int, np.ndarray] = {}
    vanished_frame: np.ndarray | None = None
    goku_hold = HoldTracker()
    goku_lines: np.ndarray | None = None
    goku_region: np.ndarray | None = None
    goku_countdown: int | None = None
    virtual_camera = None
    virtual_camera_initialized = False
    calibration_remaining = None
    last_gesture_report = 0.0
    started = time.perf_counter()
    print("[Webcam] Esci dall'inquadratura per 4 secondi: acquisisco lo sfondo.")
    if mode == "goku":
        print("[Goku] Indice e medio uniti sulla fronte per 1 secondo: sparisci. Ripeti per riapparire.")
        print("[Webcam] Comandi nella finestra: SPAZIO nuovo sfondo, F fullscreen, Q esci.")
    else:
        print("[Webcam] Comandi nella finestra: SPAZIO nuovo sfondo, R riappari, V svanisci, F fullscreen, Q esci.")

    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("La webcam ha smesso di restituire immagini.")

            if not args.no_mirror:
                camera_frame = cv2.flip(camera_frame, 1)

            height, width = camera_frame.shape[:2]
            if not virtual_camera_initialized:
                # La webcam puo' non supportare la risoluzione chiesta: adatta la
                # finestra a quella reale, senza stirare l'immagine.
                print(f"[Webcam] Risoluzione: {width}x{height}")
                if (width, height) != (args.width, args.height):
                    cv2.resizeWindow(WINDOW_NAME, width, height)
                virtual_camera = open_virtual_camera(args, width, height)
                virtual_camera_initialized = True
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
            if mode == "normal" and (dissolve_field is None or dissolve_field.shape != (height, width)):
                dissolve_field, sparkle_noise = make_dissolve_fields(height, width)
            if mode == "goku" and (goku_lines is None or goku_lines.shape[1] != width):
                goku_lines = make_speed_lines(width)
            now = time.perf_counter()

            elapsed_ms = int((now - started) * 1000)
            mp_image = None

            if state.phase == "calibrating":
                elapsed = now - state.calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining:
                    print(f"[Webcam] Acquisizione sfondo: {remaining} secondi.")
                    calibration_remaining = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    background_samples.append(camera_frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    if background_samples:
                        stack = np.stack(background_samples[-18:], axis=0)
                        background = np.median(stack, axis=0).astype(np.uint8)
                    else:
                        background = camera_frame.copy()
                    background_samples.clear()
                    state.phase = "ready"
                    print("[Webcam] Sfondo pronto. Puoi rientrare nell'inquadratura.")
                display = camera_frame

            elif background is None:
                state.phase = "calibrating"
                state.calibration_started = now
                display = camera_frame

            elif mode == "goku":
                # Il gesto e' l'unico comando: resta attivo anche da invisibili.
                if (
                    state.phase in ("ready", "vanished")
                    and now - state.last_gesture_check >= GOKU_INTERVAL_SECONDS
                    and not recognizer.busy
                ):
                    state.last_gesture_check = now
                    # Viso e mani non richiedono il 720p: meta' risoluzione basta.
                    small = camera_frame if width <= 800 else cv2.resize(
                        camera_frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA
                    )
                    recognizer.submit(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), elapsed_ms)
                pose_result = recognizer.take_result()
                if pose_result is not None and state.phase in ("ready", "vanished"):
                    posed = pose_result[0]
                    fired = goku_hold.update(posed, now)
                    remaining = goku_hold.remaining(now)
                    countdown = None if remaining is None else int(np.ceil(remaining * 4))
                    if countdown is not None and countdown != goku_countdown:
                        print(f"[Goku] Tieni la posa: {remaining:.2f}s")
                    goku_countdown = countdown
                    if args.gesture_debug and now - last_gesture_report >= 1.0:
                        print(f"[Goku] viso={recognizer.detector.face_visible}, "
                              f"mani={recognizer.detector.last_hand_count}, posa={posed}")
                        last_gesture_report = now
                    if fired:
                        state.phase = "goku_vanishing" if state.phase == "ready" else "goku_appearing"
                        state.vanish_started = now
                        goku_region = None
                        last_mask = None
                        if vanish_sound is not None:
                            vanish_sound.play()

                if state.phase in ("goku_vanishing", "goku_appearing"):
                    if last_mask is None or now - last_mask_at >= 0.05:
                        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        last_mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                        last_mask_at = now
                    if goku_region is None:
                        goku_region = teleport_region(last_mask)
                    age = now - state.vanish_started
                    progress = min(1.0, age / GOKU_EFFECT_SECONDS)
                    vanishing = state.phase == "goku_vanishing"
                    display = render_teleport(
                        camera_frame, background, last_mask, goku_region, goku_lines,
                        progress if vanishing else 1.0 - progress, age,
                    )
                    if progress >= 1.0:
                        state.phase = "vanished" if vanishing else "ready"
                        state.vanish_started = None
                        last_mask = None
                        last_mask_at = float("-inf")
                        if vanishing:
                            print("[Goku] Sparito. Abbassa la mano, poi ripeti il gesto per riapparire.")
                elif state.phase == "vanished":
                    display = background
                else:
                    display = camera_frame

            elif state.phase == "vanished":
                if vanished_frame is None:
                    vanished_frame = background.copy()
                    if mode == "naruto" and log_anchor is not None:
                        overlay_log(vanished_frame, log_sprite, landed_log_anchor(log_anchor, height),
                                    cache=log_cache)
                display = vanished_frame

            else:
                mask = last_mask
                if (
                    state.phase == "ready"
                    and now - state.last_gesture_check >=
                    (0.08 if mode == "naruto" else GESTURE_INTERVAL_SECONDS)
                ):
                    state.last_gesture_check = now
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if mode == "normal":
                        name, score = recognize_gesture(recognizer, mp_image, elapsed_ms)
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(
                                f"[Gesto] {name}, valore={score:.2f}, "
                                f"ritaglio={recognizer.preferred_crop or 'full'}, "
                                f"rotazione={recognizer.preferred_rotation or 0}"
                            )
                            last_gesture_report = now
                        start_vanish_on_gesture(state, name, score, now)
                    else:
                        joined = recognizer.recognize_joined_hands(mp_image, elapsed_ms)
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(f"[Mani] rilevate={recognizer.last_hand_count}, "
                                  f"unite={joined}, vista={recognizer.preferred_view or 'full'}")
                            last_gesture_report = now
                        if start_naruto_on_join(state, joined, time.perf_counter()):
                            vanish_sound.play()

                if state.phase in ("vanishing", "naruto_vanishing"):
                    # Normal dissolve tracks the silhouette at 20 Hz; Naruto
                    # needs it only while the person is still visible.
                    needs_mask = (
                        (mode == "normal" and (mask is None or now - last_mask_at >= 0.05))
                        or (mode == "naruto" and (log_anchor is None or now - state.vanish_started < 0.13))
                    )
                    if needs_mask:
                        if mp_image is None:
                            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                        last_mask = mask
                        last_mask_at = now
                    if mode == "naruto" and log_anchor is None:
                        log_anchor = substitution_anchor(mask)

                if mode == "naruto" and state.phase == "naruto_vanishing":
                    age = max(0.04, time.perf_counter() - state.vanish_started)
                    display = render_substitution(
                        camera_frame, background, mask, log_sprite, log_anchor, age, log_cache
                    )
                    if age >= EFFECT_SECONDS:
                        state.phase = "vanished"
                        vanished_frame = display
                elif state.phase == "ready":
                    display = camera_frame
                else:
                    progress = 0.0
                    if state.phase == "vanishing" and state.vanish_started is not None:
                        progress = min(
                            1.0, (time.perf_counter() - state.vanish_started) / VANISH_SECONDS
                        )
                        if progress >= 1.0:
                            state.phase = "vanished"
                    display = apply_dissolve(
                        camera_frame, background, mask, progress, dissolve_field, sparkle_noise
                    )
                    if state.phase == "vanished":
                        vanished_frame = display

            if virtual_camera is not None:
                send_virtual_frame(virtual_camera, display)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                state.phase = "calibrating"
                state.calibration_started = now
                state.vanish_started = None
                log_anchor = None
                last_mask = None
                last_mask_at = float("-inf")
                vanished_frame = None
                goku_hold = HoldTracker()
                goku_region = None
                background_samples.clear()
                calibration_remaining = None
                print("[Webcam] Esci dall'inquadratura per 4 secondi: acquisisco il nuovo sfondo.")
            elif mode == "goku" and key in (ord("r"), ord("v")):
                pass  # In modalita' Goku si sparisce e si riappare solo col gesto.
            elif key == ord("r"):
                if background is not None:
                    state.phase = "ready"
                    state.vanish_started = None
                    vanished_frame = None
                    log_anchor = None
                    last_mask = None
                    last_mask_at = float("-inf")
            elif key == ord("v") and background is not None:
                state.phase = "naruto_vanishing" if mode == "naruto" else "vanishing"
                state.vanish_started = time.perf_counter()
                if mode == "naruto":
                    vanish_sound.play()
                    log_anchor = None
                vanished_frame = None
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                set_fullscreen(state.fullscreen)
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
    finally:
        resources.close()

    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[Vanish Lab] ERRORE: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
