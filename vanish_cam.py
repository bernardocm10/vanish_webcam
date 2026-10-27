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
from typing import Callable
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .naruto_effect import joined_hands_from_landmarks, load_log
    from .vanish_sound import SOUND_PATH, VanishSound
    from .goku_effect import (
        SOUND_NAME as GOKU_SOUND_NAME, face_reference, forehead_fingers_pose, sound_path,
    )
    from .modes import MODES, FrameInfo, Mode
except ImportError:
    from naruto_effect import joined_hands_from_landmarks, load_log
    from vanish_sound import SOUND_PATH, VanishSound
    from goku_effect import (
        SOUND_NAME as GOKU_SOUND_NAME, face_reference, forehead_fingers_pose, sound_path,
    )
    from modes import MODES, FrameInfo, Mode

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
GESTURE_INTERVAL_SECONDS = 0.14
# Regolarizzazione del guided filter (scala 0-255 della guida): piu' basso = bordi piu' fini.
MASK_EDGE_EPS = 40.0
BACKGROUND_UPDATE_SECONDS = 0.5
BACKGROUND_UPDATE_RATE = 0.2


@dataclass
class AppState:
    """Video loop state; each mode keeps its own effect phase."""

    phase: str = "calibrating"  # calibrating | running
    calibration_started: float = 0.0
    last_gesture_check: float = float("-inf")
    last_background_update: float = float("-inf")
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
        "--static-background", action="store_true",
        help="Non aggiornare lo sfondo dopo la calibrazione (come nelle versioni precedenti)",
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
    if hasattr(cv2, "ximgproc"):
        # Il guided filter fa aderire la maschera ai bordi reali (capelli, cuffie)
        # ed elimina l'alone dello sfondo; a meta' risoluzione costa ~5 ms in 720p.
        half = (max(1, width // 2), max(1, height // 2))
        guide = cv2.resize(image.numpy_view(), half, interpolation=cv2.INTER_AREA)
        small = cv2.resize(raw, half, interpolation=cv2.INTER_LINEAR)
        small = cv2.ximgproc.guidedFilter(guide, small, 6, MASK_EDGE_EPS)
        resized = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
        resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


def update_background(
    background: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
    rate: float = BACKGROUND_UPDATE_RATE,
) -> np.ndarray:
    """Fondi il frame nello sfondo solo dove la persona e' sicuramente assente.

    Segue i cambi di luce senza ricalibrare. Il margine attorno alla sagoma
    evita che bordi incerti (mani, capelli) restino stampati nello sfondo.
    """
    height, width = mask.shape
    small_size = (max(1, width // 4), max(1, height // 4))
    occupied = (cv2.resize(mask, small_size, interpolation=cv2.INTER_AREA) > 0.05).astype(np.uint8)
    # Margine di almeno il 4% della larghezza attorno alla sagoma.
    grow = 2 * max(1, int(np.ceil(small_size[0] * 0.04))) + 1
    occupied = cv2.dilate(occupied, np.ones((grow, grow), np.uint8))
    hard_free = 1.0 - occupied.astype(np.float32)
    # La sfumatura resta tutta fuori dal margine: dentro il peso e' zero.
    free = cv2.GaussianBlur(hard_free, (0, 0), sigmaX=grow / 3.0) * hard_free
    weight = cv2.resize(free, (width, height), interpolation=cv2.INTER_NEAREST) * rate
    return cv2.blendLinear(frame, background, weight, 1.0 - weight)


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


class AsyncRecognizer:
    """Riconosce il gesto in un thread: il video non aspetta i modelli (30-200 ms).

    detect(image, timestamp_ms) gira solo nel thread; il loop consegna al massimo
    un frame alla volta e legge l'ultimo risultato quando e' pronto.
    """

    def __init__(self, detect: Callable[[mp.Image, int], object], close: Callable[[], None]):
        self.detect = detect
        self.close_detector = close
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.pending: tuple[np.ndarray, int] | None = None
        self.result: tuple[object, int] | None = None
        self.error: Exception | None = None
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="gesture", daemon=True)
        self.thread.start()

    @property
    def busy(self) -> bool:
        with self.lock:
            return self.pending is not None

    def submit(self, rgb: np.ndarray, timestamp_ms: int) -> None:
        with self.lock:
            self.pending = (rgb, timestamp_ms)
        self.wake.set()

    def take_result(self) -> tuple[object, int] | None:
        """Ultimo risultato non ancora letto, oppure None."""
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"Riconoscimento del gesto fallito: {self.error}") from self.error
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
                detected = self.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
            except Exception as error:
                with self.lock:
                    self.error = error
                    self.pending = None
                return
            with self.lock:
                self.result = (detected, timestamp_ms)
                self.pending = None

    def close(self) -> None:
        self.stopping = True
        self.wake.set()
        self.thread.join(timeout=2.0)
        self.close_detector()


class PersonSegmenter:
    """Una sola segmentazione per frame: MediaPipe VIDEO rifiuta timestamp ripetuti.

    Effetto e aggiornamento dello sfondo condividono cosi' la stessa maschera.
    """

    def __init__(self, segmenter: vision.ImageSegmenter):
        self.segmenter = segmenter
        self.last: tuple[int, np.ndarray] | None = None

    def __call__(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        if self.last is not None and self.last[0] == timestamp_ms:
            return self.last[1]
        height, width = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = person_mask(self.segmenter, image, timestamp_ms, width, height)
        self.last = (timestamp_ms, mask)
        return mask


def detection_function(mode: str, recognizer) -> Callable[[mp.Image, int], object]:
    if mode == "naruto":
        return recognizer.recognize_joined_hands
    if mode == "goku":
        return recognizer.recognize_pose
    # Cercata a ogni chiamata (non legata qui) cosi' i test possono sostituirla.
    return lambda image, timestamp_ms: recognize_gesture(recognizer, image, timestamp_ms)


def detector_status(recognizer, result) -> str:
    """Una riga di diagnostica per --gesture-debug, solo nel terminale."""
    details = [f"risultato={result}"]
    for label, attribute in (
        ("ritaglio", "preferred_crop"), ("rotazione", "preferred_rotation"),
        ("vista", "preferred_view"), ("mani", "last_hand_count"), ("viso", "face_visible"),
    ):
        if hasattr(recognizer, attribute):
            details.append(f"{label}={getattr(recognizer, attribute)}")
    return ", ".join(details)


def recognize_gesture(
    recognizer: MultiViewGestureRecognizer,
    image: mp.Image,
    timestamp_ms: int,
) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


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


def open_sound(mode: str) -> VanishSound | None:
    if mode == "naruto":
        return VanishSound()
    if mode == "goku":
        path = sound_path()
        if path is None:
            print(f"[Goku] Nessun suono: aggiungi {GOKU_SOUND_NAME} in assets/ per attivarlo.")
            return None
        print(f"[Goku] Suono: {path.name}")
        return VanishSound(path)
    return None


def run_camera(args: argparse.Namespace, resources: ExitStack, mode_name: str) -> int:
    download_models(mode_name)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, recognizer = create_tasks(mode_name)
    resources.callback(segmenter.close)
    detector = AsyncRecognizer(detection_function(mode_name, recognizer), recognizer.close)
    resources.callback(detector.close)
    sound = open_sound(mode_name)
    if sound is not None:
        resources.callback(sound.close)
    segment = PersonSegmenter(segmenter)
    mode: Mode = MODES[mode_name](segment, sound)
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    state = AppState(calibration_started=time.perf_counter(), fullscreen=args.fullscreen)
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    background_samples: list[np.ndarray] = []
    virtual_camera = None
    virtual_camera_initialized = False
    calibration_remaining = None
    last_gesture_report = float("-inf")
    started = state.calibration_started
    print("[Webcam] Esci dall'inquadratura per 4 secondi: acquisisco lo sfondo.")
    print(f"[{mode.name.capitalize()}] {mode.instructions}")
    if mode.manual_keys:
        print("[Webcam] Comandi nella finestra: SPAZIO nuovo sfondo, R riappari, V svanisci, F fullscreen, Q esci.")
    else:
        print("[Webcam] Comandi nella finestra: SPAZIO nuovo sfondo, F fullscreen, Q esci.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("La webcam ha smesso di restituire immagini.")

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
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
            now = time.perf_counter()
            elapsed_ms = int((now - started) * 1000)

            if state.phase == "calibrating":
                elapsed = now - state.calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining:
                    print(f"[Webcam] Acquisizione sfondo: {remaining} secondi.")
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
                    state.phase = "running"
                    mode.reset()
                    print("[Webcam] Sfondo pronto. Puoi rientrare nell'inquadratura.")
                display = frame

            else:
                if (
                    mode.detecting
                    and now - state.last_gesture_check >= mode.detect_interval
                    and not detector.busy
                ):
                    state.last_gesture_check = now
                    detector.submit(mode.detection_image(frame), elapsed_ms)
                detected = detector.take_result()
                if detected is not None and mode.detecting:
                    mode.on_detection(detected[0], now)
                    if args.gesture_debug and now - last_gesture_report >= 1.0:
                        print(f"[Gesto] {detector_status(recognizer, detected[0])}")
                        last_gesture_report = now
                display = mode.render(FrameInfo(frame, background, now, elapsed_ms))

                # Lo sfondo segue luce e piccoli spostamenti, anche da invisibili.
                if (
                    not args.static_background
                    and mode.phase in ("ready", "vanished")
                    and now - state.last_background_update >= BACKGROUND_UPDATE_SECONDS
                ):
                    state.last_background_update = now
                    background = update_background(background, frame, segment(frame, elapsed_ms))
                    mode.background_changed()

            if virtual_camera is not None:
                send_virtual_frame(virtual_camera, display)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            running = state.phase == "running"
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                state.phase = "calibrating"
                state.calibration_started = now
                background_samples.clear()
                calibration_remaining = None
                mode.reset()
                print("[Webcam] Esci dall'inquadratura per 4 secondi: acquisisco il nuovo sfondo.")
            elif key == ord("r") and running and mode.manual_keys:
                mode.reappear()
            elif key == ord("v") and running and mode.manual_keys:
                mode.vanish(now)
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
