"""Webcam locale: dissolvenza avviata dal gesto con indice e medio a V."""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
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
    from .particles_effect import salute_pose, salute_reference
    from .modes import MODES, FrameInfo, Mode
    from .call_audio import MIC_DESCRIPTION, CallSound, VirtualMic
    from .control_panel import ACCENTS, AMBER, FONT_DIR, FONTS, GREEN, MODE_CARDS, ControlPanel, PanelView
    from .user_data import Settings, UserData
except ImportError:
    from naruto_effect import joined_hands_from_landmarks, load_log
    from vanish_sound import SOUND_PATH, VanishSound
    from goku_effect import (
        SOUND_NAME as GOKU_SOUND_NAME, face_reference, forehead_fingers_pose, sound_path,
    )
    from particles_effect import salute_pose, salute_reference
    from modes import MODES, FrameInfo, Mode
    from call_audio import MIC_DESCRIPTION, CallSound, VirtualMic
    from control_panel import ACCENTS, AMBER, FONT_DIR, FONTS, GREEN, MODE_CARDS, ControlPanel, PanelView
    from user_data import Settings, UserData


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
# Guadagno di esposizione stimato dove la persona e' assente (serve almeno il 10% dell'immagine).
EXPOSURE_MIN_FREE_FRACTION = 0.1
EXPOSURE_GAIN_RANGE = (0.5, 2.0)


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
        description="Escolha Normal, Naruto, Goku ou Particles para a webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam")
    # 1280x720 e' il 16:9 nativo della maggior parte delle webcam (via MJPG):
    # le videochiamate non devono stirare un 4:3.
    parser.add_argument("--width", type=int, default=1280, help="Largura do vídeo")
    parser.add_argument("--height", type=int, default=720, help="Altura do vídeo")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Abre em tela cheia"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Não espelha a webcam"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="FPS máximo da saída virtual (padrão: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Exige a webcam virtual; por padrão ela é usada se existir. --no-virtual-camera desliga",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help="Driver da webcam virtual",
    )
    parser.add_argument("--virtual-device", help="Nome do dispositivo virtual")
    parser.add_argument(
        "--mode", choices=tuple(MODES),
        help="Efeito inicial (padrão: o último usado); dá para trocar no painel",
    )
    parser.add_argument(
        "--call-audio", action="store_true",
        help="Cria o microfone virtual 'VanishCam Mic' (voz + sons dos efeitos) para videochamadas",
    )
    parser.add_argument(
        "--static-background", action="store_true",
        help="Não atualiza o fundo depois da calibração (como nas versões antigas)",
    )
    parser.add_argument(
        "--background-image", type=Path, metavar="ARQUIVO",
        help="Imagem de fundo inicial (JPG/PNG; padrão: a última usada); dá para trocar no painel",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Mostra no terminal o estado do reconhecimento do gesto (sem texto no vídeo)",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


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
            "Webcam virtual indisponível. No Linux, carregue o v4l2loopback (veja o README); "
            "no Windows, instale o OBS Studio (driver OBS Virtual Camera) ou o Unity Capture "
            "e desligue a câmera virtual do OBS se estiver ligada. "
            f"Detalhe: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Webcam] {message}\n[Webcam] Continuando só com a janela de preview.")
        return None
    print(f"[Webcam] Saída virtual ativa: {camera.device} ({width}x{height}, até {args.fps:g} FPS)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def download_models(mode: str = "normal") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        # Solo Normal usa il riconoscitore di gesti; gli altri usano mani e viso.
        if mode != "normal" and destination == GESTURE_MODEL:
            continue
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[Vanish Lab] Baixando {destination.name}...")
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
                "Não consegui baixar os modelos do MediaPipe. "
                "Verifique a conexão com a internet e tente de novo."
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
            f"Webcam {index} indisponível. Feche outros apps que usam a câmera ou tente --camera 1."
        )

    # Senza MJPG molte webcam USB danno 1280x720 solo a 10 FPS (YUYV) e
    # OpenCV ripiega su 640x480.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def create_segmenter() -> vision.ImageSegmenter:
    options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    return vision.ImageSegmenter.create_from_options(options)


def create_recognizer(
    mode: str = "normal",
) -> MultiViewGestureRecognizer | MultiViewHandLandmarker | GokuPoseDetector:
    if mode in ("goku", "particles"):
        for model in (HAND_MODEL, FACE_MODEL):
            if not model.is_file():
                raise RuntimeError(f"Modelo não encontrado: {model}")
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
        face = vision.FaceLandmarker.create_from_options(face_options)
        try:
            hands = vision.HandLandmarker.create_from_options(hand_options)
        except Exception:
            face.close()
            raise
        detector = SalutePoseDetector if mode == "particles" else GokuPoseDetector
        return detector(face, hands)
    if mode == "naruto":
        if not HAND_MODEL.is_file():
            raise RuntimeError(f"Modelo das mãos não encontrado: {HAND_MODEL}")
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

    primary = task_class.create_from_options(options)
    try:
        alternate = task_class.create_from_options(
            replace(options, running_mode=vision.RunningMode.IMAGE)
        )
    except Exception:
        primary.close()
        raise
    return wrapper_class(primary, alternate)


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


def load_background_image(path: Path) -> np.ndarray:
    # imdecode legge anche percorsi con accenti su Windows, dove imread fallisce.
    try:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        image = None
    if image is None:
        raise RuntimeError(f"Não foi possível abrir a imagem de fundo: {path}")
    return image


def fit_background(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Cobre todo o quadro sem distorcer: escala e corta o excesso no centro."""
    src_height, src_width = image.shape[:2]
    scale = max(width / src_width, height / src_height)
    scaled_width = max(width, round(src_width * scale))
    scaled_height = max(height, round(src_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    scaled = cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)
    x = (scaled_width - width) // 2
    y = (scaled_height - height) // 2
    return np.ascontiguousarray(scaled[y : y + height, x : x + width])


def replace_background(frame: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mantém só a pessoa do frame e coloca a imagem escolhida atrás dela."""
    alpha = mask[..., None]
    return (frame * alpha + background * (1.0 - alpha)).astype(np.uint8)


def update_background(
    background: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
    rate: float = BACKGROUND_UPDATE_RATE,
) -> np.ndarray:
    """Segui i cambi di luce senza ricalibrare e senza stampare la persona nello sfondo.

    L'esposizione automatica della webcam cambia tutta l'immagine (una persona
    davanti a una finestra la fa schiarire): quel guadagno, misurato dove la
    persona e' assente, si applica a tutto lo sfondo, anche dietro la sagoma.
    Altrimenti la zona coperta resta ferma alla calibrazione e, da invisibili,
    appare come un'ombra della persona. Il frame si fonde poi solo lontano dalla
    sagoma: il margine evita che bordi incerti (mani, capelli) restino stampati.
    """
    height, width = mask.shape
    small_size = (max(1, width // 4), max(1, height // 4))
    occupied = (cv2.resize(mask, small_size, interpolation=cv2.INTER_AREA) > 0.05).astype(np.uint8)
    # Margine di almeno il 4% della larghezza attorno alla sagoma.
    grow = 2 * max(1, int(np.ceil(small_size[0] * 0.04))) + 1
    occupied = cv2.dilate(occupied, np.ones((grow, grow), np.uint8))
    hard_free = 1.0 - occupied.astype(np.float32)
    background = apply_exposure_gain(background, frame, hard_free > 0, small_size)
    # La sfumatura resta tutta fuori dal margine: dentro il peso e' zero.
    free = cv2.GaussianBlur(hard_free, (0, 0), sigmaX=grow / 3.0) * hard_free
    weight = cv2.resize(free, (width, height), interpolation=cv2.INTER_NEAREST) * rate
    return cv2.blendLinear(frame, background, weight, 1.0 - weight)


def apply_exposure_gain(
    background: np.ndarray,
    frame: np.ndarray,
    free: np.ndarray,
    small_size: tuple[int, int],
) -> np.ndarray:
    """Porta tutto lo sfondo all'esposizione attuale, canale per canale."""
    small_bg = cv2.resize(background, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    small_frame = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    # Pixel saturi o quasi neri non dicono nulla sul guadagno.
    valid = free & (small_bg.max(axis=2) < 250) & (small_frame.max(axis=2) < 250) & (small_bg.min(axis=2) > 8)
    if np.count_nonzero(valid) < max(16, valid.size * EXPOSURE_MIN_FREE_FRACTION):
        return background
    gain = small_frame[valid].sum(axis=0) / small_bg[valid].sum(axis=0)
    gain = np.clip(gain, EXPOSURE_GAIN_RANGE[0], EXPOSURE_GAIN_RANGE[1])
    if np.all(np.abs(gain - 1.0) < 0.004):
        return background
    return cv2.convertScaleAbs(background * gain.astype(np.float32).reshape(1, 1, 3))


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
    # Quale zona del viso serve e quale posa della mano la deve toccare.
    reference = staticmethod(face_reference)
    pose = staticmethod(forehead_fingers_pose)

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
            self.last_face = self.reference(faces.face_landmarks[0], image.width, image.height)
            self.last_face_ms = timestamp_ms
        elif timestamp_ms - self.last_face_ms > self.FACE_MEMORY_MS:
            self.last_face = None
        result = self.hands.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        return any(
            self.pose(hand, self.last_face, image.width, image.height)
            for hand in result.hand_landmarks
        )

    def close(self) -> None:
        try:
            self.face.close()
        finally:
            self.hands.close()


class SalutePoseDetector(GokuPoseDetector):
    """Continenza: mano tesa con le dita unite sulla fronte o sulle sopracciglia."""

    reference = staticmethod(salute_reference)
    pose = staticmethod(salute_pose)


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
                raise RuntimeError(f"O reconhecimento do gesto falhou: {self.error}") from self.error
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
    if mode in ("goku", "particles"):
        return recognizer.recognize_pose
    # Cercata a ogni chiamata (non legata qui) cosi' i test possono sostituirla.
    return lambda image, timestamp_ms: recognize_gesture(recognizer, image, timestamp_ms)


def detector_status(recognizer, result) -> str:
    """Una riga di diagnostica per --gesture-debug, solo nel terminale."""
    details = [f"resultado={result}"]
    for label, attribute in (
        ("recorte", "preferred_crop"), ("rotacao", "preferred_rotation"),
        ("vista", "preferred_view"), ("maos", "last_hand_count"), ("rosto", "face_visible"),
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
    fonts = tuple(FONT_DIR / names[0] for names in FONTS.values())
    required = (SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL, SOUND_PATH, *fonts)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Arquivos faltando: {', '.join(missing)}")
    load_log()
    import pygame.mixer  # The distributed build must contain its MP3 decoder.
    create_segmenter().close()
    for mode in MODES:
        create_recognizer(mode).close()
    print("[Webcam] Autoteste concluído: modelos, imagens e áudio presentes.")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    # Chiudere il terminale (SIGHUP) o un kill (SIGTERM) diventano un'uscita
    # normale: webcam, finestre e microfono virtuale vengono rilasciati.
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
        mic = VirtualMic()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        # Il video conta piu' dell'audio: avvisa e continua senza microfono virtuale.
        print(f"[Chamada] Microfone virtual indisponível: {error}")
        return None
    resources.callback(mic.close)
    print(f"[Chamada] Microfone virtual ativo: no Teams escolha '{MIC_DESCRIPTION}'.")
    return mic


def open_sound(mode: str, mic: VirtualMic | None = None):
    if mode == "naruto":
        path = SOUND_PATH
    elif mode == "goku":
        path = sound_path()
        if path is None:
            print(f"[Goku] Sem som: coloque {GOKU_SOUND_NAME} em assets/ para ativar.")
            return None
        print(f"[Goku] Som: {path.name}")
    else:
        return None
    sound = VanishSound(path) if mode == "goku" else VanishSound()
    # Con il microfono virtuale l'effetto si sente anche nella chiamata.
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
    **{ord(str(number)): ("mode", card.key) for number, card in enumerate(MODE_CARDS, start=1)},
}
MESSAGE_SECONDS = 6.0


def panel_status(phase: str, mode: Mode, calibration_remaining: int | None) -> tuple[str, tuple[int, int, int]]:
    if phase == "calibrating":
        return f"Capturando o fundo: saia da câmera ({calibration_remaining or 0} s)", AMBER
    accent = ACCENTS.get(mode.name, GREEN)
    return {
        "vanishing": ("Sumindo…", accent),
        "vanished": ("Invisível", accent),
        "appearing": ("Reaparecendo…", accent),
    }.get(mode.phase, ("Pronto: faça o gesto", GREEN))


def run_camera(args: argparse.Namespace, resources: ExitStack, mode_name: str | None = None) -> int:
    user_data = UserData.default()
    saved = user_data.load()
    mode_name = mode_name or (saved.mode if saved.mode in MODES else "normal")
    chosen: Path | None = None
    background_source: np.ndarray | None = None  # Immagine scelta, a risoluzione originale.
    if getattr(args, "background_image", None) is not None:
        chosen = Path(args.background_image)
        background_source = load_background_image(chosen)
    elif saved.background is not None:
        try:
            background_source = load_background_image(saved.background)
            chosen = saved.background
        except RuntimeError as error:
            print(f"[Webcam] {error}")
    user_data.save(Settings(mode_name, chosen))
    gallery = tuple(user_data.gallery())

    download_models(mode_name)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter = create_segmenter()
    resources.callback(segmenter.close)
    segment = PersonSegmenter(segmenter)
    mic = open_call_mic(args, resources)
    effect = EffectRuntime(mode_name, segment, mic)
    # Chiude l'effetto attivo in quel momento, anche dopo un cambio di modo.
    resources.callback(lambda: effect.close())
    resources.callback(cv2.destroyAllWindows)
    # GUI_NORMAL: solo il video, senza la barra degli strumenti di Qt.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)
    panel = ControlPanel()
    resources.callback(panel.close)

    state = AppState(calibration_started=time.perf_counter(), fullscreen=args.fullscreen)
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    real_background: np.ndarray | None = None  # Ultimo sfondo reale: torna senza ricalibrare.
    background_samples: list[np.ndarray] = []
    virtual_camera = None
    virtual_camera_initialized = False
    calibration_remaining = None
    last_gesture_report = float("-inf")
    message, message_until = "", float("-inf")
    started = state.calibration_started
    width = height = 0
    if background_source is None:
        print("[Webcam] Saia da frente da câmera por 4 segundos: capturando o fundo.")
    else:
        # L'immagine e' lo sfondo: niente calibrazione.
        state.phase = "running"
        print(f"[Webcam] Fundo: {chosen}")
    print(f"[{effect.mode.name.capitalize()}] {effect.mode.instructions}")
    print("[Webcam] Troque efeito e fundo no painel 'VanishCam' (ou teclas 1-4). Q fecha.")

    def notify(text: str, now: float) -> None:
        nonlocal message, message_until
        print(f"[Webcam] {text}")
        message, message_until = text, now + MESSAGE_SECONDS

    def start_calibration(now: float) -> None:
        nonlocal calibration_remaining
        state.phase = "calibrating"
        state.calibration_started = now
        background_samples.clear()
        calibration_remaining = None
        effect.mode.reset()
        print("[Webcam] Saia da frente da câmera por 4 segundos: capturando o novo fundo.")

    def switch_mode(name: str) -> None:
        nonlocal effect
        if name == effect.name or name not in MODES:
            return
        previous = effect.name
        # Prima chiudere: il vecchio suono possiede il mixer audio condiviso.
        effect.close()
        try:
            effect = EffectRuntime(name, segment, mic)
        except Exception as error:
            effect = EffectRuntime(previous, segment, mic)
            notify(f"Não consegui abrir o modo {name}: {error}", time.perf_counter())
            return
        user_data.save(Settings(name, chosen))
        print(f"[{effect.mode.name.capitalize()}] {effect.mode.instructions}")

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
        user_data.save(Settings(effect.name, chosen))

    def handle(action: tuple, now: float) -> None:
        nonlocal gallery
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
                notify(f"Imagem não adicionada: {error}", now)
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
            set_fullscreen(state.fullscreen)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("A webcam parou de enviar imagens.")

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            if not virtual_camera_initialized:
                # La webcam puo' non supportare la risoluzione chiesta: adatta la
                # finestra a quella reale, senza stirare l'immagine.
                print(f"[Webcam] Resolução: {width}x{height}")
                if (width, height) != (args.width, args.height):
                    cv2.resizeWindow(WINDOW_NAME, width, height)
                virtual_camera = open_virtual_camera(args, width, height)
                virtual_camera_initialized = True
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
                if background_source is not None:
                    background = fit_background(background_source, width, height)
            now = time.perf_counter()
            elapsed_ms = int((now - started) * 1000)
            camera_frame = frame
            mode = effect.mode
            if background_source is not None:
                # La maschera resta in cache per questo timestamp: gli effetti la riusano.
                frame = replace_background(frame, background, segment(frame, elapsed_ms))

            if state.phase == "calibrating":
                elapsed = now - state.calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining:
                    print(f"[Webcam] Capturando o fundo: {remaining} s.")
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
                    print("[Webcam] Fundo pronto. Pode voltar para a frente da câmera.")
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
                        print(f"[Gesto] {detector_status(effect.recognizer, detected[0])}")
                        last_gesture_report = now
                display = mode.render(FrameInfo(frame, background, now, elapsed_ms))

                # Lo sfondo segue luce e piccoli spostamenti, anche da invisibili.
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

            if virtual_camera is not None:
                send_virtual_frame(virtual_camera, display)
            cv2.imshow(WINDOW_NAME, display)
            status, status_color = panel_status(state.phase, mode, calibration_remaining)
            panel.show(PanelView(
                mode=effect.name, status=status, status_color=status_color, background=chosen,
                gallery=gallery, manual_keys=mode.manual_keys, can_recapture=background_source is None,
                fullscreen=state.fullscreen, message=message if now < message_until else "",
            ))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or panel.closed:
                break
            actions = panel.poll()
            if key in KEY_ACTIONS:
                actions.append(KEY_ACTIONS[key])
            for action in actions:
                handle(action, now)
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
        print(f"\n[Vanish Lab] ERRO: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
