"""Gesture and pose detection for each mode, run off the video thread by AsyncRecognizer."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .effects.ash import hands_up_pose, hands_up_reference
from .effects.goku import face_reference, forehead_fingers_pose
from .effects.naruto import hand_seal_from_landmarks
from .effects.particles import salute_pose, salute_reference
from .i18n import t
from .models import FACE_MODEL, GESTURE_MODEL, HAND_MODEL

GESTURE_INTERVAL_SECONDS = 0.14


def require_models(*models: Path) -> None:
    for model in models:
        if not model.is_file():
            raise RuntimeError(t("models.missing", path=model))


def create_pose_detector(detector: type[GokuPoseDetector]) -> GokuPoseDetector:
    """Face and hands on the same frame; the detector class decides which pose counts."""
    require_models(HAND_MODEL, FACE_MODEL)
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
    return detector(face, hands)


def create_hand_landmarker() -> MultiViewHandLandmarker:
    require_models(HAND_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.45,
    )
    return _multi_view(vision.HandLandmarker, options, MultiViewHandLandmarker)


def create_gesture_recognizer() -> MultiViewGestureRecognizer:
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
    return _multi_view(vision.GestureRecognizer, options, MultiViewGestureRecognizer)


def _multi_view(task_class, options, wrapper_class):
    """The VIDEO task for tracking, plus an IMAGE one for the rotated and cropped retries."""
    primary = task_class.create_from_options(options)
    try:
        alternate = task_class.create_from_options(
            replace(options, running_mode=vision.RunningMode.IMAGE)
        )
    except Exception:
        primary.close()
        raise
    return wrapper_class(primary, alternate)


def victory_landmark_score(points: np.ndarray) -> float:
    """Recognize extended index and middle fingers from 3D distances, not orientation."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP and tip: index, middle, ring, little finger.
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

    # Seen from the side, folded fingers can look straight but very short:
    # do not require clearly visible joints for the ring and little fingers.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # The two tips must be apart; the thumb may stay hidden.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Geometric quality of the extended fingers, not the classifier's probability.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # Landmarks are returned even when the classifier does not recognize
    # "Victory" (for example with the back of the hand facing the webcam).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # ALWAYS check the visible geometry too: the estimated depth can be
    # inaccurate when the palm and the folded fingers are hidden.
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
    """Retry detection on rotated frames without disturbing the tracking."""

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

        # IMAGE forces a fresh detection; the VIDEO model stays on the original
        # frame. This avoids inconsistent timestamps/tracking across rotations.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # In the side pose the palm is often too small in the full frame. The
        # two lower crops overlap and cover both hands, even when the image is
        # mirrored.
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
        # Which seal matched last ("palms" or "clasped"), shown by --gesture-debug.
        self.last_seal: str | None = None

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        self.last_seal = hand_seal_from_landmarks(result.hand_landmarks, image.width, image.height)
        if self.last_seal is not None:
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
            self.last_seal = hand_seal_from_landmarks(result.hand_landmarks, rotated.width, rotated.height)
            if self.last_seal is not None:
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
    """Forehead from the Face Landmarker and fingers from the Hand Landmarker, on the same frame."""

    FACE_MEMORY_MS = 1000
    # Which area of the face is needed and which hand pose must touch it.
    reference = staticmethod(face_reference)
    pose = staticmethod(forehead_fingers_pose)

    def __init__(self, face, hands):
        self.face = face
        self.hands = hands
        # Fingers on the forehead hide part of the face: remember the last forehead seen.
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
        return self.matches(result.hand_landmarks, self.last_face, image.width, image.height)

    def matches(self, hands, face, width: int, height: int) -> bool:
        return any(self.pose(hand, face, width, height) for hand in hands)

    def close(self) -> None:
        try:
            self.face.close()
        finally:
            self.hands.close()


class HandsUpPoseDetector(GokuPoseDetector):
    """Both hands raised beside the head: a pose of the two hands together, not of either one."""

    reference = staticmethod(hands_up_reference)

    def matches(self, hands, face, width: int, height: int) -> bool:
        return hands_up_pose(hands, face, width, height)


class SalutePoseDetector(GokuPoseDetector):
    """Salute: a flat hand with the fingers together on the forehead or the brow."""

    reference = staticmethod(salute_reference)
    pose = staticmethod(salute_pose)


class AsyncRecognizer:
    """Recognize the gesture in a thread: the video never waits for the models (30-200 ms).

    detect(image, timestamp_ms) runs only in the thread; the loop hands over at most
    one frame at a time and reads the latest result when it is ready.
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
        """The latest result not read yet, or None."""
        with self.lock:
            if self.error is not None:
                raise RuntimeError(t("gesture.failed", error=self.error)) from self.error
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


def detector_status(recognizer, result) -> str:
    """One diagnostic line for --gesture-debug, in the terminal only."""
    details = [f"result={result}"]
    for label, attribute in (
        ("crop", "preferred_crop"), ("rotation", "preferred_rotation"),
        ("view", "preferred_view"), ("hands", "last_hand_count"), ("seal", "last_seal"), ("face", "face_visible"),
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
