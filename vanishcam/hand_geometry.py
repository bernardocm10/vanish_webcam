"""Hand and face measures shared by the effects' poses and the meme gestures."""

from __future__ import annotations

import numpy as np

# MediaPipe hand landmarks.
WRIST = 0
PALM = (0, 5, 9, 13, 17)
FINGERTIPS = (8, 12, 16, 20)
# Tip and middle (PIP) joint of index, middle, ring and little finger.
FINGER_JOINTS = ((8, 6), (12, 10), (16, 14), (20, 18))


def points_of(landmarks) -> np.ndarray:
    """(21, 2) normalized x, y of MediaPipe landmarks."""
    return np.array([(landmark.x, landmark.y) for landmark in landmarks], dtype=np.float64)


def extended_finger_states(points: np.ndarray, ratio: float = 1.18) -> list[bool]:
    """Index to little finger: straight when the tip is clearly farther from the wrist than the PIP joint."""
    wrist = points[WRIST]
    return [
        bool(np.linalg.norm(points[tip] - wrist) >= ratio * np.linalg.norm(points[pip] - wrist))
        for tip, pip in FINGER_JOINTS
    ]


def extended_finger_count(points: np.ndarray, ratio: float = 1.18) -> int:
    return sum(extended_finger_states(points, ratio))


def face_box(face_landmarks) -> tuple[float, float, float, float]:
    """Normalized (x1, y1, x2, y2) around the face landmarks."""
    xs = [landmark.x for landmark in face_landmarks]
    ys = [landmark.y for landmark in face_landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def hand_features(landmarks) -> dict:
    """Palm center, wrist and fingertip heights and bounding box, in normalized coordinates."""
    points = points_of(landmarks)
    palm_x, palm_y = points[list(PALM)].mean(axis=0)
    return {
        "landmarks": landmarks,
        "center": (float(palm_x), float(palm_y)),
        "wrist_y": float(points[WRIST][1]),
        "fingertip_y": float(points[list(FINGERTIPS), 1].mean()),
        "bbox": (float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())),
    }
