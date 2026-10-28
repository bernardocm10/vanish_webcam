"""Forehead two-finger trigger and comic speed-line teleport for Goku mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..paths import ASSETS_DIR


HOLD_SECONDS = 1.0
EFFECT_SECONDS = 1.5
SOUND_NAME = "teleport.MP3"

# Face Mesh: top of the forehead, nose bridge and cheekbones (face width).
FOREHEAD_TOP = 10
NOSE_BRIDGE = 168
CHEEK_LEFT = 234
CHEEK_RIGHT = 454

LINE_COLOR = np.array((255, 246, 228), dtype=np.float32)
# Light blue halo around the white: the lines stay visible even on light walls.
GLOW_COLOR = np.array((255, 170, 60), dtype=np.float32)


def sound_path(assets: Path = ASSETS_DIR) -> Path | None:
    """assets/teleport.MP3 with any letter case: Linux file names are case-sensitive."""
    for path in sorted(assets.glob("*")):
        if path.name.lower() == SOUND_NAME.lower() and path.is_file():
            return path
    return None


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    t = min(1.0, max(0.0, (value - edge0) / max(edge1 - edge0, 1e-6)))
    return t * t * (3.0 - 2.0 * t)


def face_reference(face_landmarks, width: int, height: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return the forehead segment (top, nose bridge) and the face width in pixels."""
    if len(face_landmarks) <= CHEEK_RIGHT:
        return None
    points = np.array([(p.x * width, p.y * height) for p in face_landmarks], dtype=np.float32)
    if not np.isfinite(points).all():
        return None
    face_width = float(np.linalg.norm(points[CHEEK_LEFT] - points[CHEEK_RIGHT]))
    if face_width < 8.0:
        return None
    return points[FOREHEAD_TOP], points[NOSE_BRIDGE], face_width


def segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    t = float(np.clip(np.dot(point - start, direction) / max(float(np.dot(direction, direction)), 1e-6), 0.0, 1.0))
    return float(np.linalg.norm(point - (start + direction * t)))


def forehead_fingers_pose(hand_landmarks, face, width: int, height: int) -> bool:
    """Index and middle straight and together, ring and pinky folded, tips on the forehead."""
    if face is None or len(hand_landmarks) != 21:
        return False
    points = np.array([(p.x * width, p.y * height) for p in hand_landmarks], dtype=np.float32)
    if not np.isfinite(points).all():
        return False
    lengths = []
    for base in (5, 9):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        # A straight finger stays straight in projection, even when foreshortened.
        if length < 4.0 or np.linalg.norm(finger[3] - finger[0]) / length < 0.75:
            return False
        lengths.append(length)
    reference = min(lengths)
    if np.linalg.norm(points[8] - points[12]) > reference * 0.5:
        return False
    for base in (13, 17):
        if np.linalg.norm(points[base + 3] - points[base]) > reference * 0.7:
            return False
    top, bridge, face_width = face
    # The tips or last phalanges touching the forehead is enough, not the exact spot.
    contact = min(segment_distance(points[index], top, bridge) for index in (7, 8, 11, 12))
    return contact < face_width * 0.35


@dataclass
class HoldTracker:
    """Fire once after a pose is held; tolerate brief dropouts, then require a release."""

    hold_seconds: float = HOLD_SECONDS
    grace_seconds: float = 0.25
    release_seconds: float = 0.3
    started: float | None = None
    last_seen: float = float("-inf")
    armed: bool = True

    def update(self, active: bool, now: float) -> bool:
        if active:
            self.last_seen = now
            if not self.armed:
                return False
            if self.started is None:
                self.started = now
            if now - self.started >= self.hold_seconds:
                self.started = None
                self.armed = False
                return True
            return False
        if not self.armed:
            self.armed = now - self.last_seen >= self.release_seconds
        elif self.started is not None and now - self.last_seen > self.grace_seconds:
            self.started = None
        return False

    def remaining(self, now: float) -> float | None:
        if self.started is None:
            return None
        return max(0.0, self.hold_seconds - (now - self.started))


def make_speed_lines(width: int, seed: int = 4711) -> np.ndarray:
    """Per-column streaks: brightness (0 = gap), phase, speed and length in frame heights."""
    rng = np.random.default_rng(seed)
    lines = np.zeros((4, width), dtype=np.float32)
    thickness = max(1, width // 400)
    spacing = max(5, width // 95)
    for start in range(0, width, spacing):
        left = start + int(rng.integers(0, max(1, spacing - thickness)))
        columns = np.s_[left : min(width, left + thickness)]
        lines[0, columns] = rng.uniform(0.55, 1.0)
        lines[1, columns] = rng.uniform(0.0, 1.0)
        lines[2, columns] = rng.uniform(1.2, 2.4)
        lines[3, columns] = rng.uniform(0.12, 0.35)
    return lines


def speed_line_layer(lines: np.ndarray, rows: slice, columns: slice, frame_height: int, age: float) -> np.ndarray:
    brightness, phase, speed, length = lines[:, columns]
    layer = np.zeros((rows.stop - rows.start, brightness.shape[0]), dtype=np.float32)
    # Only the columns with a line: about a fifth of the width.
    active = np.flatnonzero(brightness)
    if active.size:
        y = np.arange(rows.start, rows.stop, dtype=np.float32)[:, None] / max(frame_height, 1)
        # The bright head rises upwards, the tail fades downwards.
        position = np.mod(y + phase[active] + speed[active] * age, 1.0)
        streak = length[active]
        layer[:, active] = np.where(position < streak, brightness[active] * (1.0 - position / streak), 0.0)
    return layer


def teleport_region(mask: np.ndarray) -> np.ndarray:
    """Soft silhouette where the lines run, kept after the person is gone."""
    height, width = mask.shape
    # It is blurred anyway: computing it at 1/4 avoids stalling a frame at 720p.
    small_size = (max(1, width // 4), max(1, height // 4))
    small = cv2.resize(mask, small_size, interpolation=cv2.INTER_AREA)
    grow = max(3, small_size[0] // 45) | 1
    region = cv2.dilate((small > 0.35).astype(np.float32), np.ones((grow, grow), np.uint8))
    region = cv2.GaussianBlur(region, (0, 0), sigmaX=max(1.0, small_size[0] / 90.0))
    return cv2.resize(region, (width, height), interpolation=cv2.INTER_LINEAR)


_solid_pixels: dict[bytes, np.ndarray] = {}


def _solid(color: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Image filled with one color; the area changes every frame, the buffer does not."""
    value = color.astype(np.uint8)
    count = shape[0] * shape[1]
    pixels = _solid_pixels.get(value.tobytes())
    if pixels is None or pixels.shape[0] < count:
        pixels = _solid_pixels[value.tobytes()] = np.full((count, 3), value, dtype=np.uint8)
    # Constant color: any prefix of the buffer is already the right image.
    return pixels[:count].reshape(shape)


def render_teleport(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    region: np.ndarray,
    lines: np.ndarray,
    progress: float,
    age: float,
) -> np.ndarray:
    """progress 0 = person fully visible, 1 = gone; the return trip plays it backwards."""
    t = float(np.clip(progress, 0.0, 1.0))
    x, y, w, h = cv2.boundingRect((cv2.max(mask, region) > 0.01).view(np.uint8))
    if w == 0 or h == 0:
        return background.copy()
    pad = max(8, frame.shape[1] // 80)
    top, left = max(0, y - pad), max(0, x - pad)
    bottom, right = min(frame.shape[0], y + h + pad), min(frame.shape[1], x + w + pad)
    rows, columns = slice(top, bottom), slice(left, right)
    area = np.s_[rows, columns]

    # Blend in uint8 with OpenCV primitives: at 720p numpy float is too slow.
    person = np.ascontiguousarray(frame[area])
    alpha = np.ascontiguousarray(mask[area], dtype=np.float32)
    backdrop = np.ascontiguousarray(background[area])
    flash = smoothstep(0.30, 0.60, t)
    if flash > 0.0:
        person = cv2.convertScaleAbs(person, alpha=1.0 - 0.65 * flash, beta=255.0 * 0.65 * flash)
    squeeze = smoothstep(0.35, 0.68, t)
    if squeeze > 0.0:
        weights = alpha.sum(axis=0)
        center = float(np.dot(np.arange(alpha.shape[1]), weights) / max(float(weights.sum()), 1e-6))
        scale = 1.0 - 0.97 * squeeze
        matrix = np.float32([[scale, 0.0, center * (1.0 - scale)], [0.0, 1.0, 0.0]])
        size = (alpha.shape[1], alpha.shape[0])
        person = cv2.warpAffine(person, matrix, size, flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(alpha, matrix, size, flags=cv2.INTER_LINEAR)
    alpha *= 1.0 - smoothstep(0.55, 0.70, t)
    output = cv2.blendLinear(person, backdrop, alpha, 1.0 - alpha)

    strength = smoothstep(0.0, 0.22, t) * (1.0 - smoothstep(0.72, 1.0, t))
    if strength > 0.01:
        streaks = speed_line_layer(lines, rows, columns, frame.shape[0], age)
        glow = cv2.GaussianBlur(streaks, (0, 0), sigmaX=max(1.5, frame.shape[1] / 640.0 * 2.2))
        region_area = np.ascontiguousarray(region[area])
        halo = cv2.multiply(cv2.min(glow * 2.2, 1.0), region_area, scale=0.75 * strength)
        output = cv2.blendLinear(output, _solid(GLOW_COLOR, output.shape), 1.0 - halo, halo)
        core = cv2.multiply(streaks, region_area, scale=0.95 * strength)
        output = cv2.blendLinear(output, _solid(LINE_COLOR, output.shape), 1.0 - core, core)

    canvas = background.copy()
    canvas[area] = output
    return canvas
