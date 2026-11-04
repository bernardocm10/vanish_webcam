"""Itachi: hands together at the chest, and the silhouette breaks into a flock of crows.

Each crow starts at a point of the captured silhouette and cuts its own winged hole
in the body as it takes off; the room turns to red ink and the Sharingan shows for a
moment at the top. Played backwards, the crows converge and rebuild the person.

The crow sheet and the Sharingan picture come from the original VanishCam
(Nicovic17/vanish_webcam), like the effect's drawing ideas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from ..i18n import t
from ..paths import ASSETS_DIR

EFFECT_SECONDS = 4.0
# By then the whole body has left as crows; the last ones are still flying away.
BODY_RELEASE_SECONDS = 3.05
CROW_COUNT = 78
FIRST_CROW_SECONDS = 0.14
LAST_CROW_SECONDS = 2.82
# Hands together at the chest for a moment: clapping passes through the pose.
HOLD_SECONDS = 0.2

CROW_SHEET_PATH = ASSETS_DIR / "itachi_crow_sheet.png"
SHARINGAN_PATH = ASSETS_DIR / "sharingan.png"
# Every flapping pose of the sheet is centered on a canvas of this size.
SPRITE_WIDTH, SPRITE_HEIGHT = 176, 128
# The sizes below were tuned for 960x540 and scale with the frame.
REFERENCE_SIZE = (960, 540)

# Fingertips and knuckles: where two hands held together touch.
CONTACT_POINTS = [4, 5, 8, 9, 12, 13, 16, 17, 20]
PALM = [0, 5, 9, 13, 17]

INK = (15, 16, 22)


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    t_ = min(1.0, max(0.0, (value - edge0) / max(edge1 - edge0, 1e-6)))
    return t_ * t_ * (3.0 - 2.0 * t_)


def chest_hands_pose(hands, face, width: int, height: int) -> bool:
    """Two hands touching in front of the chest: below the chin and under the face.

    face is (center x, chin y, face width) in pixels, from hands_up_reference.
    """
    if face is None or len(hands) < 2:
        return False
    center_x, chin_y, face_width = face
    points = []
    for hand in hands[:2]:
        if len(hand) != 21:
            return False
        hand_points = np.array([(p.x * width, p.y * height) for p in hand], dtype=np.float32)
        if not np.isfinite(hand_points).all():
            return False
        points.append(hand_points)
    sizes = [max(float(np.linalg.norm(p[9] - p[0])), float(np.linalg.norm(p[5] - p[17]))) for p in points]
    # Two hands of about the same size: not a hand and its reflection far behind.
    if min(sizes) < 6.0 or max(sizes) > min(sizes) * 2.5:
        return False
    size = sum(sizes) / 2
    palms = [p[PALM].mean(axis=0) for p in points]
    middle = (palms[0] + palms[1]) / 2
    below_chin = float(middle[1] - chin_y)
    if not 0.25 * face_width <= below_chin <= 4.0 * face_width:
        return False
    if abs(float(middle[0]) - center_x) > 1.1 * face_width:
        return False
    if np.linalg.norm(palms[0] - palms[1]) > size * 1.45:
        return False
    # Fingertips or the edges of the palms must meet: wrists close together are not enough.
    first, second = points[0][CONTACT_POINTS], points[1][CONTACT_POINTS]
    contact = float(np.min(np.linalg.norm(first[:, None] - second[None, :], axis=2)))
    return contact <= size * 0.48


@lru_cache(maxsize=1)
def load_crow_sprites() -> tuple[np.ndarray, ...]:
    """The five flapping poses of the sheet, left to right, each centered on a BGRA canvas."""
    sheet = cv2.imread(str(CROW_SHEET_PATH), cv2.IMREAD_UNCHANGED)
    if sheet is None or sheet.ndim != 3 or sheet.shape[2] != 4:
        raise RuntimeError(t("itachi.unreadable", path=CROW_SHEET_PATH))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((sheet[:, :, 3] > 30).astype(np.uint8), 8)
    birds = sorted((index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] > 5000),
                   key=lambda index: stats[index, cv2.CC_STAT_LEFT])
    if len(birds) != 5:
        raise RuntimeError(t("itachi.unreadable", path=CROW_SHEET_PATH))
    sprites = []
    for index in birds:
        x, y, width, height = (int(value) for value in stats[index, :4])
        left, top = max(0, x - 10), max(0, y - 10)
        right, bottom = min(sheet.shape[1], x + width + 10), min(sheet.shape[0], y + height + 10)
        crop = sheet[top:bottom, left:right].copy()
        # Only this bird: the wing tips of its neighbours stay out.
        crop[labels[top:bottom, left:right] != index] = 0
        factor = min((SPRITE_WIDTH - 8) / crop.shape[1], (SPRITE_HEIGHT - 8) / crop.shape[0])
        scaled = cv2.resize(crop, (max(1, round(crop.shape[1] * factor)), max(1, round(crop.shape[0] * factor))),
                            interpolation=cv2.INTER_AREA)
        canvas = np.zeros((SPRITE_HEIGHT, SPRITE_WIDTH, 4), np.uint8)
        px, py = (SPRITE_WIDTH - scaled.shape[1]) // 2, (SPRITE_HEIGHT - scaled.shape[0]) // 2
        canvas[py:py + scaled.shape[0], px:px + scaled.shape[1]] = scaled
        sprites.append(canvas)
    return tuple(sprites)


@lru_cache(maxsize=1)
def load_sharingan() -> np.ndarray:
    """The two eyes as BGRA: the black backdrop turns transparent, the dark pupils stay."""
    image = cv2.imread(str(SHARINGAN_PATH), cv2.IMREAD_COLOR)
    if image is None or image.shape[0] < 700 or image.shape[1] < 1000:
        raise RuntimeError(t("itachi.unreadable", path=SHARINGAN_PATH))
    # The eyes' band of the picture, without the empty black above and below.
    crop = image[round(image.shape[0] * 0.16):round(image.shape[0] * 0.78)].copy()
    height, width = crop.shape[:2]
    # A soft glow behind each eye keeps the black pupils from turning transparent.
    glow = np.zeros((height, width), np.uint8)
    for cx in (int(width * 0.19), int(width * 0.81)):
        cv2.ellipse(glow, (cx, round(height * 0.27)), (int(width * 0.21), round(height * 0.21)),
                    0, 0, 360, 220, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 26)
    brightness = crop.max(axis=2).astype(np.float32)
    alpha = np.maximum(glow.astype(np.float32), np.clip((brightness - 13) * 1.3, 0, 255))
    # Fade the borders: no visible rectangle around the picture.
    horizontal = np.minimum(np.arange(width), width - 1 - np.arange(width)) / 65.0
    vertical = np.minimum(np.arange(height), height - 1 - np.arange(height)) / 32.0
    alpha *= np.clip(horizontal[None, :], 0, 1) * np.clip(vertical[:, None], 0, 1)
    return np.dstack((crop, alpha.astype(np.uint8)))


def largest_person(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """The biggest silhouette as 0/1 and its box (x, y, width, height); None when nobody is there."""
    binary = (mask > 0.48).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count < 2:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = (int(value) for value in stats[component])
    if area < mask.size * 0.025 or width < 20 or height < 35:
        return None
    return (labels == component).astype(np.uint8), (x, y, width, height)


class BloodTint:
    """Translucent crimson ink over the room, the same stain on every frame of one size."""

    def __init__(self, seed: int = 1903):
        self.seed = seed
        self.layers: tuple[np.ndarray, np.ndarray] | None = None
        self.source: np.ndarray | None = None
        self.tinted: np.ndarray | None = None

    def _layers(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        if self.layers is None or self.layers[0].shape[:2] != (height, width):
            rng = np.random.default_rng(self.seed)
            coarse = rng.random((max(4, height // 58), max(5, width // 58))).astype(np.float32)
            fine = rng.random((max(8, height // 15), max(10, width // 15))).astype(np.float32)
            pattern = np.clip(cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC) * 0.66
                              + cv2.resize(fine, (width, height), interpolation=cv2.INTER_CUBIC) * 0.34, 0, 1)
            red = (0.60 + 0.20 * pattern)[:, :, None]
            ink = np.clip((0.36 - pattern) * 1.25, 0, 0.30)[:, :, None]
            self.layers = (red, ink)
        return self.layers

    def apply(self, background: np.ndarray) -> np.ndarray:
        """The room in red ink; recomputed only when the room picture changes."""
        if self.source is not background or self.tinted is None:
            red, ink = self._layers(*background.shape[:2])
            colored = background.astype(np.float32) * (1 - red) + np.array((15, 10, 200), np.float32) * red
            colored = colored * (1 - ink) + np.array((9, 10, 17), np.float32) * ink
            self.source = background
            self.tinted = np.clip(colored, 0, 255).astype(np.uint8)
        return self.tinted

    def blend(self, background: np.ndarray, timeline: float) -> np.ndarray:
        """The red spreading over the first second."""
        strength = smoothstep(0.08, 1.10, timeline)
        if strength <= 0:
            return background.copy()
        tinted = self.apply(background)
        if strength >= 1:
            return tinted.copy()
        return cv2.addWeighted(background, 1.0 - strength, tinted, strength, 0)


def cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t_: float) -> np.ndarray:
    u = 1.0 - t_
    return u ** 3 * p0 + 3 * u * u * t_ * p1 + 3 * u * t_ * t_ * p2 + t_ ** 3 * p3


@dataclass(frozen=True)
class Crow:
    # Bezier flight path, pixels.
    origin: np.ndarray
    control1: np.ndarray
    control2: np.ndarray
    destination: np.ndarray
    launch: float     # when it leaves the body, seconds
    duration: float   # flight time, seconds
    size: float       # pixels
    phase: float      # where in the wing beat it starts
    frequency: float  # wing beats per second
    depth: float      # >1 crosses in front of the camera, <0.8 flies behind the person
    hero: bool        # one of the few big crows in the foreground


@dataclass(frozen=True)
class Feather:
    origin: np.ndarray
    destination: np.ndarray
    launch: float
    duration: float
    size: float
    angle: float
    spin: float


@dataclass
class CrowFlock:
    """The silhouette captured once, the crows leaving it, and when each pixel goes."""

    source: np.ndarray              # frame captured at the start
    mask: np.ndarray                # float32 mask captured at the start
    bounds: tuple[int, int, int, int]  # the person's box: x, y, width, height
    area: tuple[int, int, int, int]    # box of every masked pixel: the only area to blend
    crows: list[Crow]
    feathers: list[Feather]
    release: np.ndarray             # float32 per pixel: when that piece of body leaves
    sprites: tuple[np.ndarray, ...]
    sharingan: Sprite               # already sized for the frame
    sprite_cache: dict = field(default_factory=dict)
    rotated_cache: dict = field(default_factory=dict)


def make_flock(frame: np.ndarray, mask: np.ndarray, seed: int = 4411) -> CrowFlock | None:
    """Pick where the crows leave from and the order the body breaks up; None without a person."""
    person = largest_person(mask)
    if person is None:
        return None
    binary, (left, top, body_width, body_height) = person
    height, width = mask.shape
    scale = min(width / REFERENCE_SIZE[0], height / REFERENCE_SIZE[1])
    rng = np.random.default_rng(seed)
    sprites = load_crow_sprites()

    candidates = np.argwhere(cv2.erode(binary, np.ones((7, 7), np.uint8)) > 0)
    if len(candidates) < CROW_COUNT:
        candidates = np.argwhere(binary > 0)
    relative_y = (candidates[:, 0] - top) / max(1, body_height)
    # More crows from the shoulders and the chest, where the eye goes first.
    weights = np.where((relative_y > 0.18) & (relative_y < 0.68), 3.0, 1.0)
    chosen = candidates[rng.choice(len(candidates), CROW_COUNT, replace=len(candidates) < CROW_COUNT,
                                   p=weights / weights.sum())]
    # Shoulders, arms and torso lead and the face follows; the random part breaks the rows
    # without scrambling the anatomy.
    vertical = (chosen[:, 0] - top) / max(1, body_height)
    priority = np.abs(vertical - 0.39) * 0.65 + np.where(vertical < 0.20, 0.42, 0.0) + rng.uniform(0, 0.27, CROW_COUNT)
    order = np.argsort(priority)
    launches = np.empty(CROW_COUNT, np.float32)
    launches[order] = np.linspace(FIRST_CROW_SECONDS, LAST_CROW_SECONDS, CROW_COUNT)
    rank = np.empty(CROW_COUNT, np.int32)
    rank[order] = np.arange(CROW_COUNT)
    hero_ranks = {10, 31, 54, 72}

    crows = []
    for index, (row, column) in enumerate(chosen):
        origin = np.array([float(column), float(row)], np.float32)
        hero = int(rank[index]) in hero_ranks
        route = 3 if hero else index % 5
        direction = -1 if index % 2 else 1
        if route == 0:  # Up, often in a shallow curve.
            destination = np.array([column + direction * rng.uniform(80, 310), -150.0])
        elif route in (1, 2):  # Out to a side, climbing.
            destination = np.array([width + 160.0 if direction > 0 else -160.0,
                                    row - rng.uniform(90, height * 0.7)])
        elif route == 3:  # A near crow crossing the foreground.
            destination = np.array([width + 210.0 if direction > 0 else -210.0, row + rng.uniform(-80, 130)])
        else:  # A far crow, up and small.
            destination = np.array([column + direction * rng.uniform(130, 350), -100.0])
        control1 = origin + np.array([direction * rng.uniform(30, 130), -rng.uniform(35, 160)])
        control2 = destination + np.array([-direction * rng.uniform(20, 150), rng.uniform(-110, 110)])
        depth = 1.75 if hero else 1.17 if route == 3 else 0.63 if route == 4 else 1.0
        size = (rng.uniform(63, 83) if hero else rng.uniform(32, 62)) * depth * scale
        launch = float(launches[index])
        crows.append(Crow(
            origin, control1, control2, destination.astype(np.float32), launch,
            float(min(rng.uniform(1.45, 1.85), EFFECT_SECONDS - launch - 0.08)), float(size),
            float(rng.uniform(0, 2 * math.pi)),
            float(rng.uniform(2.0, 2.9) if hero else rng.uniform(2.6, 3.8)), depth, hero,
        ))

    feathers = []
    for index in rng.choice(CROW_COUNT, size=34, replace=False):
        crow = crows[int(index)]
        destination = crow.origin + np.array([rng.uniform(-165, 165) * scale, -rng.uniform(45, 160) * scale],
                                             np.float32)
        feathers.append(Feather(crow.origin, destination, crow.launch + float(rng.uniform(0.06, 0.19)),
                                float(rng.uniform(0.68, 1.25)), float(rng.uniform(7, 15) * scale),
                                float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(-4.5, 4.5))))

    eyes = load_sharingan()
    eye_width = min(round(width * 0.62), round(height * 1.16))
    eye_height = max(1, round(eye_width * eyes.shape[0] / eyes.shape[1]))
    sharingan = split_sprite(cv2.resize(eyes, (eye_width, eye_height), interpolation=cv2.INTER_AREA))

    mask = np.clip(mask.astype(np.float32), 0, 1)
    ys, xs = np.nonzero(mask > 0.01)
    area = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    release = release_map(mask, (left, top, body_width, body_height), chosen, launches, crows, sprites, rng)
    return CrowFlock(frame.copy(), mask, (left, top, body_width, body_height), area, crows, feathers,
                     release, sprites, sharingan)


def release_map(mask: np.ndarray, bounds: tuple[int, int, int, int], origins: np.ndarray,
                launches: np.ndarray, crows: list[Crow], sprites: tuple[np.ndarray, ...],
                rng: np.random.Generator) -> np.ndarray:
    """When each pixel of the body leaves: the territory of the nearest crow, torn a little."""
    height, width = mask.shape
    step = 4
    small_h, small_w = max(1, -(-height // step)), max(1, -(-width // step))
    ys, xs = np.mgrid[:small_h, :small_w].astype(np.float32) * step
    nearest = np.full((small_h, small_w), np.inf, np.float32)
    release = np.zeros((small_h, small_w), np.float32)
    for (row, column), launch in zip(origins, launches):
        distance = (xs - column) ** 2 + (ys - row) ** 2
        closer = distance < nearest
        nearest[closer] = distance[closer]
        release[closer] = launch
    # A small organic tear across each territory, instead of straight tile borders.
    noise = rng.uniform(-0.055, 0.055, (max(2, small_h // 6), max(2, small_w // 6))).astype(np.float32)
    release += cv2.resize(noise, (small_w, small_h), interpolation=cv2.INTER_CUBIC) + 0.10
    release = np.clip(release, 0.09, BODY_RELEASE_SECONDS - 0.06)
    release = cv2.resize(release, (width, height), interpolation=cv2.INTER_LINEAR)
    # Each crow cuts a hole of its own shape as it takes off; the territories above
    # then remove the body between those winged holes.
    wings = sprites[2][:, :, 3]
    for crow in crows:
        bird_width = max(14, round(crow.size * 2.15))
        bird_height = max(10, round(bird_width * SPRITE_HEIGHT / SPRITE_WIDTH))
        wing = cv2.resize(wings, (bird_width, bird_height), interpolation=cv2.INTER_LINEAR)
        x, y = round(crow.origin[0] - bird_width / 2), round(crow.origin[1] - bird_height / 2)
        x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + bird_width), min(height, y + bird_height)
        if x1 < x2 and y1 < y2:
            target = release[y1:y2, x1:x2]
            np.minimum(target, crow.launch + 0.035, out=target, where=wing[y1 - y:y2 - y, x1 - x:x2 - x] > 110)
    # Keep a neck and torso core while the shoulders break up, and let the head go
    # before it: no floating face.
    left, top, body_width, body_height = bounds
    y1, y2, x1, x2 = top, min(height, top + body_height), left, min(width, left + body_width)
    region = release[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    vertical = (yy - top) / max(1, body_height)
    core = (vertical >= 0.18) & (vertical < 0.76) & (np.abs(xx - (left + body_width * 0.5)) < body_width * 0.18)
    np.maximum(region, 1.86, out=region, where=core)
    head = (vertical < 0.22) & (mask[y1:y2, x1:x2] > 0.48)
    np.clip(region, 1.03, 1.70, out=region, where=head)
    # The soft edge of the mask goes with the earliest body next to it: no outline lingers.
    fringe = (mask > 0.01) & (mask <= 0.48)
    release[fringe] = cv2.erode(release, np.ones((9, 9), np.uint8))[fringe]
    return release


def mix(destination: np.ndarray, source: np.ndarray, alpha: np.ndarray) -> None:
    """destination = source over destination, with alpha 0..1 (H, W) float32; OpenCV's SIMD blend."""
    destination[:] = cv2.blendLinear(source, destination, alpha, 1.0 - alpha)


@dataclass(frozen=True)
class Sprite:
    """A BGRA picture split for blending, trimmed to where it is not transparent."""

    color: np.ndarray  # BGR, contiguous
    alpha: np.ndarray  # float32 0..1
    left: int          # where the trimmed picture starts inside the original one
    top: int
    width: int         # size of the original picture, to center it
    height: int


def split_sprite(bgra: np.ndarray) -> Sprite:
    ys, xs = np.nonzero(bgra[:, :, 3])
    if len(xs) == 0:
        return Sprite(np.zeros((0, 0, 3), np.uint8), np.zeros((0, 0), np.float32), 0, 0, *bgra.shape[1::-1])
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    trimmed = bgra[y1:y2, x1:x2]
    return Sprite(np.ascontiguousarray(trimmed[:, :, :3]), trimmed[:, :, 3].astype(np.float32) / 255.0,
                  x1, y1, bgra.shape[1], bgra.shape[0])


def blend_sprite(frame: np.ndarray, sprite: Sprite, x: int, y: int, opacity: float = 1.0) -> None:
    """Paste a sprite with the top-left corner of its original picture at (x, y), clipped to the frame."""
    x, y = x + sprite.left, y + sprite.top
    height, width = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + sprite.alpha.shape[1]), min(height, y + sprite.alpha.shape[0])
    if x1 >= x2 or y1 >= y2 or opacity <= 0:
        return
    alpha = sprite.alpha[y1 - y:y2 - y, x1 - x:x2 - x]
    if opacity < 1:
        alpha = alpha * np.float32(opacity)
    mix(frame[y1:y2, x1:x2], np.ascontiguousarray(sprite.color[y1 - y:y2 - y, x1 - x:x2 - x]), alpha)


def draw_crow(frame: np.ndarray, flock: CrowFlock, crow: Crow, timeline: float) -> None:
    progress = (timeline - crow.launch) / crow.duration
    if not 0 <= progress <= 1:
        return
    path = (crow.origin, crow.control1, crow.control2, crow.destination)
    point = cubic(*path, progress)
    tangent = cubic(*path, min(1.0, progress + 0.015)) - point
    # The sprite faces the viewer: it leans toward its flight a little, without hiding the wings.
    angle = float(np.clip(math.degrees(math.atan2(tangent[1], tangent[0])) * 0.22, -30, 30))
    pose = int(((timeline - crow.launch) * crow.frequency + crow.phase / (2 * math.pi)) * 5) % 5
    grow = 1.0 + 0.38 * progress if crow.depth > 1.1 else 1.0 - 0.42 * progress if crow.depth < 0.8 else 1.0
    if crow.depth > 1.1 and 0.12 < progress < 0.90:
        # Two flight streaks behind the near crows.
        previous = cubic(*path, max(0.0, progress - (0.10 if crow.hero else 0.055)))
        direction = point - previous
        norm = float(np.linalg.norm(direction))
        if norm > 3:
            side = np.array([-direction[1], direction[0]]) / norm
            for sign in (-1, 1):
                offset = side * sign * crow.size * (0.27 if crow.hero else 0.18)
                cv2.line(frame, tuple(np.rint(previous + offset).astype(int)),
                         tuple(np.rint(point + offset * 0.6).astype(int)), (43, 44, 51),
                         2 if crow.hero else 1, cv2.LINE_AA)
    # Sizes in steps of 4 px and angles in steps of 10 degrees, so the caches stay small.
    target_width = max(12, int(round(crow.size * 2.15 * grow / 4) * 4))
    sprite = flock.sprite_cache.get((pose, target_width))
    if sprite is None:
        target_height = max(8, round(target_width * SPRITE_HEIGHT / SPRITE_WIDTH))
        sprite = cv2.resize(flock.sprites[pose], (target_width, target_height),
                            interpolation=cv2.INTER_AREA if target_width < SPRITE_WIDTH else cv2.INTER_LINEAR)
        if len(flock.sprite_cache) < 600:
            flock.sprite_cache[(pose, target_width)] = sprite
    bucket = round(angle / 10) * 10
    rotated = flock.rotated_cache.get((pose, target_width, bucket))
    if rotated is None:
        sh, sw = sprite.shape[:2]
        padding = max(6, int(max(sw, sh) * 0.25))
        transform = cv2.getRotationMatrix2D((sw / 2, sh / 2), bucket, 1)
        transform[:, 2] += padding
        rotated = split_sprite(cv2.warpAffine(sprite, transform, (sw + 2 * padding, sh + 2 * padding),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        if len(flock.rotated_cache) < 800:
            flock.rotated_cache[(pose, target_width, bucket)] = rotated
    # Each crow fades in over 0.1 s: the feathers form the edge of the hole it leaves.
    blend_sprite(frame, rotated, round(point[0] - rotated.width / 2), round(point[1] - rotated.height / 2),
                 min(1.0, (timeline - crow.launch) / 0.10))


def draw_feather(frame: np.ndarray, feather: Feather, timeline: float) -> None:
    progress = (timeline - feather.launch) / feather.duration
    if not 0 <= progress <= 1:
        return
    point = feather.origin * (1.0 - progress) + feather.destination * progress
    point[1] -= math.sin(math.pi * progress) * feather.size * 2.5
    angle = feather.angle + feather.spin * progress
    axis = np.array([math.cos(angle), math.sin(angle)]) * feather.size
    side = np.array([-axis[1], axis[0]])
    margin = max(3, int(feather.size * 2))
    x, y = int(point[0]), int(point[1])
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x - margin), max(0, y - margin), min(width, x + margin + 1), min(height, y + margin + 1)
    if x1 >= x2 or y1 >= y2:
        return
    shape = np.rint([point + axis, point + side * 0.30, point - axis * 0.70, point - side * 0.20]).astype(np.int32)
    shape -= np.array([x1, y1], np.int32)
    destination = frame[y1:y2, x1:x2]
    overlay = destination.copy()
    cv2.fillPoly(overlay, [shape], INK, cv2.LINE_AA)
    cv2.line(overlay, tuple(int(v) for v in shape[2]), tuple(int(v) for v in shape[0]), (53, 55, 63), 1, cv2.LINE_AA)
    opacity = min(1.0, (1.0 - progress) * 3.5, (timeline - feather.launch) * 12)
    cv2.addWeighted(overlay, opacity, destination, 1.0 - opacity, 0, dst=destination)


def draw_ink_edge(frame: np.ndarray, flock: CrowFlock, timeline: float) -> None:
    """A thin ink line where the body is breaking right now."""
    left, top, width, height = flock.bounds
    release = flock.release[top:top + height, left:left + width]
    person = flock.mask[top:top + height, left:left + width]
    front = ((release > timeline - 0.015) & (release < timeline + 0.070) & (person > 0.48)).astype(np.uint8)
    if not front.any():
        return
    edge = cv2.dilate(front, np.ones((3, 3), np.uint8)) > 0
    region = frame[top:top + height, left:left + width]
    region[edge] = (region[edge].astype(np.uint16) * 3 // 10 + np.array(INK, np.uint16) * 7 // 10).astype(np.uint8)


def draw_sharingan(frame: np.ndarray, flock: CrowFlock, timeline: float) -> None:
    """The eyes at the top of the frame, between about 0.3 s and 2.4 s."""
    opacity = smoothstep(0.28, 0.66, timeline) * (1.0 - smoothstep(1.65, 2.38, timeline))
    if opacity <= 0:
        return
    opacity *= 0.93 + 0.07 * math.sin(timeline * 6.0) ** 2
    eyes = flock.sharingan
    blend_sprite(frame, eyes, (frame.shape[1] - eyes.width) // 2, max(3, round(frame.shape[0] * 0.018)), opacity)


def render_flock(background: np.ndarray, flock: CrowFlock, tint: BloodTint, age: float,
                 appearing: bool = False) -> np.ndarray:
    """The frame age seconds into the effect; appearing plays it backwards."""
    if background.shape != flock.source.shape:
        raise ValueError("background and captured frame differ in size")
    if age >= EFFECT_SECONDS:
        return flock.source.copy() if appearing else tint.blend(background, EFFECT_SECONDS)
    timeline = EFFECT_SECONDS - max(0.0, age) if appearing else max(0.0, age)
    output = tint.blend(background, timeline)
    # Far crows fly behind what is left of the person, near ones in front of it.
    for crow in flock.crows:
        if crow.depth < 0.8:
            draw_crow(output, flock, crow, timeline)
    x1, y1, x2, y2 = flock.area
    remaining = np.clip((flock.release[y1:y2, x1:x2] - timeline) * np.float32(1 / 0.055), 0, 1)
    mix(output[y1:y2, x1:x2], np.ascontiguousarray(flock.source[y1:y2, x1:x2]), flock.mask[y1:y2, x1:x2] * remaining)
    draw_ink_edge(output, flock, timeline)
    for feather in flock.feathers:
        draw_feather(output, feather, timeline)
    draw_sharingan(output, flock, timeline)
    for crow in flock.crows:
        if crow.depth >= 0.8:
            draw_crow(output, flock, crow, timeline)
    return output
