"""Itachi's crows and Sharingan, drawn in code: no pictures to ship.

Both are drawn larger than needed with OpenCV's anti-aliased shapes and scaled down,
so the edges stay smooth. The crow is seen from the side, flying right, in five
poses of a wing beat; the eyes are two almond eyes with a red three-tomoe iris.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

SPRITE_WIDTH, SPRITE_HEIGHT = 176, 128
CROW_SUPERSAMPLE = 4
# Where the near wing points in each pose of the beat, in degrees counter-clockwise from
# "forward" (right): up, half up, level, down, and half up again on the way back.
WING_ANGLES = (100.0, 128.0, 165.0, 205.0, 140.0)

# Cel shading: near black with a cold blue sheen, BGR.
CROW_DARK = (24, 21, 22)
CROW_MID = (46, 40, 40)
CROW_SHEEN = (104, 86, 74)
CROW_EDGE = (10, 9, 10)
BEAK = (58, 56, 60)


def _points(*xy) -> np.ndarray:
    return np.rint(np.array(xy, np.float64)).astype(np.int32)


def _wing(shoulder: np.ndarray, angle: float, span: float, feathers: int = 7) -> tuple[np.ndarray, list]:
    """Wing outline and the feather separations, for a wing pointing at angle (degrees)."""
    radians = math.radians(angle)
    # u along the span, v towards the trailing edge (behind the leading edge).
    u = np.array([math.cos(radians), -math.sin(radians)])
    v = np.array([-math.sin(radians), -math.cos(radians)])

    def at(x: float, y: float) -> np.ndarray:
        return shoulder + u * (x * span) + v * (y * span)

    leading = [at(0.0, 0.0), at(0.22, -0.06), at(0.46, -0.09), at(0.70, -0.07), at(0.92, -0.02)]
    trailing = []
    tips = []
    for k in range(feathers):
        # The crow's "fingers": long primaries splayed at the tip with deep gaps between
        # them, then shorter, rounder secondaries towards the body.
        primary = k < 4
        along = 1.02 - k * 0.07 if primary else 0.76 - (k - 4) * 0.17
        depth = 0.04 + k * 0.065
        tip = at(along, depth + (0.05 if primary else 0.12))
        notch = at(along - (0.12 if primary else 0.07), depth + (0.03 if primary else 0.07))
        trailing += [tip, notch]
        tips.append(tip)
    trailing.append(at(0.03, 0.44))
    outline = np.array(leading + trailing, np.float64)
    wrist = at(0.46, -0.03)
    separations = [(wrist, tip) for tip in tips[1:]]
    return outline, separations


def _crow(angle: float) -> np.ndarray:
    """One pose, BGRA at the sprite size."""
    s = CROW_SUPERSAMPLE
    width, height = SPRITE_WIDTH * s, SPRITE_HEIGHT * s
    color = np.zeros((height, width, 3), np.uint8)
    alpha = np.zeros((height, width), np.uint8)
    line = cv2.LINE_AA

    def fill(points: np.ndarray, shade) -> None:
        points = np.rint(points).astype(np.int32)
        cv2.fillPoly(color, [points], shade, line)
        cv2.fillPoly(alpha, [points], 255, line)

    body_center = np.array([width * 0.46, height * 0.64])
    body_axes = (width * 0.19, height * 0.12)
    shoulder = body_center + np.array([width * 0.03, -height * 0.06])
    # The wing fits the sprite in every pose: up, it reaches near the top edge.
    span = width * 0.40

    # The far wing first, behind the body: a little shorter, a beat behind, darker.
    far, _ = _wing(shoulder + np.array([width * 0.05, -height * 0.02]), angle + 22, span * 0.8)
    fill(far, CROW_EDGE)

    # Tail: a fan of feathers behind the body.
    tail_root = body_center - np.array([body_axes[0] * 0.85, -height * 0.01])
    tail = [tail_root + np.array([0, -height * 0.05])]
    for k in range(5):
        spread = math.radians(-14 + k * 7)
        length = width * (0.20 + 0.02 * (2 - abs(k - 2)))
        tip = tail_root + np.array([-math.cos(spread) * length, math.sin(spread) * length + height * 0.03])
        tail.append(tip)
        if k < 4:
            tail.append(tail_root + np.array([-math.cos(spread + 0.06) * length * 0.86,
                                              math.sin(spread + 0.06) * length * 0.86 + height * 0.03]))
    tail.append(tail_root + np.array([0, height * 0.05]))
    fill(np.array(tail), CROW_DARK)

    # Body and head.
    cv2.ellipse(color, tuple(np.rint(body_center).astype(int)), tuple(int(a) for a in body_axes), -8, 0, 360,
                CROW_DARK, -1, line)
    cv2.ellipse(alpha, tuple(np.rint(body_center).astype(int)), tuple(int(a) for a in body_axes), -8, 0, 360,
                255, -1, line)
    head = body_center + np.array([body_axes[0] * 1.05, -height * 0.10])
    head_radius = int(height * 0.105)
    neck = _points(body_center + [body_axes[0] * 0.4, -body_axes[1] * 0.9],
                   head + [-head_radius * 0.3, -head_radius * 0.9],
                   head + [head_radius * 0.2, head_radius * 0.95],
                   body_center + [body_axes[0] * 0.75, body_axes[1] * 0.5])
    fill(neck, CROW_DARK)
    cv2.circle(color, tuple(np.rint(head).astype(int)), head_radius, CROW_DARK, -1, line)
    cv2.circle(alpha, tuple(np.rint(head).astype(int)), head_radius, 255, -1, line)
    # The heavy crow beak.
    beak = _points(head + [head_radius * 0.55, -head_radius * 0.45], head + [head_radius * 2.0, head_radius * 0.05],
                   head + [head_radius * 0.6, head_radius * 0.45])
    fill(beak, BEAK)
    cv2.line(color, tuple(beak[0]), tuple(beak[1]), (90, 88, 92), max(1, s // 2), line)
    # Sheen on the back and the crown: the cel-shaded highlight.
    cv2.ellipse(color, tuple(np.rint(body_center + [width * 0.01, -body_axes[1] * 0.45]).astype(int)),
                (int(body_axes[0] * 0.75), int(body_axes[1] * 0.28)), -10, 190, 350, CROW_SHEEN, max(2, s), line)
    cv2.ellipse(color, tuple(np.rint(head).astype(int)), (int(head_radius * 0.7), int(head_radius * 0.7)), 0,
                200, 300, CROW_SHEEN, max(2, s), line)
    # The eye: a pale ring and a glint.
    eye = tuple(np.rint(head + [head_radius * 0.25, -head_radius * 0.15]).astype(int))
    cv2.circle(color, eye, max(2, int(head_radius * 0.22)), (200, 205, 210), -1, line)
    cv2.circle(color, eye, max(1, int(head_radius * 0.12)), (5, 5, 8), -1, line)

    # The near wing, in front of the body, with its sheen and feathers.
    near, separations = _wing(shoulder, angle, span)
    fill(near, CROW_MID)
    # The sheen along the leading edge.
    cv2.polylines(color, [np.rint(near[:5] * 0.94 + near[-1] * 0.06).astype(np.int32)], False, CROW_SHEEN,
                  max(2, s), line)
    for start, end in separations:
        # A dark gap between feathers with a sheen line beside it.
        a, b = np.rint(start).astype(int), np.rint(end).astype(int)
        cv2.line(color, tuple(a), tuple(b), CROW_EDGE, max(1, s // 2), line)
        cv2.line(color, tuple(a + [s, s]), tuple(b + [s, s]), CROW_SHEEN, max(1, s // 3), line)
    cv2.polylines(color, [np.rint(near).astype(np.int32)], True, CROW_EDGE, max(1, s // 2), line)

    small_color = cv2.resize(color, (SPRITE_WIDTH, SPRITE_HEIGHT), interpolation=cv2.INTER_AREA)
    small_alpha = cv2.resize(alpha, (SPRITE_WIDTH, SPRITE_HEIGHT), interpolation=cv2.INTER_AREA)
    return np.dstack((small_color, small_alpha))


@lru_cache(maxsize=1)
def crow_sprites() -> tuple[np.ndarray, ...]:
    """The five poses of the wing beat, BGRA, each SPRITE_HEIGHT x SPRITE_WIDTH, the crow flying right."""
    return tuple(_crow(angle) for angle in WING_ANGLES)


EYE_SUPERSAMPLE = 2
SCLERA = (232, 234, 240)
IRIS = (30, 22, 205)
IRIS_DARK = (22, 12, 120)
INK = (12, 10, 14)
BLOOD = (24, 18, 160)


def _eye_outline(center: np.ndarray, half_width: float, half_height: float, mirror: bool) -> np.ndarray:
    """An almond eye, its outer corner a little higher, as a closed polygon."""
    steps = np.linspace(-1.0, 1.0, 41)
    side = -1.0 if mirror else 1.0
    upper = [(center[0] + t * half_width, center[1] - half_height * (1 - t * t) ** 0.8 * (1 + 0.18 * t * side)
              - half_height * 0.12 * t * side) for t in steps]
    lower = [(center[0] + t * half_width, center[1] + half_height * 0.62 * (1 - t * t) ** 0.9
              - half_height * 0.12 * t * side) for t in steps[::-1]]
    return np.array(upper + lower, np.float64)


def _tomoe(canvas: np.ndarray, center: np.ndarray, ring: float, angle: float, size: float) -> None:
    """A comma: a round head on the ring and a tail that tapers along it."""
    head = center + ring * np.array([math.cos(angle), math.sin(angle)])
    cv2.circle(canvas, tuple(np.rint(head).astype(int)), int(size), INK, -1, cv2.LINE_AA)
    outer, inner = [], []
    for k in range(13):
        a = angle - k * math.radians(5)
        taper = size * (1 - k / 12)
        point = center + ring * np.array([math.cos(a), math.sin(a)])
        normal = np.array([math.cos(a), math.sin(a)])
        outer.append(point + normal * taper)
        inner.append(point - normal * taper * 0.35)
    tail = np.array(outer + inner[::-1])
    cv2.fillPoly(canvas, [np.rint(tail).astype(np.int32)], INK, cv2.LINE_AA)


def _one_eye(color: np.ndarray, alpha: np.ndarray, center: np.ndarray, half_width: float, mirror: bool,
             tears: bool) -> None:
    half_height = half_width * 0.42
    outline = _eye_outline(center, half_width, half_height, mirror)
    shape = np.rint(outline).astype(np.int32)
    # Glow around the eye, then the white.
    glow = np.zeros(alpha.shape, np.uint8)
    cv2.fillPoly(glow, [shape], 255, cv2.LINE_AA)
    # Blurred at a quarter of the size: a wide blur is slow at full size, and a glow is soft anyway.
    small = cv2.resize(cv2.dilate(glow, np.ones((9, 9), np.uint8)), None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), half_width * 0.03)
    glow = cv2.resize(small, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
    # A pale glow, as if the eyes were lit from inside.
    np.maximum(alpha, (glow.astype(np.float32) * 0.7).astype(np.uint8), out=alpha)
    color[glow > 0] = (200, 200, 215)
    cv2.fillPoly(color, [shape], SCLERA, cv2.LINE_AA)
    cv2.fillPoly(alpha, [shape], 255, cv2.LINE_AA)

    # The iris, cut by the lids.
    iris = np.zeros_like(color)
    radius = half_height * 1.02
    iris_center = center + np.array([half_width * 0.04 * (-1 if mirror else 1), half_height * 0.05])
    c = tuple(np.rint(iris_center).astype(int))
    cv2.circle(iris, c, int(radius), IRIS_DARK, -1, cv2.LINE_AA)
    cv2.circle(iris, c, int(radius * 0.9), IRIS, -1, cv2.LINE_AA)
    # A lighter band, then the thin ring the tomoe sit on.
    cv2.circle(iris, c, int(radius * 0.72), (60, 50, 235), max(1, int(radius * 0.08)), cv2.LINE_AA)
    cv2.circle(iris, c, int(radius * 0.55), (20, 12, 110), max(1, int(radius * 0.035)), cv2.LINE_AA)
    cv2.circle(iris, c, int(radius * 0.2), INK, -1, cv2.LINE_AA)
    for k in range(3):
        _tomoe(iris, iris_center, radius * 0.55, math.radians(-90 + 120 * k), radius * 0.13)
    inside = np.zeros(alpha.shape, np.uint8)
    cv2.fillPoly(inside, [shape], 255, cv2.LINE_AA)
    lid = cv2.circle(np.zeros(alpha.shape, np.uint8), c, int(radius), 255, -1, cv2.LINE_AA)
    cut = np.minimum(inside, lid).astype(np.float32) * np.float32(1 / 255)
    color[:] = cv2.blendLinear(iris, np.ascontiguousarray(color), cut, 1.0 - cut)

    # A heavy upper lid that runs past the outer corner, and a fold above it.
    upper = shape[:41]
    side = -1 if mirror else 1
    wing = upper[-1] if not mirror else upper[0]
    extended = np.vstack([upper, [wing + np.array([side * half_width * 0.22, -half_height * 0.25]).astype(np.int32)]]) \
        if not mirror else np.vstack([[wing + np.array([side * half_width * 0.22, -half_height * 0.25]).astype(np.int32)], upper])
    thickness = max(2, int(half_height * 0.16))
    cv2.polylines(color, [extended], False, INK, thickness, cv2.LINE_AA)
    cv2.polylines(alpha, [extended], False, 255, thickness, cv2.LINE_AA)
    fold = (upper[6:36] + np.array([0, -int(half_height * 0.35)])).astype(np.int32)
    cv2.polylines(color, [fold], False, INK, max(1, thickness // 2), cv2.LINE_AA)
    cv2.polylines(alpha, [fold], False, 230, max(1, thickness // 2), cv2.LINE_AA)
    cv2.polylines(color, [shape[41:]], False, (70, 64, 70), max(1, thickness // 3), cv2.LINE_AA)

    if tears:
        # Two drops running down from the lower lid.
        for offset, length in ((-0.22, 2.6), (0.12, 1.7)):
            x = center[0] + half_width * offset
            top = center[1] + half_height * 0.55
            bottom = top + half_height * length
            width = half_height * 0.11
            trail = _points((x - width, top), (x + width, top), (x + width * 0.7, bottom), (x - width * 0.7, bottom))
            cv2.fillPoly(color, [trail], BLOOD, cv2.LINE_AA)
            cv2.fillPoly(alpha, [trail], 255, cv2.LINE_AA)
            drop = (int(x), int(bottom))
            cv2.circle(color, drop, int(width * 1.5), BLOOD, -1, cv2.LINE_AA)
            cv2.circle(alpha, drop, int(width * 1.5), 255, -1, cv2.LINE_AA)
            cv2.circle(color, (int(x - width * 0.4), int(bottom - width * 0.4)), max(1, int(width * 0.45)),
                       (150, 140, 235), -1, cv2.LINE_AA)


@lru_cache(maxsize=4)
def sharingan(width: int) -> np.ndarray:
    """The two eyes side by side, BGRA, width pixels wide, on a transparent background."""
    s = EYE_SUPERSAMPLE
    big_width = width * s
    big_height = round(big_width * 0.42)
    color = np.zeros((big_height, big_width, 3), np.uint8)
    alpha = np.zeros((big_height, big_width), np.uint8)
    half_width = big_width * 0.17
    for x, mirror, tears in ((0.22, False, True), (0.78, True, False)):
        # Each eye is drawn on its own strip of the picture: much less to blur and blend.
        x0 = max(0, round(big_width * x - half_width * 1.45))
        x1 = min(big_width, round(big_width * x + half_width * 1.45))
        _one_eye(color[:, x0:x1], alpha[:, x0:x1], np.array([big_width * x - x0, big_height * 0.40]),
                 half_width, mirror, tears)
    height = max(1, round(big_height / s))
    small_alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    # The far tail of the glow is invisible: cut it, so the picture has no faint edge.
    small_alpha[small_alpha < 4] = 0
    return np.dstack((cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA), small_alpha))
