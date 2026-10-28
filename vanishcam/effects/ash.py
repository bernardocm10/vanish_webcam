"""Ash: hands up, the silhouette turns to grey ash, crumbles from the head down and leaves a pile on the floor."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..hand_geometry import extended_finger_count
from .goku import face_reference

# The ash spreads from the head down within ASH_SECONDS; the body then crumbles, also from
# the head down, between CRUMBLE_START and CRUMBLE_START + CRUMBLE_SECONDS. Grains fall
# under gravity onto the pile and the dust settles by EFFECT_SECONDS.
ASH_SECONDS = 0.8
CRUMBLE_START = 1.0
CRUMBLE_SECONDS = 1.2
EFFECT_SECONDS = 3.8
DUST_PUFFS = 90
# Both hands up must be held for a moment: raising them to stretch passes through the pose.
HANDS_UP_HOLD_SECONDS = 0.4
# Face Mesh: the chin; hands up means both palms above it.
CHIN = 152


def hands_up_reference(face_landmarks, width: int, height: int) -> tuple[float, float, float] | None:
    """The face's horizontal center, the chin height and the face width, in pixels."""
    base = face_reference(face_landmarks, width, height)
    if base is None or len(face_landmarks) <= CHIN:
        return None
    top, bridge, face_width = base
    chin_y = face_landmarks[CHIN].y * height
    if not np.isfinite(chin_y):
        return None
    return float(bridge[0]), float(chin_y), face_width


def extended_fingers(points: np.ndarray) -> int:
    """Index to little finger whose tip is clearly farther from the wrist than its middle joint."""
    return extended_finger_count(points, ratio=1.15)


def hands_up_pose(hands, face, width: int, height: int) -> bool:
    """Both hands raised beside the head like a startled scream: open, fingers up, palms above the chin.

    One hand on each side of the face, so a single raised hand or a wave does not count.
    """
    if face is None or len(hands) < 2:
        return False
    center_x, chin_y, face_width = face
    sides = set()
    for hand in hands[:2]:
        if len(hand) != 21:
            return False
        points = np.array([(p.x * width, p.y * height) for p in hand], dtype=np.float32)
        if not np.isfinite(points).all():
            return False
        palm = points[[0, 5, 9, 13, 17]].mean(axis=0)
        fingertips = points[[8, 12, 16, 20]].mean(axis=0)
        upright = points[0][1] - fingertips[1] > 0.3 * face_width
        offset = palm[0] - center_x
        if (
            extended_fingers(points) < 3
            or not upright
            or palm[1] > chin_y
            or not 0.45 * face_width < abs(offset) < 4.0 * face_width
        ):
            return False
        sides.add(offset > 0)
    return len(sides) == 2


@dataclass
class AshPile:
    """The mound left on the floor: an ash texture plus a height level per pixel (0 floor, 1 top)."""

    box: tuple[int, int, int, int]  # x, y, width, height in the frame
    texture: np.ndarray             # BGR, uint8
    level: np.ndarray               # float32; a pixel shows once the pile has grown past its level
    edge: np.ndarray                # float32 0..1 softness of the sides
    floor: np.ndarray               # (W,) y where a grain falling at that column lands


@dataclass
class AshCloud:
    """Silhouette captured once when the effect starts; everything else is computed."""

    body: np.ndarray           # frame captured at the start
    ash: np.ndarray            # the same silhouette as grey ash
    body_mask: np.ndarray      # mask captured at the start
    box: tuple[int, int, int, int]  # silhouette bounding box: x, y, width, height
    ash_delay: np.ndarray      # per pixel: when it turns to ash
    crumble_delay: np.ndarray  # per pixel: when it breaks off
    origin: np.ndarray         # (N, 2) grain centers, pixels (x, y)
    color: np.ndarray          # (N, 3) ash color of each grain, BGR
    delay: np.ndarray          # (N,) when the grain breaks off
    velocity: np.ndarray       # (N, 2) initial velocity, pixels per second
    fall: np.ndarray           # (N,) seconds until the grain reaches the pile or the floor
    landed: np.ndarray         # (N,) sorted landing instants: how much of the pile has grown
    gravity: float             # pixels/s^2
    size: int                  # grain side in pixels
    pile: AshPile
    puffs: np.ndarray          # (M, 6) start, x, y, rise speed, max radius, life


def noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float, amount: float) -> np.ndarray:
    field = rng.standard_normal(shape, dtype=np.float32)
    if sigma > 0:
        field = cv2.GaussianBlur(field, (0, 0), sigma)
        field /= max(float(field.std()), 1e-6)
    return field * amount


def ash_color(value: np.ndarray) -> np.ndarray:
    """Grey value -> warm grey BGR (a touch less blue than red)."""
    value = value.astype(np.float32)
    return cv2.merge((value - 5.0, value - 2.0, value + 3.0)).clip(0, 255).astype(np.uint8)


def ash_texture(frame: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int],
                rng: np.random.Generator) -> np.ndarray:
    """Powdery grey ash with the person's shading: flat and matte, cracked where there are edges.

    Only the silhouette box is computed; the rest of the frame stays black (it is never shown).
    """
    ash = np.zeros_like(frame)
    x, y, w, h = box
    area = np.s_[y:y + h, x:x + w]
    gray = cv2.cvtColor(frame[area], cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside = mask[area] > 0.5
    if np.any(inside):
        low, high = np.percentile(gray[inside], (3, 97))
        gray = (gray - low) / max(float(high - low), 1.0)
    else:
        gray = gray / 255.0
    # Ash is matte: a narrow range of light greys.
    value = 105.0 + 105.0 * np.clip(gray, 0.0, 1.0)
    # Unsharp mask: eyes, mouth and folds read as cracks.
    value += 1.1 * (value - cv2.GaussianBlur(value, (0, 0), 2.5))
    # Powder: coarse grain, blotches, and dark specks of burnt material.
    value += noise(rng, gray.shape, 0.7, 8.0) + noise(rng, gray.shape, 9.0, 7.0)
    specks = rng.random(gray.shape, dtype=np.float32)
    value -= (specks < 0.012) * 40.0
    value += (specks > 0.992) * 22.0
    ash[area] = ash_color(value)
    return ash


def make_pile(mask: np.ndarray, box: tuple[int, int, int, int], rng: np.random.Generator) -> AshPile:
    height, width = mask.shape
    x, y, w, h = box
    rows = mask[max(y, y + h - max(2, h // 10)):y + h] > 0.5
    columns = x + np.nonzero(rows.any(axis=0)[x:x + w])[0] if rows.size else np.array([], int)
    center = float(columns.mean()) if columns.size else x + w / 2
    half = float(np.clip(0.6 * (np.ptp(columns) if columns.size else w), 0.12 * width, 0.32 * width))
    peak = float(np.clip(0.42 * half, 0.08 * height, 0.2 * height))
    bottom = int(min(height, y + h + 0.02 * height))

    left, right = int(max(0, center - half)), int(min(width, center + half))
    xs = np.arange(left, right, dtype=np.float32)
    u = (xs - center) / half
    # A few broad lumps, then a fine ragged crest that does not change the shading.
    mound = np.clip(1.0 - u * u, 0.0, 1.0) ** 1.4 * (1.0 + noise(rng, (1, max(1, xs.size)), half / 5, 0.16)[0])
    smooth = np.maximum(peak * mound, 0.0)
    profile = np.maximum(smooth * (1.0 + noise(rng, (1, max(1, xs.size)), 1.5, 0.02)[0]), 0.0)
    top = int(max(0, bottom - np.ceil(profile.max() if profile.size else 1) - 2))
    box_height = max(1, bottom - top)
    above = (bottom - np.arange(top, bottom, dtype=np.float32))[:, None]  # Height above the floor.
    level = above / np.maximum(profile[None, :], 1e-3)
    level[:, profile < 1.0] = 2.0
    edge = np.repeat(np.clip((1.0 - np.abs(u)) * 8.0, 0.0, 1.0)[None, :], box_height, axis=0)

    # Lit from the upper left: the slope facing left is lighter, the crest lighter than the base.
    slope = np.gradient(smooth)[None, :] / max(peak, 1.0) * half / 4.0
    shape = (box_height, max(1, xs.size))
    value = (125.0 + 55.0 * np.clip(level, 0.0, 1.0) ** 0.5 + 30.0 * np.clip(slope, -1.0, 1.0)
             + noise(rng, shape, 0.8, 11.0) + noise(rng, shape, 6.0, 9.0))
    specks = rng.random(shape, dtype=np.float32)
    value -= (specks < 0.03) * 45.0
    texture = ash_color(np.clip(value, 0.0, 255.0))

    floor = np.full(width, float(bottom), np.float32)
    floor[left:right] = bottom - profile * 0.55
    return AshPile((left, top, right - left, box_height), texture, level.astype(np.float32),
                   edge.astype(np.float32), floor)


def make_ash(frame: np.ndarray, mask: np.ndarray, seed: int = 2024) -> AshCloud:
    height, width = mask.shape
    rng = np.random.default_rng(seed)
    scale = height / 720.0
    binary = (mask > 0.5).astype(np.uint8)
    box = cv2.boundingRect(binary)
    if box[2] == 0 or box[3] == 0:
        box = (0, 0, width, height)
    x0, y0, w, h = box

    # Depth into the body, top to bottom, with blotches so the front is not a straight line.
    rows = (np.arange(height, dtype=np.float32)[:, None] - y0) / max(h, 1)
    blotches = cv2.resize(rng.random((max(2, height // 40), max(2, width // 40)), dtype=np.float32),
                          (width, height), interpolation=cv2.INTER_LINEAR)
    ash_delay = ASH_SECONDS * np.clip(rows * 0.85 + blotches * 0.15, 0.0, 1.0)

    size = max(2, width // 320)
    grid = (max(1, width // size), max(1, height // size))
    ash = ash_texture(frame, mask, box, rng)
    coverage = cv2.resize(mask.astype(np.float32), grid, interpolation=cv2.INTER_AREA)
    colors = cv2.resize(ash, grid, interpolation=cv2.INTER_AREA)
    cells_y, cells_x = np.nonzero(coverage > 0.5)
    count = cells_y.size
    origin = np.column_stack(((cells_x + 0.5) * size, (cells_y + 0.5) * size)).astype(np.float32)
    grain_depth = np.clip((origin[:, 1] - y0) / max(h, 1), 0.0, 1.0)
    delay = CRUMBLE_START + CRUMBLE_SECONDS * np.clip(grain_depth * 0.85 + rng.uniform(0.0, 0.15, count), 0.0, 1.0)

    cell_delay = np.zeros((grid[1], grid[0]), np.float32)
    cell_delay[cells_y, cells_x] = delay
    # Edge cells break off with their neighbors, or a ghost outline would remain.
    neighbours = cv2.dilate(cell_delay, np.ones((3, 3), np.uint8))
    is_grain = np.zeros_like(cell_delay, bool)
    is_grain[cells_y, cells_x] = True
    cell_delay = np.where(is_grain, cell_delay, neighbours)
    crumble_delay = cv2.resize(cell_delay, (width, height), interpolation=cv2.INTER_NEAREST)

    pile = make_pile(mask, box, rng)
    gravity = 1500.0 * scale
    velocity = np.column_stack((rng.normal(0.0, 22.0, count) * scale,
                                rng.uniform(-25.0, 35.0, count) * scale)).astype(np.float32)
    columns = np.clip(origin[:, 0].astype(int), 0, width - 1)
    drop = np.maximum(pile.floor[columns] - origin[:, 1], 1.0)
    fall = (-velocity[:, 1] + np.sqrt(velocity[:, 1] ** 2 + 2.0 * gravity * drop)) / gravity
    # Every grain must land before the effect ends.
    fall = np.minimum(fall, EFFECT_SECONDS - 0.25 - delay).clip(0.05)

    # Dust: puffs where the body crumbles, then from the growing pile.
    puffs = np.zeros((DUST_PUFFS, 6), np.float32)
    if count:
        picked = rng.integers(0, count, DUST_PUFFS)
        from_pile = np.arange(DUST_PUFFS) >= DUST_PUFFS * 0.6
        puffs[:, 0] = delay[picked] + rng.uniform(0.0, 0.15, DUST_PUFFS)
        puffs[:, 1] = origin[picked, 0]
        puffs[:, 2] = np.where(from_pile, pile.floor[columns[picked]], origin[picked, 1])
        puffs[from_pile, 0] = np.minimum((delay + fall)[picked[from_pile]], EFFECT_SECONDS - 0.9)
    else:
        puffs[:, 0] = EFFECT_SECONDS
    puffs[:, 3] = rng.uniform(45.0, 120.0, DUST_PUFFS) * scale
    puffs[:, 4] = rng.uniform(0.07, 0.16, DUST_PUFFS) * height
    # Every puff fades out before the effect ends: the last frame is just the room and the pile.
    puffs[:, 5] = np.maximum(0.2, np.minimum(rng.uniform(1.1, 1.8, DUST_PUFFS), EFFECT_SECONDS - puffs[:, 0]))

    return AshCloud(
        body=frame.copy(), ash=ash, body_mask=mask.astype(np.float32), box=box,
        ash_delay=ash_delay, crumble_delay=crumble_delay,
        origin=origin, color=colors[cells_y, cells_x].astype(np.float32), delay=delay.astype(np.float32),
        velocity=velocity, fall=fall.astype(np.float32), landed=np.sort(delay + fall).astype(np.float32),
        gravity=gravity, size=size, pile=pile, puffs=puffs,
    )


def draw_pile(canvas: np.ndarray, pile: AshPile, grown: float) -> None:
    if grown <= 0.0:
        return
    x, y, w, h = pile.box
    if w <= 0 or h <= 0:
        return
    # Grows from the floor up: a pixel appears once the pile has grown past its level.
    alpha = np.clip((grown - pile.level) / 0.08, 0.0, 1.0) * pile.edge
    area = np.s_[y:y + h, x:x + w]
    canvas[area] = cv2.blendLinear(pile.texture, np.ascontiguousarray(canvas[area]),
                                   np.ascontiguousarray(alpha), np.ascontiguousarray(1.0 - alpha))


_DUST_SPRITES: dict[tuple[int, int], np.ndarray] = {}


def dust_sprite(radius: int, variant: int) -> np.ndarray:
    """A soft, wispy smoke puff, wider than tall; a few variants so the puffs do not repeat."""
    key = (radius, variant)
    sprite = _DUST_SPRITES.get(key)
    if sprite is None:
        rng = np.random.default_rng(variant)
        ys = np.arange(-radius, radius + 1, dtype=np.float32)[:, None] / max(radius, 1)
        xs = np.arange(-radius, radius + 1, dtype=np.float32)[None, :] / max(radius, 1)
        distance = np.sqrt(xs * xs * 0.7 + ys * ys * 1.3)
        wisps = noise(rng, distance.shape, max(1.0, radius / 6), 0.45)
        sprite = np.clip(np.exp(-2.6 * distance * distance) * (1.0 + wisps), 0.0, 1.0)
        sprite[distance >= 1.0] = 0.0
        _DUST_SPRITES[key] = sprite = sprite.astype(np.float32)
    return sprite


def draw_dust(canvas: np.ndarray, cloud: AshCloud, elapsed: float) -> None:
    height, width = canvas.shape[:2]
    for index, (start, x, y, rise, radius_max, life) in enumerate(cloud.puffs):
        age = elapsed - start
        if age <= 0.0 or age >= life:
            continue
        progress = age / life
        # Quantized radius: a handful of cached sprites instead of one per frame.
        radius = int(radius_max * (0.3 + 0.7 * progress ** 0.5))
        radius = max(3, radius - radius % 4)
        strength = 0.5 * np.sin(np.pi * progress) ** 0.7
        cx = int(x + 0.4 * radius * np.sin(start * 9.0 + age * 1.5))
        cy = int(y - rise * age)
        sprite = dust_sprite(radius, index % 4)
        x1, y1 = cx - radius, cy - radius
        left, top = max(0, x1), max(0, y1)
        right, bottom = min(width, x1 + sprite.shape[1]), min(height, y1 + sprite.shape[0])
        if left >= right or top >= bottom:
            continue
        alpha = sprite[top - y1:bottom - y1, left - x1:right - x1] * strength
        area = np.ascontiguousarray(canvas[top:bottom, left:right])
        dust = np.empty_like(area)
        dust[:] = (188, 191, 195)
        canvas[top:bottom, left:right] = cv2.blendLinear(dust, area, alpha, 1.0 - alpha)


def render_ash(background: np.ndarray, cloud: AshCloud, elapsed: float) -> np.ndarray:
    canvas = background.copy()
    height, width = canvas.shape[:2]

    # The body remains, turning to ash, where it has not crumbled yet.
    attached = cloud.body_mask * np.clip((cloud.crumble_delay - elapsed) / 0.08, 0.0, 1.0)
    x, y, w, h = cv2.boundingRect((attached > 0.01).view(np.uint8))
    if w and h:
        area = np.s_[y:y + h, x:x + w]
        if elapsed >= ASH_SECONDS + 0.25:
            look = cloud.ash[area]
        else:
            turned = np.clip((elapsed - cloud.ash_delay[area]) / 0.25, 0.0, 1.0)
            look = cv2.blendLinear(np.ascontiguousarray(cloud.ash[area]), np.ascontiguousarray(cloud.body[area]),
                                   turned, 1.0 - turned)
        alpha = np.ascontiguousarray(attached[area])
        canvas[area] = cv2.blendLinear(np.ascontiguousarray(look), np.ascontiguousarray(canvas[area]),
                                       alpha, 1.0 - alpha)

    grown = np.searchsorted(cloud.landed, elapsed) / max(len(cloud.landed), 1)
    draw_pile(canvas, cloud.pile, float(grown))

    flight = elapsed - cloud.delay
    active = (flight >= 0.0) & (flight < cloud.fall)
    if np.any(active):
        flight = flight[active, None]
        position = cloud.origin[active] + cloud.velocity[active] * flight
        position[:, 1] += 0.5 * cloud.gravity * flight[:, 0] ** 2
        offsets = np.arange(cloud.size) - cloud.size // 2
        xs = (np.rint(position[:, 0])[:, None, None] + offsets[None, None, :]).astype(np.int64)
        ys = (np.rint(position[:, 1])[:, None, None] + offsets[None, :, None]).astype(np.int64)
        xs, ys = np.broadcast_arrays(xs, ys)
        paint = np.broadcast_to(cloud.color[active][:, None, None, :], xs.shape + (3,))
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        canvas[ys[inside], xs[inside]] = paint[inside].astype(np.uint8)

    draw_dust(canvas, cloud, elapsed)
    return canvas


def render_remains(background: np.ndarray, cloud: AshCloud) -> np.ndarray:
    """What stays once the effect is over: the room and the pile of ash."""
    canvas = background.copy()
    draw_pile(canvas, cloud.pile, 1.0)
    return canvas
