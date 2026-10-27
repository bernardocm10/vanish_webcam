"""Particles: the silhouette breaks into colored fragments carried by the wind."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


EFFECT_SECONDS = 2.1
# I frammenti partono entro DEPART_SECONDS e svaniscono entro LIFE_SECONDS[1]:
# la somma resta dentro EFFECT_SECONDS.
DEPART_SECONDS = 1.0
LIFE_SECONDS = (0.7, 1.1)


@dataclass
class ParticleCloud:
    """Silhouette captured once when the effect starts; everything else is computed."""

    origin: np.ndarray      # (N, 2) centro di ogni frammento, in pixel (x, y)
    color: np.ndarray       # (N, 3) colore medio della cella, BGR
    delay: np.ndarray       # (N,) secondi prima che il frammento si stacchi
    life: np.ndarray        # (N,) durata del volo
    velocity: np.ndarray    # (N, 2) pixel al secondo
    flutter: np.ndarray     # (N,) fase dell'oscillazione verticale
    size: int               # lato del frammento in pixel
    body: np.ndarray        # frame catturato all'avvio
    body_mask: np.ndarray   # maschera catturata all'avvio
    body_delay: np.ndarray  # ritardo per pixel: il corpo si sgretola insieme ai frammenti
    wind: np.ndarray        # accelerazione del vento, pixel/s^2


def make_particles(frame: np.ndarray, mask: np.ndarray, seed: int = 2024) -> ParticleCloud:
    height, width = mask.shape
    rng = np.random.default_rng(seed)
    size = max(3, width // 240)
    grid = (max(1, width // size), max(1, height // size))
    coverage = cv2.resize(mask, grid, interpolation=cv2.INTER_AREA)
    colors = cv2.resize(frame, grid, interpolation=cv2.INTER_AREA)
    rows, cols = np.nonzero(coverage > 0.5)
    count = rows.size
    scale = width / 1280.0

    x_norm = (cols + 0.5) / grid[0]
    y_norm = (rows + 0.5) / grid[1]
    # Il vento soffia verso destra: si sgretola prima il lato destro e l'alto.
    sweep = 0.65 * (1.0 - x_norm) + 0.35 * y_norm
    if count:
        sweep = (sweep - sweep.min()) / max(float(np.ptp(sweep)), 1e-6)
    delay = DEPART_SECONDS * np.clip(sweep * 0.85 + rng.uniform(0.0, 0.15, count), 0.0, 1.0)

    cell_delay = np.zeros((grid[1], grid[0]), dtype=np.float32)
    cell_delay[rows, cols] = delay
    # Le celle di bordo (coperte a meta') non diventano frammenti: si staccano
    # con i vicini, altrimenti resterebbe un contorno fantasma.
    neighbours = cv2.dilate(cell_delay, np.ones((3, 3), np.uint8))
    is_particle = np.zeros_like(cell_delay, dtype=bool)
    is_particle[rows, cols] = True
    cell_delay = np.where(is_particle, cell_delay, neighbours)
    body_delay = cv2.resize(cell_delay, (width, height), interpolation=cv2.INTER_NEAREST)

    velocity = np.column_stack((
        rng.uniform(180.0, 420.0, count) * scale,
        rng.uniform(-120.0, 20.0, count) * scale,
    )).astype(np.float32)
    return ParticleCloud(
        origin=np.column_stack(((cols + 0.5) * size, (rows + 0.5) * size)).astype(np.float32),
        color=colors[rows, cols].astype(np.float32),
        delay=delay.astype(np.float32),
        life=rng.uniform(*LIFE_SECONDS, count).astype(np.float32),
        velocity=velocity,
        flutter=rng.uniform(0.0, 2.0 * np.pi, count).astype(np.float32),
        size=size,
        body=frame.copy(),
        body_mask=mask.astype(np.float32),
        body_delay=body_delay,
        wind=np.array((260.0, -70.0), dtype=np.float32) * scale,
    )


def render_particles(background: np.ndarray, cloud: ParticleCloud, elapsed: float) -> np.ndarray:
    canvas = background.copy()
    height, width = canvas.shape[:2]

    # Il corpo resta dove i frammenti non sono ancora partiti.
    attached = cloud.body_mask * np.clip((cloud.body_delay - elapsed) / 0.06, 0.0, 1.0)
    x, y, w, h = cv2.boundingRect((attached > 0.01).view(np.uint8))
    if w and h:
        area = np.s_[y:y + h, x:x + w]
        alpha = np.ascontiguousarray(attached[area])
        canvas[area] = cv2.blendLinear(
            np.ascontiguousarray(cloud.body[area]), np.ascontiguousarray(canvas[area]),
            alpha, 1.0 - alpha,
        )

    flight = elapsed - cloud.delay
    active = (flight >= 0.0) & (flight < cloud.life)
    if not np.any(active):
        return canvas
    flight = flight[active, None]
    position = (
        cloud.origin[active]
        + cloud.velocity[active] * flight
        + 0.5 * cloud.wind * flight * flight
    )
    position[:, 1] += 6.0 * np.sin(flight[:, 0] * 8.0 + cloud.flutter[active]) * (width / 1280.0)
    fade = 1.0 - (flight[:, 0] / cloud.life[active]) ** 1.5
    # In volo i frammenti schiariscono un poco, come polvere controluce.
    color = cloud.color[active] + (255.0 - cloud.color[active]) * (0.35 * (1.0 - fade))[:, None]

    offsets = np.arange(cloud.size) - cloud.size // 2
    xs = (np.rint(position[:, 0])[:, None, None] + offsets[None, None, :]).astype(np.int64)
    ys = (np.rint(position[:, 1])[:, None, None] + offsets[None, :, None]).astype(np.int64)
    xs, ys = np.broadcast_arrays(xs, ys)
    alpha = np.broadcast_to(fade[:, None, None], xs.shape)
    paint = np.broadcast_to(color[:, None, None, :], xs.shape + (3,))
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys, alpha, paint = xs[inside], ys[inside], alpha[inside], paint[inside]
    under = canvas[ys, xs].astype(np.float32)
    canvas[ys, xs] = (under + (paint - under) * alpha[:, None]).astype(np.uint8)
    return canvas
