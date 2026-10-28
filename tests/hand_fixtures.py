"""Synthetic MediaPipe landmarks: a hand or a face where a test wants it, in normalized coordinates."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

# Knuckle (MCP) of index, middle, ring and little finger, across the palm.
KNUCKLES_X = (-0.15, -0.05, 0.05, 0.15)
KNUCKLES_Y = 0.45
THUMBS = {
    # CMC, MCP, IP, tip.
    "side": ((-0.12, 0.12), (-0.22, 0.22), (-0.32, 0.30), (-0.42, 0.36)),
    "up": ((-0.12, 0.15), (-0.15, 0.40), (-0.15, 0.58), (-0.15, 0.75)),
    "folded": ((-0.12, 0.12), (-0.18, 0.22), (-0.10, 0.30), (0.00, 0.30)),
}


def make_hand(cx, cy, size, extended=(True, True, True, True), thumb="side", angle=0.0):
    """21 landmarks with the palm centered on (cx, cy).

    size is the wrist to middle fingertip length with the fingers straight; angle turns the
    hand clockwise in degrees, 0 with the fingers pointing up.
    """
    local = [(0.0, 0.0), *THUMBS[thumb]]
    for x, straight in zip(KNUCKLES_X, extended):
        if straight:
            local += [(x, KNUCKLES_Y), (x, KNUCKLES_Y + 0.2), (x, KNUCKLES_Y + 0.35), (x, KNUCKLES_Y + 0.5)]
        else:
            # Curled into the palm: the tip ends near the knuckle, closer to the wrist than the PIP joint.
            local += [(x, KNUCKLES_Y), (x, KNUCKLES_Y + 0.15), (x, KNUCKLES_Y + 0.05), (x, KNUCKLES_Y - 0.02)]
    points = np.array(local) * size
    turn = math.radians(angle)
    rotation = np.array([[math.cos(turn), math.sin(turn)], [-math.sin(turn), math.cos(turn)]])
    points = points @ rotation.T
    palm = points[[0, 5, 9, 13, 17]].mean(axis=0)
    # Image y grows downwards: "up" in the hand is a smaller y.
    return [SimpleNamespace(x=float(cx + x - palm[0]), y=float(cy - (y - palm[1])), z=0.0) for x, y in points]


def make_face(x1, y1, x2, y2):
    """478 face landmarks filling the box, its corners included."""
    xs = np.linspace(x1, x2, 22)
    ys = np.linspace(y1, y2, 22)
    grid = [(x, y) for y in ys for x in xs][:478]
    return [SimpleNamespace(x=float(x), y=float(y), z=0.0) for x, y in grid]
