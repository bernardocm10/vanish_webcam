"""Meme reactions: a held gesture pops the owner's picture over the middle of the video.

Memes is what the owner chose (saved with the preferences); MemeLayer is what is on screen:
one picture at a time, popping in, staying a few seconds and popping out by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .effects.goku import HoldTracker
from .i18n import t
from .meme_poses import GESTURES

SHOW_SECONDS = 3.0
POP_IN_SECONDS = 0.28
POP_OUT_SECONDS = 0.25
OVERSHOOT_SECONDS = 0.18
OVERSHOOT = 1.1
# The picture fits inside this share of the frame's width and height.
BOX = 0.65
HOLD_SECONDS = 0.25
# Two frames in the prototype: joined hands are hard to hold still for the tracker.
PRAYER_HOLD_SECONDS = 0.1
DETECT_INTERVAL = 0.1
# Meme gestures close to an effect's own trigger: both could fire at once.
CONFLICTS: dict[str, str] = {"prayer": "naruto", "hands_behind_head": "ash"}

_UNSET = object()


@dataclass(frozen=True)
class MemeGesture:
    image: Path | None = None
    on: bool = False


@dataclass(frozen=True)
class Memes:
    enabled: bool = True
    gestures: tuple[tuple[str, MemeGesture], ...] = tuple((name, MemeGesture()) for name in GESTURES)

    def gesture(self, name: str) -> MemeGesture:
        return dict(self.gestures)[name]

    def active(self) -> frozenset[str]:
        """The gestures to look for: on, with a picture, and the memes switched on."""
        if not self.enabled:
            return frozenset()
        return frozenset(name for name, gesture in self.gestures if gesture.on and gesture.image is not None)

    def with_enabled(self, enabled: bool) -> Memes:
        return Memes(enabled, self.gestures)

    def with_gesture(self, name: str, image=_UNSET, on=_UNSET) -> Memes:
        """A new picture or switch for one gesture; without a picture it cannot be on."""
        current = self.gesture(name)
        image = current.image if image is _UNSET else image
        on = current.on if on is _UNSET else bool(on)
        changed = MemeGesture(image, on and image is not None)
        return Memes(self.enabled, tuple((key, changed if key == name else value) for key, value in self.gestures))

    def to_json(self, root: Path) -> dict:
        def stored(image: Path | None) -> str | None:
            if image is None:
                return None
            try:
                return Path(image).relative_to(root).as_posix()
            except ValueError:
                return str(image)

        return {
            "enabled": self.enabled,
            "gestures": {name: {"image": stored(gesture.image), "on": gesture.on} for name, gesture in self.gestures},
        }

    @classmethod
    def from_json(cls, data: object, root: Path) -> Memes:
        """Malformed data gives the defaults; a picture deleted meanwhile, a gesture with none."""
        if not isinstance(data, dict):
            return cls()
        enabled = data.get("enabled") is not False
        saved = data.get("gestures") if isinstance(data.get("gestures"), dict) else {}
        gestures = []
        for name in GESTURES:
            entry = saved.get(name)
            gesture = MemeGesture()
            if isinstance(entry, dict) and isinstance(entry.get("image"), str):
                image = root / entry["image"]
                if image.is_file():
                    gesture = MemeGesture(image, entry.get("on") is True)
            gestures.append((name, gesture))
        return cls(enabled, tuple(gestures))


def scale_at(age: float) -> float:
    """Size of the picture age seconds after it fired: pop in with an overshoot, stay, shrink away."""
    if age <= 0:
        return 0.0
    if age < OVERSHOOT_SECONDS:
        progress = age / OVERSHOOT_SECONDS
        return OVERSHOOT * (1 - (1 - progress) ** 2)
    if age < POP_IN_SECONDS:
        progress = (age - OVERSHOOT_SECONDS) / (POP_IN_SECONDS - OVERSHOOT_SECONDS)
        return OVERSHOOT + (1.0 - OVERSHOOT) * progress
    leaving = SHOW_SECONDS - POP_OUT_SECONDS
    if age < leaving:
        return 1.0
    return 1.0 - 0.4 * min(1.0, (age - leaving) / POP_OUT_SECONDS)


def opacity_at(age: float) -> float:
    leaving = SHOW_SECONDS - POP_OUT_SECONDS
    if age < leaving:
        return 1.0
    return max(0.0, 1.0 - (age - leaving) / POP_OUT_SECONDS)


class MemeLayer:
    """The picture on screen, if any, drawn over whatever the effect shows."""

    def __init__(self):
        self.trackers = {
            name: HoldTracker(hold_seconds=PRAYER_HOLD_SECONDS if name == "prayer" else HOLD_SECONDS)
            for name in GESTURES
        }
        self.current: str | None = None
        self.path: Path | None = None
        self.started = 0.0
        self.fitted: dict[tuple[Path, int, int], np.ndarray | None] = {}
        self.warned: set[Path] = set()

    @property
    def showing(self) -> bool:
        return self.current is not None

    def on_detection(self, gesture: str | None, now: float, memes: Memes) -> None:
        active = memes.active()
        for name, tracker in self.trackers.items():
            # Every tracker sees whether its gesture is the one made: the others release.
            fired = tracker.update(name == gesture, now)
            if fired and self.current is None and name in active:
                self.current, self.path, self.started = name, memes.gesture(name).image, now

    def clear(self) -> None:
        self.current = self.path = None

    def render(self, frame: np.ndarray, now: float) -> np.ndarray:
        """The frame with the picture on top: a new array, never the effect's own."""
        if self.current is None:
            return frame
        age = now - self.started
        if age >= SHOW_SECONDS:
            self.clear()
            return frame
        picture = self._fitted(self.path, frame.shape[1], frame.shape[0])
        if picture is None:
            self.clear()
            return frame
        scale = scale_at(age)
        size = (round(picture.shape[1] * scale), round(picture.shape[0] * scale))
        if size[0] < 1 or size[1] < 1:
            return frame
        scaled = cv2.resize(picture, size, interpolation=cv2.INTER_LINEAR)
        out = frame.copy()
        height, width = frame.shape[:2]
        x0, y0 = (width - size[0]) // 2, (height - size[1]) // 2
        # A picture larger than the frame (the overshoot) is cut at its edges.
        left, top = max(0, -x0), max(0, -y0)
        right, bottom = min(size[0], width - x0), min(size[1], height - y0)
        piece = scaled[top:bottom, left:right].astype(np.float32)
        alpha = piece[..., 3:4] / 255.0 * opacity_at(age)
        region = out[y0 + top : y0 + bottom, x0 + left : x0 + right].astype(np.float32)
        blended = piece[..., :3] * alpha + region * (1.0 - alpha)
        out[y0 + top : y0 + bottom, x0 + left : x0 + right] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
        return out

    def _fitted(self, path: Path, width: int, height: int) -> np.ndarray | None:
        """The picture as BGRA, as large as fits in the box of this frame size; None if unreadable."""
        key = (path, width, height)
        if key not in self.fitted:
            self.fitted[key] = None
            # Pillow, not cv2.imread: it reads paths with accents on Windows (C:\Users\João)
            # and turns phone photos upright from their EXIF orientation.
            try:
                with Image.open(path) as source:
                    rgba = np.asarray(ImageOps.exif_transpose(source).convert("RGBA"))
            except (OSError, ValueError, Image.DecompressionBombError):
                rgba = None
            if rgba is None or rgba.size == 0:
                if path not in self.warned:
                    self.warned.add(path)
                    print(f"[Memes] {t('memes.unreadable', name=path.name)}")
                return None
            image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            fit = min(BOX * width / image.shape[1], BOX * height / image.shape[0])
            size = (max(1, round(image.shape[1] * fit)), max(1, round(image.shape[0] * fit)))
            self.fitted[key] = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return self.fitted[key]
