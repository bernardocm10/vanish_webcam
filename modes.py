"""One class per effect: when it triggers, how it animates, what stays on screen.

The video loop in vanish_cam.py only handles the camera, the background and the
keys. A mode receives detection results (computed off the video thread) and
draws each frame. To add an effect, subclass Mode and register it in MODES.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

try:
    from .goku_effect import (
        EFFECT_SECONDS as GOKU_EFFECT_SECONDS, HoldTracker, make_speed_lines,
        render_teleport, teleport_region,
    )
    from .naruto_effect import (
        EFFECT_SECONDS as NARUTO_EFFECT_SECONDS, landed_log_anchor, load_log,
        overlay_log, render_substitution, substitution_anchor,
    )
    from .normal_effect import VANISH_SECONDS, apply_dissolve, make_dissolve_fields
except ImportError:
    from goku_effect import (
        EFFECT_SECONDS as GOKU_EFFECT_SECONDS, HoldTracker, make_speed_lines,
        render_teleport, teleport_region,
    )
    from naruto_effect import (
        EFFECT_SECONDS as NARUTO_EFFECT_SECONDS, landed_log_anchor, load_log,
        overlay_log, render_substitution, substitution_anchor,
    )
    from normal_effect import VANISH_SECONDS, apply_dissolve, make_dissolve_fields


# frame BGR, timestamp in ms -> maschera della persona (0..1) a piena risoluzione.
Segment = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class FrameInfo:
    frame: np.ndarray
    background: np.ndarray
    now: float
    timestamp_ms: int


class Mode:
    """Phases: ready -> vanishing -> vanished (-> appearing -> ready, Goku only)."""

    name = ""
    detect_interval = 0.14
    # V e R funzionano da tastiera; Goku li disattiva: conta solo il gesto.
    manual_keys = True
    detect_while_vanished = False
    instructions = ""

    def __init__(self, segment: Segment, sound=None):
        self.segment = segment
        self.sound = sound
        self.phase = "ready"
        self.started: float | None = None

    @property
    def detecting(self) -> bool:
        return self.phase == "ready" or (self.detect_while_vanished and self.phase == "vanished")

    def detection_image(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def on_detection(self, result, now: float) -> None:
        raise NotImplementedError

    def vanish(self, now: float) -> None:
        self.phase = "vanishing"
        self.started = now
        if self.sound is not None:
            self.sound.play()

    def reappear(self) -> None:
        self.phase = "ready"
        self.started = None

    def reset(self) -> None:
        """New background: back to the visible person."""
        self.reappear()

    def background_changed(self) -> None:
        """The background was refreshed; drop anything drawn on top of the old one."""

    def render(self, info: FrameInfo) -> np.ndarray:
        raise NotImplementedError


class NormalMode(Mode):
    name = "normal"
    instructions = "Segno a V con indice e medio: dissolvenza."

    def __init__(self, segment: Segment, sound=None):
        super().__init__(segment, sound)
        self.fields: tuple[np.ndarray, np.ndarray] | None = None
        self.mask: np.ndarray | None = None
        self.mask_at = float("-inf")

    def on_detection(self, result, now: float) -> None:
        name, score = result
        # Avvia al primo riconoscimento affidabile della V.
        if self.phase == "ready" and name == "Victory" and score >= 0.62:
            self.vanish(now)

    def reappear(self) -> None:
        super().reappear()
        self.mask = None
        self.mask_at = float("-inf")

    def render(self, info: FrameInfo) -> np.ndarray:
        if self.phase == "ready":
            return info.frame
        if self.phase == "vanished":
            return info.background
        height, width = info.frame.shape[:2]
        if self.fields is None or self.fields[0].shape != (height, width):
            self.fields = make_dissolve_fields(height, width)
        # La sagoma si aggiorna a 20 Hz durante la dissolvenza.
        if self.mask is None or info.now - self.mask_at >= 0.05:
            self.mask = self.segment(info.frame, info.timestamp_ms)
            self.mask_at = info.now
        progress = min(1.0, (info.now - self.started) / VANISH_SECONDS)
        if progress >= 1.0:
            self.phase = "vanished"
            return info.background
        return apply_dissolve(info.frame, info.background, self.mask, progress, *self.fields)


class NarutoMode(Mode):
    name = "naruto"
    detect_interval = 0.08
    instructions = "Mani unite davanti alla webcam: fumo e tronco."

    def __init__(self, segment: Segment, sound=None):
        super().__init__(segment, sound)
        self.log_sprite = load_log()
        self.log_cache: dict[int, np.ndarray] = {}
        self.mask: np.ndarray | None = None
        self.anchor: tuple[int, int, int] | None = None
        self.vanished_frame: np.ndarray | None = None

    def on_detection(self, joined, now: float) -> None:
        if self.phase == "ready" and joined:
            self.vanish(now)

    def vanish(self, now: float) -> None:
        super().vanish(now)
        self.anchor = None
        self.vanished_frame = None

    def reappear(self) -> None:
        super().reappear()
        self.mask = None
        self.anchor = None
        self.vanished_frame = None

    def background_changed(self) -> None:
        self.vanished_frame = None

    def render(self, info: FrameInfo) -> np.ndarray:
        if self.phase == "ready":
            return info.frame
        if self.phase == "vanished":
            if self.vanished_frame is None:
                self.vanished_frame = info.background.copy()
                if self.anchor is not None:
                    landed = landed_log_anchor(self.anchor, info.frame.shape[0])
                    overlay_log(self.vanished_frame, self.log_sprite, landed, cache=self.log_cache)
            return self.vanished_frame
        age = max(0.04, info.now - self.started)
        # La sagoma serve solo mentre la persona e' ancora visibile.
        if self.anchor is None or age < 0.13:
            self.mask = self.segment(info.frame, info.timestamp_ms)
            if self.anchor is None:
                self.anchor = substitution_anchor(self.mask)
        display = render_substitution(
            info.frame, info.background, self.mask, self.log_sprite, self.anchor, age, self.log_cache
        )
        if age >= NARUTO_EFFECT_SECONDS:
            self.phase = "vanished"
            self.vanished_frame = display
        return display


class GokuMode(Mode):
    name = "goku"
    detect_interval = 0.066
    manual_keys = False
    detect_while_vanished = True
    instructions = "Indice e medio uniti sulla fronte per 1 secondo: sparisci. Ripeti per riapparire."

    def __init__(self, segment: Segment, sound=None):
        super().__init__(segment, sound)
        self.hold = HoldTracker()
        self.countdown: int | None = None
        self.lines: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.mask_at = float("-inf")
        self.region: np.ndarray | None = None

    def detection_image(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        # Viso e mani non richiedono il 720p: meta' risoluzione basta.
        if width > 800:
            frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def on_detection(self, posed, now: float) -> None:
        fired = self.hold.update(bool(posed), now)
        remaining = self.hold.remaining(now)
        countdown = None if remaining is None else int(np.ceil(remaining * 4))
        if countdown is not None and countdown != self.countdown:
            print(f"[Goku] Tieni la posa: {remaining:.2f}s")
        self.countdown = countdown
        if not fired:
            return
        if self.phase == "ready":
            self.vanish(now)
        elif self.phase == "vanished":
            self.phase = "appearing"
            self.started = now
            if self.sound is not None:
                self.sound.play()
        self.region = None
        self.mask = None

    def reset(self) -> None:
        super().reset()
        self.hold = HoldTracker()
        self.region = None
        self.mask = None

    def render(self, info: FrameInfo) -> np.ndarray:
        if self.phase == "ready":
            return info.frame
        if self.phase == "vanished":
            return info.background
        width = info.frame.shape[1]
        if self.lines is None or self.lines.shape[1] != width:
            self.lines = make_speed_lines(width)
        if self.mask is None or info.now - self.mask_at >= 0.05:
            self.mask = self.segment(info.frame, info.timestamp_ms)
            self.mask_at = info.now
        if self.region is None:
            self.region = teleport_region(self.mask)
        age = info.now - self.started
        progress = min(1.0, age / GOKU_EFFECT_SECONDS)
        vanishing = self.phase == "vanishing"
        display = render_teleport(
            info.frame, info.background, self.mask, self.region, self.lines,
            progress if vanishing else 1.0 - progress, age,
        )
        if progress >= 1.0:
            self.phase = "vanished" if vanishing else "ready"
            self.started = None
            self.mask = None
            self.mask_at = float("-inf")
            if vanishing:
                print("[Goku] Sparito. Abbassa la mano, poi ripeti il gesto per riapparire.")
        return display


MODES: dict[str, type[Mode]] = {mode.name: mode for mode in (NormalMode, NarutoMode, GokuMode)}
