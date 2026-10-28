"""Control window: the panel on the left and the live video filling the rest.

Every button lives in the panel; the video beside it and the virtual camera stay
clean. The panel is drawn with Pillow (anti-aliased, with the bundled IBM Plex
fonts) and shown with OpenCV, so it shares the video loop's event handling and
needs no GUI toolkit.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .i18n import t
from .meme_poses import GESTURES
from .memes import CONFLICTS, Memes
from .modes import MODES
from .shortcuts import ACTIONS, Shortcuts
from .paths import FONT_DIR
from .user_data import pick_image_file

PANEL_TITLE = "VanishCam"
WIDTH = 440
PAD = 20
# Draw at twice the resolution and scale down: edges and text stay smooth.
SCALE = 2
GALLERY_COLUMNS = 3
TILE_GAP = 10
TILE_HEIGHT = 71
CAPTION = 18  # Caption under each thumbnail.
SELECT_HEIGHT = 52  # The effect selector and each row of its menu.
CAMERA_HEIGHT = 40  # The camera selector and each row of its menu.
KEY_HEIGHT = 46
BAR = 40
HEADER = 79  # Top bar and status line.
SECTION = 38  # From the previous content to the next, section title included.
FOOTER = 34
# Initial room for the video beside the panel; the window can then be resized freely.
VIDEO_WIDTH = 960
FULLSCREEN_HINT_SECONDS = 4.0
TABS = ("live", "memes", "settings")
MEME_ROW = 44  # Each gesture on the MEMES tab.
MEME_THUMB = (52, 36)
SHORTCUT_ROW = 28  # Each action on the SETTINGS tab.

# Broadcast control surface: light faceplate, black ink, a single signal color.
FACE = (232, 229, 222)
KEY = (244, 242, 237)
KEY_EDGE = (168, 163, 152)
HOVER = (221, 217, 208)
INK = (26, 25, 23)
GREY = (108, 104, 96)
RULE = (190, 185, 175)
PALE = (172, 167, 157)
TALLY = (212, 46, 36)
GREEN = (34, 138, 72)
AMBER = (196, 122, 14)
# Behind the video (BGR): the letterbox bars around it.
STAGE = (31, 33, 34)


@dataclass(frozen=True)
class ModeCard:
    key: str

    # Read when drawn, so they follow the language chosen at startup.
    @property
    def title(self) -> str:
        return t(f"mode.{self.key}.title")

    @property
    def hint(self) -> str:
        return t(f"mode.{self.key}.hint")


MODE_CARDS = tuple(ModeCard(key) for key in MODES)
# While the effect runs the lamp turns red, like the tally of a camera on air.
ACCENTS = {card.key: TALLY for card in MODE_CARDS}
ADD_TILE = "add"


@dataclass(frozen=True)
class PanelView:
    """Everything the panel shows; a new value redraws it."""

    mode: str
    status: str
    status_color: tuple[int, int, int] = GREEN
    background: Path | None = None
    gallery: tuple[Path, ...] = ()
    manual_keys: bool = True
    can_recapture: bool = True
    fullscreen: bool = False
    message: str = ""
    # The webcams to choose from (camera.CameraInfo) and the one in use.
    cameras: tuple = ()
    camera_index: int = 0
    camera_state: str = "live"  # live | opening | restarting | reconnecting
    camera_detail: str = ""
    can_adjust: bool = False
    detached: bool = False  # The video in its own window instead of beside the panel.
    shortcuts: Shortcuts = Shortcuts()
    memes: Memes = Memes()

    @property
    def camera_name(self) -> str:
        for camera in self.cameras:
            if camera.index == self.camera_index:
                return camera.name
        return t("panel.camera_number", index=self.camera_index)


@dataclass(frozen=True)
class Region:
    box: tuple[float, float, float, float]
    action: tuple

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.box
        return x0 <= x <= x1 and y0 <= y <= y1


def action_name(action: str) -> str:
    """How a shortcut's action is called in the panel: "mode.goku" -> "Goku"."""
    kind, _, name = action.partition(".")
    if kind == "mode":
        return t(f"mode.{name}.title")
    return t({"recapture": "panel.recapture", "quit": "panel.quit"}.get(action, f"panel.{action}"))


def key_legend(view: PanelView, action: str) -> str:
    """The key printed on a LIVE button: none while the shortcuts are off."""
    if not view.shortcuts.enabled:
        return ""
    return Shortcuts.label(view.shortcuts.key_for(action))


def hit(regions: list[Region], x: float, y: float) -> Region | None:
    # Regions on top (an open menu, the remove x) are registered before what they cover.
    return next((region for region in regions if region.contains(x, y)), None)


# IBM Plex (OFL) is bundled: same look on Windows and Linux. System fonts are the fallback.
FONTS = {
    "mono": ("IBMPlexMono-Regular.ttf", "DejaVuSansMono.ttf", "consola.ttf", "Menlo.ttc"),
    "mono-bold": ("IBMPlexMono-SemiBold.ttf", "DejaVuSansMono-Bold.ttf", "consolab.ttf", "Menlo.ttc"),
    "title": ("IBMPlexSansCondensed-SemiBold.ttf", "DejaVuSansCondensed-Bold.ttf", "arialnb.ttf", "DejaVuSans-Bold.ttf"),
}


@lru_cache(maxsize=None)
def font_file(style: str) -> str | None:
    for name in FONTS[style]:
        for candidate in (str(FONT_DIR / name), name):
            try:
                ImageFont.truetype(candidate, 12)
                return candidate
            except OSError:
                continue
    return None


@lru_cache(maxsize=None)
def font(size: float, style: str = "mono") -> ImageFont.FreeTypeFont:
    name = font_file(style)
    if name is None:
        return ImageFont.load_default(size * SCALE)
    return ImageFont.truetype(name, round(size * SCALE))


def tiles(view: PanelView) -> list:
    """Real background, gallery images (plus an outside image chosen by command line), add."""
    images = list(view.gallery)
    if view.background is not None and view.background not in images:
        images.insert(0, view.background)
    return [None, *images, ADD_TILE]


def gallery_height(view: PanelView) -> int:
    rows = -(-len(tiles(view)) // GALLERY_COLUMNS)
    return rows * (TILE_HEIGHT + CAPTION) + (rows - 1) * TILE_GAP


@dataclass(frozen=True)
class Layout:
    """Top of each section's content, in panel pixels."""

    effect: float
    gallery: float
    camera: float
    controls: float
    footer: float
    height: int


def layout(view: PanelView) -> Layout:
    effect = HEADER + SECTION
    gallery = effect + SELECT_HEIGHT + SECTION
    camera = gallery + gallery_height(view) + SECTION
    # Selector, the status line under it, then reopen and settings.
    controls = camera + CAMERA_HEIGHT + 24 + KEY_HEIGHT + SECTION
    footer = controls + 2 * KEY_HEIGHT + TILE_GAP + 16
    return Layout(effect, gallery, camera, controls, footer, round(footer + FOOTER))


def panel_height(view: PanelView) -> int:
    return layout(view).height


def place(image: np.ndarray, canvas: np.ndarray, area: tuple[int, int, int, int]) -> None:
    """Draw the whole image into area (x, y, width, height) of canvas, as large as fits, centered.

    Linear scaling: a preview only, and several times cheaper than INTER_AREA at 1080p.
    """
    x, y, width, height = area
    if image is None or width < 2 or height < 2:
        return
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
    x, y = x + (width - size[0]) // 2, y + (height - size[1]) // 2
    canvas[y : y + size[1], x : x + size[0]] = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def fit_into(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """The whole image, as large as fits, centered on the stage color."""
    stage = np.empty((height, width, 3), np.uint8)
    stage[:] = STAGE
    place(image, stage, (0, 0, width, height))
    return stage


def footer(view: PanelView, spots: Layout, text, line, fit) -> None:
    """A message for a few seconds, otherwise whether the shortcuts are on."""
    y = spots.footer
    line(0, y, WIDTH, y, RULE)
    if view.message:
        text((PAD, y + 17), fit(view.message, 10.5, "mono-bold", WIDTH - 2 * PAD), 10.5, TALLY, "mono-bold", "lm")
    else:
        hint = t("panel.footer_on" if view.shortcuts.enabled else "panel.footer_off")
        text((PAD, y + 17), hint, 9.5, GREY, "mono", "lm", spacing=0.6)


class PanelRenderer:
    def __init__(self):
        self.thumbs: dict[Path, Image.Image | None] = {}
        self.real_thumb: Image.Image | None = None
        self.version = 0

    def set_real_background(self, bgr: np.ndarray | None) -> None:
        if bgr is None:
            self.real_thumb = None
        else:
            self.real_thumb = self._fit(Image.fromarray(np.ascontiguousarray(bgr[..., ::-1])))
        self.version += 1

    def _fit(self, image: Image.Image) -> Image.Image:
        size = (round(self.tile_width * SCALE), TILE_HEIGHT * SCALE)
        return ImageOps.fit(image.convert("RGB"), size, Image.Resampling.LANCZOS)

    @property
    def tile_width(self) -> float:
        return (WIDTH - 2 * PAD - (GALLERY_COLUMNS - 1) * TILE_GAP) / GALLERY_COLUMNS

    def thumb(self, path: Path) -> Image.Image | None:
        if path not in self.thumbs:
            try:
                with Image.open(path) as image:
                    self.thumbs[path] = self._fit(image)
            except (OSError, ValueError):
                self.thumbs[path] = None
        return self.thumbs[path]

    def render(
        self, view: PanelView, hover: tuple[float, float] | None = None, menu: str | None = None,
        tab: str = "live", capturing: str | None = None,
    ) -> tuple[np.ndarray, list[Region]]:
        """The panel image and its click regions.

        menu is the open dropdown ("mode", "camera" or None), tab the page shown ("live" or
        "settings"), capturing the shortcut waiting for its new key.
        """
        spots = layout(view)
        height = spots.height
        canvas = Image.new("RGB", (WIDTH * SCALE, height * SCALE), FACE)
        draw = ImageDraw.Draw(canvas)
        regions: list[Region] = []
        # With a menu open only the menu reacts: the rest of the panel is covered.
        covered = menu is not None and tab == "live"

        def box(x0, y0, x1, y1):
            return [round(x0 * SCALE), round(y0 * SCALE), round(x1 * SCALE) - 1, round(y1 * SCALE) - 1]

        def rect(r, fill=None, outline=None, width=1):
            draw.rectangle(box(*r), fill=fill, outline=outline, width=round(width * SCALE) if outline else 0)

        def line(x0, y0, x1, y1, color, width=1):
            draw.line([(x0 * SCALE, y0 * SCALE), (x1 * SCALE, y1 * SCALE)], fill=color, width=round(width * SCALE))

        def text(xy, value, size, color, style="mono", anchor="la", spacing=0.0):
            face = font(size, style)
            if not spacing:
                draw.text((xy[0] * SCALE, xy[1] * SCALE), value, fill=color, font=face, anchor=anchor)
                return
            # Manual letter spacing: Pillow does not support it. Aligned on the total width.
            advances = [face.getlength(char) / SCALE + spacing for char in value]
            total = sum(advances) - spacing
            x = xy[0] - {"l": 0, "m": total / 2, "r": total}[anchor[0]]
            for char, advance in zip(value, advances):
                draw.text((x * SCALE, xy[1] * SCALE), char, fill=color, font=face, anchor="l" + anchor[1])
                x += advance

        def fit(value, size, style, room):
            face = font(size, style)
            while value and face.getlength(value) / SCALE > room:
                value = value[:-2] + "…"
            return value

        def hovered(r, on_top=False) -> bool:
            return (on_top or not covered) and hover is not None and Region(r, ()).contains(*hover)

        def clickable(r, action) -> None:
            if not covered:
                regions.append(Region(r, action))

        def section(y, letter, label):
            text((PAD, y), f"{letter}  {label}", 10, GREY, "mono-bold", "lm", spacing=1.2)
            start = PAD + font(10, "mono-bold").getlength(f"{letter}  {label}") / SCALE + 1.2 * (len(label) + 3) + 8
            line(start, y, WIDTH - PAD, y, RULE)

        def hatch(r, color):
            x0, y0, x1, y1 = r
            step = 7
            for offset in np.arange(-(y1 - y0), x1 - x0, step):
                ax, ay = x0 + max(offset, 0), y0 + max(-offset, 0)
                length = min(x1 - ax, y1 - ay)
                line(ax, ay, ax + length, ay + length, color)

        def chevron(x, y, color, up=False):
            sign = -1 if up else 1
            line(x - 5, y - 2.5 * sign, x, y + 2.5 * sign, color, 1.5)
            line(x, y + 2.5 * sign, x + 5, y - 2.5 * sign, color, 1.5)

        def key(r, label, legend, action, enabled=True):
            x0, y0, x1, y1 = r
            face = (x0, y0, x1, y1 - 3)
            if enabled:
                # A physical key: light face and a thicker bottom edge.
                over = hovered(r)
                rect((x0, y0 + 3, x1, y1), fill=KEY_EDGE)
                rect(face, fill=HOVER if over else KEY, outline=INK if over else KEY_EDGE)
                color, soft = INK, GREY
                clickable(r, action)
            else:
                rect(face, outline=RULE)
                color = soft = PALE
            text((x0 + 12, y0 + (y1 - y0 - 3) / 2 + 3), label, 12, color, "mono-bold", "lm", spacing=0.6)
            if legend:
                text((x1 - 9, y0 + 7), legend, 9, soft, "mono", "ra")

        def mode_row(y, number, card, selected, row_height, over):
            if over:
                rect((PAD, y + 1, WIDTH - PAD, y + row_height), fill=HOVER)
            cap = (PAD + 10, y + row_height / 2 - 13, PAD + 36, y + row_height / 2 + 13)
            rect(cap, outline=INK, width=1)
            text(((cap[0] + cap[2]) / 2, (cap[1] + cap[3]) / 2), str(number), 12.5, INK, "mono-bold", "mm")
            text((PAD + 50, y + row_height / 2 - 19), card.title, 18, INK, "title")
            text((PAD + 50, y + row_height / 2 + 7), fit(card.hint, 10.5, "mono", WIDTH - 2 * PAD - 110), 10.5, GREY)
            if selected:
                rect((WIDTH - PAD - 60, y + row_height / 2 - 4, WIDTH - PAD - 52, y + row_height / 2 + 4), fill=TALLY)
                text((WIDTH - PAD - 12, y + row_height / 2), t("panel.active"), 10, INK, "mono-bold", "rm", spacing=1)

        def selector(r, menu_name):
            """The closed dropdown: a key-like box that opens its menu."""
            opened = menu == menu_name
            over = hovered(r, on_top=opened)
            x0, y0, x1, y1 = r
            rect(r, fill=HOVER if over or opened else KEY, outline=INK if over or opened else KEY_EDGE, width=1.5 if opened else 1)
            chevron(x1 - 18, (y0 + y1) / 2, INK, up=opened)
            if opened:
                regions.append(Region(r, ("menu", menu_name)))
            else:
                clickable(r, ("menu", menu_name))

        # Top bar and status line.
        rect((0, 0, WIDTH, BAR), fill=INK)
        text((PAD, BAR / 2), "VANISHCAM", 12.5, FACE, "mono-bold", "lm", spacing=2.2)
        lamp = (WIDTH - PAD - 12, BAR / 2 - 6, WIDTH - PAD, BAR / 2 + 6)
        rect(lamp, fill=view.status_color, outline=(70, 68, 64))
        # The tabs live in the bar: the panel stays as tall as before.
        x = 150
        for name in TABS:
            label = t(f"panel.tab_{name}")
            width = font(10, "mono-bold").getlength(label) / SCALE + 1 * len(label) + 24
            r = (x, 0, x + width, BAR)
            active = name == tab
            color = FACE if active or hovered(r) else PALE
            text((x + width / 2, BAR / 2), label, 10, color, "mono-bold", "mm", spacing=1)
            if active:
                rect((x + 8, BAR - 3, x + width - 8, BAR), fill=TALLY)
            clickable(r, ("tab", name))
            x += width
        rect((PAD, 56, PAD + 8, 64), fill=view.status_color)
        text((PAD + 16, 60), fit(view.status.upper(), 11.5, "mono-bold", WIDTH - 2 * PAD - 16), 11.5, INK, "mono-bold", "lm")
        line(0, 78.5, WIDTH, 78.5, INK)

        if tab == "settings":
            self._settings(view, spots, capturing, text, rect, line, fit, hovered, clickable, section, key)
            return self._finish(canvas, height, regions)
        if tab == "memes":
            self._memes(view, spots, canvas, text, rect, line, fit, hovered, clickable, section, hatch)
            return self._finish(canvas, height, regions)

        # A · Effect: the active one; the others in its menu.
        y = spots.effect
        section(y - 16, "A", t("panel.effect"))
        effect_box = (PAD, y, WIDTH - PAD, y + SELECT_HEIGHT)
        number, card = next((n, c) for n, c in enumerate(MODE_CARDS, start=1) if c.key == view.mode)
        selector(effect_box, "mode")
        mode_row(y, number, card, False, SELECT_HEIGHT, False)

        # B · Background.
        y = spots.gallery
        section(y - 16, "B", t("panel.background"))
        width = self.tile_width
        number = 0
        for index, item in enumerate(tiles(view)):
            row, column = divmod(index, GALLERY_COLUMNS)
            x0 = PAD + column * (width + TILE_GAP)
            y0 = y + row * (TILE_HEIGHT + CAPTION + TILE_GAP)
            frame = (x0, y0, x0 + width, y0 + TILE_HEIGHT)
            over = hovered((x0, y0, x0 + width, y0 + TILE_HEIGHT + CAPTION))
            selected = item == view.background
            if item == ADD_TILE:
                rect(frame, fill=HOVER if over else KEY, outline=INK if over else KEY_EDGE)
                text((x0 + width / 2, y0 + TILE_HEIGHT / 2 - 1), "+", 22, INK, "mono", "mm")
                text((x0, y0 + TILE_HEIGHT + 5), t("panel.add"), 9.5, INK if over else GREY, "mono-bold", spacing=0.8)
                clickable((x0, y0, x0 + width, y0 + TILE_HEIGHT + CAPTION), ("add",))
                continue
            picture = self.real_thumb if item is None else self.thumb(item)
            if picture is not None:
                canvas.paste(picture, (round(x0 * SCALE), round(y0 * SCALE)))
            else:
                rect(frame, fill=KEY)
                hatch(frame, RULE)
            if item is None:
                caption = t("panel.real")
            else:
                number += 1
                caption = t("panel.image", number=number)
            if selected:
                rect((x0 - 3, y0 - 3, x0 + width + 3, y0 + TILE_HEIGHT + 3), outline=INK, width=3)
                rect((x0, y0 + TILE_HEIGHT + 8, x0 + 7, y0 + TILE_HEIGHT + 15), fill=TALLY)
                text((x0 + 12, y0 + TILE_HEIGHT + 5), caption, 9.5, INK, "mono-bold", spacing=0.8)
            else:
                rect(frame, outline=INK if over else (120, 116, 108), width=2 if over else 1)
                text((x0, y0 + TILE_HEIGHT + 5), caption, 9.5, INK if over else GREY, "mono-bold", spacing=0.8)
            if over and item is not None and item in view.gallery:
                close = (x0 + width - 24, y0 + 2, x0 + width - 2, y0 + 24)
                rect(close, fill=INK)
                cx, cy = close[0] + 11, close[1] + 11
                line(cx - 4, cy - 4, cx + 4, cy + 4, FACE, 1.5)
                line(cx - 4, cy + 4, cx + 4, cy - 4, FACE, 1.5)
                clickable(close, ("remove", item))
            if not selected:
                clickable((x0, y0, x0 + width, y0 + TILE_HEIGHT + CAPTION), ("background", item))

        # C · Camera: which webcam, how it is doing, reopen and the driver's settings.
        y = spots.camera
        section(y - 16, "C", t("panel.camera"))
        camera_box = (PAD, y, WIDTH - PAD, y + CAMERA_HEIGHT)
        selector(camera_box, "camera")
        lamp_color = GREEN if view.camera_state == "live" else AMBER
        rect((PAD + 12, y + CAMERA_HEIGHT / 2 - 4, PAD + 20, y + CAMERA_HEIGHT / 2 + 4), fill=lamp_color)
        name = fit(view.camera_name, 15, "title", WIDTH - 2 * PAD - 70)
        text((PAD + 30, y + CAMERA_HEIGHT / 2), name, 15, INK, "title", "lm")
        detail = {
            "opening": t("panel.camera_opening"), "restarting": t("panel.camera_restarting"),
            "reconnecting": t("panel.camera_reconnecting"),
        }.get(view.camera_state, view.camera_detail)
        text((PAD + 30, y + CAMERA_HEIGHT + 11), detail, 10, AMBER if view.camera_state != "live" else GREY, "mono-bold", "lm", spacing=0.6)
        y += CAMERA_HEIGHT + 24
        third = (WIDTH - 2 * PAD - 2 * TILE_GAP) / 3
        camera_keys = (
            (t("panel.reopen"), "", ("reopen_camera",), True),
            (t("panel.camera_settings"), "", ("camera_settings",), view.can_adjust and view.camera_state == "live"),
            (t("panel.attach" if view.detached else "panel.detach"), key_legend(view, "detach"), ("detach",), True),
        )
        for column, (label, legend, action, enabled) in enumerate(camera_keys):
            x0 = PAD + column * (third + TILE_GAP)
            key((x0, y, x0 + third, y + KEY_HEIGHT), label, legend, action, enabled)
        half = (WIDTH - 2 * PAD - TILE_GAP) / 2

        # D · Controls.
        y = spots.controls
        section(y - 16, "D", t("panel.controls"))
        buttons = (
            (t("panel.vanish"), key_legend(view, "vanish"), ("vanish",), view.manual_keys),
            (t("panel.reappear"), key_legend(view, "reappear"), ("reappear",), view.manual_keys),
            (t("panel.recapture"), key_legend(view, "recapture"), ("recapture",), view.can_recapture),
            (t("panel.window" if view.fullscreen else "panel.fullscreen"), key_legend(view, "fullscreen"),
             ("fullscreen",), True),
        )
        for index, (label, legend, action, enabled) in enumerate(buttons):
            row, column = divmod(index, 2)
            x0 = PAD + column * (half + TILE_GAP)
            y0 = y + row * (KEY_HEIGHT + TILE_GAP)
            key((x0, y0, x0 + half, y0 + KEY_HEIGHT), label, legend, action, enabled)

        footer(view, spots, text, line, fit)

        # The open menu, drawn last so it covers the sections below its selector.
        if menu is not None:
            if menu == "mode":
                options = [(("mode", c.key), c) for c in MODE_CARDS]
                row_height, anchor = SELECT_HEIGHT, effect_box
            else:
                cameras = view.cameras or (None,)
                options = [(("camera", c.index if c else view.camera_index), c) for c in cameras]
                row_height, anchor = CAMERA_HEIGHT, camera_box
            total = len(options) * row_height
            top = anchor[3] - 1
            if top + total > height - 4:
                top = anchor[1] - total + 1  # No room below: open upward.
            menu_box = (PAD, top, WIDTH - PAD, top + total)
            # A margin of faceplate around it hides the edges of what it covers (a selected tile's frame).
            rect((PAD - 6, top, WIDTH - PAD + 6, top + total + 6), fill=FACE)
            rect((menu_box[0] + 3, menu_box[1] + 3, menu_box[2] + 3, menu_box[3] + 3), fill=RULE)  # Shadow.
            rect(menu_box, fill=KEY, outline=INK, width=1.5)
            items = []
            for row, (action, item) in enumerate(options):
                y0 = top + row * row_height
                r = (PAD, y0, WIDTH - PAD, y0 + row_height)
                over = hovered(r, on_top=True)
                if menu == "mode":
                    mode_row(y0, row + 1, item, item.key == view.mode, row_height, over)
                else:
                    if over:
                        rect((PAD + 1, y0 + 1, WIDTH - PAD - 1, y0 + row_height), fill=HOVER)
                    label = item.name if item else view.camera_name
                    text((PAD + 30, y0 + row_height / 2), fit(label, 14, "title", WIDTH - 2 * PAD - 100), 14, INK, "title", "lm")
                    if action[1] == view.camera_index:
                        rect((PAD + 12, y0 + row_height / 2 - 4, PAD + 20, y0 + row_height / 2 + 4), fill=TALLY)
                if row:
                    line(PAD + 10, y0, WIDTH - PAD - 10, y0, RULE)
                items.append(Region(r, action))
            # On top of everything else; a click anywhere else just closes the menu.
            regions[:0] = items
            regions.append(Region((0, 0, WIDTH, height), ("menu", None)))

        return self._finish(canvas, height, regions)

    @staticmethod
    def _finish(canvas: Image.Image, height: int, regions: list[Region]) -> tuple[np.ndarray, list[Region]]:
        small = canvas.resize((WIDTH, height), Image.Resampling.BOX)
        return cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR), regions

    def meme_thumb(self, path: Path) -> Image.Image | None:
        """The gesture's picture, whole, inside the row's thumbnail."""
        key = ("meme", path)
        if key not in self.thumbs:
            size = (MEME_THUMB[0] * SCALE, MEME_THUMB[1] * SCALE)
            try:
                with Image.open(path) as image:
                    upright = ImageOps.exif_transpose(image)  # As the meme will show in the call.
                    picture = ImageOps.contain(upright.convert("RGBA"), size, Image.Resampling.LANCZOS)
                tile = Image.new("RGB", size, KEY)
                tile.paste(picture, ((size[0] - picture.width) // 2, (size[1] - picture.height) // 2), picture)
                self.thumbs[key] = tile
            except (OSError, ValueError):
                self.thumbs[key] = None
        return self.thumbs[key]

    def _memes(self, view, spots, canvas, text, rect, line, fit, hovered, clickable, section, hatch) -> None:
        """MEMES tab: the master switch and, per gesture, its picture, a collision warning and its switch."""
        memes = view.memes
        y = spots.effect
        section(y - 16, "A", t("panel.memes"))
        switch = (PAD, y, WIDTH - PAD, y + CAMERA_HEIGHT)
        over = hovered(switch)
        rect(switch, fill=HOVER if over else KEY, outline=INK if over else KEY_EDGE)
        lamp = GREEN if memes.enabled else RULE
        rect((PAD + 12, y + CAMERA_HEIGHT / 2 - 4, PAD + 20, y + CAMERA_HEIGHT / 2 + 4), fill=lamp)
        text((PAD + 30, y + CAMERA_HEIGHT / 2), t("panel.memes_on" if memes.enabled else "panel.memes_off"), 15, INK,
             "title", "lm")
        clickable(switch, ("memes_enabled", not memes.enabled))
        y += CAMERA_HEIGHT + 10
        for name in GESTURES:
            gesture = memes.gesture(name)
            thumb = (PAD, y + 4, PAD + MEME_THUMB[0], y + 4 + MEME_THUMB[1])
            picture = self.meme_thumb(gesture.image) if gesture.image is not None else None
            if picture is not None:
                canvas.paste(picture, (round(thumb[0] * SCALE), round(thumb[1] * SCALE)))
                rect(thumb, outline=INK if hovered(thumb) else KEY_EDGE)
            else:
                rect(thumb, fill=HOVER if hovered(thumb) else KEY)
                hatch(thumb, RULE)
                text(((thumb[0] + thumb[2]) / 2, (thumb[1] + thumb[3]) / 2 - 1), "+", 20, INK, "mono", "mm")
            clickable(thumb, ("meme_pick", name))
            usable = memes.enabled and gesture.image is not None
            text((PAD + MEME_THUMB[0] + 12, y + 15), t(f"meme.{name}").upper(), 11, INK if usable else GREY,
                 "mono-bold", "lm", spacing=0.6)
            effect = CONFLICTS.get(name)
            if effect == view.mode:
                note, color = t("panel.meme_conflict", effect=t(f"mode.{effect}.title")), AMBER
            else:
                note, color = t("panel.change_image" if gesture.image is not None else "panel.choose_image"), GREY
            room = WIDTH - 2 * PAD - MEME_THUMB[0] - 70
            text((PAD + MEME_THUMB[0] + 12, y + 31), fit(note, 9, "mono", room), 9, color, "mono", "lm")
            # A small switch: green with the knob right when on.
            toggle = (WIDTH - PAD - 40, y + 13, WIDTH - PAD - 4, y + 31)
            on = gesture.on and usable
            rect(toggle, fill=GREEN if on else (KEY if usable else FACE), outline=INK if usable and hovered(toggle) else KEY_EDGE)
            knob = toggle[2] - 16 if on else toggle[0] + 2
            rect((knob, toggle[1] + 2, knob + 14, toggle[3] - 2), fill=FACE if on else (RULE if usable else PALE))
            if usable:
                clickable(toggle, ("meme_on", name, not gesture.on))
            y += MEME_ROW
            line(PAD, y, WIDTH - PAD, y, RULE)
        footer(view, spots, text, line, fit)

    @staticmethod
    def _settings(view, spots, capturing, text, rect, line, fit, hovered, clickable, section, key) -> None:
        """SETTINGS tab: the shortcuts switch, one row per action, and restoring the defaults."""
        shortcuts = view.shortcuts
        y = spots.effect
        section(y - 16, "A", t("panel.shortcuts"))
        switch = (PAD, y, WIDTH - PAD, y + CAMERA_HEIGHT)
        over = hovered(switch)
        rect(switch, fill=HOVER if over else KEY, outline=INK if over else KEY_EDGE)
        lamp = GREEN if shortcuts.enabled else RULE
        rect((PAD + 12, y + CAMERA_HEIGHT / 2 - 4, PAD + 20, y + CAMERA_HEIGHT / 2 + 4), fill=lamp)
        state = t("panel.shortcuts_on" if shortcuts.enabled else "panel.shortcuts_off")
        text((PAD + 30, y + CAMERA_HEIGHT / 2), state, 15, INK, "title", "lm")
        clickable(switch, ("shortcuts_enabled", not shortcuts.enabled))
        y += CAMERA_HEIGHT + 14
        text((PAD, y), fit(t("panel.shortcuts_note"), 9.5, "mono", WIDTH - 2 * PAD), 9.5, GREY, "mono", "lm")
        y += 16
        for action in ACTIONS:
            r = (PAD, y, WIDTH - PAD, y + SHORTCUT_ROW)
            waiting = action == capturing
            if shortcuts.enabled and hovered(r):
                rect((PAD, y + 1, WIDTH - PAD, y + SHORTCUT_ROW), fill=HOVER)
            color = INK if shortcuts.enabled else PALE
            text((PAD + 4, y + SHORTCUT_ROW / 2), action_name(action).upper(), 11, color, "mono-bold", "lm", spacing=0.6)
            if waiting:
                text((WIDTH - PAD - 4, y + SHORTCUT_ROW / 2), t("panel.press_key"), 10, TALLY, "mono-bold", "rm")
            else:
                label = Shortcuts.label(shortcuts.key_for(action)) or "—"
                cap = (WIDTH - PAD - 70, y + 4, WIDTH - PAD - 4, y + SHORTCUT_ROW - 4)
                rect(cap, outline=KEY_EDGE if shortcuts.enabled else RULE)
                text(((cap[0] + cap[2]) / 2, (cap[1] + cap[3]) / 2), label, 10, color, "mono-bold", "mm")
            if shortcuts.enabled:
                clickable(r, ("capture", action))
            y += SHORTCUT_ROW
            line(PAD, y, WIDTH - PAD, y, RULE)
        y += TILE_GAP
        key((PAD, y, WIDTH - PAD, y + KEY_HEIGHT), t("panel.reset_shortcuts"), "", ("reset_shortcuts",),
            shortcuts.enabled)
        footer(view, spots, text, line, fit)


class ControlPanel:
    """The app window: shows a PanelView beside the video and turns clicks into actions.

    Actions are tuples: ("mode", key), ("background", Path | None), ("remove", Path),
    ("import", Path), ("error", text), ("vanish",), ("reappear",), ("recapture",),
    ("fullscreen",), ("camera", index), ("reopen_camera",), ("camera_settings",), ("detach",),
    ("cameras",) when the camera menu opens, to list the webcams again, and from the SETTINGS
    tab ("shortcuts_enabled", bool) and ("reset_shortcuts",), and from the MEMES tab
    ("memes_enabled", bool), ("meme_on", gesture, bool) and ("meme_import", gesture, Path).
    """

    def __init__(self, title: str = PANEL_TITLE, renderer: PanelRenderer | None = None):
        self.title = title
        self.renderer = renderer or PanelRenderer()
        self.regions: list[Region] = []
        self.hover: tuple[int, int] | None = None
        self.clicks: list[tuple[int, int]] = []
        self.picked: queue.SimpleQueue[tuple] = queue.SimpleQueue()
        self.picking = False
        self.menu: str | None = None
        self.tab = "live"
        # The shortcut whose new key is awaited (SETTINGS tab); the app reads the key.
        self.capturing: str | None = None
        self.fullscreen = False
        self.fullscreen_since = 0.0
        self.detached = False
        self.drawn = None
        self.panel: np.ndarray | None = None
        self.size: tuple[int, int] | None = None
        # The window image is kept and only its changed parts redrawn: the panel when it
        # changes, the video when a new frame comes. Rebuilt when the layout changes.
        self.canvas: np.ndarray | None = None
        self.layout = None
        self.video: np.ndarray | None = None
        # NORMAL + FREERATIO: the window can be resized and maximized; each frame is drawn
        # at the window's own size, so nothing is stretched and clicks map 1:1.
        # GUI_NORMAL: without the Qt toolbar and zoom.
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(title, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.clicks.append((x, y))

    def set_real_background(self, bgr: np.ndarray | None) -> None:
        self.renderer.set_real_background(bgr)

    def set_fullscreen(self, enabled: bool) -> None:
        """Fullscreen shows only the video; F (or the button, from the window) brings the panel back."""
        self.fullscreen = enabled
        self.fullscreen_since = time.monotonic()
        mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, mode)

    def set_detached(self, enabled: bool) -> None:
        """With the video in its own window this one shrinks to just the panel; attached, it grows back."""
        self.detached = enabled
        self.size = None  # Resized to fit on the next frame.

    def _window_size(self, panel_height: int) -> tuple[int, int]:
        if self.size is None:
            # Room for the panel, and for a 16:9 video beside it unless the video has its own window.
            if self.detached:
                self.size = (WIDTH, panel_height)
            else:
                self.size = (WIDTH + VIDEO_WIDTH, max(panel_height, VIDEO_WIDTH * 9 // 16))
            cv2.resizeWindow(self.title, *self.size)
            return self.size
        try:
            _, _, width, height = cv2.getWindowImageRect(self.title)
        except cv2.error:
            width = height = 0
        if width > 0 and height > 0:
            self.size = (width, height)
        return self.size  # Minimized: keep the last size.

    def show(self, view: PanelView, video: np.ndarray | None = None) -> None:
        # The panel is redrawn only when something changes: the view, the item under the
        # mouse, a thumbnail, the open menu. The video beside it, when a new frame comes.
        over = hit(self.regions, *self.hover) if self.hover else None
        key = (view, over.box if over else None, self.renderer.version, self.menu, self.tab, self.capturing)
        panel_changed = key != self.drawn
        if panel_changed:
            self.panel, self.regions = self.renderer.render(
                view, self.hover, self.menu, tab=self.tab, capturing=self.capturing,
            )
            self.drawn = key
        panel = self.panel
        width, height = self._window_size(panel.shape[0])
        hinting = self.fullscreen and time.monotonic() - self.fullscreen_since < FULLSCREEN_HINT_SECONDS
        layout = ((width, height), self.fullscreen, self.detached, hinting,
                  None if video is None else video.shape, panel.shape)
        if layout == self.layout and not panel_changed and video is self.video:
            return  # Nothing new: no frame since the last call (a slow webcam), no hover.
        if layout != self.layout:
            self.canvas = np.empty((height, width, 3), np.uint8)
            self.canvas[:] = FACE[::-1] if self.detached and not self.fullscreen else STAGE
            self.layout = layout
            panel_changed = video_changed = True
        else:
            video_changed = video is not self.video
        canvas = self.canvas
        if self.fullscreen:
            area = (0, 0, width, height)
        else:
            area = None if self.detached or width <= WIDTH else (WIDTH, 0, width - WIDTH, height)
            if panel_changed:
                columns, rows = min(width, WIDTH), min(height, panel.shape[0])
                canvas[:, :columns] = FACE[::-1]  # Below the panel, when the window is taller.
                canvas[:rows, :columns] = panel[:rows, :columns]
        if area is not None and video_changed:
            place(video, canvas, area)
        self.video = video
        if hinting:
            # Only on this window, never in the virtual camera; drawn on a copy so it goes away.
            canvas = canvas.copy()
            cv2.putText(canvas, t("panel.fullscreen_hint"), (24, height - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.imshow(self.title, canvas)

    def poll(self) -> list[tuple]:
        actions = []
        clicks, self.clicks = self.clicks, []
        if self.fullscreen:
            clicks = []  # Only the video is on screen.
        for x, y in clicks:
            region = hit(self.regions, x, y)
            if region is None:
                continue
            if region.action[0] == "tab":
                self.tab, self.menu, self.capturing = region.action[1], None, None
                continue
            if region.action[0] == "capture":
                self.capturing = None if self.capturing == region.action[1] else region.action[1]
                continue
            if region.action[0] == "menu":
                opening = region.action[1] if region.action[1] != self.menu else None
                self.menu = opening
                if opening == "camera":
                    actions.append(("cameras",))
                continue
            if self.menu is not None:
                self.menu = None
            actions.append(region.action)
        if ("add",) in actions:
            # A double click opens a single dialog.
            actions = [action for action in actions if action != ("add",)]
            self._start_picker(lambda path: ("import", path))
        for pick in [action for action in actions if action[0] == "meme_pick"]:
            actions.remove(pick)
            self._start_picker(lambda path, name=pick[1]: ("meme_import", name, path))
        while not self.picked.empty():
            actions.append(self.picked.get())
        return actions

    def _start_picker(self, chosen) -> None:
        """One file dialog at a time; chosen turns the picked path into the action to send."""
        if self.picking:
            return
        self.picking = True
        # The picker blocks: off the video thread, so the webcam keeps running.
        threading.Thread(target=self._pick, args=(chosen,), name="picture-picker", daemon=True).start()

    def _pick(self, chosen) -> None:
        try:
            path = pick_image_file()
            if path is not None:
                self.picked.put(chosen(path))
        except (OSError, RuntimeError) as error:
            self.picked.put(("error", str(error)))
        finally:
            self.picking = False

    @property
    def closed(self) -> bool:
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass
