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


MODE_CARDS = tuple(ModeCard(key) for key in ("normal", "naruto", "goku", "particles", "ash"))
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
    camera_state: str = "live"  # live | opening | reconnecting
    camera_detail: str = ""
    can_adjust: bool = False

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


def fit_into(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """The whole image, as large as fits, centered on the stage color."""
    stage = np.empty((height, width, 3), np.uint8)
    stage[:] = STAGE
    if image is None or width < 2 or height < 2:
        return stage
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    x, y = (width - size[0]) // 2, (height - size[1]) // 2
    stage[y : y + size[1], x : x + size[0]] = cv2.resize(image, size, interpolation=interpolation)
    return stage


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
    ) -> tuple[np.ndarray, list[Region]]:
        """The panel image and its click regions; menu is the open dropdown ("mode", "camera" or None)."""
        spots = layout(view)
        height = spots.height
        canvas = Image.new("RGB", (WIDTH * SCALE, height * SCALE), FACE)
        draw = ImageDraw.Draw(canvas)
        regions: list[Region] = []
        # With a menu open only the menu reacts: the rest of the panel is covered.
        covered = menu is not None

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
        text((lamp[0] - 10, BAR / 2), "TALLY", 9, (130, 126, 118), "mono", "rm", spacing=1)
        rect((PAD, 56, PAD + 8, 64), fill=view.status_color)
        text((PAD + 16, 60), fit(view.status.upper(), 11.5, "mono-bold", WIDTH - 2 * PAD - 16), 11.5, INK, "mono-bold", "lm")
        line(0, 78.5, WIDTH, 78.5, INK)

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
        lamp_color = {"live": GREEN, "opening": AMBER, "reconnecting": AMBER}.get(view.camera_state, PALE)
        rect((PAD + 12, y + CAMERA_HEIGHT / 2 - 4, PAD + 20, y + CAMERA_HEIGHT / 2 + 4), fill=lamp_color)
        name = fit(view.camera_name, 15, "title", WIDTH - 2 * PAD - 70)
        text((PAD + 30, y + CAMERA_HEIGHT / 2), name, 15, INK, "title", "lm")
        detail = {
            "opening": t("panel.camera_opening"), "reconnecting": t("panel.camera_reconnecting"),
        }.get(view.camera_state, view.camera_detail)
        text((PAD + 30, y + CAMERA_HEIGHT + 11), detail, 10, AMBER if view.camera_state != "live" else GREY, "mono-bold", "lm", spacing=0.6)
        y += CAMERA_HEIGHT + 24
        half = (WIDTH - 2 * PAD - TILE_GAP) / 2
        key((PAD, y, PAD + half, y + KEY_HEIGHT), t("panel.reopen"), "", ("reopen_camera",))
        key((PAD + half + TILE_GAP, y, WIDTH - PAD, y + KEY_HEIGHT), t("panel.camera_settings"), "",
            ("camera_settings",), view.can_adjust and view.camera_state == "live")

        # D · Controls.
        y = spots.controls
        section(y - 16, "D", t("panel.controls"))
        buttons = (
            (t("panel.vanish"), "V", ("vanish",), view.manual_keys),
            (t("panel.reappear"), "R", ("reappear",), view.manual_keys),
            (t("panel.recapture"), t("panel.space"), ("recapture",), view.can_recapture),
            (t("panel.window" if view.fullscreen else "panel.fullscreen"), "F", ("fullscreen",), True),
        )
        for index, (label, legend, action, enabled) in enumerate(buttons):
            row, column = divmod(index, 2)
            x0 = PAD + column * (half + TILE_GAP)
            y0 = y + row * (KEY_HEIGHT + TILE_GAP)
            key((x0, y0, x0 + half, y0 + KEY_HEIGHT), label, legend, action, enabled)

        y = spots.footer
        line(0, y, WIDTH, y, RULE)
        if view.message:
            text((PAD, y + 17), fit(view.message, 10.5, "mono-bold", WIDTH - 2 * PAD), 10.5, TALLY, "mono-bold", "lm")
        else:
            text((PAD, y + 17), t("panel.footer", count=len(MODE_CARDS)), 9.5, GREY, "mono", "lm", spacing=0.6)

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

        small = canvas.resize((WIDTH, height), Image.Resampling.BOX)
        return cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR), regions


class ControlPanel:
    """The app window: shows a PanelView beside the video and turns clicks into actions.

    Actions are tuples: ("mode", key), ("background", Path | None), ("remove", Path),
    ("import", Path), ("error", text), ("vanish",), ("reappear",), ("recapture",),
    ("fullscreen",), ("camera", index), ("reopen_camera",), ("camera_settings",),
    and ("cameras",) when the camera menu opens, to list the webcams again.
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
        self.fullscreen = False
        self.fullscreen_since = 0.0
        self.drawn = None
        self.panel: np.ndarray | None = None
        self.size: tuple[int, int] | None = None
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

    def _window_size(self, panel_height: int) -> tuple[int, int]:
        if self.size is None:
            # First frame: room for the panel and a 16:9 video beside it.
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
        # mouse, a thumbnail, the open menu. The video beside it changes every frame.
        over = hit(self.regions, *self.hover) if self.hover else None
        key = (view, over.box if over else None, self.renderer.version, self.menu)
        if key != self.drawn:
            self.panel, self.regions = self.renderer.render(view, self.hover, self.menu)
            self.drawn = key
        panel = self.panel
        width, height = self._window_size(panel.shape[0])
        if self.fullscreen:
            canvas = fit_into(video, width, height)
            if time.monotonic() - self.fullscreen_since < FULLSCREEN_HINT_SECONDS:
                # Only on this window: the virtual camera never shows it.
                cv2.putText(canvas, t("panel.fullscreen_hint"), (24, height - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        else:
            canvas = np.empty((height, width, 3), np.uint8)
            columns, rows = min(width, WIDTH), min(height, panel.shape[0])
            canvas[:, :columns] = FACE[::-1]  # Below the panel, when the window is taller.
            canvas[:rows, :columns] = panel[:rows, :columns]
            if width > WIDTH:
                canvas[:, WIDTH:] = fit_into(video, width - WIDTH, height)
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
            self._start_picker()
        while not self.picked.empty():
            actions.append(self.picked.get())
        return actions

    def _start_picker(self) -> None:
        if self.picking:
            return
        self.picking = True
        # The picker blocks: off the video thread, so the webcam keeps running.
        threading.Thread(target=self._pick, name="background-picker", daemon=True).start()

    def _pick(self) -> None:
        try:
            path = pick_image_file()
            if path is not None:
                self.picked.put(("import", path))
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
