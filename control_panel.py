"""Control window: switch effect and background while the camera keeps running.

The preview and the virtual camera stay clean; every button lives here. The panel
is drawn with Pillow (anti-aliased shapes, accented text) and shown with OpenCV,
so it shares the video loop's event handling and needs no GUI toolkit.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from .user_data import pick_image_file
except ImportError:
    from user_data import pick_image_file

PANEL_TITLE = "VanishCam"
WIDTH = 420
PAD = 20
# Disegna al doppio della risoluzione e riduce: bordi e testo restano lisci.
SCALE = 2
GALLERY_COLUMNS = 3
TILE_GAP = 9
TILE_HEIGHT = 68
CARD_HEIGHT = 54
CARD_GAP = 8
BUTTON_HEIGHT = 38

BG = (24, 25, 30)
CARD = (37, 39, 46)
HOVER = (47, 50, 59)
BORDER = (60, 63, 74)
TEXT = (236, 238, 243)
MUTED = (150, 156, 170)
GREEN = (125, 207, 140)
AMBER = (240, 190, 100)


@dataclass(frozen=True)
class ModeCard:
    key: str
    title: str
    hint: str
    accent: tuple[int, int, int]


MODE_CARDS = (
    ModeCard("normal", "Normal", "Sinal de V · dissolução", (122, 162, 247)),
    ModeCard("naruto", "Naruto", "Palmas juntas · tronco e fumaça", (255, 158, 100)),
    ModeCard("goku", "Goku", "Dois dedos na testa por 1 s · teletransporte", (240, 200, 90)),
    ModeCard("particles", "Partículas", "Sinal de V · fragmentos no vento", (187, 154, 247)),
)
ACCENTS = {card.key: card.accent for card in MODE_CARDS}
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


@dataclass(frozen=True)
class Region:
    box: tuple[float, float, float, float]
    action: tuple

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.box
        return x0 <= x <= x1 and y0 <= y <= y1


def hit(regions: list[Region], x: float, y: float) -> Region | None:
    # Le regioni piccole (la x di rimozione) sono registrate prima della miniatura.
    return next((region for region in regions if region.contains(x, y)), None)


REGULAR_FONTS = (
    "Inter-Regular.ttf", "Ubuntu-R.ttf", "Cantarell-Regular.otf", "segoeui.ttf",
    "DejaVuSans.ttf", "Arial.ttf",
)
BOLD_FONTS = (
    "Inter-Bold.ttf", "Ubuntu-B.ttf", "Cantarell-Bold.otf", "segoeuib.ttf",
    "DejaVuSans-Bold.ttf", "Arial Bold.ttf",
)


@lru_cache(maxsize=None)
def font_file(bold: bool) -> str | None:
    for name in BOLD_FONTS if bold else REGULAR_FONTS:
        try:
            ImageFont.truetype(name, 12)
            return name
        except OSError:
            continue
    return None


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = font_file(bold)
    if name is None:
        return ImageFont.load_default(size * SCALE)
    return ImageFont.truetype(name, size * SCALE)


def mix(color, other, amount: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * amount) for a, b in zip(color, other))


def tiles(view: PanelView) -> list:
    """Real background, gallery images (plus an outside image chosen by command line), add."""
    images = list(view.gallery)
    if view.background is not None and view.background not in images:
        images.insert(0, view.background)
    return [None, *images, ADD_TILE]


def panel_height(view: PanelView) -> int:
    rows = -(-len(tiles(view)) // GALLERY_COLUMNS)
    cards = len(MODE_CARDS) * CARD_HEIGHT + (len(MODE_CARDS) - 1) * CARD_GAP
    gallery = rows * TILE_HEIGHT + (rows - 1) * TILE_GAP
    buttons = 2 * BUTTON_HEIGHT + TILE_GAP
    return round(86 + 20 + cards + 26 + 20 + gallery + 26 + 20 + buttons + 22 + 16 + PAD)


class PanelRenderer:
    def __init__(self):
        self.thumbs: dict[Path, Image.Image | None] = {}
        self.real_thumb: Image.Image | None = None
        self.version = 0
        self.tile_mask = Image.new("L", (round(self.tile_width * SCALE), TILE_HEIGHT * SCALE), 0)
        ImageDraw.Draw(self.tile_mask).rounded_rectangle(
            [0, 0, self.tile_mask.width - 1, self.tile_mask.height - 1], radius=10 * SCALE, fill=255,
        )

    def set_real_background(self, bgr: np.ndarray | None) -> None:
        if bgr is None:
            self.real_thumb = None
        else:
            rgb = Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]))
            # Scurita: la scritta "Fundo real" resta leggibile su qualsiasi stanza.
            thumb = self._fit(rgb)
            self.real_thumb = Image.blend(thumb, Image.new("RGB", thumb.size, BG), 0.45)
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

    def render(self, view: PanelView, hover: tuple[float, float] | None = None) -> tuple[np.ndarray, list[Region]]:
        height = panel_height(view)
        canvas = Image.new("RGB", (WIDTH * SCALE, height * SCALE), BG)
        draw = ImageDraw.Draw(canvas)
        regions: list[Region] = []
        accent = ACCENTS.get(view.mode, MODE_CARDS[0].accent)

        def box(x0, y0, x1, y1):
            return [round(x0 * SCALE), round(y0 * SCALE), round(x1 * SCALE), round(y1 * SCALE)]

        def text(xy, value, size, color, bold=False, anchor="la"):
            draw.text((xy[0] * SCALE, xy[1] * SCALE), value, fill=color, font=font(size, bold), anchor=anchor)

        def hovered(rect) -> bool:
            return hover is not None and Region(rect, ()).contains(*hover)

        def section(y, label):
            text((PAD, y), label, 11, MUTED, bold=True)

        text((PAD, 18), "VanishCam", 22, TEXT, bold=True)
        draw.ellipse(box(PAD, 58, PAD + 9, 67), fill=view.status_color)
        text((PAD + 17, 62.5), view.status, 13, view.status_color, anchor="lm")

        y = 86
        section(y, "EFEITO")
        y += 20
        for number, card in enumerate(MODE_CARDS, start=1):
            rect = (PAD, y, WIDTH - PAD, y + CARD_HEIGHT)
            selected = card.key == view.mode
            fill = mix(CARD, card.accent, 0.16) if selected else HOVER if hovered(rect) else CARD
            outline = card.accent if selected else BORDER
            draw.rounded_rectangle(box(*rect), radius=12 * SCALE, fill=fill, outline=outline,
                                   width=(2 if selected else 1) * SCALE)
            cx, cy = PAD + 27, y + CARD_HEIGHT / 2
            draw.ellipse(box(cx - 14, cy - 14, cx + 14, cy + 14),
                         fill=card.accent if selected else mix(CARD, card.accent, 0.25))
            text((cx, cy), str(number), 14, BG if selected else TEXT, bold=True, anchor="mm")
            text((PAD + 54, y + 10), card.title, 15, TEXT, bold=True)
            text((PAD + 54, y + 31), card.hint, 12, MUTED)
            if not selected:
                regions.append(Region(rect, ("mode", card.key)))
            y += CARD_HEIGHT + CARD_GAP
        y += 26 - CARD_GAP

        section(y, "FUNDO")
        y += 20
        width = self.tile_width
        for index, item in enumerate(tiles(view)):
            row, column = divmod(index, GALLERY_COLUMNS)
            x0 = PAD + column * (width + TILE_GAP)
            y0 = y + row * (TILE_HEIGHT + TILE_GAP)
            rect = (x0, y0, x0 + width, y0 + TILE_HEIGHT)
            over = hovered(rect)
            if item == ADD_TILE:
                draw.rounded_rectangle(box(*rect), radius=10 * SCALE, fill=HOVER if over else CARD,
                                       outline=BORDER, width=SCALE)
                text((x0 + width / 2, y0 + 27), "+", 24, TEXT, anchor="mm")
                text((x0 + width / 2, y0 + 50), "Adicionar", 11, MUTED, anchor="mm")
                regions.append(Region(rect, ("add",)))
                continue
            picture = self.real_thumb if item is None else self.thumb(item)
            if picture is not None:
                canvas.paste(picture, (round(x0 * SCALE), round(y0 * SCALE)), self.tile_mask)
            else:
                draw.rounded_rectangle(box(*rect), radius=10 * SCALE, fill=CARD)
            if item is None:
                text((x0 + width / 2, y0 + TILE_HEIGHT / 2), "Fundo real", 12, TEXT, bold=True, anchor="mm")
            selected = item == view.background
            if selected:
                draw.rounded_rectangle(box(*rect), radius=10 * SCALE, outline=accent, width=3 * SCALE)
            elif over:
                draw.rounded_rectangle(box(*rect), radius=10 * SCALE, outline=TEXT, width=SCALE)
            else:
                draw.rounded_rectangle(box(*rect), radius=10 * SCALE, outline=BORDER, width=SCALE)
            if over and item is not None and item in view.gallery:
                close = (x0 + width - 24, y0 + 4, x0 + width - 4, y0 + 24)
                draw.ellipse(box(*close), fill=(0, 0, 0), outline=TEXT, width=SCALE)
                text((close[0] + 10, close[1] + 9.5), "×", 14, TEXT, anchor="mm")
                regions.append(Region(close, ("remove", item)))
            if not selected:
                regions.append(Region(rect, ("background", item)))
        rows = -(-len(tiles(view)) // GALLERY_COLUMNS)
        y += rows * TILE_HEIGHT + (rows - 1) * TILE_GAP + 26

        section(y, "CONTROLES")
        y += 20
        buttons = (
            ("Sumir", "V", ("vanish",), view.manual_keys),
            ("Reaparecer", "R", ("reappear",), view.manual_keys),
            ("Novo fundo", "Espaço", ("recapture",), view.can_recapture),
            ("Sair da tela cheia" if view.fullscreen else "Tela cheia", "F", ("fullscreen",), True),
        )
        width = (WIDTH - 2 * PAD - TILE_GAP) / 2
        for index, (label, key, action, enabled) in enumerate(buttons):
            row, column = divmod(index, 2)
            x0 = PAD + column * (width + TILE_GAP)
            y0 = y + row * (BUTTON_HEIGHT + TILE_GAP)
            rect = (x0, y0, x0 + width, y0 + BUTTON_HEIGHT)
            fill = HOVER if enabled and hovered(rect) else CARD
            draw.rounded_rectangle(box(*rect), radius=10 * SCALE, fill=fill, outline=BORDER, width=SCALE)
            text((x0 + 14, y0 + BUTTON_HEIGHT / 2), label, 13, TEXT if enabled else mix(MUTED, BG, 0.4),
                 bold=True, anchor="lm")
            chip = font(11).getlength(key) / SCALE + 12
            chip_rect = (x0 + width - 10 - chip, y0 + 10, x0 + width - 10, y0 + BUTTON_HEIGHT - 10)
            draw.rounded_rectangle(box(*chip_rect), radius=5 * SCALE, fill=BG)
            text(((chip_rect[0] + chip_rect[2]) / 2, y0 + BUTTON_HEIGHT / 2), key, 11, MUTED, anchor="mm")
            if enabled:
                regions.append(Region(rect, action))
        y += 2 * BUTTON_HEIGHT + TILE_GAP + 22

        if view.message:
            text((PAD, y), view.message, 12, AMBER)
        else:
            text((PAD, y), "Teclas 1–4 trocam o efeito · Q fecha o app", 12, MUTED)

        small = canvas.resize((WIDTH, height), Image.Resampling.BOX)
        return cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR), regions


class ControlPanel:
    """The panel window: shows a PanelView and turns clicks into actions.

    Actions are tuples: ("mode", key), ("background", Path | None), ("remove", Path),
    ("import", Path), ("error", text), ("vanish",), ("reappear",), ("recapture",),
    ("fullscreen",).
    """

    def __init__(self, title: str = PANEL_TITLE, renderer: PanelRenderer | None = None):
        self.title = title
        self.renderer = renderer or PanelRenderer()
        self.regions: list[Region] = []
        self.hover: tuple[int, int] | None = None
        self.clicks: list[tuple[int, int]] = []
        self.picked: queue.SimpleQueue[tuple] = queue.SimpleQueue()
        self.picking = False
        self.shown = None
        # GUI_NORMAL: senza la barra degli strumenti e lo zoom di Qt.
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(title, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.clicks.append((x, y))

    def set_real_background(self, bgr: np.ndarray | None) -> None:
        self.renderer.set_real_background(bgr)

    def show(self, view: PanelView) -> None:
        # Ridisegna solo se cambia qualcosa: vista, elemento sotto il mouse, miniatura.
        over = hit(self.regions, *self.hover) if self.hover else None
        key = (view, over.box if over else None, self.renderer.version)
        if key == self.shown:
            return
        image, self.regions = self.renderer.render(view, self.hover)
        cv2.imshow(self.title, image)
        self.shown = key

    def poll(self) -> list[tuple]:
        actions = []
        clicks, self.clicks = self.clicks, []
        for x, y in clicks:
            region = hit(self.regions, x, y)
            if region is not None:
                actions.append(region.action)
        if ("add",) in actions:
            # Un doppio clic apre una sola finestra.
            actions = [action for action in actions if action != ("add",)]
            self._start_picker()
        while not self.picked.empty():
            actions.append(self.picked.get())
        return actions

    def _start_picker(self) -> None:
        if self.picking:
            return
        self.picking = True
        # La finestra di scelta blocca: fuori dal thread video, la webcam non si ferma.
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
