"""Control window: switch effect and background while the camera keeps running.

The preview and the virtual camera stay clean; every button lives here. The panel
is drawn with Pillow (anti-aliased, with the bundled IBM Plex fonts) and shown with OpenCV,
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
FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
WIDTH = 440
PAD = 20
# Disegna al doppio della risoluzione e riduce: bordi e testo restano lisci.
SCALE = 2
GALLERY_COLUMNS = 3
TILE_GAP = 10
TILE_HEIGHT = 71
CAPTION = 18  # Didascalia sotto ogni miniatura.
ROW_HEIGHT = 52
KEY_HEIGHT = 46
BAR = 40
ROWS_TOP = 112
SECTION = 38  # Dal contenuto precedente al successivo, titolo di sezione compreso.

# Pannello di controllo da regia: frontale chiaro, inchiostro nero, un solo colore di segnale.
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


@dataclass(frozen=True)
class ModeCard:
    key: str
    title: str
    hint: str


MODE_CARDS = (
    ModeCard("normal", "Normal", "sinal de V — dissolução"),
    ModeCard("naruto", "Naruto", "palmas juntas — tronco e fumaça"),
    ModeCard("goku", "Goku", "dois dedos na testa, 1 s — teletransporte"),
    ModeCard("particles", "Partículas", "continência na testa — fragmentos no vento"),
)
# Quando l'effetto e' in corso la spia diventa rossa, come il tally di una telecamera in onda.
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


# IBM Plex (OFL) e' incluso: stesso aspetto su Windows e Linux. I font di sistema sono il ripiego.
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


def rows_bottom() -> int:
    return ROWS_TOP + len(MODE_CARDS) * ROW_HEIGHT


def panel_height(view: PanelView) -> int:
    keys = 2 * KEY_HEIGHT + TILE_GAP
    return rows_bottom() + SECTION + gallery_height(view) + SECTION + keys + 50


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

    def render(self, view: PanelView, hover: tuple[float, float] | None = None) -> tuple[np.ndarray, list[Region]]:
        height = panel_height(view)
        canvas = Image.new("RGB", (WIDTH * SCALE, height * SCALE), FACE)
        draw = ImageDraw.Draw(canvas)
        regions: list[Region] = []

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
            # Spaziatura manuale: Pillow non la gestisce. Si allinea sulla larghezza totale.
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

        def hovered(r) -> bool:
            return hover is not None and Region(r, ()).contains(*hover)

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

        # Barra superiore e riga di stato.
        rect((0, 0, WIDTH, BAR), fill=INK)
        text((PAD, BAR / 2), "VANISHCAM", 12.5, FACE, "mono-bold", "lm", spacing=2.2)
        lamp = (WIDTH - PAD - 12, BAR / 2 - 6, WIDTH - PAD, BAR / 2 + 6)
        rect(lamp, fill=view.status_color, outline=(70, 68, 64))
        text((lamp[0] - 10, BAR / 2), "TALLY", 9, (130, 126, 118), "mono", "rm", spacing=1)
        rect((PAD, 56, PAD + 8, 64), fill=view.status_color)
        text((PAD + 16, 60), fit(view.status.upper(), 11.5, "mono-bold", WIDTH - 2 * PAD - 16), 11.5, INK, "mono-bold", "lm")
        line(0, 78.5, WIDTH, 78.5, INK)

        section(ROWS_TOP - 16, "A", "EFEITO")
        y = ROWS_TOP
        line(PAD, y, WIDTH - PAD, y, RULE)
        for number, card in enumerate(MODE_CARDS, start=1):
            row = (PAD, y, WIDTH - PAD, y + ROW_HEIGHT)
            selected = card.key == view.mode
            if selected:
                rect((PAD, y, WIDTH - PAD, y + ROW_HEIGHT + 1), fill=INK)
            elif hovered(row):
                rect((PAD, y + 1, WIDTH - PAD, y + ROW_HEIGHT), fill=HOVER)
            ink, soft = (FACE, PALE) if selected else (INK, GREY)
            cap = (PAD + 10, y + 13, PAD + 36, y + 39)
            rect(cap, outline=ink, width=1)
            text(((cap[0] + cap[2]) / 2, (cap[1] + cap[3]) / 2), str(number), 12.5, ink, "mono-bold", "mm")
            text((PAD + 50, y + 7), card.title, 18, ink, "title")
            text((PAD + 50, y + 33), card.hint, 10.5, soft)
            if selected:
                rect((WIDTH - PAD - 60, y + 22, WIDTH - PAD - 52, y + 30), fill=TALLY)
                text((WIDTH - PAD - 12, y + ROW_HEIGHT / 2), "ATIVO", 10, FACE, "mono-bold", "rm", spacing=1)
            else:
                regions.append(Region(row, ("mode", card.key)))
            y += ROW_HEIGHT
            line(PAD, y, WIDTH - PAD, y, RULE)

        y = rows_bottom() + SECTION
        section(y - 16, "B", "FUNDO")
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
                text((x0, y0 + TILE_HEIGHT + 5), "ADICIONAR", 9.5, INK if over else GREY, "mono-bold", spacing=0.8)
                regions.append(Region((x0, y0, x0 + width, y0 + TILE_HEIGHT + CAPTION), ("add",)))
                continue
            picture = self.real_thumb if item is None else self.thumb(item)
            if picture is not None:
                canvas.paste(picture, (round(x0 * SCALE), round(y0 * SCALE)))
            else:
                rect(frame, fill=KEY)
                hatch(frame, RULE)
            if item is None:
                caption = "REAL"
            else:
                number += 1
                caption = f"IMG {number:02d}"
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
                regions.append(Region(close, ("remove", item)))
            if not selected:
                regions.append(Region((x0, y0, x0 + width, y0 + TILE_HEIGHT + CAPTION), ("background", item)))

        y += gallery_height(view) + SECTION
        section(y - 16, "C", "CONTROLES")
        buttons = (
            ("SUMIR", "V", ("vanish",), view.manual_keys),
            ("REAPARECER", "R", ("reappear",), view.manual_keys),
            ("NOVO FUNDO", "ESPAÇO", ("recapture",), view.can_recapture),
            ("JANELA" if view.fullscreen else "TELA CHEIA", "F", ("fullscreen",), True),
        )
        width = (WIDTH - 2 * PAD - TILE_GAP) / 2
        for index, (label, key, action, enabled) in enumerate(buttons):
            row, column = divmod(index, 2)
            x0 = PAD + column * (width + TILE_GAP)
            y0 = y + row * (KEY_HEIGHT + TILE_GAP)
            face = (x0, y0, x0 + width, y0 + KEY_HEIGHT - 3)
            if enabled:
                # Tasto fisico: faccia chiara e bordo inferiore piu' spesso.
                over = hovered((x0, y0, x0 + width, y0 + KEY_HEIGHT))
                rect((x0, y0 + 3, x0 + width, y0 + KEY_HEIGHT), fill=KEY_EDGE)
                rect(face, fill=HOVER if over else KEY, outline=INK if over else KEY_EDGE)
                color, legend = INK, GREY
                regions.append(Region((x0, y0, x0 + width, y0 + KEY_HEIGHT), action))
            else:
                rect(face, outline=RULE)
                color = legend = PALE
            text((x0 + 12, y0 + (KEY_HEIGHT - 3) / 2 + 3), label, 12, color, "mono-bold", "lm", spacing=0.6)
            text((x0 + width - 9, y0 + 7), key, 9, legend, "mono", "ra")
        y += 2 * KEY_HEIGHT + TILE_GAP + 16

        line(0, y, WIDTH, y, RULE)
        if view.message:
            text((PAD, y + 17), fit(view.message, 10.5, "mono-bold", WIDTH - 2 * PAD), 10.5, TALLY, "mono-bold", "lm")
        else:
            text((PAD, y + 17), "1–4 EFEITO   V SUMIR   R VOLTAR   Q SAIR", 9.5, GREY, "mono", "lm", spacing=0.6)

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
