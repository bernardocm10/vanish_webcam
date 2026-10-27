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
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

try:
    from .user_data import pick_image_file
except ImportError:
    from user_data import pick_image_file

PANEL_TITLE = "VanishCam"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "vanish_icon.png"
WIDTH = 440
PAD = 22
# Disegna al doppio della risoluzione e riduce: bordi e testo restano lisci.
SCALE = 2
GALLERY_COLUMNS = 3
TILE_GAP = 10
TILE_HEIGHT = 72
CARD_HEIGHT = 60
CARD_GAP = 10
BUTTON_HEIGHT = 42
CARDS_TOP = 146
TILES_OFFSET = 44  # Dalla fine delle card all'inizio della galleria (titolo compreso).

BG_TOP = (23, 22, 34)
BG_BOTTOM = (12, 12, 18)
BG = BG_BOTTOM
CARD = (30, 31, 42)
HOVER = (40, 42, 56)
BORDER = (52, 55, 72)
TEXT = (240, 241, 247)
MUTED = (148, 152, 172)
FAINT = (92, 96, 116)
GREEN = (112, 214, 150)
AMBER = (246, 190, 92)


@dataclass(frozen=True)
class ModeCard:
    key: str
    title: str
    hint: str
    accent: tuple[int, int, int]


MODE_CARDS = (
    ModeCard("normal", "Normal", "Sinal de V · dissolução", (110, 160, 255)),
    ModeCard("naruto", "Naruto", "Palmas juntas · tronco e fumaça", (255, 150, 90)),
    ModeCard("goku", "Goku", "Dois dedos na testa por 1 s · teletransporte", (250, 204, 80)),
    ModeCard("particles", "Partículas", "Sinal de V · fragmentos no vento", (190, 140, 255)),
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
def font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = font_file(bold)
    if name is None:
        return ImageFont.load_default(size * SCALE)
    return ImageFont.truetype(name, round(size * SCALE))


def mix(color, other, amount: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * amount) for a, b in zip(color, other))


def alpha(color, amount: float) -> tuple[int, int, int, int]:
    return (*color, round(255 * amount))


def tiles(view: PanelView) -> list:
    """Real background, gallery images (plus an outside image chosen by command line), add."""
    images = list(view.gallery)
    if view.background is not None and view.background not in images:
        images.insert(0, view.background)
    return [None, *images, ADD_TILE]


def gallery_height(view: PanelView) -> int:
    rows = -(-len(tiles(view)) // GALLERY_COLUMNS)
    return rows * TILE_HEIGHT + (rows - 1) * TILE_GAP


def cards_bottom() -> int:
    return CARDS_TOP + len(MODE_CARDS) * CARD_HEIGHT + (len(MODE_CARDS) - 1) * CARD_GAP


def panel_height(view: PanelView) -> int:
    buttons = 2 * BUTTON_HEIGHT + TILE_GAP
    return cards_bottom() + TILES_OFFSET + gallery_height(view) + TILES_OFFSET + buttons + 64


@lru_cache(maxsize=8)
def backdrop(height: int, accent: tuple[int, int, int]) -> Image.Image:
    """Vertical gradient with a soft glow of the current effect's color behind the header."""
    ramp = np.linspace(0.0, 1.0, height * SCALE, dtype=np.float32)[:, None, None]
    pixels = np.array(BG_TOP, np.float32) * (1 - ramp) + np.array(BG_BOTTOM, np.float32) * ramp
    image = Image.fromarray(np.repeat(pixels, WIDTH * SCALE, axis=1).astype(np.uint8))
    glow = Image.new("L", (WIDTH * SCALE // 4, height * SCALE // 4), 0)
    ImageDraw.Draw(glow).ellipse([-30, -60, glow.width * 0.75, 70], fill=52)
    glow = glow.filter(ImageFilter.GaussianBlur(18)).resize(image.size, Image.Resampling.BILINEAR)
    return Image.composite(Image.new("RGB", image.size, accent), image, glow)


@lru_cache(maxsize=1)
def logo() -> Image.Image | None:
    try:
        with Image.open(LOGO_PATH) as image:
            return image.convert("RGBA").resize((34 * SCALE, 34 * SCALE), Image.Resampling.LANCZOS)
    except OSError:
        return None


def draw_icon(draw: ImageDraw.ImageDraw, key: str, x: float, y: float, color) -> None:
    """Small vector icon for each effect, in a 40x40 box at (x, y)."""
    s = SCALE

    def dot(cx, cy, r):
        draw.ellipse([(x + cx - r) * s, (y + cy - r) * s, (x + cx + r) * s, (y + cy + r) * s], fill=color)

    if key == "normal":  # Una sagoma che si disperde.
        dot(15, 20, 7.5)
        for cx, cy, r in ((25, 14, 2.6), (28.5, 22, 2.2), (33, 16.5, 1.6), (31.5, 27, 1.4), (35, 23, 1.1)):
            dot(cx, cy, r)
    elif key == "naruto":  # Nuvola di fumo.
        for cx, cy, r in ((13, 23, 6), (20, 17.5, 8), (28, 22, 6.5), (20, 25, 6)):
            dot(cx, cy, r)
    elif key == "goku":  # Fulmine del teletrasporto.
        points = ((23, 6), (12, 22), (19, 22), (16, 34), (29, 16.5), (21.5, 16.5), (26, 6))
        draw.polygon([((x + px) * s, (y + py) * s) for px, py in points], fill=color)
    else:  # Frammenti nel vento.
        for cx, cy, half in ((11, 24, 4.5), (19, 15, 3.6), (24, 26, 3.2), (30, 13.5, 2.6), (33, 23, 2.2), (27, 7.5, 1.6)):
            draw.rounded_rectangle([(x + cx - half) * s, (y + cy - half) * s, (x + cx + half) * s, (y + cy + half) * s],
                                   radius=half * 0.35 * s, fill=color)


def draw_check(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, color) -> None:
    s = SCALE
    draw.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s], fill=color)
    draw.line([((cx - r * 0.42) * s, cy * s), ((cx - r * 0.08) * s, (cy + r * 0.34) * s),
               ((cx + r * 0.45) * s, (cy - r * 0.32) * s)], fill=BG, width=round(r * 0.32 * s), joint="curve")


class PanelRenderer:
    def __init__(self):
        self.thumbs: dict[Path, Image.Image | None] = {}
        self.real_thumb: Image.Image | None = None
        self.version = 0
        self.tile_mask = Image.new("L", (round(self.tile_width * SCALE), TILE_HEIGHT * SCALE), 0)
        ImageDraw.Draw(self.tile_mask).rounded_rectangle(
            [0, 0, self.tile_mask.width - 1, self.tile_mask.height - 1], radius=12 * SCALE, fill=255,
        )

    def set_real_background(self, bgr: np.ndarray | None) -> None:
        if bgr is None:
            self.real_thumb = None
        else:
            rgb = Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]))
            self.real_thumb = self._fit(rgb)
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
        accent = ACCENTS.get(view.mode, MODE_CARDS[0].accent)
        canvas = backdrop(height, accent).copy()
        # Modalita' RGBA: i colori con trasparenza si fondono con lo sfondo.
        draw = ImageDraw.Draw(canvas, "RGBA")
        regions: list[Region] = []

        def box(x0, y0, x1, y1):
            return [round(x0 * SCALE), round(y0 * SCALE), round(x1 * SCALE), round(y1 * SCALE)]

        def text(xy, value, size, color, bold=False, anchor="la"):
            draw.text((xy[0] * SCALE, xy[1] * SCALE), value, fill=color, font=font(size, bold), anchor=anchor)

        def hovered(rect) -> bool:
            return hover is not None and Region(rect, ()).contains(*hover)

        def glow(rect, color, radius, strength=0.55):
            # Alone sfocato dietro l'elemento selezionato.
            margin = 16
            x0, y0, x1, y1 = rect
            layer = Image.new("L", (round((x1 - x0 + 2 * margin) * SCALE), round((y1 - y0 + 2 * margin) * SCALE)), 0)
            ImageDraw.Draw(layer).rounded_rectangle(
                [margin * SCALE, (margin + 3) * SCALE, layer.width - margin * SCALE, layer.height - (margin - 3) * SCALE],
                radius=radius * SCALE, fill=round(255 * strength),
            )
            layer = layer.filter(ImageFilter.GaussianBlur(7 * SCALE))
            canvas.paste(Image.new("RGB", layer.size, color), (round((x0 - margin) * SCALE), round((y0 - margin) * SCALE)), layer)

        def section(y, label):
            # Maiuscoletto spaziato.
            x = PAD
            for char in label:
                text((x, y), char, 10.5, FAINT, bold=True)
                x += font(10.5, True).getlength(char) / SCALE + 1.6

        def key_chip(right, cy, label, color=MUTED, fill=alpha((255, 255, 255), 0.06)):
            width = font(10.5).getlength(label) / SCALE + 12
            draw.rounded_rectangle(box(right - width, cy - 10, right, cy + 10), radius=6 * SCALE, fill=fill,
                                   outline=alpha((255, 255, 255), 0.08), width=SCALE)
            text((right - width / 2, cy), label, 10.5, color, anchor="mm")

        # Intestazione: logo, nome e stato.
        draw.rounded_rectangle(box(PAD, 22, PAD + 44, 66), radius=13 * SCALE, fill=alpha(accent, 0.18),
                               outline=alpha(accent, 0.45), width=SCALE)
        mark = logo()
        if mark is not None:
            canvas.paste(mark, (round((PAD + 5) * SCALE), round(27 * SCALE)), mark)
        text((PAD + 58, 24), "VanishCam", 21, TEXT, bold=True)
        text((PAD + 58, 51), "Efeitos de sumir na webcam", 12, MUTED)

        status_width = font(12.5, True).getlength(view.status) / SCALE + 40
        pill = (PAD, 82, PAD + status_width, 110)
        draw.rounded_rectangle(box(*pill), radius=14 * SCALE, fill=alpha(view.status_color, 0.13),
                               outline=alpha(view.status_color, 0.38), width=SCALE)
        draw.ellipse(box(PAD + 13, 92, PAD + 21, 100), fill=view.status_color)
        draw.ellipse(box(PAD + 10, 89, PAD + 24, 103), outline=alpha(view.status_color, 0.35), width=SCALE)
        text((PAD + 31, 96), view.status, 12.5, view.status_color, bold=True, anchor="lm")

        section(CARDS_TOP - 20, "EFEITO")
        y = CARDS_TOP
        for number, card in enumerate(MODE_CARDS, start=1):
            rect = (PAD, y, WIDTH - PAD, y + CARD_HEIGHT)
            selected = card.key == view.mode
            if selected:
                glow(rect, card.accent, 16, 0.5)
                gradient = Image.linear_gradient("L").rotate(90).resize(
                    (round((rect[2] - rect[0]) * SCALE), CARD_HEIGHT * SCALE))
                tint = Image.composite(Image.new("RGB", gradient.size, mix(CARD, card.accent, 0.08)),
                                       Image.new("RGB", gradient.size, mix(CARD, card.accent, 0.30)), gradient)
                mask = Image.new("L", tint.size, 0)
                ImageDraw.Draw(mask).rounded_rectangle([0, 0, mask.width - 1, mask.height - 1], radius=16 * SCALE, fill=255)
                canvas.paste(tint, (round(rect[0] * SCALE), round(rect[1] * SCALE)), mask)
                draw.rounded_rectangle(box(*rect), radius=16 * SCALE, outline=card.accent, width=2 * SCALE)
            else:
                fill = HOVER if hovered(rect) else CARD
                draw.rounded_rectangle(box(*rect), radius=16 * SCALE, fill=alpha(fill, 0.92),
                                       outline=mix(BORDER, card.accent, 0.25) if hovered(rect) else BORDER, width=SCALE)
            icon_x, icon_y = PAD + 10, y + 10
            draw.rounded_rectangle(box(icon_x, icon_y, icon_x + 40, icon_y + 40), radius=12 * SCALE,
                                   fill=card.accent if selected else alpha(card.accent, 0.16))
            draw_icon(draw, card.key, icon_x, icon_y, BG if selected else card.accent)
            text((PAD + 64, y + 12), card.title, 15.5, TEXT, bold=True)
            text((PAD + 64, y + 35), card.hint, 11.5, mix(MUTED, card.accent, 0.25) if selected else MUTED)
            if selected:
                draw_check(draw, WIDTH - PAD - 22, y + CARD_HEIGHT / 2, 10, card.accent)
            else:
                key_chip(WIDTH - PAD - 12, y + CARD_HEIGHT / 2, str(number))
                regions.append(Region(rect, ("mode", card.key)))
            y += CARD_HEIGHT + CARD_GAP

        y = cards_bottom() + TILES_OFFSET
        section(y - 20, "FUNDO")
        width = self.tile_width
        for index, item in enumerate(tiles(view)):
            row, column = divmod(index, GALLERY_COLUMNS)
            x0 = PAD + column * (width + TILE_GAP)
            y0 = y + row * (TILE_HEIGHT + TILE_GAP)
            rect = (x0, y0, x0 + width, y0 + TILE_HEIGHT)
            over = hovered(rect)
            if item == ADD_TILE:
                draw.rounded_rectangle(box(*rect), radius=12 * SCALE,
                                       fill=alpha(accent, 0.12) if over else alpha((255, 255, 255), 0.03))
                # Bordo tratteggiato.
                dash_color = accent if over else BORDER
                for start in np.arange(x0 + 14, x0 + width - 14, 9):
                    for edge in (y0 + 0.5, y0 + TILE_HEIGHT - 0.5):
                        draw.line(box(start, edge, min(start + 5, x0 + width - 14), edge), fill=dash_color, width=SCALE)
                for start in np.arange(y0 + 14, y0 + TILE_HEIGHT - 14, 9):
                    for edge in (x0 + 0.5, x0 + width - 0.5):
                        draw.line(box(edge, start, edge, min(start + 5, y0 + TILE_HEIGHT - 14)), fill=dash_color, width=SCALE)
                cx = x0 + width / 2
                draw.ellipse(box(cx - 13, y0 + 13, cx + 13, y0 + 39), fill=alpha(accent, 0.9 if over else 0.22))
                text((cx, y0 + 25.5), "+", 20, BG if over else accent, bold=True, anchor="mm")
                text((cx, y0 + 54), "Adicionar", 11, TEXT if over else MUTED, bold=over, anchor="mm")
                regions.append(Region(rect, ("add",)))
                continue
            selected = item == view.background
            if selected:
                glow(rect, accent, 12, 0.6)
            picture = self.real_thumb if item is None else self.thumb(item)
            if picture is not None:
                canvas.paste(picture, (round(x0 * SCALE), round(y0 * SCALE)), self.tile_mask)
            else:
                draw.rounded_rectangle(box(*rect), radius=12 * SCALE, fill=CARD)
                if item is None:
                    draw_icon(draw, "normal", x0 + width / 2 - 20, y0 + 8, FAINT)
            if item is None:
                label_width = font(11, True).getlength("Fundo real") / SCALE + 14
                draw.rounded_rectangle(box(x0 + 6, y0 + TILE_HEIGHT - 26, x0 + 6 + label_width, y0 + TILE_HEIGHT - 6),
                                       radius=7 * SCALE, fill=(10, 10, 14, 190))
                text((x0 + 6 + label_width / 2, y0 + TILE_HEIGHT - 16), "Fundo real", 11, TEXT, bold=True, anchor="mm")
            if selected:
                draw.rounded_rectangle(box(*rect), radius=12 * SCALE, outline=accent, width=round(2.5 * SCALE))
                draw_check(draw, x0 + width - 15, y0 + TILE_HEIGHT - 15, 9, accent)
            elif over:
                draw.rounded_rectangle(box(*rect), radius=12 * SCALE, fill=alpha((255, 255, 255), 0.08),
                                       outline=TEXT, width=SCALE)
            else:
                draw.rounded_rectangle(box(*rect), radius=12 * SCALE, outline=alpha((255, 255, 255), 0.10), width=SCALE)
            if over and item is not None and item in view.gallery:
                close = (x0 + width - 25, y0 + 5, x0 + width - 5, y0 + 25)
                draw.ellipse(box(*close), fill=(10, 10, 14, 220), outline=alpha(TEXT, 0.8), width=SCALE)
                cx, cy = close[0] + 10, close[1] + 10
                for dx in (-1, 1):
                    draw.line(box(cx - 3.5, cy - 3.5 * dx, cx + 3.5, cy + 3.5 * dx), fill=TEXT, width=round(1.6 * SCALE))
                regions.append(Region(close, ("remove", item)))
            if not selected:
                regions.append(Region(rect, ("background", item)))

        y += gallery_height(view) + TILES_OFFSET
        section(y - 20, "CONTROLES")
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
            over = enabled and hovered(rect)
            primary = index == 0 and enabled
            if primary:
                # Il pulsante principale prende il colore dell'effetto.
                if over:
                    glow(rect, accent, 12, 0.45)
                draw.rounded_rectangle(box(*rect), radius=12 * SCALE, fill=mix(accent, TEXT, 0.15) if over else accent)
                label_color, chip_color, chip_fill = BG, mix(BG, accent, 0.35), alpha(BG, 0.14)
            else:
                fill = alpha((255, 255, 255), 0.09 if over else 0.04)
                draw.rounded_rectangle(box(*rect), radius=12 * SCALE, fill=fill,
                                       outline=alpha(TEXT, 0.35) if over else alpha((255, 255, 255), 0.10), width=SCALE)
                label_color = TEXT if enabled else FAINT
                chip_color, chip_fill = (MUTED if enabled else FAINT), alpha((255, 255, 255), 0.06)
            text((x0 + 15, y0 + BUTTON_HEIGHT / 2), label, 13, label_color, bold=True, anchor="lm")
            key_chip(x0 + width - 11, y0 + BUTTON_HEIGHT / 2, key, chip_color, chip_fill)
            if enabled:
                regions.append(Region(rect, action))
        y += 2 * BUTTON_HEIGHT + TILE_GAP + 20

        draw.line(box(PAD, y, WIDTH - PAD, y), fill=alpha((255, 255, 255), 0.07), width=SCALE)
        if view.message:
            text((PAD, y + 14), view.message, 12, AMBER)
        else:
            text((PAD, y + 14), "Teclas 1–4 trocam o efeito · Q fecha o app", 11.5, FAINT)

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
