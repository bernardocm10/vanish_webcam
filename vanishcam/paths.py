"""Where the bundled files live: the project folder, or the unpacked PyInstaller build."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
FONT_DIR = ASSETS_DIR / "fonts"
MODEL_DIR = ROOT / "models"
