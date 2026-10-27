"""Preferences that survive a restart: last mode, chosen background and the image gallery.

Everything lives in one folder per user (settings.json plus a backgrounds/ copy of
each image), so the gallery keeps working even if the original file is moved.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_GALLERY = 7
PICKER_TITLE = "Escolha uma imagem de fundo"


def config_dir() -> Path:
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / "VanishCam"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "VanishCam"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "vanishcam"


@dataclass
class Settings:
    mode: str | None = None
    background: Path | None = None


class UserData:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.settings_file = self.root / "settings.json"
        self.backgrounds_dir = self.root / "backgrounds"

    @classmethod
    def default(cls) -> "UserData":
        return cls(config_dir())

    def load(self) -> Settings:
        try:
            data = json.loads(self.settings_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Settings()
        if not isinstance(data, dict):
            return Settings()
        mode = data.get("mode") if isinstance(data.get("mode"), str) else None
        background = data.get("background")
        # Un'immagine cancellata nel frattempo torna allo sfondo reale.
        if not isinstance(background, str) or not Path(background).is_file():
            background = None
        return Settings(mode, Path(background) if background else None)

    def save(self, settings: Settings) -> None:
        data = {
            "mode": settings.mode,
            "background": str(settings.background) if settings.background else None,
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.settings_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(temporary, self.settings_file)
        except OSError as error:
            # Non salvare le preferenze non deve fermare la webcam.
            print(f"[Webcam] Não consegui salvar as preferências: {error}")

    def gallery(self) -> list[Path]:
        """Gallery images, newest first."""
        if not self.backgrounds_dir.is_dir():
            return []
        images = [
            path for path in self.backgrounds_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        return sorted(images, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)

    def import_image(self, source: Path) -> Path:
        """Copy an image into the gallery; the oldest ones go beyond MAX_GALLERY."""
        source = Path(source)
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Formato não suportado: {source.name}")
        self.backgrounds_dir.mkdir(parents=True, exist_ok=True)
        if source.parent.resolve() == self.backgrounds_dir.resolve():
            return source
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in source.stem)[:40] or "fundo"
        target = self.backgrounds_dir / f"{time.time_ns()}-{stem}{source.suffix.lower()}"
        shutil.copyfile(source, target)
        for old in self.gallery()[MAX_GALLERY:]:
            old.unlink(missing_ok=True)
        return target

    def remove_image(self, path: Path) -> None:
        path = Path(path)
        # Solo le copie della galleria: mai i file originali dell'utente.
        if path.parent.resolve() == self.backgrounds_dir.resolve():
            path.unlink(missing_ok=True)


def picker_command() -> list[str] | None:
    """The native file-open dialog of this system, as a command that prints the chosen path."""
    system = platform.system()
    if system == "Windows":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d = New-Object System.Windows.Forms.OpenFileDialog;"
            f"$d.Title = '{PICKER_TITLE}';"
            "$d.Filter = 'Imagens|*.jpg;*.jpeg;*.png;*.bmp;*.webp';"
            "if ($d.ShowDialog() -eq 'OK') { [Console]::OutputEncoding = [Text.Encoding]::UTF8; $d.FileName }"
        )
        return ["powershell", "-NoProfile", "-STA", "-Command", script]
    if system == "Darwin":
        return ["osascript", "-e", f'POSIX path of (choose file of type {{"public.image"}} with prompt "{PICKER_TITLE}")']
    patterns = " ".join(f"*{suffix} *{suffix.upper()}" for suffix in IMAGE_SUFFIXES)
    if shutil.which("zenity"):
        return ["zenity", "--file-selection", f"--title={PICKER_TITLE}", f"--file-filter=Imagens | {patterns}"]
    if shutil.which("kdialog"):
        return ["kdialog", "--title", PICKER_TITLE, "--getopenfilename", str(Path.home()), patterns]
    return None


def pick_image_file() -> Path | None:
    """Blocking: run it off the video thread. None if cancelled."""
    command = picker_command()
    if command is None:
        raise RuntimeError("Nenhuma janela de seleção de arquivo disponível (instale o zenity).")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", creationflags=flags)
    chosen = result.stdout.strip()
    return Path(chosen) if result.returncode == 0 and chosen else None
