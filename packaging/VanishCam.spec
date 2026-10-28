"""PyInstaller one-file build for the Windows webcam app."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs


root = Path(SPECPATH).resolve().parent
icon_file = root / "assets" / "vanish_icon.ico"
resources = [
    (root / "models" / "hand_landmarker.task", "models"),
    (root / "models" / "face_landmarker.task", "models"),
    (root / "models" / "gesture_recognizer.task", "models"),
    (root / "models" / "selfie_segmenter_landscape.tflite", "models"),
    (root / "assets" / "substitution_log.png", "assets"),
    (root / "assets" / "vanish.MP3", "assets"),
    (root / "assets" / "fonts" / "IBMPlexMono-Regular.ttf", "assets/fonts"),
    (root / "assets" / "fonts" / "IBMPlexMono-SemiBold.ttf", "assets/fonts"),
    (root / "assets" / "fonts" / "IBMPlexSansCondensed-SemiBold.ttf", "assets/fonts"),
    (root / "assets" / "fonts" / "OFL.txt", "assets/fonts"),
]
missing = [str(path) for path, _ in resources if not path.is_file()]
if not icon_file.is_file():
    missing.append(str(icon_file))
if missing:
    raise SystemExit("Files required by the build are missing:\n" + "\n".join(missing))

a = Analysis(
    [str(root / "vanishcam" / "__main__.py")],
    pathex=[str(root)],
    binaries=collect_dynamic_libs("mediapipe"),
    datas=[(str(path), destination) for path, destination in resources],
    hiddenimports=["pygame.mixer", "mediapipe.tasks.c"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VanishCam",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=str(icon_file),
)
