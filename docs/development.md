# Development

[← Documentation](README.md)

## Running the tests

From the project folder:

```bash
.venv/bin/python -m unittest
```

GitHub Actions runs them on Windows and Linux for every push to `main` and every pull request (`.github/workflows/tests.yml`).

To run the app from source: `.venv/bin/python -m vanishcam`.

## How the code is organized

```
vanishcam/          the app, a Python package
  effects/          drawing and pose detection for each effect
assets/             images, sounds and fonts bundled with the app
models/             MediaPipe models (the hand and face ones are versioned; the rest download on first run)
tests/              unittest suite
scripts/windows/    start.bat and its variants, build.bat
scripts/linux/      teams.sh, the GNOME launcher and its installer
packaging/          PyInstaller spec for the Windows EXE
docs/               these guides
prototypes/         experiments that are not part of the app
```

| File | Responsibility |
| --- | --- |
| `vanishcam/app.py` | The video loop: calibration, the active effect (`EffectRuntime`), keys and panel actions |
| `vanishcam/cli.py` | Command-line options |
| `vanishcam/camera.py` | Opening the webcam, reopening it when it drops, and the card shown meanwhile |
| `vanishcam/virtual_camera.py` | The virtual webcam output |
| `vanishcam/models.py` | MediaPipe model paths and downloads |
| `vanishcam/segmentation.py` | The person mask, once per frame |
| `vanishcam/background.py` | Picture backgrounds and keeping the real room up to date |
| `vanishcam/detectors.py` | Gesture and pose detection for each mode, and `AsyncRecognizer` |
| `vanishcam/modes.py` | One class per effect: `NormalMode`, `NarutoMode`, `GokuMode`, `ParticlesMode`, `AshMode` |
| `vanishcam/effects/` | `normal.py`, `naruto.py`, `goku.py`, `particles.py`, `ash.py`: drawing for each effect |
| `vanishcam/control_panel.py` | Draws the panel with Pillow and turns clicks into actions |
| `vanishcam/user_data.py` | Saved preferences and the background gallery |
| `vanishcam/call_audio.py`, `vanishcam/sound.py` | Effect sounds and the call microphone (PipeWire on Linux, VB-CABLE on Windows) |
| `vanishcam/i18n.py` | Every text the user reads, in English and Portuguese, and the language detection |
| `vanishcam/paths.py` | Where the bundled assets and models are, from source and from the EXE |

A few things worth knowing:

- **Gesture detection runs in a background thread** (`AsyncRecognizer`), so the video never waits for the models.
- **Switching effect replaces only `EffectRuntime`:** the recognizer, the sound, and the `Mode`. The camera, segmentation, virtual devices, and background stay open.
- **Each `Mode` is a small state machine:** it turns detection results into phases (`ready`, `vanishing`, `vanished`, and `appearing` for Goku) and draws each frame.

## Adding an effect

1. Put its drawing and pose detection in `vanishcam/effects/<name>.py`.
2. Subclass `Mode` in `vanishcam/modes.py` and register it in `MODES`.
3. Add its detector in `create_recognizer` and `detection_function` in `vanishcam/detectors.py`.
4. Add its row to `MODE_CARDS` in `vanishcam/control_panel.py`.
5. Add its `mode.<name>.title`, `.hint` and `.instructions` texts to both languages in `vanishcam/i18n.py`.

## Language conventions

Code, comments, docs and commit messages are in English. Anything the user reads in the app goes through `t()` in `vanishcam/i18n.py`, with an English and a Portuguese text; `tests/test_i18n.py` fails if a key or a `{placeholder}` is missing in one of them.

## Building the Windows EXE

Run `scripts\windows\build.bat` to build a standalone `dist\VanishCam.exe` that runs without Python. The build is defined in `packaging/VanishCam.spec` (PyInstaller) and bundles the models, sounds, the log image, and the panel fonts.

## Fonts

The panel uses [IBM Plex](https://github.com/IBM/plex) Mono and Sans Condensed, bundled in `assets/fonts` under the [SIL Open Font License](../assets/fonts/OFL.txt), so it looks the same on every system.
