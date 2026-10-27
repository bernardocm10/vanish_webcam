# Development

[← Documentation](README.md)

## Running the tests

From the project folder:

```bash
.venv/bin/python -m unittest
```

## How the code is organized

| File | Responsibility |
| --- | --- |
| `vanish_cam.py` | Camera, background capture and updates, keyboard, virtual webcam output, and the main loop |
| `modes.py` | One class per effect: `NormalMode`, `NarutoMode`, `GokuMode`, `ParticlesMode`, `AshMode` |
| `normal_effect.py`, `naruto_effect.py`, `goku_effect.py`, `particles_effect.py`, `ash_effect.py` | Drawing for each effect |
| `control_panel.py` | Draws the panel with Pillow and turns clicks into actions |
| `user_data.py` | Saved preferences and the background gallery |
| `call_audio.py`, `vanish_sound.py` | Effect sounds and the VanishCam Mic |
| `i18n.py` | Every text the user reads, in English and Portuguese, and the language detection |

A few things worth knowing:

- **Gesture detection runs in a background thread** (`AsyncRecognizer`), so the video never waits for the models.
- **Switching effect replaces only `EffectRuntime`:** the recognizer, the sound, and the `Mode`. The camera, segmentation, virtual devices, and background stay open.
- **Each `Mode` is a small state machine:** it turns detection results into phases (`ready`, `vanishing`, `vanished`, and `appearing` for Goku) and draws each frame.

## Adding an effect

1. Subclass `Mode` in `modes.py` and register it in `MODES`.
2. Add its detector in `create_recognizer` and `detection_function` in `vanish_cam.py`.
3. Add its row to `MODE_CARDS` in `control_panel.py`.
4. Add its `mode.<name>.title`, `.hint` and `.instructions` texts to both languages in `i18n.py`.

## Language conventions

Code, comments, docs and commit messages are in English. Anything the user reads in the app goes through `t()` in `i18n.py`, with an English and a Portuguese text; `test_i18n.py` fails if a key or a `{placeholder}` is missing in one of them.

## Building the Windows EXE

Run `build-windows.bat` to build a standalone `VanishCam.exe` that runs without Python. The build is defined in `VanishCam.spec` (PyInstaller) and bundles the models, sounds, the log image, and the panel fonts.

## Fonts

The panel uses [IBM Plex](https://github.com/IBM/plex) Mono and Sans Condensed, bundled in `assets/fonts` under the [SIL Open Font License](../assets/fonts/OFL.txt), so it looks the same on every system.
