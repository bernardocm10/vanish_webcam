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
| `vanishcam/detectors.py` | The recognizers (gesture, hands, face and hands) and `AsyncRecognizer` |
| `vanishcam/modes.py` | One class per effect, declaring its models, recognizer and sound: `NormalMode`, `NarutoMode`, `GokuMode`, `ParticlesMode`, `AshMode`; `MODES` sets the panel order |
| `vanishcam/effects/` | `normal.py`, `naruto.py`, `goku.py`, `particles.py`, `ash.py`: drawing for each effect |
| `vanishcam/hand_geometry.py` | Finger and face measures shared by the effects' poses and the memes |
| `vanishcam/meme_poses.py` | The eight meme gestures and their precedence |
| `vanishcam/memes.py` | The meme settings (`Memes`) and the picture on screen (`MemeLayer`) |
| `vanishcam/shortcuts.py` | Keyboard shortcuts: off by default, one key per action |
| `vanishcam/control_panel.py` | Draws the panel with Pillow and turns clicks into actions |
| `vanishcam/user_data.py` | Saved preferences and the background gallery |
| `vanishcam/call_audio.py`, `vanishcam/sound.py` | Effect sounds and the call microphone (PipeWire on Linux, VB-CABLE on Windows) |
| `vanishcam/i18n.py` | Every text the user reads, in English and Portuguese, and the language detection |
| `vanishcam/paths.py` | Where the bundled assets and models are, from source and from the EXE |

A few things worth knowing:

- **Gesture detection runs in a background thread** (`AsyncRecognizer`), so the video never waits for the models.
- **The webcam is read in a background thread** (`FrameReader`), so a slow or stuck webcam never freezes the window: while no new image comes, the loop keeps answering clicks and keys without redoing any video work.
- **The window image is reused** (`ControlPanel.show`): the panel is copied in only when it changes and the video only when a new frame comes.
- **Switching effect replaces only `EffectRuntime`:** the recognizer, the sound, and the `Mode`. The camera, segmentation, virtual devices, and background stay open.
- **Each `Mode` is a small state machine:** it turns detection results into phases (`ready`, `vanishing`, `vanished`, and `appearing` for Goku) and draws each frame.

## Adding an effect

1. Put its drawing and pose in `vanishcam/effects/<name>.py`.
2. Subclass `Mode` in `vanishcam/modes.py` and declare what it needs: `models`, `create_recognizer` and `detect`, and optionally `sound_file` and `sound_optional`. Register it in `MODES` at the position it should have in the panel: that order also sets its `1`–`5` key.
3. Add its `mode.<name>.title`, `.hint` and `.instructions` texts to both languages in `vanishcam/i18n.py`.

The panel, the model downloads, the sound, and the self-test read everything else from `MODES`.

## Language conventions

Code, comments, docs and commit messages are in English. Anything the user reads in the app goes through `t()` in `vanishcam/i18n.py`, with an English and a Portuguese text; `tests/test_i18n.py` fails if a key or a `{placeholder}` is missing in one of them.

## Building the Windows EXE

Run `scripts\windows\build.bat` to build a standalone `dist\VanishCam.exe` that runs without Python. The build is defined in `packaging/VanishCam.spec` (PyInstaller) and bundles the models, sounds, the log image, and the panel fonts.

### Publishing a release

Push a version tag and GitHub Actions builds the EXE, runs its self-test, and publishes it on the [Releases](https://github.com/bernardocm10/vanish_webcam/releases) page (`.github/workflows/release.yml`):

```bash
git tag v1.0.0
git push origin v1.0.0
```

To only build it, run the **Release** workflow by hand from the Actions tab and download the EXE from the run's artifacts.

## Fonts

The panel uses [IBM Plex](https://github.com/IBM/plex) Mono and Sans Condensed, bundled in `assets/fonts` under the [SIL Open Font License](../assets/fonts/OFL.txt), so it looks the same on every system.
