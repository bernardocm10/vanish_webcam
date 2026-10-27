# VanishCam

Vanish from your webcam with a hand gesture, live in Teams, OBS, or any video call. Dissolve into thin air, swap yourself for a log in a cloud of smoke, teleport like Goku, or scatter into fragments in the wind — in front of your real room or any picture you choose.

<p align="center"><img src="docs/panel.png" alt="VanishCam control panel" width="360"></p>

Everything runs on your computer: the video is never recorded or sent anywhere, and the video you send to a call shows only you and the effect, with no menus or text on top.

## Features

- **Four effects**, each triggered by a hand gesture.
- **Control panel** to switch effect and background while the camera keeps running.
- **Custom backgrounds:** put any picture behind you; when you vanish, the picture is what remains.
- **Works in calls:** virtual webcam for Teams, Zoom, Meet, and OBS, and a virtual microphone so the call hears the effect sounds (Linux).
- **Remembers your choices** and starts with the last effect and background.

## Quick start

**Windows:** double-click `AVVIA_LOCALE.bat`. The first run installs everything it needs (internet required once).

**Linux:**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python vanish_cam.py
```

Two windows open: the video and the **VanishCam** control panel. Then:

1. **Step out of the frame for 4 seconds** so the app can capture your room. Skipped if you pick a picture as the background.
2. **Pick an effect and a background** in the panel.
3. **Make the gesture** and vanish. Press `R` (or the gesture again, in Goku) to come back.

Using it in a call? See [Calls, Teams and OBS](docs/calls.md).

## Effects at a glance

| Effect | Gesture | What happens |
| --- | --- | --- |
| **Normal** | V sign (index and middle finger) | You dissolve into the background |
| **Naruto** | Palms pressed together, held for a moment | A cloud of smoke, and a log takes your place |
| **Goku** | Two fingers on your forehead for 1 s | Speed lines and you teleport; repeat to come back |
| **Partículas** | V sign | You break into fragments carried by the wind |

Details for each one in [Effects](docs/effects.md).

## Documentation

| Guide | What is in it |
| --- | --- |
| [Effects](docs/effects.md) | Each effect, its gesture, sounds, and how it is drawn |
| [Control panel and backgrounds](docs/control-panel.md) | The panel, picture backgrounds, keyboard shortcuts, saved preferences |
| [Calls, Teams and OBS](docs/calls.md) | Virtual webcam and microphone, Teams on Linux, OBS, GNOME launcher |
| [Options and troubleshooting](docs/configuration.md) | Command-line options, camera and resolution, fixing background residue and gestures |
| [Development](docs/development.md) | Tests, code structure, adding an effect, building the Windows EXE |

## Credits

VanishCam is a fork of [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam) by [Nicola Auriemma](https://github.com/Nicovic17), who developed the first two effects, **Normal** (dissolve) and **Naruto** (log and smoke), and the original app this repository is built on: the webcam pipeline, person segmentation, and gesture recognition with MediaPipe.
