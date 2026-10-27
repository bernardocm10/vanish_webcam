# VanishCam — webcam disappearance effects

Vanish from your webcam with a hand gesture, live in Teams, OBS, or any video call: dissolve into thin air, swap yourself for a log in a cloud of smoke, teleport like Goku, or scatter into fragments in the wind. You can put any picture behind you and switch effect or background from a control panel without restarting.

<p align="center"><img src="docs/panel.png" alt="VanishCam control panel" width="360"></p>

Video is processed on your computer. It is neither recorded nor sent to a server. The preview window and virtual camera show only the video and effects, without overlaid menus, instructions, or text.

## Features

- **Four gesture-triggered effects:** Normal (dissolve), Naruto (log and smoke), Goku (teleport), and Partículas (fragments in the wind).
- **Control panel:** switch effect and background while the camera keeps running, with no restart and no new calibration.
- **Custom backgrounds:** replace your room with any picture from a built-in gallery. When you vanish, the picture is what remains.
- **Remembers your choices:** the next launch starts with the last effect and background.
- **Works in calls:** virtual webcam for Teams, Zoom, Meet, and OBS, plus an optional virtual microphone so the people in the call hear the effect sounds (Linux).
- **Background that adapts:** it follows lighting and exposure changes, so you do not leave a shadow behind.

## Getting started

**Windows:** double-click `AVVIA_LOCALE.bat`. The first run creates a Python environment and installs the dependencies, which needs an internet connection once. To build a standalone `VanishCam.exe` that runs without Python, run `BUILD_WINDOWS.bat`.

**Linux:** create the environment once, then start the app:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python vanish_cam.py
```

For Microsoft Teams on Linux and a launcher in the GNOME app menu, see [Linux and Microsoft Teams](#linux-and-microsoft-teams).

Two windows open: the video preview and the **VanishCam** control panel.

1. Step out of the camera frame for 4 seconds while your real background is captured. The panel tells you when you can step back in. If you choose a picture as your background, this step is skipped.
2. Pick an effect and a background in the panel.
3. Make the effect's gesture and vanish.

## Control panel

The panel sits next to the video, so the preview and the virtual camera still show only the video. It is styled like a broadcast control surface: a light faceplate, black ink, and a single red signal color for whatever is live.

- **Status line and tally lamp:** the line under the title bar says what is happening, and the lamp in the corner shows the same thing at a glance: green when ready, amber while capturing the background, red while you vanish or are invisible.
- **A · Efeito:** click a row, or press `1`–`4`, to switch effect. The active effect is the inverted row marked **■ ATIVO**. The camera, the virtual webcam, the call microphone, and the background stay as they are, and there is no new calibration.
- **B · Fundo:** **REAL** is your room (hatched until it has been captured); **IMG 01**, **IMG 02**… are your images, and the chosen one has a thick black frame. **ADICIONAR** opens the system file picker and copies the image into the gallery (up to 7, oldest removed first). Hover over an image and click **×** to remove it from the gallery (the original file is never touched). Going back to **REAL** reuses the background already captured; it only asks you to step out of the frame if it was never captured in this session.
- **C · Controles:** keys for **SUMIR** (`V`), **REAPARECER** (`R`), **NOVO FUNDO** (`SPACE`, only with the real background), and **TELA CHEIA** (`F`). Keys that do not apply are greyed out; SUMIR and REAPARECER are disabled in Goku mode.

The app remembers the last effect and background and starts with them next time. They are stored in `~/.config/vanishcam/` on Linux and `%APPDATA%\VanishCam\` on Windows. Closing the panel or pressing `Q` quits the app.

## Effects

**Normal:** Raise your index and middle fingers in a V sign. The dissolve starts as soon as the gesture is recognized and lasts about 3.6 seconds.

**Naruto:** Bring your palms together in front of the webcam, with your fingers side by side rather than interlaced. Keep both hands visible. As soon as the gesture is recognized, you vanish in a cloud of smoke and a wooden log appears, falls, and remains visible until you press `R`. No microphone is needed. Naruto mode uses `hand_landmarker.task` to track both hands. `assets/vanish.MP3` plays through your computer speakers when the disappearance starts, including when triggered with `V`. You can trigger it again after pressing `R`.

**Goku:** Hold your index and middle fingers straight and together against your forehead for 1 second. Animated vertical comic-style lines run across your silhouette just before you teleport away and linger briefly afterward. To reappear, lower your hand, then repeat the gesture for another 1 second. The return uses the reverse effect. If you break the pose, the countdown starts over. If you add an `assets/teleport.MP3` (not included), it plays on both disappearance and reappearance. The terminal shows the countdown for each gesture. The `V` and `R` keys are disabled in this mode: the gesture is the only way to disappear and reappear.

**Partículas:** Make the same V sign as in Normal. Your silhouette breaks into small colored fragments that drift sideways as if carried by the wind. The effect lasts about 2.1 seconds. It captures your silhouette once when the effect starts to keep the video smooth.

## Custom background image

The easiest way is the **Fundo** section of the control panel. The image fills the whole frame without being stretched (the excess is cropped from the center), and it is not mirrored. You appear in front of it, and when you vanish the image is what remains. Turn off the call app's own background effects so they do not stack with this one.

To start with a specific image from the command line, pass `--background-image` with a JPG or PNG file:

```powershell
local_app\AVVIA_LOCALE.bat --mode normal --background-image C:\Users\me\Pictures\beach.jpg
```

```bash
./teams.sh --mode goku --background-image ~/Pictures/beach.jpg
```

## Linux and Microsoft Teams

Teams on Linux runs in the browser (teams.microsoft.com). Create a virtual webcam once per boot with [v4l2loopback](https://github.com/umlaeute/v4l2loopback), then start the app with `teams.sh`, which sends the video to `/dev/video10` without mirroring it:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1 card_label="VanishCam"
./teams.sh
```

To get a **VanishCam** entry in the GNOME app menu, run `./install-launcher.sh` once. The entry runs `vanishcam-launcher.sh`, which asks for your password in a system dialog only when the virtual webcam is missing (after a reboot), then opens the app with its control panel. The **ADICIONAR** tile in the panel uses `zenity` (preinstalled on GNOME) or `kdialog` for the file picker.

In Teams, choose **VanishCam** under **Settings → Devices → Camera**. Start the app before joining the call, and turn off Teams' own background effects.

`teams.sh` also passes `--call-audio`, which creates a **VanishCam Mic** microphone with PipeWire: your real microphone plus the effect sounds, so the people in the call hear the Naruto and Goku sounds too. Choose it under **Settings → Devices → Microphone**. It exists only while the app runs (nothing is saved in the audio settings), so keep the app open during the call or switch Teams back to your normal microphone.

## Virtual camera and OBS

If [OBS Studio](https://obsproject.com/) is installed, the app automatically tries to open **OBS Virtual Camera**. The terminal shows the device name. Use `AVVIA_VIRTUAL.bat` to require a virtual camera. Select **OBS Virtual Camera** in the app that will receive the video. OBS's virtual camera must be stopped while this app is producing its output.

To use the effect in an OBS scene, run `AVVIA_OBS.bat`, add a **Window Capture** source, select **Webcam - Anteprima**, and then start the virtual camera from OBS. Alternatively, use [Unity Capture](https://github.com/schellingb/UnityCapture) as a separate virtual video source:

```powershell
local_app\AVVIA_VIRTUAL.bat --virtual-backend unitycapture
```

OBS has only one virtual camera. Do not use it as both an OBS input and output at the same time. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Controls

The keys work in both the preview and the panel:

| Key | Action |
| --- | --- |
| `1`–`4` | Switch effect (Normal, Naruto, Goku, Partículas) |
| `V` | Vanish now, without the gesture |
| `R` | Reappear |
| `SPACE` | Capture a new real background |
| `F` | Toggle fullscreen |
| `Q` or `ESC` | Quit and release the cameras |

The `V` and `R` keys are disabled in Goku mode.

## Command-line options

| Option | Description |
| --- | --- |
| `--mode normal\|naruto\|goku\|particles` | Effect to start with (default: the last one used) |
| `--background-image FILE` | Picture to start with as the background (default: the last one used) |
| `--camera N` | Webcam index (default: 0) |
| `--width`, `--height` | Requested resolution (default: 1280x720) |
| `--fps N` | Maximum virtual-output FPS (default: 30) |
| `--virtual-camera`, `--no-virtual-camera` | Require or disable the virtual webcam (default: use it if available) |
| `--virtual-backend`, `--virtual-device` | Pick the virtual webcam driver and device |
| `--call-audio` | Create the **VanishCam Mic** virtual microphone (Linux, PipeWire) |
| `--no-mirror` | Do not mirror the video |
| `--fullscreen` | Start in fullscreen |
| `--static-background` | Keep the calibrated background instead of updating it |
| `--gesture-debug` | Print gesture recognition diagnostics in the terminal |

The default resolution is requested as MJPG so USB webcams can deliver it at full frame rate; if the webcam does not support it, the app uses the resolution the webcam returns and keeps its aspect ratio.

## How the real background stays clean

After calibration the background keeps updating wherever you are clearly not in the frame, so it follows lighting changes even while you are invisible. Changes in the webcam's automatic exposure, for example when you stand in front of a bright window, are also applied to the area behind you, so you do not leave a shadow where you were standing. If you still see a residue, turn off auto exposure in the webcam settings and press `SPACE` to capture a new background. Pass `--static-background` to keep the original snapshot instead.

To troubleshoot gesture recognition without adding text to the video, pass `--gesture-debug`. Diagnostic information appears only in the terminal.

## Visual assets

The [wooden log](assets/substitution_log.png) is an original illustration generated with ImageGen using the supplied image as a visual reference. Smoke is drawn in real time by `naruto_effect.py` so it follows the person's position and scales with the video. We also considered a [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet), but the procedural animation adapts better to dynamic positioning without an additional asset. No anime frames or assets are included in the project.

The control panel uses [IBM Plex](https://github.com/IBM/plex) Mono and Sans Condensed, bundled in `assets/fonts` under the [SIL Open Font License](assets/fonts/OFL.txt), so it looks the same on every system.

Goku mode (`goku_effect.py`) uses `face_landmarker.task` to locate the forehead and `hand_landmarker.task` to track the fingers, in a background thread so the video never waits for the models. The teleport's speed lines are drawn in code, without anime frames.

## Development

Run the tests from the project folder:

```bash
.venv/bin/python -m unittest
```

- `vanish_cam.py` runs the camera, calibration, background updates, keys, and outputs. Gesture detection runs in a background thread (`AsyncRecognizer`). Everything that depends on the effect (recognizer, sound, and `Mode`) lives in `EffectRuntime`, which is the only part replaced when you switch effect.
- `modes.py` has one class per effect (`NormalMode`, `NarutoMode`, `GokuMode`, `ParticlesMode`). Each turns detection results into phases (`ready`, `vanishing`, `vanished`, and `appearing` for Goku) and draws each frame.
- `control_panel.py` draws the panel with Pillow, using the IBM Plex fonts in `assets/fonts`, and turns clicks into actions. `user_data.py` stores preferences and the background gallery.

To add an effect, subclass `Mode`, register it in `MODES`, add its detector in `create_recognizer` and `detection_function`, and add its row to `MODE_CARDS`.

## Credits

VanishCam is a fork of [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam) by [Nicola Auriemma](https://github.com/Nicovic17), who developed the first two effects, **Normal** (dissolve) and **Naruto** (log and smoke), and the original app this repository is built on: the webcam pipeline, person segmentation, and gesture recognition with MediaPipe.
