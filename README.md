# VanishCam — webcam disappearance effects

Video is processed on your computer. It is neither recorded nor sent to a server. The preview window and virtual camera show only the video and effects, without overlaid menus, instructions, or text.

For step-by-step instructions in English and Italian, see the [user guide](USER_GUIDE_EN_IT.md).

## ☕ Buy me a coffee

If you like this project and want to support it, you can buy me a coffee through [PayPal](https://paypal.me/nicovic17) or [Revolut](https://revolut.me/nicoauriemma).

## Download for Windows

[Download VanishCam.exe]( https://nicoaauriemma.vercel.app/shop ) and double-click it to run the app. You do not need to install Python.

The published EXE may predate the Particles and Goku modes. To use them now, run the local scripts or rebuild the EXE from these sources.

## Getting started

1. Double-click `AVVIA_LOCALE.bat`.
2. In the start menu, choose **Normal Vanish**, **Naruto Vanish**, **Particles**, or **Goku** with the mouse or the `1`, `2`, `3`, and `4` keys.
3. Step out of the camera frame for 4 seconds while the background is captured. The terminal tells you when you can step back in.

**Normal Vanish:** Raise your index and middle fingers in a V sign. The dissolve starts as soon as the gesture is recognized and lasts about 3.6 seconds.

**Particles:** *(described upstream but not included in this source code; `--mode particles` is not available.)* Make the same V sign as in Normal Vanish. Your silhouette breaks into small colored fragments that drift sideways as if carried by the wind. The effect lasts about 2.1 seconds. It captures your silhouette once when the effect starts to keep the video smooth.

**Goku:** Hold your index and middle fingers straight and together against your forehead for 1 second. Animated vertical comic-style lines run across your silhouette just before you teleport away and linger briefly afterward. To reappear, lower your hand, then repeat the gesture for another 1 second. The return uses the reverse effect. If you break the pose, the countdown starts over. If you add an `assets/teleport.MP3` (not included), it plays on both disappearance and reappearance. The terminal shows the countdown for each gesture. The `V` and `R` keys are disabled in this mode: the gesture is the only way to disappear and reappear. No text is overlaid on the video.

**Naruto Vanish:** Bring your palms together in front of the webcam, with your fingers side by side rather than interlaced. Keep both hands visible. As soon as the gesture is recognized, you vanish in a cloud of smoke and a wooden log appears, falls, and remains visible until you press `R`. No microphone is needed. Naruto mode uses `hand_landmarker.task` to track both hands. `assets/vanish.MP3` plays through your computer speakers when the disappearance starts, including when triggered with `V`. You can trigger it again after pressing `R`.

To start a mode without the menu:

```powershell
local_app\AVVIA_LOCALE.bat --mode naruto
local_app\AVVIA_LOCALE.bat --mode normal
local_app\AVVIA_LOCALE.bat --mode goku
```

## Linux and Microsoft Teams

Teams on Linux runs in the browser (teams.microsoft.com). Create a virtual webcam once per boot with [v4l2loopback](https://github.com/umlaeute/v4l2loopback), then start the app with `teams.sh`, which sends the video to `/dev/video10` without mirroring it:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1 card_label="VanishCam"
./teams.sh --mode goku
```

In Teams, choose **VanishCam** under **Settings → Devices → Camera**. Start the app before joining the call, and turn off Teams' own background effects.

## Virtual camera and OBS

If [OBS Studio](https://obsproject.com/) is installed, the app automatically tries to open **OBS Virtual Camera**. The terminal shows the device name. Use `AVVIA_VIRTUAL.bat` to require a virtual camera. Select **OBS Virtual Camera** in the app that will receive the video. OBS's virtual camera must be stopped while this app is producing its output.

To use the effect in an OBS scene, run `AVVIA_OBS.bat`, add a **Window Capture** source, select **Webcam - Anteprima**, and then start the virtual camera from OBS. Alternatively, use [Unity Capture](https://github.com/schellingb/UnityCapture) as a separate virtual video source:

```powershell
local_app\AVVIA_VIRTUAL.bat --virtual-backend unitycapture
```

OBS has only one virtual camera. Do not use it as both an OBS input and output at the same time. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Controls

- `SPACE`: capture a new background.
- `R`: make the person reappear.
- `V`: manually start the selected mode's effect.
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit and release the cameras.

The `V` and `R` keys are disabled in Goku mode.

The default resolution is 1280x720 (16:9), requested as MJPG so USB webcams can deliver it at full frame rate; if the webcam does not support it, the app uses the resolution the webcam returns and keeps its aspect ratio. You can change the camera, resolution, and maximum virtual-output FPS:

```powershell
local_app\AVVIA_LOCALE.bat --camera 1 --width 1280 --height 720 --fps 30
```

To troubleshoot gesture recognition without adding text to the video, pass `--gesture-debug`. Diagnostic information appears only in the terminal.

## Visual assets

The [wooden log](assets/substitution_log.png) is an original illustration generated with ImageGen using the supplied image as a visual reference. Smoke is drawn in real time by `naruto_effect.py` so it follows the person's position and scales with the video. We also considered a [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet), but the procedural animation adapts better to dynamic positioning without an additional asset. No anime frames or assets are included in the project.

Goku mode (`goku_effect.py`) uses `face_landmarker.task` to locate the forehead and `hand_landmarker.task` to track the fingers, in a background thread so the video never waits for the models. The teleport's speed lines are drawn in code, without anime frames.

## Development

Run the tests from the project folder:

```bash
.venv/bin/python -m unittest
```
