# Options and troubleshooting

[← Documentation](README.md)

## Command-line options

Pass them after the start command, for example `.venv/bin/python -m vanishcam --camera 1` or `scripts\windows\start.bat --camera 1`.

| Option | What it does |
| --- | --- |
| `--mode normal\|naruto\|goku\|particles\|ash` | Effect to start with (default: the last one used) |
| `--background-image FILE` | Picture to start with as the background (default: the last one used) |
| `--camera N` | Webcam number (default: the one used last time, or the first found); you can also pick it in the panel |
| `--width`, `--height` | Requested resolution (default: 1280x720) |
| `--fps N` | Maximum frame rate sent to the virtual webcam (default: 30) |
| `--virtual-camera` / `--no-virtual-camera` | Require or disable the virtual webcam (default: use it if available) |
| `--virtual-backend`, `--virtual-device` | Pick the virtual webcam driver and device |
| `--call-audio` | Send your voice plus the effect sounds to the call: **VanishCam Mic** on Linux (PipeWire), **CABLE Output** on Windows ([VB-CABLE](calls.md#windows-call-microphone-with-vb-cable)) |
| `--no-mirror` | Do not mirror the video |
| `--fullscreen` | Start in fullscreen, with only the video |
| `--video-window` | Start with the video detached in its own **VanishCam - Video** window (**DETACH** in the panel does the same) |
| `--static-background` | Keep the captured background as it is instead of updating it |
| `--gesture-debug` | Print gesture recognition details in the terminal |
| `--lang en\|pt` | Interface language (default: the system language) |

## Language

The panel, the terminal messages, the errors, and `--help` are in English or Brazilian Portuguese. The app follows the system language and uses English for any other language. To choose one yourself:

- `--lang pt` or `--lang en` for one launch, or
- the `VANISHCAM_LANG` environment variable (`pt` or `en`), which also applies to the start scripts.

## Camera and resolution

The app asks the webcam for 1280x720 in MJPG, the format that lets most USB webcams run at full frame rate. If the webcam does not support it, the app uses whatever resolution the webcam returns and keeps its aspect ratio.

If the wrong camera opens, pick the right one in **C · Camera** in the panel; the app remembers it. If the app says the webcam is unavailable, close other apps that are using it, then click **REOPEN**.

### If the webcam disconnects

If the webcam stops sending images for half a second (unplugged, or taken by another app), the app keeps running: it tries to reopen it every second, and meanwhile the preview and the call show a **Reconnecting the camera** card over a blurred, dimmed copy of the last image, so the call does not freeze on a still frame. This card is the only text the app ever puts on the video. As soon as the webcam is back the video resumes, with the same effect and background; if it comes back at another resolution, it is scaled to the one the call already receives. Close the window to give up and quit.

## Troubleshooting

### I leave a shadow or residue where I was standing

The captured room keeps updating wherever you are clearly not in the frame, so it follows lighting changes even while you are invisible. Changes in the webcam's automatic exposure (for example when you stand in front of a bright window) are also applied to the area behind you.

If you still see a residue:

1. Turn off auto exposure in the webcam settings.
2. Click **NEW BACKGROUND** and step out of the frame to capture the room again.

To keep the original snapshot and never update it, use `--static-background`.

### The gesture is not recognized

- Keep your hand fully visible and well lit, facing the camera.
- For Naruto, keep **both** hands in the frame, and either press the palms together (wrists touching too, fingers pointing up) or clasp your hands with the fingers interlaced, and hold for a moment. If `--gesture-debug` shows `hands=1`, one hand is hiding the other: turn them a little towards the camera.
- For Goku, hold the pose for a full second: breaking it restarts the countdown.
- For Ash, keep your face and **both** hands in the frame: open hands, one on each side of your head, above your chin.
- For Particles, keep your face visible, and the four fingers straight and together on your brow.
- Run with `--gesture-debug` to see in the terminal what the app detects. Nothing is drawn on the video.

You can always click **VANISH** to vanish without the gesture (except in Goku).
