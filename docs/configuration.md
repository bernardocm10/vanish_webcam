# Options and troubleshooting

[← Documentation](README.md)

## Command-line options

Pass them after the start command, for example `.venv/bin/python vanish_cam.py --camera 1` or `start.bat --camera 1`.

| Option | What it does |
| --- | --- |
| `--mode normal\|naruto\|goku\|particles\|ash` | Effect to start with (default: the last one used) |
| `--background-image FILE` | Picture to start with as the background (default: the last one used) |
| `--camera N` | Webcam number (default: 0) |
| `--width`, `--height` | Requested resolution (default: 1280x720) |
| `--fps N` | Maximum frame rate sent to the virtual webcam (default: 30) |
| `--virtual-camera` / `--no-virtual-camera` | Require or disable the virtual webcam (default: use it if available) |
| `--virtual-backend`, `--virtual-device` | Pick the virtual webcam driver and device |
| `--call-audio` | Create the **VanishCam Mic** virtual microphone (Linux, PipeWire) |
| `--no-mirror` | Do not mirror the video |
| `--fullscreen` | Start in fullscreen |
| `--static-background` | Keep the captured background as it is instead of updating it |
| `--gesture-debug` | Print gesture recognition details in the terminal |
| `--lang en\|pt` | Interface language (default: the system language) |

## Language

The panel, the terminal messages, the errors, and `--help` are in English or Brazilian Portuguese. The app follows the system language and uses English for any other language. To choose one yourself:

- `--lang pt` or `--lang en` for one launch, or
- the `VANISHCAM_LANG` environment variable (`pt` or `en`), which also applies to the start scripts.

## Camera and resolution

The app asks the webcam for 1280x720 in MJPG, the format that lets most USB webcams run at full frame rate. If the webcam does not support it, the app uses whatever resolution the webcam returns and keeps its aspect ratio.

If the wrong camera opens, try `--camera 1` (or 2, 3…). If the app says the webcam is unavailable, close other apps that are using it.

## Troubleshooting

### I leave a shadow or residue where I was standing

The captured room keeps updating wherever you are clearly not in the frame, so it follows lighting changes even while you are invisible. Changes in the webcam's automatic exposure (for example when you stand in front of a bright window) are also applied to the area behind you.

If you still see a residue:

1. Turn off auto exposure in the webcam settings.
2. Press `SPACE` (**NEW BACKGROUND**) and step out of the frame to capture the room again.

To keep the original snapshot and never update it, use `--static-background`.

### The gesture is not recognized

- Keep your hand fully visible and well lit, facing the camera.
- For Naruto, keep **both** hands in the frame, press the palms together (wrists touching too), fingers side by side and pointing up, and hold for a moment.
- For Goku, hold the pose for a full second: breaking it restarts the countdown.
- For Particles, keep your face visible, and the four fingers straight and together on your brow.
- Run with `--gesture-debug` to see in the terminal what the app detects. Nothing is drawn on the video.

You can always press `V` to vanish without the gesture (except in Goku).
