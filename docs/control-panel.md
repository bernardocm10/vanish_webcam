# Control panel and backgrounds

[← Documentation](README.md)

<img src="panel.png" alt="VanishCam control panel" width="300" align="right">

VanishCam opens a single window: the panel on the left and the live video filling the rest. Resize or maximize it and the video grows with it, never stretched. Everything you click lives in the panel, so the video you send to a call shows only you and the effect.

The panel is styled like a broadcast control surface: a light faceplate, black ink, and a single red signal color for whatever is live.

It follows your system language (see [Language](configuration.md#language)). This guide uses the English labels, with the Portuguese ones in parentheses.

The panel has two tabs in its black top bar: **LIVE** (*AO VIVO*), where the app opens, with everything below, and **SETTINGS** (*AJUSTES*), with the [keyboard shortcuts](#keyboard-shortcuts). The screenshot shows the panel before the tabs were added.

## What is on the panel

**Status line and tally lamp** — say what is happening right now:

| Lamp | Meaning |
| --- | --- |
| Green | Ready: make the gesture |
| Amber | Capturing the background (step out of the frame), or opening the camera |
| Red | Vanishing, invisible, or reappearing |

**A · Effect** (*Efeito*) — the active effect. Click it to open the list and pick another, or press `1`–`5`; the active one is marked **■ LIVE** (*ATIVO*). Switching keeps the camera, the virtual webcam, the call microphone, and the background as they are: no restart, no new calibration.

**B · Background** (*Fundo*) — choose what is behind you:

- **REAL** is your room. It is hatched until it has been captured.
- **IMG 01**, **IMG 02**… are your pictures. The chosen one has a thick black frame.
- **ADD** (*ADICIONAR*) opens your system's file picker and copies the picture into the gallery (up to 7; the oldest is removed first).
- Hover over a picture and click **×** to remove it from the gallery. Your original file is never touched.

**C · Camera** (*Câmera*) — the webcam in use and how it is doing:

- The lamp is green while the webcam sends images and amber while it is opening or reconnecting. Under the name: the real resolution and frame rate.
- Click the name to pick another webcam. The list is read again each time it opens, so a webcam plugged in meanwhile shows up. Virtual cameras (OBS Virtual Camera and the like) are left out: they are outputs, and reading one back would loop the video.
- **REOPEN** (*REABRIR*) lets the webcam go, restarts its driver, and opens it again, without stopping the app: it fixes frozen or black images that reopening alone does not. Restarting a driver needs administrator rights, so Windows asks for permission each time; if you decline, the webcam is only reopened. On Linux it only reopens the webcam.
- **SETTINGS** (*AJUSTES*) opens the webcam driver's own settings window (exposure, focus, white balance). Windows only.
- **DETACH** (*DESTACAR*) moves the video to its own **VanishCam - Video** window, for a second monitor or OBS's window capture, and the panel shrinks to its own width. **ATTACH** (*ACOPLAR*), or closing the video window, brings it back. The app remembers where you left it.

**D · Controls** (*Controles*) — **VANISH** (*SUMIR*), **REAPPEAR** (*REAPARECER*), **NEW BACKGROUND** (*NOVO FUNDO*), **FULLSCREEN** (*TELA CHEIA*). Buttons that do not apply right now are greyed out. With keyboard shortcuts on, each button shows its key in the corner.

<br clear="right">

## Using a picture as your background

Pick it in **B · Background**, or start the app with one from the command line:

```bash
.venv/bin/python -m vanishcam --background-image ~/Pictures/beach.jpg
```

```powershell
.\scripts\windows\start.bat --background-image C:\Users\me\Pictures\beach.jpg
```

- The picture fills the whole frame without being stretched (the excess is cropped from the center) and is never mirrored.
- You appear in front of it, and when you vanish, the picture is what remains.
- There is no 4-second calibration while a picture is selected, and **NEW BACKGROUND** is disabled.
- Going back to **REAL** reuses the room already captured. It only asks you to step out of the frame if the room was never captured in this session.
- Turn off the call app's own background effects (blur, virtual background) so they do not stack with this one.

## Keyboard shortcuts

Shortcuts are **off** until you turn them on, so typing with the VanishCam window focused (thinking you are in the call's chat, say) does nothing. The footer of the panel says whether they are on.

To use them, open **SETTINGS** (*AJUSTES*) and click **Shortcuts off** (*Atalhos desligados*). The list below it shows one key per action:

| Action | Key when you turn them on |
| --- | --- |
| Vanish now, without the gesture | `V` |
| Reappear | `R` |
| Capture a new real background | `SPACE` |
| Fullscreen with only the video (in its own window when detached) | `F` |
| Detach the video to its own window, or attach it back | `D` |
| Switch effect (Normal, Naruto, Goku, Particles, Ash) | `1`–`5` |
| Quit | none |

- **To change a key**, click its row and press the new key: a letter, a digit, punctuation, or space. Combinations with Ctrl or Alt and the function keys are not supported, because the window cannot tell them apart reliably on every system. `ESC` cancels. A key already used by another action is refused, and the panel says which one has it.
- **RESTORE DEFAULTS** (*RESTAURAR PADRÃO*) puts back the keys in the table and keeps the shortcuts on.
- **`ESC`** always leaves fullscreen, with shortcuts on or off. It never quits.
- **Closing the window** quits the app and releases the camera.

Vanish and reappear do nothing in Goku mode: only the gesture counts there.

## Saved preferences

The app remembers the last effect, background, webcam, and your keyboard shortcuts, and starts with them next time. The webcam is remembered by name, so it is found again even if its number changes. `--mode`, `--background-image`, and `--camera` override them for one launch.

| System | Folder |
| --- | --- |
| Linux | `~/.config/vanishcam/` |
| Windows | `%APPDATA%\VanishCam\` |

The folder holds `settings.json` and a `backgrounds/` folder with the copies of your gallery pictures. Delete it to start fresh.
