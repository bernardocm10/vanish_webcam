# Calls, Teams and OBS

[← Documentation](README.md)

To use VanishCam in a call, the app sends its video to a **virtual webcam**, which you then pick as the camera in Teams, Zoom, Meet, or OBS.

- [Linux and Microsoft Teams](#linux-and-microsoft-teams)
- [Windows: virtual camera and OBS](#windows-virtual-camera-and-obs)

## Linux and Microsoft Teams

Teams on Linux runs in the browser (teams.microsoft.com).

### 1. Create the virtual webcam

Once per boot, with [v4l2loopback](https://github.com/umlaeute/v4l2loopback):

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1 card_label="VanishCam"
```

### 2. Start the app with `scripts/linux/teams.sh`

```bash
scripts/linux/teams.sh
```

It sends the video to `/dev/video10` without mirroring it, and creates the **VanishCam Mic** microphone (see below). Extra options are passed through, for example `scripts/linux/teams.sh --mode goku`.

### 3. Pick the devices in Teams

- **Settings → Devices → Camera:** choose **VanishCam**.
- **Settings → Devices → Microphone:** choose **VanishCam Mic** if you want the call to hear the effect sounds.
- Start the app **before** joining the call, and turn off Teams' own background effects.

### The VanishCam Mic

`teams.sh` passes `--call-audio`, which uses PipeWire to create a **VanishCam Mic**: your real microphone plus the effect sounds, so the people in the call hear the Naruto and Goku sounds too.

It exists only while the app runs and nothing is saved in your audio settings. Keep the app open during the call, or switch Teams back to your normal microphone.

### App menu launcher (GNOME)

Run this once to add **VanishCam** to the GNOME app menu:

```bash
scripts/linux/install-launcher.sh
```

The launcher runs `scripts/linux/launcher.sh`. After a reboot, when the virtual webcam is missing, it asks for your password in a system dialog to create it; then it opens the app. The panel's **ADD** button uses `zenity` (preinstalled on GNOME) or `kdialog` as the file picker.

To remove the launcher: `rm ~/.local/share/applications/vanishcam.desktop`.

## Windows: virtual camera and OBS

### Virtual camera

If [OBS Studio](https://obsproject.com/) is installed, the app automatically tries to open **OBS Virtual Camera**; the terminal shows the device name. Then select **OBS Virtual Camera** as the camera in your call app.

- `scripts\windows\start-virtual.bat` starts the app and **requires** a virtual camera.
- OBS's own virtual camera must be **stopped** while VanishCam is sending video to it.

### Inside an OBS scene

1. Run `scripts\windows\start-obs.bat`.
2. In OBS, add a **Window Capture** source and select the **Webcam - Anteprima** window.
3. Start the virtual camera from OBS.

OBS has only one virtual camera, so do not use it as both an OBS input and output at the same time.

### Unity Capture

As an alternative virtual video source, use [Unity Capture](https://github.com/schellingb/UnityCapture):

```powershell
.\scripts\windows\start-virtual.bat --virtual-backend unitycapture
```

See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras) for all supported virtual cameras.
