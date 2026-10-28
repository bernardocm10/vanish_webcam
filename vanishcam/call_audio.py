"""Virtual microphone for video calls: your real mic plus the effect sounds.

Linux (PipeWire): while the app runs, two pw-loopback processes create
"VanishCam Mic"; nothing is saved in the audio configuration and the device
disappears when the app closes.

Windows (VB-CABLE): Windows cannot create an audio device without a driver, so
the free VB-CABLE provides one. The app plays your microphone plus the effects
into "CABLE Input", and the call app records them from "CABLE Output".
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .i18n import t

MIX_SINK = "vanishcam_mix"
MIC_SOURCE = "vanishcam_mic"
MIC_DESCRIPTION = "VanishCam Mic"
# Zero priority: the system never picks them as defaults on its own
# (otherwise all the computer's audio would end up in the call).
_QUIET = "priority.session=0 priority.driver=0 node.virtual=true"

# VB-CABLE: the app plays into CABLE_PLAYBACK; calls record from CABLE_RECORDING.
CABLE_PLAYBACK = "CABLE Input"
CABLE_RECORDING = "CABLE Output"
CABLE_DESCRIPTION = "CABLE Output (VB-Audio Virtual Cable)"
CABLE_URL = "https://vb-audio.com/Cable/"
EFFECT_GAIN = 0.8


def default_source() -> str:
    """node.name of the current default microphone."""
    output = subprocess.run(
        ["wpctl", "inspect", "@DEFAULT_AUDIO_SOURCE@"],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout
    match = re.search(r'node\.name = "([^"]+)"', output)
    if match is None:
        raise RuntimeError(t("mic.no_default"))
    return match.group(1)


def node_exists(name: str) -> bool:
    output = subprocess.run(
        ["pw-cli", "ls", "Node"], capture_output=True, text=True, timeout=5
    ).stdout
    return f'node.name = "{name}"' in output


class PipeWireMic:
    description = MIC_DESCRIPTION

    def __init__(self, source: str | None = None):
        missing = [tool for tool in ("pw-loopback", "pw-play", "pw-cli", "wpctl") if shutil.which(tool) is None]
        if missing:
            raise RuntimeError(t("mic.no_pipewire", tools=", ".join(missing)))
        self.source = source or default_source()
        if self.source in (MIC_SOURCE, MIX_SINK):
            raise RuntimeError(t("mic.default_is_virtual", name=MIC_DESCRIPTION))
        if node_exists(MIX_SINK):
            raise RuntimeError(t("mic.already_running", name=MIC_DESCRIPTION))
        self.processes: list[subprocess.Popen] = []
        self.players: list[subprocess.Popen] = []
        try:
            # Mix: whatever enters the sink comes out of the virtual microphone.
            self._spawn(
                "--name", "vanishcam-mic",
                "--capture-props",
                f'media.class=Audio/Sink node.name={MIX_SINK} node.description="VanishCam mix" {_QUIET}',
                "--playback-props",
                f'media.class=Audio/Source node.name={MIC_SOURCE} node.description="{MIC_DESCRIPTION}" {_QUIET}',
            )
            self._wait_for(MIX_SINK)
            # Voice: the real microphone goes into the mix.
            self._spawn(
                # Short options: PipeWire 1.0 does not accept --channels/--channel-map.
                "--name", "vanishcam-voice", "-c", "1", "-m", "[ MONO ]",
                "-C", self.source, "-P", MIX_SINK,
                "--playback-props", "node.dont-reconnect=true",
            )
        except Exception:
            self.close()
            raise

    def _spawn(self, *arguments: str) -> None:
        # setpriv --pdeathsig: if the app dies (even with kill -9) the loopback dies
        # too, so "VanishCam Mic" is not left orphaned in the system.
        guard = ["setpriv", "--pdeathsig", "TERM"] if shutil.which("setpriv") else []
        self.processes.append(subprocess.Popen(
            [*guard, "pw-loopback", *arguments],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))

    def _wait_for(self, name: str, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in self.processes):
                raise RuntimeError(t("mic.loopback_exited"))
            if node_exists(name):
                return
            time.sleep(0.05)
        raise RuntimeError(t("mic.timeout", name=name, seconds=timeout))

    def prepare(self, path: Path) -> None:
        """pw-play reads the file itself: nothing to decode ahead."""

    def play(self, path: Path) -> None:
        """Send a sound into the call (the local speakers are handled by VanishSound)."""
        self.players = [player for player in self.players if player.poll() is None]
        self.players.append(subprocess.Popen(
            ["pw-play", "--target", MIX_SINK, str(path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))

    def close(self) -> None:
        for process in self.players + self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.players + self.processes:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self.players.clear()
        self.processes.clear()


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear resampling, channel by channel: enough for short effect sounds."""
    if from_rate == to_rate or len(samples) < 2:
        return samples
    count = max(1, round(len(samples) * to_rate / from_rate))
    positions = np.linspace(0, len(samples) - 1, count)
    source = np.arange(len(samples))
    return np.stack([np.interp(positions, source, samples[:, c]) for c in range(samples.shape[1])], axis=1)


def decode_sound(path: Path, rate: int) -> np.ndarray:
    """Stereo float32 samples at the given rate, decoded by the mixer VanishSound already opened."""
    from pygame import mixer, sndarray

    settings = mixer.get_init()
    if settings is None:
        raise RuntimeError(t("mic.no_mixer"))
    frequency, size, _ = settings
    samples = sndarray.array(mixer.Sound(str(path))).astype(np.float32)
    samples /= float(2 ** (abs(size) - 1))
    if samples.ndim == 1:
        samples = samples[:, None]
    stereo = np.repeat(samples, 2, axis=1) if samples.shape[1] == 1 else samples[:, :2]
    return resample(stereo, frequency, rate).astype(np.float32)


def find_cable(devices, hostapi: int) -> int | None:
    """Index of VB-CABLE's playback side in the same host API as the microphone (PortAudio needs both in one)."""
    for index, device in enumerate(devices):
        if (device["hostapi"] == hostapi and device["max_output_channels"] > 0
                and device["name"].startswith(CABLE_PLAYBACK)):
            return index
    return None


class CableMic:
    """Your microphone plus the effect sounds, played into VB-CABLE for the call to record."""

    description = CABLE_DESCRIPTION

    def __init__(self):
        try:
            import sounddevice
        except (ImportError, OSError) as error:
            raise RuntimeError(t("mic.no_sounddevice", error=error)) from error
        devices = sounddevice.query_devices()
        source, speakers = sounddevice.default.device
        if source is None or source < 0:
            raise RuntimeError(t("mic.no_default_input"))
        microphone = devices[source]
        if microphone["name"].startswith(CABLE_RECORDING):
            raise RuntimeError(t("mic.default_is_virtual", name=CABLE_DESCRIPTION))
        cable = find_cable(devices, microphone["hostapi"])
        if cable is None:
            raise RuntimeError(t("mic.no_cable", url=CABLE_URL))
        if speakers is not None and speakers >= 0 and devices[speakers]["name"].startswith(CABLE_PLAYBACK):
            # The installer sometimes makes the cable the default output: you would hear
            # nothing and the whole computer's audio would go into the call.
            print(f"[VanishCam Mic] {t('mic.cable_is_default_output', name=devices[speakers]['name'])}")

        self.rate = int(devices[cable]["default_samplerate"])
        self.lock = threading.Lock()
        self.sounds: dict[Path, np.ndarray | None] = {}
        self.playing: list[tuple[np.ndarray, int]] = []
        self.stream = sounddevice.Stream(
            device=(source, cable), samplerate=self.rate, channels=(1, 2),
            dtype="float32", latency="low", callback=self._mix,
        )
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            raise

    def prepare(self, path: Path) -> None:
        """Decode a sound before it is needed: play() runs on the video thread."""
        if path in self.sounds:
            return
        try:
            self.sounds[path] = decode_sound(path, self.rate)
        except Exception as error:
            # The call just misses this sound; the speakers still play it.
            print(f"[VanishCam Mic] {t('mic.sound_unavailable', name=path.name, error=error)}")
            self.sounds[path] = None

    def play(self, path: Path) -> None:
        self.prepare(path)
        samples = self.sounds[path]
        if samples is not None:
            with self.lock:
                self.playing.append((samples, 0))

    def _mix(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio thread: voice on both channels, plus whatever effects are still playing."""
        outdata[:] = indata[:, :1]
        with self.lock:
            still = []
            for samples, offset in self.playing:
                chunk = samples[offset : offset + frames]
                outdata[: len(chunk)] += chunk * EFFECT_GAIN
                if offset + frames < len(samples):
                    still.append((samples, offset + frames))
            self.playing = still
        np.clip(outdata, -1.0, 1.0, out=outdata)

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


VirtualMic = PipeWireMic | CableMic


def open_virtual_mic() -> VirtualMic:
    return CableMic() if sys.platform == "win32" else PipeWireMic()


class CallSound:
    """Play an effect sound on the speakers and into the call at the same time."""

    def __init__(self, local, mic: VirtualMic, path: Path):
        self.local = local
        self.mic = mic
        self.path = path
        mic.prepare(path)

    def play(self) -> None:
        self.local.play()
        self.mic.play(self.path)

    def close(self) -> None:
        self.local.close()
