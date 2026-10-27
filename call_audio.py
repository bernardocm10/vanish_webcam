"""Virtual microphone for video calls: your real mic plus the effect sounds.

PipeWire only (Linux). While the app runs, two pw-loopback processes create
"VanishCam Mic"; nothing is saved in the audio configuration and the device
disappears when the app closes. In Teams, pick it as the microphone.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

try:
    from .i18n import t
except ImportError:
    from i18n import t


MIX_SINK = "vanishcam_mix"
MIC_SOURCE = "vanishcam_mic"
MIC_DESCRIPTION = "VanishCam Mic"
# Zero priority: the system never picks them as defaults on its own
# (otherwise all the computer's audio would end up in the call).
_QUIET = "priority.session=0 priority.driver=0 node.virtual=true"


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


class VirtualMic:
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


class CallSound:
    """Play an effect sound on the speakers and into the call at the same time."""

    def __init__(self, local, mic: VirtualMic, path: Path):
        self.local = local
        self.mic = mic
        self.path = path

    def play(self) -> None:
        self.local.play()
        self.mic.play(self.path)

    def close(self) -> None:
        self.local.close()
