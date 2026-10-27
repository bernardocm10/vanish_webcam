import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import call_audio
import vanish_cam as app
from call_audio import MIC_SOURCE, MIX_SINK, CallSound, VirtualMic, default_source
from test_video_output import arguments


def fake_process():
    process = Mock()
    process.poll.return_value = None
    return process


class VirtualMicTests(unittest.TestCase):
    def setUp(self):
        self.mocks = ExitStack()
        self.mocks.enter_context(patch.object(call_audio.shutil, "which", return_value="/usr/bin/tool"))
        self.popen = self.mocks.enter_context(
            patch.object(call_audio.subprocess, "Popen", side_effect=lambda *a, **k: fake_process())
        )
        # La prima verifica (istanza gia' attiva?) dice no, poi il nodo compare.
        self.exists = self.mocks.enter_context(
            patch.object(call_audio, "node_exists", side_effect=[False, True])
        )

    def tearDown(self):
        self.mocks.close()

    def test_mix_then_real_mic_into_the_mix(self):
        mic = VirtualMic(source="alsa_input.headset")
        mix, voice = (call.args[0] for call in self.popen.call_args_list)
        self.assertIn(f"node.name={MIX_SINK}", " ".join(mix))
        self.assertIn(f"node.name={MIC_SOURCE}", " ".join(mix))
        self.assertIn("priority.session=0", " ".join(mix))
        self.assertEqual(voice[voice.index("-C") + 1], "alsa_input.headset")
        self.assertEqual(voice[voice.index("-P") + 1], MIX_SINK)
        mic.play(Path("teleport.mp3"))
        self.assertEqual(self.popen.call_args.args[0], ["pw-play", "--target", MIX_SINK, "teleport.mp3"])
        processes = [result for result in mic.processes + mic.players]
        mic.close()
        for process in processes:
            process.terminate.assert_called_once()

    def test_refuses_to_loop_the_virtual_mic_into_itself(self):
        with self.assertRaisesRegex(RuntimeError, "já é"):
            VirtualMic(source=MIC_SOURCE)
        self.popen.assert_not_called()

    def test_missing_pipewire_tools(self):
        with patch.object(call_audio.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "PipeWire indisponível"):
                VirtualMic(source="alsa_input.headset")

    def test_default_source_reads_wpctl(self):
        output = '  * media.class = "Audio/Source"\n  * node.name = "alsa_input.usb-headset"\n'
        with patch.object(call_audio.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
            self.assertEqual(default_source(), "alsa_input.usb-headset")


class CallSoundTests(unittest.TestCase):
    def test_effect_sound_goes_to_speakers_and_call(self):
        local, mic = Mock(), Mock()
        sound = CallSound(local, mic, Path("vanish.MP3"))
        sound.play()
        local.play.assert_called_once()
        mic.play.assert_called_once_with(Path("vanish.MP3"))
        sound.close()
        local.close.assert_called_once()

    def test_unavailable_mic_never_stops_the_video(self):
        with ExitStack() as resources, patch.object(app, "VirtualMic", side_effect=RuntimeError("sem PipeWire")), \
                patch("builtins.print") as printed:
            self.assertIsNone(app.open_call_mic(arguments(call_audio=True), resources))
        self.assertIn("sem PipeWire", printed.call_args.args[0])

    def test_naruto_sound_is_shared_with_the_call(self):
        mic = Mock()
        with patch.object(app, "VanishSound", return_value=Mock()):
            sound = app.open_sound("naruto", mic)
        self.assertIsInstance(sound, CallSound)
        self.assertIsNone(app.open_sound("normal", mic))


if __name__ == "__main__":
    unittest.main()
