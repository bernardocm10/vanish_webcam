import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from vanishcam import call_audio
from vanishcam import app
from vanishcam.call_audio import (
    CABLE_DESCRIPTION, CABLE_URL, EFFECT_GAIN, MIC_DESCRIPTION, MIC_SOURCE, MIX_SINK, CableMic, CallSound,
    PipeWireMic, default_source, open_virtual_mic, resample,
)
from vanishcam.i18n import t
from vanishcam.modes import GokuMode, NarutoMode
from .test_video_output import arguments


def fake_process():
    process = Mock()
    process.poll.return_value = None
    return process


class PipeWireMicTests(unittest.TestCase):
    def setUp(self):
        self.mocks = ExitStack()
        self.mocks.enter_context(patch.object(call_audio.shutil, "which", return_value="/usr/bin/tool"))
        self.popen = self.mocks.enter_context(
            patch.object(call_audio.subprocess, "Popen", side_effect=lambda *a, **k: fake_process())
        )
        # The first check (instance already running?) says no, then the node appears.
        self.exists = self.mocks.enter_context(
            patch.object(call_audio, "node_exists", side_effect=[False, True])
        )

    def tearDown(self):
        self.mocks.close()

    def test_mix_then_real_mic_into_the_mix(self):
        mic = PipeWireMic(source="alsa_input.headset")
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
        with self.assertRaises(RuntimeError) as raised:
            PipeWireMic(source=MIC_SOURCE)
        self.assertEqual(str(raised.exception), t("mic.default_is_virtual", name=MIC_DESCRIPTION))
        self.popen.assert_not_called()

    def test_missing_pipewire_tools(self):
        with patch.object(call_audio.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "pw-loopback"):
                PipeWireMic(source="alsa_input.headset")

    def test_default_source_reads_wpctl(self):
        output = '  * media.class = "Audio/Source"\n  * node.name = "alsa_input.usb-headset"\n'
        with patch.object(call_audio.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
            self.assertEqual(default_source(), "alsa_input.usb-headset")


def audio_device(name, hostapi=0, inputs=0, outputs=0, rate=48000.0):
    return {"name": name, "hostapi": hostapi, "max_input_channels": inputs,
            "max_output_channels": outputs, "default_samplerate": rate}


WINDOWS_DEVICES = [
    audio_device("Microfone (Headset)", inputs=1),
    audio_device("Alto-falantes (Headset)", outputs=2),
    audio_device("CABLE Output (VB-Audio Virtual C", inputs=2),
    audio_device("CABLE Input (VB-Audio Virtual C", outputs=2, rate=44100.0),
    audio_device("CABLE Input (VB-Audio Virtual Cable)", hostapi=2, outputs=2),
]


def fake_sounddevice(devices=WINDOWS_DEVICES, default=(0, 1)):
    stream = Mock()
    module = SimpleNamespace(
        query_devices=Mock(return_value=devices),
        default=SimpleNamespace(device=default),
        Stream=Mock(return_value=stream),
    )
    return module, stream


class CableMicTests(unittest.TestCase):
    def open(self, **options):
        module, stream = fake_sounddevice(**options)
        with patch.dict("sys.modules", sounddevice=module):
            return CableMic(), module, stream

    def test_default_microphone_plays_into_the_cable_of_the_same_host_api(self):
        mic, module, stream = self.open()
        options = module.Stream.call_args.kwargs
        self.assertEqual(options["device"], (0, 3))
        self.assertEqual(options["samplerate"], 44100)
        self.assertEqual(options["channels"], (1, 2))
        stream.start.assert_called_once()
        self.assertEqual(mic.description, CABLE_DESCRIPTION)
        mic.close()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_missing_cable_says_where_to_get_it(self):
        devices = [device for device in WINDOWS_DEVICES if "CABLE" not in device["name"]]
        with self.assertRaises(RuntimeError) as raised:
            self.open(devices=devices)
        self.assertIn(CABLE_URL, str(raised.exception))

    def test_refuses_to_loop_the_cable_into_itself(self):
        with self.assertRaises(RuntimeError) as raised:
            self.open(default=(2, 1))
        self.assertEqual(str(raised.exception), t("mic.default_is_virtual", name=CABLE_DESCRIPTION))

    def test_warns_when_the_cable_is_the_default_output(self):
        with patch("builtins.print") as printed:
            self.open(default=(0, 3))
        self.assertIn("CABLE Input", printed.call_args.args[0])

    def test_voice_on_both_channels_plus_the_effect_until_it_ends(self):
        mic, _, _ = self.open()
        effect = np.full((5, 2), 0.5, np.float32)
        with patch.object(call_audio, "decode_sound", return_value=effect) as decode:
            mic.prepare(Path("vanish.MP3"))
            mic.play(Path("vanish.MP3"))
        decode.assert_called_once_with(Path("vanish.MP3"), 44100)
        voice = np.full((4, 1), 0.25, np.float32)
        out = np.zeros((4, 2), np.float32)
        mic._mix(voice, out, 4, None, None)
        np.testing.assert_allclose(out, 0.25 + 0.5 * EFFECT_GAIN)
        mic._mix(voice, out, 4, None, None)
        np.testing.assert_allclose(out[0], 0.25 + 0.5 * EFFECT_GAIN)
        np.testing.assert_allclose(out[1:], 0.25)
        self.assertEqual(mic.playing, [])

    def test_a_sound_that_cannot_be_decoded_only_misses_the_call(self):
        mic, _, _ = self.open()
        with patch.object(call_audio, "decode_sound", side_effect=RuntimeError("no mixer")), \
                patch("builtins.print") as printed:
            mic.play(Path("teleport.MP3"))
        self.assertIn("teleport.MP3", printed.call_args.args[0])
        self.assertEqual(mic.playing, [])

    def test_platform_picks_the_microphone(self):
        with patch.object(call_audio.sys, "platform", "win32"), \
                patch.object(call_audio, "CableMic", return_value="cable"):
            self.assertEqual(open_virtual_mic(), "cable")
        with patch.object(call_audio.sys, "platform", "linux"), \
                patch.object(call_audio, "PipeWireMic", return_value="pipewire"):
            self.assertEqual(open_virtual_mic(), "pipewire")


class ResampleTests(unittest.TestCase):
    def test_keeps_length_proportional_and_values_interpolated(self):
        samples = np.array([[0.0, 0.0], [1.0, -1.0]], np.float32).repeat(441, axis=0)
        resampled = resample(samples, 44100, 48000)
        self.assertEqual(len(resampled), 960)
        self.assertAlmostEqual(float(resampled[0, 0]), 0.0)
        self.assertAlmostEqual(float(resampled[-1, 1]), -1.0)
        self.assertIs(resample(samples, 48000, 48000), samples)


class CallSoundTests(unittest.TestCase):
    def test_effect_sound_goes_to_speakers_and_call(self):
        local, mic = Mock(), Mock()
        sound = CallSound(local, mic, Path("vanish.MP3"))
        mic.prepare.assert_called_once_with(Path("vanish.MP3"))
        sound.play()
        local.play.assert_called_once()
        mic.play.assert_called_once_with(Path("vanish.MP3"))
        sound.close()
        local.close.assert_called_once()

    def test_unavailable_mic_never_stops_the_video(self):
        with ExitStack() as resources, patch.object(app, "open_virtual_mic", side_effect=RuntimeError("no PipeWire")), \
                patch("builtins.print") as printed:
            self.assertIsNone(app.open_call_mic(arguments(call_audio=True), resources))
        self.assertIn("no PipeWire", printed.call_args.args[0])

    def test_naruto_sound_is_shared_with_the_call(self):
        mic = Mock()
        with patch.object(app, "VanishSound", return_value=Mock()):
            sound = app.open_sound("naruto", mic)
        self.assertIsInstance(sound, CallSound)
        self.assertIsNone(app.open_sound("normal", mic))

    def test_missing_optional_sound_runs_silent(self):
        with patch.object(GokuMode, "sound_file", return_value=Path("nowhere/teleport.MP3")), \
                patch("builtins.print") as printed:
            self.assertIsNone(app.open_sound("goku", None))
        self.assertIn("teleport.MP3", printed.call_args.args[0])
        self.assertTrue(printed.call_args.args[0].startswith("[Goku]"))

    def test_missing_required_sound_is_an_error(self):
        with patch.object(NarutoMode, "sound_file", return_value=Path("nowhere/vanish.MP3")):
            with self.assertRaises(RuntimeError):
                app.open_sound("naruto", None)

    def test_self_test_requires_every_declared_file(self):
        with patch.object(GokuMode, "sound_file", return_value=Path("nowhere/teleport.MP3")):
            with self.assertRaisesRegex(RuntimeError, "teleport.MP3"):
                app.self_test()


if __name__ == "__main__":
    unittest.main()
