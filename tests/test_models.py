import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from vanishcam import models
from vanishcam.models import FACE_MODEL, GESTURE_MODEL, HAND_MODEL
from vanishcam.modes import MODES


class DownloadModelsTests(unittest.TestCase):
    def requested(self, declared):
        """URLs fetched by download_models(declared) into an empty folder."""
        with tempfile.TemporaryDirectory() as folder, ExitStack() as mocks:
            folder = Path(folder)
            segmenter, gesture = folder / "segmenter.tflite", folder / "gesture.task"
            mocks.enter_context(patch.object(models, "MODEL_DIR", folder))
            mocks.enter_context(patch.object(models, "SEGMENTER_MODEL", segmenter))
            mocks.enter_context(patch.object(models, "GESTURE_MODEL", gesture))
            mocks.enter_context(patch.object(models, "MODEL_URLS", {segmenter: "https://seg", gesture: "https://gesture"}))
            response = MagicMock()
            response.__enter__.return_value = response
            response.headers.get.return_value = 0
            response.read.side_effect = lambda size: b""
            opened = mocks.enter_context(patch.object(models.urllib.request, "urlopen", return_value=response))
            mocks.enter_context(patch("builtins.print"))
            # The declared paths are the real ones: map them onto the temporary files.
            renamed = {GESTURE_MODEL: gesture}
            models.download_models(tuple(renamed.get(path, path) for path in declared))
            return [call.args[0].full_url for call in opened.call_args_list]

    def test_segmenter_always_and_only_the_declared_models(self):
        self.assertEqual(self.requested(()), ["https://seg"])
        self.assertEqual(self.requested((GESTURE_MODEL,)), ["https://seg", "https://gesture"])
        self.assertEqual(self.requested((HAND_MODEL, FACE_MODEL)), ["https://seg"])

    def test_every_effect_declares_its_models(self):
        self.assertEqual(MODES["normal"].models, (GESTURE_MODEL,))
        self.assertEqual(MODES["naruto"].models, (HAND_MODEL,))
        for name in ("goku", "particles", "ash"):
            self.assertEqual(MODES[name].models, (HAND_MODEL, FACE_MODEL))


if __name__ == "__main__":
    unittest.main()
