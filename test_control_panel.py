import json
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import control_panel
import vanish_cam as app
from control_panel import (
    ADD_TILE, AMBER, MODE_CARDS, WIDTH, ControlPanel, PanelRenderer, PanelView, hit, panel_height, tiles,
)
from test_video_output import InlineRecognizer, app_mocks, arguments, video_clock
from user_data import MAX_GALLERY, Settings, UserData


def write_image(path, value=30, size=(48, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((*size, 3), value, np.uint8))
    return path


class UserDataTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.data = UserData(self.folder / "config")

    def test_missing_or_corrupt_settings_fall_back_to_defaults(self):
        self.assertEqual(self.data.load(), Settings())
        self.data.root.mkdir()
        self.data.settings_file.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.data.load(), Settings())
        self.data.settings_file.write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(self.data.load(), Settings())

    def test_round_trip_and_deleted_image_returns_to_real_background(self):
        image = write_image(self.folder / "praia.png")
        self.data.save(Settings("goku", image))
        self.assertEqual(self.data.load(), Settings("goku", image))
        image.unlink()
        self.assertEqual(self.data.load(), Settings("goku", None))
        self.assertEqual(json.loads(self.data.settings_file.read_text())["mode"], "goku")

    def test_import_copies_newest_first_and_keeps_at_most_max_gallery(self):
        source = write_image(self.folder / "minha praia.PNG")
        copies = []
        for _ in range(MAX_GALLERY + 2):
            copies.append(self.data.import_image(source))
            time.sleep(0.002)
        gallery = self.data.gallery()
        self.assertEqual(len(gallery), MAX_GALLERY)
        self.assertEqual(gallery[0], copies[-1])
        self.assertTrue(source.is_file())  # L'originale resta dov'era.
        self.assertTrue(all(path.suffix == ".png" for path in gallery))
        self.assertEqual(self.data.import_image(gallery[0]), gallery[0])  # Gia' in galleria.
        with self.assertRaises(ValueError):
            self.data.import_image(self.folder / "notas.txt")

    def test_remove_only_touches_gallery_copies(self):
        source = write_image(self.folder / "praia.png")
        copy = self.data.import_image(source)
        self.data.remove_image(source)
        self.assertTrue(source.is_file())
        self.data.remove_image(copy)
        self.assertEqual(self.data.gallery(), [])


class PanelRendererTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.gallery = tuple(write_image(self.folder / f"bg{i}.png", 40 * i) for i in range(3))
        self.renderer = PanelRenderer()

    def regions(self, view, hover=None):
        image, regions = self.renderer.render(view, hover)
        self.assertEqual(image.shape, (panel_height(view), WIDTH, 3))
        return regions

    def actions(self, regions):
        return [region.action for region in regions]

    def test_mode_cards_switch_to_other_modes_only(self):
        actions = self.actions(self.regions(PanelView("naruto", "Pronto")))
        modes = [action[1] for action in actions if action[0] == "mode"]
        self.assertEqual(modes, [card.key for card in MODE_CARDS if card.key != "naruto"])

    def test_gallery_tiles_and_selection(self):
        view = PanelView("normal", "Pronto", background=self.gallery[1], gallery=self.gallery)
        self.assertEqual(tiles(view), [None, *self.gallery, ADD_TILE])
        actions = self.actions(self.regions(view))
        self.assertIn(("background", None), actions)
        self.assertIn(("background", self.gallery[0]), actions)
        self.assertNotIn(("background", self.gallery[1]), actions)  # Gia' scelto.
        self.assertIn(("add",), actions)
        # Un'immagine da riga di comando, fuori dalla galleria, compare comunque.
        outside = self.folder / "fora.png"
        self.assertEqual(tiles(PanelView("normal", "", background=outside))[:2], [None, outside])

    def test_remove_button_appears_only_over_a_gallery_tile(self):
        view = PanelView("normal", "Pronto", gallery=self.gallery)
        self.assertFalse(any(a[0] == "remove" for a in self.actions(self.regions(view))))
        tile = next(r for r in self.regions(view) if r.action == ("background", self.gallery[0]))
        center = ((tile.box[0] + tile.box[2]) / 2, (tile.box[1] + tile.box[3]) / 2)
        regions = self.regions(view, hover=center)
        x0, y0, x1, _ = tile.box
        self.assertEqual(hit(regions, x1 - 14, y0 + 14).action, ("remove", self.gallery[0]))
        self.assertEqual(hit(regions, x0 + 10, y0 + 30).action, ("background", self.gallery[0]))

    def test_disabled_controls_have_no_click_area(self):
        actions = self.actions(self.regions(PanelView("goku", "Pronto", manual_keys=False, can_recapture=False)))
        for action in (("vanish",), ("reappear",), ("recapture",)):
            self.assertNotIn(action, actions)
        self.assertIn(("fullscreen",), actions)
        enabled = self.actions(self.regions(PanelView("normal", "Pronto")))
        for action in (("vanish",), ("reappear",), ("recapture",), ("fullscreen",)):
            self.assertIn(action, enabled)

    def test_unreadable_gallery_image_still_renders(self):
        broken = self.folder / "quebrada.png"
        broken.write_bytes(b"nada")
        self.regions(PanelView("normal", "Pronto", gallery=(broken,)))


class ControlPanelWindowTests(unittest.TestCase):
    def setUp(self):
        mocks = self.enterContext(ExitStack())
        for name in ("namedWindow", "setMouseCallback", "destroyWindow"):
            mocks.enter_context(patch.object(control_panel.cv2, name))
        self.imshow = mocks.enter_context(patch.object(control_panel.cv2, "imshow"))
        self.panel = ControlPanel()

    def click(self, action):
        region = next(r for r in self.panel.regions if r.action == action)
        x = (region.box[0] + region.box[2]) / 2
        y = (region.box[1] + region.box[3]) / 2
        self.panel._on_mouse(cv2.EVENT_LBUTTONUP, x, y, 0, None)

    def test_redraws_only_on_change_and_turns_clicks_into_actions(self):
        view = PanelView("normal", "Pronto")
        self.panel.show(view)
        self.panel.show(view)
        self.assertEqual(self.imshow.call_count, 1)
        self.click(("mode", "goku"))
        self.panel._on_mouse(cv2.EVENT_LBUTTONUP, 2, 2, 0, None)  # Fuori dai pulsanti.
        self.assertEqual(self.panel.poll(), [("mode", "goku")])
        self.assertEqual(self.panel.poll(), [])
        self.panel.show(PanelView("goku", "Pronto"))
        self.assertEqual(self.imshow.call_count, 2)

    def pick(self, **picker):
        self.panel.show(PanelView("normal", "Pronto"))
        with patch.object(control_panel, "pick_image_file", **picker) as dialog:
            self.click(("add",))
            self.click(("add",))  # Doppio clic: una sola finestra.
            actions = self.panel.poll()
            for _ in range(200):
                if not self.panel.picking:
                    break
                time.sleep(0.01)
        dialog.assert_called_once()
        return actions + self.panel.poll()

    def test_add_opens_picker_off_the_video_thread(self):
        chosen = Path("/tmp/praia.jpg")
        self.assertEqual(self.pick(return_value=chosen), [("import", chosen)])

    def test_cancelled_picker_does_nothing(self):
        self.assertEqual(self.pick(return_value=None), [])

    def test_picker_errors_become_a_message(self):
        self.assertEqual(self.pick(side_effect=RuntimeError("sem zenity")), [("error", "sem zenity")])


class LiveSwitchTests(unittest.TestCase):
    """Effect and background change while camera, segmenter and virtual camera stay open."""

    def run_app(self, actions=(), frames=6, args=None, setup=None, keys=None):
        capture = Mock()
        images = [np.full((24, 40, 3), 100 + 10 * i, np.uint8) for i in range(frames)]
        read, clock = video_clock(images, (0.0, 4.1) + tuple(4.3 + 0.2 * i for i in range(frames - 2)))
        capture.read.side_effect = read
        mask = np.zeros((24, 40), np.float32)
        mask[:, :20] = 1.0  # Persona a sinistra.
        recognizers = [Mock() for _ in range(3)]
        for recognizer in recognizers:
            recognizer.recognize_joined_hands.return_value = False
            recognizer.recognize_pose.return_value = False
        displayed = []
        keys = list(keys or [0] * (frames - 1)) + [ord("q")]
        with ExitStack() as mocks:
            panel, user_data = app_mocks(mocks, Mock(), None, actions)
            self.creating = mocks.enter_context(patch.object(app, "create_recognizer", side_effect=recognizers))
            if setup is not None:
                setup(user_data)
            mocks.enter_context(patch.object(app, "parse_args", return_value=args or arguments(virtual_camera=False)))
            mocks.enter_context(patch.object(app, "download_models"))
            opening = mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "VanishSound", return_value=Mock()))
            mocks.enter_context(patch.object(app, "person_mask", return_value=mask))
            mocks.enter_context(patch.object(app, "recognize_gesture", return_value=("No_Hand", 0.0)))
            mocks.enter_context(patch.object(app, "AsyncRecognizer", InlineRecognizer))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=clock))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, f: displayed.append(f.copy())))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
            # La cartella temporanea sparisce all'uscita: leggi subito cosa e' stato salvato.
            self.saved, self.final_gallery = user_data.load(), user_data.gallery()
            # Webcam e segmentazione restano aperte durante i cambi.
            opening.assert_called_once()
            self.assertEqual(app.create_segmenter.call_count, 1)
        return images, displayed, panel, user_data, recognizers

    def gallery_image(self, user_data, value=30):
        return write_image(user_data.backgrounds_dir / "praia.png", value)

    def test_mode_switch_keeps_background_without_recalibrating(self):
        images, displayed, panel, user_data, recognizers = self.run_app([[], [], [("mode", "naruto")]])
        self.assertEqual([view.mode for view in panel.views], ["normal"] * 3 + ["naruto"] * 3)
        self.assertTrue(all(view.status_color != AMBER for view in panel.views[1:]))
        recognizers[0].close.assert_called_once()
        recognizers[1].close.assert_called_once()  # Chiuso all'uscita.
        self.assertEqual(self.creating.call_args_list[1].args, ("naruto",))
        self.assertEqual(self.saved.mode, "naruto")
        self.assertEqual(len(panel.real_backgrounds), 1)  # Calibrato una volta sola.

    def test_number_keys_switch_mode(self):
        _, _, panel, user_data, _ = self.run_app(keys=[0, 0, ord("3"), 0, ord("4")])
        self.assertEqual([view.mode for view in panel.views], ["normal"] * 3 + ["goku"] * 2 + ["particles"])
        self.assertFalse(panel.views[3].manual_keys)  # Goku: niente V e R.
        self.assertEqual(self.saved.mode, "particles")

    def test_failed_mode_switch_keeps_the_previous_mode(self):
        # Il nuovo modo non si apre: si torna a quello di prima, con un messaggio.
        original = app.EffectRuntime.__init__

        def flaky(runtime, name, segment, mic):
            if name == "naruto":
                raise RuntimeError("modelo quebrado")
            original(runtime, name, segment, mic)

        with patch.object(app.EffectRuntime, "__init__", flaky):
            _, _, panel, user_data, _ = self.run_app([[], [], [("mode", "naruto")]])
        self.assertEqual({view.mode for view in panel.views}, {"normal"})
        self.assertIn("modelo quebrado", panel.views[3].message)
        self.assertEqual(self.saved.mode, "normal")

    def test_image_from_gallery_replaces_room_and_is_remembered(self):
        images, displayed, panel, user_data, _ = self.run_app(
            [[], [], [lambda p: ("background", p.views[-1].gallery[0])]],
            setup=self.gallery_image,
        )
        image = panel.views[0].gallery[0]
        self.assertEqual(panel.views[3].background, image)
        self.assertFalse(panel.views[3].can_recapture)
        self.assertTrue((displayed[3][:, 20:] == 30).all())
        np.testing.assert_array_equal(displayed[3][:, :20], images[3][:, :20])
        self.assertEqual(self.saved.background, image)

    def test_back_to_real_background_reuses_the_captured_one(self):
        images, displayed, panel, user_data, _ = self.run_app(
            [[], [], [lambda p: ("background", p.views[-1].gallery[0])], [], [("background", None)]],
            setup=self.gallery_image, keys=[0, 0, 0, 0, ord("v")],
        )
        self.assertIsNone(panel.views[5].background)
        self.assertTrue(all(view.status_color != AMBER for view in panel.views[1:]))
        self.assertEqual(len(panel.real_backgrounds), 1)
        self.assertIsNone(self.saved.background)

    def test_startup_uses_saved_mode_and_background(self):
        def setup(user_data):
            user_data.save(Settings("particles", self.gallery_image(user_data)))

        images, displayed, panel, _, _ = self.run_app(
            frames=3, args=arguments(virtual_camera=False, mode=None), setup=setup,
        )
        self.assertEqual(panel.views[0].mode, "particles")
        self.assertIsNotNone(panel.views[0].background)
        self.assertNotEqual(panel.views[0].status_color, AMBER)  # Nessuna calibrazione.
        self.assertTrue((displayed[0][:, 20:] == 30).all())
        self.assertEqual(panel.real_backgrounds, [])

    def test_import_adds_to_gallery_and_removing_it_without_real_background_recalibrates(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        outside = write_image(folder / "minha praia.png", 30)
        images, displayed, panel, user_data, _ = self.run_app(
            [[("import", outside)], [], [lambda p: ("remove", p.views[-1].background)]], frames=4,
        )
        copy = panel.views[1].background
        self.assertEqual(copy.parent, user_data.backgrounds_dir)
        self.assertEqual(panel.views[1].gallery, (copy,))
        self.assertTrue((displayed[1][:, 20:] == 30).all())
        self.assertEqual(self.final_gallery, [])
        self.assertTrue(outside.is_file())
        self.assertEqual(panel.views[3].gallery, ())
        self.assertEqual(panel.views[3].status_color, AMBER)  # Mai calibrato: ora si calibra.

    def test_bad_import_shows_a_message(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        broken = folder / "quebrada.png"
        broken.write_bytes(b"nada")
        _, _, panel, user_data, _ = self.run_app([[], [], [("import", broken)]], frames=4)
        self.assertIn("Imagem não adicionada", panel.views[3].message)
        self.assertEqual(self.final_gallery, [])


if __name__ == "__main__":
    unittest.main()
