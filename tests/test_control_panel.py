import json
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from vanishcam import control_panel
from vanishcam import app
from vanishcam import detectors, segmentation
from vanishcam.camera import CameraInfo
from vanishcam.control_panel import (
    ADD_TILE, AMBER, MODE_CARDS, STAGE, WIDTH, ControlPanel, PanelRenderer, PanelView, fit_into, hit, panel_height,
    tiles,
)
from .test_video_output import InlineRecognizer, app_mocks, arguments, video_clock
from vanishcam.i18n import t
from vanishcam.user_data import MAX_GALLERY, Settings, UserData


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
        self.assertTrue(source.is_file())  # The original stays where it was.
        self.assertTrue(all(path.suffix == ".png" for path in gallery))
        self.assertEqual(self.data.import_image(gallery[0]), gallery[0])  # Already in the gallery.
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

    def regions(self, view, hover=None, menu=None):
        image, regions = self.renderer.render(view, hover, menu)
        self.assertEqual(image.shape, (panel_height(view), WIDTH, 3))
        return regions

    def actions(self, regions):
        return [region.action for region in regions]

    def test_effect_menu_lists_every_effect_and_covers_the_rest(self):
        view = PanelView("naruto", "Pronto", gallery=self.gallery)
        closed = self.actions(self.regions(view))
        self.assertIn(("menu", "mode"), closed)
        self.assertFalse(any(action[0] == "mode" for action in closed))
        opened = self.actions(self.regions(view, menu="mode"))
        self.assertEqual([a[1] for a in opened if a[0] == "mode"], [card.key for card in MODE_CARDS])
        # Under the open menu nothing else reacts; a click elsewhere only closes it.
        self.assertNotIn(("background", None), opened)
        self.assertNotIn(("vanish",), opened)
        self.assertEqual(opened[-1], ("menu", None))

    def test_camera_section_and_its_menu(self):
        cameras = (CameraInfo(0, "USB 2.0 PC Cam"), CameraInfo(2, "Logitech C920"))
        view = PanelView("normal", "Pronto", cameras=cameras, camera_index=2, can_adjust=True)
        self.assertEqual(view.camera_name, "Logitech C920")
        self.assertEqual(PanelView("normal", "", camera_index=3).camera_name, t("panel.camera_number", index=3))
        closed = self.actions(self.regions(view))
        for action in (("menu", "camera"), ("reopen_camera",), ("camera_settings",)):
            self.assertIn(action, closed)
        opened = self.actions(self.regions(view, menu="camera"))
        self.assertEqual([a for a in opened if a[0] == "camera"], [("camera", 0), ("camera", 2)])
        # Without a list (it could not be read) the menu still shows the webcam in use.
        alone = self.actions(self.regions(PanelView("normal", "", camera_index=1), menu="camera"))
        self.assertEqual([a for a in alone if a[0] == "camera"], [("camera", 1)])

    def test_camera_settings_only_when_available_and_live(self):
        for view in (PanelView("normal", "", can_adjust=False),
                     PanelView("normal", "", can_adjust=True, camera_state="opening")):
            self.assertNotIn(("camera_settings",), self.actions(self.regions(view)))

    def test_gallery_tiles_and_selection(self):
        view = PanelView("normal", "Pronto", background=self.gallery[1], gallery=self.gallery)
        self.assertEqual(tiles(view), [None, *self.gallery, ADD_TILE])
        actions = self.actions(self.regions(view))
        self.assertIn(("background", None), actions)
        self.assertIn(("background", self.gallery[0]), actions)
        self.assertNotIn(("background", self.gallery[1]), actions)  # Already chosen.
        self.assertIn(("add",), actions)
        # A picture from the command line, outside the gallery, shows up anyway.
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
        broken = self.folder / "broken.png"
        broken.write_bytes(b"nada")
        self.regions(PanelView("normal", "Pronto", gallery=(broken,)))


class FitIntoTests(unittest.TestCase):
    def test_whole_video_centered_without_distortion(self):
        video = np.full((90, 160, 3), 200, np.uint8)
        stage = fit_into(video, 400, 300)
        self.assertEqual(stage.shape, (300, 400, 3))
        # 16:9 in 4:3: full width, bars above and below.
        self.assertTrue((stage[150, :] == 200).all())
        self.assertTrue((stage[0, :] == STAGE).all())
        self.assertTrue((stage[-1, :] == STAGE).all())
        self.assertTrue((fit_into(None, 40, 30) == STAGE).all())


class ControlPanelWindowTests(unittest.TestCase):
    SIZE = (WIDTH + 400, 760)

    def setUp(self):
        mocks = self.enterContext(ExitStack())
        for name in ("namedWindow", "setMouseCallback", "destroyWindow", "resizeWindow", "setWindowProperty"):
            mocks.enter_context(patch.object(control_panel.cv2, name))
        mocks.enter_context(patch.object(control_panel.cv2, "getWindowImageRect", return_value=(0, 0, *self.SIZE)))
        self.imshow = mocks.enter_context(patch.object(control_panel.cv2, "imshow"))
        self.panel = ControlPanel()
        self.render = mocks.enter_context(patch.object(self.panel.renderer, "render", wraps=self.panel.renderer.render))
        self.video = np.full((90, 160, 3), 200, np.uint8)

    def click(self, action):
        region = next(r for r in self.panel.regions if r.action == action)
        x = (region.box[0] + region.box[2]) / 2
        y = (region.box[1] + region.box[3]) / 2
        self.panel._on_mouse(cv2.EVENT_LBUTTONUP, x, y, 0, None)

    def test_panel_redraws_only_on_change_while_the_video_updates_every_frame(self):
        view = PanelView("normal", "Pronto")
        self.panel.show(view, self.video)
        self.panel.show(view, self.video)
        self.assertEqual(self.render.call_count, 1)
        self.assertEqual(self.imshow.call_count, 2)
        self.panel.show(PanelView("goku", "Pronto"), self.video)
        self.assertEqual(self.render.call_count, 2)

    def test_one_window_with_the_panel_beside_the_video(self):
        view = PanelView("normal", "Pronto")
        self.panel.show(view, self.video)
        # The first frame sets the window's starting size; then each frame follows the window.
        first = self.imshow.call_args.args[1]
        self.assertEqual(first.shape, (panel_height(view), WIDTH + control_panel.VIDEO_WIDTH, 3))
        self.panel.show(view, self.video)
        shown = self.imshow.call_args.args[1]
        self.assertEqual(shown.shape, (self.SIZE[1], self.SIZE[0], 3))
        panel, _ = PanelRenderer().render(view)
        np.testing.assert_array_equal(shown[: panel.shape[0], :WIDTH], panel)
        self.assertTrue((shown[self.SIZE[1] // 2, WIDTH + 10 : -10] == 200).all())

    def test_effect_menu_opens_then_picks(self):
        self.panel.show(PanelView("normal", "Pronto"), self.video)
        self.click(("menu", "mode"))
        self.assertEqual(self.panel.poll(), [])  # Opening the menu is not an action.
        self.assertEqual(self.panel.menu, "mode")
        self.panel.show(PanelView("normal", "Pronto"), self.video)
        self.click(("mode", "goku"))
        self.assertEqual(self.panel.poll(), [("mode", "goku")])
        self.assertIsNone(self.panel.menu)

    def test_click_outside_closes_the_menu_without_acting(self):
        self.panel.show(PanelView("normal", "Pronto"), self.video)
        self.click(("menu", "mode"))
        self.panel.poll()
        self.panel.show(PanelView("normal", "Pronto"), self.video)
        self.panel._on_mouse(cv2.EVENT_LBUTTONUP, WIDTH - 5, panel_height(PanelView("normal", "")) - 5, 0, None)
        self.assertEqual(self.panel.poll(), [])
        self.assertIsNone(self.panel.menu)

    def test_camera_menu_asks_for_a_fresh_list(self):
        self.panel.show(PanelView("normal", "Pronto"), self.video)
        self.click(("menu", "camera"))
        self.assertEqual(self.panel.poll(), [("cameras",)])

    def test_fullscreen_shows_only_the_video_and_ignores_clicks(self):
        self.panel.show(PanelView("normal", "Pronto"), self.video)
        self.panel.set_fullscreen(True)
        self.panel.show(PanelView("normal", "Pronto", fullscreen=True), self.video)
        shown = self.imshow.call_args.args[1]
        self.assertTrue((shown[self.SIZE[1] // 2, 10:-10] == 200).all())  # No panel on the left.
        self.click(("menu", "mode"))
        self.assertEqual(self.panel.poll(), [])

    def pick(self, **picker):
        self.panel.show(PanelView("normal", "Pronto"), self.video)
        with patch.object(control_panel, "pick_image_file", **picker) as dialog:
            self.click(("add",))
            self.click(("add",))  # Double click: a single dialog.
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

    def run_app(self, actions=(), frames=6, args=None, setup=None, keys=None, cameras=()):
        capture = Mock()
        images = [np.full((24, 40, 3), 100 + 10 * i, np.uint8) for i in range(frames)]
        read, clock = video_clock(images, (0.0, 4.1) + tuple(4.3 + 0.2 * i for i in range(frames - 2)))
        capture.read.side_effect = read
        mask = np.zeros((24, 40), np.float32)
        mask[:, :20] = 1.0  # Person on the left.
        recognizers = [Mock() for _ in range(3)]
        for recognizer in recognizers:
            recognizer.recognize_joined_hands.return_value = False
            recognizer.recognize_pose.return_value = False
        displayed = []
        keys = list(keys or [0] * (frames - 1)) + [ord("q")]
        with ExitStack() as mocks:
            panel, user_data = app_mocks(mocks, Mock(), None, actions, cameras)
            self.creating = mocks.enter_context(patch.object(app, "create_recognizer", side_effect=recognizers))
            if setup is not None:
                setup(user_data)
            mocks.enter_context(patch.object(app, "parse_args", return_value=args or arguments(virtual_camera=False)))
            mocks.enter_context(patch.object(app, "download_models"))
            opening = mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "VanishSound", return_value=Mock()))
            mocks.enter_context(patch.object(segmentation, "person_mask", return_value=mask))
            mocks.enter_context(patch.object(detectors, "recognize_gesture", return_value=("No_Hand", 0.0)))
            mocks.enter_context(patch.object(app, "AsyncRecognizer", InlineRecognizer))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=clock))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
                mocks.enter_context(patch.object(app.cv2, name))
            displayed = panel.frames
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
            # The temporary folder disappears on exit: read what was saved right away.
            self.saved, self.final_gallery = user_data.load(), user_data.gallery()
            # Webcam and segmentation stay open across the switches.
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
        recognizers[1].close.assert_called_once()  # Closed on exit.
        self.assertEqual(self.creating.call_args_list[1].args, ("naruto",))
        self.assertEqual(self.saved.mode, "naruto")
        self.assertEqual(len(panel.real_backgrounds), 1)  # Calibrated only once.

    def test_number_keys_switch_mode(self):
        _, _, panel, user_data, _ = self.run_app(keys=[0, 0, ord("3"), 0, ord("4")])
        self.assertEqual([view.mode for view in panel.views], ["normal"] * 3 + ["goku"] * 2 + ["particles"])
        self.assertFalse(panel.views[3].manual_keys)  # Goku: no V and R.
        self.assertEqual(self.saved.mode, "particles")

    def test_failed_mode_switch_keeps_the_previous_mode(self):
        # The new mode fails to open: back to the previous one, with a message.
        original = app.EffectRuntime.__init__

        def flaky(runtime, name, segment, mic):
            if name == "naruto":
                raise RuntimeError("broken model")
            original(runtime, name, segment, mic)

        with patch.object(app.EffectRuntime, "__init__", flaky):
            _, _, panel, user_data, _ = self.run_app([[], [], [("mode", "naruto")]])
        self.assertEqual({view.mode for view in panel.views}, {"normal"})
        self.assertIn("broken model", panel.views[3].message)
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
        self.assertNotEqual(panel.views[0].status_color, AMBER)  # No calibration.
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
        self.assertEqual(panel.views[3].status_color, AMBER)  # Never calibrated: calibrate now.

    def test_bad_import_shows_a_message(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        broken = folder / "broken.png"
        broken.write_bytes(b"nada")
        _, _, panel, user_data, _ = self.run_app([[], [], [("import", broken)]], frames=4)
        self.assertTrue(panel.views[3].message.startswith(t("loop.image_not_added", error="")))
        self.assertEqual(self.final_gallery, [])

    def test_choosing_a_camera_reopens_it_and_remembers_it_by_name(self):
        cameras = (CameraInfo(0, "USB 2.0 PC Cam"), CameraInfo(2, "Logitech C920"))
        with patch.object(app.CameraFeed, "reopen") as reopen:
            _, _, panel, _, _ = self.run_app([[], [("cameras",)], [("camera", 2)]], cameras=cameras)
        reopen.assert_called_once()
        self.assertEqual(self.saved.camera, "Logitech C920")
        self.assertEqual(panel.views[-1].camera_index, 2)
        self.assertEqual(panel.views[-1].cameras, cameras)
        self.assertEqual(panel.views[-1].message, t("camera.switched", name="Logitech C920"))

    def test_reopen_and_settings_buttons(self):
        with patch.object(app.CameraFeed, "reopen") as reopen, \
                patch.object(app.CameraFeed, "open_settings", return_value=False):
            _, _, panel, _, _ = self.run_app([[], [("reopen_camera",)], [("camera_settings",)]], frames=4)
        reopen.assert_called_once_with()
        self.assertEqual(panel.views[-1].message, t("camera.settings_unavailable"))

    def test_escape_leaves_fullscreen_instead_of_quitting(self):
        _, _, panel, _, _ = self.run_app([[("fullscreen",)]], frames=4, keys=[0, 27, 0])
        self.assertEqual(panel.fullscreen, [True, False])
        self.assertEqual(len(panel.views), 4)  # Still running after Esc, until Q.

    def test_settings_remember_the_camera(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        data = UserData(folder)
        data.save(Settings("goku", None, "Logitech C920"))
        self.assertEqual(data.load(), Settings("goku", None, "Logitech C920"))


if __name__ == "__main__":
    unittest.main()
