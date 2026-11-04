"""Starting without capturing the room: the guessed background, learning it, the panel and the loop."""

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from vanishcam import app, detectors, segmentation
from vanishcam.background import covered_area, guess_background, learn_unseen, paint_unseen
from vanishcam.control_panel import PanelRenderer, PanelView, panel_height
from vanishcam.i18n import t
from vanishcam.user_data import Settings, UserData
from .test_video_output import TEST_SHORTCUTS, InlineRecognizer, app_mocks, arguments, video_clock

HEIGHT, WIDTH = 120, 200


def room():
    """A room with a smooth gradient, so a painted guess can be close to it."""
    x = np.linspace(80, 200, WIDTH, dtype=np.float32)
    gradient = np.repeat(x[None, :], HEIGHT, axis=0)
    return np.dstack([gradient, gradient * 0.9, gradient * 0.8]).astype(np.uint8)


def person_at(left, right):
    mask = np.zeros((HEIGHT, WIDTH), np.float32)
    mask[20:, left:right] = 1.0
    return mask


def frame_with(mask, background):
    frame = background.copy()
    frame[mask > 0.5] = (30, 160, 40)  # A green person.
    return frame


class GuessedRoomTests(unittest.TestCase):
    def setUp(self):
        self.room = room()
        self.mask = person_at(80, 120)
        self.frame = frame_with(self.mask, self.room)

    def test_what_the_person_covers_is_painted_from_the_room(self):
        background, unseen = guess_background(self.frame, self.mask)
        # Unseen: the person and a margin around them; the rest is the frame as it is.
        self.assertTrue(unseen[self.mask > 0.5].all())
        self.assertFalse(unseen[:, :60].any())
        np.testing.assert_array_equal(background[~unseen], self.frame[~unseen])
        # No green person left behind: the hole is painted in room colors.
        painted = background[60:, 90:110].astype(int)
        self.assertLess(int(np.abs(painted - self.room[60:, 90:110].astype(int)).mean()), 25)
        self.assertGreater(int(painted[..., 2].mean()), int(painted[..., 1].mean()) - 30)

    def test_nothing_to_paint_leaves_the_background_alone(self):
        unseen = np.zeros((HEIGHT, WIDTH), bool)
        self.assertIs(paint_unseen(self.room, unseen), self.room)

    def test_what_the_camera_sees_replaces_the_guess_at_once(self):
        background, unseen = guess_background(self.frame, self.mask)
        # The person steps right: the left part of the old hole is seen as it is.
        moved = person_at(110, 150)
        frame = frame_with(moved, self.room)
        learned, still = learn_unseen(background, frame, moved, unseen)
        revealed = unseen & ~covered_area(moved)
        self.assertTrue(revealed.any())
        np.testing.assert_array_equal(learned[revealed], self.room[revealed])
        self.assertLess(np.count_nonzero(still), np.count_nonzero(unseen))
        # Gone from the frame: all of it is seen, and the room counts as learned.
        empty = np.zeros_like(self.mask)
        final, done = learn_unseen(learned, self.room, empty, still)
        self.assertIsNone(done)
        np.testing.assert_array_equal(final, self.room)


class QuickStartSettingsTests(unittest.TestCase):
    def test_quick_start_is_saved_with_the_other_preferences(self):
        with tempfile.TemporaryDirectory() as folder:
            data = UserData(Path(folder))
            self.assertFalse(data.load().quick_start)
            data.save(Settings(mode="goku", quick_start=True))
            loaded = data.load()
            self.assertTrue(loaded.quick_start)
            self.assertEqual(loaded.mode, "goku")

    def test_panel_offers_skip_while_capturing_and_the_switch_on_settings(self):
        renderer = PanelRenderer()
        _, regions = renderer.render(PanelView("normal", "", calibrating=True))
        actions = [region.action for region in regions]
        self.assertIn(("skip_calibration",), actions)
        self.assertNotIn(("recapture",), actions)
        _, regions = renderer.render(PanelView("normal", ""))
        self.assertIn(("recapture",), [region.action for region in regions])
        for on in (False, True):
            view = PanelView("normal", "", quick_start=on, shortcuts=TEST_SHORTCUTS)
            image, regions = renderer.render(view, tab="settings")
            self.assertIn(("quick_start", not on), [region.action for region in regions])
            # Every tab keeps the same height, the settings content included.
            self.assertEqual(image.shape[0], panel_height(PanelView("normal", "")))
        reset = next(r for r in regions if r.action == ("reset_shortcuts",))
        self.assertLess(reset.box[3], panel_height(PanelView("normal", "")) - 30)


class QuickStartLoopTests(unittest.TestCase):
    def run_loop(self, times, keys, masks, quick_start=False, actions=(), args=None):
        """Run the app on scripted frames; masks(timestamp_ms) says where the person is."""
        background = room()
        frames = []
        for at in times:
            frames.append(frame_with(masks(int(at * 1000)), background))
        capture = Mock()
        read, clock = video_clock(frames, times)
        capture.read.side_effect = read
        printed = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=args or arguments(
                virtual_camera=False, width=WIDTH, height=HEIGHT, quick_start=False)))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            panel, user_data = app_mocks(mocks, Mock(), Mock(), actions=actions)
            user_data.save(Settings(shortcuts=TEST_SHORTCUTS, quick_start=quick_start))
            mocks.enter_context(patch.object(
                segmentation, "person_mask", side_effect=lambda segmenter, image, ms, w, h: masks(ms)))
            mocks.enter_context(patch.object(detectors, "recognize_gesture", return_value=("No_Hand", 0.0)))
            mocks.enter_context(patch.object(app, "AsyncRecognizer", InlineRecognizer))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=clock))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "waitKeyEx", side_effect=keys))
            mocks.enter_context(patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(map(str, a)))))
            self.assertEqual(app.run(), 0)
            # Read before the temporary folder goes away.
            saved = user_data.load()
        return panel, background, printed, saved

    def test_quick_start_runs_from_the_first_frame_and_vanishes_into_the_guess(self):
        panel, room_image, printed, _ = self.run_loop(
            (0, 0.1, 0.2, 4.5), (0, ord("v"), 0, ord("q")), lambda ms: person_at(80, 120), quick_start=True)
        self.assertFalse(panel.views[0].calibrating)
        self.assertEqual(len(panel.real_backgrounds), 1)
        self.assertTrue(any(t("loop.room_guessed") in line for line in printed))
        # Gone: where the person stood shows the painted room, not the green person.
        gone = panel.frames[-1].astype(int)
        self.assertLess(int(np.abs(gone[60:, 90:110] - room_image[60:, 90:110].astype(int)).mean()), 25)

    def test_the_room_is_learned_once_the_person_has_moved_away(self):
        def masks(ms):
            return person_at(80, 120) if ms < 1000 else person_at(150, 190) if ms < 2000 else np.zeros((HEIGHT, WIDTH), np.float32)

        panel, room_image, printed, _ = self.run_loop(
            (0, 0.6, 1.2, 1.8, 2.4, 3.0), (0, 0, 0, 0, 0, ord("q")), masks, quick_start=True)
        self.assertTrue(any(t("loop.room_learned") in line for line in printed))
        # The panel's REAL thumbnail gets the learned room.
        self.assertEqual(len(panel.real_backgrounds), 2)
        np.testing.assert_array_equal(panel.real_backgrounds[-1][:, 60:140], room_image[:, 60:140])

    def test_skip_during_the_capture_starts_right_away(self):
        panel, _, printed, _ = self.run_loop(
            (0, 0.5, 1.0), (0, 0, ord("q")), lambda ms: person_at(80, 120),
            actions=([("skip_calibration",)],))
        self.assertTrue(panel.views[0].calibrating)
        self.assertFalse(panel.views[-1].calibrating)
        self.assertTrue(any(t("loop.room_guessed") in line for line in printed))

    def test_the_settings_switch_is_saved(self):
        _, _, _, saved = self.run_loop(
            (0, 0.5), (0, ord("q")), lambda ms: person_at(80, 120), actions=([("quick_start", True)],))
        self.assertTrue(saved.quick_start)

    def test_quick_start_on_the_command_line(self):
        args = arguments(virtual_camera=False, width=WIDTH, height=HEIGHT, quick_start=True)
        panel, _, printed, saved = self.run_loop(
            (0, 0.5), (0, ord("q")), lambda ms: person_at(80, 120), args=args)
        self.assertFalse(panel.views[0].calibrating)
        # For this run only: the saved preference stays off.
        self.assertFalse(saved.quick_start)


if __name__ == "__main__":
    unittest.main()
