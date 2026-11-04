"""Goku's gesture hold time: the choices, the saved preference, the SETTINGS stepper and the mode."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vanishcam import app

from vanishcam.control_panel import PanelRenderer, PanelView
from vanishcam.effects.goku import HOLD_CHOICES, HOLD_SECONDS, step_hold, valid_hold
from vanishcam.i18n import t
from vanishcam.modes import GokuMode
from vanishcam.user_data import Settings, UserData
from . import test_quick_start
from .test_video_output import arguments


class HoldChoiceTests(unittest.TestCase):
    def test_steps_walk_the_choices_and_stop_at_the_ends(self):
        self.assertEqual(step_hold(1.0, 1), 1.5)
        self.assertEqual(step_hold(1.0, -1), 0.75)
        self.assertIsNone(step_hold(HOLD_CHOICES[-1], 1))
        self.assertIsNone(step_hold(HOLD_CHOICES[0], -1))
        # A value between two choices steps to the next one either way.
        self.assertEqual(step_hold(0.6, 1), 0.75)
        self.assertEqual(step_hold(0.6, -1), 0.5)

    def test_only_usable_times_are_accepted(self):
        self.assertEqual(valid_hold(0.5), 0.5)
        self.assertEqual(valid_hold(2), 2.0)
        for bad in (0, -1, 0.01, 10, float("nan"), "1", None, True):
            self.assertIsNone(valid_hold(bad))


class HoldSettingTests(unittest.TestCase):
    def test_saved_with_the_preferences_and_bad_values_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            data = UserData(Path(folder))
            self.assertEqual(data.load().goku_hold, HOLD_SECONDS)
            data.save(Settings(goku_hold=0.25, quick_start=True))
            loaded = data.load()
            self.assertEqual((loaded.goku_hold, loaded.quick_start), (0.25, True))
            data.settings_file.write_text('{"goku_hold": 99}', encoding="utf-8")
            self.assertEqual(data.load().goku_hold, HOLD_SECONDS)

    def test_the_time_shows_under_the_effect_only_for_goku(self):
        renderer = PanelRenderer()
        image, regions = renderer.render(PanelView("goku", "", goku_hold=0.5))
        actions = [region.action for region in regions]
        self.assertIn(("goku_hold", 0.25), actions)
        self.assertIn(("goku_hold", 0.75), actions)
        self.assertIn("0.5", t("panel.goku_hold", seconds="0.5"))
        # Right under the effect selector, above the backgrounds.
        menu = next(r for r in regions if r.action == ("menu", "mode"))
        step = next(r for r in regions if r.action == ("goku_hold", 0.75))
        background = next(r for r in regions if r.action == ("add",))
        self.assertLess(menu.box[3], step.box[1])
        self.assertLess(step.box[3], background.box[1])
        # At the longest time there is no + to click.
        _, regions = renderer.render(PanelView("goku", "", goku_hold=HOLD_CHOICES[-1]))
        holds = [region.action for region in regions if region.action[0] == "goku_hold"]
        self.assertEqual(holds, [("goku_hold", 2.0)])
        # Not with the other effects, nor on the other tabs; the panel keeps its height.
        for view, tab in ((PanelView("normal", ""), "live"), (PanelView("goku", ""), "settings")):
            other, regions = renderer.render(view, tab=tab)
            self.assertFalse(any(region.action[0] == "goku_hold" for region in regions))
            self.assertEqual(other.shape, image.shape)


class GokuModeHoldTests(unittest.TestCase):
    def test_the_mode_uses_the_configured_time_to_vanish_and_after_reset(self):
        mode = GokuMode(Mock())
        mode.configure(Settings(goku_hold=0.25))
        mode.on_detection(True, 10.0)
        mode.on_detection(True, 10.2)
        self.assertEqual(mode.phase, "ready")
        mode.on_detection(True, 10.26)
        self.assertEqual(mode.phase, "vanishing")
        mode.reset()
        self.assertEqual(mode.hold.hold_seconds, 0.25)

    def test_a_longer_time_needs_a_longer_hold(self):
        mode = GokuMode(Mock())
        mode.configure(Settings(goku_hold=2.0))
        for now in (0.0, 0.5, 1.0, 1.5, 1.9):
            mode.on_detection(True, now)
        self.assertEqual(mode.phase, "ready")
        mode.on_detection(True, 2.0)
        self.assertEqual(mode.phase, "vanishing")

    def test_other_effects_ignore_the_setting(self):
        from vanishcam.modes import NarutoMode
        mode = NarutoMode.__new__(NarutoMode)
        mode.configure(Settings(goku_hold=0.1))  # No error, nothing to take.


class GokuHoldLoopTests(unittest.TestCase):
    def test_the_stepper_changes_the_running_effect_and_is_saved(self):
        configured = []
        original = GokuMode.configure

        def spy(mode, settings):
            configured.append(settings.goku_hold)
            return original(mode, settings)

        loop = test_quick_start.QuickStartLoopTests("test_quick_start_on_the_command_line")
        args = arguments(virtual_camera=False, width=200, height=120, mode="goku")
        with patch.object(GokuMode, "configure", spy), patch.object(app, "open_sound", return_value=None):
            _, _, _, saved = loop.run_loop(
                (0, 0.5), (0, ord("q")), lambda ms: test_quick_start.person_at(80, 120), actions=([("goku_hold", 0.5)],), args=args)
        self.assertEqual(saved.goku_hold, 0.5)
        # Configured at start with the saved time, then again with the new one.
        self.assertEqual(configured, [HOLD_SECONDS, 0.5])

if __name__ == "__main__":
    unittest.main()
