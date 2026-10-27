import unittest
from unittest.mock import Mock

import numpy as np

import vanish_cam as app
from ash_effect import ASH_SECONDS, CRUMBLE_START, EFFECT_SECONDS, make_ash, render_ash, render_remains
from control_panel import MODE_CARDS
from modes import MODES, AshMode, FrameInfo

HEIGHT, WIDTH = 180, 320


def person():
    """A green person standing on the floor of a grey room."""
    frame = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
    mask = np.zeros((HEIGHT, WIDTH), np.float32)
    mask[40:HEIGHT, 130:190] = 1.0
    frame[mask > 0.5] = (30, 170, 40)
    background = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
    return frame, mask, background


def grey(pixels: np.ndarray) -> bool:
    """Ash: the three channels differ by a few levels at most."""
    pixels = pixels.astype(np.int16)
    return int(np.abs(pixels[..., 0] - pixels[..., 2]).max()) <= 12


class AshEffectTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.mask, self.background = person()
        self.cloud = make_ash(self.frame, self.mask)

    def test_grains_are_ash_coloured_and_the_head_crumbles_first(self):
        self.assertGreater(len(self.cloud.origin), 200)
        self.assertTrue(grey(self.cloud.color))
        head = self.cloud.origin[:, 1] < 70
        feet = self.cloud.origin[:, 1] > 150
        self.assertLess(self.cloud.delay[head].mean(), self.cloud.delay[feet].mean())
        self.assertTrue((self.cloud.delay >= CRUMBLE_START).all())

    def test_every_grain_lands_before_the_effect_ends(self):
        self.assertLessEqual(float(self.cloud.landed.max()), EFFECT_SECONDS)

    def test_person_first_then_a_grey_statue(self):
        start = render_ash(self.background, self.cloud, 0.0)
        np.testing.assert_array_equal(start[100, 160], self.frame[100, 160])
        statue = render_ash(self.background, self.cloud, ASH_SECONDS + 0.3)
        body = statue[self.mask > 0.5]
        self.assertTrue(grey(body))
        self.assertGreater(float(np.abs(body.astype(int) - 90).mean()), 10)  # Still visible over the room.

    def test_the_effect_ends_on_the_room_and_the_pile(self):
        last = render_ash(self.background, self.cloud, EFFECT_SECONDS - 1e-3).astype(int)
        remains = render_remains(self.background, self.cloud)
        self.assertLess(float(np.abs(last - remains).mean()), 1.0)  # No jump into the vanished frame.
        # The pile sits on the floor where the person stood; the rest is the room.
        np.testing.assert_array_equal(remains[:100], self.background[:100])
        self.assertTrue(np.any(remains[-10:, 130:190] != self.background[-10:, 130:190]))
        self.assertTrue(grey(remains[-5:, 150:170]))

    def test_empty_mask_does_not_fail(self):
        cloud = make_ash(self.frame, np.zeros_like(self.mask))
        np.testing.assert_array_equal(render_ash(self.background, cloud, 2.0), self.background)


class AshModeTests(unittest.TestCase):
    def test_registered_with_a_panel_row_and_the_salute_detector(self):
        self.assertIs(MODES["ash"], AshMode)
        self.assertEqual(MODE_CARDS[-1].key, "ash")
        self.assertEqual(app.KEY_ACTIONS[ord("5")], ("mode", "ash"))
        recognizer = Mock()
        self.assertEqual(app.detection_function("ash", recognizer), recognizer.recognize_pose)

    def test_salute_turns_to_ash_and_the_pile_stays_until_r(self):
        frame, mask, background = person()
        segment = Mock(return_value=mask)
        mode = AshMode(segment)
        mode.on_detection(True, 0.6)
        mode.on_detection(True, 1.0)
        self.assertEqual(mode.phase, "vanishing")
        for now in (1.0, 1.5, 2.5, 3.5):
            mode.render(FrameInfo(frame, background, now, int(now * 1000)))
        segment.assert_called_once()  # Silhouette captured once.
        gone = mode.render(FrameInfo(frame, background, 1.0 + EFFECT_SECONDS, 5000))
        self.assertEqual(mode.phase, "vanished")
        self.assertTrue(np.any(gone != background))  # The pile.
        np.testing.assert_array_equal(gone[:100], background[:100])
        # The pile is redrawn on a refreshed background, and goes away with R.
        brighter = np.full_like(background, 120)
        mode.background_changed()
        again = mode.render(FrameInfo(frame, brighter, 6.0, 6000))
        np.testing.assert_array_equal(again[:100], brighter[:100])
        mode.reappear()
        np.testing.assert_array_equal(mode.render(FrameInfo(frame, background, 7.0, 7000)), frame)


if __name__ == "__main__":
    unittest.main()
