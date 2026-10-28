import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from vanishcam import app
from vanishcam.detectors import HandsUpPoseDetector
from vanishcam.effects.ash import (
    ASH_SECONDS, CHIN, CRUMBLE_START, EFFECT_SECONDS, hands_up_pose, hands_up_reference,
    make_ash, render_ash, render_remains,
)
from vanishcam.control_panel import MODE_CARDS
from vanishcam.effects.goku import CHEEK_LEFT, CHEEK_RIGHT, FOREHEAD_TOP, NOSE_BRIDGE
from vanishcam.shortcuts import Shortcuts
from vanishcam.modes import MODES, AshMode, FrameInfo

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


def face_landmarks():
    points = [SimpleNamespace(x=0.5, y=0.4) for _ in range(478)]
    points[FOREHEAD_TOP] = SimpleNamespace(x=0.5, y=0.22)
    points[NOSE_BRIDGE] = SimpleNamespace(x=0.5, y=0.36)
    points[CHEEK_LEFT] = SimpleNamespace(x=0.38, y=0.45)
    points[CHEEK_RIGHT] = SimpleNamespace(x=0.62, y=0.45)
    points[CHIN] = SimpleNamespace(x=0.5, y=0.55)
    return points


def hand(x, y, up=True, open_hand=True):
    """A hand with its knuckles at (x, y): fingers spread and pointing up (or down), or folded into a fist."""
    sign = -1 if up else 1
    points = [SimpleNamespace(x=x, y=y) for _ in range(21)]
    points[0] = SimpleNamespace(x=x, y=y - sign * 0.12)
    for number, base in enumerate((5, 9, 13, 17)):
        spread = (number - 1.5) * 0.025
        reach = (0.03, 0.05, 0.07) if open_hand else (0.03, 0.01, -0.02)
        points[base] = SimpleNamespace(x=x + spread, y=y)
        for joint, step in zip((base + 1, base + 2, base + 3), reach):
            points[joint] = SimpleNamespace(x=x + spread * (1 + step * 8), y=y + sign * step)
    return points


class HandsUpPoseTests(unittest.TestCase):
    def setUp(self):
        self.face = hands_up_reference(face_landmarks(), 640, 480)

    def pose(self, hands, face="default"):
        return hands_up_pose(hands, self.face if face == "default" else face, 640, 480)

    def test_both_open_hands_raised_beside_the_head(self):
        self.assertTrue(self.pose([hand(0.25, 0.3), hand(0.75, 0.3)]))
        self.assertTrue(self.pose([hand(0.3, 0.45), hand(0.72, 0.2)]))  # Not at the same height.

    def test_one_hand_or_no_face_is_not_enough(self):
        self.assertFalse(self.pose([hand(0.25, 0.3)]))
        self.assertFalse(self.pose([hand(0.25, 0.3), hand(0.75, 0.3)], face=None))

    def test_hands_below_the_chin_are_not_up(self):
        self.assertFalse(self.pose([hand(0.25, 0.65), hand(0.75, 0.65)]))

    def test_both_hands_on_the_same_side_are_not_the_pose(self):
        self.assertFalse(self.pose([hand(0.2, 0.3), hand(0.3, 0.3)]))

    def test_hands_in_front_of_the_face_are_not_the_pose(self):
        self.assertFalse(self.pose([hand(0.47, 0.3), hand(0.53, 0.3)]))

    def test_fists_or_hands_pointing_down_do_not_count(self):
        self.assertFalse(self.pose([hand(0.25, 0.3, open_hand=False), hand(0.75, 0.3, open_hand=False)]))
        self.assertFalse(self.pose([hand(0.25, 0.3, up=False), hand(0.75, 0.3, up=False)]))

    def test_detector_checks_both_hands_together(self):
        face = Mock()
        face.detect_for_video.return_value = SimpleNamespace(face_landmarks=[face_landmarks()])
        hands = Mock()
        hands.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[hand(0.25, 0.3), hand(0.75, 0.3)])
        detector = HandsUpPoseDetector(face, hands)
        image = SimpleNamespace(width=640, height=480)
        self.assertTrue(detector.recognize_pose(image, 0))
        hands.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[hand(0.25, 0.3)])
        self.assertFalse(detector.recognize_pose(image, 100))


class AshModeTests(unittest.TestCase):
    def test_registered_with_a_panel_row_and_a_pose_detector(self):
        self.assertIs(MODES["ash"], AshMode)
        self.assertEqual(MODE_CARDS[-1].key, "ash")
        self.assertEqual(Shortcuts().with_enabled(True).action_for(ord("5")), ("mode", "ash"))
        recognizer, image = Mock(), object()
        self.assertIs(AshMode.detect(recognizer, image, 0), recognizer.recognize_pose.return_value)
        recognizer.recognize_pose.assert_called_once_with(image, 0)

    def test_hands_up_turns_to_ash_and_the_pile_stays_until_r(self):
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
