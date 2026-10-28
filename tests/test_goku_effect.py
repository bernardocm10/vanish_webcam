import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from vanishcam.detectors import GokuPoseDetector
from vanishcam.modes import FrameInfo, GokuMode
from vanishcam.effects.goku import (
    CHEEK_LEFT, CHEEK_RIGHT, FOREHEAD_TOP, NOSE_BRIDGE, HoldTracker, face_reference,
    forehead_fingers_pose, make_speed_lines, render_teleport, sound_path, teleport_region,
)


WIDTH, HEIGHT = 640, 480


def face_landmarks():
    points = [SimpleNamespace(x=0.5, y=0.4) for _ in range(478)]
    points[FOREHEAD_TOP] = SimpleNamespace(x=0.5, y=0.25)
    points[NOSE_BRIDGE] = SimpleNamespace(x=0.5, y=0.40)
    points[CHEEK_LEFT] = SimpleNamespace(x=0.38, y=0.45)
    points[CHEEK_RIGHT] = SimpleNamespace(x=0.62, y=0.45)
    return points


def finger(base, tip):
    return [SimpleNamespace(x=base[0] + (tip[0] - base[0]) * t, y=base[1] + (tip[1] - base[1]) * t)
            for t in (0.0, 0.35, 0.7, 1.0)]


def hand(index_tip=(0.49, 0.31), middle_tip=(0.51, 0.30), ring_extended=False, shift=0.0):
    points = [SimpleNamespace(x=0.47, y=0.52) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.52, y=0.60)
    points[5:9] = finger((0.49, 0.48), index_tip)
    points[9:13] = finger((0.51, 0.48), middle_tip)
    if ring_extended:
        points[13:17] = finger((0.53, 0.49), (0.53, 0.31))
    else:
        points[13:17] = [SimpleNamespace(x=0.53, y=0.49), SimpleNamespace(x=0.535, y=0.46),
                         SimpleNamespace(x=0.53, y=0.47), SimpleNamespace(x=0.525, y=0.50)]
    points[17:21] = [SimpleNamespace(x=0.55, y=0.50), SimpleNamespace(x=0.555, y=0.48),
                     SimpleNamespace(x=0.55, y=0.49), SimpleNamespace(x=0.545, y=0.51)]
    return [SimpleNamespace(x=p.x + shift, y=p.y) for p in points]


class ForeheadPoseTests(unittest.TestCase):
    def setUp(self):
        self.face = face_reference(face_landmarks(), WIDTH, HEIGHT)

    def test_two_fingers_together_on_forehead(self):
        self.assertTrue(forehead_fingers_pose(hand(), self.face, WIDTH, HEIGHT))

    def test_rejects_pose_away_from_forehead_or_without_face(self):
        self.assertFalse(forehead_fingers_pose(hand(shift=0.25), self.face, WIDTH, HEIGHT))
        self.assertFalse(forehead_fingers_pose(hand(), None, WIDTH, HEIGHT))

    def test_rejects_v_sign_and_open_hand(self):
        v_sign = hand(index_tip=(0.43, 0.31), middle_tip=(0.57, 0.30))
        self.assertFalse(forehead_fingers_pose(v_sign, self.face, WIDTH, HEIGHT))
        self.assertFalse(forehead_fingers_pose(hand(ring_extended=True), self.face, WIDTH, HEIGHT))


class HoldTrackerTests(unittest.TestCase):
    def test_fires_once_after_one_second(self):
        tracker = HoldTracker()
        self.assertFalse(tracker.update(True, 0.0))
        self.assertAlmostEqual(tracker.remaining(0.4), 0.6)
        self.assertTrue(tracker.update(True, 1.0))
        self.assertFalse(tracker.update(True, 2.5))

    def test_short_dropout_keeps_countdown_but_broken_pose_restarts_it(self):
        tracker = HoldTracker()
        tracker.update(True, 0.0)
        tracker.update(False, 0.15)
        self.assertTrue(tracker.update(True, 1.0))
        tracker = HoldTracker()
        tracker.update(True, 0.0)
        tracker.update(False, 0.5)
        self.assertIsNone(tracker.remaining(0.5))
        self.assertFalse(tracker.update(True, 1.0))
        self.assertTrue(tracker.update(True, 2.0))

    def test_hand_must_be_lowered_before_next_trigger(self):
        tracker = HoldTracker()
        tracker.update(True, 0.0)
        self.assertTrue(tracker.update(True, 1.0))
        tracker.update(False, 1.1)
        self.assertFalse(tracker.update(True, 1.2))  # Lowered for less than 0.3s.
        tracker.update(False, 1.3)
        tracker.update(False, 1.7)
        tracker.update(True, 1.8)
        self.assertTrue(tracker.update(True, 2.85))


class TeleportRenderTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        self.background = np.full((120, 160, 3), 200, dtype=np.uint8)
        self.mask = np.zeros((120, 160), dtype=np.float32)
        self.mask[20:110, 60:100] = 1.0
        self.region = teleport_region(self.mask)
        self.lines = make_speed_lines(160)

    def render(self, progress, age=0.3):
        return render_teleport(self.frame, self.background, self.mask, self.region,
                               self.lines, progress, age)

    def test_person_visible_at_start_and_background_at_end(self):
        start = self.render(0.0)
        np.testing.assert_array_equal(start[30:100, 65:95], self.frame[30:100, 65:95])
        np.testing.assert_array_equal(self.render(1.0), self.background)

    def test_person_squeezes_and_lines_linger_after_disappearing(self):
        squeezed = self.render(0.5)
        self.assertTrue((squeezed[60, 62] > 120).all())  # Silhouette edge already pulled in.
        gone = self.render(0.8)
        self.assertGreater(int(gone[20:110, 60:100].max()), 210)  # Bright streaks remain.
        np.testing.assert_array_equal(gone[:, :20], self.background[:, :20])

    def test_empty_mask_returns_background(self):
        empty = np.zeros_like(self.mask)
        result = render_teleport(self.frame, self.background, empty, empty, self.lines, 0.4, 0.2)
        np.testing.assert_array_equal(result, self.background)


class SoundPathTests(unittest.TestCase):
    def test_finds_teleport_sound_with_any_letter_case(self):
        with tempfile.TemporaryDirectory() as folder:
            assets = Path(folder)
            self.assertIsNone(sound_path(assets))
            (assets / "teleport.mp3").write_bytes(b"x")
            self.assertEqual(sound_path(assets), assets / "teleport.mp3")


class GokuModeTests(unittest.TestCase):
    def setUp(self):
        self.sound = Mock()
        self.mode = GokuMode(segment=lambda frame, timestamp: self.mask, sound=self.sound)
        self.frame = np.full((60, 80, 3), 40, np.uint8)
        self.background = np.full((60, 80, 3), 200, np.uint8)
        self.mask = np.zeros((60, 80), np.float32)
        self.mask[10:60, 30:50] = 1.0

    def render(self, now):
        return self.mode.render(FrameInfo(self.frame, self.background, now, int(now * 1000)))

    def hold_pose(self, start):
        for step in range(0, 11):
            self.mode.on_detection(True, start + step * 0.1)

    def test_pose_held_vanishes_and_repeated_pose_brings_person_back(self):
        with patch("builtins.print"):
            self.hold_pose(0.0)
            self.assertEqual(self.mode.phase, "vanishing")
            self.render(0.5)
            np.testing.assert_array_equal(self.render(2.6), self.background)
            self.assertEqual(self.mode.phase, "vanished")
            self.assertTrue(self.mode.detecting)
            self.hold_pose(3.0)  # Hand never lowered: does not reappear.
            self.assertEqual(self.mode.phase, "vanished")
            self.mode.on_detection(False, 4.2)
            self.mode.on_detection(False, 4.6)
            self.hold_pose(4.7)
            self.assertEqual(self.mode.phase, "appearing")
            self.assertFalse(self.mode.detecting)
            self.render(6.0)
            last = self.render(7.3)  # Last frame of the effect: the whole person over the background.
            np.testing.assert_array_equal(last[20:60, 35:45], self.frame[20:60, 35:45])
            self.assertEqual(self.mode.phase, "ready")
            np.testing.assert_array_equal(self.render(7.4), self.frame)
        self.assertEqual(self.sound.play.call_count, 2)

    def test_keyboard_is_disabled_and_detection_uses_half_resolution(self):
        self.assertFalse(self.mode.manual_keys)
        big = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(self.mode.detection_image(big).shape, (360, 640, 3))


class GokuDetectorTests(unittest.TestCase):
    def test_remembers_forehead_while_fingers_hide_the_face(self):
        face = Mock()
        face.detect_for_video.side_effect = [
            SimpleNamespace(face_landmarks=[face_landmarks()]),
            SimpleNamespace(face_landmarks=[]),
            SimpleNamespace(face_landmarks=[]),
        ]
        hands = Mock()
        hands.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[hand()])
        detector = GokuPoseDetector(face, hands)
        image = SimpleNamespace(width=WIDTH, height=HEIGHT)
        self.assertTrue(detector.recognize_pose(image, 0))
        self.assertTrue(detector.recognize_pose(image, 500))
        self.assertFalse(detector.recognize_pose(image, 1600))


if __name__ == "__main__":
    unittest.main()
