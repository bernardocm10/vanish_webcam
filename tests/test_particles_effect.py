import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from vanishcam import app
from vanishcam.effects.goku import CHEEK_LEFT, CHEEK_RIGHT, FOREHEAD_TOP, NOSE_BRIDGE
from vanishcam.modes import FrameInfo, ParticlesMode
from vanishcam.effects.particles import (
    BROW_LEFT, BROW_RIGHT, DEPART_SECONDS, EFFECT_SECONDS, make_particles, render_particles,
    salute_pose, salute_reference,
)

WIDTH, HEIGHT = 640, 480


def face_landmarks():
    points = [SimpleNamespace(x=0.5, y=0.4) for _ in range(478)]
    points[FOREHEAD_TOP] = SimpleNamespace(x=0.5, y=0.25)
    points[NOSE_BRIDGE] = SimpleNamespace(x=0.5, y=0.40)
    points[CHEEK_LEFT] = SimpleNamespace(x=0.38, y=0.45)
    points[CHEEK_RIGHT] = SimpleNamespace(x=0.62, y=0.45)
    points[BROW_LEFT] = SimpleNamespace(x=0.42, y=0.33)
    points[BROW_RIGHT] = SimpleNamespace(x=0.58, y=0.33)
    return points


def finger(base, tip, bend=0.0):
    """Four joints from base to tip; bend pulls the middle joints sideways (a curled finger)."""
    return [SimpleNamespace(x=base[0] + (tip[0] - base[0]) * t, y=base[1] + (tip[1] - base[1]) * t + bend * t * (1 - t) * 4)
            for t in (0.0, 0.35, 0.7, 1.0)]


def salute(shift=0.0, spread=0.0, folded=False, bend=0.0):
    """Hand coming from the side, fingers pointing at the brow on the right of the image."""
    points = [SimpleNamespace(x=0.74 + shift, y=0.40) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.76 + shift, y=0.42)
    rows = ((5, 0.30, 0.59), (9, 0.32, 0.58), (13, 0.34, 0.585), (17, 0.36, 0.60))
    for number, (base, y, tip_x) in enumerate(rows):
        tip_y = y + (number - 1.5) * spread
        if folded and base in (13, 17):
            points[base : base + 4] = finger((0.68 + shift, y), (0.665 + shift, y + 0.01))
        else:
            points[base : base + 4] = finger((0.68 + shift, y), (tip_x + shift, tip_y), bend)
    return points


class ParticlesEffectTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((120, 160, 3), np.uint8)
        self.frame[:, :, 1] = 180  # Green person on a grey background.
        self.background = np.full((120, 160, 3), 90, np.uint8)
        self.mask = np.zeros((120, 160), np.float32)
        self.mask[30:110, 60:100] = 1.0
        self.cloud = make_particles(self.frame, self.mask)

    def test_fragments_take_the_silhouette_colors(self):
        self.assertGreater(len(self.cloud.origin), 50)
        np.testing.assert_allclose(self.cloud.color, [[0, 180, 0]] * len(self.cloud.color))
        self.assertTrue(np.all(self.cloud.delay <= DEPART_SECONDS))

    def test_whole_silhouette_leaves_no_ghost_outline(self):
        inside = self.cloud.body_delay[self.mask > 0]
        self.assertTrue(np.all(np.isfinite(inside)))
        self.assertTrue(np.all(inside <= DEPART_SECONDS))

    def test_body_first_then_wind_then_only_background(self):
        start = render_particles(self.background, self.cloud, 0.0)
        np.testing.assert_array_equal(start[40:100, 65:95], self.frame[40:100, 65:95])
        middle = render_particles(self.background, self.cloud, 0.8)
        # The wind blows to the right: fragments past the right edge of the silhouette.
        self.assertTrue(np.any(middle[:, 105:] != self.background[:, 105:]))
        np.testing.assert_array_equal(middle[:, :50], self.background[:, :50])
        np.testing.assert_array_equal(
            render_particles(self.background, self.cloud, EFFECT_SECONDS), self.background
        )

    def test_empty_mask_is_just_background(self):
        cloud = make_particles(self.frame, np.zeros_like(self.mask))
        np.testing.assert_array_equal(render_particles(self.background, cloud, 0.5), self.background)


class SalutePoseTests(unittest.TestCase):
    def setUp(self):
        self.face = salute_reference(face_landmarks(), WIDTH, HEIGHT)

    def test_flat_hand_on_the_brow_is_a_salute(self):
        self.assertTrue(salute_pose(salute(), self.face, WIDTH, HEIGHT))

    def test_rejects_poses_that_are_not_a_salute(self):
        self.assertFalse(salute_pose(salute(), None, WIDTH, HEIGHT))  # No face.
        self.assertFalse(salute_pose(salute(shift=0.15), self.face, WIDTH, HEIGHT))  # Hand far away.
        self.assertFalse(salute_pose(salute(spread=0.06), self.face, WIDTH, HEIGHT))  # Fingers fanned out.
        self.assertFalse(salute_pose(salute(folded=True), self.face, WIDTH, HEIGHT))  # Goku pose.
        self.assertFalse(salute_pose(salute(bend=0.05), self.face, WIDTH, HEIGHT))  # Bent fingers.
        self.assertFalse(salute_pose(salute()[:20], self.face, WIDTH, HEIGHT))

    def test_detector_remembers_the_brow_while_the_hand_covers_it(self):
        face = Mock()
        face.detect_for_video.side_effect = [
            SimpleNamespace(face_landmarks=[face_landmarks()]),
            SimpleNamespace(face_landmarks=[]),
            SimpleNamespace(face_landmarks=[]),
        ]
        hands = Mock()
        hands.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[salute()])
        detector = app.SalutePoseDetector(face, hands)
        image = SimpleNamespace(width=WIDTH, height=HEIGHT)
        self.assertTrue(detector.recognize_pose(image, 0))
        self.assertTrue(detector.recognize_pose(image, 500))
        self.assertFalse(detector.recognize_pose(image, 1600))

    def test_particles_mode_uses_the_salute_detector(self):
        recognizer = Mock()
        detect = app.detection_function("particles", recognizer)
        self.assertEqual(detect, recognizer.recognize_pose)


class ParticlesModeTests(unittest.TestCase):
    def test_salute_must_be_held_and_released_before_the_next_one(self):
        mode = ParticlesMode(Mock())
        mode.on_detection(True, 1.0)
        mode.on_detection(True, 1.2)
        self.assertEqual(mode.phase, "ready")
        mode.on_detection(True, 1.45)
        self.assertEqual(mode.phase, "vanishing")
        mode.reappear()
        mode.on_detection(True, 1.6)
        mode.on_detection(True, 2.5)
        self.assertEqual(mode.phase, "ready")  # Hand still on the forehead.
        mode.on_detection(False, 2.6)
        mode.on_detection(False, 3.0)
        mode.on_detection(True, 3.1)
        mode.on_detection(True, 3.6)
        self.assertEqual(mode.phase, "vanishing")

    def test_hand_passing_over_the_forehead_does_not_trigger(self):
        mode = ParticlesMode(Mock())
        for now, saluting in ((1.0, True), (1.07, True), (1.14, False), (1.5, False), (1.6, True), (1.7, False)):
            mode.on_detection(saluting, now)
        self.assertEqual(mode.phase, "ready")

    def test_salute_captures_silhouette_once_and_ends_on_background(self):
        frame = np.full((60, 80, 3), 40, np.uint8)
        background = np.full((60, 80, 3), 200, np.uint8)
        mask = np.zeros((60, 80), np.float32)
        mask[10:60, 30:50] = 1.0
        segment = Mock(return_value=mask)
        mode = ParticlesMode(segment)
        mode.on_detection(True, 0.6)
        mode.on_detection(True, 1.0)
        self.assertEqual(mode.phase, "vanishing")
        for now in (1.0, 1.5, 2.0, 2.5):
            mode.render(FrameInfo(frame, background, now, int(now * 1000)))
        segment.assert_called_once()
        np.testing.assert_array_equal(mode.render(FrameInfo(frame, background, 3.2, 3200)), background)
        self.assertEqual(mode.phase, "vanished")


if __name__ == "__main__":
    unittest.main()
