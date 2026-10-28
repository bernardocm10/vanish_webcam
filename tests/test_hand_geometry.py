import unittest

from vanishcam.effects import ash
from vanishcam.hand_geometry import extended_finger_count, extended_finger_states, face_box, hand_features, points_of
from .hand_fixtures import make_face, make_hand


class HandGeometryTests(unittest.TestCase):
    def test_extended_fingers_follow_the_fixture(self):
        points = points_of(make_hand(0.5, 0.5, 0.2, extended=(True, False, True, False)))
        self.assertEqual(extended_finger_states(points), [True, False, True, False])
        self.assertEqual(extended_finger_count(points_of(make_hand(0.5, 0.5, 0.2))), 4)

    def test_face_box_and_features(self):
        self.assertEqual(face_box(make_face(0.4, 0.2, 0.6, 0.5)), (0.4, 0.2, 0.6, 0.5))
        features = hand_features(make_hand(0.5, 0.6, 0.2))
        self.assertLess(features["fingertip_y"], features["wrist_y"])  # Fingers up.
        self.assertAlmostEqual(features["center"][0], 0.5)

    def test_ash_keeps_its_threshold(self):
        points = points_of(make_hand(0.5, 0.5, 0.2)) * (640, 480)
        self.assertEqual(ash.extended_fingers(points), extended_finger_count(points, ratio=1.15))


if __name__ == "__main__":
    unittest.main()
