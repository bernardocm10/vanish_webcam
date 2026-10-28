import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from vanishcam.detectors import MemePoseDetector
from vanishcam.meme_poses import GESTURES, POSES, match_gesture
from .hand_fixtures import make_face, make_hand

# A face in the middle of the frame, normalized (x1, y1, x2, y2).
FACE = (0.4, 0.2, 0.6, 0.5)
FIST = (False, False, False, False)


def fist_thumb_up():
    return make_hand(0.5, 0.75, 0.16, extended=FIST, thumb="up")


class MemePoseTests(unittest.TestCase):
    def matches(self, name, hands, face=FACE):
        return POSES[name](hands, face)[0]

    def test_thumbs_up(self):
        self.assertTrue(self.matches("thumbs_up", [fist_thumb_up()]))
        self.assertFalse(self.matches("thumbs_up", [make_hand(0.5, 0.75, 0.16)]))  # Open hand.
        self.assertFalse(self.matches("thumbs_up", [fist_thumb_up()], face=None))

    def test_calm(self):
        self.assertTrue(self.matches("calm", [make_hand(0.5, 0.7, 0.2)]))
        self.assertFalse(self.matches("calm", [make_hand(0.5, 0.7, 0.2, extended=FIST)]))
        self.assertFalse(self.matches("calm", [make_hand(0.5, 0.7, 0.2, angle=90)]))  # Fingers sideways.

    def test_hearing(self):
        self.assertTrue(self.matches("hearing", [make_hand(0.66, 0.35, 0.18)]))
        self.assertFalse(self.matches("hearing", [make_hand(0.5, 0.35, 0.18)]))  # In front of the face.

    def test_hands_behind_head(self):
        hands = [make_hand(0.33, 0.3, 0.15), make_hand(0.67, 0.3, 0.15)]
        self.assertTrue(self.matches("hands_behind_head", hands))
        self.assertFalse(self.matches("hands_behind_head", hands[:1]))
        self.assertFalse(self.matches("hands_behind_head", [make_hand(0.33, 0.3, 0.15), make_hand(0.67, 0.75, 0.15)]))

    def test_prayer(self):
        joined = [make_hand(0.5, 0.6, 0.2, extended=FIST), make_hand(0.51, 0.6, 0.2, extended=FIST)]
        self.assertTrue(self.matches("prayer", joined))
        self.assertFalse(self.matches("prayer", [make_hand(0.3, 0.6, 0.2), make_hand(0.7, 0.6, 0.2)]))

    def test_timeout(self):
        upright = make_hand(0.5, 0.75, 0.15)
        across = make_hand(0.5, 0.64, 0.15, angle=90)
        self.assertTrue(self.matches("timeout", [upright, across]))
        self.assertFalse(self.matches("timeout", [upright, make_hand(0.56, 0.75, 0.15)]))  # Parallel.

    def test_pointing_at_self(self):
        pointing = make_hand(0.6, 0.72, 0.2, extended=(True, False, False, False), angle=-40)
        self.assertTrue(self.matches("pointing_at_self", [pointing]))
        self.assertFalse(self.matches("pointing_at_self", [make_hand(0.6, 0.72, 0.2, angle=-40)]))  # Open hand.

    def test_hands_on_face(self):
        hands = [make_hand(0.45, 0.4, 0.15), make_hand(0.55, 0.4, 0.15)]
        self.assertTrue(self.matches("hands_on_face", hands))
        self.assertFalse(self.matches("hands_on_face", hands[:1]))

    def test_precedence_and_disabled_gestures(self):
        self.assertEqual(match_gesture([fist_thumb_up()], FACE, frozenset(GESTURES)), "thumbs_up")
        self.assertIsNone(match_gesture([fist_thumb_up()], FACE, frozenset(GESTURES) - {"thumbs_up"}))
        self.assertIsNone(match_gesture([], None, frozenset(GESTURES)))
        self.assertEqual(
            GESTURES,
            ("thumbs_up", "hands_behind_head", "timeout", "calm", "hands_on_face", "prayer", "pointing_at_self", "hearing"),
        )


class MemeDetectorTests(unittest.TestCase):
    def test_face_is_remembered_and_only_enabled_gestures_match(self):
        face, hands = Mock(), Mock()
        face.detect_for_video.return_value = SimpleNamespace(face_landmarks=[make_face(*FACE)])
        hands.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[fist_thumb_up()])
        detector = MemePoseDetector(face, hands)
        detector.enabled = frozenset({"thumbs_up"})
        self.assertEqual(detector.recognize(object(), 0), "thumbs_up")
        face.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        self.assertEqual(detector.recognize(object(), 1000), "thumbs_up")  # Face hidden for 1 s: remembered.
        self.assertIsNone(detector.recognize(object(), 2300))  # Too long ago.
        detector.enabled = frozenset()
        face.detect_for_video.return_value = SimpleNamespace(face_landmarks=[make_face(*FACE)])
        self.assertIsNone(detector.recognize(object(), 2400))
        detector.close()
        face.close.assert_called_once()
        hands.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
