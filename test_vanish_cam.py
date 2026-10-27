import unittest
from types import SimpleNamespace

import numpy as np
import mediapipe as mp

from modes import NormalMode
from vanish_cam import (
    MultiViewGestureRecognizer, fit_background, recognize_gesture, replace_background,
    update_background, victory_landmark_score,
)


def hand_points(extended=(True, True, False, False)):
    points = np.zeros((21, 3))
    for base, x, opened, spread in zip(
        (5, 9, 13, 17), (-0.45, 0.0, 0.4, 0.75), extended, (-0.35, 0.25, 0.0, 0.0)
    ):
        if opened:
            points[base : base + 4] = [
                (x, 1.0, 0.0), (x + spread * 0.45, 1.7, 0.0),
                (x + spread * 0.75, 2.15, 0.0), (x + spread, 2.5, 0.0),
            ]
        else:
            points[base : base + 4] = [
                (x, 1.0, 0.0), (x, 1.5, 0.2),
                (x, 1.3, 0.5), (x, 0.95, 0.3),
            ]
    return points


def landmarks(points):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]


class VictoryGestureTests(unittest.TestCase):
    def test_victory_with_back_palm_rotation_mirror_and_scale(self):
        points = hand_points()
        for angle in (0.0, 0.7, 1.57, 2.6, np.pi):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            tilt = np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
            for mirror in (1, -1):
                for scale in (0.02, 1.0, 100.0):
                    with self.subTest(angle=angle, mirror=mirror, scale=scale):
                        transformed = points @ rotation @ tilt
                        transformed[:, 0] *= mirror
                        transformed = transformed * scale + (0.5, 0.3, 0.1)
                        self.assertGreater(victory_landmark_score(transformed), 0.62)

    def test_other_gestures_do_not_trigger(self):
        for fingers in (
            (False, False, False, False),  # Fist.
            (True, True, True, True),  # Open palm.
            (True, False, False, False),  # Index only.
            (True, False, False, True),  # Horns.
            (True, True, True, False),  # Three fingers.
        ):
            with self.subTest(fingers=fingers):
                self.assertEqual(victory_landmark_score(hand_points(fingers)), 0.0)

    def test_invalid_or_degenerate_landmarks(self):
        for points in (np.zeros((21, 3)), np.zeros((20, 3)), np.full((21, 3), np.nan)):
            self.assertEqual(victory_landmark_score(points), 0.0)

    def test_landmarks_trigger_without_classifier_category(self):
        result = SimpleNamespace(
            hand_world_landmarks=[landmarks(hand_points())], hand_landmarks=[], gestures=[]
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        name, score = recognize_gesture(recognizer, None, 1000)
        self.assertEqual(name, "Victory")
        self.assertGreater(score, 0.62)

    def test_normalized_landmark_fallback_handles_video_aspect_ratio(self):
        image = SimpleNamespace(width=960, height=540)
        points = hand_points() * 40 + (480, 200, 0)
        normalized = points / (image.width, image.height, image.width)
        result = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[landmarks(normalized)], gestures=[]
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, image, 1000)[0], "Victory")

    def test_existing_classifier_still_works(self):
        result = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[],
            gestures=[[SimpleNamespace(category_name="Victory", score=0.9)]],
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, None, 1000), ("Victory", 0.9))

    def test_occluded_folded_fingers_can_appear_short_and_straight(self):
        points = hand_points()
        for base in (13, 17):
            points[base : base + 4] = [
                (0.4, 1.0, 0.0), (0.4, 1.1, 0.0),
                (0.4, 1.2, 0.0), (0.4, 1.3, 0.0),
            ]
        self.assertGreater(victory_landmark_score(points), 0.62)

    def test_hidden_fingers_with_coincident_landmarks(self):
        points = hand_points()
        points[13:16] = points[13]
        points[16] = points[13]
        points[17:20] = points[17]
        points[20] = points[17]
        self.assertGreater(victory_landmark_score(points), 0.62)

    def test_visible_geometry_is_used_even_with_inaccurate_world_landmarks(self):
        image = SimpleNamespace(width=960, height=540)
        points = hand_points() * 40 + (480, 200, 0)
        world = hand_points()
        world[8] = world[6]
        self.assertEqual(victory_landmark_score(world), 0.0)
        result = SimpleNamespace(
            hand_world_landmarks=[landmarks(world)],
            hand_landmarks=[landmarks(points / (960, 540, 960))], gestures=[],
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, image, 1000)[0], "Victory")

    def test_diagnostics_distinguish_missing_hand_and_rejected_gesture(self):
        for hands, expected in (([], "No_Hand"), ([landmarks(hand_points((False,) * 4))], "Hand_Detected")):
            result = SimpleNamespace(hand_world_landmarks=hands, hand_landmarks=[], gestures=[])
            recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
            self.assertEqual(recognize_gesture(recognizer, None, 1000)[0], expected)


class ImmediateAnimationTests(unittest.TestCase):
    def setUp(self):
        self.mode = NormalMode(segment=lambda frame, timestamp: None)

    def test_first_recognized_v_starts_animation_immediately(self):
        self.mode.on_detection(("Victory", 0.9), 12.5)
        self.assertEqual(self.mode.phase, "vanishing")
        self.assertEqual(self.mode.started, 12.5)

    def test_other_or_weak_gestures_do_not_vanish(self):
        for name, score in (("No_Hand", 0.0), ("Hand_Detected", 0.0), ("Victory", 0.61)):
            with self.subTest(name=name, score=score):
                self.mode.on_detection((name, score), 12.5)
                self.assertEqual(self.mode.phase, "ready")
                self.assertIsNone(self.mode.started)

    def test_only_ready_phase_responds(self):
        for phase in ("vanishing", "vanished"):
            self.mode.phase, self.mode.started = phase, 3.0
            self.mode.on_detection(("Victory", 0.9), 12.5)
            self.assertEqual(self.mode.phase, phase)
            self.assertEqual(self.mode.started, 3.0)


class BackgroundUpdateTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((180, 320, 3), 100, dtype=np.uint8)
        self.mask = np.zeros((180, 320), dtype=np.float32)
        self.mask[40:180, 140:180] = 1.0
        # The same background shot at twice the exposure, with the (dark) person in front.
        self.frame = np.full((180, 320, 3), 200, dtype=np.uint8)
        self.frame[40:180, 140:180] = 30

    def test_exposure_change_reaches_the_area_behind_the_person(self):
        updated = update_background(self.background, self.frame, self.mask, rate=0.5)
        # No shadow of the person: the covered area follows the new exposure.
        np.testing.assert_array_equal(updated[40:180, 140:180], 200)
        self.assertEqual(updated[5, 5].tolist(), [200, 200, 200])

    def test_local_change_far_from_person_blends_and_person_never_leaks(self):
        # Half brighter and half darker: the mean light does not change (gain 1).
        background = self.background.copy()
        background[:, :160] = 50
        background[:, 160:] = 150
        frame = np.empty_like(background)
        frame[:, :160] = 150
        frame[:, 160:] = 50
        frame[40:180, 140:180] = 30
        updated = update_background(background, frame, self.mask, rate=0.5)
        self.assertEqual(updated[5, 5].tolist(), [100, 100, 100])
        np.testing.assert_array_equal(updated[40:180, 140:180], background[40:180, 140:180])
        margin = round(320 * 0.04)
        np.testing.assert_array_equal(updated[100, 140 - margin:140], background[100, 140 - margin:140])

    def test_saturated_or_mostly_hidden_scenes_keep_the_background(self):
        mask = np.ones((180, 320), dtype=np.float32)
        updated = update_background(self.background, self.frame, mask)
        np.testing.assert_array_equal(updated, self.background)
        bright = np.full((180, 320, 3), 255, dtype=np.uint8)
        np.testing.assert_array_equal(update_background(bright, self.frame, self.mask)[40:180, 140:180], 255)


class BackgroundImageTests(unittest.TestCase):
    def test_fit_covers_frame_without_stretching(self):
        # A 4:3 picture with a red band at the top and bottom: at 16:9 they are cropped.
        image = np.zeros((300, 400, 3), np.uint8)
        image[:25] = (0, 0, 255)
        image[-25:] = (0, 0, 255)
        fitted = fit_background(image, 320, 180)
        self.assertEqual(fitted.shape, (180, 320, 3))
        self.assertEqual(int(fitted[:, :, 2].max()), 0)

    def test_small_image_is_upscaled(self):
        fitted = fit_background(np.full((10, 10, 3), 7, np.uint8), 64, 48)
        self.assertEqual(fitted.shape, (48, 64, 3))
        self.assertTrue(np.all(fitted == 7))

    def test_person_stays_and_image_fills_the_rest(self):
        frame = np.full((4, 4, 3), 200, np.uint8)
        background = np.full((4, 4, 3), 10, np.uint8)
        mask = np.zeros((4, 4), np.float32)
        mask[:, :2] = 1.0
        mask[0, 2] = 0.5
        result = replace_background(frame, background, mask)
        self.assertTrue(np.all(result[:, :2] == 200))
        self.assertTrue(np.all(result[1:, 2:] == 10))
        self.assertEqual(int(result[0, 2, 0]), 105)


class MultiViewTests(unittest.TestCase):
    def setUp(self):
        self.empty = SimpleNamespace(hand_world_landmarks=[], hand_landmarks=[], gestures=[])
        self.victory = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[],
            gestures=[[SimpleNamespace(category_name="Victory", score=0.9)]],
        )
        self.image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.zeros((32, 48, 3), dtype=np.uint8)
        )

    def test_rotated_detection_and_preferred_view_are_used(self):
        calls = []

        def recognize(image):
            calls.append((image.width, image.height))
            return self.victory if image.width == 32 else self.empty

        detector = MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.empty),
            SimpleNamespace(recognize=recognize),
        )
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "Victory")
        self.assertEqual(calls, [(48, 32), (32, 48)])
        self.assertEqual(detector.preferred_rotation, 1)
        calls.clear()
        self.assertEqual(recognize_gesture(detector, self.image, 1140)[0], "Victory")
        self.assertEqual(calls, [(32, 48)])

    def test_failed_search_is_throttled_without_reusing_old_positive(self):
        calls = []
        alternate = SimpleNamespace(recognize=lambda image: calls.append(image) or self.empty)
        detector = MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.empty), alternate
        )
        # At most 3 expensive views per frame, then a pause until the next interval.
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "No_Hand")
        self.assertEqual(len(calls), 3)
        recognize_gesture(detector, self.image, 1100)
        self.assertEqual(len(calls), 3)
        recognize_gesture(detector, self.image, 1140)
        self.assertEqual(len(calls), 6)
        alternate.recognize = lambda image: self.victory
        self.assertEqual(recognize_gesture(detector, self.image, 1420)[0], "Victory")
        alternate.recognize = lambda image: self.empty
        self.assertEqual(recognize_gesture(detector, self.image, 1560)[0], "No_Hand")

    def test_primary_victory_avoids_extra_inference(self):
        calls = []
        detector = MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.victory),
            SimpleNamespace(recognize=lambda image: calls.append(image) or self.empty),
        )
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "Victory")
        self.assertEqual(calls, [])

    def test_lower_hand_crop_recovers_a_small_sideways_v(self):
        calls = []

        def recognize(image):
            calls.append((image.width, image.height))
            return self.victory if (image.width, image.height) == (37, 24) else self.empty

        detector = MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.empty),
            SimpleNamespace(recognize=recognize),
        )
        # The alternative views are spread over several frames: the lower
        # crop comes at the second attempt.
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "No_Hand")
        self.assertEqual(recognize_gesture(detector, self.image, 1140)[0], "Victory")
        self.assertEqual(detector.preferred_crop, "lower_left")
        self.assertEqual(detector.preferred_rotation, 0)
        calls.clear()
        self.assertEqual(recognize_gesture(detector, self.image, 1280)[0], "Victory")
        self.assertEqual(calls, [(37, 24)])


if __name__ == "__main__":
    unittest.main()
