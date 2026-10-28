import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from vanishcam import app
from vanishcam.modes import NarutoMode
from vanishcam.effects.naruto import (
    EFFECT_SECONDS, falling_log_anchor, hand_seal_from_landmarks, joined_hands_from_landmarks,
    landed_log_anchor, load_log, overlay_log, render_substitution, substitution_anchor,
)
from .test_video_output import InlineRecognizer, app_mocks, arguments, video_clock


PHOTO_CLASP = [[SimpleNamespace(x=x, y=y) for x, y in hand] for hand in (
    [(0.4888, 0.5882), (0.5101, 0.5389), (0.5299, 0.4977), (0.5573, 0.4664), (0.5941, 0.4437),
     (0.4463, 0.4416), (0.5024, 0.3985), (0.563, 0.3987), (0.6066, 0.4168), (0.4484, 0.4527),
     (0.5182, 0.3979), (0.5841, 0.4043), (0.6221, 0.4296), (0.4628, 0.4762), (0.5212, 0.4141),
     (0.5804, 0.4167), (0.6197, 0.4336), (0.486, 0.5069), (0.5201, 0.4814), (0.5667, 0.468),
     (0.6056, 0.4635)],
    [(0.6263, 0.6054), (0.5951, 0.5557), (0.5632, 0.5043), (0.53, 0.4653), (0.5089, 0.4327),
     (0.6163, 0.4469), (0.5242, 0.3852), (0.4559, 0.3983), (0.4421, 0.4305), (0.621, 0.4507),
     (0.5264, 0.3792), (0.4587, 0.3972), (0.4523, 0.4353), (0.6144, 0.4675), (0.5308, 0.3928),
     (0.4649, 0.4035), (0.4582, 0.4343), (0.6032, 0.4917), (0.5368, 0.451), (0.4955, 0.4408),
     (0.4898, 0.4524)],
)]


class JoinedHandsTests(unittest.TestCase):
    def test_joined_hands_must_be_held_briefly(self):
        sound = Mock()
        mode = NarutoMode(segment=lambda frame, timestamp: None, sound=sound)
        mode.on_detection(False, 1.0)
        mode.on_detection(True, 1.1)
        mode.on_detection(True, 1.2)
        self.assertEqual(mode.phase, "ready")  # Hands just joined: nothing yet.
        mode.on_detection(True, 1.35)
        self.assertEqual(mode.phase, "vanishing")
        self.assertEqual(mode.started, 1.35)
        mode.on_detection(True, 1.5)
        self.assertEqual(mode.started, 1.35)
        sound.play.assert_called_once()

    def test_hands_passing_through_the_pose_do_not_trigger(self):
        mode = NarutoMode(segment=lambda frame, timestamp: None, sound=Mock())
        for now, joined in ((1.0, True), (1.08, True), (1.16, False), (1.24, False), (1.6, False), (1.68, True)):
            mode.on_detection(joined, now)
        self.assertEqual(mode.phase, "ready")

    def test_one_missed_detection_does_not_cancel_the_gesture(self):
        mode = NarutoMode(segment=lambda frame, timestamp: None, sound=Mock())
        for now, joined in ((1.0, True), (1.08, False), (1.16, True), (1.24, True)):
            mode.on_detection(joined, now)
        self.assertEqual(mode.phase, "vanishing")

    def test_hands_must_separate_before_vanishing_again(self):
        mode = NarutoMode(segment=lambda frame, timestamp: None, sound=Mock())
        mode.on_detection(True, 1.0)
        mode.on_detection(True, 1.3)
        mode.reappear()
        mode.on_detection(True, 1.4)
        mode.on_detection(True, 2.0)
        self.assertEqual(mode.phase, "ready")
        mode.on_detection(False, 2.1)
        mode.on_detection(False, 2.5)
        mode.on_detection(True, 2.6)
        mode.on_detection(True, 2.9)
        self.assertEqual(mode.phase, "vanishing")

    @staticmethod
    def hand(center_x, upward=True):
        points = [SimpleNamespace(x=center_x, y=0.53) for _ in range(21)]
        points[0] = SimpleNamespace(x=center_x, y=0.70)
        for index, offset in ((5, -0.045), (9, -0.015), (13, 0.015), (17, 0.045)):
            points[index] = SimpleNamespace(x=center_x + offset, y=0.54 if upward else 0.84)
        for index, offset in ((8, -0.045), (12, -0.015), (16, 0.015), (20, 0.045)):
            points[index] = SimpleNamespace(x=center_x + offset, y=0.34 if upward else 1.04)
        return points

    @staticmethod
    def tilted(hand, degrees, pivot=None):
        """Rotate a hand in the image around its wrist (in pixels, so the angle is real)."""
        angle = np.radians(degrees)
        c, s = np.cos(angle), np.sin(angle)
        px, py = (hand[0].x * 640, hand[0].y * 480) if pivot is None else pivot
        out = []
        for p in hand:
            x, y = p.x * 640 - px, p.y * 480 - py
            out.append(SimpleNamespace(x=(px + c * x - s * y) / 640, y=(py + s * x + c * y) / 480))
        return out

    def test_touching_palms_recognized_but_one_or_separated_hands_are_not(self):
        left, right = self.hand(0.47), self.hand(0.53)
        self.assertTrue(joined_hands_from_landmarks([left, right], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left, self.hand(0.83)], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left, self.hand(0.53, upward=False)], 640, 480))

    def test_open_palms_side_by_side_before_the_gesture_are_not_joined(self):
        # This was the false positive: open hands close together, but not joined yet.
        self.assertFalse(joined_hands_from_landmarks([self.hand(0.45), self.hand(0.55)], 640, 480))
        # Almost touching is enough: the hand model is not pixel accurate.
        self.assertTrue(joined_hands_from_landmarks([self.hand(0.46), self.hand(0.54)], 640, 480))

    def test_slight_tilt_is_tolerated_but_open_or_crossed_hands_are_not(self):
        left, right = self.hand(0.47), self.hand(0.53)
        # Tips 15 degrees inwards: palms still joined.
        self.assertTrue(joined_hands_from_landmarks([self.tilted(left, 15), self.tilted(right, -15)], 640, 480))
        # Hands open in a V, right before closing them.
        self.assertFalse(joined_hands_from_landmarks([self.tilted(left, -15), self.tilted(right, 15)], 640, 480))
        # Fingers crossed: not the seal.
        self.assertFalse(joined_hands_from_landmarks([self.tilted(left, 30), self.tilted(right, -30)], 640, 480))

    def test_wrists_apart_are_not_joined(self):
        # Tips close but wrists apart: a "V" with the arms, not joined palms.
        left = self.tilted(self.hand(0.47), 20, pivot=(0.47 * 640, 0.34 * 480))
        right = self.tilted(self.hand(0.53), -20, pivot=(0.53 * 640, 0.34 * 480))
        self.assertFalse(joined_hands_from_landmarks([left, right], 640, 480))

    def test_rotated_joined_hands_still_recognized(self):
        hands = [self.hand(0.47), self.hand(0.53)]
        rotated = [[SimpleNamespace(x=point.y, y=1 - point.x) for point in hand]
                   for hand in hands]
        self.assertTrue(joined_hands_from_landmarks(rotated, 480, 640))

    def test_pressed_palms_are_the_palms_seal(self):
        self.assertEqual(hand_seal_from_landmarks([self.hand(0.47), self.hand(0.53)], 640, 480), "palms")

    def test_clasped_hands_from_a_real_photo(self):
        # Hand Landmarker output on a photo of two clasped hands (452x678), thumbs up, fingers wrapped
        # around the other hand. Each hand's fingertips sit 0.2 hand sizes from the other's knuckles.
        self.assertIsNotNone(hand_seal_from_landmarks(PHOTO_CLASP, 452, 678))

    @staticmethod
    def interlaced(center=0.5, reach=0.03, curl=0.0):
        """Own hands interlaced in front of the chest: wrists on the sides, knuckle rows side by side.

        reach: how far past the other hand's knuckles the fingertips go; curl moves them back to their own.
        """
        hands = []
        for side in (-1, 1):
            wrist_x = center + side * 0.12
            knuckle_x = center + side * 0.03
            tip_x = center - side * reach + side * curl
            points = [SimpleNamespace(x=(wrist_x + knuckle_x) / 2, y=0.6) for _ in range(21)]
            points[0] = SimpleNamespace(x=wrist_x, y=0.6)
            for index, offset in ((5, -0.05), (9, -0.017), (13, 0.017), (17, 0.05)):
                points[index] = SimpleNamespace(x=knuckle_x, y=0.6 + offset)
            for index, offset in ((8, -0.05), (12, -0.017), (16, 0.017), (20, 0.05)):
                points[index] = SimpleNamespace(x=tip_x, y=0.6 + offset)
            hands.append(points)
        return hands

    def test_interlaced_fingers_are_the_clasped_seal(self):
        self.assertEqual(hand_seal_from_landmarks(self.interlaced(), 640, 480), "clasped")
        self.assertEqual(hand_seal_from_landmarks(self.interlaced(center=0.3), 640, 480), "clasped")

    def test_fist_bump_is_not_a_clasp(self):
        # Fingers curled back into each palm: the tips stay closer to their own knuckles.
        self.assertIsNone(hand_seal_from_landmarks(self.interlaced(reach=0.0, curl=0.045), 640, 480))

    def test_fingers_on_the_back_of_the_other_hand_only_one_way_are_not_a_clasp(self):
        left, right = self.interlaced()
        open_right = self.interlaced(reach=-0.09)[1]  # Fingers stretched back, away from the other hand.
        self.assertIsNone(hand_seal_from_landmarks([left, open_right], 640, 480))

    def test_hands_far_apart_are_not_a_clasp(self):
        left = self.interlaced(center=0.3)[0]
        right = self.interlaced(center=0.7)[1]
        self.assertIsNone(hand_seal_from_landmarks([left, right], 640, 480))

    def test_debug_line_says_which_seal_matched(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(hand_landmarks=self.interlaced())
        recognizer = app.MultiViewHandLandmarker(primary, alternate)
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(recognizer.recognize_joined_hands(image, 100))
        self.assertIn("seal=clasped", app.detector_status(recognizer, True))

    def test_recognizer_uses_two_hands_in_the_current_frame(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[self.hand(0.47), self.hand(0.53)])
        recognizer = app.MultiViewHandLandmarker(primary, alternate)
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(recognizer.recognize_joined_hands(image, 100))
        self.assertEqual(recognizer.last_hand_count, 2)
        alternate.detect.assert_not_called()

    def test_hand_model_recovers_joined_hands_in_a_cropped_view(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[self.hand(0.47)])
        alternate.detect.side_effect = [
            SimpleNamespace(hand_landmarks=[]),
            SimpleNamespace(hand_landmarks=[self.hand(0.47), self.hand(0.53)]),
        ]
        recognizer = app.MultiViewHandLandmarker(primary, alternate)
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(recognizer.recognize_joined_hands(image, 100))
        self.assertEqual(recognizer.preferred_view, ("full", 3))
        self.assertEqual(recognizer.last_hand_count, 2)


class EffectTests(unittest.TestCase):
    def test_generated_asset_has_real_transparency(self):
        sprite = load_log()
        self.assertEqual(sprite.shape[2], 4)
        self.assertEqual(int(sprite[:, :, 3].min()), 0)
        self.assertEqual(int(sprite[:, :, 3].max()), 255)

    def test_smoke_animation_leaves_log_behind(self):
        frame = np.full((160, 240, 3), 130, np.uint8)
        background = np.full_like(frame, 35)
        mask = np.ones(frame.shape[:2], np.float32)
        sprite = load_log()
        anchor = (120, 85, 100)
        early = render_substitution(frame, background, mask, sprite, anchor, 0.04)
        middle = render_substitution(frame, background, mask, sprite, anchor, 0.55)
        final = render_substitution(frame, background, mask, sprite, anchor, EFFECT_SECONDS)
        self.assertTrue(np.any(early != frame))
        self.assertTrue(np.any(middle != frame))
        self.assertTrue(np.any(middle != background))
        self.assertTrue(np.any(final != background))
        np.testing.assert_array_equal(final[0, 0], background[0, 0])

    def test_log_anchor_tracks_person_region(self):
        mask = np.zeros((160, 240), np.float32)
        mask[20:130, 140:210] = 1.0
        x, y, size = substitution_anchor(mask)
        self.assertGreater(x, 140)
        self.assertGreater(size, 35)
        self.assertTrue(20 < y < 130)

    def test_log_falls_then_stays_on_floor(self):
        anchor = (120, 80, 90)
        ys = [falling_log_anchor(anchor, 240, age)[1] for age in (0.1, 0.3, 0.5, 0.7)]
        self.assertEqual(ys[0], anchor[1])
        self.assertGreater(ys[2], ys[1])
        self.assertGreater(ys[3], ys[2])
        floor = landed_log_anchor(anchor, 240)
        self.assertGreater(floor[1], ys[3])
        background = np.full((240, 240, 3), 35, np.uint8)
        output = render_substitution(background, background, np.zeros((240, 240), np.float32),
                                     load_log(), anchor, EFFECT_SECONDS)
        expected = background.copy()
        overlay_log(expected, load_log(), floor)
        np.testing.assert_array_equal(output, expected)


class NarutoVideoLoopTests(unittest.TestCase):
    def test_effect_is_sent_to_preview_and_virtual_camera(self):
        frames = [np.full((80, 120, 3), v, np.uint8) for v in (10, 40, 80, 120, 160, 200)]
        capture = Mock()
        # Hands joined from 4.3: the effect starts only after the hold, at 4.6.
        read, clock = video_clock(frames, (0, 4.1, 4.3, 4.6, 4.8, 6.3))
        capture.read.side_effect = read
        segmenter, recognizer = Mock(), Mock()
        recognizer.recognize_joined_hands.return_value = True
        sound = Mock()
        virtual = Mock(width=120, height=80)
        preview = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(mode="naruto")))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            app_mocks(mocks, segmenter, recognizer)
            mocks.enter_context(patch.object(app, "VanishSound", return_value=sound))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=virtual))
            mocks.enter_context(patch.object(app, "person_mask", return_value=np.ones((80, 120), np.float32)))
            mocks.enter_context(patch.object(app, "AsyncRecognizer", InlineRecognizer))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=clock))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: preview.append(frame.copy())))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=(0, 0, 0, 0, 0, ord("q"))))
            mocks.enter_context(patch.object(app.cv2, "putText", side_effect=AssertionError("Overlay vietato")))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
        self.assertEqual(len(preview), 6)
        for sent, shown in zip(virtual.send.call_args_list, preview):
            np.testing.assert_array_equal(sent.args[0], shown)
        np.testing.assert_array_equal(preview[2], frames[2])  # Still holding.
        self.assertTrue(np.any(preview[3] != frames[3]))
        self.assertTrue(np.any(preview[4] != frames[4]))
        self.assertTrue(np.any(preview[5] != frames[1]))  # The log stays.
        self.assertEqual(recognizer.recognize_joined_hands.call_count, 2)  # Start and confirmation of the hold.
        sound.play.assert_called_once()
        sound.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
