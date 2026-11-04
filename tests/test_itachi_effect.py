import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import mediapipe as mp
import numpy as np

from vanishcam import app
from vanishcam.control_panel import MODE_CARDS
from vanishcam.detectors import ChestHandsPoseDetector
from vanishcam.effects import itachi
from vanishcam.effects.ash import CHIN, hands_up_reference
from vanishcam.effects.goku import CHEEK_LEFT, CHEEK_RIGHT, FOREHEAD_TOP, NOSE_BRIDGE
from vanishcam.effects.itachi import (
    BODY_RELEASE_SECONDS, CROW_COUNT, EFFECT_SECONDS, SPRITE_HEIGHT, SPRITE_WIDTH, BloodTint, chest_hands_pose,
    make_flock, render_flock,
)
from vanishcam.effects.itachi_art import crow_sprites, sharingan
from vanishcam.modes import MODES, FrameInfo, ItachiMode
from vanishcam.shortcuts import Shortcuts
from tests.hand_fixtures import make_hand

WIDTH, HEIGHT = 640, 480


def face_landmarks():
    """A face in the upper middle of the frame: chin at y=0.35, about 77 px wide."""
    points = [SimpleNamespace(x=0.5, y=0.25) for _ in range(478)]
    points[FOREHEAD_TOP] = SimpleNamespace(x=0.5, y=0.12)
    points[NOSE_BRIDGE] = SimpleNamespace(x=0.5, y=0.22)
    points[CHEEK_LEFT] = SimpleNamespace(x=0.44, y=0.26)
    points[CHEEK_RIGHT] = SimpleNamespace(x=0.56, y=0.26)
    points[CHIN] = SimpleNamespace(x=0.5, y=0.35)
    return points


def hands_at(x, y=0.6, gap=0.06):
    """Two upright hands side by side around (x, y), their palms gap apart."""
    return [make_hand(x - gap / 2, y, 0.18), make_hand(x + gap / 2, y, 0.18)]


def scene(height=240, width=320):
    """An orange person in a blue room."""
    background = np.full((height, width, 3), (191, 157, 96), np.uint8)
    mask = np.zeros((height, width), np.float32)
    cv2.ellipse(mask, (width // 2, height // 2 - 5), (width // 5, int(height * 0.44)), 0, 0, 360, 1, -1)
    person = background.copy()
    person[mask > 0.5] = (55, 95, 145)
    return person, background, mask


class ChestHandsPoseTests(unittest.TestCase):
    def setUp(self):
        self.face = hands_up_reference(face_landmarks(), WIDTH, HEIGHT)

    def pose(self, hands, face="default"):
        return chest_hands_pose(hands, self.face if face == "default" else face, WIDTH, HEIGHT)

    def test_hands_touching_in_front_of_the_chest(self):
        self.assertTrue(self.pose(hands_at(0.5)))
        self.assertTrue(self.pose(hands_at(0.55, 0.75)))  # A little to the side, lower down.

    def test_one_hand_or_no_face_is_not_enough(self):
        self.assertFalse(self.pose(hands_at(0.5)[:1]))
        self.assertFalse(self.pose(hands_at(0.5), face=None))

    def test_hands_apart_do_not_count(self):
        self.assertFalse(self.pose(hands_at(0.5, gap=0.4)))

    def test_hands_at_the_face_or_off_to_the_side_are_not_at_the_chest(self):
        self.assertFalse(self.pose(hands_at(0.5, 0.25)))
        self.assertFalse(self.pose(hands_at(0.15)))

    def test_detector_retries_the_center_when_the_hands_merge(self):
        face, hands, fallback = Mock(), Mock(), Mock()
        face.detect_for_video.return_value = SimpleNamespace(face_landmarks=[face_landmarks()])
        # The tracking sees only one hand; the retry finds both in the crop.
        hands.detect_for_video.return_value = SimpleNamespace(hand_landmarks=hands_at(0.5)[:1])
        x1, y1, x2, y2 = ChestHandsPoseDetector.CROP

        def in_crop(hand):
            return [SimpleNamespace(x=(p.x - x1) / (x2 - x1), y=(p.y - y1) / (y2 - y1)) for p in hand]

        fallback.detect.return_value = SimpleNamespace(hand_landmarks=[in_crop(h) for h in hands_at(0.5)])
        detector = ChestHandsPoseDetector(face, hands, fallback)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((HEIGHT, WIDTH, 3), np.uint8))
        self.assertTrue(detector.recognize_pose(image, 100))
        self.assertEqual(fallback.detect.call_count, 1)
        # Not again until the retry interval has passed.
        detector.recognize_pose(image, 150)
        self.assertEqual(fallback.detect.call_count, 1)
        detector.close()
        fallback.close.assert_called_once()

    def test_detector_needs_no_retry_when_the_tracking_sees_both_hands(self):
        face, hands, fallback = Mock(), Mock(), Mock()
        face.detect_for_video.return_value = SimpleNamespace(face_landmarks=[face_landmarks()])
        hands.detect_for_video.return_value = SimpleNamespace(hand_landmarks=hands_at(0.5))
        detector = ChestHandsPoseDetector(face, hands, fallback)
        self.assertTrue(detector.recognize_pose(SimpleNamespace(width=WIDTH, height=HEIGHT), 0))
        fallback.detect.assert_not_called()


class CrowFlockTests(unittest.TestCase):
    def setUp(self):
        self.person, self.background, self.mask = scene()
        self.flock = make_flock(self.person, self.mask)
        self.tint = BloodTint()

    def test_crows_are_drawn_in_five_poses_inside_the_sprite(self):
        sprites = crow_sprites()
        self.assertEqual(len(sprites), 5)
        for sprite in sprites:
            self.assertEqual(sprite.shape, (SPRITE_HEIGHT, SPRITE_WIDTH, 4))
            # A dark bird, whole: nothing cut by the sprite's edges.
            self.assertLess(int(sprite[sprite[:, :, 3] > 200][:, :3].mean()), 80)
            for border in (sprite[0], sprite[-1], sprite[:, 0], sprite[:, -1]):
                self.assertLess(int(border[:, 3].max()), 40)
        # The wing beats: every pose differs from the next.
        for first, second in zip(sprites, sprites[1:]):
            self.assertFalse(np.array_equal(first[:, :, 3], second[:, :, 3]))

    def test_sharingan_is_two_red_eyes_on_nothing(self):
        eyes = sharingan(600)
        self.assertEqual(eyes.shape[1], 600)
        self.assertEqual(eyes.shape[2], 4)
        # No visible rectangle: the borders are transparent.
        for border in (eyes[0], eyes[-1], eyes[:, 0], eyes[:, -1]):
            self.assertTrue((border[:, 3] == 0).all())
        # Red irises around black pupils, one in each half.
        for center_x in (0.22, 0.78):
            x, y = round(600 * center_x), round(eyes.shape[0] * 0.42)
            iris = eyes[y - 30:y + 30, x - 45:x + 45]
            red = (iris[:, :, 2] > 150) & (iris[:, :, 1] < 90) & (iris[:, :, 3] > 200)
            self.assertGreater(int(red.sum()), 300)
            self.assertLess(int(eyes[y, x, :3].max()), 40)

    def test_crows_flying_left_are_mirrored(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        rightward = itachi.Crow(np.float32([100, 120]), np.float32([140, 120]), np.float32([180, 120]),
                                np.float32([220, 120]), 0.0, 2.0, 30.0, 0.0, 3.0, 1.0, False)
        leftward = itachi.Crow(np.float32([220, 120]), np.float32([180, 120]), np.float32([140, 120]),
                               np.float32([100, 120]), 0.0, 2.0, 30.0, 0.0, 3.0, 1.0, False)
        right, left = frame.copy(), frame.copy()
        itachi.draw_crow(right, self.flock, rightward, 1.0)
        itachi.draw_crow(left, self.flock, leftward, 1.0)
        # Both at the middle of the same line, one the mirror image of the other.
        self.assertTrue(right.any())
        self.assertLess(float(np.abs(right.astype(int) - left[:, ::-1].astype(int)).mean()), 1.0)

    def test_every_crow_leaves_from_the_body(self):
        self.assertEqual(len(self.flock.crows), CROW_COUNT)
        for crow in self.flock.crows:
            self.assertGreater(self.mask[int(crow.origin[1]), int(crow.origin[0])], 0.5)
        self.assertEqual(sum(crow.hero for crow in self.flock.crows), 4)

    def test_shoulders_then_head_then_the_torso_core(self):
        top = int(np.nonzero(self.mask.any(axis=1))[0][0])
        release = self.flock.release
        head = float(np.median(release[top:top + 30, 140:180]))
        core = float(np.median(release[110:150, 150:170]))
        shoulders = float(np.median(release[150:200, 100:120][self.mask[150:200, 100:120] > 0.5]))
        self.assertLess(shoulders, head)
        self.assertLess(head, core)
        self.assertLess(float(release[self.mask > 0.01].max()), BODY_RELEASE_SECONDS)

    def test_one_piece_of_body_per_crow(self):
        self.assertEqual(len(self.flock.shards), CROW_COUNT)
        at_home = 0
        for shard, crow in zip(self.flock.shards, self.flock.crows):
            height, width = shard.alpha.shape
            # The crow leaves from its own piece (the torn borders move a few off it),
            # and the piece holds the source pixels.
            at_home += (shard.left <= crow.origin[0] < shard.left + width
                        and shard.top <= crow.origin[1] < shard.top + height)
            inside = shard.alpha > 0.5
            np.testing.assert_array_equal(
                shard.color[inside], self.person[shard.top:shard.top + height, shard.left:shard.left + width][inside])
        self.assertGreater(at_home, CROW_COUNT * 0.9)
        # Together the pieces cover the whole body, once.
        covered = np.zeros(self.mask.shape, np.int32)
        for shard in self.flock.shards:
            height, width = shard.alpha.shape
            covered[shard.top:shard.top + height, shard.left:shard.left + width] += shard.alpha > 0
        self.assertTrue((covered[self.mask > 0.5] == 1).all())

    def test_a_torn_piece_is_dark_as_it_flies_into_its_crow(self):
        index = min(range(CROW_COUNT), key=lambda i: self.flock.crows[i].launch)
        crow = self.flock.crows[index]
        frame = self.background.copy()
        itachi.draw_shard(frame, self.flock.shards[index], crow, crow.launch + itachi.TEAR_SECONDS * 0.4)
        changed = np.any(frame != self.background, axis=2)
        self.assertTrue(changed.any())
        # Darker than the person was: ink, on its way to being a crow.
        self.assertLess(int(frame[changed].mean()), int(self.person[self.mask > 0.5].mean()))
        # Gone once the crow has taken over.
        later = self.background.copy()
        itachi.draw_shard(later, self.flock.shards[index], crow, crow.launch + itachi.TEAR_SECONDS)
        np.testing.assert_array_equal(later, self.background)

    def test_cracks_show_before_a_piece_tears_off(self):
        crow = max(self.flock.crows, key=lambda c: c.launch)
        before = render_flock(self.background, self.flock, self.tint, crow.launch - itachi.CRACK_SECONDS - 0.05)
        cracking = render_flock(self.background, self.flock, self.tint, crow.launch - 0.05)
        x, y = int(crow.origin[0]), int(crow.origin[1])
        # The body around the last crow's start goes darker just before it leaves.
        self.assertLess(int(cracking[y - 3:y + 4, x - 3:x + 4].mean()), int(before[y - 3:y + 4, x - 3:x + 4].mean()))

    def test_starts_on_the_person_and_ends_on_the_red_room(self):
        start = render_flock(self.background, self.flock, self.tint, 0.0)
        np.testing.assert_array_equal(start, self.person)
        middle = render_flock(self.background, self.flock, self.tint, BODY_RELEASE_SECONDS * 0.55)
        self.assertGreater(np.count_nonzero(np.any(middle != self.person, axis=2)), 100)
        final = render_flock(self.background, self.flock, self.tint, EFFECT_SECONDS)
        np.testing.assert_array_equal(final, render_flock(self.background, self.flock, self.tint, EFFECT_SECONDS + 1))
        # Red ink: more red, less blue than the room, and no person left.
        self.assertGreater(int(final[..., 2].mean()), int(self.background[..., 2].mean()))
        self.assertLess(int(final[..., 0].mean()), int(self.background[..., 0].mean()))
        np.testing.assert_array_equal(final, self.tint.apply(self.background))

    def test_sharingan_shows_only_early_on(self):
        def without_eyes(age):
            with patch.object(itachi, "draw_sharingan"):
                return render_flock(self.background, self.flock, self.tint, age)

        early = render_flock(self.background, self.flock, self.tint, 1.0)
        self.assertFalse(np.array_equal(early[:20], without_eyes(1.0)[:20]))
        np.testing.assert_array_equal(render_flock(self.background, self.flock, self.tint, 3.0), without_eyes(3.0))

    def test_appearing_plays_backwards_to_the_person(self):
        self.assertTrue(np.array_equal(render_flock(self.background, self.flock, self.tint, 0.0, appearing=True),
                                       render_flock(self.background, self.flock, self.tint, EFFECT_SECONDS)))
        np.testing.assert_array_equal(
            render_flock(self.background, self.flock, self.tint, EFFECT_SECONDS, appearing=True), self.person)

    def test_nobody_in_the_frame_makes_no_flock(self):
        self.assertIsNone(make_flock(self.person, np.zeros_like(self.mask)))

    def test_tint_follows_a_new_room(self):
        brighter = np.full_like(self.background, 200)
        self.assertFalse(np.array_equal(self.tint.apply(self.background), self.tint.apply(brighter)))


class ItachiModeTests(unittest.TestCase):
    def setUp(self):
        self.person, self.background, self.mask = scene()
        self.segment = Mock(return_value=self.mask)
        self.mode = ItachiMode(self.segment)

    def frame(self, now, frame=None, background=None):
        info = FrameInfo(self.person if frame is None else frame,
                         self.background if background is None else background, now, int(now * 1000))
        return self.mode.render(info)

    def vanish(self):
        self.mode.on_detection(True, 0.0)
        self.mode.on_detection(True, 0.25)
        self.assertEqual(self.mode.phase, "vanishing")
        for now in (0.25, 1.0, 2.0, 3.0):
            self.frame(now)
        gone = self.frame(0.25 + EFFECT_SECONDS)
        self.assertEqual(self.mode.phase, "vanished")
        return gone

    def test_registered_last_with_key_six_and_the_chest_detector(self):
        self.assertIs(MODES["itachi"], ItachiMode)
        self.assertEqual(MODE_CARDS[-1].key, "itachi")
        self.assertEqual(Shortcuts().with_enabled(True).action_for(ord("6")), ("mode", "itachi"))
        recognizer, image = Mock(), object()
        self.assertIs(ItachiMode.detect(recognizer, image, 0), recognizer.recognize_pose.return_value)
        self.assertTrue(ItachiMode.manual_keys)
        self.assertTrue(ItachiMode.detect_while_vanished)

    def test_gesture_vanishes_into_the_red_room_with_one_capture(self):
        gone = self.vanish()
        self.segment.assert_called_once()  # Silhouette captured once.
        self.assertGreater(int(gone[..., 2].mean()), int(self.background[..., 2].mean()))
        np.testing.assert_array_equal(self.frame(5.0), gone)
        # The red follows a refreshed room.
        brighter = np.full_like(self.background, 200)
        self.mode.background_changed()
        self.assertFalse(np.array_equal(self.frame(5.1, background=brighter), gone))

    def test_gesture_again_brings_the_crows_back(self):
        self.vanish()
        self.mode.on_detection(False, 5.0)
        self.mode.on_detection(False, 5.4)
        self.mode.on_detection(True, 5.5)
        self.mode.on_detection(True, 5.75)
        self.assertEqual(self.mode.phase, "appearing")
        # The person back in the frame is captured again, at a new place.
        moved = np.roll(self.person, 30, axis=1)
        self.segment.return_value = np.roll(self.mask, 30, axis=1)
        self.frame(5.8, frame=moved)
        self.assertEqual(self.segment.call_count, 2)
        self.frame(7.0, frame=moved)
        self.assertEqual(self.mode.phase, "appearing")
        back = self.frame(5.8 + EFFECT_SECONDS, frame=moved)
        self.assertEqual(self.mode.phase, "ready")
        np.testing.assert_array_equal(back, moved)

    def test_holding_the_hands_after_vanishing_does_not_bring_you_back(self):
        self.vanish()
        self.mode.on_detection(True, 5.0)
        self.mode.on_detection(True, 6.0)
        self.assertEqual(self.mode.phase, "vanished")

    def test_r_while_invisible_also_animates_the_return(self):
        self.vanish()
        self.mode.reappear()
        self.assertEqual(self.mode.phase, "appearing")

    def test_nobody_to_bring_back_stays_invisible(self):
        gone = self.vanish()
        self.mode.reappear()
        self.segment.return_value = np.zeros_like(self.mask)
        np.testing.assert_array_equal(self.frame(6.0), gone)
        self.assertEqual(self.mode.phase, "vanished")

    def test_nobody_to_vanish_stays_ready(self):
        self.segment.return_value = np.zeros_like(self.mask)
        self.mode.vanish(0.0)
        np.testing.assert_array_equal(self.frame(0.1), self.person)
        self.assertEqual(self.mode.phase, "ready")

    def test_new_background_shows_the_person_right_away(self):
        self.vanish()
        self.mode.reset()
        self.assertEqual(self.mode.phase, "ready")
        np.testing.assert_array_equal(self.frame(6.0), self.person)

    def test_self_test_draws_the_crows_and_the_eyes(self):
        with patch.object(app, "crow_sprites", wraps=app.crow_sprites) as crows, \
                patch.object(app, "sharingan", wraps=app.sharingan) as eyes, \
                patch.object(app, "create_segmenter"), patch.object(app, "create_recognizer"), \
                patch.object(app, "create_meme_detector"), \
                patch.object(Path, "is_file", return_value=True):  # The downloaded models may be missing.
            self.assertEqual(app.self_test(), 0)
        crows.assert_called_once()
        eyes.assert_called_once()


if __name__ == "__main__":
    unittest.main()
