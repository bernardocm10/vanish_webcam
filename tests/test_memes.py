import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from vanishcam.memes import SHOW_SECONDS, MemeGesture, MemeLayer, Memes, scale_at

# The thumbs up held from 0 s fires at the 0.3 s detection (0.25 s hold).
FIRED = 0.3


class MemeLayerTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.image = self.folder / "joinha.png"
        cv2.imwrite(str(self.image), np.full((100, 200, 3), 255, np.uint8))  # 2:1, white.
        self.memes = Memes().with_gesture("thumbs_up", image=self.image, on=True)
        self.frame = np.zeros((480, 640, 3), np.uint8)

    def fire(self, layer, start=0.0, memes=None):
        for now in (start, start + 0.1, start + 0.2, start + 0.3):
            layer.on_detection("thumbs_up", now, memes or self.memes)

    def white(self, image):
        spots = np.argwhere(image.max(axis=2) > 200)
        height, width = np.ptp(spots, axis=0) + 1
        return spots, width, height

    def test_pops_in_settles_and_leaves(self):
        layer = MemeLayer()
        self.fire(layer)
        self.assertTrue(layer.showing)
        spots, width, height = self.white(layer.render(self.frame, FIRED + 1.0))
        self.assertAlmostEqual(width, 0.65 * 640, delta=3)  # Width-bound: 2:1 into 416x312.
        self.assertAlmostEqual(height, width / 2, delta=3)
        self.assertAlmostEqual(spots[:, 1].mean(), 320, delta=2)  # Centered.
        self.assertAlmostEqual(spots[:, 0].mean(), 240, delta=2)
        self.assertFalse((self.frame != 0).any())  # The input frame is untouched.
        self.assertGreater(scale_at(0.18), 1.05)
        self.assertLess(scale_at(0.05), 0.6)
        self.assertIs(layer.render(self.frame, FIRED + SHOW_SECONDS + 0.01), self.frame)
        self.assertFalse(layer.showing)

    def test_a_held_gesture_fires_once_until_released(self):
        layer = MemeLayer()
        self.fire(layer)
        layer.render(self.frame, FIRED + SHOW_SECONDS + 0.01)  # Gone.
        for now in (3.5, 3.6, 3.7, 3.8):  # Still holding the pose.
            layer.on_detection("thumbs_up", now, self.memes)
        self.assertFalse(layer.showing)
        for now in (3.9, 4.3):  # Released...
            layer.on_detection(None, now, self.memes)
        self.fire(layer, 4.4)  # ...and made again.
        self.assertTrue(layer.showing)

    def test_one_at_a_time(self):
        memes = self.memes.with_gesture("calm", image=self.image, on=True)
        layer = MemeLayer()
        self.fire(layer, memes=memes)
        for now in (0.4, 0.5, 0.6, 0.7):
            layer.on_detection("calm", now, memes)
        self.assertEqual(layer.current, "thumbs_up")

    def test_gesture_without_an_active_image_does_nothing(self):
        layer = MemeLayer()
        self.fire(layer, memes=Memes())
        self.assertFalse(layer.showing)

    def test_image_fits_the_box_for_any_aspect(self):
        for shape in ((50, 2000, 3), (2000, 50, 3), (10, 10, 4)):
            with self.subTest(shape):
                image = self.folder / f"meme{shape[0]}x{shape[1]}.png"
                cv2.imwrite(str(image), np.full(shape, 255, np.uint8))
                layer = MemeLayer()
                self.fire(layer, memes=Memes().with_gesture("thumbs_up", image=image, on=True))
                _, width, height = self.white(layer.render(self.frame, FIRED + 1.0))
                self.assertLessEqual(width, 0.65 * 640 + 1)
                self.assertLessEqual(height, 0.65 * 480 + 1)

    def test_transparent_pixels_show_the_video(self):
        rgba = np.zeros((100, 100, 4), np.uint8)
        rgba[:50] = (255, 255, 255, 255)  # Top half opaque, bottom half transparent.
        cv2.imwrite(str(self.image), rgba)
        layer = MemeLayer()
        self.fire(layer)
        out = layer.render(self.frame, FIRED + 1.0)
        self.assertTrue((out[300:340, 300:340] == 0).all())  # Below the middle: transparent.
        self.assertTrue((out[200:230, 300:340] == 255).all())

    def test_unreadable_image_is_skipped(self):
        self.image.write_bytes(b"not an image")
        layer = MemeLayer()
        with patch("builtins.print") as printed:
            self.fire(layer)
            self.assertIs(layer.render(self.frame, FIRED + 1.0), self.frame)
            self.fire(layer, 5.0)
            layer.render(self.frame, 6.0)
        self.assertEqual(printed.call_count, 1)  # Said once, not on every frame.

    def test_reads_pictures_where_opencv_cannot(self):
        # cv2.imread fails on Windows paths with accents (C:\\Users\\João\\...).
        layer = MemeLayer()
        self.fire(layer)
        with patch("vanishcam.memes.cv2.imread", return_value=None):
            out = layer.render(self.frame, FIRED + 1.0)
        self.assertTrue((out[240, 320] == 255).all())

    def test_phone_photos_are_turned_upright(self):
        photo = self.folder / "phone.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6  # Orientation: shot sideways, to be turned 90 degrees.
        Image.new("RGB", (200, 100), "white").save(photo, exif=exif)
        layer = MemeLayer()
        self.fire(layer, memes=Memes().with_gesture("thumbs_up", image=photo, on=True))
        _, width, height = self.white(layer.render(self.frame, FIRED + 1.0))
        self.assertGreater(height, width)  # Upright: taller than wide.

    def test_clear(self):
        layer = MemeLayer()
        self.fire(layer)
        layer.clear()
        self.assertIs(layer.render(self.frame, FIRED + 1.0), self.frame)


class MemesConfigTests(unittest.TestCase):
    def test_defaults_and_active(self):
        memes = Memes()
        self.assertTrue(memes.enabled)
        self.assertEqual(memes.active(), frozenset())
        self.assertFalse(memes.with_gesture("calm", on=True).gesture("calm").on)  # No image: stays off.
        on = memes.with_gesture("calm", image=Path("a.png"), on=True)
        self.assertEqual(on.active(), frozenset({"calm"}))
        self.assertEqual(on.with_enabled(False).active(), frozenset())
        self.assertEqual(on.with_gesture("calm", on=False).active(), frozenset())

    def test_json_round_trip_relative_to_the_folder(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        image = root / "memes" / "a.png"
        image.parent.mkdir()
        cv2.imwrite(str(image), np.zeros((4, 4, 3), np.uint8))
        memes = Memes().with_gesture("calm", image=image, on=True).with_enabled(False)
        data = memes.to_json(root)
        self.assertEqual(data["gestures"]["calm"], {"image": "memes/a.png", "on": True})
        self.assertEqual(Memes.from_json(data, root), memes)
        self.assertEqual(Memes.from_json(7, root), Memes())
        self.assertEqual(Memes.from_json({"gestures": {"calm": "x"}}, root), Memes())
        image.unlink()
        self.assertEqual(Memes.from_json(data, root).gesture("calm"), MemeGesture())


if __name__ == "__main__":
    unittest.main()
