import unittest
from unittest.mock import Mock

import numpy as np

from modes import FrameInfo, ParticlesMode
from particles_effect import DEPART_SECONDS, EFFECT_SECONDS, make_particles, render_particles


class ParticlesEffectTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((120, 160, 3), np.uint8)
        self.frame[:, :, 1] = 180  # Persona verde su sfondo grigio.
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
        # Il vento soffia verso destra: frammenti oltre il bordo destro della sagoma.
        self.assertTrue(np.any(middle[:, 105:] != self.background[:, 105:]))
        np.testing.assert_array_equal(middle[:, :50], self.background[:, :50])
        np.testing.assert_array_equal(
            render_particles(self.background, self.cloud, EFFECT_SECONDS), self.background
        )

    def test_empty_mask_is_just_background(self):
        cloud = make_particles(self.frame, np.zeros_like(self.mask))
        np.testing.assert_array_equal(render_particles(self.background, cloud, 0.5), self.background)


class ParticlesModeTests(unittest.TestCase):
    def test_v_sign_captures_silhouette_once_and_ends_on_background(self):
        frame = np.full((60, 80, 3), 40, np.uint8)
        background = np.full((60, 80, 3), 200, np.uint8)
        mask = np.zeros((60, 80), np.float32)
        mask[10:60, 30:50] = 1.0
        segment = Mock(return_value=mask)
        mode = ParticlesMode(segment)
        mode.on_detection(("Victory", 0.9), 1.0)
        self.assertEqual(mode.phase, "vanishing")
        for now in (1.0, 1.5, 2.0, 2.5):
            mode.render(FrameInfo(frame, background, now, int(now * 1000)))
        segment.assert_called_once()
        np.testing.assert_array_equal(mode.render(FrameInfo(frame, background, 3.2, 3200)), background)
        self.assertEqual(mode.phase, "vanished")


if __name__ == "__main__":
    unittest.main()
