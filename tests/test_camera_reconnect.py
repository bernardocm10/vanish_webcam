import threading
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import numpy as np

from vanishcam import app
from vanishcam import detectors, segmentation
from vanishcam.camera import CameraFeed, dim_for_card, reconnecting_card
from vanishcam.i18n import t
from .test_video_output import InlineRecognizer, app_mocks, arguments

HEIGHT, WIDTH = 90, 160


def scripted_capture(script, clock):
    """A capture whose reads follow script: (ok, frame, instant); then it keeps failing."""
    capture = Mock()

    def read():
        if not script:
            return False, None
        ok, frame, clock[0] = script.pop(0)
        return ok, frame

    capture.read.side_effect = read
    return capture


class CameraFeedTests(unittest.TestCase):
    def setUp(self):
        self.clock = [0.0]
        patcher = patch.object(app.time, "perf_counter", side_effect=lambda: self.clock[0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_one_dropped_frame_does_not_reconnect(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        capture = scripted_capture([(True, frame, 0.0), (False, None, 0.1), (True, frame, 0.2)], self.clock)
        opener = Mock(return_value=capture)
        feed = CameraFeed(opener, lost_after=0.5)
        self.assertIs(feed.read(), frame)
        self.assertIsNone(feed.read())
        self.assertFalse(feed.reconnecting)
        self.assertIs(feed.read(), frame)
        opener.assert_called_once()
        capture.release.assert_not_called()

    def test_reopens_after_the_webcam_is_gone_for_a_while(self):
        frame = np.full((HEIGHT, WIDTH, 3), 7, np.uint8)
        lost = scripted_capture([(False, None, 1.0), (False, None, 1.6)], self.clock)
        back = Mock()
        back.read.return_value = (True, frame)
        opener = Mock(side_effect=[lost, RuntimeError("busy"), back])
        feed = CameraFeed(opener, lost_after=0.5, retry_seconds=0.01)
        self.assertIsNone(feed.read())
        self.assertIsNone(feed.read())
        self.assertTrue(feed.reconnecting)
        self.assertEqual(feed.lost_since, 1.6)
        lost.release.assert_called_once()
        feed.reopener.thread.join(timeout=2.0)
        self.assertIs(feed.read(), frame)
        self.assertFalse(feed.reconnecting)
        self.assertIs(feed.capture, back)
        self.assertEqual(opener.call_count, 3)  # Startup, one failed retry, then back.
        feed.release()
        back.release.assert_called_once()

    def test_quitting_while_reconnecting_stops_the_retries(self):
        lost = scripted_capture([(False, None, 0.0), (False, None, 1.0)], self.clock)
        retried = threading.Event()

        def opener():
            if lost.read.call_count == 0:
                return lost  # Startup.
            retried.set()
            raise RuntimeError("unplugged")

        feed = CameraFeed(opener, lost_after=0.5, retry_seconds=0.01)
        feed.read()
        feed.read()
        self.assertTrue(retried.wait(2.0))
        reopener = feed.reopener
        feed.release()
        self.assertFalse(reopener.thread.is_alive())
        self.assertIsNone(feed.reopener)


class ReconnectingCardTests(unittest.TestCase):
    def test_card_keeps_the_frame_size_and_draws_the_status(self):
        base = dim_for_card(np.full((HEIGHT * 4, WIDTH * 4, 3), 200, np.uint8), WIDTH * 4, HEIGHT * 4)
        self.assertLess(int(base.max()), 200)  # Dimmed.
        card = reconnecting_card(base, "RECONECTANDO A CÂMERA", "Webcam 0 · tentando há 3 s", True)
        self.assertEqual(card.shape, base.shape)
        self.assertGreater(int(card.max()), int(base.max()) + 60)  # Light text on the dimmed frame.

    def test_card_without_any_previous_frame(self):
        base = dim_for_card(None, WIDTH, HEIGHT)
        self.assertEqual(base.shape, (HEIGHT, WIDTH, 3))


class ReconnectLoopTests(unittest.TestCase):
    def test_call_sees_the_card_while_the_webcam_is_gone_then_the_video_again(self):
        clock = [0.0]
        before = np.full((HEIGHT, WIDTH, 3), 200, np.uint8)
        after = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
        lost = scripted_capture(
            [(True, before, 0.0), (True, before, 0.1), (False, None, 0.2), (False, None, 0.8)], clock,
        )
        back = Mock()
        back.read.return_value = (True, after)
        displayed, sent = [], []
        camera = Mock(width=WIDTH, height=HEIGHT)
        camera.send.side_effect = lambda frame: sent.append(frame.copy())

        def wait_key(delay):
            # Quit once the video is back, with a safety limit if it never comes back.
            done = (displayed and (displayed[-1] == 90).all()) or len(displayed) > 500
            return ord("q") if done else 0

        with ExitStack() as mocks:
            args = arguments(background_image="beach.png")
            mocks.enter_context(patch.object(app, "parse_args", return_value=args))
            mocks.enter_context(patch.object(app, "download_models"))
            opening = mocks.enter_context(patch.object(app, "open_camera", side_effect=[lost, back]))
            panel, _ = app_mocks(mocks, Mock(), Mock())
            mocks.enter_context(patch.object(
                app, "load_background_image", return_value=np.full((HEIGHT, WIDTH, 3), 30, np.uint8),
            ))
            mocks.enter_context(patch.object(segmentation, "person_mask", return_value=np.ones((HEIGHT, WIDTH), np.float32)))
            mocks.enter_context(patch.object(detectors, "recognize_gesture", return_value=("No_Hand", 0.0)))
            mocks.enter_context(patch.object(app, "AsyncRecognizer", InlineRecognizer))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=lambda: clock[0]))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: displayed.append(frame.copy())))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=wait_key))
            printed = mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)

        self.assertTrue((displayed[0] == 200).all())
        cards = [frame for frame in displayed if not (frame == 200).all() and not (frame == 90).all()]
        self.assertTrue(cards, "the reconnecting card was never shown")
        self.assertTrue((displayed[-1] == 90).all())
        # The call keeps receiving frames, the card included: it does not freeze on the last image.
        self.assertEqual(len(sent), len(displayed))
        self.assertTrue(any(np.array_equal(frame, cards[0]) for frame in sent))
        statuses = [view.status for view in panel.views]
        self.assertIn(t("status.reconnecting"), statuses)
        self.assertNotEqual(statuses[-1], t("status.reconnecting"))
        lines = [str(call.args[0]) for call in printed.call_args_list if call.args]
        self.assertTrue(any(t("camera.lost") in line for line in lines))
        self.assertTrue(any(t("camera.back") in line for line in lines))
        self.assertEqual(opening.call_count, 2)
        lost.release.assert_called_once()
        back.release.assert_called_once()
        camera.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
