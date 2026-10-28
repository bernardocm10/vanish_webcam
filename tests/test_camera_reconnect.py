import queue
import threading
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from vanishcam import app
from vanishcam import camera as camera_module
from vanishcam import detectors, segmentation
from vanishcam.app import choose_camera
from vanishcam.camera import (
    CameraFeed, CameraInfo, FrameReader, device_instance_id, dim_for_card, list_cameras, reconnecting_card,
    restart_driver,
)
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

    def test_reopen_switches_to_another_webcam_in_the_background(self):
        first = np.full((HEIGHT, WIDTH, 3), 1, np.uint8)
        second = np.full((HEIGHT * 2, WIDTH * 2, 3), 2, np.uint8)
        old = scripted_capture([(True, first, 0.0), (True, first, 0.1)], self.clock)
        new = Mock()
        new.read.return_value = (True, second)
        feed = CameraFeed(Mock(return_value=old))
        feed.read()
        self.clock[0] = 0.1
        feed.read()
        self.assertEqual(feed.frame_size, (WIDTH, HEIGHT))
        self.assertAlmostEqual(feed.fps, 10.0)
        feed.reopen(Mock(return_value=new))
        old.release.assert_called_once()
        self.assertTrue(feed.opening and feed.reconnecting)
        feed.reopener.thread.join(timeout=2.0)
        self.assertIs(feed.read(), second)
        self.assertFalse(feed.opening or feed.reconnecting)
        self.assertEqual(feed.frame_size, (WIDTH * 2, HEIGHT * 2))
        self.assertIsNone(feed.fps)  # Measured again for the new webcam.
        feed.release()

    def test_choosing_again_while_opening_lets_the_first_one_go(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        started = scripted_capture([(True, frame, 0.0)], self.clock)
        opening, release_slow = threading.Event(), threading.Event()
        slow = Mock()
        slow.read.return_value = (True, frame)

        def open_slowly():
            opening.set()
            release_slow.wait(2.0)
            return slow

        feed = CameraFeed(Mock(return_value=started))
        feed.reopen(open_slowly)
        self.assertTrue(opening.wait(2.0))
        abandoned = feed.reopener
        quick = Mock()
        quick.read.return_value = (True, frame)
        feed.reopen(Mock(return_value=quick))
        release_slow.set()
        abandoned.thread.join(timeout=2.0)
        slow.release.assert_called_once()  # Opened too late: let go, not kept busy.
        feed.reopener.thread.join(timeout=2.0)
        feed.read()
        self.assertIs(feed.capture, quick)
        feed.release()

    def test_settings_dialog_only_on_windows_with_a_live_webcam(self):
        capture = Mock()
        capture.set.return_value = True
        feed = CameraFeed(Mock(return_value=capture))
        with patch.object(camera_module.platform, "system", return_value="Windows"):
            self.assertTrue(feed.open_settings())
        capture.set.assert_called_once_with(camera_module.cv2.CAP_PROP_SETTINGS, 1)
        with patch.object(camera_module.platform, "system", return_value="Linux"):
            self.assertFalse(feed.open_settings())


class FrameReaderTests(unittest.TestCase):
    def capture(self, reads):
        """A capture whose reads come from a queue; it blocks like a slow webcam until fed."""
        feed = queue.Queue()
        capture = Mock()
        capture.read.side_effect = lambda: feed.get(timeout=2.0)
        for read in reads:
            feed.put(read)
        return capture, feed

    def wait_for(self, reader, expected):
        for _ in range(200):
            result = reader.read()
            if result[0] == expected[0] and (result[1] is None) == (expected[1] is None):
                return result
            time.sleep(0.005)
        self.fail(f"never read {expected}")

    def test_each_frame_once_and_waiting_is_not_a_failure(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        capture, feed = self.capture([(True, frame)])
        reader = FrameReader(capture)
        self.assertIs(self.wait_for(reader, (True, frame))[1], frame)
        # The next image is still on its way (the read blocks): alive, nothing new.
        self.assertEqual(reader.read(), (True, None))
        feed.put((False, None))
        self.wait_for(reader, (False, None))
        feed.put((True, frame))
        reader.release()
        capture.release.assert_called_once()  # By the reading thread, after its last read.

    def test_a_webcam_that_stops_answering_counts_as_failed(self):
        capture, feed = self.capture([])
        reader = FrameReader(capture, stall_seconds=0.05)
        self.assertEqual(reader.read(), (True, None))
        time.sleep(0.1)
        self.assertEqual(reader.read(), (False, None))
        feed.put((False, None))
        reader.release()

    def test_feed_waits_without_counting_failures(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        capture = Mock()
        capture.read.side_effect = [(True, frame)] + [(True, None)] * 50
        feed = CameraFeed(Mock(return_value=capture), lost_after=0.0)
        self.assertIs(feed.read(), frame)
        for _ in range(50):
            self.assertIsNone(feed.read())
        self.assertFalse(feed.reconnecting)
        capture.release.assert_not_called()

    def test_feed_wraps_every_capture_after_its_first_image(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        first, second = Mock(), Mock()
        second.read.return_value = (True, frame)
        wrapped = []

        def wrap(capture):
            wrapped.append(capture)
            return capture

        feed = CameraFeed(Mock(return_value=first), wrap=wrap)
        feed.reopen(Mock(return_value=second))
        feed.reopener.thread.join(timeout=2.0)
        feed.read()
        self.assertEqual(wrapped, [first, second])
        feed.release()


class DriverRestartTests(unittest.TestCase):
    PATH = "\\\\?\\usb#vid_090c&pid_b371&mi_00#6&341aedb9&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global"
    INSTANCE = "USB\\VID_090C&PID_B371&MI_00\\6&341AEDB9&0&0000"

    def test_device_path_becomes_the_pnputil_instance_id(self):
        self.assertEqual(device_instance_id(self.PATH), self.INSTANCE)
        for path in (None, "", "/dev/video0"):
            self.assertIsNone(device_instance_id(path))

    def restart(self, code):
        with patch.object(camera_module.platform, "system", return_value="Windows"), \
                patch.object(camera_module.subprocess, "run", return_value=SimpleNamespace(returncode=code)) as run:
            result = restart_driver(self.INSTANCE)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "powershell")
        self.assertIn("/restart-device", command[-1])
        self.assertIn(f'"{self.INSTANCE}"', command[-1])
        self.assertIn("-Verb RunAs", command[-1])  # pnputil needs the permission prompt.
        return result

    def test_outcomes(self):
        self.assertIsNone(self.restart(0))
        self.assertEqual(self.restart(camera_module.DRIVER_CANCELLED), t("camera.driver_cancelled"))
        self.assertEqual(self.restart(camera_module.DRIVER_REBOOT), t("camera.driver_reboot"))
        self.assertEqual(self.restart(5), t("camera.driver_exit", code=5))
        with patch.object(camera_module.platform, "system", return_value="Linux"):
            self.assertEqual(restart_driver(self.INSTANCE), t("camera.driver_windows_only"))

    def test_driver_restarts_after_the_webcam_is_released_and_before_it_opens(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        capture, reopened = Mock(), Mock()
        reopened.read.return_value = (True, frame)
        events, started, finish = [], threading.Event(), threading.Event()
        capture.release.side_effect = lambda: events.append("released")

        def restart():
            events.append("restart")
            started.set()
            finish.wait(2.0)

        def open_again():
            events.append("open")
            return reopened

        feed = CameraFeed(Mock(return_value=capture))
        feed.reopen(open_again, prepare=restart)
        self.assertTrue(started.wait(2.0))
        self.assertTrue(feed.restarting)
        finish.set()
        feed.reopener.thread.join(timeout=2.0)
        self.assertFalse(feed.restarting)
        self.assertIs(feed.read(), frame)
        self.assertEqual(events, ["released", "restart", "open"])
        feed.release()


class CameraListTests(unittest.TestCase):
    def enumerated(self, *cameras):
        found = [SimpleNamespace(index=index, name=name) for index, name in cameras]
        module = SimpleNamespace(enumerate_cameras=Mock(return_value=found))
        with patch.dict("sys.modules", cv2_enumerate_cameras=module):
            return list_cameras()

    def test_virtual_cameras_and_repeats_are_left_out(self):
        cameras = self.enumerated((0, "USB 2.0 PC Cam"), (1, "OBS Virtual Camera"), (0, "USB 2.0 PC Cam"),
                                  (10, "VanishCam"), (2, "Logitech C920"))
        self.assertEqual(cameras, (CameraInfo(0, "USB 2.0 PC Cam"), CameraInfo(2, "Logitech C920")))

    def test_listing_failure_is_an_empty_list(self):
        with patch.dict("sys.modules", cv2_enumerate_cameras=None):
            self.assertEqual(list_cameras(), ())

    def test_startup_camera_choice(self):
        cameras = (CameraInfo(0, "USB 2.0 PC Cam"), CameraInfo(2, "Logitech C920"))
        self.assertEqual(choose_camera(5, "Logitech C920", cameras), 5)  # --camera wins.
        self.assertEqual(choose_camera(None, "Logitech C920", cameras), 2)  # Found by name.
        self.assertEqual(choose_camera(None, "Unplugged", cameras), 0)
        self.assertEqual(choose_camera(None, None, ()), 0)


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
            displayed = panel.frames
            mocks.enter_context(patch.object(app.cv2, "waitKeyEx", side_effect=wait_key))
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
