import argparse
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from vanishcam import app
from vanishcam import detectors, segmentation
from vanishcam.segmentation import PersonSegmenter
from vanishcam.memes import Memes
from vanishcam.shortcuts import Shortcuts
from vanishcam.user_data import Settings


def arguments(**overrides):
    values = dict(
        camera=0, width=960, height=540, fps=30.0, fullscreen=False,
        no_mirror=True, virtual_camera=None, virtual_backend=None, virtual_device=None,
        gesture_debug=False, mode="normal", static_background=False, call_audio=False, background_image=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class InlineRecognizer:
    """Like AsyncRecognizer but on the same thread: deterministic results in tests."""

    busy = False

    def __init__(self, detect, close):
        self.detect, self.close_detector, self.result = detect, close, None

    def submit(self, rgb, timestamp_ms):
        self.result = (self.detect(rgb, timestamp_ms), timestamp_ms)

    def take_result(self):
        result, self.result = self.result, None
        return result

    def close(self):
        self.close_detector()


class FakePanel:
    """A panel without a window: records the views and returns one action per frame."""

    def __init__(self, actions=()):
        self.actions = list(actions)
        self.views, self.frames, self.real_backgrounds = [], [], []
        self.fullscreen, self.detached = [], []
        self.last = None
        self.closed = False
        self.capturing = None

    def show(self, view, video=None):
        # The video shown beside the panel: what the user sees. While no new frame comes the
        # app shows the same one again, like ControlPanel nothing new is drawn then.
        self.views.append(view)
        if video is not None and (not self.frames or video is not self.last):
            self.frames.append(video.copy())
        self.last = video

    def set_fullscreen(self, enabled):
        self.fullscreen.append(enabled)

    def set_detached(self, enabled):
        self.detached.append(enabled)

    def poll(self):
        # An action can be a function of the panel: useful when it depends on what already happened.
        actions = self.actions.pop(0) if self.actions else []
        return [action(self) if callable(action) else action for action in actions]

    def set_real_background(self, bgr):
        self.real_backgrounds.append(bgr.copy())

    def close(self):
        pass


# The key scripts of the tests use the old keys: shortcuts on, with Q to quit.
TEST_SHORTCUTS = Shortcuts().with_enabled(True).rebind("quit", ord("q"))


def app_mocks(mocks, segmenter, recognizer, actions=(), cameras=(), shortcuts=TEST_SHORTCUTS):
    """Fake models, a fake panel and preferences in a temporary folder."""
    mocks.enter_context(patch.object(app, "create_segmenter", return_value=segmenter))
    mocks.enter_context(patch.object(app, "create_recognizer", return_value=recognizer))
    panel = FakePanel(actions)
    mocks.enter_context(patch.object(app, "ControlPanel", return_value=panel))
    # No real webcam listing in tests: the camera number from the arguments is used.
    mocks.enter_context(patch.object(app, "list_cameras", return_value=cameras))
    # Frames read on the video thread, in order: the scripted captures stay deterministic.
    mocks.enter_context(patch.object(app, "FrameReader", lambda capture: capture))
    folder = mocks.enter_context(tempfile.TemporaryDirectory())
    user_data = app.UserData(Path(folder))
    user_data.save(Settings(shortcuts=shortcuts))
    mocks.enter_context(patch.object(app.UserData, "default", return_value=user_data))
    return panel, user_data


def video_clock(frames, times):
    """Fake capture.read and perf_counter: each frame read brings its own instant."""
    clock = [times[0]]
    queue = list(zip(frames, times))

    def read():
        frame, clock[0] = queue.pop(0)
        return True, frame

    return read, (lambda: clock[0])


class VirtualCameraTests(unittest.TestCase):
    def test_bgr_format_and_actual_frame_dimensions(self):
        camera = Mock(device="OBS Virtual Camera")
        module = SimpleNamespace(Camera=Mock(return_value=camera), PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIs(app.open_virtual_camera(arguments(), 640, 480), camera)
        module.Camera.assert_called_once_with(
            width=640, height=480, fps=30.0, fmt="BGR", backend=None, device=None,
        )

    def test_unavailable_driver_auto_falls_back_but_explicit_mode_fails(self):
        module = SimpleNamespace(Camera=Mock(side_effect=RuntimeError("driver missing")),
                                 PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module), patch("builtins.print"):
            self.assertIsNone(app.open_virtual_camera(arguments(), 640, 480))
            for options in (dict(virtual_camera=True), dict(virtual_backend="obs")):
                with self.assertRaisesRegex(RuntimeError, "driver missing"):
                    app.open_virtual_camera(arguments(**options), 640, 480)

    def test_disabled_camera_does_not_open_device(self):
        module = SimpleNamespace(Camera=Mock())
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIsNone(app.open_virtual_camera(arguments(virtual_camera=False), 640, 480))
        module.Camera.assert_not_called()

    def test_send_preserves_colors_and_handles_noncontiguous_frames(self):
        camera = Mock(width=40, height=24)
        frame = np.zeros((24, 40, 3), np.uint8)
        frame[:] = (10, 70, 200)  # BGR: do not swap red and blue.
        app.send_virtual_frame(camera, frame[:, ::-1])
        sent = camera.send.call_args.args[0]
        self.assertTrue(sent.flags.c_contiguous)
        np.testing.assert_array_equal(sent, frame)

    def test_send_resizes_if_input_resolution_changes(self):
        camera = Mock(width=40, height=24)
        app.send_virtual_frame(camera, np.full((48, 80, 3), 42, np.uint8))
        sent = camera.send.call_args.args[0]
        self.assertEqual(sent.shape, (24, 40, 3))
        self.assertTrue((sent == 42).all())


class VideoLoopTests(unittest.TestCase):
    def run_mock_video(self, send_error=None, gesture=("No_Hand", 0.0)):
        capture = Mock()
        frames = [np.full((24, 40, 3), color, np.uint8) for color in (10, 40, 80, 120, 160)]
        read, clock = video_clock(frames, (0, 4.1, 4.3, 5.0, 8.2))
        capture.read.side_effect = read
        segmenter, recognizer = Mock(), Mock()
        camera = Mock(width=40, height=24)
        camera.send.side_effect = send_error
        displayed = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments()))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            panel, _ = app_mocks(mocks, segmenter, recognizer)
            opening = mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            mocks.enter_context(patch.object(segmentation, "person_mask", return_value=np.ones((24, 40), np.float32)))
            mocks.enter_context(patch.object(detectors, "recognize_gesture", return_value=gesture))
            mocks.enter_context(patch.object(app, "AsyncRecognizer", InlineRecognizer))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=clock))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            destroy = mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            displayed = panel.frames
            keys = (0, 0, ord("v") if gesture[0] != "Victory" else 0, 0, ord("q"))
            mocks.enter_context(patch.object(app.cv2, "waitKeyEx", side_effect=keys))
            mocks.enter_context(patch.object(app.cv2, "putText", side_effect=AssertionError("No text may be drawn over the video")))
            mocks.enter_context(patch("builtins.print"))
            if send_error is None:
                self.assertEqual(app.run(), 0)
            else:
                with self.assertRaisesRegex(RuntimeError, "send failed"):
                    app.run()
            opening.assert_called_once()
            self.assertEqual(opening.call_args.args[1:], (40, 24))
            capture.release.assert_called_once()
            segmenter.close.assert_called_once()
            recognizer.close.assert_called_once()
            camera.close.assert_called_once()
            destroy.assert_called_once()
        return frames, camera, displayed

    def test_background_image_replaces_room_without_calibration(self):
        capture = Mock()
        frames = [np.full((24, 40, 3), 200, np.uint8) for _ in range(3)]
        read, clock = video_clock(frames, (0, 0.1, 0.2))
        capture.read.side_effect = read
        mask = np.zeros((24, 40), np.float32)
        mask[:, :20] = 1.0  # Person on the left, room on the right.
        displayed = []
        with ExitStack() as mocks:
            args = arguments(virtual_camera=False, background_image="beach.png")
            mocks.enter_context(patch.object(app, "parse_args", return_value=args))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            panel, _ = app_mocks(mocks, Mock(), Mock())
            loaded = mocks.enter_context(patch.object(
                app, "load_background_image", return_value=np.full((48, 80, 3), 30, np.uint8),
            ))
            mocks.enter_context(patch.object(segmentation, "person_mask", return_value=mask))
            mocks.enter_context(patch.object(detectors, "recognize_gesture", return_value=("No_Hand", 0.0)))
            mocks.enter_context(patch.object(app, "AsyncRecognizer", InlineRecognizer))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=clock))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            displayed = panel.frames
            mocks.enter_context(patch.object(app.cv2, "waitKeyEx", side_effect=(0, ord("v"), ord("q"))))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
        loaded.assert_called_once_with(Path("beach.png"))
        # From the very first frame: the person in front of the picture, no calibration.
        self.assertTrue((displayed[0][:, :20] == 200).all())
        self.assertTrue((displayed[0][:, 20:] == 30).all())
        # After V the person starts dissolving into the picture, not into the room.
        self.assertTrue((displayed[2][:, 20:] == 30).all())
        self.assertLess(int(displayed[2][:, :20].min()), 200)

    def test_preview_and_virtual_output_are_clean_in_all_phases(self):
        frames, camera, displayed = self.run_mock_video()
        self.assertEqual(len(displayed), 5)
        for sent, preview in zip(camera.send.call_args_list, displayed):
            np.testing.assert_array_equal(sent.args[0], preview)
        np.testing.assert_array_equal(displayed[0], frames[0])  # Calibration.
        np.testing.assert_array_equal(displayed[2], frames[2])  # Ready.
        np.testing.assert_array_equal(displayed[4], frames[1])  # Gone: background only.

    def test_first_gesture_starts_dissolve_in_preview_and_virtual_output(self):
        frames, camera, displayed = self.run_mock_video(gesture=("Victory", 0.9))
        np.testing.assert_array_equal(camera.send.call_args_list[2].args[0], displayed[2])
        self.assertTrue(np.any(displayed[3] != frames[3]))  # Animation running.
        self.assertTrue(np.any(displayed[3] != frames[1]))  # Not gone yet.
        np.testing.assert_array_equal(displayed[4], frames[1])  # Animation finished.

    def test_all_devices_close_on_send_error(self):
        self.run_mock_video(send_error=RuntimeError("send failed"))


class AsyncRecognizerTests(unittest.TestCase):
    def wait_result(self, recognizer):
        for _ in range(200):
            result = recognizer.take_result()
            if result is not None:
                return result
            time.sleep(0.01)
        self.fail("Nessun risultato dal thread")

    def test_detects_off_the_video_thread_and_closes_detector(self):
        threads, closed = [], []

        def detect(image, timestamp_ms):
            threads.append(threading.current_thread().name)
            return image.width, timestamp_ms

        recognizer = app.AsyncRecognizer(detect, lambda: closed.append(True))
        try:
            recognizer.submit(np.zeros((6, 9, 3), np.uint8), 120)
            self.assertEqual(self.wait_result(recognizer), ((9, 120), 120))
            self.assertIsNone(recognizer.take_result())  # Each result is read once.
            self.assertEqual(threads, ["gesture"])
        finally:
            recognizer.close()
        self.assertEqual(closed, [True])

    def test_detector_error_reaches_the_video_loop(self):
        def detect(image, timestamp_ms):
            raise ValueError("broken model")

        recognizer = app.AsyncRecognizer(detect, lambda: None)
        try:
            recognizer.submit(np.zeros((4, 4, 3), np.uint8), 1)
            for _ in range(200):
                if not recognizer.busy:
                    break
                time.sleep(0.01)
            with self.assertRaisesRegex(RuntimeError, "broken model"):
                recognizer.take_result()
        finally:
            recognizer.close()


class PersonSegmenterTests(unittest.TestCase):
    def test_one_segmentation_per_timestamp(self):
        masks = [np.zeros((4, 6), np.float32), np.ones((4, 6), np.float32)]
        with patch.object(segmentation, "person_mask", side_effect=masks) as segment:
            segmenter = PersonSegmenter(Mock())
            frame = np.zeros((4, 6, 3), np.uint8)
            first = segmenter(frame, 100)
            self.assertIs(segmenter(frame, 100), first)
            self.assertIsNot(segmenter(frame, 133), first)
        self.assertEqual(segment.call_count, 2)
        self.assertEqual(segment.call_args.args[2:], (133, 6, 4))


class ShortcutKeysTests(unittest.TestCase):
    def run_keys(self, keys, shortcuts, fullscreen=False):
        """The app on a picture background, one key press per frame; the window closes after the last."""
        keys = list(keys)
        capture = Mock()
        frames = [np.full((24, 40, 3), 200, np.uint8) for _ in keys]
        read, clock = video_clock(frames, [0.1 * index for index in range(len(keys))])
        capture.read.side_effect = read
        mask = np.zeros((24, 40), np.float32)
        mask[:, :20] = 1.0  # Person on the left.
        with ExitStack() as mocks:
            args = arguments(virtual_camera=False, background_image="beach.png", fullscreen=fullscreen)
            mocks.enter_context(patch.object(app, "parse_args", return_value=args))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            panel, _ = app_mocks(mocks, Mock(), Mock(), shortcuts=shortcuts)
            mocks.enter_context(patch.object(
                app, "load_background_image", return_value=np.full((48, 80, 3), 30, np.uint8),
            ))
            mocks.enter_context(patch.object(segmentation, "person_mask", return_value=mask))
            mocks.enter_context(patch.object(detectors, "recognize_gesture", return_value=("No_Hand", 0.0)))
            mocks.enter_context(patch.object(app, "AsyncRecognizer", InlineRecognizer))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=clock))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
                mocks.enter_context(patch.object(app.cv2, name))

            def wait_key(delay):
                key = keys.pop(0)
                panel.closed = not keys
                return key

            mocks.enter_context(patch.object(app.cv2, "waitKeyEx", side_effect=wait_key))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
            self.recognizers = app.create_recognizer.call_count
        self.assertEqual(keys, [])  # Every key was read: nothing quit early.
        return panel

    def test_typing_does_nothing_while_shortcuts_are_off(self):
        panel = self.run_keys([ord(c) for c in "vou sair rapidinho 12345"], Shortcuts())
        # Nobody vanished, no fullscreen, no detach, no other effect.
        self.assertTrue(all((frame[:, :20] == 200).all() for frame in panel.frames))
        self.assertEqual(panel.fullscreen, [])
        self.assertEqual(panel.detached, [])
        self.assertEqual(self.recognizers, 1)

    def test_escape_leaves_fullscreen_and_never_quits(self):
        for escape in (27, 0xFF1B):
            with self.subTest(escape):
                panel = self.run_keys([ord("f"), escape, escape, 0, 0], Shortcuts().with_enabled(True))
                self.assertEqual(panel.fullscreen, [True, False])

    def test_escape_leaves_fullscreen_with_shortcuts_off(self):
        # The default: no shortcuts, started fullscreen, no panel to click. ESC is the way out.
        for escape in (27, 0xFF1B):
            with self.subTest(escape):
                panel = self.run_keys([0, escape, 0], Shortcuts(), fullscreen=True)
                self.assertEqual(panel.fullscreen[-1], False)



class MemeLoopTests(unittest.TestCase):
    STEP = 0.125  # Exact in binary: every frame is one meme detection interval or more apart.

    def setUp(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.white_png = folder / "white.png"
        cv2.imwrite(str(self.white_png), np.full((2, 2, 3), 255, np.uint8))

    def run_memes(self, memes, results, mode="normal", keys=None):
        """The app on a picture background; the meme detector answers results, one per detection."""
        frames = len(results) + 1
        capture = Mock()
        images = [np.full((24, 40, 3), 100, np.uint8) for _ in range(frames)]
        read, clock = video_clock(images, [self.STEP * index for index in range(frames)])
        capture.read.side_effect = read
        answers = list(results)
        detector = Mock()
        detector.recognize.side_effect = lambda image, timestamp_ms: answers.pop(0) if answers else None
        recognizer = Mock()
        recognizer.recognize_joined_hands.return_value = False
        camera = Mock(width=40, height=24)
        keys = list(keys or [0] * (frames - 1)) + [ord("q")]
        with ExitStack() as mocks:
            args = arguments(background_image="beach.png", mode=mode)
            mocks.enter_context(patch.object(app, "parse_args", return_value=args))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            panel, user_data = app_mocks(mocks, Mock(), recognizer)
            user_data.save(Settings(shortcuts=TEST_SHORTCUTS, memes=memes))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            mocks.enter_context(patch.object(
                app, "load_background_image", return_value=np.full((48, 80, 3), 30, np.uint8),
            ))
            mocks.enter_context(patch.object(app, "VanishSound", return_value=Mock()))
            mask = np.zeros((24, 40), np.float32)
            mask[:, 10:30] = 1.0  # Person in the middle.
            mocks.enter_context(patch.object(segmentation, "person_mask", return_value=mask))
            mocks.enter_context(patch.object(detectors, "recognize_gesture", return_value=("No_Hand", 0.0)))
            mocks.enter_context(patch.object(app, "AsyncRecognizer", InlineRecognizer))
            self.creating = mocks.enter_context(patch.object(app, "create_meme_detector", return_value=detector))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=clock))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "waitKeyEx", side_effect=keys))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
        self.detector = detector
        return panel.frames, [call.args[0] for call in camera.send.call_args_list]

    def test_meme_pops_over_the_video_and_the_call(self):
        # Held from the first detection: fired at the third, 0.25 s later; still on at the end.
        memes = Memes().with_gesture("thumbs_up", image=self.white_png, on=True)
        frames, sent = self.run_memes(memes, results=["thumbs_up"] * 6 + [None] * 4)
        self.assertTrue((frames[0][12, 20] == 100).all())  # Before: the person.
        self.assertTrue((frames[-1][12, 20] == 255).all())  # The white picture in the middle.
        self.assertTrue((frames[-1][2, 2] != 255).all())  # Only in the middle.
        np.testing.assert_array_equal(sent[-1], frames[-1])  # The call gets the same picture.

    def test_no_detector_while_nothing_is_enabled(self):
        self.run_memes(Memes(), results=[None] * 4)
        self.creating.assert_not_called()
        self.run_memes(Memes().with_gesture("calm", image=self.white_png, on=True).with_enabled(False), [None] * 4)
        self.creating.assert_not_called()

    def test_meme_never_changes_the_effect_frame(self):
        # Naruto vanishes with V at the second frame and keeps its frozen frame; a meme pops up
        # from 3 s and is gone by 6.5 s: the frame after it is the frame before it.
        memes = Memes().with_gesture("thumbs_up", image=self.white_png, on=True)
        results = [None] * 24 + ["thumbs_up"] * 4 + [None] * 32
        keys = [0, ord("v")] + [0] * (len(results) - 2)
        _, sent = self.run_memes(memes, results, mode="naruto", keys=keys)  # Sent: one per frame.
        before, during, after = sent[22], sent[36], sent[-1]
        self.assertFalse((during == before).all())  # The meme was on screen...
        np.testing.assert_array_equal(after, before)  # ...and left nothing behind.
        self.assertEqual(self.detector.close.call_count, 1)  # Closed on exit.


if __name__ == "__main__":
    unittest.main()
