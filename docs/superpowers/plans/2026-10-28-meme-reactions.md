# Meme Reactions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A gesture held in front of the camera pops one of the owner's images over the middle of the video for about 3 seconds, on top of any effect, in the call and in the preview. The gestures, their images, and the on/off switches are set on a new MEMES tab.

**Architecture:**
- **Poses.** The eight pose functions of `prototypes/run3.py` are ported, in English, into `vanishcam/meme_poses.py` on top of a shared `vanishcam/hand_geometry.py`.
- **Layer.** `vanishcam/memes.py` holds the saved configuration (`Memes`) and the on-screen layer (`MemeLayer`): hold, pop in, stay, pop out, one at a time.
- **Detection.** A `MemePoseDetector` (face and hands) runs in its own `AsyncRecognizer`, next to the effect's, only while the layer has something enabled.
- **Panel.** It gets a MEMES tab between LIVE and SETTINGS.
- **Storage.** Images are copied into a `memes/` folder next to `backgrounds/`.

**Tech Stack:** Python 3.12, MediaPipe face and hand landmarkers, OpenCV, Pillow, unittest.

**Spec:** `docs/superpowers/specs/2026-10-28-registry-shortcuts-memes-design.md`, section "Phase 3: meme reactions".

## Global Constraints

- **Image size and position:** fit inside 65% of the frame's width and 65% of its height, aspect ratio kept, centered on the video. PNG transparency is respected. No border, no label, no tracking boxes.
- **Timing:**
  - pop in: about 0.28 s, growing from nothing with a slight overshoot (about 110%) before settling at 100%;
  - stay: until 3.0 s after the start;
  - pop out: 0.25 s, shrinking and fading.
- **One at a time.** While an image is on screen, gestures do nothing. The same gesture must be released before it can fire again (`HoldTracker` does this).
- **Always-on layer.** It works over every effect and every phase (ready, vanishing, vanished), but not while the background is being captured. It never changes cached effect frames: it draws on a copy.
- **Precedence** when two enabled gestures match: thumbs_up, hands_behind_head, timeout, calm, hands_on_face, prayer, pointing_at_self, hearing. Disabled gestures are not evaluated.
- **No bundled images.** Every gesture starts with no image and off. Choosing an image switches that gesture on, and a gesture without an image cannot be switched on. A saved image that has gone missing makes the gesture "no image, off".
- **Master switch:** on by default. Nothing is detected while it is off or while no gesture is on.
- **Collision warning, no auto-off:** `prayer` against Naruto, `hands_behind_head` against Ash.
- **Panel:** keeps its width and height. The tabs read LIVE, MEMES, SETTINGS. Every text goes through `t()` in EN and PT.
- **Code, docs, commits:** in English; commit messages are one subject line.
- **Tests:** run with `.venv/bin/python -m unittest`. The baseline on `main` is 214 tests, all passing.

## Review Focus

- **A held pose:** must fire once, not loop. Pinned by `test_a_held_gesture_fires_once_until_released` (Task 3).
- **An image file deleted or unreadable after it was chosen:** must not crash the loop. The gesture becomes "no image" when loading, and a broken file is skipped at runtime. Pinned by `test_unreadable_image_is_skipped` (Task 3) and `test_missing_image_turns_the_gesture_off` (Task 4).
- **The effect's frozen frame while vanished** (`NarutoMode.vanished_frame` and similar): the meme must not be burnt into it, so the effect is clean again after the meme leaves. Pinned by `test_meme_never_changes_the_effect_frame` (Task 6).
- **A very wide or very tall image:** must stay inside 65%/65% and inside the frame. Pinned by `test_image_fits_the_box_for_any_aspect` (Task 3).
- **Turning the master switch off while a meme is on screen:** the image goes away at once and detection stops. Pinned by `test_master_off_clears_and_stops_detection` (Task 6).

---

### Task 1: Shared hand geometry

**Files:**
- Create: `vanishcam/hand_geometry.py`
- Modify: `vanishcam/effects/ash.py` (`extended_fingers` uses the shared helper)
- Test: `tests/test_hand_geometry.py`, plus `tests/hand_fixtures.py` (a builder used again in Task 2)

**Interfaces:**
- Produces, in `vanishcam/hand_geometry.py`:
  - `WRIST = 0`, `PALM = (0, 5, 9, 13, 17)`, `FINGERTIPS = (8, 12, 16, 20)`
  - `extended_finger_states(points: np.ndarray, ratio: float = 1.18) -> list[bool]` for index, middle, ring, and little finger. A finger is extended when the distance from its tip to the wrist is at least `ratio` times the distance from its PIP joint to the wrist. `points` is a `(21, 2)` array in any unit.
  - `extended_finger_count(points, ratio=1.18) -> int`
  - `points_of(landmarks) -> np.ndarray`: normalized `(21, 2)` from MediaPipe landmarks.
  - `face_box(face_landmarks) -> tuple[float, float, float, float]`: normalized `(x1, y1, x2, y2)`.
  - `hand_features(landmarks) -> dict` with the run3 keys `landmarks`, `center`, `wrist_y`, `fingertip_y`, and `bbox`.
- In `effects/ash.py`, `extended_fingers(points)` returns `extended_finger_count(points, ratio=1.15)`. Its threshold stays exactly as today.
- In `tests/hand_fixtures.py`: `make_hand(cx, cy, size, extended=(True, True, True, True), thumb="side", angle=0.0) -> list[SimpleNamespace]` returns 21 normalized landmarks with `.x`/`.y`.
  - `thumb` is one of `"up"`, `"side"`, or `"folded"`.
  - `angle` rotates the hand in degrees, with 0 meaning fingers up.
  - `size` is the wrist-to-middle-fingertip length when extended.
  - Also: `make_face(x1, y1, x2, y2)` returns 478 landmarks spread over the box (corners included), enough for `face_box`.

- [ ] **Step 1: Write the failing tests**

```python
class HandGeometryTests(unittest.TestCase):
    def test_extended_fingers_follow_the_fixture(self):
        points = points_of(make_hand(0.5, 0.5, 0.2, extended=(True, False, True, False)))
        self.assertEqual(extended_finger_states(points), [True, False, True, False])
        self.assertEqual(extended_finger_count(points_of(make_hand(0.5, 0.5, 0.2))), 4)

    def test_face_box_and_features(self):
        self.assertEqual(face_box(make_face(0.4, 0.2, 0.6, 0.5)), (0.4, 0.2, 0.6, 0.5))
        features = hand_features(make_hand(0.5, 0.6, 0.2))
        self.assertLess(features["fingertip_y"], features["wrist_y"])  # Fingers up.

    def test_ash_keeps_its_threshold(self):
        points = points_of(make_hand(0.5, 0.5, 0.2)) * (640, 480)
        self.assertEqual(ash.extended_fingers(points), extended_finger_count(points, ratio=1.15))
```

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_hand_geometry -v`
Expected: ERROR, `No module named 'vanishcam.hand_geometry'`.

- [ ] **Step 3: Implement** the module, the fixtures, and the ash change.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 217 tests`, `OK`. The existing Ash tests are unchanged and pass.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/hand_geometry.py vanishcam/effects/ash.py tests/test_hand_geometry.py tests/hand_fixtures.py
git commit -m "Share the hand geometry helpers between the effects and the memes"
```

---

### Task 2: The eight meme poses

**Files:**
- Create: `vanishcam/meme_poses.py`
- Test: `tests/test_meme_poses.py`

**Interfaces:**
- Consumes: `hand_geometry` (Task 1) and `tests/hand_fixtures.py`.
- Produces:
  - `GESTURES: tuple[str, ...]`, in precedence order: `("thumbs_up", "hands_behind_head", "timeout", "calm", "hands_on_face", "prayer", "pointing_at_self", "hearing")`.
  - `POSES: dict[str, Callable[[list, tuple | None], tuple[bool, float]]]`, mapping each name to its ported function. The run3 names translate as follows:

    | run3 function | name |
    | --- | --- |
    | `thumbs_up_pose` | `thumbs_up` |
    | `hands_behind_head_pose` | `hands_behind_head` |
    | `timeout_hands_pose` | `timeout` |
    | `calma_hand_pose` | `calm` |
    | `hands_on_face_pose` | `hands_on_face` |
    | `prayer_hands_pose` | `prayer` |
    | `pointing_at_self_pose` | `pointing_at_self` |
    | `hearing_hand_pose` | `hearing` |
  - `match_gesture(hands: list, face: tuple | None, enabled: frozenset[str]) -> str | None` returns the first enabled gesture in `GESTURES` order whose pose matches. It only calls the poses that are enabled.
- **Porting rules:**
  - Thresholds, logic, and the `(matched, score)` return stay exactly as in run3 ([prototypes/run3.py:108-701](../../../prototypes/run3.py)).
  - Comments and docstrings are translated to English.
  - The finger helpers come from `hand_geometry`.
  - `yuta_hand_pose` is not ported.

- [ ] **Step 1: Write the failing tests.** There is one positive and at least one negative case per pose, built with `make_hand` and `make_face`. The face box `(0.4, 0.2, 0.6, 0.5)` is a face in the middle of the frame.

```python
FACE = (0.4, 0.2, 0.6, 0.5)

class MemePoseTests(unittest.TestCase):
    def test_thumbs_up(self):
        fist_thumb_up = make_hand(0.5, 0.75, 0.16, extended=(False,) * 4, thumb="up")
        self.assertTrue(POSES["thumbs_up"]([fist_thumb_up], FACE)[0])
        self.assertFalse(POSES["thumbs_up"]([make_hand(0.5, 0.75, 0.16)], FACE)[0])  # Open hand.

    def test_calm(self): ...              # Open hand, fingers together, in front of the chest; not spread fingers.
    def test_hearing(self): ...           # Open hand just outside one side of the face at ear height; not in front of it.
    def test_hands_behind_head(self): ... # Two hands at both sides of the head; not one hand.
    def test_prayer(self): ...            # Two hands overlapping with fingers folded or interlaced; not two open hands apart.
    def test_timeout(self): ...           # One hand horizontal, one vertical, touching, in front of the chest; not two parallel.
    def test_pointing_at_self(self): ...  # Only the index finger, below the chin, pointing inwards; not an open hand.
    def test_hands_on_face(self): ...     # Two open upright hands over the face; not one hand.

    def test_precedence_and_disabled_gestures(self):
        fist_thumb_up = make_hand(0.5, 0.75, 0.16, extended=(False,) * 4, thumb="up")
        self.assertEqual(match_gesture([fist_thumb_up], FACE, frozenset(GESTURES)), "thumbs_up")
        self.assertIsNone(match_gesture([fist_thumb_up], FACE, frozenset(GESTURES) - {"thumbs_up"}))
        self.assertIsNone(match_gesture([], None, frozenset(GESTURES)))
```

Each `...` above is a real test with the positive and negative case its comment names, using concrete `make_hand` arguments. Tune the fixture arguments, never the ported thresholds, until the positives pass.

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_meme_poses -v`
Expected: ERROR, `No module named 'vanishcam.meme_poses'`.

- [ ] **Step 3: Port the poses** following the rules above.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 226 tests`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/meme_poses.py tests/test_meme_poses.py
git commit -m "Port the eight meme poses from the run3 prototype"
```

---

### Task 3: The meme layer

**Files:**
- Create: `vanishcam/memes.py`
- Test: `tests/test_memes.py`

**Interfaces:**
- Consumes: `GESTURES` (Task 2), and `HoldTracker` from `effects/goku.py`.
- Produces:
  - Constants: `SHOW_SECONDS = 3.0`, `POP_IN_SECONDS = 0.28`, `POP_OUT_SECONDS = 0.25`, `BOX = 0.65`, `HOLD_SECONDS = 0.25`, `PRAYER_HOLD_SECONDS = 0.1`, `DETECT_INTERVAL = 0.1`.
  - `CONFLICTS: dict[str, str] = {"prayer": "naruto", "hands_behind_head": "ash"}`, mapping a gesture to the effect it collides with.
  - `@dataclass(frozen=True) class MemeGesture` with `image: Path | None = None` and `on: bool = False`.
  - `@dataclass(frozen=True) class Memes`:
    - fields `enabled: bool = True` and `gestures: tuple[tuple[str, MemeGesture], ...]`, defaulting to every name in `GESTURES` with `MemeGesture()`;
    - `gesture(name) -> MemeGesture`;
    - `active() -> frozenset[str]`: the names that are on and have an image, or empty when `enabled` is off;
    - `with_enabled(bool) -> Memes`;
    - `with_gesture(name, image=..., on=...) -> Memes`: `on=True` without an image is ignored;
    - `to_json(root: Path) -> dict`, storing image paths relative to `root`;
    - `from_json(data, root: Path)`, a classmethod: a missing file gives `MemeGesture()`, and malformed data gives the defaults.
  - `scale_at(age: float) -> float`: 0 at 0; about 1.1 at about 0.18 s; 1.0 from `POP_IN_SECONDS` until `SHOW_SECONDS - POP_OUT_SECONDS`; then it shrinks to 0.6 at `SHOW_SECONDS`.
  - `opacity_at(age: float) -> float`: 1 until the pop out, then down to 0 at `SHOW_SECONDS`.
  - `class MemeLayer`:
    - `__init__(self)`;
    - `showing -> bool`, a property;
    - `on_detection(self, gesture: str | None, now: float, memes: Memes) -> None`:
      - one `HoldTracker` per gesture, with `PRAYER_HOLD_SECONDS` for prayer and `HOLD_SECONDS` for the rest;
      - every tracker is updated with whether its gesture is the one detected, so the others release;
      - a tracker that fires while nothing is showing starts that gesture's image;
    - `render(self, frame: np.ndarray, now: float) -> np.ndarray`:
      - returns `frame` itself when nothing is showing;
      - otherwise a new array with the image composited at the center, scaled by `scale_at` and blended by `opacity_at` times its alpha;
      - the image keeps its aspect and fits the `BOX`×`BOX` box of the frame, and the scaled image is clipped to the frame;
      - once `SHOW_SECONDS` has passed, it clears itself;
    - `clear(self) -> None`;
    - images are loaded with `cv2.IMREAD_UNCHANGED` (BGR or BGRA) and cached by path and frame size;
    - an image that cannot be read is skipped: nothing shows, and it prints `t("memes.unreadable", name=...)` once.

- [ ] **Step 1: Write the failing tests**

```python
class MemeLayerTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.image = self.folder / "joinha.png"
        cv2.imwrite(str(self.image), np.full((100, 200, 3), 255, np.uint8))  # 2:1, white.
        self.memes = Memes().with_gesture("thumbs_up", image=self.image, on=True)
        self.frame = np.zeros((480, 640, 3), np.uint8)

    def fire(self, layer, start=0.0):
        for now in (start, start + 0.1, start + 0.2, start + 0.3):
            layer.on_detection("thumbs_up", now, self.memes)

    def test_pops_in_settles_and_leaves(self):
        layer = MemeLayer()
        self.fire(layer)
        self.assertTrue(layer.showing)
        start = 0.25  # Fired at the 0.25 s hold.
        settled = layer.render(self.frame, start + 1.0)
        white = np.argwhere(settled.max(axis=2) > 200)
        height, width = np.ptp(white, axis=0) + 1
        self.assertAlmostEqual(width, 0.65 * 640, delta=3)  # Width-bound: 2:1 into 416x312.
        self.assertAlmostEqual(height, width / 2, delta=3)
        self.assertAlmostEqual(white[:, 1].mean(), 320, delta=2)  # Centered.
        self.assertFalse((self.frame != 0).any())  # The input frame is untouched.
        self.assertGreater(scale_at(0.18), 1.05)
        self.assertLess(scale_at(0.05), 0.6)
        self.assertIs(layer.render(self.frame, start + SHOW_SECONDS + 0.01), self.frame)
        self.assertFalse(layer.showing)

    def test_a_held_gesture_fires_once_until_released(self):
        layer = MemeLayer()
        self.fire(layer)
        layer.render(self.frame, 0.25 + SHOW_SECONDS + 0.01)  # Gone.
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
        self.fire(layer)
        for now in (0.4, 0.5, 0.6, 0.7):
            layer.on_detection("calm", now, memes)
        self.assertEqual(layer.current, "thumbs_up")

    def test_image_fits_the_box_for_any_aspect(self):
        for shape in ((50, 2000, 3), (2000, 50, 3), (10, 10, 4)):
            with self.subTest(shape):
                cv2.imwrite(str(self.image), np.full(shape, 255, np.uint8))
                layer = MemeLayer()
                self.fire(layer)
                out = layer.render(self.frame, 1.0)
                white = np.argwhere(out.max(axis=2) > 200)
                height, width = np.ptp(white, axis=0) + 1
                self.assertLessEqual(width, 0.65 * 640 + 1)
                self.assertLessEqual(height, 0.65 * 480 + 1)

    def test_transparent_pixels_show_the_video(self):
        rgba = np.zeros((100, 100, 4), np.uint8)
        rgba[:50] = (255, 255, 255, 255)  # Top half opaque, bottom half transparent.
        cv2.imwrite(str(self.image), rgba)
        layer = MemeLayer()
        self.fire(layer)
        out = layer.render(self.frame, 1.0)
        self.assertTrue((out[300:340, 300:340] == 0).all())  # Below the middle: transparent.
        self.assertTrue((out[200:230, 300:340] == 255).all())

    def test_unreadable_image_is_skipped(self):
        self.image.write_bytes(b"not an image")
        layer = MemeLayer()
        with patch("builtins.print"):
            self.fire(layer)
            self.assertIs(layer.render(self.frame, 1.0), self.frame)


class MemesConfigTests(unittest.TestCase):
    def test_defaults_and_active(self):
        memes = Memes()
        self.assertTrue(memes.enabled)
        self.assertEqual(memes.active(), frozenset())
        self.assertFalse(memes.with_gesture("calm", on=True).gesture("calm").on)  # No image: stays off.

    def test_json_round_trip_relative_to_the_folder(self):
        ...  # with_gesture("calm", image=root/"memes"/"a.png", on=True) -> to_json(root) stores "memes/a.png";
             # from_json(that, root) equals the original; from_json(7, root) == Memes().
```

`MemeLayer.current: str | None` is the gesture on screen.

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_memes -v`
Expected: ERROR, `No module named 'vanishcam.memes'`.

- [ ] **Step 3: Implement** `vanishcam/memes.py`, and add `memes.unreadable` to `i18n.py` in both languages.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 234 tests`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/memes.py vanishcam/i18n.py tests/test_memes.py
git commit -m "Add the meme layer: hold, pop in, stay and pop out, one at a time"
```

---

### Task 4: Memes are saved with the preferences, and their images are copied

**Files:**
- Modify: `vanishcam/user_data.py`
- Modify: `vanishcam/app.py` (`save_settings` and the `memes` local)
- Test: `tests/test_control_panel.py` (`UserDataTests`)

**Interfaces:**
- Consumes: `Memes.to_json` and `Memes.from_json` (Task 3).
- Produces:
  - `Settings.memes: Memes = field(default_factory=Memes)` as the sixth field.
  - `UserData.memes_dir` is `root / "memes"`. `save` writes `"memes": settings.memes.to_json(self.root)`, and `load` reads `Memes.from_json(data.get("memes"), self.root)`.
  - `UserData.import_meme(gesture: str, source: Path) -> Path` copies the image to `memes/<gesture>-<time_ns><suffix>`. It raises `ValueError(t("gallery.unsupported", ...))` for a suffix not in `IMAGE_SUFFIXES`, and deletes that gesture's older copies (`memes/<gesture>-*`).
  - In `run_camera`: a local `memes`, loaded from `saved.memes`. `save_settings()` writes it as well.

- [ ] **Step 1: Write the failing tests**

```python
    def test_memes_are_saved_and_loaded(self):
        image = self.data.import_meme("calm", write_image(self.folder / "calma.png"))
        memes = Memes().with_gesture("calm", image=image, on=True)
        self.data.save(Settings(memes=memes))
        self.assertEqual(self.data.load().memes, memes)
        self.assertEqual(image.parent, self.data.memes_dir)

    def test_choosing_again_replaces_the_old_copy(self):
        first = self.data.import_meme("calm", write_image(self.folder / "a.png"))
        second = self.data.import_meme("calm", write_image(self.folder / "b.png"))
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())
        with self.assertRaises(ValueError):
            self.data.import_meme("calm", self.folder / "notes.txt")

    def test_missing_image_turns_the_gesture_off(self):
        image = self.data.import_meme("calm", write_image(self.folder / "calma.png"))
        self.data.save(Settings(memes=Memes().with_gesture("calm", image=image, on=True)))
        image.unlink()
        self.assertEqual(self.data.load().memes.gesture("calm"), MemeGesture())
```

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_control_panel.UserDataTests -v`
Expected: ERROR (no `import_meme`, no `memes` field).

- [ ] **Step 3: Implement** the interfaces above.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 237 tests`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/user_data.py vanishcam/app.py tests/test_control_panel.py
git commit -m "Save the meme settings and copy their images into the config folder"
```

---

### Task 5: The meme detector

**Files:**
- Modify: `vanishcam/detectors.py`
- Test: `tests/test_meme_poses.py`

**Interfaces:**
- Consumes: `match_gesture` (Task 2) and `face_box` (Task 1).
- Produces:
  - `class MemePoseDetector`:
    - `__init__(self, face, hands)`;
    - `enabled: frozenset[str]`, set by the app before each submit;
    - `recognize(image: mp.Image, timestamp_ms: int) -> str | None` runs the face landmarker and the hand landmarker on the same frame. It remembers the last face box for 1200 ms, as run3 did, then calls `match_gesture(hands, face, self.enabled)`;
    - `close()`.
  - `create_meme_detector() -> MemePoseDetector`: `require_models(HAND_MODEL, FACE_MODEL)`, with run3's confidences (0.45 for detection, presence, and tracking), `num_faces=1`, `num_hands=2`, and VIDEO mode.
  - `self_test` in `app.py` also builds and closes one with `create_meme_detector().close()`.

- [ ] **Step 1: Write the failing test**

```python
class MemeDetectorTests(unittest.TestCase):
    def test_face_is_remembered_and_only_enabled_gestures_match(self):
        face, hands = Mock(), Mock()
        face.detect_for_video.return_value = SimpleNamespace(face_landmarks=[make_face(*FACE)])
        fist_thumb_up = make_hand(0.5, 0.75, 0.16, extended=(False,) * 4, thumb="up")
        hands.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[fist_thumb_up])
        detector = MemePoseDetector(face, hands)
        detector.enabled = frozenset({"thumbs_up"})
        self.assertEqual(detector.recognize(object(), 0), "thumbs_up")
        face.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        self.assertEqual(detector.recognize(object(), 1000), "thumbs_up")  # Face hidden for 1 s: remembered.
        self.assertIsNone(detector.recognize(object(), 2300))  # Too long ago.
        detector.enabled = frozenset()
        face.detect_for_video.return_value = SimpleNamespace(face_landmarks=[make_face(*FACE)])
        self.assertIsNone(detector.recognize(object(), 2400))
```

- [ ] **Step 2: Run it and check that it fails**

Run: `.venv/bin/python -m unittest tests.test_meme_poses -v`
Expected: ERROR, `cannot import name 'MemePoseDetector'`.

- [ ] **Step 3: Implement** the interfaces above.

- [ ] **Step 4: Run all the tests and the self-test**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 238 tests`, `OK`.

Run the self-test with `cv2_enumerate_cameras` stubbed, as in phase 1 (it is missing from the local venv):
`SDL_AUDIODRIVER=dummy .venv/bin/python -c "import sys,types; sys.modules['cv2_enumerate_cameras']=types.ModuleType('x'); from vanishcam import app; sys.exit(app.self_test())"`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/detectors.py vanishcam/app.py tests/test_meme_poses.py
git commit -m "Add the meme pose detector with face memory"
```

---

### Task 6: The layer runs in the video loop

**Files:**
- Modify: `vanishcam/app.py`
- Test: `tests/test_video_output.py`

**Interfaces:**
- Consumes:
  - `MemeLayer`, `Memes.active`, and `DETECT_INTERVAL` (Task 3);
  - `create_meme_detector` and `MemePoseDetector.enabled` (Task 5);
  - the `memes` local and `save_settings` (Task 4);
  - `half_resolution_rgb` from `modes.py`.
- Produces, in `run_camera`:
  - `meme_layer = MemeLayer()` and `meme_detector: AsyncRecognizer | None = None`, plus a helper `sync_meme_detector()`:
    - it creates the detector (`AsyncRecognizer(recognizer.recognize, recognizer.close)`) when `memes.active()` is non-empty;
    - it closes it and calls `meme_layer.clear()` when `memes.active()` is empty;
    - it is called at startup and after every change to `memes`;
    - the detector is closed on exit through `resources`.
  - In the running branch, after `display = mode.render(...)`:
    1. When a detector exists, at least `DETECT_INTERVAL` has passed, and it is not busy: set `recognizer.enabled = memes.active()`, then submit `half_resolution_rgb(camera_frame)` with `elapsed_ms`.
    2. Read its result. If there is one, call `meme_layer.on_detection(result, now, memes)`.
    3. `display = meme_layer.render(display, now)`.
  - Nothing of the above runs while calibrating.
  - A module-level seam `create_meme_detector`, imported into `app`, is what the tests patch.

- [ ] **Step 1: Write the failing tests.** They reuse `run_keys`' harness style, with a helper `run_memes(memes, results, steps)`:
  - it patches `app.create_meme_detector` to return a `Mock` whose `recognize` side effect yields `results`, and patches `app.AsyncRecognizer` with `InlineRecognizer`;
  - it saves `Settings(memes=memes, shortcuts=...)` to the temporary `UserData`;
  - `steps` is a list of `(time, actions)` entries, one per frame, played through `FakePanel` actions and `video_clock`.

```python
    def test_meme_pops_over_the_video_and_the_call(self):
        # thumbs_up held from 0.1 s: shown from the 0.35 s hold, still on at 1.5 s.
        frames, sent = self.run_memes(Memes().with_gesture("thumbs_up", image=self.white_png, on=True),
                                      results=["thumbs_up"] * 6 + [None] * 4)
        self.assertTrue((frames[-1][12, 20] == 255).all())        # Preview: the white image in the middle.
        np.testing.assert_array_equal(sent[-1], frames[-1])        # The call gets the same picture.

    def test_no_detector_while_nothing_is_enabled(self):
        self.run_memes(Memes(), results=[])
        app.create_meme_detector.assert_not_called()

    def test_meme_never_changes_the_effect_frame(self):
        # Naruto vanished (its vanished_frame cached), meme shown then gone: the frame after it
        # equals the frame before it, pixel for pixel.
        ...
```

`self.white_png` is a 2×2 white PNG in the test's temporary folder. The `...` test is written in full, with the scenario its comment describes. The master-off test needs Task 7's panel action, so it lives in Task 7.

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_video_output -v`
Expected: the new tests FAIL (no meme in the frames; `create_meme_detector` does not exist in `app`).

- [ ] **Step 3: Implement** the interfaces above.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: all pass, about 241 tests.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/app.py tests/test_video_output.py
git commit -m "Run the meme layer over every effect, in the preview and the call"
```

---

### Task 7: The MEMES tab

**Files:**
- Modify: `vanishcam/control_panel.py`, `vanishcam/app.py`, and `vanishcam/i18n.py`
- Test: `tests/test_control_panel.py`, plus the master-off test moved from Task 6 into `tests/test_video_output.py`

**Interfaces:**
- Consumes:
  - `Memes`, `GESTURES`, and `CONFLICTS` (Tasks 2 and 3);
  - `UserData.import_meme` (Task 4);
  - `sync_meme_detector` (Task 6).
- Produces:
  - `TABS = ("live", "memes", "settings")`. The tab labels need to fit the bar: about 150–386 px, between "VANISHCAM" and the lamp. If they do not fit at size 10, the tab font goes down to 9.5.
  - `PanelView.memes: Memes = Memes()`.
  - **MEMES tab body**, from `spots.effect`:
    - section `"A"` with `t("panel.memes")`;
    - the master switch row, styled like SETTINGS' switch. Its text is `t("panel.memes_on")` or `t("panel.memes_off")`, and a click sends `("memes_enabled", not enabled)`;
    - one row per gesture in `GESTURES` order, 44 px high:
      - a 52×36 thumbnail of its image, or a hatched box saying `t("panel.choose_image")` when it has none. A click on the thumbnail sends `("meme_pick", name)`;
      - the gesture's name, `t(f"meme.{name}")`;
      - under the name, `t("panel.change_image")` in grey when it has an image. When the current effect (`view.mode`) collides with it per `CONFLICTS`, `t("panel.meme_conflict", effect=<effect title>)` in `AMBER` replaces that line;
      - a switch on the right: `("meme_on", name, not on)`. It is clickable only if the gesture has an image and the master switch is on. Otherwise it is drawn `PALE`.
  - Thumbnails come from `PanelRenderer.meme_thumb(path)`, cached like `thumb` but sized 52×36.
  - `ControlPanel.poll`:
    - turns `("meme_pick", name)` into a picker thread (reusing `pick_image_file`) that queues `("meme_import", name, path)`;
    - the picker is single-flight, shared with the background picker's `picking` flag.
  - App `handle`:
    - `("memes_enabled", value)`: `memes = memes.with_enabled(value)`, then `sync_meme_detector()` and `save_settings()`;
    - `("meme_on", name, value)`: `memes.with_gesture(name, on=value)`, then sync and save;
    - `("meme_import", name, path)`: `user_data.import_meme(name, path)` (on `ValueError` or `OSError`, `notify(t("loop.image_not_added", ...))`), then `memes.with_gesture(name, image=copy, on=True)`, then sync and save.
  - `PanelView(...)` gains `memes=memes`.
- **New i18n keys (EN / PT):**
  - `panel.tab_memes`: MEMES / MEMES
  - `panel.memes`: MEME REACTIONS / REAÇÕES COM MEMES
  - `panel.memes_on`: Memes on / Memes ligados
  - `panel.memes_off`: Memes off / Memes desligados
  - `panel.choose_image`: CHOOSE / ESCOLHER
  - `panel.change_image`: click the picture to change it / clique na imagem para trocar
  - `panel.meme_conflict`: "same gesture as {effect}" / "mesmo gesto do {effect}"
  - `meme.<name>`, eight of them:

    | Gesture | EN | PT |
    | --- | --- | --- |
    | `thumbs_up` | Thumbs up | Joinha |
    | `calm` | Calm down | Calma |
    | `hearing` | Hand to the ear | Mão na orelha |
    | `hands_behind_head` | Hands behind the head | Mãos atrás da cabeça |
    | `prayer` | Hands together | Mãos juntas |
    | `timeout` | Time-out | Tempo |
    | `pointing_at_self` | Pointing at myself | Apontando pra mim |
    | `hands_on_face` | Hands on the face | Mãos no rosto |

- [ ] **Step 1: Write the failing tests**

```python
class MemesTabTests(unittest.TestCase):
    def render(self, view):
        return PanelRenderer().render(view, tab="memes")

    def test_three_tabs_fit_in_the_bar(self):
        _, regions = PanelRenderer().render(PanelView("normal", ""))
        tabs = [r for r in regions if r.action[0] == "tab"]
        self.assertEqual([r.action[1] for r in tabs], ["live", "memes", "settings"])
        self.assertTrue(all(r.box[2] < WIDTH - PAD - 20 for r in tabs))  # Clear of the lamp.

    def test_rows_switches_and_pickers(self):
        image = write_image(Path(self.enterContext(tempfile.TemporaryDirectory())) / "a.png")
        memes = Memes().with_gesture("calm", image=image, on=True)
        image_out, regions = self.render(PanelView("normal", "", memes=memes))
        actions = [r.action for r in regions]
        self.assertEqual(image_out.shape[0], panel_height(PanelView("normal", "")))
        self.assertIn(("memes_enabled", False), actions)
        self.assertEqual([a[1] for a in actions if a[0] == "meme_pick"], list(GESTURES))
        self.assertIn(("meme_on", "calm", False), actions)
        self.assertFalse(any(a[:2] == ("meme_on", "thumbs_up") for a in actions))  # No image yet.

    def test_master_off_greys_the_switches(self):
        ...  # memes with calm on but enabled=False: ("memes_enabled", True) present, no ("meme_on", ...) at all.

    def test_conflict_warning_follows_the_effect(self):
        ...  # drawn text (joined, as in ControlPanelTabsTests) contains t("panel.meme_conflict", effect="Naruto")
             # for view.mode == "naruto", and not for "goku".
```

Then, in `LiveSwitchTests`:
- `test_importing_a_meme_turns_it_on_and_saves`: the panel action `("meme_import", "calm", outside_png)` leads to the saved `memes.gesture("calm").on` being true, with its image inside `memes_dir`.
- `test_master_off_clears_and_stops_detection`, moved from Task 6: a meme is shown, then `("memes_enabled", False)` arrives. The next frame equals the camera frame, and the detector's `close` was called.

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_control_panel tests.test_video_output -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement** the interfaces above. Render all three tabs to a PNG in the scratchpad and look at it with Read: the tabs must fit, the rows must not overlap, and the panel must be as tall as LIVE.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/control_panel.py vanishcam/app.py vanishcam/i18n.py tests/
git commit -m "Add the MEMES tab to pick an image and switch on each gesture"
```

---

### Task 8: Docs and removing the prototype

**Files:**
- Create: `docs/memes.md`. It covers what memes are, each gesture with how to make it, choosing images, the master and per-gesture switches, the collisions with Naruto and Ash, and that nothing is bundled. Its last section, "Performance", says that a second detector runs while memes are on, and that turning them off if the video stutters brings back the previous speed.
- Modify: `docs/README.md` (the guide table), `README.md` (features, and the docs table), `docs/development.md` (file table rows for `hand_geometry.py`, `meme_poses.py`, `memes.py`, and the "prototypes/" line), and `docs/control-panel.md` (the MEMES tab and the "Saved preferences" folder, which now holds `memes/`).
- Delete: `prototypes/run3.py` and the now-empty `prototypes/`.

- [ ] **Step 1: Check that nothing references the prototype**

Run: `grep -rn "run3\|prototypes" --include=*.py --include=*.md --include=*.spec . | grep -v "^./docs/superpowers"`
Expected: only the `docs/development.md` line about `prototypes/`, which this task removes.

- [ ] **Step 2: Write the docs and delete the prototype.**

- [ ] **Step 3: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A docs README.md prototypes
git commit -m "Document the meme reactions and remove the run3 prototype"
```
