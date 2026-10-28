# Effect Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each effect declares its models, recognizer, detection call, and sound on its `Mode` subclass, so no module switches on effect names any more.

**Architecture:** `Mode` (in `vanishcam/modes.py`) gains class-level declarations: `models`, `create_recognizer()`, `detect(recognizer, image, timestamp_ms)`, `sound_file()`, and `sound_optional`. `detectors.py` exposes one factory per kind of recognizer instead of `create_recognizer(mode)`. `app.py`, `models.py`, and `control_panel.py` read `MODES[name]`. The `MODES` dict order becomes the panel order and the `1`–`5` keys.

**Tech Stack:** Python 3.12, MediaPipe tasks, OpenCV, unittest.

**Spec:** `docs/superpowers/specs/2026-10-28-registry-shortcuts-memes-design.md`, section "Phase 1: one place per effect".

## Global Constraints

- No behavior change: the same detectors, thresholds, sounds, order, and messages.
- No string comparison on effect names in `detectors.py`, `models.py`, `app.py`, or `control_panel.py`. A lookup `MODES[name]` is fine.
- Every user-visible text goes through `t()` with English and Portuguese entries (`tests/test_i18n.py` enforces it).
- Code, comments, docs, and commit messages are in English; commit messages are one subject line, no body.
- Run tests with `.venv/bin/python -m unittest`. The baseline is 179 tests, all passing.
- `app.create_recognizer(name: str)` stays as the seam the tests patch (`tests/test_video_output.py` `app_mocks`, `tests/test_control_panel.py:313`, which asserts it was called with `("naruto",)`).
- `detectors.recognize_gesture` must be looked up at call time (tests patch `detectors.recognize_gesture`).

## Review Focus

- **Switching effects at runtime:** `switch_mode` builds an `EffectRuntime` for the new name; every effect must reach its own recognizer method. Pinned by `test_each_effect_detects_with_its_own_method` (Task 2).
- **Goku sound missing:** Goku still starts, silent, and prints the "no sound" message. Pinned by `test_missing_optional_sound_runs_silent` (Task 3).
- **Naruto sound missing:** still an error, as today (`VanishSound` raises `sound.missing`). Pinned by `test_missing_required_sound_is_an_error` (Task 3).
- **`--self-test` in the EXE build:** it must still require every model, both sounds, and the fonts, and build every recognizer. Pinned by `test_self_test_requires_every_declared_file` (Task 3) and a manual `--self-test` run (Task 4).
- **Keys `1`–`5` after reordering `MODES`:** `3` must stay Goku and `4` Particles. Pinned by `test_panel_and_keys_follow_the_registry_order` (Task 4).

---

### Task 1: Each effect declares its models

**Files:**
- Modify: `vanishcam/models.py` (`download_models`)
- Modify: `vanishcam/modes.py` (`Mode` and the five subclasses)
- Modify: `vanishcam/app.py:142` (`EffectRuntime.__init__`) and `vanishcam/app.py:228` (`run_camera`)
- Test: `tests/test_models.py` (new)

**Interfaces:**
- Produces: `Mode.models: tuple[Path, ...]`, the MediaPipe models the effect needs besides the segmenter. Normal: `(GESTURE_MODEL,)`. Naruto: `(HAND_MODEL,)`. Goku, Particles, Ash: `(HAND_MODEL, FACE_MODEL)`.
- Produces: `download_models(models: tuple[Path, ...] = ()) -> None` downloads `SEGMENTER_MODEL` plus every path in `models` that has an entry in `MODEL_URLS`. Hand and face models are versioned and have no URL, so they are skipped.

- [ ] **Step 1: Write the failing test** in `tests/test_models.py`

```python
class DownloadModelsTests(unittest.TestCase):
    def requested(self, declared):
        """URLs fetched by download_models(declared) into an empty folder."""
        # Patch MODEL_DIR, SEGMENTER_MODEL, GESTURE_MODEL and MODEL_URLS onto a temp folder,
        # with MODEL_URLS = {segmenter: "https://seg", gesture: "https://gesture"}.
        # urllib.request.urlopen is a Mock returning a context manager whose .headers.get gives 0
        # and whose .read returns b"x" once, then b"". Silence print. Return the fetched URLs in order.

    def test_segmenter_always_and_only_the_declared_models(self):
        self.assertEqual(self.requested(()), ["https://seg"])
        self.assertEqual(self.requested((models.GESTURE_MODEL,)), ["https://seg", "https://gesture"])
        self.assertEqual(self.requested((models.HAND_MODEL, models.FACE_MODEL)), ["https://seg"])

    def test_every_effect_declares_its_models(self):
        self.assertEqual(MODES["normal"].models, (GESTURE_MODEL,))
        self.assertEqual(MODES["naruto"].models, (HAND_MODEL,))
        for name in ("goku", "particles", "ash"):
            self.assertEqual(MODES[name].models, (HAND_MODEL, FACE_MODEL))
```

`requested` reads the URL from the `Request` passed to `urlopen` (`request.full_url`).

- [ ] **Step 2: Run it and check that it fails**

Run: `.venv/bin/python -m unittest tests.test_models -v`
Expected: FAIL (`AttributeError: ... has no attribute 'models'`, and `download_models` treats the tuple as a mode name).

- [ ] **Step 3: Implement**

- Change the signature to `download_models(models: tuple[Path, ...] = ())`. Iterate `MODEL_URLS` and download a destination only if it is `SEGMENTER_MODEL` or is in `models`.
- Add `models` to `Mode` (default `()`) and to each subclass. `modes.py` imports the paths from `.models`.
- In `app.py`, call `download_models(MODES[name].models)` in `EffectRuntime.__init__` and `download_models(MODES[mode_name].models)` in `run_camera`.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 181 tests`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/models.py vanishcam/modes.py vanishcam/app.py tests/test_models.py
git commit -m "Let each effect declare the models it downloads"
```

---

### Task 2: Each effect declares its recognizer and how it detects

**Files:**
- Modify: `vanishcam/detectors.py`: replace `create_recognizer(mode)` with three factories and delete `detection_function`.
- Modify: `vanishcam/modes.py`
- Modify: `vanishcam/app.py`: imports, a new `create_recognizer(name)`, `EffectRuntime.__init__`, and the `self_test` loop.
- Modify: `tests/test_particles_effect.py:114-117` and `tests/test_ash_effect.py:139-144`, which call `detection_function`.
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `Mode.models` (Task 1).
- Produces, in `detectors.py`:
  - `create_gesture_recognizer() -> MultiViewGestureRecognizer`: the current `else` branch of `create_recognizer`, `Victory` allowlist included.
  - `create_hand_landmarker() -> MultiViewHandLandmarker`: the current `naruto` branch.
  - `create_pose_detector(detector: type[GokuPoseDetector]) -> GokuPoseDetector`: the current face-and-hands branch, building `detector(face, hands)`.
  - Each factory keeps raising `RuntimeError(t("models.missing", path=...))` for a missing hand or face model, as today.
- Produces, on `Mode`:
  - `create_recognizer()`, a staticmethod (or a classmethod on pose effects) that returns the recognizer. Normal: `create_gesture_recognizer()`. Naruto: `create_hand_landmarker()`. Goku: `create_pose_detector(GokuPoseDetector)`. Particles: `SalutePoseDetector`. Ash: `HandsUpPoseDetector`.
  - `detect(recognizer, image: mp.Image, timestamp_ms: int) -> object`, a staticmethod. Normal: `detectors.recognize_gesture(recognizer, image, timestamp_ms)`, looked up through the module at call time. Naruto: `recognizer.recognize_joined_hands(image, timestamp_ms)`. Goku, Particles, Ash: `recognizer.recognize_pose(image, timestamp_ms)`.
  - The base `Mode` raises `NotImplementedError` for both.
- Produces, in `app.py`: `create_recognizer(name: str)` returning `MODES[name].create_recognizer()`. `EffectRuntime` builds its detector with `AsyncRecognizer(functools.partial(MODES[name].detect, self.recognizer), self.recognizer.close)`.

- [ ] **Step 1: Write the failing test** in `tests/test_models.py`

```python
class RecognizerDeclarationTests(unittest.TestCase):
    def test_each_effect_detects_with_its_own_method(self):
        image = object()
        with patch.object(detectors, "recognize_gesture", return_value=("Victory", 0.9)) as gesture:
            recognizer = Mock()
            self.assertEqual(MODES["normal"].detect(recognizer, image, 7), ("Victory", 0.9))
            gesture.assert_called_once_with(recognizer, image, 7)
        expected = {"naruto": "recognize_joined_hands", "goku": "recognize_pose",
                    "particles": "recognize_pose", "ash": "recognize_pose"}
        for name, method in expected.items():
            with self.subTest(name):
                recognizer = Mock()
                self.assertIs(MODES[name].detect(recognizer, image, 7), getattr(recognizer, method).return_value)
                getattr(recognizer, method).assert_called_once_with(image, 7)

    def test_pose_effects_build_their_own_detector_class(self):
        expected = {"goku": GokuPoseDetector, "particles": SalutePoseDetector, "ash": HandsUpPoseDetector}
        for name, detector in expected.items():
            with self.subTest(name), patch.object(modes, "create_pose_detector") as create:
                MODES[name].create_recognizer()
                create.assert_called_once_with(detector)
```

`modes.py` must import `create_pose_detector` into its namespace so that `patch.object(modes, "create_pose_detector")` takes effect.

Replace the `detection_function` assertions in `tests/test_particles_effect.py` and `tests/test_ash_effect.py` with the same check through `ParticlesMode.detect` and `AshMode.detect`: the call returns `recognizer.recognize_pose.return_value`, called with `(image, 0)`.

- [ ] **Step 2: Run it and check that it fails**

Run: `.venv/bin/python -m unittest tests.test_models tests.test_particles_effect tests.test_ash_effect -v`
Expected: FAIL (`NotImplementedError` or `AttributeError: ... 'detect'`).

- [ ] **Step 3: Implement** the interfaces above. Move the `"Looked up on every call (not bound here) so the tests can replace it"` comment from `detection_function` to `NormalMode.detect`. Remove the now unused imports from `app.py` (`detection_function`, and `create_recognizer` from `.detectors`). `self_test` loops `for name in MODES: create_recognizer(name).close()`.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 183 tests`, `OK`. `tests/test_control_panel.py` still sees `create_recognizer` called with `("naruto",)`.

- [ ] **Step 5: Check that no name switch is left in detectors.py**

Run: `grep -nE '"(normal|naruto|goku|particles|ash)"' vanishcam/detectors.py`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add vanishcam/detectors.py vanishcam/modes.py vanishcam/app.py tests/
git commit -m "Let each effect build its own recognizer and choose how it detects"
```

---

### Task 3: Each effect declares its sound

**Files:**
- Modify: `vanishcam/modes.py` (`Mode`, `NarutoMode`, `GokuMode`)
- Modify: `vanishcam/app.py` (`open_sound`, `self_test`, imports)
- Modify: `vanishcam/i18n.py`: rename keys `goku.sound` to `sound.found` and `goku.no_sound` to `sound.optional_missing`, in both languages, with the texts unchanged.
- Test: `tests/test_call_audio.py`

**Interfaces:**
- Produces, on `Mode`:
  - `sound_file() -> Path | None`, a staticmethod; `None` means silent (the base default).
    - Naruto: `SOUND_PATH` (`assets/vanish.MP3`).
    - Goku: `sound_path() or ASSETS_DIR / SOUND_NAME`. When the file is missing it still returns a path, so the message can name it.
  - `sound_optional: bool = False`. `True` on Goku only: a missing file means a silent effect, not an error.
- Produces, in `app.py`: `open_sound(name: str, mic: VirtualMic | None = None)`, the same signature with this body:
  1. `path = MODES[name].sound_file()`. If it is `None`, return `None`.
  2. If the effect is optional:
     - missing file: print `f"[{title}] {t('sound.optional_missing', name=path.name)}"` and return `None`;
     - otherwise print `f"[{title}] {t('sound.found', name=path.name)}"`;
     - `title` is `t(f"mode.{name}.title")`.
  3. Build `VanishSound(path)` and wrap it in `CallSound(sound, mic, path)` when `mic` is not `None`.
- `self_test` builds its `required` tuple as `SEGMENTER_MODEL`, the union of every `MODES[...].models`, every non-`None` `sound_file()`, and the fonts. Drop the `GOKU_SOUND_NAME`/`sound_path`/`SOUND_PATH` imports from `app.py` if nothing else uses them.

- [ ] **Step 1: Write the failing tests** in `tests/test_call_audio.py`, next to `test_naruto_sound_is_shared_with_the_call`

```python
    def test_missing_optional_sound_runs_silent(self):
        with patch.object(GokuMode, "sound_file", return_value=Path("nowhere/teleport.MP3")), \
                patch("builtins.print") as printed:
            self.assertIsNone(app.open_sound("goku", None))
        self.assertIn("teleport.MP3", printed.call_args.args[0])
        self.assertTrue(printed.call_args.args[0].startswith("[Goku]"))

    def test_missing_required_sound_is_an_error(self):
        with patch.object(NarutoMode, "sound_file", return_value=Path("nowhere/vanish.MP3")):
            with self.assertRaises(RuntimeError):
                app.open_sound("naruto", None)

    def test_self_test_requires_every_declared_file(self):
        with patch.object(GokuMode, "sound_file", return_value=Path("nowhere/teleport.MP3")):
            with self.assertRaisesRegex(RuntimeError, "teleport.MP3"):
                app.self_test()
```

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_call_audio -v`
Expected: FAIL (`AttributeError: ... 'sound_file'`).

- [ ] **Step 3: Implement** the interfaces above.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 186 tests`, `OK`. `tests/test_i18n.py` passes with the renamed keys.

- [ ] **Step 5: Check that no name switch is left in app.py or models.py**

Run: `grep -nE '"(normal|naruto|goku|particles|ash)"' vanishcam/app.py vanishcam/models.py`
Expected: only the default effect at `run_camera` (`saved.mode if saved.mode in MODES else "normal"`). It picks a default rather than switching behavior, so it stays.

- [ ] **Step 6: Commit**

```bash
git add vanishcam/modes.py vanishcam/app.py vanishcam/i18n.py tests/test_call_audio.py
git commit -m "Let each effect declare its sound"
```

---

### Task 4: The panel follows the registry, and the docs say how to add an effect

**Files:**
- Modify: `vanishcam/modes.py`: reorder `MODES` to `NormalMode, NarutoMode, GokuMode, ParticlesMode, AshMode` and update the module docstring.
- Modify: `vanishcam/control_panel.py:76`: `MODE_CARDS = tuple(ModeCard(key) for key in MODES)`.
- Modify: `docs/development.md`: "Adding an effect" becomes the three steps of the spec, and the `modes.py` / `detectors.py` rows in the file table describe the new split.
- Test: `tests/test_control_panel.py`

**Interfaces:**
- Consumes: `MODES` (all tasks).

- [ ] **Step 1: Write the failing test** in `tests/test_control_panel.py`

```python
    def test_panel_and_keys_follow_the_registry_order(self):
        order = ["normal", "naruto", "goku", "particles", "ash"]
        self.assertEqual(list(MODES), order)
        self.assertEqual([card.key for card in MODE_CARDS], order)
        for number, name in enumerate(order, start=1):
            self.assertEqual(app.KEY_ACTIONS[ord(str(number))], ("mode", name))
```

- [ ] **Step 2: Run it and check that it fails**

Run: `.venv/bin/python -m unittest tests.test_control_panel -v`
Expected: FAIL on `list(MODES)` (particles comes before goku today).

- [ ] **Step 3: Implement** the reorder and `MODE_CARDS`, and update the docs. "Adding an effect" in `docs/development.md` becomes:
  1. `vanishcam/effects/<name>.py` with its drawing and pose,
  2. a `Mode` subclass that declares `models`, `create_recognizer`, `detect`, and optionally `sound_file`/`sound_optional`, registered in `MODES` at the position it should have in the panel,
  3. its `mode.<name>.title`, `.hint`, and `.instructions` in both languages in `vanishcam/i18n.py`.

- [ ] **Step 4: Run all the tests and the self-test**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 187 tests`, `OK`.

Run: `SDL_AUDIODRIVER=dummy .venv/bin/python -m vanishcam --self-test`
Expected: exit code 0, with the `selftest.ok` message as the last line.

- [ ] **Step 5: Check the whole package for name switches**

Run: `grep -nE '"(normal|naruto|goku|particles|ash)"' vanishcam/detectors.py vanishcam/models.py vanishcam/app.py vanishcam/control_panel.py`
Expected: only the `run_camera` default noted in Task 3.

- [ ] **Step 6: Commit**

```bash
git add vanishcam/modes.py vanishcam/control_panel.py docs/development.md tests/test_control_panel.py
git commit -m "Order the panel by the effect registry and document adding an effect"
```
