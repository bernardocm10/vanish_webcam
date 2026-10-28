# Effect registry, panel tabs with configurable shortcuts, and meme reactions

Date: 2026-10-28. Status: design, awaiting approval.

## Why

Three things the owner wants, in this order:

1. **Adding an effect should be cheap.** Today one effect is spread over seven places, keyed by its name as a string. A future effect, or the meme layer below, means editing all of them.
2. **Shortcuts fire by accident during calls.** The owner typed in Teams while the VanishCam window still had focus, and the bare-letter shortcuts (`V`, `R`, `1`–`5`, `SPACE`, `F`, `D`) triggered effects. `Q` even quits the app mid-call. Shortcuts must stop being keys that ordinary typing produces, and the owner wants to choose them in the app.
3. **Meme reactions in calls**, reusing the pose detectors of `prototypes/run3.py`. You make a gesture, an image pops up over the middle of the video, stays a few seconds, and goes away by itself. Everyone in the call sees it.

Each phase gets its own implementation plan and ships on its own. Phase 1 must not change behavior.

## Constraints (all phases)

- Windows and Linux, from source and from the PyInstaller EXE.
- The output sent to the call (virtual camera) shows only the person, the effect, and now the meme. Never menus, debug text, or tracking boxes.
- Every user-visible text goes through `t()` in `vanishcam/i18n.py`, in English and Portuguese (`tests/test_i18n.py` enforces it). Code, comments, docs, and commit messages are in English.
- The panel is drawn with Pillow and shown in the OpenCV window (`vanishcam/control_panel.py`). No GUI toolkit is added.
- Tests use `unittest` and run on GitHub Actions on Windows and Linux. Each phase keeps them green and adds its own.
- Gesture detection stays off the video thread (`AsyncRecognizer`).
- Commit messages are one subject line, no body.

---

## Phase 1: one place per effect

### Today

What one effect touches, besides its own `vanishcam/effects/<name>.py`:

| Where | What depends on the name |
| --- | --- |
| `modes.py` `MODES` | the `Mode` subclass |
| `detectors.py` `create_recognizer` | which MediaPipe tasks to build (`mode in ("goku", "particles", "ash")`, `mode == "naruto"`, otherwise the gesture recognizer) and which detector class wraps them |
| `detectors.py` `detection_function` | which detector method to call |
| `models.py` `download_models` | `mode != "normal"` skips the gesture model |
| `app.py` `open_sound` | `naruto` plays `vanish.MP3`, `goku` plays `teleport.mp3` (also looked up by `sound_path()`), the rest are silent |
| `control_panel.py` `MODE_CARDS` | the order in the panel and the `1`–`5` keys |
| `i18n.py` | `mode.<name>.title`, `.hint`, `.instructions` |

### Goal

An effect is declared in one place: its `Mode` subclass (or a small descriptor next to it) says which detector it uses, which models it needs, and which sound it plays. Everything else reads that declaration. After this phase, adding an effect means:

1. `vanishcam/effects/<name>.py` with its drawing and pose,
2. a `Mode` subclass registered in `MODES`,
3. its texts in `i18n.py`.

Update `docs/development.md` ("Adding an effect") to match.

### Requirements

- No string switches on effect names remain in `detectors.py`, `models.py`, `app.py`, or `control_panel.py`.
- The panel order and the `1`–`5` order come from the registry, not a separate tuple. Note that `MODES` today is ordered normal, naruto, particles, goku, ash, while the panel is normal, naruto, goku, particles, ash: the registry takes the panel's order.
- `download_models` always downloads the segmenter, plus whatever the chosen effect declares.
- Behavior is identical: the same detectors, thresholds, sounds, order, and messages. The existing tests pass, updated only where they import names that moved or were renamed.

### Left to the builder

- Descriptor on the `Mode` class versus a separate registry dataclass; names.
- Whether `create_recognizer` / `detection_function` / `open_sound` survive as thin wrappers or disappear (tests in `test_video_output.py`, `test_particles_effect.py`, `test_ash_effect.py`, and `test_call_audio.py` patch or call them; update them as needed).
- Pulling repeated helpers (`smoothstep` exists in both `effects/goku.py` and `effects/normal.py`) into one place if it falls out naturally.

### Not in this phase

- Sharing face and hand landmarks between the effect detector and the meme detector. Phase 3 runs its own detector (see there).

---

## Phase 2: panel tabs and configurable shortcuts

### Tabs

A row of tabs goes right under the black VANISHCAM top bar: **LIVE** (*AO VIVO*), **MEMES**, **SETTINGS** (*AJUSTES*). The owner picked this layout from three mockups.

- **LIVE** is today's panel, unchanged: effect, background, camera, controls. The app opens on LIVE every time; the last tab is not remembered.
- **SETTINGS** holds the shortcuts (below). It is built so more settings can be added later.
- **MEMES** arrives in phase 3. Until then it is not shown.
- Shortcuts work on every tab, not only on LIVE.
- The panel keeps its width and does not grow taller than today. A tab whose content does not fit scrolls, or is laid out to fit; the builder decides.
- Visual style follows the existing broadcast control surface: light faceplate, black ink, red for whatever is live. The active tab is marked the way the mockup did it (ink text plus a red underline).

### Shortcuts

**Actions:** vanish, reappear, switch to effect N (one per effect), new background, fullscreen, detach/attach, quit.

**Fixed, not configurable:** `ESC` leaves fullscreen (as today). Closing the window quits. Outside fullscreen, `ESC` no longer quits (today it does, like `Q`): quitting must not be a single key.

**Defaults** must not be produced by typing ordinary text: no bare letters, digits, space, punctuation, Enter, or Backspace. Quit must not be a single key; by default it is a modifier combo if those work (below), or has no shortcut at all, leaving the window's close button.

**Technical risk to settle first.** The loop reads keys with `cv2.waitKey(...) & 0xFF` (`app.py`), which drops function keys and modifiers. `cv2.waitKeyEx` returns more, but what it reports for Ctrl/Alt/Shift differs by OpenCV backend (Win32 on Windows; GTK or Qt on Linux depending on the wheel). The first task of this phase is a small probe that logs what `waitKeyEx` returns for Ctrl+letter, Alt+letter, Ctrl+Shift+letter, and F1–F12 on Linux; the owner runs it on Windows by hand, since CI cannot press keys. The probe must check that a combo can be told apart from the plain key, not only that `waitKeyEx` returns something: the Linux `opencv-contrib-python` wheel uses the Qt backend, which may swallow Ctrl-only combos or report Alt+letter and Ctrl+Shift+letter as the bare letter (unverified). Then:

- If modifier combos are reliable on both systems: defaults are combos, and any key or combo can be bound.
- If not: defaults are function keys, and binding accepts only keys the probe showed to be reliable on both systems.

The builder chooses the exact defaults within these rules and lists them in `docs/control-panel.md`.

**Rebinding in SETTINGS:**

- Each action shows its current shortcut. Clicking it waits for the next key or combo; `ESC` cancels.
- While waiting, that key press goes to the binder and does not trigger anything.
- A shortcut already used by another action is refused, and the panel says which action has it.
- A "restore defaults" button.
- Every place that shows a key reflects the current bindings: the key labels on the LIVE buttons (today `V`, `R`, `ESPAÇO`, `F`, `D`), the footer hint (`1–5 EFEITO · V SUMIR · R VOLTAR · Q SAIR`), and the terminal hints.

**Saved** in `settings.json` with the other preferences. A missing or invalid binding falls back to its default without losing the others.

### Docs

Update the shortcut tables and mentions in `docs/control-panel.md`, `docs/effects.md` (`V`, `R`, `1`–`5`), and `README.md`.

---

## Phase 3: meme reactions

### What the owner sees

- In a call, the owner makes one of the meme gestures and holds it for a moment.
- An image pops in from the center of the video: it grows quickly from nothing, overshoots slightly, and settles. It then covers about 60–70% of the frame (fit inside 65% of the width and 65% of the height, aspect ratio kept, centered on the video, not on the face).
- After about 3 seconds it leaves with a short shrink and fade, by itself.
- It is part of the output: the call and the preview both see it. Transparent PNGs keep their transparency. No border, no label.

### Behavior

- Memes are a **layer that is always on**, independent of the chosen effect. They work in every effect and in every phase of it (ready, vanished...), drawn on top of the effect's output. They do not run while the background is being captured.
- **One at a time.** While an image is on screen, other gestures do nothing. After it leaves, the same gesture has to be released before it can fire again, so holding a pose does not loop it.
- A master on/off switch turns the whole layer off, including its detection.

### Gestures

The eight working detectors from `prototypes/run3.py`, translated to English and ported into the package. The disabled "Yuta" pose stays out.

| run3 name | Gesture | Default label (EN / PT) |
| --- | --- | --- |
| `thumbs_up_pose` | thumbs up | the builder writes the labels |
| `calma_hand_pose` | open hand in front of the chest, fingers together, palm forward | |
| `hearing_hand_pose` | open hand beside the ear | |
| `hands_behind_head_pose` | both hands at the sides of the head | |
| `prayer_hands_pose` | hands pressed together or interlaced | |
| `timeout_hands_pose` | the "T" of time-out | |
| `pointing_at_self_pose` | index finger pointing at your own chest | |
| `hands_on_face_pose` | both hands up in front of the face | |

- Gestures that are switched off are not evaluated. When two enabled gestures match at once, run3's precedence applies: thumbs up, hands behind head, time-out, calm, hands on face, prayer, pointing at self, hearing.
- A gesture fires after being held for a moment. run3 counted frames (2 for prayer, 5 for the rest); convert that to a hold time, reusing `HoldTracker` from `effects/goku.py`.
- Detectors take normalized landmarks and a face box and return `(matched, score)` in run3. Keep them as pure functions with unit tests on synthetic landmarks, like the existing effect tests.
- run3's finger helpers (`extended_finger_states`, `hand_features`) overlap with `extended_fingers` in `effects/ash.py`. Put the shared hand geometry in one module.

### Detection

The memes run their own `AsyncRecognizer` with a face landmarker and a hand landmarker (both models are already bundled and versioned), next to the effect's detector. It runs only when the master switch is on and at least one gesture is enabled.

This is the designer's call: sharing one landmark pass between the effect and the memes would save CPU for Goku, Particles, and Ash, but it is left for later unless the second detector makes the video visibly slower on the owner's machine. Measure frame rate with memes on and off; if it drops noticeably, raise it with the owner before optimizing.

### Gestures that collide with effects

Two meme gestures are close to effect triggers: **prayer** is the Naruto seal, and **hands behind head** is near Ash's both-hands-up. With both active, one pose could fire the effect and the meme together. The MEMES tab shows a warning on a gesture that collides with the effect currently selected. It does not switch anything off by itself; the owner decides with the per-gesture switch.

### MEMES tab

From the chosen mockup:

- the master switch "Memes on" at the top;
- one row per gesture: thumbnail of its image, gesture name, "change image", and its own on/off switch;
- the collision warning under the gesture name when it applies.

**Images:** the app ships with **no meme images**: no third-party images in the repository (the project already promises no copyrighted assets). Every gesture starts with no image and switched off. Its row shows "choose image" instead of a thumbnail. Choosing an image (the system file picker, `pick_image_file()` in `user_data.py`) copies it into the user's config folder, next to `backgrounds/`, and switches that gesture on. A gesture without an image cannot be switched on.

**Saved** in `settings.json`: the master switch, and per gesture its image and whether it is on. A missing image file turns that gesture back to "no image".

### Cleanup

Delete `prototypes/run3.py` (and `prototypes/` if empty) once everything useful is ported. Update `docs/` with a guide for memes and the development table.

### Not in this phase

- Bundled meme images, meme sounds.
- Per-gesture size, position, or duration settings.
- Remembering the last panel tab.

---

## Anti-goals

- Anything triggering from typing ordinary text while the window has focus.
- Debug text, tracking boxes, or panel elements in the call output.
- Third-party images committed to the repository.
- A visibly slower video because of the meme layer.
