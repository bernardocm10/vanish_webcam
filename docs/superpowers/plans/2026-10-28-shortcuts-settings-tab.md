# Shortcuts and Settings Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keyboard shortcuts are off by default. Tabs in the panel lead to a SETTINGS page where the owner turns them on and picks one key per action, so typing with the VanishCam window focused no longer triggers effects or quits the app.

**Architecture:** A new pure module, `vanishcam/shortcuts.py`, holds an immutable `Shortcuts` value: whether shortcuts are on, plus one key per action. It is saved in `settings.json` through `Settings`. The video loop reads full key codes with `cv2.waitKeyEx` and turns them into actions only through `Shortcuts`. `ESC` is handled apart and only leaves fullscreen. The panel gets two tabs in its black top bar (LIVE and SETTINGS), and the SETTINGS tab renders the switch and the key list. A row being rebound captures the next key press.

**Tech Stack:** Python 3.12, OpenCV (Qt backend on Linux, Win32 on Windows), Pillow, unittest.

**Spec:** `docs/superpowers/specs/2026-10-28-registry-shortcuts-memes-design.md`, section "Phase 2", including the "Decision after the key probe" paragraph, which overrides the rules about combos and function keys below it.

## Global Constraints

- Shortcuts are **off by default**. While off, no key does anything except `ESC`, which leaves fullscreen.
- **Bindable keys:** single keys with codes 32–126 (letters, digits, punctuation, space). Letters are stored and matched in lowercase.
- **Defaults when turned on:** vanish `v`, reappear `r`, new background space, fullscreen `f`, detach `d`, effects `1`–`5` in `MODES` order. Quit has no key.
- **`ESC`:** leaves fullscreen and cancels a rebind in progress. It never quits; today it does.
- **Closing the window** still quits.
- **Panel:** keeps its width (`WIDTH = 440`) and does not grow taller. The app opens on the LIVE tab every time, and LIVE looks as it does today apart from the key legends.
- **Text:** every user-visible string goes through `t()` with English and Portuguese entries. Code, docs, and commits are in English; commit messages are one subject line.
- **Tests:** run with `.venv/bin/python -m unittest`. The baseline on `main` is 188 tests, all passing.
- **MEMES tab:** not shown in this phase.

## Review Focus

- **Typing with the window focused and shortcuts off:** a sentence like `"vou sair rapidinho"` must trigger nothing. Pinned by `test_typing_does_nothing_while_shortcuts_are_off` (Task 3).
- **`ESC` on Linux/Qt:** it may arrive as `27` or as the X keysym `0xFF1B`. Both must leave fullscreen, and neither may quit. Pinned by `test_escape_leaves_fullscreen_and_never_quits` (Task 3).
- **Upgrading from an older `settings.json`:** a file without `"shortcuts"`, or with garbage in it, must load as off with the default keys, without losing the other preferences. Pinned by `test_old_or_broken_shortcuts_load_as_defaults` (Task 2).
- **Saving another preference** (switching effect, background, camera, or detach) must not wipe the shortcuts. Pinned by `test_switching_effect_keeps_the_shortcuts` (Task 5).
- **Shift or Alt pressed while rebinding:** on Qt they arrive as keysyms `0xFFE1` and `0xFFE9`. They must not bind and must not show an error; the row keeps waiting. Pinned by `test_modifier_keys_are_ignored_while_capturing` (Task 5).

---

### Task 1: The shortcuts model

**Files:**
- Create: `vanishcam/shortcuts.py`
- Test: `tests/test_shortcuts.py`

**Interfaces:**
- Produces:
  - `ESCAPE_CODES = (27, 0xFF1B)`
  - `ACTIONS: tuple[str, ...]`: `"vanish"`, `"reappear"`, `"recapture"`, `"fullscreen"`, `"detach"`, then `f"mode.{name}"` for each `name in MODES`, then `"quit"`.
  - `DEFAULT_KEYS: dict[str, int | None]`: the Global Constraints defaults. `mode.<name>` maps to the digit of its position (only up to 9); quit maps to `None`.
  - `normalize(code: int) -> int | None`: lowercases `A`–`Z` and returns codes in 32–126; anything else gives `None`.
  - `action_tuple(action: str) -> tuple`: `"mode.goku"` gives `("mode", "goku")`, `"vanish"` gives `("vanish",)`, `"quit"` gives `("quit",)`.
  - `@dataclass(frozen=True) class Shortcuts`:
    - fields `enabled: bool = False` and `keys: tuple[tuple[str, int | None], ...] = tuple(DEFAULT_KEYS.items())`;
    - `key_for(action: str) -> int | None`;
    - `action_for(code: int) -> tuple | None` returns `None` when disabled, unbound, or unsupported; it normalizes `code` first;
    - `label(code: int | None) -> str`, a staticmethod: `""` for `None`, `t("panel.space")` for 32, otherwise `chr(code).upper()`;
    - `owner(code: int, exclude: str | None = None) -> str | None` returns the action that already has the key;
    - `rebind(action: str, code: int) -> Shortcuts` (the caller checks `owner` first);
    - `with_enabled(enabled: bool) -> Shortcuts`;
    - `reset() -> Shortcuts` restores the default keys and keeps `enabled`;
    - `to_json() -> dict`, and `from_json(data: object) -> Shortcuts` as a classmethod.
- JSON shape: `{"enabled": bool, "keys": {action: int | null}}`. `from_json` falls back as follows:
  - not a dict: defaults;
  - `enabled` that is not `True`: `False`;
  - unknown actions: ignored;
  - a missing action, or a code that `normalize` rejects: the default for that action;
  - two actions on the same key: all keys back to the defaults.

- [ ] **Step 1: Write the failing tests** in `tests/test_shortcuts.py`

```python
class ShortcutsTests(unittest.TestCase):
    def test_off_by_default_and_nothing_fires(self):
        shortcuts = Shortcuts()
        self.assertFalse(shortcuts.enabled)
        self.assertIsNone(shortcuts.action_for(ord("v")))

    def test_defaults_when_turned_on(self):
        on = Shortcuts().with_enabled(True)
        self.assertEqual(on.action_for(ord("v")), ("vanish",))
        self.assertEqual(on.action_for(ord("V")), ("vanish",))  # Shift or caps lock.
        self.assertEqual(on.action_for(ord(" ")), ("recapture",))
        self.assertEqual(on.action_for(ord("3")), ("mode", "goku"))
        self.assertIsNone(on.action_for(ord("q")))  # Quit has no key by default.
        self.assertIsNone(on.key_for("quit"))

    def test_only_single_printable_keys(self):
        self.assertEqual(normalize(ord("Q")), ord("q"))
        for code in (27, 0xFF1B, 0xFFE1, 0xFFE9, 0xFF57, 0x1008FF12, -1, 127):
            self.assertIsNone(normalize(code))

    def test_rebind_and_owner(self):
        on = Shortcuts().with_enabled(True)
        self.assertEqual(on.owner(ord("v")), "vanish")
        self.assertIsNone(on.owner(ord("v"), exclude="vanish"))
        moved = on.rebind("vanish", ord("x"))
        self.assertEqual(moved.action_for(ord("x")), ("vanish",))
        self.assertIsNone(moved.action_for(ord("v")))
        self.assertEqual(moved.reset().key_for("vanish"), ord("v"))
        self.assertTrue(moved.reset().enabled)

    def test_labels(self):
        self.assertEqual(Shortcuts.label(ord("v")), "V")
        self.assertEqual(Shortcuts.label(32), t("panel.space"))
        self.assertEqual(Shortcuts.label(None), "")

    def test_json_round_trip_and_fallbacks(self):
        custom = Shortcuts().with_enabled(True).rebind("quit", ord("q"))
        self.assertEqual(Shortcuts.from_json(custom.to_json()), custom)
        self.assertEqual(Shortcuts.from_json(None), Shortcuts())
        self.assertEqual(Shortcuts.from_json({"enabled": "yes"}), Shortcuts())
        broken = Shortcuts.from_json({"enabled": True, "keys": {"vanish": 0xFFE1, "nonsense": 120}})
        self.assertEqual(broken.key_for("vanish"), ord("v"))
        clash = Shortcuts.from_json({"enabled": True, "keys": {"vanish": ord("r")}})
        self.assertEqual(clash, Shortcuts().with_enabled(True))
```

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_shortcuts -v`
Expected: ERROR, `ModuleNotFoundError: No module named 'vanishcam.shortcuts'`.

- [ ] **Step 3: Implement `vanishcam/shortcuts.py`** to the interface above. It imports `MODES` from `.modes` and `t` from `.i18n`.

- [ ] **Step 4: Run the tests and check that they pass**

Run: `.venv/bin/python -m unittest tests.test_shortcuts -v`
Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/shortcuts.py tests/test_shortcuts.py
git commit -m "Add a model for single-key shortcuts that are off by default"
```

---

### Task 2: Shortcuts are saved with the preferences

**Files:**
- Modify: `vanishcam/user_data.py` (`Settings`, `UserData.load`, `UserData.save`)
- Modify: `vanishcam/app.py` (`run_camera`): the five `user_data.save(Settings(...))` calls at lines 230, 295, 337, 356, and 379 become one closure.
- Test: `tests/test_control_panel.py` (`UserDataTests`)

**Interfaces:**
- Consumes: `Shortcuts`, `Shortcuts.to_json`, and `Shortcuts.from_json` (Task 1).
- Produces:
  - `Settings.shortcuts: Shortcuts = field(default_factory=Shortcuts)` as the fifth field, so positional calls still work.
  - In `run_camera`: a local `shortcuts: Shortcuts`, loaded from `saved.shortcuts`, and a closure `save_settings() -> None` that writes `Settings(effect.name, chosen, camera_name(), detached, shortcuts)`. Before the first `effect` exists (line 230), it uses `mode_name`. Every save in `run_camera` goes through `save_settings()`.

- [ ] **Step 1: Write the failing tests** in `UserDataTests`

```python
    def test_shortcuts_are_saved_and_loaded(self):
        custom = Shortcuts().with_enabled(True).rebind("quit", ord("q"))
        self.data.save(Settings("goku", None, None, False, custom))
        self.assertEqual(self.data.load().shortcuts, custom)

    def test_old_or_broken_shortcuts_load_as_defaults(self):
        self.data.root.mkdir(parents=True, exist_ok=True)
        for shortcuts in ({}, {"shortcuts": 7}, {"shortcuts": {"enabled": True, "keys": "x"}}):
            with self.subTest(shortcuts):
                self.data.settings_file.write_text(json.dumps({"mode": "naruto", **shortcuts}), encoding="utf-8")
                loaded = self.data.load()
                self.assertEqual(loaded.mode, "naruto")
                self.assertEqual(loaded.shortcuts.keys, Shortcuts().keys)
        self.assertFalse(self.data.load().shortcuts.enabled)
```

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_control_panel.UserDataTests -v`
Expected: FAIL or ERROR (`Settings` takes 5 positional arguments; there is no `shortcuts` attribute).

- [ ] **Step 3: Implement** the field, the `"shortcuts"` key in `save` (`settings.shortcuts.to_json()`), and `Shortcuts.from_json(data.get("shortcuts"))` in `load`. Then add the `save_settings()` closure in `app.py` and route the five save calls through it.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 196 tests`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/user_data.py vanishcam/app.py tests/test_control_panel.py
git commit -m "Save the keyboard shortcuts with the other preferences"
```

---

### Task 3: The video loop reads keys only through the shortcuts

**Files:**
- Modify: `vanishcam/app.py`:
  - delete `KEY_ACTIONS` (line 166);
  - the key handling around line 554;
  - `handle()` gains `("quit",)`.
- Modify: `vanishcam/i18n.py`:
  - `loop.panel_hint` becomes "Switch effect and background in the 'VanishCam' panel." / "Troque efeito e fundo no painel 'VanishCam'.", with no keys and without the `{count}` placeholder;
  - `panel.fullscreen_hint` becomes "ESC · EXIT FULLSCREEN" / "ESC · SAIR DA TELA CHEIA".
- Modify these tests:
  - `tests/test_video_output.py`: `app_mocks`, and the `waitKey` patches at lines 178 and 220.
  - `tests/test_control_panel.py`: lines 83 and 339.
  - `tests/test_naruto_effect.py`: line 284.
  - `tests/test_camera_reconnect.py`: line 356.
  - `tests/test_ash_effect.py`: line 142.
- Test: `tests/test_video_output.py`

**Interfaces:**
- Consumes:
  - `Shortcuts.action_for` and `ESCAPE_CODES` (Task 1);
  - the `shortcuts` local in `run_camera` (Task 2).
- Produces: the loop calls `key = cv2.waitKeyEx(5 if waiting else 1)`, then:
  1. if `key in ESCAPE_CODES`: leave fullscreen when `state.fullscreen` is on; otherwise do nothing;
  2. otherwise, if `shortcuts.action_for(key)` is not `None`, append that action;
  3. `panel.closed` still breaks the loop;
  4. `handle(("quit",))` makes the loop exit the same way closing the window does.
- Test seam: `app_mocks` gets a parameter `shortcuts=Shortcuts().with_enabled(True).rebind("quit", ord("q"))`, saved into the temporary `UserData` before the app starts, so the existing key scripts (`ord("v")`, `ord("q")`) keep working. Every `patch.object(app.cv2, "waitKey", ...)` becomes `"waitKeyEx"`.
- The registry-order assertions that used `KEY_ACTIONS` (`test_control_panel.py:83`, `test_ash_effect.py:142`) now check `Shortcuts().with_enabled(True).action_for(ord(str(number)))`.

- [ ] **Step 1: Write the failing tests** in `tests/test_video_output.py`, reusing the setup of the existing test at line 200 (`app_mocks`, `video_clock`, and `InlineRecognizer`) through a helper `run_keys(keys, shortcuts) -> (panel, mode_phases)`.

```python
    def test_typing_does_nothing_while_shortcuts_are_off(self):
        typed = [ord(c) for c in "vou sair rapidinho 12345"]
        panel, phases = self.run_keys(typed, shortcuts=Shortcuts(), close_after=len(typed))
        self.assertEqual(set(phases), {"ready"})
        self.assertEqual(panel.fullscreen_calls, [])
        self.assertFalse(any(name == "detach" for name in panel.detach_calls))

    def test_escape_leaves_fullscreen_and_never_quits(self):
        for escape in (27, 0xFF1B):
            with self.subTest(escape):
                keys = [ord("f"), escape, escape, 0, 0]
                panel, _ = self.run_keys(keys, shortcuts=Shortcuts().with_enabled(True), close_after=len(keys))
                self.assertEqual(panel.fullscreen_calls, [True, False])
                self.assertEqual(panel.loops, len(keys))  # Still running after both escapes.
```

`run_keys` ends the loop by setting `FakePanel.closed = True` after `close_after` calls to `waitKeyEx`. If `FakePanel` lacks `fullscreen_calls`, `detach_calls`, and `loops`, add them, recorded from `set_fullscreen`, `set_detached`, and `show`.

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_video_output -v`
Expected: the new tests FAIL. They either hang, fixed by the `close_after` guard, or show `v` vanishing and `ESC` quitting.

- [ ] **Step 3: Implement** the interface above. Delete `KEY_ACTIONS` and the `ord("q")` and `27` quit branch. Update the i18n strings. Update the existing tests to the new seam.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 198 tests`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/app.py vanishcam/i18n.py tests/
git commit -m "Read keys only through the shortcuts and stop quitting on Q or ESC"
```

---

### Task 4: LIVE and SETTINGS tabs in the panel

**Files:**
- Modify: `vanishcam/control_panel.py`: `PanelView`, `layout`, `PanelRenderer.render`, `ControlPanel.__init__`, `ControlPanel.show`, `ControlPanel.poll`, and the `ControlPanel` docstring's action list.
- Modify: `vanishcam/i18n.py`
- Test: `tests/test_control_panel.py` (`PanelRendererTests`, plus a new `ControlPanelTabsTests`)

**Interfaces:**
- Consumes: `Shortcuts`, `ACTIONS`, and `Shortcuts.label` (Task 1).
- Produces:
  - `PanelView.shortcuts: Shortcuts = Shortcuts()`.
  - `PanelRenderer.render(view, hover=None, menu=None, tab="live", capturing=None)`, where `capturing` is the action being rebound.
  - `ControlPanel.tab: str` (`"live"` at start) and `ControlPanel.capturing: str | None`. Both are part of the redraw key in `show`.
  - New region actions:
    - `("tab", "live" | "settings")`: `poll` handles it like `"menu"`. It sets `self.tab`, clears `capturing`, and is not returned.
    - `("capture", action)`: `poll` sets `self.capturing = action` (clicking the same row again clears it) and does not return it.
    - `("shortcuts_enabled", bool)` and `("reset_shortcuts",)`: returned to the app.
- **Top bar layout.** The tabs go inside the existing black bar (`BAR = 40`), so the panel does not grow. "VANISHCAM" stays on the left, the TALLY lamp on the right without the word "TALLY", and between them the tab labels `t("panel.tab_live")` / `t("panel.tab_settings")` in `mono-bold`, size 10. The active tab is drawn in `FACE` color with a 3 px `TALLY` underline at the bottom of the bar; inactive ones are in `PALE`.
- **SETTINGS tab body**, from `spots.effect` down:
  - section `"A"` with `t("panel.shortcuts")`;
  - a key-like toggle row: `t("panel.shortcuts_on")` / `t("panel.shortcuts_off")`, with a lamp that is `GREEN` when on and `RULE` when off. Clicking it sends `("shortcuts_enabled", not enabled)`;
  - one grey line under it, `t("panel.shortcuts_note")`: off means typing in this window does nothing, and ESC always leaves fullscreen;
  - one row per `ACTIONS` entry, 28 px high: the action name on the left and a key cap on the right with `Shortcuts.label`, or `"—"` when unbound;
    - action names: `t("panel.vanish")`, `t("panel.reappear")`, `t("panel.recapture")`, `t("panel.fullscreen")`, `t("panel.detach")`, `t(f"mode.{name}.title")`, and `t("panel.quit")`;
    - the row being captured shows `t("panel.press_key")` in `TALLY` instead of the key;
    - while shortcuts are off, the rows are `PALE` and not clickable;
  - a `t("panel.reset_shortcuts")` key at the bottom.
  - The footer keeps showing `view.message`.
- **LIVE tab:**
  - When shortcuts are on, the button legends come from the bindings: `Shortcuts.label(view.shortcuts.key_for(...))` in place of the literal `"V"`, `"R"`, `t("panel.space")`, `"F"`, `"D"`. When off, there are no legends.
  - The idle footer shows `t("panel.footer_on")` ("SHORTCUTS ON · SEE SETTINGS") or `t("panel.footer_off")` ("SHORTCUTS OFF · TURN ON IN SETTINGS"). This replaces `panel.footer`, which is deleted.
- **New i18n keys (EN / PT):**
  - `panel.tab_live`: LIVE / AO VIVO
  - `panel.tab_settings`: SETTINGS / AJUSTES
  - `panel.shortcuts`: KEYBOARD SHORTCUTS / ATALHOS DE TECLADO
  - `panel.shortcuts_on`: Shortcuts on / Atalhos ligados
  - `panel.shortcuts_off`: Shortcuts off / Atalhos desligados
  - `panel.shortcuts_note`
  - `panel.quit`: QUIT / SAIR
  - `panel.press_key`: PRESS A KEY / APERTE UMA TECLA
  - `panel.reset_shortcuts`: RESTORE DEFAULTS / RESTAURAR PADRÃO
  - `panel.footer_on`, `panel.footer_off`
  - `panel.shortcut_taken`: "{key} is already the shortcut for {action}" / "{key} já é o atalho de {action}". It is used in Task 5.

- [ ] **Step 1: Write the failing tests**

```python
class ControlPanelTabsTests(unittest.TestCase):
    def render(self, view, **kwargs):
        return PanelRenderer().render(view, **kwargs)

    def test_tabs_sit_in_the_top_bar_and_the_panel_does_not_grow(self):
        view = PanelView(mode="normal", status="ready")
        image, regions = self.render(view)
        tabs = [r for r in regions if r.action[0] == "tab"]
        self.assertEqual([r.action[1] for r in tabs], ["live", "settings"])
        self.assertTrue(all(r.box[3] <= BAR for r in tabs))
        self.assertEqual(image.shape[0], panel_height(view))

    def test_settings_tab_lists_every_action_and_the_switch(self):
        view = PanelView(mode="normal", status="ready", shortcuts=Shortcuts().with_enabled(True))
        _, regions = self.render(view, tab="settings")
        actions = [r.action for r in regions]
        self.assertIn(("shortcuts_enabled", False), actions)
        self.assertIn(("reset_shortcuts",), actions)
        self.assertEqual([a[1] for a in actions if a[0] == "capture"], list(ACTIONS))

    def test_rows_are_not_clickable_while_shortcuts_are_off(self):
        _, regions = self.render(PanelView(mode="normal", status="ready"), tab="settings")
        actions = [r.action for r in regions]
        self.assertIn(("shortcuts_enabled", True), actions)
        self.assertFalse(any(a[0] == "capture" for a in actions))

    def test_live_legends_follow_the_bindings_and_vanish_when_off(self):
        # Spy on the legend text: render with a patched ImageDraw.text and collect the strings drawn.
        on = Shortcuts().with_enabled(True).rebind("vanish", ord("x"))
        self.assertIn("X", self.drawn_text(PanelView(mode="normal", status="ready", shortcuts=on)))
        self.assertNotIn("V", self.drawn_text(PanelView(mode="normal", status="ready")))
```

In `PanelRendererTests`, add `test_poll_switches_tab_and_starts_capture`. With a `ControlPanel` (its constructor patched the way the existing `ControlPanel` tests do it), simulate clicks on the regions of `("tab", "settings")` and `("capture", "vanish")`. Then assert `panel.tab == "settings"`, `panel.capturing == "vanish"`, and `poll()` returning `[]`. A click on `("tab", "live")` clears `capturing`.

`drawn_text` is a small helper on the test class. It patches `ImageDraw.ImageDraw.text`, wrapping the original, records each `text` argument, and joins them. The test's assertion on `"V"` only checks that a single-letter legend `"V"` is absent: assert against the list of drawn strings, not a substring.

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_control_panel -v`
Expected: the new tests FAIL or ERROR (no `tab` parameter, no `shortcuts` field).

- [ ] **Step 3: Implement** the interfaces above, with the copy in both languages.

- [ ] **Step 4: Run all the tests, and look at the panel**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 203 tests`, `OK`.

Then render both tabs to PNG files in the scratchpad (with `PanelRenderer().render(...)` and `cv2.imwrite`) and look at them with the Read tool. The tabs must be readable in the bar, the SETTINGS rows must not overlap, and the panel height must be the same as LIVE's.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/control_panel.py vanishcam/i18n.py tests/test_control_panel.py
git commit -m "Add LIVE and SETTINGS tabs, with the keyboard shortcuts on SETTINGS"
```

---

### Task 5: Turning shortcuts on and rebinding a key

**Files:**
- Modify: `vanishcam/app.py`: `handle()`, the key handling in the loop, and the `PanelView(...)` call, which gains `shortcuts=shortcuts`.
- Test: `tests/test_control_panel.py`, in the class around line 300 that drives `run_camera` with `app_mocks` and scripted panel actions.

**Interfaces:**
- Consumes:
  - `Shortcuts.with_enabled`, `owner`, `rebind`, `reset`, `normalize`, and `ESCAPE_CODES` (Task 1);
  - `save_settings()` (Task 2);
  - `ControlPanel.capturing` and the actions `("shortcuts_enabled", bool)` and `("reset_shortcuts",)` (Task 4).
- Produces:
  - `handle(("shortcuts_enabled", value))`: `shortcuts = shortcuts.with_enabled(value)`, then `save_settings()`.
  - `handle(("reset_shortcuts",))`: `shortcuts = shortcuts.reset()`, then `save_settings()`.
  - **Key capture.** At the top of key handling, when `panel.capturing` is set and `key != -1`:
    - `ESC`: `panel.capturing = None`;
    - `normalize(key) is None` (Shift, Alt, media keys): ignore it and keep waiting;
    - the key belongs to another action, per `owner(code, exclude=panel.capturing)`: `notify(t("panel.shortcut_taken", key=Shortcuts.label(code), action=<that action's name>), now)` and keep waiting;
    - otherwise: `shortcuts = shortcuts.rebind(panel.capturing, code)`, `panel.capturing = None`, `save_settings()`.
    - A key handled by the capture is never dispatched as an action.
  - The action-name lookup used by the message and by the SETTINGS rows lives in `control_panel.py` as `action_name(action: str) -> str`. Task 4 introduced it, or this task extracts it from the row rendering.
  - `FakePanel` gains `capturing = None`.

- [ ] **Step 1: Write the failing tests**

```python
    def test_turning_shortcuts_on_and_rebinding_is_saved(self):
        # Panel actions: ("shortcuts_enabled", True); then capturing="vanish" while the key "x" comes.
        saved = self.run_with(actions=[("shortcuts_enabled", True)], capture=("vanish", [ord("x")]),
                              shortcuts=Shortcuts())
        self.assertTrue(saved.shortcuts.enabled)
        self.assertEqual(saved.shortcuts.key_for("vanish"), ord("x"))

    def test_modifier_keys_are_ignored_while_capturing(self):
        saved = self.run_with(capture=("vanish", [0xFFE1, 0xFFE9, ord("x")]),
                              shortcuts=Shortcuts().with_enabled(True))
        self.assertEqual(saved.shortcuts.key_for("vanish"), ord("x"))

    def test_a_key_already_used_is_refused(self):
        saved = self.run_with(capture=("vanish", [ord("r")]), shortcuts=Shortcuts().with_enabled(True))
        self.assertEqual(saved.shortcuts.key_for("vanish"), ord("v"))
        self.assertIn(Shortcuts.label(ord("r")), self.last_message)

    def test_captured_key_is_not_dispatched(self):
        # Rebinding vanish to "1" must not also switch to effect 1... "1" is taken, so use "x" bound to nothing:
        saved = self.run_with(capture=("detach", [ord("x")]), shortcuts=Shortcuts().with_enabled(True))
        self.assertEqual(self.detach_calls, [])

    def test_switching_effect_keeps_the_shortcuts(self):
        custom = Shortcuts().with_enabled(True).rebind("quit", ord("q"))
        saved = self.run_with(actions=[("mode", "naruto")], shortcuts=custom)
        self.assertEqual(saved.mode, "naruto")
        self.assertEqual(saved.shortcuts, custom)
```

`run_with` builds on the existing harness in that class: `app_mocks(..., shortcuts=...)`, scripted panel actions, and `FakePanel.capturing` set to `capture[0]` while `capture[1]` is fed to `waitKeyEx`, then closed. It returns `user_data.load()` and keeps the last `notify` text in `self.last_message`, read from the `PanelView.message` passed to `show`.

- [ ] **Step 2: Run them and check that they fail**

Run: `.venv/bin/python -m unittest tests.test_control_panel -v`
Expected: the new tests FAIL (actions not handled, capture not implemented).

- [ ] **Step 3: Implement** the interface above.

- [ ] **Step 4: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 208 tests`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add vanishcam/app.py vanishcam/control_panel.py tests/
git commit -m "Turn shortcuts on and rebind keys from the SETTINGS tab"
```

---

### Task 6: Docs

**Files:**
- Modify:
  - `docs/control-panel.md`: the tabs, the SETTINGS tab, the "Keyboard shortcuts" section (off by default, the defaults when on, ESC, closing the window quits), and the screenshot caveat.
  - `docs/effects.md`: the mentions of the `V`/`R` keys and `1`–`5`.
  - `README.md`: step 3 says "Press `R`", and the features list.
  - `docs/configuration.md`: any mention of `Q`, and a `--help` text that names keys.
- Modify: `vanishcam/i18n.py`, for any `cli.*` text that names keys.

- [ ] **Step 1: Find every key mention left**

Run: `grep -rnE "\`(V|R|Q|F|D|SPACE|ESC)\`|1–5|1-5|Q quits|Q fecha" README.md docs/*.md vanishcam/i18n.py`
Expected: a list to go through. Each mention either describes shortcuts as off by default and configurable in SETTINGS, or is removed.

- [ ] **Step 2: Update the docs.** Note in `docs/control-panel.md` that `docs/panel.png` shows the panel before the tabs; replacing the screenshot needs the real app and is left to the owner.

- [ ] **Step 3: Run all the tests**

Run: `.venv/bin/python -m unittest`
Expected: `Ran 208 tests`, `OK`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/ vanishcam/i18n.py
git commit -m "Document the SETTINGS tab and the shortcuts being off by default"
```
