import unittest

from vanishcam.i18n import t
from vanishcam.shortcuts import Shortcuts, normalize


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
        # The same key saved for two actions: a hand-edited file, back to the defaults.
        clash = Shortcuts.from_json({"enabled": True, "keys": {"vanish": ord("r"), "reappear": ord("r")}})
        self.assertEqual(clash, Shortcuts().with_enabled(True))

    def test_a_default_never_takes_a_saved_key(self):
        # Saved before an action existed (a new effect): its default key is already one of the owner's.
        saved = {"vanish": ord("x"), "quit": ord("5")}
        loaded = Shortcuts.from_json({"enabled": True, "keys": saved})
        self.assertEqual(loaded.key_for("vanish"), ord("x"))
        self.assertEqual(loaded.key_for("quit"), ord("5"))
        self.assertIsNone(loaded.key_for("mode.ash"))  # Its default 5 was taken: no key, not a reset.
        self.assertEqual(loaded.key_for("reappear"), ord("r"))


if __name__ == "__main__":
    unittest.main()
