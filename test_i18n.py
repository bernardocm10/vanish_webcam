import string
import unittest
from unittest.mock import patch

import i18n
import vanish_cam as app
from i18n import STRINGS, detect_language, normalize, set_language, t


def placeholders(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


class TranslationTableTests(unittest.TestCase):
    def test_every_language_has_the_same_keys_and_placeholders(self):
        english = STRINGS["en"]
        for code in i18n.LANGUAGES:
            with self.subTest(language=code):
                self.assertEqual(set(STRINGS[code]), set(english))
                for key, text in STRINGS[code].items():
                    self.assertEqual(placeholders(text), placeholders(english[key]), key)


class DetectionTests(unittest.TestCase):
    def test_locale_names_from_every_system(self):
        for name, expected in (
            ("pt_BR.UTF-8", "pt"), ("pt-BR", "pt"), ("pt_PT", "pt"), ("Portuguese_Brazil", "pt"),
            ("en_US.UTF-8", "en"), ("English_United States", "en"), ("it_IT.UTF-8", "en"),
            ("C.UTF-8", None), ("POSIX", None), ("", None), (None, None),
        ):
            with self.subTest(name=name):
                self.assertEqual(normalize(name), expected)

    def test_environment_variable_wins_over_the_system(self):
        environ = {"VANISHCAM_LANG": "en", "LANG": "pt_BR.UTF-8"}
        self.assertEqual(detect_language(environ, "Linux"), "en")

    def test_gettext_order_and_c_locale_is_skipped(self):
        self.assertEqual(detect_language({"LANGUAGE": "pt_BR:en", "LANG": "en_US.UTF-8"}, "Linux"), "pt")
        self.assertEqual(detect_language({"LC_ALL": "C", "LANG": "pt_BR.UTF-8"}, "Linux"), "pt")

    def test_unknown_system_falls_back_to_english(self):
        with patch.object(i18n.locale, "getlocale", return_value=(None, None)):
            self.assertEqual(detect_language({}, "Linux"), "en")

    def test_windows_uses_the_ui_language(self):
        with patch.object(i18n, "windows_language", return_value="pt_BR"):
            self.assertEqual(detect_language({}, "Windows"), "pt")


class CommandLineLanguageTests(unittest.TestCase):
    def tearDown(self):
        set_language(None)

    def test_lang_option_switches_messages(self):
        app.parse_args(["--lang", "pt"])
        self.assertEqual(t("panel.vanish"), "SUMIR")
        app.parse_args(["--lang", "en"])
        self.assertEqual(t("panel.vanish"), "VANISH")

    def test_help_text_follows_the_language(self):
        set_language("en")
        with patch("sys.stdout") as stdout, self.assertRaises(SystemExit):
            app.parse_args(["--lang", "pt", "--help"])
        printed = "".join(call.args[0] for call in stdout.write.call_args_list)
        self.assertIn("Índice da webcam", printed)

    def test_mode_titles_follow_the_language(self):
        from modes import ParticlesMode

        mode = ParticlesMode(segment=lambda frame, timestamp: None)
        set_language("pt")
        self.assertEqual(mode.title, "Partículas")
        set_language("en")
        self.assertEqual(mode.title, "Particles")


if __name__ == "__main__":
    unittest.main()
