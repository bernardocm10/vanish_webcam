"""Keyboard shortcuts: off by default, one single key per action when on.

The OpenCV window cannot tell Ctrl or Alt combinations from plain keys on every
system (Qt drops Ctrl, and sends Shift and Alt as keys of their own), so each
action gets one key, and the owner turns them on only if they want them: typing
with this window focused by mistake then does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .i18n import t
from .modes import MODES

# ESC as Win32 reports it, and as the X keysym Qt reports on Linux.
ESCAPE_CODES = (27, 0xFF1B)

ACTIONS: tuple[str, ...] = (
    "vanish", "reappear", "recapture", "fullscreen", "detach", *(f"mode.{name}" for name in MODES), "quit",
)

DEFAULT_KEYS: dict[str, int | None] = {
    "vanish": ord("v"),
    "reappear": ord("r"),
    "recapture": ord(" "),
    "fullscreen": ord("f"),
    "detach": ord("d"),
    **{f"mode.{name}": ord(str(number)) if number <= 9 else None for number, name in enumerate(MODES, start=1)},
    "quit": None,
}


def normalize(code: int) -> int | None:
    """The key as stored: printable ASCII, letters in lowercase; None for anything else."""
    if ord("A") <= code <= ord("Z"):
        return code + 32
    return code if 32 <= code <= 126 else None


def action_tuple(action: str) -> tuple:
    """The panel action for a shortcut: "mode.goku" -> ("mode", "goku")."""
    kind, _, argument = action.partition(".")
    return (kind, argument) if argument else (kind,)


@dataclass(frozen=True)
class Shortcuts:
    enabled: bool = False
    keys: tuple[tuple[str, int | None], ...] = tuple(DEFAULT_KEYS.items())

    def key_for(self, action: str) -> int | None:
        return dict(self.keys).get(action)

    def action_for(self, code: int) -> tuple | None:
        code = normalize(code)
        if not self.enabled or code is None:
            return None
        action = self.owner(code)
        return action_tuple(action) if action is not None else None

    @staticmethod
    def label(code: int | None) -> str:
        if code is None:
            return ""
        return t("panel.space") if code == 32 else chr(code).upper()

    def owner(self, code: int, exclude: str | None = None) -> str | None:
        return next((action for action, key in self.keys if key == code and action != exclude), None)

    def rebind(self, action: str, code: int) -> Shortcuts:
        return Shortcuts(self.enabled, tuple((name, code if name == action else key) for name, key in self.keys))

    def with_enabled(self, enabled: bool) -> Shortcuts:
        return Shortcuts(enabled, self.keys)

    def reset(self) -> Shortcuts:
        return Shortcuts(self.enabled)

    def to_json(self) -> dict:
        return {"enabled": self.enabled, "keys": dict(self.keys)}

    @classmethod
    def from_json(cls, data: object) -> Shortcuts:
        """Anything missing or invalid falls back to its default, unless the owner saved that key
        for another action (a new effect after an upgrade): then it has none. The same key saved
        for two actions (a hand-edited file) puts back all the defaults."""
        if not isinstance(data, dict):
            return cls()
        enabled = data.get("enabled") is True
        saved = data.get("keys") if isinstance(data.get("keys"), dict) else {}
        chosen = {}
        for action in DEFAULT_KEYS:
            code = saved.get(action, ...)
            if code is None or (isinstance(code, int) and not isinstance(code, bool) and normalize(code) == code):
                chosen[action] = code
        taken = [code for code in chosen.values() if code is not None]
        if len(taken) != len(set(taken)):
            return cls(enabled)
        keys = tuple(
            (action, chosen[action] if action in chosen else None if default in taken else default)
            for action, default in DEFAULT_KEYS.items()
        )
        return cls(enabled, keys)
