"""Key-event to hotkey-token conversion for the pynput listener and the Qt capture field."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

Qt = pytest.importorskip("PyQt6.QtCore").Qt

from app.hotkeys import key_tokens  # noqa: E402
from app.hotkeys.key_tokens import pynput_key_tokens, qt_key_tokens  # noqa: E402


def key(name=None, char=None, vk=None):
    return SimpleNamespace(name=name, char=char, vk=vk)


@pytest.mark.parametrize("event,expected", [
    (key(name="f9"), {"<f9>"}),
    (key(name="media_next"), {"<f9>"}),  # macOS media mode of the F-row
    (key(char="A"), {"a"}),
    (key(name="alt_gr"), {"<alt_gr>", "<ctrl>", "<alt>"}),
    (key(char="+"), {"<plus>"}),
    (key(char="\x03"), set()),  # Ctrl+C as reported on Windows: a control char is not a key name
    (key(char=" "), set()),
])
def test_pynput_events(event, expected):
    assert pynput_key_tokens(event) == expected


def test_windows_ctrl_letter_is_named_by_its_virtual_key(monkeypatch):
    monkeypatch.setattr(key_tokens, "is_WINDOWS", True)
    assert pynput_key_tokens(key(char="\x03", vk=0x43)) == {"c"}


def test_qt_capture_maps_modifiers_function_keys_and_punctuation(monkeypatch):
    monkeypatch.setattr(key_tokens, "is_MACOS", False)  # macOS swaps Ctrl/Meta, see test_macos_hotkeys.py
    ctrl_shift = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
    assert qt_key_tokens(int(Qt.Key.Key_F5), ctrl_shift, "") == {"<ctrl>", "<shift>", "<f5>"}
    assert qt_key_tokens(int(Qt.Key.Key_Shift), Qt.KeyboardModifier.ShiftModifier, "") == {"<shift>"}
    assert qt_key_tokens(int(Qt.Key.Key_Slash), Qt.KeyboardModifier.NoModifier, "") == {"/"}
