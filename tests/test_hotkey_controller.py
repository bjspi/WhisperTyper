"""HotkeyController behaviour that needs Qt (platform flags patched; runs on every OS)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

QtCore = pytest.importorskip("PyQt6.QtCore")
QtGui = pytest.importorskip("PyQt6.QtGui")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from app.controllers import hotkeys as hotkey_controller  # noqa: E402
from app.core import hotkeys  # noqa: E402
from app.hotkeys import key_tokens  # noqa: E402

Qt, QEvent = QtCore.Qt, QtCore.QEvent


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def controller():
    recording = SimpleNamespace(is_recording=False, push_to_talk_active=False)
    ctx = SimpleNamespace(config={}, tr=lambda key, **_kw: key)
    return hotkey_controller.HotkeyController(ctx, recording, Mock())


def test_own_injected_keystrokes_are_never_suppressed():
    hk = controller()
    hk._manual_bindings = [hotkeys.parse_hotkey_binding("<ctrl>+<shift>+c", "post_rephrase")]
    hk._pressed = {"<ctrl>", "<shift>"}
    hk._manual_listener = SimpleNamespace(_suppress=None)
    c_key = 0x43
    hk._win32_event_filter(0x0100, SimpleNamespace(vkCode=c_key, flags=0))
    assert hk._manual_listener._suppress is True  # the physical hotkey press is swallowed
    hk._win32_event_filter(0x0100, SimpleNamespace(vkCode=c_key, flags=0x10))  # LLKHF_INJECTED
    assert hk._manual_listener._suppress is False  # the app's own simulated Ctrl+C reaches the target


def test_macos_capture_records_a_chord_until_the_first_release(qapp, monkeypatch):
    for module in (key_tokens, hotkey_controller):
        monkeypatch.setattr(module, "is_MACOS", True)
        monkeypatch.setattr(module, "is_WINDOWS", False)
    hk = controller()
    hk.restart = Mock()
    field, button = QtWidgets.QLineEdit(), QtWidgets.QPushButton()
    hk.start_capture(field, button)

    def send(kind, key, auto_repeat=False):
        event = QtGui.QKeyEvent(kind, key, Qt.KeyboardModifier.NoModifier, "", auto_repeat)
        assert hk.eventFilter(field, event)

    send(QEvent.Type.KeyPress, Qt.Key.Key_F6)
    send(QEvent.Type.KeyPress, Qt.Key.Key_F6, auto_repeat=True)
    send(QEvent.Type.KeyRelease, Qt.Key.Key_F6, auto_repeat=True)  # auto-repeat must not end it
    assert hk.capturing
    send(QEvent.Type.KeyPress, Qt.Key.Key_F7)
    send(QEvent.Type.KeyRelease, Qt.Key.Key_F7)
    assert not hk.capturing
    assert hotkeys.normalize_hotkey_string(field.text()) in ("<f6>+<f7>", "<f7>+<f6>")
