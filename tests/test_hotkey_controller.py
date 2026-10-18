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


def controller(config=None, hotkey="<ctrl>+<f8>"):
    recording = SimpleNamespace(is_recording=False, push_to_talk_active=False)
    ctx = SimpleNamespace(config=config or {}, tr=lambda key, **_kw: key)
    hk = hotkey_controller.HotkeyController(ctx, recording, Mock())
    hk.bindings = [hotkeys.parse_hotkey_binding(hotkey, "transcription")]
    hk._manual_bindings = list(hk.bindings)
    return hk


def pynput_key(name):
    return SimpleNamespace(name=name, char=None, vk=None)


def test_push_to_talk_quick_tap_stops_before_the_gui_thread_started_recording(qapp):
    hk = controller({"push_to_talk": True})
    actions = []
    hk.action_triggered.connect(lambda action, _ns: actions.append(action))
    hk._on_press(pynput_key("ctrl"))
    hk._on_press(pynput_key("f8"))
    assert actions == ["transcription"] and hk._recording.push_to_talk_active
    # The key-up arrives on the listener thread while is_recording is still False.
    hk._on_release(pynput_key("f8"))
    assert actions == ["transcription", "stop_transcription"]
    assert not hk._recording.push_to_talk_active


def test_stale_pressed_keys_are_dropped_before_a_new_press(qapp, monkeypatch):
    hk = controller()
    actions = []
    hk.action_triggered.connect(lambda action, _ns: actions.append(action))
    clock = [100.0]
    monkeypatch.setattr(hotkey_controller.time, "monotonic", lambda: clock[0])
    hk._on_press(pynput_key("alt"))  # its key-up gets lost (UAC prompt, lock screen)
    clock[0] += hotkey_controller._STALE_PRESSED_S + 1
    hk._on_press(pynput_key("ctrl"))
    assert hk._pressed == {"<ctrl>"}
    hk._on_press(pynput_key("f8"))
    assert actions == ["transcription"]


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
