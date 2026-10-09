"""macOS hotkey behaviour. The platform flags are patched, so these tests run on every OS."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

Qt = pytest.importorskip("PyQt6.QtCore").Qt

from app.controllers import hotkeys as hotkey_controller  # noqa: E402
from app.core import hotkeys  # noqa: E402
from app.hotkeys import key_tokens  # noqa: E402


@pytest.fixture
def macos(monkeypatch):
    for module in (key_tokens, hotkey_controller):
        monkeypatch.setattr(module, "is_MACOS", True)
        monkeypatch.setattr(module, "is_WINDOWS", False)


def key(name=None, char=None, vk=None):
    return SimpleNamespace(name=name, char=char, vk=vk)


def test_hardware_keycodes_are_not_read_as_windows_virtual_keys(macos):
    # macOS keycode 0x11 is 't'; the Windows table would read it as Ctrl and fire Ctrl hotkeys while typing.
    assert key_tokens.pynput_key_tokens(key(char="t", vk=0x11)) == {"t"}


def test_qt_capture_records_the_physical_control_and_command_keys(macos):
    assert key_tokens.qt_key_tokens(int(Qt.Key.Key_Plus), Qt.KeyboardModifier.MetaModifier, "+") == {"<ctrl>", "<plus>"}
    assert key_tokens.qt_key_tokens(int(Qt.Key.Key_K), Qt.KeyboardModifier.ControlModifier, "k") == {"<cmd>", "k"}


def test_injected_plus_does_not_count_as_a_special_key(macos):
    assert not key_tokens.injected_event_counts({"<plus>"})
    assert key_tokens.injected_event_counts({"<f6>"})


def post_rephrase_controller():
    recording = SimpleNamespace(is_recording=False, push_to_talk_active=False)
    controller = hotkey_controller.HotkeyController(SimpleNamespace(config={}), recording, Mock())
    controller.bindings = controller._manual_bindings = [hotkeys.parse_hotkey_binding("<ctrl>+<plus>", "post_rephrase")]
    fired = []
    controller.action_triggered.connect(lambda action, _ns: fired.append(action))
    return controller, fired


def test_post_rephrase_fires_only_after_its_keys_are_released(macos):
    controller, fired = post_rephrase_controller()
    ctrl, plus = key(name="ctrl"), key(char="+")

    controller._on_press(ctrl)
    controller._on_press(plus)
    controller._on_release(plus)
    assert fired == []  # Ctrl still held: the simulated Cmd+C would arrive as Ctrl+Cmd+C
    controller._on_release(ctrl)
    assert fired == ["post_rephrase"]


def test_post_rephrase_release_after_the_timeout_is_dropped(macos, monkeypatch):
    controller, fired = post_rephrase_controller()
    ctrl, plus = key(name="ctrl"), key(char="+")
    controller._on_press(ctrl)
    controller._on_press(plus)
    controller._on_release(plus)
    later = hotkey_controller.time.monotonic() + hotkey_controller._RELEASE_ACTION_TIMEOUT_S + 1
    monkeypatch.setattr(hotkey_controller.time, "monotonic", lambda: later)
    controller._on_release(ctrl)  # e.g. a key-up that got lost and only shows up much later
    assert fired == []
