"""macOS-only regression tests for the Qt-based hotkey capture flow.

Every test patches ``is_MACOS``, so the suite also runs on Windows and Linux CI.
The platform-independent hotkey engine is covered by ``tests/test_hotkey_guard.py``.
"""
from __future__ import annotations

from typing import Any, Set

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent

from app.core import hotkeys
from tests.hotkey_harness import HotkeyHarness


class FakeWidget:
    """Small QWidget stand-in for testing the capture lifecycle."""

    def __init__(self) -> None:
        """Initialize observable widget state."""
        self.text = ""
        self.enabled = True
        self.filter_installed = False
        self.keyboard_grabbed = False

    def setText(self, text: str) -> None:
        self.text = text

    def selectAll(self) -> None:
        pass

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def clear(self) -> None:
        self.text = ""

    def installEventFilter(self, _event_filter: Any) -> None:
        self.filter_installed = True

    def removeEventFilter(self, _event_filter: Any) -> None:
        self.filter_installed = False

    def setFocus(self, _reason: Any) -> None:
        pass

    def grabKeyboard(self) -> None:
        self.keyboard_grabbed = True

    def releaseKeyboard(self) -> None:
        self.keyboard_grabbed = False


class FakeTranslator:
    def tr(self, key: str) -> str:
        return key


class CaptureHarness(HotkeyHarness):
    """Capture-flow harness that records destructive listener lifecycle calls."""

    def __init__(self) -> None:
        """Initialize fake controls and listener lifecycle counters."""
        super().__init__()
        self.set_hotkey_button = FakeWidget()
        self.set_pr_hotkey_button = FakeWidget()
        self.hotkey_display = FakeWidget()
        self.pr_hotkey_display = FakeWidget()
        self.capturing_for_widget = None
        self.capturing_button = None
        self.captured_keys: Set[str] = set()
        self.hotkey_capture_listener = None
        self.translator = FakeTranslator()
        self.stop_calls = 0
        self.init_calls = 0

    def sender(self) -> FakeWidget:
        return self.set_pr_hotkey_button

    def _check_and_warn_macos_permissions(self, _permission: str) -> None:
        pass

    def _stop_hotkey_listeners(self) -> None:
        self.stop_calls += 1

    def init_manual_hotkey_listener(self) -> None:
        self.init_calls += 1


def _key_event(
    event_type: QEvent.Type,
    key: Qt.Key,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    text: str = "",
    *,
    auto_repeat: bool = False,
) -> QKeyEvent:
    return QKeyEvent(event_type, key, modifiers, text, auto_repeat, 1)


def test_global_listener_is_logically_muted_during_macos_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_MACOS", True)
    harness = HotkeyHarness()
    harness._hotkey_capture_suppresses_global_actions = True

    harness._on_hotkey_press({"<ctrl>"})
    harness._on_hotkey_press({"c"})
    harness._on_hotkey_release({"c"})

    assert harness.hotkey_action_signal.emissions == []
    assert harness.pressed_hotkey_tokens == set()
    assert harness.active_hotkey_actions == set()


def test_capture_mute_flag_is_ignored_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows and Linux must keep dispatching even if the flag were ever set."""
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_MACOS", False)
    harness = HotkeyHarness()
    harness._hotkey_capture_suppresses_global_actions = True

    harness._on_hotkey_press({"<ctrl>"})
    harness._on_hotkey_press({"c"})
    harness._on_hotkey_release({"c"})
    harness._on_hotkey_release({"<ctrl>"})

    assert harness.hotkey_action_signal.emissions == ["post_rephrase"]


def test_macos_capture_keeps_global_listener_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_MACOS", True)
    harness = CaptureHarness()

    harness.start_hotkey_capture()

    assert harness.stop_calls == 0
    assert harness._hotkey_capture_suppresses_global_actions is True
    assert harness.capturing_for_widget is harness.pr_hotkey_display
    assert harness.pr_hotkey_display.keyboard_grabbed is True

    harness._finish_hotkey_capture()

    assert harness.init_calls == 0
    assert harness._hotkey_capture_suppresses_global_actions is False
    assert harness.pr_hotkey_display.keyboard_grabbed is False


class FakeListener:
    """Stand-in for the temporary pynput capture listener used off macOS."""

    def __init__(self, **_kwargs: Any) -> None:
        """Record that the listener was constructed but never really started."""
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


def test_capture_off_macos_still_stops_and_restarts_listeners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Windows/Linux capture lifecycle is untouched by the macOS path."""
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_MACOS", False)
    monkeypatch.setattr("app.mixins.hotkey_mixin.keyboard.Listener", FakeListener)
    harness = CaptureHarness()

    harness.start_hotkey_capture()
    assert harness.stop_calls == 1
    assert harness.hotkey_capture_listener.started is True

    harness._finish_hotkey_capture()
    assert harness.init_calls == 1


def test_macos_capture_collects_modifier_chord_until_key_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_MACOS", True)
    harness = CaptureHarness()
    harness.start_hotkey_capture()

    assert harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Meta,
            Qt.KeyboardModifier.MetaModifier,
        ),
    )
    assert harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Comma,
            Qt.KeyboardModifier.MetaModifier,
            ",",
        ),
    )

    assert harness.pr_hotkey_display.text == "<ctrl>+,"
    assert harness.capturing_for_widget is harness.pr_hotkey_display

    assert harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(
            QEvent.Type.KeyRelease,
            Qt.Key.Key_Comma,
            Qt.KeyboardModifier.MetaModifier,
            ",",
        ),
    )
    assert harness.capturing_for_widget is None
    assert harness.pr_hotkey_display.text == "<ctrl>+,"


def test_macos_capture_supports_two_non_modifier_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_MACOS", True)
    harness = CaptureHarness()
    harness.start_hotkey_capture()

    harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(QEvent.Type.KeyPress, Qt.Key.Key_F6),
    )
    harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(QEvent.Type.KeyPress, Qt.Key.Key_F7),
    )

    assert harness.pr_hotkey_display.text == "<f6>+<f7>"
    assert harness.capturing_for_widget is harness.pr_hotkey_display

    harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(QEvent.Type.KeyRelease, Qt.Key.Key_F7),
    )
    assert harness.capturing_for_widget is None
    assert harness.pr_hotkey_display.text == "<f6>+<f7>"


def test_macos_capture_keeps_plus_key_unambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_MACOS", True)
    harness = CaptureHarness()
    harness.start_hotkey_capture()

    harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Meta,
            Qt.KeyboardModifier.MetaModifier,
        ),
    )
    harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Plus,
            Qt.KeyboardModifier.MetaModifier,
            "+",
        ),
    )

    assert harness.pr_hotkey_display.text == "<ctrl>+<plus>"

    harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(
            QEvent.Type.KeyRelease,
            Qt.Key.Key_Plus,
            Qt.KeyboardModifier.MetaModifier,
            "+",
        ),
    )
    binding = hotkeys.parse_hotkey_binding(harness.pr_hotkey_display.text, "post_rephrase")
    assert binding is not None
    assert binding["tokens"] == {"<ctrl>", "<plus>"}


def test_macos_capture_maps_physical_command_to_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_MACOS", True)
    harness = CaptureHarness()
    harness.start_hotkey_capture()

    harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Control,
            Qt.KeyboardModifier.ControlModifier,
        ),
    )
    harness.eventFilter(
        harness.pr_hotkey_display,
        _key_event(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Plus,
            Qt.KeyboardModifier.ControlModifier,
            "+",
        ),
    )

    assert harness.pr_hotkey_display.text == "<cmd>+<plus>"
