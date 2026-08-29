"""Regression tests for self-generated keyboard-event suppression.

These cover the platform-independent hotkey engine: the guard around synthetic
clipboard input, deferred dispatch on release, chord matching, and the fact that
the Windows virtual-key table is only consulted on Windows. macOS-only capture
behaviour is covered in ``tests/test_macos_hotkeys.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core import hotkeys
from app.mixins.hotkey_mixin import HotkeyMixin
from tests.hotkey_harness import HotkeyHarness


def test_synthetic_copy_cannot_trigger_post_rephrase() -> None:
    harness = HotkeyHarness()
    harness._suppress_hotkeys_for_simulated_input("clipboard_c", duration_seconds=1.0)

    harness._on_hotkey_press({"<ctrl>"})
    harness._on_hotkey_press({"c"})

    assert harness.hotkey_action_signal.emissions == []
    assert harness.pressed_hotkey_tokens == set()


def test_real_hotkey_still_triggers_outside_guard_window() -> None:
    harness = HotkeyHarness()

    harness._on_hotkey_press({"<ctrl>"})
    harness._on_hotkey_press({"c"})
    assert harness.hotkey_action_signal.emissions == []

    harness._on_hotkey_release({"c"})
    assert harness.hotkey_action_signal.emissions == []

    harness._on_hotkey_release({"<ctrl>"})

    assert harness.hotkey_action_signal.emissions == ["post_rephrase"]


def test_control_plus_hotkey_triggers_with_layout_alias() -> None:
    harness = HotkeyHarness()
    binding = hotkeys.parse_hotkey_binding("<ctrl>+<plus>", "post_rephrase")
    assert binding is not None
    harness.hotkey_bindings = [binding]
    harness.manual_hotkey_bindings = [binding]

    harness._on_hotkey_press({"<ctrl>"})
    harness._on_hotkey_press({"<plus>", "e"})
    assert harness.hotkey_action_signal.emissions == []

    harness._on_hotkey_release({"<plus>", "e"})
    assert harness.hotkey_action_signal.emissions == []

    harness._on_hotkey_release({"<ctrl>"})

    assert harness.hotkey_action_signal.emissions == ["post_rephrase"]


def test_arming_simulated_input_guard_clears_completed_chord() -> None:
    harness = HotkeyHarness()
    harness.pressed_hotkey_tokens = {"<ctrl>", "<plus>"}
    harness.active_hotkey_actions = {"post_rephrase"}
    harness.deferred_hotkey_actions = {"post_rephrase"}

    harness._suppress_hotkeys_for_simulated_input("clipboard_c")

    assert harness.pressed_hotkey_tokens == set()
    assert harness.active_hotkey_actions == set()
    assert harness.deferred_hotkey_actions == set()


def test_release_inside_guard_keeps_cleared_listener_state_empty() -> None:
    harness = HotkeyHarness()
    harness.pressed_hotkey_tokens = {"<ctrl>", "c"}
    harness.active_hotkey_actions = {"post_rephrase"}
    harness._suppress_hotkeys_for_simulated_input("clipboard_c", duration_seconds=1.0)

    harness._on_hotkey_release({"c"})

    assert harness.pressed_hotkey_tokens == set()
    assert harness.active_hotkey_actions == set()


def test_non_windows_keycodes_are_not_read_as_windows_virtual_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VK_TO_TOKEN describes Windows codes; macOS and Linux report different numbers."""
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_WINDOWS", False)
    harness = HotkeyHarness()

    assert HotkeyMixin._key_to_hotkey_tokens(
        harness,
        SimpleNamespace(name=None, char="t", vk=0x11)
    ) == {"t"}
    assert HotkeyMixin._key_to_hotkey_tokens(
        harness,
        SimpleNamespace(name=None, char="c", vk=0x08)
    ) == {"c"}
    assert HotkeyMixin._key_to_hotkey_tokens(
        harness,
        SimpleNamespace(name=None, char="n", vk=0x2D)
    ) == {"n"}
    assert HotkeyMixin._key_to_hotkey_tokens(
        harness,
        SimpleNamespace(name=None, char="+", vk=0x45)
    ) == {"<plus>"}


def test_windows_keycodes_are_still_read_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Windows path must keep resolving virtual-key codes."""
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_WINDOWS", True)
    harness = HotkeyHarness()

    assert HotkeyMixin._key_to_hotkey_tokens(
        harness,
        SimpleNamespace(name=None, char=None, vk=0x78)
    ) == {"<f9>"}


def test_realistic_control_plus_events_trigger_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.mixins.hotkey_mixin.is_WINDOWS", False)
    harness = HotkeyHarness()
    binding = hotkeys.parse_hotkey_binding("<ctrl>+<plus>", "post_rephrase")
    assert binding is not None
    harness.hotkey_bindings = [binding]
    harness.manual_hotkey_bindings = [binding]

    ctrl_tokens = HotkeyMixin._key_to_hotkey_tokens(
        harness,
        SimpleNamespace(name="ctrl", char=None, vk=0x3B),
    )
    plus_tokens = HotkeyMixin._key_to_hotkey_tokens(
        harness,
        SimpleNamespace(name=None, char="+", vk=0x45),
    )
    harness._on_hotkey_press(ctrl_tokens)
    harness._on_hotkey_press(plus_tokens)
    assert harness.hotkey_action_signal.emissions == []

    harness._on_hotkey_release(plus_tokens)
    assert harness.hotkey_action_signal.emissions == []

    harness._on_hotkey_release(ctrl_tokens)

    assert harness.hotkey_action_signal.emissions == ["post_rephrase"]
