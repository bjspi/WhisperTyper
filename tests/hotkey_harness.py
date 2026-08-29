"""Shared test harness for the global hotkey listener callbacks.

Kept separate from the test modules so the cross-platform guard tests and the
macOS-specific capture tests can use the same minimal stand-in for the app.
"""
from __future__ import annotations

from typing import Any, Set

from app.core import hotkeys
from app.mixins.hotkey_mixin import HotkeyMixin


class FakeSignal:
    """Collect signal emissions without starting a Qt event loop."""

    def __init__(self) -> None:
        """Start with no recorded emissions."""
        self.emissions: list[str] = []

    def emit(self, value: str) -> None:
        self.emissions.append(value)


class HotkeyHarness(HotkeyMixin):
    """Minimal state needed by the listener callbacks."""

    def __init__(self) -> None:
        """Initialize one Ctrl+C post-rephrase binding and its listener state."""
        binding = hotkeys.parse_hotkey_binding("<ctrl>+c", "post_rephrase")
        assert binding is not None
        self.config = {"push_to_talk": False}
        self.hotkey_bindings = [binding]
        self.manual_hotkey_bindings = [binding]
        self.pressed_hotkey_tokens: Set[str] = set()
        self.active_hotkey_actions: Set[str] = set()
        self.deferred_hotkey_actions: Set[str] = set()
        self.push_to_talk_active = False
        self.is_recording = False
        self.hotkey_action_signal = FakeSignal()
        self._hotkey_suppressed_until = 0.0
        self._hotkey_suppression_reason = ""

    def _key_to_hotkey_tokens(self, key: Any) -> Set[str]:
        return set(key)
