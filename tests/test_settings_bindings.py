"""Declarative settings bindings: load/save round-trip and platform gating (no Qt needed)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from app.ui.settings import bindings
from app.ui.settings.bindings import Binding, load_bindings, save_bindings


class FakeControl:
    """Enough of the Qt widget API for every binding kind; ``state`` holds the shown value."""

    def __init__(self) -> None:
        """Start empty and visible, like a fresh widget."""
        self.state: Any = None
        self.visible = True
        self._slots: List[Any] = []
        self.valueChanged = SimpleNamespace(connect=self._slots.append)

    def _set(self, value: Any) -> None:
        self.state = value

    setChecked = setText = setPlainText = _set  # noqa: N815 - Qt API

    def isChecked(self) -> Any:  # noqa: N802 - Qt API
        return self.state

    def text(self) -> Any:
        return self.state

    toPlainText = text  # noqa: N815 - Qt API

    def value(self) -> Any:
        return self.state

    def setValue(self, value: Any) -> None:  # noqa: N802 - Qt API
        self.state = value
        for slot in self._slots:
            slot(value)

    def setRange(self, *_bounds: int) -> None:  # noqa: N802 - Qt API
        pass

    def setVisible(self, visible: bool) -> None:  # noqa: N802 - Qt API
        self.visible = visible


class Window:
    """Creates a fake control for every attribute a binding touches."""

    def __getattr__(self, name: str) -> FakeControl:
        """Create the missing control on first access."""
        control = FakeControl()
        setattr(self, name, control)
        return control


TABLE = (
    Binding("enabled", "enabled", "check"),
    Binding("url", "url", "stripped"),
    Binding("prompt", "prompt", "plain"),
    Binding("windows_only", "windows_only", "check", "windows", companions=("windows_label",)),
    Binding("context", "context", "check", "not_macos", unsupported_value=False),
)


@pytest.mark.parametrize("supported", [True, False])
def test_round_trip_keeps_values_of_unsupported_platforms(monkeypatch, supported):
    monkeypatch.setitem(bindings.PLATFORMS, "windows", supported)
    monkeypatch.setitem(bindings.PLATFORMS, "not_macos", supported)
    config = {"enabled": True, "url": "https://x", "prompt": "<b>keep</b>", "windows_only": True, "context": True}
    window = Window()
    load_bindings(window, config, TABLE)
    assert window.windows_only.visible is supported and window.windows_label.visible is supported
    window.url.state = "  https://edited  "
    window.windows_only.state = False
    saved = dict(config)
    save_bindings(window, saved, TABLE)
    assert saved["url"] == "https://edited"
    assert saved["prompt"] == "<b>keep</b>"  # Plain text, never interpreted as rich text.
    # A control hidden on this platform must not overwrite the saved choice ...
    assert saved["windows_only"] is (False if supported else True)
    # ... unless the binding forces a value there (selected-text context is off on macOS).
    assert saved["context"] is supported


def test_temperature_slider_round_trips_two_decimals_and_updates_its_label():
    window = Window()
    table = (Binding("temp", "temperature", "temperature", companions=("temp_label",)),)
    load_bindings(window, {"temperature": 0.29}, table)
    assert window.temp.value() == 29  # 0.29 * 100 is 28.999…; rounding keeps the saved value.
    assert window.temp_label.state == "0.29"
    window.temp.setValue(70)
    assert window.temp_label.state == "0.70"
    saved: Dict[str, Any] = {}
    save_bindings(window, saved, table)
    assert saved == {"temperature": 0.7}
