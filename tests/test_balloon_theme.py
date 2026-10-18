"""The mouse-following balloon follows the app's resolved colour theme, not only the OS scheme."""
from __future__ import annotations

import json

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from app.audio.store import RecordingStore  # noqa: E402
from app.context import AppContext, Notifier  # noqa: E402
from app.core.config_store import ConfigStore  # noqa: E402
from app.core.hotkeys import normalize_hotkey_string  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.tooltip import MouseFollowerTooltip  # noqa: E402


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    MouseFollowerTooltip.hide_tooltip()


def balloon_style() -> str:
    balloon = MouseFollowerTooltip._instance
    assert balloon is not None
    return balloon.label.styleSheet()


def test_notifier_passes_the_resolved_theme_to_the_balloon(qapp):
    Notifier(dark=lambda: True).show("dark", 10)
    assert theme.DARK["panel"] in balloon_style()
    Notifier(dark=lambda: False).show("light", 10)
    assert theme.LIGHT["panel"] in balloon_style()


def test_direct_balloons_without_a_theme_follow_the_os(qapp):
    MouseFollowerTooltip.show_tooltip("os", 10)
    expected = theme.palette(theme.is_dark_mode(qapp))["panel"]
    assert expected in balloon_style()


def test_context_resolves_the_configured_theme_for_the_balloon(qapp, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ui_language": "en", "color_theme": "dark"}), encoding="utf-8")
    ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
    ctx.notifier.show("configured", 10)
    assert theme.DARK["panel"] in balloon_style()
    ctx.config["color_theme"] = "light"  # the theme is resolved per balloon, so a switch applies at once
    ctx.notifier.show("switched", 10)
    assert theme.LIGHT["panel"] in balloon_style()
