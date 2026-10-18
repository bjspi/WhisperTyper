"""Rephrase hotkey palette: chips wrap, a click reports prompt and selection, Escape closes, focus (Qt offscreen)."""
from __future__ import annotations

import pytest

QtCore = pytest.importorskip("PyQt6.QtCore")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
QtTest = pytest.importorskip("PyQt6.QtTest")

from app.ui.floating_buttons import FloatingButtonWindow  # noqa: E402

ENTRIES = [
    {"caption": "✨ Instruction", "text": "RUN"},
    {"caption": "Polish", "text": "POLISH"},
]


@pytest.fixture
def qapp(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(FloatingButtonWindow, "show", lambda self: None)
    yield app
    if FloatingButtonWindow._instance:
        FloatingButtonWindow._instance.close()


def window(clicks, entries=ENTRIES, selected="Selected text to rephrase"):
    def on_click(system_prompt, selected_text, palette):  # same parameters as the controller's slot
        clicks.append((system_prompt, selected_text, palette))
    return FloatingButtonWindow(entries, selected, on_click, title="Rephrase")


def test_a_chip_click_reports_its_prompt_the_selection_and_the_window(qapp):
    clicks = []
    palette = window(clicks)
    palette._prompt_buttons[1].click()
    assert clicks == [("POLISH", "Selected text to rephrase", palette)]


def test_many_long_captions_wrap_within_the_maximum_width(qapp):
    clicks = []
    one_line = window(clicks, ENTRIES[:1])
    entries = [{"caption": f"Prompt with a fairly long caption number {i}", "text": str(i)} for i in range(6)]
    wrapped = window(clicks, entries)
    assert wrapped.card.width() <= FloatingButtonWindow.MAX_CARD_WIDTH
    assert wrapped.card.height() > one_line.card.height()


def test_selection_preview_is_one_shortened_line(qapp):
    palette = window([], selected="word " * 200)
    assert palette.preview_label is not None
    assert palette.preview_label.text().endswith("…")
    assert "\n" not in palette.preview_label.text()


def test_escape_closes_the_palette(qapp):
    palette = window([])
    QtTest.QTest.keyClick(palette, QtCore.Qt.Key.Key_Escape)
    assert FloatingButtonWindow._instance is None


def test_the_palette_takes_the_focus_when_shown(qapp, monkeypatch):
    monkeypatch.setattr(FloatingButtonWindow, "show", lambda self: QtWidgets.QWidget.show(self))  # undo the fixture's no-op
    palette = window([])
    qapp.processEvents()
    assert palette.isVisible() and palette.isActiveWindow()
    assert palette.focusPolicy() == QtCore.Qt.FocusPolicy.StrongFocus and palette.hasFocus()
    assert not palette.testAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
