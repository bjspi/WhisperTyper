"""Searching the settings: matching, the result list in the sidebar and opening a found setting."""
from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from app.core.settings_search import SettingEntry, find, place

ROW = object()  # stands in for a row widget


def _entries():
    return [
        SettingEntry(0, "Recording", "Recording hotkey", "The key combination to start a recording."),
        SettingEntry(0, "Recording", "Use push-to-talk", "Records only while the hotkey is held.", "Hotkeys", ROW),
        SettingEntry(1, "Advanced", "Keep log files for (days)", "", "Logging", ROW),
    ]


def test_every_word_must_match_and_titles_come_first():
    entries = _entries()
    assert [entry.title for entry in find("HOTKEY", entries)] == ["Recording hotkey", "Use push-to-talk"]
    assert [entry.title for entry in find("held hotkey", entries)] == ["Use push-to-talk"]
    assert [entry.title for entry in find("logging", entries)] == ["Keep log files for (days)"]  # its card
    assert [entry.title for entry in find("advanced", entries)] == ["Keep log files for (days)"]  # its page
    assert find("  ", entries) == [] and find("hotkey days", entries) == []


def test_place_names_the_card_only_when_a_title_repeats_on_the_page():
    model = SettingEntry(0, "Transcription", "Model", "", "API", ROW)
    other = SettingEntry(0, "Transcription", "model", "", "Second pass", ROW)
    elsewhere = SettingEntry(1, "Local", "Model", "", "Local engine", ROW)
    assert place(model, [model, other, elsewhere]) == "Transcription › API"
    assert place(elsewhere, [model, other, elsewhere]) == "Local"
    assert place(model, [model, elsewhere]) == "Transcription"


QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from app.audio.store import RecordingStore  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.core.config_store import ConfigStore  # noqa: E402
from app.core.hotkeys import normalize_hotkey_string  # noqa: E402
from app.ui.settings.window import SettingsWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qapp, tmp_path, delete_windows_after_test):
    """The modern design with expert mode off, shown, so visibility checks mean something."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ui_language": "en", "modern_design": True, "expert_mode": False}), encoding="utf-8")
    ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
    instance = SettingsWindow(ctx, recording=Mock(selectable_input_devices=Mock(return_value=[])), hotkeys=Mock(),
                              files=Mock(log_file_exists=Mock(return_value=False)), quit_app=Mock())
    instance.show()
    yield instance
    instance.deleteLater()


def _type(window, text):
    """Type ``text`` into the search field and return the result texts."""
    window._search.field.setText(text)
    results = window._search.results
    return [results.item(row).text() for row in range(results.count())]


def test_results_replace_the_pages_until_escape_clears_the_text(window):
    assert window._nav.isVisible()
    assert _type(window, "push to talk")[0] == "Use push-to-talk\nRecording & hotkeys"
    assert not window._nav.isVisible() and window._search.results.isVisible()
    QTest.keyClick(window._search.field, Qt.Key.Key_Escape)
    assert window._search.field.text() == "" and window.isVisible()
    assert window._nav.isVisible() and not window._search.results.isVisible()


def test_enter_opens_the_page_and_lights_up_the_row_without_saving(window, qapp):
    saved = Mock()
    window.saved.connect(saved)
    _type(window, "push to talk")
    QTest.keyClick(window._search.field, Qt.Key.Key_Return)
    qapp.processEvents()
    assert window.tabs.currentWidget().isAncestorOf(window.push_to_talk_checkbox)
    row = window.push_to_talk_checkbox.parentWidget()
    assert row.objectName() == "settingRow" and row.property("searchHit") is True
    saved.assert_not_called()


def test_opening_an_expert_setting_turns_expert_mode_on(window, qapp):
    _type(window, "palette opacity")
    QTest.keyClick(window._search.field, Qt.Key.Key_Return)
    qapp.processEvents()
    assert window.expert_mode_checkbox.isChecked()
    assert window.recording_overlay_opacity_input.isVisibleTo(window.tabs.currentWidget())


def test_a_search_without_matches_says_so(window):
    page = window.tabs.currentIndex()
    assert _type(window, "no such setting anywhere") == [window.translator.tr("settings_search_no_match")]
    QTest.keyClick(window._search.field, Qt.Key.Key_Return)
    assert window.tabs.currentIndex() == page
