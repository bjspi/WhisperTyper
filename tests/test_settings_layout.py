"""Layout of the merged Transcription tab: section order, collapsible sections, temperature support."""
from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from app.audio.store import RecordingStore  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.core.config_store import ConfigStore  # noqa: E402
from app.core.hotkeys import normalize_hotkey_string  # noqa: E402
from app.ui.settings.collapsible import CollapsibleSection  # noqa: E402
from app.ui.settings.window import SettingsWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ui_language": "en"}), encoding="utf-8")
    ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
    instance = SettingsWindow(ctx, recording=Mock(selectable_input_devices=Mock(return_value=[])), hotkeys=Mock(),
                              files=Mock(log_file_exists=Mock(return_value=False)), quit_app=Mock())
    yield instance
    instance.deleteLater()


def test_collapsible_section_toggles_its_content(qapp):
    section = CollapsibleSection()
    section.content_layout.addWidget(QtWidgets.QLabel("inside"))
    assert not section.is_expanded() and section.content.isHidden()
    QTest.mouseClick(section.header, Qt.MouseButton.LeftButton)  # anywhere on the header row
    assert section.is_expanded() and not section.content.isHidden()


def test_one_tab_holds_api_hotkeys_recording_and_ffmpeg_in_order(window):
    assert not hasattr(window, "rephrasing_tab")
    layout = window.transcription_layout
    order = [window.transcription_api_group, window.shared_api_group, window.hotkeys_group,
             window.recording_group, window.ffmpeg_section]
    assert [layout.indexOf(widget) for widget in order] == sorted(layout.indexOf(widget) for widget in order)
    assert window.transcription_api_group.isAncestorOf(window.transcription_prompt_section)
    assert window.hotkeys_group.isAncestorOf(window.pr_hotkey_display)
    assert window.hotkeys_group.isAncestorOf(window.push_to_talk_checkbox)
    assert not window.transcription_prompt_section.is_expanded() and not window.ffmpeg_section.is_expanded()


def test_model_without_temperature_disables_the_slider_and_says_why(window):
    window.rephrasing_provider_selector.setCurrentIndex(window.rephrasing_provider_selector.findData("openai"))
    window.rephrasing_model_input.setCurrentText("gpt-5-mini")
    reason = window.translator.tr("temperature_unsupported_tooltip")
    assert not window.rephrasing_temp_slider.isEnabled()
    assert window.rephrasing_temp_slider.toolTip() == reason == window.rephrasing_temp_label.toolTip()
    window.rephrasing_model_input.setCurrentText("gpt-4.1-mini")
    assert window.rephrasing_temp_slider.isEnabled() and window.rephrasing_temp_label.toolTip() == ""


def test_tab_order_puts_api_keys_right_after_the_api_providers(window):
    pages = [window.tabs.widget(index) for index in range(window.tabs.count())]
    assert pages == [window.transcription_tab, window._api_keys_tab, window.post_rephrasing_tab,
                     window._replacements_tab, window.general_tab]
    assert window.tabs.tabText(0) == "API Providers"
