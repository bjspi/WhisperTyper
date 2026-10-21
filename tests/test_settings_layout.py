"""Layout of the settings window: classic tabs, the modern sidebar pages and rows, temperature support."""
from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
from PyQt6 import QtCore  # noqa: E402
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


def _make_window(tmp_path, modern):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ui_language": "en", "modern_design": modern}), encoding="utf-8")
    ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
    return SettingsWindow(ctx, recording=Mock(selectable_input_devices=Mock(return_value=[])), hotkeys=Mock(),
                          files=Mock(log_file_exists=Mock(return_value=False)), quit_app=Mock())


@pytest.fixture
def window(qapp, tmp_path):
    """The classic design: tabs as laid out in the .ui file."""
    instance = _make_window(tmp_path, modern=False)
    yield instance
    instance.deleteLater()


@pytest.fixture
def modern_window(qapp, tmp_path):
    """The modern design: sidebar pages and settings rows."""
    instance = _make_window(tmp_path, modern=True)
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


def test_muted_labels_are_styled_by_the_theme_not_per_widget(window):
    from app.ui import theme
    assert window.prompt_token_label.styleSheet() == "" and window.transformations_tab_description_label.styleSheet() == ""
    for dark in (False, True):
        qss = theme.build_stylesheet(dark)
        assert f'QLabel#transformations_tab_description_label {{ color: {theme.palette(dark)["muted"]};' in qss
        assert f'QLabel#prompt_token_label[level="over"] {{ color: {theme.palette(dark)["warn"]}; }}' in qss


def test_open_log_action_follows_the_log_file_in_refresh_actions(window):
    window._files.log_file_exists.return_value = True
    window.refresh_actions()
    assert window.open_log_file_action.isEnabled()
    window._files.log_file_exists.return_value = False
    window.refresh_actions()
    assert not window.open_log_file_action.isEnabled()


def test_classic_design_shows_tabs_and_menu_bar(window):
    assert not window.is_modern() and window.design_extra_width() == 0
    assert not window.tabs.tabBar().isHidden() and not window.menu_bar.isHidden()
    assert window._sidebar.isHidden() and window._page_header.isHidden()


def test_modern_design_splits_pages_and_mirrors_them_in_the_sidebar(modern_window):
    window = modern_window
    assert window.is_modern() and window.design_extra_width() > 0
    assert window.tabs.tabBar().isHidden() and window.menu_bar.isHidden() and not window._sidebar.isHidden()
    pages = [window.tabs.widget(index) for index in range(window.tabs.count())]
    assert window.tabs.count() == window._nav.count() == 8
    assert [pages.index(page) for page in (window.transcription_tab, window._api_keys_tab, window.post_rephrasing_tab,
                                           window._replacements_tab, window.general_tab)] == [0, 2, 4, 5, 6]
    assert pages[1].isAncestorOf(window.shared_api_group)
    assert pages[3].isAncestorOf(window.hotkeys_group) and pages[3].isAncestorOf(window.recording_group)
    assert pages[7].isAncestorOf(window.text_insertion_group) and pages[7].isAncestorOf(window.ffmpeg_group)
    assert window.transcription_prompt_section.is_expanded() and window.second_pass_section.is_expanded()
    assert all(window.tabs.tabText(index) for index in range(window.tabs.count()))
    window._nav.setCurrentRow(3)
    assert window.tabs.currentIndex() == 3 and window._page_title.text() == "Hotkeys & recording"
    window.tabs.setCurrentWidget(window.general_tab)
    assert window._nav.currentRow() == 6


def test_modern_rows_follow_their_control_and_toggle_on_click(modern_window):
    window = modern_window
    checkbox = window.restore_clipboard_checkbox
    row = next(row for row in window._rows if row.checkbox is checkbox)
    assert row.title.text() == checkbox.text() and window.misc_group.isAncestorOf(row)
    before = checkbox.isChecked()
    QTest.mouseClick(row, Qt.MouseButton.LeftButton, pos=row.rect().topLeft() + QtCore.QPoint(5, 5))
    assert checkbox.isChecked() != before
    checkbox.hide()
    assert row.isHidden()
    checkbox.show()
    assert not row.isHidden()
    checkbox.setEnabled(False)
    assert not row.title.isEnabled()


def test_modern_rows_strip_the_caption_colon_and_keep_every_control(modern_window):
    window = modern_window
    row = next(row for row in window._rows if row.title is window.transcription_provider_label)
    assert not row.title.text().endswith(":")
    for group in (window.transcription_api_group, window.shared_api_group, window.hotkeys_group,
                  window.recording_group, window.ffmpeg_group, window.misc_group, window.text_insertion_group, window.logging_group):
        children = group.findChildren(QtWidgets.QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
        assert [type(child).__name__ for child in children] == ["CardBody"]
    assert window.ffmpeg_group.isAncestorOf(window.ffmpeg_path_input) and window.ffmpeg_section.isHidden()
    assert window.microphone_group.isAncestorOf(window.input_device_selector)
    assert window.transcription_layout.indexOf(window.microphone_group) == 0


def test_modern_prompts_page_puts_list_and_editor_into_cards(modern_window):
    window = modern_window
    assert window.prompts_list_group.isAncestorOf(window.post_rp_list)
    assert window.prompts_list_group.isAncestorOf(window.post_rp_add_btn)
    assert window.prompts_editor_group.isAncestorOf(window.post_rp_text_edit)
    assert window.prompts_palette_group.isAncestorOf(window.recording_prompt_overlay_digit_keys_checkbox)
    assert window.prompts_list_group.title() == "Your prompts"
    count = window.post_rp_list.count()
    window.post_rp_add_btn.click()
    assert window.post_rp_list.count() == count + 1
    delegate = window.post_rp_list.itemDelegate()
    assert type(delegate).__name__ == "PromptItemDelegate"
    window.post_rp_list.grab()  # paints every row through the delegate


def test_building_the_modern_rows_opens_no_windows(qapp, tmp_path):
    shown = []

    class Watch(QtCore.QObject):
        def eventFilter(self, watched, event):  # noqa: N802 - Qt API
            if event.type() == QtCore.QEvent.Type.Show and isinstance(watched, QtWidgets.QWidget) \
                    and watched.isWindow() and type(watched).__name__ in ("SettingRow", "CardBody", "QLabel"):
                shown.append(type(watched).__name__)
            return False

    watch = Watch()
    qapp.installEventFilter(watch)
    try:
        window = _make_window(tmp_path, modern=True)
    finally:
        qapp.removeEventFilter(watch)
    assert shown == []
    window.deleteLater()
