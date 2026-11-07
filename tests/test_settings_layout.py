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
from app.controllers.local_engine import LOCAL_ENGINE_SUPPORTED  # noqa: E402
from app.core.config_store import ConfigStore  # noqa: E402
from app.core.env import is_WINDOWS  # noqa: E402
from app.core.hotkeys import normalize_hotkey_string  # noqa: E402
from app.ui.settings.collapsible import CollapsibleSection  # noqa: E402
from app.ui.settings.window import SettingsWindow  # noqa: E402

# Every test builds a whole settings window: delete it right after the test (see conftest).
pytestmark = pytest.mark.usefixtures("delete_windows_after_test")


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_window(tmp_path, modern, **config):
    """A settings window built from a config file like at startup."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ui_language": "en", "modern_design": modern, **config}), encoding="utf-8")
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


def test_tab_order_puts_api_keys_and_local_transcription_right_after_the_api_providers(window):
    pages = [window.tabs.widget(index) for index in range(window.tabs.count())]
    local = [window._local_engine_tab] if LOCAL_ENGINE_SUPPORTED else []
    assert pages == [window.transcription_tab, window._api_keys_tab, *local,
                     window.post_rephrasing_tab, window._replacements_tab, window.general_tab]
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


@pytest.mark.parametrize("modern", [True, False], ids=["modern", "classic"])
def test_switching_rephrasing_off_hides_prompts_and_its_hotkey(qapp, tmp_path, modern):
    window = _make_window(tmp_path, modern)
    prompts_page = window.tabs.indexOf(window.post_rephrasing_tab)
    selector = window.rephrasing_provider_selector
    assert selector.itemData(0) == "off" and selector.currentData() == "openai"
    assert window.tabs.isTabVisible(prompts_page) and not window.pr_hotkey_label.isHidden()

    selector.setCurrentIndex(0)
    assert not window.tabs.isTabVisible(prompts_page)
    assert window.pr_hotkey_display.isHidden() and window.set_pr_hotkey_button.isHidden()
    assert window.rephrasing_model_input.isHidden() and window.test_rephrasing_api_button.isHidden()
    assert not selector.isHidden()
    assert not modern or window._nav.isRowHidden(prompts_page)
    assert window.rephrasing_api_url_input.text() == "https://api.openai.com/v1/chat/completions"

    selector.setCurrentIndex(selector.findData("groq"))
    assert window.tabs.isTabVisible(prompts_page) and not window.pr_hotkey_display.isHidden()
    assert not window.rephrasing_model_input.isHidden() and not window.test_rephrasing_api_button.isHidden()
    assert not modern or not window._nav.isRowHidden(prompts_page)
    assert "groq" in window.rephrasing_api_url_input.text()


def test_a_saved_off_opens_switched_off_and_is_stored(qapp, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ui_language": "en", "modern_design": True, "rephrasing_enabled": False}), encoding="utf-8")
    ctx = AppContext(ConfigStore(str(path), normalize_hotkey_string), RecordingStore(str(tmp_path)))
    window = SettingsWindow(ctx, recording=Mock(selectable_input_devices=Mock(return_value=[])), hotkeys=Mock(),
                            files=Mock(log_file_exists=Mock(return_value=False)), quit_app=Mock())
    prompts_page = window.tabs.indexOf(window.post_rephrasing_tab)
    assert window.rephrasing_provider_selector.currentData() == "off" and window._nav.isRowHidden(prompts_page)
    assert window._save_api_settings() and ctx.config["rephrasing_enabled"] is False
    window.rephrasing_provider_selector.setCurrentIndex(1)
    assert window._save_api_settings() and ctx.config["rephrasing_enabled"] is True


def test_classic_design_shows_tabs_and_menu_bar(window):
    assert not window.is_modern() and window.design_extra_width() == 0
    assert not window.tabs.tabBar().isHidden() and not window.menu_bar.isHidden()
    assert window._sidebar.isHidden() and window._page_header.isHidden()


def test_modern_design_splits_pages_and_mirrors_them_in_the_sidebar(modern_window):
    window = modern_window
    assert window.is_modern() and window.design_extra_width() > 0
    assert window.tabs.tabBar().isHidden() and window.menu_bar.isHidden() and not window._sidebar.isHidden()
    pages = [window.tabs.widget(index) for index in range(window.tabs.count())]
    # The local transcription page (after API keys) is not offered on macOS.
    shift = 1 if LOCAL_ENGINE_SUPPORTED else 0
    assert window.tabs.count() == window._nav.count() == 8 + shift
    assert [pages.index(page) for page in (window.transcription_tab, window._api_keys_tab, window.post_rephrasing_tab,
                                           window._replacements_tab, window.general_tab)] == [
        0, 2, 4 + shift, 5 + shift, 6 + shift]
    assert not LOCAL_ENGINE_SUPPORTED or pages.index(window._local_engine_tab) == 3
    assert pages[1].isAncestorOf(window.shared_api_group)
    recording, advanced = pages[3 + shift], pages[7 + shift]
    assert recording.isAncestorOf(window.hotkeys_group) and recording.isAncestorOf(window.recording_group)
    assert recording.isAncestorOf(window.input_device_selector)
    assert advanced.isAncestorOf(window.text_insertion_group) and advanced.isAncestorOf(window.ffmpeg_group)
    assert window.transcription_prompt_section.is_expanded() and window.second_pass_section.is_expanded()
    assert all(window.tabs.tabText(index) for index in range(window.tabs.count()))
    window._nav.setCurrentRow(3 + shift)
    assert window.tabs.currentIndex() == 3 + shift and window._page_title.text() == "Recording & hotkeys"
    window.tabs.setCurrentWidget(window.general_tab)
    assert window._nav.currentRow() == 6 + shift


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
    assert window.recording_group.isAncestorOf(window.input_device_selector)


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


# macOS: the digit keys checkbox is Windows-only (hidden elsewhere), so there is no anchor to sit below.
@pytest.mark.skipif(not is_WINDOWS, reason="digit keys checkbox is Windows-only")
@pytest.mark.parametrize("modern", [False, True])
def test_palette_opacity_sits_below_the_palette_options(qapp, tmp_path, modern):
    window = _make_window(tmp_path, modern)
    spin, anchor = window.recording_overlay_opacity_input, window.recording_prompt_overlay_digit_keys_checkbox
    window.tabs.setCurrentWidget(window.tabs.widget(next(
        index for index in range(window.tabs.count()) if window.tabs.widget(index).isAncestorOf(anchor))))
    window.show()
    qapp.processEvents()
    try:
        # In the palette options (not floating at the window's corner), right below the digit keys.
        assert spin.isVisible() and anchor.isVisible()
        assert spin.mapTo(window, spin.rect().topLeft()).y() > anchor.mapTo(window, anchor.rect().topLeft()).y()
        assert spin.value() == 100 and spin.minimum() == 30
    finally:
        window.close()


def _make_expert_window(tmp_path, expert_mode, **config):
    """A modern window with the given expert mode."""
    return _make_window(tmp_path, True, expert_mode=expert_mode, **config)


def test_expert_mode_off_hides_rows_cards_and_pages_in_the_sidebar(qapp, tmp_path):
    window = _make_expert_window(tmp_path, False)
    assert not window.expert_mode_checkbox.isChecked()
    replacements = window.tabs.indexOf(window._replacements_tab)
    assert not window.tabs.isTabVisible(replacements) and window._nav.isRowHidden(replacements)
    gain_row = next(row for row in window._rows if row.follow is window.gain_input)
    play_sounds_row = next(row for row in window._rows if row.follow is window.play_sounds_checkbox)
    assert gain_row.isHidden() and not play_sounds_row.isHidden()
    # Every row of these cards is an expert row, so the card goes with its title.
    assert window.prompts_palette_group.isHidden() and window.system_integration_group.isHidden()
    assert not window.recording_group.isHidden() and not window.misc_group.isHidden()
    assert window.second_pass_section.isHidden()
    assert window._local_engine_tab.server_group.isHidden() and not window._local_engine_tab.engine_group.isHidden()

    window.expert_mode_checkbox.click()
    assert window.tabs.isTabVisible(replacements) and not window._nav.isRowHidden(replacements)
    assert not gain_row.isHidden() and not window.prompts_palette_group.isHidden()
    assert not window.second_pass_section.isHidden() and not window._local_engine_tab.server_group.isHidden()


def test_expert_mode_is_remembered_and_keeps_hidden_values_on_save(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("app.ui.message_dialog.inform", Mock())
    window = _make_expert_window(tmp_path, True, gain_db=6.0)
    window.expert_mode_checkbox.click()
    stored = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert stored["expert_mode"] is False

    window.save_settings()
    stored = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert stored["gain_db"] == 6.0 and stored["expert_mode"] is False


def test_a_control_shown_by_its_page_stays_hidden_while_expert_mode_is_off(qapp, tmp_path):
    window = _make_expert_window(tmp_path, False)
    temperature_row = next(row for row in window._rows if row.follow is window.rephrasing_temp_slider)
    window.rephrasing_temp_slider.setVisible(False)
    window.rephrasing_temp_slider.setVisible(True)  # as the API page does on every refresh
    assert temperature_row.isHidden()
    window.expert_mode_checkbox.click()
    assert not temperature_row.isHidden()


def test_a_row_whose_control_is_hidden_stays_hidden_when_expert_mode_comes_on(qapp, tmp_path):
    window = _make_expert_window(tmp_path, False)
    format_row = next(row for row in window._rows if row.follow is window.recording_format_selector)
    assert window.recording_format_selector.isHidden()  # AAC is only offered after the encoder probe
    window.expert_mode_checkbox.click()
    assert format_row.isHidden()
    window.recording_format_selector.setVisible(True)
    assert not format_row.isHidden()


def test_switching_expert_mode_off_leaves_a_hidden_page_for_a_visible_one(qapp, tmp_path):
    window = _make_expert_window(tmp_path, True)
    window.tabs.setCurrentWidget(window._replacements_tab)
    window.expert_mode_checkbox.click()
    assert window.tabs.currentWidget() is not window._replacements_tab
    assert window.tabs.isTabVisible(window.tabs.currentIndex())
    assert window._nav.currentRow() == window.tabs.currentIndex()
    assert window._page_title.text() == window.tabs.tabText(window.tabs.currentIndex())


def test_invalid_rules_on_a_hidden_page_switch_expert_mode_on_to_show_them(qapp, tmp_path, monkeypatch):
    window = _make_expert_window(tmp_path, False)
    monkeypatch.setattr("app.ui.message_dialog.inform", Mock())
    window._replacements_tab.editor.setPlainText("only one field")
    window.save_settings()
    assert window.expert_mode_checkbox.isChecked() and window.tabs.currentWidget() is window._replacements_tab


def test_classic_design_ignores_expert_mode(qapp, tmp_path):
    window = _make_window(tmp_path, False, expert_mode=False)
    assert window._sidebar.isHidden()
    assert not window.gain_input.isHidden() and not window.second_pass_section.isHidden()
    assert all(window.tabs.isTabVisible(index) for index in range(window.tabs.count()))
