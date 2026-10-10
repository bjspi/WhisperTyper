"""Typed surface of the settings window shared by its page modules.

The window's controls come from ``resources/main_window.ui`` (via ``uic.loadUi``) plus the
widgets the pages add in code. Declaring them here lets every page be type-checked on its own.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMenuBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.context import AppContext
from app.core.i18n import TranslationManager

if TYPE_CHECKING:
    from PyQt6.QtCore import QTimer

    from app.controllers.recording import RecordingController
    from app.ui.api_keys import ApiKeysTab
    from app.ui.replacements import ReplacementsTab
    from app.ui.transformations_tab import TransformationsEditor


class SettingsWindowBase(QWidget):
    """Controls, collaborators and cross-page hooks of the settings window."""

    #: Result of the background AAC encoder probe (a tuple of bitrates), delivered on the GUI thread.
    aac_bitrates_ready = pyqtSignal(object)

    ctx: AppContext
    config: Dict[str, Any]
    translator: TranslationManager
    _recording: RecordingController
    _api_keys_tab: ApiKeysTab
    _replacements_tab: ReplacementsTab
    _transformations: TransformationsEditor
    _theme_palette: Optional[Dict[str, str]] = None

    # Window chrome
    main_layout: QVBoxLayout
    menu_bar: QMenuBar
    file_menu: QMenu
    help_menu: QMenu
    open_config_action: QAction
    exit_action: QAction
    open_log_file_action: QAction
    play_last_recording_action: QAction
    about_action: QAction
    github_action: QAction
    tabs: QTabWidget
    save_button: QPushButton
    transcription_tab: QWidget
    rephrasing_tab: QWidget
    post_rephrasing_tab: QWidget
    general_tab: QWidget
    transcription_scroll_area: QScrollArea
    rephrasing_scroll_area: QScrollArea

    # Transcription page
    transcription_layout: QVBoxLayout
    transcription_api_group: QGroupBox
    api_key_label: QLabel
    transcription_key_profile_selector: QComboBox
    api_endpoint_label: QLabel
    api_endpoint_input: QLineEdit
    transcription_provider_label: QLabel
    transcription_provider_selector: QComboBox
    test_transcription_api_button: QPushButton
    transcription_key_status_label: QLabel
    transcription_key_choose_button: QPushButton
    transcription_key_add_button: QPushButton
    model_label: QLabel
    model_dropdown: QComboBox
    transcription_temp_label_title: QLabel
    transcription_temp_slider: QSlider
    transcription_temp_label: QLabel
    transcription_prompt_label: QLabel
    prompt_input: QTextEdit
    prompt_token_label: QLabel
    ffmpeg_label: QLabel
    ffmpeg_path_input: QLineEdit
    ffmpeg_browse_button: QPushButton
    ffmpeg_status_label: QLabel
    _ffmpeg_status_debounce: QTimer

    # Recording card
    recording_group: QGroupBox
    recording_group_layout: QVBoxLayout
    controls_layout: QHBoxLayout
    hotkey_label: QLabel
    hotkey_display: QLineEdit
    set_hotkey_button: QPushButton
    push_to_talk_checkbox: QCheckBox
    windows_keep_mic_hot_checkbox: QCheckBox
    windows_keep_mic_hot_idle_label: QLabel
    windows_keep_mic_hot_idle_input: QSpinBox
    min_recording_label: QLabel
    min_recording_input: QDoubleSpinBox
    recording_format_label: QLabel
    recording_format_selector: QComboBox
    recording_bitrate_label: QLabel
    recording_bitrate_selector: QComboBox
    input_language_label: QLabel
    language_input: QComboBox
    gain_label: QLabel
    gain_input: QLineEdit

    # Rephrasing API page
    shared_api_group: QGroupBox
    rephrasing_api_url_label: QLabel
    rephrasing_api_url_input: QLineEdit
    rephrasing_provider_label: QLabel
    rephrasing_provider_selector: QComboBox
    rephrasing_api_key_label: QLabel
    rephrasing_key_profile_selector: QComboBox
    rephrasing_model_label: QLabel
    rephrasing_model_input: QComboBox
    rephrasing_temp_label_title: QLabel
    rephrasing_temp_slider: QSlider
    rephrasing_temp_label: QLabel
    test_rephrasing_api_button: QPushButton
    rephrasing_key_status_label: QLabel
    rephrasing_key_choose_button: QPushButton
    rephrasing_key_add_button: QPushButton

    # Prompts page
    transformations_tab_description_label: QLabel
    transformations_unavailable_label: QLabel
    transformations_info_label: QLabel
    splitter: QSplitter
    post_rp_list: QListWidget
    post_rp_list_placeholder: QWidget
    post_rp_instruction_enabled_checkbox: QCheckBox
    caption_label: QLabel
    post_rp_caption_edit: QLineEdit
    post_rp_show_during_recording_checkbox: QCheckBox
    post_rp_auto_apply_checkbox: QCheckBox
    post_rp_instruction_options: QWidget
    liveprompt_enabled_checkbox: QCheckBox
    liveprompt_help_button: QPushButton
    liveprompt_trigger_label: QLabel
    liveprompt_trigger_words_input: QLineEdit
    liveprompt_trigger_scan_depth_label: QLabel
    liveprompt_trigger_scan_depth_input: QSpinBox
    liveprompt_strip_trigger_checkbox: QCheckBox
    rephrase_context_checkbox: QCheckBox
    text_label: QLabel
    post_rp_text_edit: QTextEdit
    post_rp_add_btn: QPushButton
    post_rp_remove_btn: QPushButton
    pr_hotkey_group: QGroupBox
    pr_hotkey_label: QLabel
    pr_hotkey_display: QLineEdit
    set_pr_hotkey_button: QPushButton

    # General page
    general_layout: QVBoxLayout
    ui_language_label: QLabel
    ui_language_selector: QComboBox
    color_theme_label: QLabel
    color_theme_selector: QComboBox
    input_device_label: QLabel
    input_device_selector: QComboBox
    proxy_url_label: QLabel
    proxy_url_input: QLineEdit
    use_px_proxy_checkbox: QCheckBox
    test_internet_button: QPushButton
    log_retention_label: QLabel
    log_retention_input: QSpinBox
    restore_clipboard_checkbox: QCheckBox
    debug_logging_checkbox: QCheckBox
    file_logging_checkbox: QCheckBox
    redact_log_checkbox: QCheckBox
    systray_double_click_copy_checkbox: QCheckBox
    recording_prompt_overlay_system_position_checkbox: QCheckBox
    quit_without_confirmation_checkbox: QCheckBox
    misc_group: QGroupBox
    misc_layout: QVBoxLayout
    text_insertion_group: QGroupBox
    logging_group: QGroupBox
    proxy_row_layout: QHBoxLayout
    alt_clipboard_lib_checkbox: QCheckBox
    windows_sendinput_text_checkbox: QCheckBox
    windows_sendinput_fallback_checkbox: QCheckBox
    fast_paste_checkbox: QCheckBox
    post_rephrase_auto_select_all_checkbox: QCheckBox
    play_g_button: QPushButton

    # Hooks implemented by other pages or the window itself.
    def retranslate_ui(self) -> None:
        """Apply the current UI language to every text of the window."""
        raise NotImplementedError

    def apply_theme(self) -> None:
        """Apply the configured light/dark stylesheet."""
        raise NotImplementedError

    def _update_prompt_token_counter(self) -> None:
        """Refresh the prompt token badge for the selected model."""
        raise NotImplementedError

    def _refresh_api_state(self, *_args: object) -> None:
        """Mark incomplete API sections."""
        raise NotImplementedError
