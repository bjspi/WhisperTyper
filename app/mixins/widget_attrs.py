"""WidgetAttrs — group-box style templates + uic-loaded widget type hints.

These are static class attributes / annotations that document the widgets injected by
uic.loadUi. Kept on a base class so IDEs still resolve them, out of the main file.
Annotations are strings (PEP 563) so no Qt imports are needed here.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMenuBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class WidgetAttrs:
    """uic-injected widget type hints and group-box style templates."""

    # --- Stylesheets for dynamic group box border ---
    NORMAL_GROUP_STYLE = """
        QGroupBox#{group_name} {{
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 1ex;
        }}
        QGroupBox#{group_name}::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            border-radius: 3px;
        }}
    """

    HIGHLIGHT_GROUP_STYLE = """
        QGroupBox#{group_name} {{
            border: 2px solid red;
            border-radius: 5px;
            margin-top: 1ex;
        }}
        QGroupBox#{group_name}::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            border-radius: 3px;
        }}
    """

    # --- UI Element Type Hints (widgets injected by uic.loadUi) ---
    main_layout: "QVBoxLayout"
    menu_bar: "QMenuBar"
    file_menu: "QMenu"
    help_menu: "QMenu"
    open_config_action: "QAction"
    exit_action: "QAction"
    about_action: "QAction"
    github_action: "QAction"
    tabs: "QTabWidget"
    save_button: "QPushButton"
    transcription_api_group: "QGroupBox"
    api_key_label: "QLabel"
    api_key_input: "QLineEdit"
    api_endpoint_label: "QLabel"
    api_endpoint_input: "QLineEdit"
    openai_button: "QPushButton"
    groq_button: "QPushButton"
    test_transcription_api_button: "QPushButton"
    model_label: "QLabel"
    model_dropdown: "QComboBox"
    model_input: "QLineEdit"
    transcription_temp_label_title: "QLabel"
    transcription_temp_slider: "QSlider"
    transcription_temp_label: "QLabel"
    hotkey_label: "QLabel"
    hotkey_display: "QLineEdit"
    set_hotkey_button: "QPushButton"
    push_to_talk_checkbox: "QCheckBox"
    windows_keep_mic_hot_checkbox: "QCheckBox"
    windows_keep_mic_hot_idle_label: "QLabel"
    windows_keep_mic_hot_idle_input: "QSpinBox"
    input_language_label: "QLabel"
    language_input: "QComboBox"
    gain_label: "QLabel"
    gain_input: "QLineEdit"
    transcription_prompt_label: "QLabel"
    prompt_input: "QTextEdit"
    prompt_token_label: "QLabel"
    liveprompt_group: "QGroupBox"
    liveprompt_enabled_checkbox: "QCheckBox"
    liveprompt_help_button: "QPushButton"
    liveprompt_trigger_label: "QLabel"
    liveprompt_trigger_words_input: "QLineEdit"
    liveprompt_trigger_scan_depth_label: "QLabel"
    liveprompt_trigger_scan_depth_input: "QSpinBox"
    liveprompt_system_prompt_label: "QLabel"
    liveprompt_system_prompt_input: "QTextEdit"
    rephrase_context_checkbox: "QCheckBox"
    generic_rephrase_group: "QGroupBox"
    generic_rephrase_enabled_checkbox: "QCheckBox"
    generic_rephrase_prompt_label: "QLabel"
    generic_rephrase_prompt_input: "QTextEdit"
    shared_api_group: "QGroupBox"
    rephrasing_api_url_label: "QLabel"
    rephrasing_api_url_input: "QLineEdit"
    rephrasing_api_key_label: "QLabel"
    rephrasing_api_key_input: "QLineEdit"
    rephrasing_model_label: "QLabel"
    rephrasing_model_input: "QLineEdit"
    rephrasing_temp_label_title: "QLabel"
    rephrasing_temp_slider: "QSlider"
    rephrasing_temp_label: "QLabel"
    test_rephrasing_api_button: "QPushButton"
    transformations_tab_description_label: "QLabel"
    transformations_unavailable_label: "QLabel"
    transformations_info_label: "QLabel"
    post_rephrasing_tab: "QWidget"
    splitter: "QSplitter"
    post_rp_list: "QListWidget"
    post_rp_list_placeholder: "QWidget"
    caption_label: "QLabel"
    post_rp_caption_edit: "QLineEdit"
    text_label: "QLabel"
    post_rp_text_edit: "QTextEdit"
    post_rp_add_btn: "QPushButton"
    post_rp_remove_btn: "QPushButton"
    pr_hotkey_group: "QGroupBox"
    pr_hotkey_label: "QLabel"
    pr_hotkey_display: "QLineEdit"
    set_pr_hotkey_button: "QPushButton"
    ui_language_label: "QLabel"
    ui_language_selector: "QComboBox"
    color_theme_label: "QLabel"
    color_theme_selector: "QComboBox"
    input_device_label: "QLabel"
    input_device_selector: "QComboBox"
    proxy_url_label: "QLabel"
    proxy_url_input: "QLineEdit"
    use_px_proxy_checkbox: "QCheckBox"
    test_internet_button: "QPushButton"
    log_retention_label: "QLabel"
    log_retention_input: "QSpinBox"
    restore_clipboard_checkbox: "QCheckBox"
    debug_logging_checkbox: "QCheckBox"
    file_logging_checkbox: "QCheckBox"
    redact_log_checkbox: "QCheckBox"
    systray_double_click_copy_checkbox: "QCheckBox"
    quit_without_confirmation_checkbox: "QCheckBox"
    alt_clipboard_lib_checkbox: "QCheckBox"
    post_rephrase_auto_select_all_checkbox: "QCheckBox"
    play_g_button: "QPushButton"

    # macOS: name of the app to re-activate after showing a floating window.
    macos_active_application: Optional[str] = None
