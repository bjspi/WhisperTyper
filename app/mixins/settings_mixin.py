"""SettingsMixin — settings window: build, bind, validate, API tests, config & logging."""
from __future__ import annotations

import logging
import logging.handlers
import os
from typing import Any, Dict, List, Optional

from PyQt6 import uic
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
)

from app.core.config_store import ConfigStore
from app.core.constants import (
    CONFIG_FILE,
    LANGUAGES,
    LOG_FILE_PATH,
    TRANSCRIPTION_MODEL_OPTIONS,
    WHISPER_PROMPT_TOKEN_LIMIT,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from app.core.env import is_MACOS, is_WINDOWS, open_with_default_app
from app.core.ffmpeg import probe_version, resolve_ffmpeg
from app.core.hotkeys import normalize_hotkey_string
from app.core.paths import resource_path
from app.core.prompts import (
    DEFAULT_GENERIC_REPHRASE_PROMPTS,
    DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS,
    DEFAULT_TRANSCRIPTION_PROMPTS,
    _default_prompt_for,
    _is_known_default_prompt,
)
from app.core.redaction import LOG_REDACTION_STATE
from app.core.textutil import estimate_tokens
from app.services import net
from app.ui.connection_tester import ConnectionTester


class SettingsMixin:
    """Settings window: build, bind, validate, retranslate, API/connection tests, config & logging."""

    def _group_style(self, group_name: str, highlighted: bool = False) -> str:
        """Return a theme-aware group box style (accent/warn border on the card)."""
        pal = getattr(self, "_theme_palette", None)
        if pal:
            border = pal["warn"] if highlighted else pal["border"]
            width = 2 if highlighted else 1
            return (
                f"QGroupBox#{group_name} {{ background: {pal['panel']}; "
                f"border: {width}px solid {border}; border-radius: 12px; margin-top: 14px; "
                f"padding: 14px 12px 10px 12px; font-weight: 600; }} "
                f"QGroupBox#{group_name}::title {{ subcontrol-origin: margin; "
                f"subcontrol-position: top left; left: 12px; padding: 0 6px; "
                f"color: {pal['accent']}; background: {pal['bg']}; }}"
            )
        template = self.HIGHLIGHT_GROUP_STYLE if highlighted else self.NORMAL_GROUP_STYLE
        return template.format(group_name=group_name)

    def init_ui(self) -> None:
        """Initializes the settings window by loading it from main_window.ui."""
        # Load the UI from the .ui file
        ui_path = resource_path("resources", "main_window.ui")
        uic.loadUi(ui_path, self)

        # --- Tab 3 (Transformations) Layout Adjustments ---
        # Set the labels to take up minimum vertical space
        self.transformations_tab_description_label.setSizePolicy(self.transformations_tab_description_label.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Maximum)
        self.transformations_info_label.setSizePolicy(self.transformations_info_label.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Maximum)
        # Ensure the splitter takes up all remaining space
        self.splitter.setSizePolicy(self.splitter.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        # Token counter renders as a compact pill badge, so it should hug its content.
        self.prompt_token_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)


        # --- Menu Bar Setup ---
        # The menu bar is not part of the .ui file for a QWidget, so we create it manually.
        self.menu_bar = QMenuBar(self)
        self.main_layout.insertWidget(0, self.menu_bar) # Insert at the top of the main layout
        # Move menu higher and to the left by adjusting layout margins
        self.main_layout.setContentsMargins(0, 0, 0, 5)
        # Cross-platform branding header (icon + name + hotkey badge), macOS included.
        self._install_brand_header()

        self.file_menu = self.menu_bar.addMenu("") # Text set in retranslate
        self.open_config_action = QAction("", self)
        self.open_config_action.triggered.connect(self.open_config_file)
        self.file_menu.addAction(self.open_config_action)

        # Add "Open Log File" from systray
        self.open_log_file_action = QAction("", self)
        self.open_log_file_action.triggered.connect(self.open_log_file)
        self.file_menu.addAction(self.open_log_file_action)

        # Add "Play Last Recording" from systray
        self.play_last_recording_action = QAction("", self)
        self.play_last_recording_action.triggered.connect(self.play_latest_recording)
        self.file_menu.addAction(self.play_last_recording_action)

        self.file_menu.addSeparator()
        self.exit_action = QAction("", self)
        self.exit_action.triggered.connect(self.quit_app)
        self.file_menu.addAction(self.exit_action)

        self.help_menu = self.menu_bar.addMenu("") # Text set in retranslate
        self.about_action = QAction("", self)
        self.about_action.triggered.connect(self.show_about_dialog)
        self.help_menu.addAction(self.about_action)
        self.github_action = QAction("", self)
        self.github_action.triggered.connect(self.open_github_link)
        self.help_menu.addAction(self.github_action)

        # --- Post-UI Load Configuration and Connections ---

        # Transcription Tab Connections
        self.api_key_input.setText(self.config["api_key"])
        self.api_endpoint_input.setText(self.config["api_endpoint"])
        self.openai_button.clicked.connect(
            lambda: self.api_endpoint_input.setText("https://api.openai.com/v1/audio/transcriptions"))
        self.groq_button.clicked.connect(
            lambda: self.api_endpoint_input.setText("https://api.groq.com/openai/v1/audio/transcriptions"))
        self._connection_tester = ConnectionTester(self)
        self.test_transcription_api_button.clicked.connect(self._connection_tester.test_transcription)

        # Connect for live validation
        self.api_key_input.textChanged.connect(self._update_transcription_api_group_style)
        self.api_endpoint_input.textChanged.connect(self._update_transcription_api_group_style)

        self.model_dropdown.addItems(TRANSCRIPTION_MODEL_OPTIONS)
        model_value = self.config["model"]
        if self.model_dropdown.findText(model_value) != -1:
            self.model_dropdown.setCurrentText(model_value)
            self.model_input.setVisible(False)
        else:
            self.model_dropdown.setCurrentText("Custom")
            self.model_input.setText(model_value)
            self.model_input.setVisible(True)
        self.model_dropdown.currentTextChanged.connect(lambda text: self.model_input.setVisible(text == "Custom"))
        self.model_dropdown.currentTextChanged.connect(lambda _t: self._update_prompt_token_counter())
        self.model_input.textChanged.connect(lambda _t: self._update_prompt_token_counter())

        self.transcription_temp_slider.setRange(0, 100)
        self.transcription_temp_slider.setValue(int(self.config["transcription_temperature"] * 100))
        self.transcription_temp_label.setText(f"{self.config['transcription_temperature']:.2f}")
        self.transcription_temp_slider.valueChanged.connect(self._update_transcription_temp_label)
        # Model and temperature share one row at equal width (50:50), with only a minimal gap
        # between the two columns.
        self.model_temp_row.setStretch(0, 1)  # model column
        self.model_temp_row.setStretch(1, 1)  # temperature column
        self.model_temp_row.setSpacing(8)
        # Top-align both columns so the "Temperature:" label sits on the same line as "Model:"
        # (the temperature column is a touch shorter and would otherwise drift downward).
        self.model_temp_row.setAlignment(self.model_col, Qt.AlignmentFlag.AlignTop)
        self.model_temp_row.setAlignment(self.temp_col, Qt.AlignmentFlag.AlignTop)

        self.hotkey_display.setText(self.hotkey_str)
        self.set_hotkey_button.clicked.connect(self.start_hotkey_capture)

        # These recording controls live inside the "Recording & Hotkey" card next to the
        # hotkey/language/gain row (recording_group from the .ui), not scattered in the tab.
        self.push_to_talk_checkbox = QCheckBox(self)
        self.push_to_talk_checkbox.setChecked(self.config["push_to_talk"])
        self.recording_group_layout.addWidget(self.push_to_talk_checkbox)

        self.windows_keep_mic_hot_checkbox = QCheckBox(self)
        self.windows_keep_mic_hot_checkbox.setChecked(self.config["windows_keep_mic_hot"])
        self.recording_group_layout.addWidget(self.windows_keep_mic_hot_checkbox)
        self.windows_keep_mic_hot_checkbox.setVisible(is_WINDOWS)
        self.windows_keep_mic_hot_checkbox.stateChanged.connect(self._update_windows_keep_mic_hot_ui_state)

        self.windows_keep_mic_hot_idle_label = QLabel(self)
        self.windows_keep_mic_hot_idle_input = QSpinBox(self)
        self.windows_keep_mic_hot_idle_input.setRange(1, 240)
        self.windows_keep_mic_hot_idle_input.setValue(self.config["windows_keep_mic_hot_idle_minutes"])
        windows_keep_mic_hot_idle_layout = QHBoxLayout()
        windows_keep_mic_hot_idle_layout.addWidget(self.windows_keep_mic_hot_idle_label)
        windows_keep_mic_hot_idle_layout.addWidget(self.windows_keep_mic_hot_idle_input)
        windows_keep_mic_hot_idle_layout.addStretch()
        self.recording_group_layout.addLayout(windows_keep_mic_hot_idle_layout)
        self.windows_keep_mic_hot_idle_label.setVisible(is_WINDOWS)
        self.windows_keep_mic_hot_idle_input.setVisible(is_WINDOWS)
        self._update_windows_keep_mic_hot_ui_state()

        # Minimum recording length (all platforms): shorter recordings are discarded as mis-taps.
        self.min_recording_label = QLabel(self)
        self.min_recording_input = QDoubleSpinBox(self)
        self.min_recording_input.setRange(0.0, 10.0)
        self.min_recording_input.setSingleStep(0.1)
        self.min_recording_input.setDecimals(1)
        self.min_recording_input.setSuffix(" s")
        self.min_recording_input.setValue(float(self.config.get("min_recording_seconds", 1.0)))
        min_recording_layout = QHBoxLayout()
        min_recording_layout.addWidget(self.min_recording_label)
        min_recording_layout.addWidget(self.min_recording_input)
        min_recording_layout.addStretch()
        self.recording_group_layout.addLayout(min_recording_layout)

        self.lang_code_to_name = {v: k for k, v in LANGUAGES.items()}
        self.language_input.addItems(LANGUAGES.keys())
        lang_code = self.config["input_language"].lower()
        display_name = self.lang_code_to_name.get(lang_code, "English")
        self.language_input.setCurrentText(display_name)

        self.gain_input.setText(str(self.config["gain_db"]))
        # Rebalance the hotkey / language / gain row: the hotkey field needs room for long combos
        # like "<ctrl>+<caps_lock>", while gain only ever holds a couple of digits. Give the
        # hotkey column the width freed up by halving the gain column (3 : 2 : 1).
        self.controls_layout.setStretch(0, 3)  # hotkey
        self.controls_layout.setStretch(1, 2)  # language
        self.controls_layout.setStretch(2, 1)  # gain
        self._build_ffmpeg_settings_row()

        self.prompt_input.setText(self.config["prompt"])
        self.prompt_input.textChanged.connect(self._update_prompt_token_counter)
        QTimer.singleShot(0, self._update_prompt_token_counter)

        # Rephrasing Tab Connections
        self.liveprompt_enabled_checkbox.setChecked(self.config["liveprompt_enabled"])
        # Render the help button as a small round "?" badge (see theme QSS). The .ui can't set a
        # fixed size reliably, so pin it here.
        self.liveprompt_help_button.setFixedSize(22, 22)
        self.liveprompt_help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.liveprompt_help_button.clicked.connect(self.show_liveprompt_help)
        self.liveprompt_trigger_words_input.setText(self.config["liveprompt_trigger_words"])
        self.liveprompt_trigger_scan_depth_input.setValue(self.config["liveprompt_trigger_word_scan_depth"])
        self.liveprompt_strip_trigger_checkbox.setChecked(self.config["liveprompt_strip_trigger"])
        self.liveprompt_system_prompt_input.setText(self.config["liveprompt_system_prompt"])
        if is_MACOS:
            # On macOS, we don't use the context checkbox due to permission issues ...
            self.rephrase_context_checkbox.setChecked(False)
            self.rephrase_context_checkbox.setVisible(False)
        else:
            self.rephrase_context_checkbox.setChecked(self.config["rephrase_use_selection_context"])

        self.generic_rephrase_enabled_checkbox.setChecked(self.config["generic_rephrase_enabled"])
        self.generic_rephrase_prompt_input.setText(self.config["generic_rephrase_prompt"])

        # Connect checkboxes to update the API group styling
        self.liveprompt_enabled_checkbox.stateChanged.connect(self._update_rephrase_api_group_style)
        self.generic_rephrase_enabled_checkbox.stateChanged.connect(self._update_rephrase_api_group_style)

        self.rephrasing_api_url_input.setText(self.config["rephrasing_api_url"])
        self.rephrasing_api_key_input.setText(self.config["rephrasing_api_key"])
        self.rephrasing_model_input.setText(self.config["rephrasing_model"])

        # Connect text inputs to update the API group styling
        self.rephrasing_api_url_input.textChanged.connect(self._update_rephrase_api_group_style)
        self.rephrasing_api_key_input.textChanged.connect(self._update_rephrase_api_group_style)
        self.rephrasing_model_input.textChanged.connect(self._update_rephrase_api_group_style)

        self.rephrasing_temp_slider.setRange(0, 100)
        self.rephrasing_temp_slider.setValue(int(self.config["rephrasing_temperature"] * 100))
        self.rephrasing_temp_label.setText(f"{self.config['rephrasing_temperature']:.2f}")
        self.rephrasing_temp_slider.valueChanged.connect(self._update_rephrasing_temp_label)

        # Connect the new test button
        self.test_rephrasing_api_button.clicked.connect(self._connection_tester.test_rephrasing)

        # Transformations Tab Setup
        self.max_post_rephrasing_entries = 10

        # Replace the placeholder QListWidget with our custom drag-enabled one
        class _PostRPList(QListWidget):
            def __init__(self, outer: Any) -> None:
                super().__init__()
                self._outer = outer
                self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
                self.setDragEnabled(True)
                self.setAcceptDrops(True)
                self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
                self.setDefaultDropAction(Qt.DropAction.MoveAction)

            def dropEvent(self, event: Any) -> None:
                super().dropEvent(event)
                if hasattr(self._outer, '_sync_post_rp_data_from_list'):
                    self._outer._sync_post_rp_data_from_list()

        placeholder = self.post_rp_list_placeholder
        self.post_rp_list = _PostRPList(self)
        self.splitter.replaceWidget(0, self.post_rp_list)
        placeholder.deleteLater()

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        loaded_post_rephrasing_entries = self.config.get("post_rephrasing_entries", [])
        self.post_rephrasing_data: List[Dict[str, str]] = [
            self._normalize_post_rp_entry(entry)
            for entry in loaded_post_rephrasing_entries
            if isinstance(entry, dict)
        ]
        if len(self.post_rephrasing_data) > self.max_post_rephrasing_entries:
             self.post_rephrasing_data = self.post_rephrasing_data[:self.max_post_rephrasing_entries]
        self._sync_config_post_rp_entries()

        self._current_pr_row = -1
        self._current_pr_entry_id: Optional[str] = None
        self._post_rp_updating = False
        self._load_post_rp_entries_into_list()
        self.post_rp_list.currentRowChanged.connect(self._on_post_rp_selection_changed)
        self.post_rp_add_btn.clicked.connect(self._on_post_rp_add_clicked)
        self.post_rp_remove_btn.clicked.connect(self._on_post_rp_remove_clicked)
        self._update_post_rp_ui_state()
        if self.post_rp_list.count() > 0:
            self.post_rp_list.setCurrentRow(0)

        # Connect changes in the main API key to the rephrase group style check
        self.api_key_input.textChanged.connect(self._update_rephrase_api_group_style)

        self.pr_hotkey_display.setText(self.config["post_rephrase_hotkey"])
        self.set_pr_hotkey_button.clicked.connect(self.start_hotkey_capture)

        # General Tab Connections
        self.ui_language_selector.addItems(["English", "Deutsch", "Español", "Français"])
        lang_map = {"en": "English", "de": "Deutsch", "es": "Español", "fr": "Français"}
        current_lang_name = lang_map.get(self.config.get("ui_language", "en"), "English")
        self.ui_language_selector.setCurrentText(current_lang_name)
        self.ui_language_selector.currentTextChanged.connect(self.change_language)

        # Color theme selector: System (auto) / Light / Dark. Data holds the config key.
        self.color_theme_selector.addItem("", "system")
        self.color_theme_selector.addItem("", "light")
        self.color_theme_selector.addItem("", "dark")
        idx = self.color_theme_selector.findData(self.config.get("color_theme", "system"))
        self.color_theme_selector.setCurrentIndex(idx if idx >= 0 else 0)
        self.color_theme_selector.currentIndexChanged.connect(self._on_color_theme_changed)

        self._populate_input_device_selector()
        self.input_device_selector.currentIndexChanged.connect(self._on_input_device_changed)

        self.post_rephrase_auto_select_all_checkbox = QCheckBox(self)
        play_button_index = self.general_layout.indexOf(self.play_g_button)
        if play_button_index < 0:
            play_button_index = self.general_layout.count()
        self.general_layout.insertWidget(play_button_index, self.post_rephrase_auto_select_all_checkbox)

        self.proxy_url_input.setText(self.config.get("proxy_url", ""))
        self.use_px_proxy_checkbox.setChecked(self.config.get("use_local_px_proxy", False))
        self.test_internet_button.clicked.connect(self._connection_tester.test_internet)
        self.log_retention_input.setValue(int(self.config.get("log_retention_days", 3)))
        self.restore_clipboard_checkbox.setChecked(self.config["restore_clipboard"])
        self.debug_logging_checkbox.setChecked(self.config["debug_logging"])
        self.file_logging_checkbox.setChecked(self.config["file_logging"])
        self.redact_log_checkbox.setChecked(self.config["redact_transcription_in_log"])
        self.systray_double_click_copy_checkbox.setChecked(self.config["systray_double_click_copy"])
        self.quit_without_confirmation_checkbox.setChecked(self.config["quit_without_confirmation"])
        self.alt_clipboard_lib_checkbox.setChecked(self.config["alt_clipboard_lib"])
        self.post_rephrase_auto_select_all_checkbox.setChecked(self.config["post_rephrase_auto_select_all"])

        self.play_g_button.clicked.connect(self.play_latest_recording)

        # Main Save Button
        self.save_button.clicked.connect(self.save_settings)
        # Save button: compact, right-aligned with breathing room from the edge — wrap it in a
        # row with a leading stretch instead of letting it stretch the full window width.
        self.save_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.save_button.setMinimumWidth(150)
        self.save_button.setMaximumWidth(200)
        self.main_layout.removeWidget(self.save_button)
        save_row = QHBoxLayout()
        save_row.setContentsMargins(0, 0, 16, 6)
        save_row.addStretch()
        save_row.addWidget(self.save_button)
        self.main_layout.addLayout(save_row)

        # Set window icon
        app_icon = self._get_app_icon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        else:
            self.setWindowIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))

        # Cross-platform light/dark teal theme (applied on macOS too, with dark support).
        self.apply_theme()

        # Set all translatable texts
        self.retranslate_ui()

        # Enforce a minimum window size (can't be shrunk past it) and restore the last size.
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        try:
            width = max(int(self.config.get("window_width", 760)), WINDOW_MIN_WIDTH)
            height = max(int(self.config.get("window_height", WINDOW_MIN_HEIGHT)), WINDOW_MIN_HEIGHT)
        except (TypeError, ValueError):
            width, height = 760, WINDOW_MIN_HEIGHT
        self.resize(width, height)

    def _build_ffmpeg_settings_row(self) -> None:
        """Add the FFmpeg path row (label + path field + Browse + status) to the Transcription tab.

        With ffmpeg available, video files (mp4/mov/mkv/…) can be picked for transcription; their
        audio track is extracted to a temp MP3 first. Empty path = auto-detect on PATH.
        """
        self.ffmpeg_label = QLabel(self)
        self.ffmpeg_path_input = QLineEdit(self)
        self.ffmpeg_path_input.setText(self.config.get("ffmpeg_path", ""))
        self.ffmpeg_browse_button = QPushButton(self)
        self.ffmpeg_browse_button.clicked.connect(self._browse_ffmpeg_path)
        # Re-check availability as the user types/clears the path — debounced, because the
        # probe runs `ffmpeg -version` as a blocking subprocess. Without the debounce every
        # single keystroke would spawn (and wait for) one subprocess on the GUI thread.
        self._ffmpeg_status_debounce = QTimer(self)
        self._ffmpeg_status_debounce.setSingleShot(True)
        self._ffmpeg_status_debounce.setInterval(400)
        self._ffmpeg_status_debounce.timeout.connect(self._refresh_ffmpeg_status)
        self.ffmpeg_path_input.textChanged.connect(
            lambda _t: self._ffmpeg_status_debounce.start()
        )

        ffmpeg_row = QHBoxLayout()
        ffmpeg_row.addWidget(self.ffmpeg_path_input)
        ffmpeg_row.addWidget(self.ffmpeg_browse_button)

        self.ffmpeg_status_label = QLabel(self)
        self.ffmpeg_status_label.setWordWrap(True)

        # Insert as its own block just above the transcription prompt.
        insert_at = self.transcription_layout.indexOf(self.transcription_prompt_label)
        if insert_at < 0:
            insert_at = self.transcription_layout.count()
        self.transcription_layout.insertWidget(insert_at, self.ffmpeg_status_label)
        self.transcription_layout.insertLayout(insert_at, ffmpeg_row)
        self.transcription_layout.insertWidget(insert_at, self.ffmpeg_label)

        self._refresh_ffmpeg_status()

    def _browse_ffmpeg_path(self) -> None:
        """Open a file picker for the ffmpeg binary and store the chosen path in the field."""
        start_dir = self.ffmpeg_path_input.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, self.translator.tr("ffmpeg_browse_dialog_title"), start_dir
        )
        if path:
            self.ffmpeg_path_input.setText(path)

    def _refresh_ffmpeg_status(self) -> None:
        """Probe the current ffmpeg path/PATH and reflect availability in the status label."""
        if not hasattr(self, "ffmpeg_status_label"):
            return
        exe = resolve_ffmpeg(self.ffmpeg_path_input.text())
        version = probe_version(exe)
        pal = getattr(self, "_theme_palette", None) or {}
        ok_color = pal.get("accent", "#0e8aa8")
        warn_color = pal.get("warn", "#cf6a3f")
        if version:
            self.ffmpeg_status_label.setText(
                self.translator.tr("ffmpeg_status_detected", version=version)
            )
            self.ffmpeg_status_label.setStyleSheet(f"color: {ok_color};")
        else:
            self.ffmpeg_status_label.setText(self.translator.tr("ffmpeg_status_not_found"))
            self.ffmpeg_status_label.setStyleSheet(f"color: {warn_color};")

    def show_liveprompt_help(self) -> None:
        """Shows a detailed help tooltip for the LivePrompting feature."""
        tooltip_text = self.translator.tr("liveprompt_help_tooltip")
        self.show_tray_balloon(tooltip_text, 5000) # Show for 5 seconds

    def show_about_dialog(self) -> None:
        """Shows the 'About' dialog with application information."""
        QMessageBox.about(
            self,
            self.translator.tr("about_dialog_title"),
            self.translator.tr("about_dialog_text")
        )

    def open_config_file(self) -> None:
        """Opens the config.json file in the default system editor."""
        if os.path.exists(CONFIG_FILE):
            QDesktopServices.openUrl(QUrl.fromLocalFile(CONFIG_FILE))
        else:
            self.show_tray_balloon(self.translator.tr("config_file_not_found"), 3000)

    def open_github_link(self) -> None:
        """Opens the project's GitHub repository in the default browser."""
        # Replace with the actual URL of your repository
        url = QUrl("https://github.com/bjspi/WhisperTyper")
        QDesktopServices.openUrl(url)

    def _update_transcription_temp_label(self, value: int) -> None:
        """Updates the label for the transcription temperature slider."""
        self.transcription_temp_label.setText(f"{value / 100.0:.2f}")

    def _update_rephrasing_temp_label(self, value: int) -> None:
        """Updates the label for the rephrasing temperature slider."""
        self.rephrasing_temp_label.setText(f"{value / 100.0:.2f}")

    def save_settings(self) -> None:
        """Saves settings, restarts the hotkey listener."""
        # Clean model name: remove anything in parentheses and trailing whitespace
        model_raw = self.model_input.text() if self.model_dropdown.currentText() == "Custom" else self.model_dropdown.currentText()
        # Perform validation (warnings only)
        warnings = self._collect_validation_warnings(model_raw)
        if warnings:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle(self.translator.tr("validation_warning_title"))
            msg.setText(self.translator.tr("validation_warning_text"))
            msg.setInformativeText("\n".join(f"- {w}" for w in warnings))
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()

        self.config["api_key"] = self.api_key_input.text()
        self.config["api_endpoint"] = self.api_endpoint_input.text()
        self.config["model"] = model_raw
        self.config["transcription_temperature"] = self.transcription_temp_slider.value() / 100.0
        self.config["ffmpeg_path"] = self.ffmpeg_path_input.text().strip()

        # Save the language code, not the display name
        lang_display_name = self.language_input.currentText()
        self.config["input_language"] = LANGUAGES.get(lang_display_name, "en")
        self.config["prompt"] = self.prompt_input.toPlainText()
        self.config["restore_clipboard"] = self.restore_clipboard_checkbox.isChecked()
        self.config["push_to_talk"] = self.push_to_talk_checkbox.isChecked()
        if is_WINDOWS:
            self.config["windows_keep_mic_hot"] = self.windows_keep_mic_hot_checkbox.isChecked()
            self.config["windows_keep_mic_hot_idle_minutes"] = self.windows_keep_mic_hot_idle_input.value()
        self.config["min_recording_seconds"] = self.min_recording_input.value()
        self.config["debug_logging"] = self.debug_logging_checkbox.isChecked()
        try:
            self.config["gain_db"] = float(self.gain_input.text() or 0)
        except (ValueError, TypeError):
            # Keep the previous gain value if the field contains non-numeric text,
            # so an invalid entry never aborts the whole save (which would also skip
            # the config write and hotkey re-init below).
            logging.warning(
                f"Invalid gain value '{self.gain_input.text()}'; keeping previous gain {self.config.get('gain_db')}."
            )
            self.gain_input.setText(str(self.config.get("gain_db", 0)))
        # File logging
        self.config["file_logging"] = self.file_logging_checkbox.isChecked()
        # Redact transcript/prompt text in log
        self.config["redact_transcription_in_log"] = self.redact_log_checkbox.isChecked()
        # Log rotation / retention (days)
        self.config["log_retention_days"] = int(self.log_retention_input.value())
        # Corporate proxy
        self.config["proxy_url"] = self.proxy_url_input.text().strip()
        self.config["use_local_px_proxy"] = self.use_px_proxy_checkbox.isChecked()
        # Systray double-click
        self.config["systray_double_click_copy"] = self.systray_double_click_copy_checkbox.isChecked()
        self.config["quit_without_confirmation"] = self.quit_without_confirmation_checkbox.isChecked()
        # Alternative clipboard lib
        self.config["alt_clipboard_lib"] = self.alt_clipboard_lib_checkbox.isChecked()
        self.config["post_rephrase_auto_select_all"] = self.post_rephrase_auto_select_all_checkbox.isChecked()

        # Rephrasing settings
        self.config["liveprompt_enabled"] = self.liveprompt_enabled_checkbox.isChecked()
        self.config["liveprompt_trigger_words"] = self.liveprompt_trigger_words_input.text()
        self.config["liveprompt_trigger_word_scan_depth"] = self.liveprompt_trigger_scan_depth_input.value()
        self.config["liveprompt_strip_trigger"] = self.liveprompt_strip_trigger_checkbox.isChecked()
        self.config["liveprompt_system_prompt"] = self.liveprompt_system_prompt_input.toPlainText()
        self.config["rephrase_use_selection_context"] = self.rephrase_context_checkbox.isChecked()
        self.config["generic_rephrase_enabled"] = self.generic_rephrase_enabled_checkbox.isChecked()
        self.config["generic_rephrase_prompt"] = self.generic_rephrase_prompt_input.toPlainText()

        # Shared API settings
        self.config["rephrasing_api_url"] = self.rephrasing_api_url_input.text()
        self.config["rephrasing_api_key"] = self.rephrasing_api_key_input.text()
        self.config["rephrasing_model"] = self.rephrasing_model_input.text()
        self.config["rephrasing_temperature"] = self.rephrasing_temp_slider.value() / 100.0
        # Post Rewording entries (new)
        if hasattr(self, 'post_rephrasing_data'):
            self._save_current_post_rp_edits()
            self._sync_config_post_rp_entries()

        # Get the pending hotkey string from the UI display
        pending_hotkey_str = normalize_hotkey_string(self.hotkey_display.text()) or self.hotkey_display.text().strip()
        pending_pr_hotkey_str = normalize_hotkey_string(self.pr_hotkey_display.text()) or self.pr_hotkey_display.text().strip()
        self.hotkey_display.setText(pending_hotkey_str)
        self.pr_hotkey_display.setText(pending_pr_hotkey_str)

        hotkey_changed = (pending_hotkey_str != self.hotkey_str) or \
                         (pending_pr_hotkey_str != self.post_rephrase_hotkey_str)

        if hotkey_changed:
            self.hotkey_str = pending_hotkey_str
            self.config["hotkey"] = self.hotkey_str
            self.post_rephrase_hotkey_str = pending_pr_hotkey_str
            self.config["post_rephrase_hotkey"] = self.post_rephrase_hotkey_str

            if is_MACOS:
                # On macOS, dynamically restarting the listener is problematic due to accessibility permissions.
                # It's safer to ask the user to restart the app.
                QMessageBox.information(
                    self,
                    self.translator.tr("macos_hotkey_restart_title"),
                    self.translator.tr("macos_hotkey_restart_text")
                )
            else:
                # On other systems, we can safely re-initialize the listener.
                self.init_manual_hotkey_listener()

        self.save_config()
        if self._use_windows_keep_mic_hot():
            self._touch_transcription_activity()
            self._start_background_audio_capture()
        else:
            self._stop_background_audio_capture()
        self.show_tray_balloon(self.translator.tr("settings_saved_message", hotkey=self.hotkey_str), 2000)
        self._update_brand_header()  # refresh the hotkey badge
        # Apply logging changes (level + file handler)
        self.apply_logging_configuration()
        # Update menu item states
        self.update_logfile_menu_action()
        self.update_play_last_recording_action()

    def _update_prompt_token_counter(self) -> None:
        """Update the token counter label (rendered as a pill badge) below the prompt."""
        text = self.prompt_input.toPlainText()
        tokens = estimate_tokens(text)
        limit = WHISPER_PROMPT_TOKEN_LIMIT if 'whisper' in (self.model_dropdown.currentText().lower() + ' ' + self.model_input.text().lower()) else None
        pal = getattr(self, "_theme_palette", None)
        if limit:
            over = tokens > limit
            near = tokens > int(limit * 0.85)
            if pal:
                color = pal["warn"] if over else (pal["accent"] if near else pal["muted"])
            else:
                color = 'red' if over else ('#aa7700' if near else '#555')
            exceeded_text = self.translator.tr("token_exceeded_text") if over else ""
            self.prompt_token_label.setText(self.translator.tr("token_counter_exceeded_label", tokens=tokens, limit=limit, exceeded_text=exceeded_text))
        else:
            color = pal["muted"] if pal else '#555'
            self.prompt_token_label.setText(self.translator.tr("token_counter_label", tokens=tokens))
        # Render as a rounded "pill" badge on the themed platforms.
        if pal:
            self.prompt_token_label.setStyleSheet(
                f"background: {pal['panel2']}; border: 1px solid {pal['border']}; "
                f"border-radius: 10px; padding: 2px 10px; color: {color};"
            )
        else:
            self.prompt_token_label.setStyleSheet(f"color: {color};")

    def _get_config_store(self) -> ConfigStore:
        """Return the (lazily created) config persistence helper."""
        if getattr(self, "_config_store", None) is None:
            self._config_store = ConfigStore(CONFIG_FILE, normalize_hotkey_string)
        return self._config_store

    def load_config(self) -> None:
        """
        Loads configuration from JSON file (with migrations, via ConfigStore).
        Ensures all default keys are present, and saves back if any were added.
        """
        self.config, config_updated = self._get_config_store().load()
        self.hotkey_str = self.config["hotkey"]
        self.post_rephrase_hotkey_str = self.config["post_rephrase_hotkey"]

        # If anything was migrated or defaulted, save the file back.
        if config_updated:
            self.save_config()

    def save_config(self) -> None:
        """Saves the current configuration to the JSON file (via ConfigStore)."""
        self._get_config_store().save(self.config)

    def _collect_validation_warnings(self, model_raw: str) -> List[str]:
        """Return a list of validation warning strings based on current form values.
        Rules (all case-insensitive):
        - If endpoint contains 'openai': model must contain 'openai' AND key must start with 'sk-'
        - If endpoint contains 'groq': model must contain 'groq' AND key must start with 'gsk'
        (Warnings are hints only; saving proceeds regardless.)
        """
        warnings: List[str] = []
        endpoint = self.api_endpoint_input.text().strip().lower()
        model_lc = model_raw.strip().lower()
        api_key = self.api_key_input.text().strip()
        api_key_lc = api_key.lower()
        is_custom_model = self.model_dropdown.currentText() == "Custom"

        if 'openai.com' in endpoint:
            if not is_custom_model and 'openai' not in model_lc:
                warnings.append("API endpoint contains 'openai', but the selected model does not contain 'openai'.")
            if not api_key_lc.startswith('sk-'):
                warnings.append("OpenAI API Key should start with 'sk-'.")
        if 'groq' in endpoint:
            if not is_custom_model and 'groq' not in model_lc:
                warnings.append("API endpoint contains 'groq', but the selected model does not contain 'groq'.")
            if not api_key_lc.startswith('gsk'):
                warnings.append("Groq API Key should start with 'gsk'.")

        # Whisper prompt length validation (soft warning only)
        if 'whisper' in model_lc:
            prompt_txt = self.prompt_input.toPlainText()
            tokens = estimate_tokens(prompt_txt)
            if tokens > WHISPER_PROMPT_TOKEN_LIMIT:
                warnings.append(f"Transcription Prompt exceeds the estimated limited the Whisper model can handle (max {WHISPER_PROMPT_TOKEN_LIMIT} tokens). ")
        return warnings

    # ---------------- Post Rewording (New Implementation) ----------------

    def _has_valid_api_settings(self) -> bool:
        """Return True if a transcription API key and endpoint are configured."""
        return bool(self.config.get("api_key", "").strip() and self.config.get("api_endpoint", "").strip())

    def _resolve_proxies(self, proxy_url: str, use_px: bool) -> Optional[Dict[str, str]]:
        """Decide outbound proxies (delegates to services.net.resolve_proxies)."""
        return net.resolve_proxies(proxy_url, use_px)

    def _config_proxies(self) -> Optional[Dict[str, str]]:
        """Resolve proxies from the saved configuration (used by background workers)."""
        return self._resolve_proxies(
            self.config.get("proxy_url", ""),
            self.config.get("use_local_px_proxy", False),
        )

    def _ui_proxies(self) -> Optional[Dict[str, str]]:
        """Resolve proxies from the current (unsaved) settings widgets (used by test buttons)."""
        use_px = hasattr(self, "use_px_proxy_checkbox") and self.use_px_proxy_checkbox.isChecked()
        proxy_url = self.proxy_url_input.text() if hasattr(self, "proxy_url_input") else ""
        return self._resolve_proxies(proxy_url, use_px)

    def _update_transcription_api_group_style(self) -> None:
        """Highlights the transcription API groupbox if its settings are incomplete."""
        url_missing = not self.api_endpoint_input.text().strip()
        key_missing = not self.api_key_input.text().strip()
        settings_incomplete = url_missing or key_missing

        if settings_incomplete:
            self.transcription_api_group.setStyleSheet(
                self._group_style("transcription_api_group", highlighted=True)
            )
        else:
            self.transcription_api_group.setStyleSheet(
                self._group_style("transcription_api_group")
            )

    def _update_rephrase_api_group_style(self) -> None:
        """Highlights the shared API groupbox if its settings are incomplete by directly setting the stylesheet."""
        # Check if any of the required fields are empty.
        # This validation is for the UI highlight only and is intentionally strict.
        url_missing = not self.rephrasing_api_url_input.text().strip()
        # Per user request, this check MUST NOT use the fallback key from tab 1.
        # The rephrasing key field must be filled on its own.
        key_missing = not self.rephrasing_api_key_input.text().strip()
        model_missing = not self.rephrasing_model_input.text().strip()

        settings_incomplete = url_missing or key_missing or model_missing

        # Directly set the stylesheet for the group box. This is more reliable
        # than using dynamic properties and repolishing.
        if settings_incomplete:
            self.shared_api_group.setStyleSheet(self._group_style("shared_api_group", highlighted=True))
        else:
            self.shared_api_group.setStyleSheet(self._group_style("shared_api_group"))

    def retranslate_ui(self) -> None:
        """Updates all UI texts to the currently selected language."""
        # Window Title
        self.setWindowTitle(self.translator.tr("window_title"))

        # Menus
        self.file_menu.setTitle(self.translator.tr("menu_file"))
        self.open_config_action.setText(self.translator.tr("menu_file_open_config"))
        self.open_log_file_action.setText(self.translator.tr("tray_log_action")) # Re-use systray translation
        self.play_last_recording_action.setText(self.translator.tr("tray_play_action")) # Re-use systray translation
        self.exit_action.setText(self.translator.tr("menu_file_exit"))
        self.help_menu.setTitle(self.translator.tr("menu_help"))
        self.about_action.setText(self.translator.tr("menu_help_about"))
        self.github_action.setText(self.translator.tr("menu_help_github"))

        # Tabs
        self.tabs.setTabText(0, self.translator.tr("tab_transcription"))
        self.tabs.setTabText(1, self.translator.tr("tab_rephrase"))
        self.tabs.setTabText(2, self.translator.tr("tab_transformations"))
        self.tabs.setTabText(3, self.translator.tr("tab_general"))
        self.tabs.setTabToolTip(0, self.translator.tr("tooltip_tab_transcription"))
        self.tabs.setTabToolTip(1, self.translator.tr("tooltip_tab_rephrase"))
        self.tabs.setTabToolTip(2, self.translator.tr("tooltip_tab_transformations"))
        self.tabs.setTabToolTip(3, self.translator.tr("tooltip_tab_general"))

        # Transcription Tab
        self.transcription_api_group.setTitle(self.translator.tr("transcription_api_group_title"))
        self.api_key_label.setText(self.translator.tr("api_key_label"))
        api_key_tooltip = self.translator.tr("api_key_tooltip")
        self.api_key_label.setToolTip(api_key_tooltip)
        self.api_key_input.setToolTip(api_key_tooltip)

        self.api_endpoint_label.setText(self.translator.tr("api_endpoint_label"))
        api_endpoint_tooltip = self.translator.tr("api_endpoint_tooltip")
        self.api_endpoint_label.setToolTip(api_endpoint_tooltip)
        self.api_endpoint_input.setToolTip(api_endpoint_tooltip)
        self.openai_button.setToolTip(api_endpoint_tooltip)
        self.groq_button.setToolTip(api_endpoint_tooltip)

        self.openai_button.setText("🤖  " + self.translator.tr("openai_button"))
        self.groq_button.setText("⚡  " + self.translator.tr("groq_button"))
        self.test_transcription_api_button.setText("🔌  " + self.translator.tr("test_connection_button"))
        self.test_transcription_api_button.setToolTip(self.translator.tr("test_connection_tooltip"))
        self.model_label.setText(self.translator.tr("model_label"))
        model_tooltip = self.translator.tr("model_tooltip")
        self.model_label.setToolTip(model_tooltip)
        self.model_dropdown.setToolTip(model_tooltip)
        self.model_input.setToolTip(model_tooltip)

        if is_MACOS:
            self.model_input.setPlaceholderText("whisper-1")
        else:
            self.model_input.setPlaceholderText(self.translator.tr("custom_model_placeholder"))
        self.transcription_temp_label_title.setText(self.translator.tr("temperature_label"))
        temp_tooltip = self.translator.tr("temperature_tooltip")
        self.transcription_temp_label_title.setToolTip(temp_tooltip)
        self.transcription_temp_slider.setToolTip(temp_tooltip)

        self.hotkey_label.setText(self.translator.tr("hotkey_label"))
        hotkey_tooltip = self.translator.tr("hotkey_tooltip")
        self.hotkey_label.setToolTip(hotkey_tooltip)
        self.hotkey_display.setToolTip(hotkey_tooltip)
        self.set_hotkey_button.setToolTip(hotkey_tooltip)

        self.set_hotkey_button.setText(self.translator.tr("set_hotkey_button"))
        self.push_to_talk_checkbox.setText(self.translator.tr("push_to_talk_checkbox"))
        self.push_to_talk_checkbox.setToolTip(self.translator.tr("push_to_talk_tooltip"))
        self.windows_keep_mic_hot_checkbox.setText(self.translator.tr("windows_keep_mic_hot_checkbox"))
        self.windows_keep_mic_hot_checkbox.setToolTip(self.translator.tr("windows_keep_mic_hot_tooltip"))
        self.windows_keep_mic_hot_idle_label.setText(self.translator.tr("windows_keep_mic_hot_idle_label"))
        self.windows_keep_mic_hot_idle_input.setToolTip(self.translator.tr("windows_keep_mic_hot_idle_tooltip"))
        self.windows_keep_mic_hot_idle_label.setToolTip(self.translator.tr("windows_keep_mic_hot_idle_tooltip"))
        self.min_recording_label.setText(self.translator.tr("min_recording_label"))
        min_recording_tooltip = self.translator.tr("min_recording_tooltip")
        self.min_recording_label.setToolTip(min_recording_tooltip)
        self.min_recording_input.setToolTip(min_recording_tooltip)
        self.input_language_label.setText(self.translator.tr("input_language_label"))
        input_lang_tooltip = self.translator.tr("input_language_tooltip")
        self.input_language_label.setToolTip(input_lang_tooltip)
        self.language_input.setToolTip(input_lang_tooltip)

        self.gain_label.setText(self.translator.tr("gain_label"))
        gain_tooltip = self.translator.tr("gain_tooltip")
        self.gain_label.setToolTip(gain_tooltip)
        self.gain_input.setToolTip(gain_tooltip)

        self.ffmpeg_label.setText(self.translator.tr("ffmpeg_label"))
        ffmpeg_tooltip = self.translator.tr("ffmpeg_tooltip")
        self.ffmpeg_label.setToolTip(ffmpeg_tooltip)
        self.ffmpeg_path_input.setToolTip(ffmpeg_tooltip)
        self.ffmpeg_path_input.setPlaceholderText(self.translator.tr("ffmpeg_path_placeholder"))
        self.ffmpeg_browse_button.setText(self.translator.tr("ffmpeg_browse_button"))
        # Status text depends on live detection, not just language — recompute on retranslate.
        self._refresh_ffmpeg_status()

        self.transcription_prompt_label.setText(self.translator.tr("transcription_prompt_label"))
        prompt_tooltip = self.translator.tr("transcription_prompt_tooltip")
        self.transcription_prompt_label.setToolTip(prompt_tooltip)
        self.prompt_input.setToolTip(prompt_tooltip)
        self.prompt_input.setPlaceholderText(self.translator.tr("transcription_prompt_placeholder"))

        # Rephrase Tab
        self.liveprompt_group.setTitle(self.translator.tr("liveprompt_group_title"))
        self.liveprompt_enabled_checkbox.setText(self.translator.tr("liveprompt_enable_checkbox"))
        self.liveprompt_enabled_checkbox.setToolTip(self.translator.tr("liveprompt_enable_tooltip"))
        self.liveprompt_help_button.setToolTip(self.translator.tr("liveprompt_help_button_tooltip"))
        self.liveprompt_trigger_label.setText(self.translator.tr("liveprompt_trigger_label"))
        lp_trigger_tooltip = self.translator.tr("liveprompt_trigger_words_tooltip")
        self.liveprompt_trigger_label.setToolTip(lp_trigger_tooltip)
        self.liveprompt_trigger_words_input.setToolTip(lp_trigger_tooltip)

        self.liveprompt_trigger_scan_depth_label.setText(self.translator.tr("liveprompt_trigger_scan_depth_label"))
        lp_scan_depth_tooltip = self.translator.tr("liveprompt_trigger_scan_depth_tooltip")
        self.liveprompt_trigger_scan_depth_label.setToolTip(lp_scan_depth_tooltip)
        self.liveprompt_trigger_scan_depth_input.setToolTip(lp_scan_depth_tooltip)

        self.liveprompt_strip_trigger_checkbox.setText(self.translator.tr("liveprompt_strip_trigger_checkbox"))
        self.liveprompt_strip_trigger_checkbox.setToolTip(self.translator.tr("liveprompt_strip_trigger_tooltip"))

        self.liveprompt_system_prompt_label.setText(self.translator.tr("liveprompt_system_prompt_label"))
        lp_prompt_tooltip = self.translator.tr("liveprompt_system_prompt_tooltip")
        self.liveprompt_system_prompt_label.setToolTip(lp_prompt_tooltip)
        self.liveprompt_system_prompt_input.setToolTip(lp_prompt_tooltip)

        self.rephrase_context_checkbox.setText(self.translator.tr("rephrase_context_checkbox"))
        self.rephrase_context_checkbox.setToolTip(self.translator.tr("rephrase_context_tooltip"))

        self.generic_rephrase_group.setTitle(self.translator.tr("generic_rephrase_group_title"))
        self.generic_rephrase_enabled_checkbox.setText(self.translator.tr("generic_rephrase_enable_checkbox"))
        self.generic_rephrase_prompt_label.setText(self.translator.tr("generic_rephrase_prompt_label"))
        gr_prompt_tooltip = self.translator.tr("generic_rephrase_prompt_tooltip")
        self.generic_rephrase_prompt_label.setToolTip(gr_prompt_tooltip)
        self.generic_rephrase_prompt_input.setToolTip(gr_prompt_tooltip)

        self.shared_api_group.setTitle(self.translator.tr("shared_api_group_title"))
        self.shared_api_group.setToolTip(self.translator.tr("shared_api_group_tooltip"))
        self.rephrasing_api_url_label.setText(self.translator.tr("rephrase_api_url_label"))
        self.rephrasing_api_url_label.setToolTip(api_endpoint_tooltip)
        self.rephrasing_api_url_input.setToolTip(api_endpoint_tooltip)

        self.rephrasing_api_key_label.setText(self.translator.tr("rephrase_api_key_label"))
        self.rephrasing_api_key_label.setToolTip(api_key_tooltip)
        self.rephrasing_api_key_input.setToolTip(api_key_tooltip)

        self.rephrasing_model_label.setText(self.translator.tr("rephrase_model_label"))
        self.rephrasing_model_label.setToolTip(model_tooltip)
        self.rephrasing_model_input.setToolTip(model_tooltip)

        self.rephrasing_temp_label_title.setText(self.translator.tr("temperature_label"))
        self.rephrasing_temp_label_title.setToolTip(temp_tooltip)
        self.rephrasing_temp_slider.setToolTip(temp_tooltip)

        # Test API Button
        self.test_rephrasing_api_button.setText("🔌  " + self.translator.tr("test_api_button"))
        self.test_rephrasing_api_button.setToolTip(self.translator.tr("test_api_button_tooltip"))

        # Transformations Tab
        self.transformations_tab_description_label.setText(self.translator.tr("transformations_tab_description"))
        self.transformations_info_label.setText(self.translator.tr("transformations_info", max_entries=self.max_post_rephrasing_entries))
        self.caption_label.setText(self.translator.tr("caption_label"))
        self.text_label.setText(self.translator.tr("text_label"))
        self.post_rp_text_edit.setPlaceholderText(self.translator.tr("text_placeholder"))
        self.post_rp_add_btn.setText(self.translator.tr("add_button"))
        self.post_rp_remove_btn.setText(self.translator.tr("remove_button"))

        # Post Rephrase Hotkey
        self.pr_hotkey_group.setTitle(self.translator.tr("post_rephrase_hotkey_group_title"))
        self.pr_hotkey_label.setText(self.translator.tr("post_rephrase_hotkey_label"))
        pr_hotkey_tooltip = self.translator.tr("post_rephrase_hotkey_tooltip")
        self.pr_hotkey_group.setToolTip(pr_hotkey_tooltip)
        self.pr_hotkey_display.setToolTip(pr_hotkey_tooltip)
        self.set_pr_hotkey_button.setToolTip(pr_hotkey_tooltip)
        self.set_pr_hotkey_button.setText(self.translator.tr("set_hotkey_button"))

        # General Tab
        self.ui_language_label.setText(self.translator.tr("ui_language_label"))
        self.color_theme_label.setText(self.translator.tr("color_theme_label"))
        self.color_theme_selector.setItemText(0, self.translator.tr("color_theme_system"))
        self.color_theme_selector.setItemText(1, self.translator.tr("color_theme_light"))
        self.color_theme_selector.setItemText(2, self.translator.tr("color_theme_dark"))
        self.input_device_label.setText(self.translator.tr("input_device_label"))
        if self.input_device_selector.count() > 0:
            self.input_device_selector.setItemText(0, self.translator.tr("input_device_default"))
        self.restore_clipboard_checkbox.setText(self.translator.tr("restore_clipboard_checkbox"))
        self.debug_logging_checkbox.setText(self.translator.tr("debug_logging_checkbox"))
        self.file_logging_checkbox.setText(self.translator.tr("file_logging_checkbox"))
        self.redact_log_checkbox.setText(self.translator.tr("redact_log_checkbox"))
        self.redact_log_checkbox.setToolTip(self.translator.tr("redact_log_tooltip"))
        self.proxy_url_label.setText(self.translator.tr("proxy_url_label"))
        self.proxy_url_input.setToolTip(self.translator.tr("proxy_url_tooltip"))
        self.use_px_proxy_checkbox.setText(self.translator.tr("use_px_proxy_checkbox"))
        self.use_px_proxy_checkbox.setToolTip(self.translator.tr("use_px_proxy_tooltip"))
        self.test_internet_button.setText("🌐  " + self.translator.tr("test_internet_button"))
        self.test_internet_button.setToolTip(self.translator.tr("test_internet_tooltip"))
        self.log_retention_label.setText(self.translator.tr("log_retention_label"))
        self.log_retention_input.setToolTip(self.translator.tr("log_retention_tooltip"))
        self.systray_double_click_copy_checkbox.setText(self.translator.tr("systray_double_click_copy_checkbox"))
        self.quit_without_confirmation_checkbox.setText(self.translator.tr("quit_without_confirmation_checkbox"))
        self.quit_without_confirmation_checkbox.setToolTip(self.translator.tr("quit_without_confirmation_tooltip"))
        self.play_g_button.setText(self.translator.tr("play_last_recording_button"))
        self.play_g_button.setToolTip(self.translator.tr("play_last_recording_tooltip"))
        self.alt_clipboard_lib_checkbox.setText(self.translator.tr("alt_clipboard_lib_checkbox"))
        self.alt_clipboard_lib_checkbox.setToolTip(self.translator.tr("alt_clipboard_lib_tooltip"))
        self.post_rephrase_auto_select_all_checkbox.setText(self.translator.tr("post_rephrase_auto_select_all_checkbox"))
        self.post_rephrase_auto_select_all_checkbox.setToolTip(
            self.translator.tr("post_rephrase_auto_select_all_tooltip")
        )

        # Save Button
        self.save_button.setText(self.translator.tr("save_button"))

        # Recording & Hotkey card (wraps hotkey/language/gain/push-to-talk)
        if hasattr(self, "recording_group"):
            self.recording_group.setTitle(self.translator.tr("recording_group_title"))

        # Update other dynamic texts
        self._update_prompt_token_counter()
        self.init_tray_icon() # Re-init to update menu item texts
        if hasattr(self, "_update_brand_header"):
            self._update_brand_header()

    def _on_color_theme_changed(self, *_args: object) -> None:
        """Persist the chosen colour theme and re-apply it immediately."""
        self.config["color_theme"] = self.color_theme_selector.currentData() or "system"
        self.save_config()
        self.apply_theme()

    def _populate_input_device_selector(self) -> None:
        """Fill the input-device dropdown with the system default + available microphones."""
        self.input_device_selector.blockSignals(True)
        self.input_device_selector.clear()
        self.input_device_selector.addItem(self.translator.tr("input_device_default"), "")
        for device in self.selectable_input_devices():
            self.input_device_selector.addItem(device["name"], device["name"])
        configured = self.config.get("input_device_name", "") or ""
        idx = self.input_device_selector.findData(configured)
        self.input_device_selector.setCurrentIndex(idx if idx >= 0 else 0)
        self.input_device_selector.blockSignals(False)

    def _on_input_device_changed(self, *_args: object) -> None:
        """Persist the chosen input device and reopen the capture stream on it."""
        self.config["input_device_name"] = self.input_device_selector.currentData() or ""
        self.save_config()
        self.apply_input_device_selection()

    def change_language(self, lang_name: str) -> None:
        """
        Changes the application's UI language.

        Args:
            lang_name (str): The display name of the language to switch to.
        """
        lang_map = {"English": "en", "Deutsch": "de", "Español": "es", "Français": "fr"}
        lang_code = lang_map.get(lang_name, "en")
        # Swap the default prompt texts to the new language, but only where the user
        # has not manually customized them (i.e. they still match a known default).
        self._maybe_swap_default_prompts(lang_code)
        self.translator.set_language(lang_code)
        self.config["ui_language"] = lang_code
        self.retranslate_ui()

    def _maybe_swap_default_prompts(self, new_lang_code: str) -> None:
        """
        Replace the prompt fields with the new language's defaults, but only for prompts the
        user has not manually edited. A prompt counts as "not edited" if it still matches one
        of the known default prompts in any language.
        """
        prompt_fields = [
            (self.prompt_input, DEFAULT_TRANSCRIPTION_PROMPTS),
            (self.liveprompt_system_prompt_input, DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS),
            (self.generic_rephrase_prompt_input, DEFAULT_GENERIC_REPHRASE_PROMPTS),
        ]
        for widget, prompt_map in prompt_fields:
            current_text = widget.toPlainText()
            if _is_known_default_prompt(prompt_map, current_text):
                new_default = _default_prompt_for(prompt_map, new_lang_code)
                if current_text.strip() != new_default:
                    widget.setPlainText(new_default)
                    logging.info("Swapped a default prompt to the new UI language.")

    def apply_logging_configuration(self) -> None:
        """Apply logging level and file handler based on current config."""
        logger = logging.getLogger()
        # Update level
        logger.setLevel(logging.DEBUG if self.config["debug_logging"] else logging.INFO)
        # Update the shared log-redaction state so all log sites (including worker threads) honor it
        LOG_REDACTION_STATE["enabled"] = bool(self.config.get("redact_transcription_in_log", True))

        # In windowed/pythonw mode, sys.stderr is the crash-log file (see run.py). The default
        # basicConfig StreamHandler points at it, so at DEBUG it would append EVERY log line and
        # the file would grow without bound over a long-running session. Raise the stream
        # handler(s) to WARNING there — full detail still goes to the rotating WhisperTyper.log,
        # and crash tracebacks are written directly by the excepthook regardless.
        if os.environ.get("WHISPERTYPER_WINDOWED") == "1":
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setLevel(logging.WARNING)
        # Remove existing file handler if present
        if getattr(self, '_file_log_handler', None):
            try:
                logger.removeHandler(self._file_log_handler)
                self._file_log_handler.close()
            except Exception:
                pass
            self._file_log_handler = None

        # Add file handler if enabled
        if self.config["file_logging"]:
            try:
                # Daily (time-based) rotation: at midnight a new log file is started and the
                # previous day's file is renamed to "WhisperTyper.log.YYYY-MM-DD".
                # Only the configured number of old daily files is kept.
                retention_days = int(self.config.get("log_retention_days", 3) or 0)
                fh = logging.handlers.TimedRotatingFileHandler(
                    LOG_FILE_PATH,
                    when='midnight',
                    interval=1,
                    backupCount=max(0, retention_days),
                    encoding='utf-8',
                    delay=True,
                )
                fh.suffix = "%Y-%m-%d"
                fh.setLevel(logging.DEBUG)  # always capture full detail in file
                fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
                logger.addHandler(fh)
                self._file_log_handler = fh
                logging.info(
                    f"File logging enabled (daily rotation, keeping {max(0, retention_days)} days): {LOG_FILE_PATH}"
                )
            except Exception as e:
                logging.error(f"Failed to enable file logging: {e}")

    def update_logfile_menu_action(self) -> None:
        """Updates the enabled/disabled state of the 'Open Log File' action in the tray menu."""
        if hasattr(self, 'open_log_action'):
            log_file_exists = os.path.isfile(LOG_FILE_PATH)
            self.open_log_action.setEnabled(log_file_exists)
            if hasattr(self, 'open_log_file_action'): # Also update the main menu action
                self.open_log_file_action.setEnabled(log_file_exists)

    def open_log_file(self) -> None:
        """Opens the log file with the system's default application (text editor)."""
        if not os.path.isfile(LOG_FILE_PATH):
            self.show_tray_balloon(self.translator.tr("log_file_not_exist_message"), 2000)
            self.update_logfile_menu_action()  # Update menu state
            return

        try:
            open_with_default_app(LOG_FILE_PATH)
        except Exception as e:
            logging.error(f"Failed to open log file: {e}")
            self.show_tray_balloon(self.translator.tr("log_file_open_fail_message", error=e), 3000)
