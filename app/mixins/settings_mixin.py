"""SettingsMixin — the settings window: composes the pages, loads/saves the form, retranslates."""
from __future__ import annotations

import logging

from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QHBoxLayout, QMenuBar, QMessageBox, QSizePolicy, QStyle

from app.core.config_store import ConfigStore
from app.core.constants import CONFIG_FILE, WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH
from app.core.env import is_MACOS
from app.core.hotkeys import normalize_hotkey_string
from app.core.paths import resource_path
from app.core.replacements import ReplacementError, Replacements
from app.mixins.settings_pages.api_page import ApiSettingsMixin
from app.mixins.settings_pages.general_page import GeneralSettingsMixin
from app.mixins.settings_pages.recording_page import RecordingSettingsMixin
from app.mixins.settings_pages.transcription_page import TranscriptionPageMixin
from app.ui.api_keys import ApiKeysTab
from app.ui.connection_tester import ConnectionTester
from app.ui.durations import BALLOON_SHORT_MS
from app.ui.replacements import ReplacementsTab
from app.ui.settings.bindings import load_bindings, save_bindings
from app.ui.settings.texts import apply_texts
from app.ui.settings.widgets import SliderWheelToScrollArea
from app.ui.transformations_tab import TransformationsEditor

#: Default window width on first launch (the height default lives in the config).
_DEFAULT_WINDOW_WIDTH = 760


class SettingsMixin(ApiSettingsMixin, TranscriptionPageMixin, RecordingSettingsMixin, GeneralSettingsMixin):
    """Settings window: build, bind, validate, retranslate, config persistence."""

    def init_ui(self) -> None:
        """Build the settings window from main_window.ui and load the saved configuration into it."""
        uic.loadUi(resource_path("resources", "main_window.ui"), self)
        self._build_menu_bar()

        # Pages and controls the .ui file does not contain.
        self._api_keys_tab = ApiKeysTab(self.config["api_key_profiles"], self.translator, self)
        self._api_keys_tab.groq_rotation.setChecked(self.config["groq_key_rotation"])
        self.tabs.insertTab(self.tabs.indexOf(self.general_tab), self._api_keys_tab, "")
        self._replacements_tab = ReplacementsTab(self.config["replacements_rules"], self.config["replacements_enabled"],
                                                 self.translator, self)
        self.tabs.insertTab(self.tabs.indexOf(self.general_tab), self._replacements_tab, "")
        try:
            self._replacement_rules = Replacements(self.config["replacements_rules"])
        except ReplacementError as error:
            self._replacement_rules = Replacements("")
            logging.warning("replacements_config_invalid line=%s; corrections disabled until settings are fixed", error.line)
        self._init_transcription_page()
        self._build_recording_controls()
        self._build_general_page()
        self._init_transformations_page()

        load_bindings(self, self.config)
        self.hotkey_display.setText(self.hotkey_str)
        self.pr_hotkey_display.setText(self.config["post_rephrase_hotkey"])

        # Live behaviour, connected after loading so the saved values do not trigger handlers.
        self._connect_transcription_page()
        self._connect_recording_controls()
        self._connect_general_page()
        self._init_api_settings()
        self._sync_transformations_to_config()  # Store the normalized templates once.
        self._connection_tester = ConnectionTester(self)
        self.test_transcription_api_button.clicked.connect(self._connection_tester.test_transcription)
        self.test_rephrasing_api_button.clicked.connect(self._connection_tester.test_rephrasing)
        self.test_internet_button.clicked.connect(self._connection_tester.test_internet)
        self.set_hotkey_button.clicked.connect(self.start_hotkey_capture)
        self.set_pr_hotkey_button.clicked.connect(self.start_hotkey_capture)
        self.play_g_button.clicked.connect(self.play_latest_recording)
        self.liveprompt_help_button.setFixedSize(22, 22)  # Rendered as a round "?" badge by the theme.
        self.liveprompt_help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.liveprompt_help_button.clicked.connect(self.show_liveprompt_help)
        if is_MACOS:
            # Trackpad scrolling over a slider should scroll the page, not change the temperature.
            SliderWheelToScrollArea(self.transcription_temp_slider, self.transcription_scroll_area)
            SliderWheelToScrollArea(self.rephrasing_temp_slider, self.rephrasing_scroll_area)
        self._build_save_row()

        app_icon = self._get_app_icon()
        self.setWindowIcon(app_icon if not app_icon.isNull()
                           else self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        self.apply_theme()
        self.retranslate_ui()

        # Enforce a minimum window size and restore the last size.
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        try:
            width = max(int(self.config.get("window_width", _DEFAULT_WINDOW_WIDTH)), WINDOW_MIN_WIDTH)
            height = max(int(self.config.get("window_height", WINDOW_MIN_HEIGHT)), WINDOW_MIN_HEIGHT)
        except (TypeError, ValueError):
            width, height = _DEFAULT_WINDOW_WIDTH, WINDOW_MIN_HEIGHT
        self.resize(width, height)

    def _build_menu_bar(self) -> None:
        """File/Help menus and the branding header above the tabs (captions come from retranslate)."""
        self.menu_bar = QMenuBar(self)
        self.main_layout.insertWidget(0, self.menu_bar)
        self.main_layout.setContentsMargins(0, 0, 0, 5)
        self._install_brand_header()
        self.file_menu = self.menu_bar.addMenu("")
        self.help_menu = self.menu_bar.addMenu("")
        for menu, attr, handler in (
            (self.file_menu, "open_config_action", self.open_config_file),
            (self.file_menu, "open_log_file_action", self.open_log_file),
            (self.file_menu, "play_last_recording_action", self.play_latest_recording),
            (self.file_menu, None, None),
            (self.file_menu, "exit_action", self.quit_app),
            (self.help_menu, "about_action", self.show_about_dialog),
            (self.help_menu, "github_action", self.open_github_link),
        ):
            if attr is None:
                menu.addSeparator()
                continue
            action = QAction("", self)
            action.triggered.connect(handler)
            menu.addAction(action)
            setattr(self, attr, action)

    def _build_save_row(self) -> None:
        """Compact, right-aligned Save button instead of a full-width one."""
        self.save_button.clicked.connect(self.save_settings)
        self.save_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.save_button.setMinimumWidth(150)
        self.save_button.setMaximumWidth(200)
        self.main_layout.removeWidget(self.save_button)
        save_row = QHBoxLayout()
        save_row.setContentsMargins(0, 0, 16, 6)
        save_row.addStretch()
        save_row.addWidget(self.save_button)
        self.main_layout.addLayout(save_row)

    def _init_transformations_page(self) -> None:
        """Template editor on the transformations tab; edits are mirrored into the live config."""
        # Description texts take minimal height; the editor splitter takes the rest.
        for label in (self.transformations_tab_description_label, self.transformations_unavailable_label,
                      self.transformations_info_label):
            label.setSizePolicy(label.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Maximum)
        self.splitter.setSizePolicy(self.splitter.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        self._transformations = TransformationsEditor(
            self.config.get("post_rephrasing_entries", []), self.translator, splitter=self.splitter,
            list_placeholder=self.post_rp_list_placeholder, caption_edit=self.post_rp_caption_edit,
            text_edit=self.post_rp_text_edit, show_during_recording=self.post_rp_show_during_recording_checkbox,
            add_button=self.post_rp_add_btn, remove_button=self.post_rp_remove_btn,
        )
        self.post_rp_list = self._transformations.list
        self._transformations.changed.connect(self._sync_transformations_to_config)

    def _sync_transformations_to_config(self) -> None:
        """Mirror the edited templates into the live config, which the palettes read."""
        self.config["post_rephrasing_entries"] = self._transformations.entries()
        self._refresh_api_state()

    def save_settings(self) -> None:
        """Validate and store the form, then apply hotkeys, capture, logging and warmup."""
        raw_rules = self._replacements_tab.editor.toPlainText()
        try:
            replacement_rules = Replacements(raw_rules)
        except ReplacementError as error:
            QMessageBox.warning(self, self.translator.tr("tab_replacements"),
                                self.translator.tr("replacements_invalid", line=error.line))
            self.tabs.setCurrentWidget(self._replacements_tab)
            self._replacements_tab.editor.setFocus()
            return
        warnings = self._collect_validation_warnings(self.model_dropdown.currentText().strip())
        if warnings:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle(self.translator.tr("validation_warning_title"))
            msg.setText(self.translator.tr("validation_warning_text"))
            msg.setInformativeText("\n".join(f"- {w}" for w in warnings))
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        if not self._save_api_settings():
            return
        self.config["replacements_enabled"] = self._replacements_tab.enabled.isChecked()
        self.config["replacements_rules"] = raw_rules
        self._replacement_rules = replacement_rules
        save_bindings(self, self.config)
        self._save_recording_controls()
        self._sync_transformations_to_config()
        self._apply_pending_hotkeys()

        self.save_config()
        if self._use_windows_keep_mic_hot():
            self._touch_transcription_activity()
            self._start_background_audio_capture()
        else:
            self._stop_background_audio_capture()
        self.show_tray_balloon(self.translator.tr("settings_saved_message", hotkey=self.hotkey_str), BALLOON_SHORT_MS)
        self._update_brand_header()
        self.apply_logging_configuration()
        self._schedule_http_warmup(activate=True)
        self.update_logfile_menu_action()
        self.update_play_last_recording_action()

    def _apply_pending_hotkeys(self) -> None:
        """Adopt edited hotkeys; restart the listener (macOS asks for an app restart instead)."""
        pending = normalize_hotkey_string(self.hotkey_display.text()) or self.hotkey_display.text().strip()
        pending_pr = normalize_hotkey_string(self.pr_hotkey_display.text()) or self.pr_hotkey_display.text().strip()
        self.hotkey_display.setText(pending)
        self.pr_hotkey_display.setText(pending_pr)
        if pending == self.hotkey_str and pending_pr == self.post_rephrase_hotkey_str:
            return
        self.hotkey_str = self.config["hotkey"] = pending
        self.post_rephrase_hotkey_str = self.config["post_rephrase_hotkey"] = pending_pr
        if is_MACOS:
            # Restarting the listener at runtime conflicts with the macOS accessibility permissions.
            QMessageBox.information(self, self.translator.tr("macos_hotkey_restart_title"),
                                    self.translator.tr("macos_hotkey_restart_text"))
        else:
            self.init_manual_hotkey_listener()

    def _get_config_store(self) -> ConfigStore:
        """Return the (lazily created) config persistence helper."""
        if getattr(self, "_config_store", None) is None:
            self._config_store = ConfigStore(CONFIG_FILE, normalize_hotkey_string)
        return self._config_store

    def load_config(self) -> None:
        """Load the configuration (with migrations) and write it back if anything was added."""
        self.config, config_updated = self._get_config_store().load()
        self.hotkey_str = self.config["hotkey"]
        self.post_rephrase_hotkey_str = self.config["post_rephrase_hotkey"]
        if config_updated:
            self.save_config()

    def save_config(self) -> None:
        """Write the current configuration to disk."""
        self._get_config_store().save(self.config)

    def retranslate_ui(self) -> None:
        """Apply the current UI language to every text of the window."""
        tr = self.translator.tr
        self.setWindowTitle(tr("window_title"))
        apply_texts(self, tr)
        for tab, title_key, tooltip_key in (
            (self.transcription_tab, "tab_transcription", "tooltip_tab_transcription"),
            (self.rephrasing_tab, "tab_rephrase", "tooltip_tab_rephrase"),
            (self.post_rephrasing_tab, "tab_transformations", "tooltip_tab_transformations"),
            (self._api_keys_tab, "tab_api_keys", "tooltip_tab_api_keys"),
            (self._replacements_tab, "tab_replacements", "tooltip_tab_replacements"),
            (self.general_tab, "tab_general", "tooltip_tab_general"),
        ):
            index = self.tabs.indexOf(tab)
            self.tabs.setTabText(index, tr(title_key))
            self.tabs.setTabToolTip(index, tr(tooltip_key))
        self.transformations_info_label.setText(tr("transformations_info", max_entries=self._transformations.max_entries))
        self._api_keys_tab.retranslate_ui()
        self._replacements_tab.retranslate_ui()
        self._retranslate_api_settings()
        self._retranslate_general_page()
        self._update_recording_format_controls()
        self._refresh_ffmpeg_status()
        self._update_prompt_token_counter()
        self.init_tray_icon()  # Rebuild the tray menu with the new texts.
        self._update_brand_header()
