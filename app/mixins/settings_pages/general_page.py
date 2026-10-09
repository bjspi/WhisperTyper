"""General page and window actions: languages, theme, input device, text insertion, logging, files."""
from __future__ import annotations

import logging
import os

from PyQt6.QtCore import QSignalBlocker, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QCheckBox, QMessageBox

from app.core.constants import CONFIG_FILE, LOG_FILE_PATH
from app.core.env import is_WINDOWS
from app.core.i18n import UI_LANGUAGES
from app.core.prompts import (
    DEFAULT_GENERIC_REPHRASE_PROMPTS,
    DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS,
    DEFAULT_TRANSCRIPTION_PROMPTS,
    _default_prompt_for,
    _is_known_default_prompt,
)
from app.platform.system import open_with_default_app
from app.services.logging_config import apply_logging_config
from app.ui.durations import BALLOON_ERROR_MS, BALLOON_MAX_MS, BALLOON_SHORT_MS

GITHUB_URL = "https://github.com/bjspi/WhisperTyper"
_COLOR_THEMES = (("system", "color_theme_system"), ("light", "color_theme_light"), ("dark", "color_theme_dark"))


class GeneralSettingsMixin:
    """General settings plus the window's file/help actions."""

    def _build_general_page(self) -> None:
        """Fill the selectors and add the controls the .ui file does not contain."""
        for code, name in UI_LANGUAGES.items():
            self.ui_language_selector.addItem(name, code)
        self.ui_language_selector.setCurrentIndex(max(0, self.ui_language_selector.findData(self.config["ui_language"])))
        for theme, _key in _COLOR_THEMES:
            self.color_theme_selector.addItem("", theme)
        self.color_theme_selector.setCurrentIndex(max(0, self.color_theme_selector.findData(
            self.config.get("color_theme", "system"))))
        self._populate_input_device_selector()

        self.post_rephrase_auto_select_all_checkbox = QCheckBox(self)
        play_button_index = self.general_layout.indexOf(self.play_g_button)
        self.general_layout.insertWidget(play_button_index if play_button_index >= 0 else self.general_layout.count(),
                                         self.post_rephrase_auto_select_all_checkbox)

    def _connect_general_page(self) -> None:
        """React to immediate-apply selectors and dependent checkboxes."""
        self.ui_language_selector.currentIndexChanged.connect(
            lambda _index: self.change_language(self.ui_language_selector.currentData()))
        self.color_theme_selector.currentIndexChanged.connect(self._on_color_theme_changed)
        self.input_device_selector.currentIndexChanged.connect(self._on_input_device_changed)
        # The clipboard fallback only matters while direct Unicode input is enabled.
        self.windows_sendinput_fallback_checkbox.setEnabled(is_WINDOWS and self.windows_sendinput_text_checkbox.isChecked())
        self.windows_sendinput_text_checkbox.toggled.connect(
            lambda checked: self.windows_sendinput_fallback_checkbox.setEnabled(is_WINDOWS and checked))

    def _retranslate_general_page(self) -> None:
        """Translate the selector entries whose data stays fixed."""
        for index, (_theme, key) in enumerate(_COLOR_THEMES):
            self.color_theme_selector.setItemText(index, self.translator.tr(key))
        if self.input_device_selector.count() > 0:
            self.input_device_selector.setItemText(0, self.translator.tr("input_device_default"))

    def _on_color_theme_changed(self, *_args: object) -> None:
        """Persist the chosen colour theme and re-apply it immediately."""
        self.config["color_theme"] = self.color_theme_selector.currentData() or "system"
        self.save_config()
        self.apply_theme()

    def _populate_input_device_selector(self) -> None:
        """Fill the input-device dropdown with the system default + available microphones."""
        with QSignalBlocker(self.input_device_selector):
            self.input_device_selector.clear()
            self.input_device_selector.addItem(self.translator.tr("input_device_default"), "")
            for device in self.selectable_input_devices():
                self.input_device_selector.addItem(device["name"], device["name"])
            index = self.input_device_selector.findData(self.config.get("input_device_name", "") or "")
            self.input_device_selector.setCurrentIndex(max(0, index))

    def _on_input_device_changed(self, *_args: object) -> None:
        """Persist the chosen input device and reopen the capture stream on it."""
        self.config["input_device_name"] = self.input_device_selector.currentData() or ""
        self.save_config()
        self.apply_input_device_selection()

    def change_language(self, lang_code: str) -> None:
        """Switch the UI language (and untouched default prompts) to ``lang_code``."""
        lang_code = lang_code if lang_code in UI_LANGUAGES else "en"
        self._maybe_swap_default_prompts(lang_code)
        self.translator.set_language(lang_code)
        self.config["ui_language"] = lang_code
        self.retranslate_ui()

    def _maybe_swap_default_prompts(self, new_lang_code: str) -> None:
        """Replace prompts the user never edited (still a known default) with the new language's default."""
        for widget, prompt_map in (
            (self.prompt_input, DEFAULT_TRANSCRIPTION_PROMPTS),
            (self.liveprompt_system_prompt_input, DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS),
            (self.generic_rephrase_prompt_input, DEFAULT_GENERIC_REPHRASE_PROMPTS),
        ):
            current_text = widget.toPlainText()
            if _is_known_default_prompt(prompt_map, current_text):
                new_default = _default_prompt_for(prompt_map, new_lang_code)
                if current_text.strip() != new_default:
                    widget.setPlainText(new_default)
                    logging.info("Swapped a default prompt to the new UI language.")

    def apply_logging_configuration(self) -> None:
        """Apply the saved logging level, redaction and file logging."""
        self._file_log_handler = apply_logging_config(self.config, getattr(self, "_file_log_handler", None))

    def show_liveprompt_help(self) -> None:
        """Explain LivePrompting in a long balloon."""
        self.show_tray_balloon(self.translator.tr("liveprompt_help_tooltip"), BALLOON_MAX_MS)

    def show_about_dialog(self) -> None:
        """Show the 'About' dialog."""
        QMessageBox.about(self, self.translator.tr("about_dialog_title"), self.translator.tr("about_dialog_text"))

    def open_config_file(self) -> None:
        """Open config.json in the default editor."""
        if os.path.exists(CONFIG_FILE):
            QDesktopServices.openUrl(QUrl.fromLocalFile(CONFIG_FILE))
        else:
            self.show_tray_balloon(self.translator.tr("config_file_not_found"), BALLOON_ERROR_MS)

    def open_github_link(self) -> None:
        """Open the project's GitHub repository."""
        QDesktopServices.openUrl(QUrl(GITHUB_URL))

    def update_logfile_menu_action(self) -> None:
        """Enable the 'Open Log File' actions only when the log file exists."""
        if hasattr(self, "open_log_action"):
            log_file_exists = os.path.isfile(LOG_FILE_PATH)
            self.open_log_action.setEnabled(log_file_exists)
            if hasattr(self, "open_log_file_action"):
                self.open_log_file_action.setEnabled(log_file_exists)

    def open_log_file(self) -> None:
        """Open the log file with the system's default application."""
        if not os.path.isfile(LOG_FILE_PATH):
            self.show_tray_balloon(self.translator.tr("log_file_not_exist_message"), BALLOON_SHORT_MS)
            self.update_logfile_menu_action()
            return
        try:
            open_with_default_app(LOG_FILE_PATH)
        except Exception as e:
            logging.error(f"Failed to open log file: {e}")
            self.show_tray_balloon(self.translator.tr("log_file_open_fail_message", error=e), BALLOON_ERROR_MS)
