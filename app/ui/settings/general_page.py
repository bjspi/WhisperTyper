"""General page: languages, theme, input device and text insertion options."""
from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QCheckBox, QMessageBox

from app import __version__
from app.controllers.file_actions import GITHUB_URL
from app.core.env import is_WINDOWS
from app.core.i18n import UI_LANGUAGES
from app.ui.durations import BALLOON_MAX_MS
from app.ui.settings.base import SettingsWindowBase

_COLOR_THEMES = (("system", "color_theme_system"), ("light", "color_theme_light"), ("dark", "color_theme_dark"))
_IP_FAMILIES = (("auto", "network_ip_family_auto"), ("ipv4", "network_ip_family_ipv4"),
                ("ipv6", "network_ip_family_ipv6"))


class GeneralPage(SettingsWindowBase):
    """General settings plus the window's help dialogs."""

    def _build_general_page(self) -> None:
        """Fill the selectors and add the controls the .ui file does not contain."""
        for code, name in UI_LANGUAGES.items():
            self.ui_language_selector.addItem(name, code)
        self.ui_language_selector.setCurrentIndex(max(0, self.ui_language_selector.findData(self.config["ui_language"])))
        for theme, _key in _COLOR_THEMES:
            self.color_theme_selector.addItem("", theme)
        self.color_theme_selector.setCurrentIndex(max(0, self.color_theme_selector.findData(
            self.config.get("color_theme", "system"))))
        for family, _key in _IP_FAMILIES:
            self.network_ip_family_selector.addItem("", family)
        self.populate_input_devices()
        # The proxy field takes the row's free width; the test button keeps a fixed width.
        self.proxy_row_layout.setStretch(0, 1)
        self.network_ip_family_layout.addStretch(1)

        self.post_rephrase_auto_select_all_checkbox = QCheckBox(self)
        self.misc_layout.addWidget(self.post_rephrase_auto_select_all_checkbox)

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
        for index, (_family, key) in enumerate(_IP_FAMILIES):
            self.network_ip_family_selector.setItemText(index, self.translator.tr(key))
        if self.input_device_selector.count() > 0:
            self.populate_input_devices()

    def _on_color_theme_changed(self, *_args: object) -> None:
        """Persist the chosen colour theme and re-apply it immediately."""
        self.config["color_theme"] = self.color_theme_selector.currentData() or "system"
        self.ctx.save_config_keys("color_theme")
        self.apply_theme()

    def populate_input_devices(self) -> None:
        """Fill the input-device dropdown with the system default + available microphones."""
        with QSignalBlocker(self.input_device_selector):
            self.input_device_selector.clear()
            self.input_device_selector.addItem(self.translator.tr("input_device_default"), "")
            for device in self._recording.selectable_input_devices():
                self.input_device_selector.addItem(device["name"], device["name"])
            configured = (self.config.get("input_device_name", "") or "").strip()
            if self.input_device_selector.findData(configured) < 0:
                # Keep a device that is not connected right now selected instead of switching to the default.
                self.input_device_selector.addItem(
                    self.translator.tr("input_device_not_connected", device=configured), configured)
            self.input_device_selector.setCurrentIndex(self.input_device_selector.findData(configured))

    def _on_input_device_changed(self, *_args: object) -> None:
        """Persist the chosen input device and reopen the capture stream on it."""
        self.config["input_device_name"] = self.input_device_selector.currentData() or ""
        self.ctx.save_config_keys("input_device_name")
        self._recording.apply_input_device_selection()

    def change_language(self, lang_code: str) -> None:
        """Switch the UI language (and untouched default prompts) to ``lang_code``."""
        lang_code = lang_code if lang_code in UI_LANGUAGES else "en"
        self._maybe_swap_default_prompts(lang_code)
        self.translator.set_language(lang_code)
        self.config["ui_language"] = lang_code
        self.retranslate_ui()

    def _maybe_swap_default_prompts(self, new_lang_code: str) -> None:
        """Replace prompts the user never edited (still a known default) with the new language's default.

        The transcription prompt follows the input language first (see ``swap_default_transcription_prompt``).
        """
        self.swap_default_transcription_prompt(self.language_input.currentData() or "", new_lang_code)
        self._transformations.apply_language_defaults(new_lang_code)

    def show_liveprompt_help(self) -> None:
        """Explain LivePrompting in a long balloon."""
        self.ctx.notifier.show(self.translator.tr("liveprompt_help_tooltip"), BALLOON_MAX_MS)

    def show_about_dialog(self) -> None:
        """Show the 'About' dialog."""
        QMessageBox.about(self, self.translator.tr("about_dialog_title"),
                          self.translator.tr("about_dialog_text", version=__version__, url=GITHUB_URL))
