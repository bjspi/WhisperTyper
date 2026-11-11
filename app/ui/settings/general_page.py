"""General page: languages, theme, input device, system integration and text insertion options."""
from __future__ import annotations

from typing import List, Tuple

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QBoxLayout, QCheckBox, QGroupBox, QHBoxLayout, QLabel, QLayout, QSpinBox, QVBoxLayout, QWidget

from app.controllers.system_integration import COMMAND_KEY, START_MENU_KEY, switch_start_at_login
from app.core.env import is_WINDOWS
from app.core.i18n import UI_LANGUAGES
from app.platform import autostart, command_shim, start_menu
from app.ui.about_dialog import show_about
from app.ui.durations import BALLOON_MAX_MS
from app.ui.settings.base import SettingsWindowBase

_COLOR_THEMES = (("system", "color_theme_system"), ("light", "color_theme_light"), ("dark", "color_theme_dark"))
_IP_FAMILIES = (("auto", "network_ip_family_auto"), ("ipv4", "network_ip_family_ipv4"),
                ("ipv6", "network_ip_family_ipv6"))


def _layout_holding(widget: QWidget) -> "QBoxLayout | None":
    """The box layout that directly contains ``widget`` (searching nested layouts of its parent)."""
    parent = widget.parentWidget()

    def search(layout: "QLayout | None") -> "QBoxLayout | None":
        if layout is None:
            return None
        if isinstance(layout, QBoxLayout) and layout.indexOf(widget) >= 0:
            return layout
        for index in range(layout.count()):
            item = layout.itemAt(index)
            found = search(item.layout() if item is not None else None)
            if found is not None:
                return found
        return None

    return search(parent.layout() if parent is not None else None)


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

        # Below the palette options of the .ui file: how opaque the palette and the floating buttons are.
        self.recording_overlay_opacity_label = QLabel(self)
        self.recording_overlay_opacity_input = QSpinBox(self)
        self.recording_overlay_opacity_input.setRange(30, 100)
        self.recording_overlay_opacity_input.setSingleStep(5)
        self.recording_overlay_opacity_input.setSuffix(" %")
        anchor = self.recording_prompt_overlay_digit_keys_checkbox
        layout = _layout_holding(anchor)
        if layout is not None:
            row = QHBoxLayout()
            row.addWidget(self.recording_overlay_opacity_label)
            row.addWidget(self.recording_overlay_opacity_input)
            row.addStretch(1)
            layout.insertLayout(layout.indexOf(anchor) + 1, row)

        self.post_rephrase_auto_select_all_checkbox = QCheckBox(self)
        self.misc_layout.addWidget(self.post_rephrase_auto_select_all_checkbox)
        self._build_system_integration_group()

    def _build_system_integration_group(self) -> None:
        """The terminal command, Start menu and start at login switches; they apply at once and only where supported.

        The start at login has no config key: like the tray switch it reads and writes the registry.
        """
        self.system_integration_group = QGroupBox(self.misc_group.parentWidget())
        layout = QVBoxLayout(self.system_integration_group)
        self.command_on_path_checkbox = QCheckBox(self.system_integration_group)
        self.start_menu_entry_checkbox = QCheckBox(self.system_integration_group)
        for checkbox, key, supported in self._system_integration_switches():
            checkbox.setChecked(bool(self.config.get(key, True)))
            checkbox.setVisible(supported)
            layout.addWidget(checkbox)
        self.start_at_login_checkbox = QCheckBox(self.system_integration_group)
        self.start_at_login_checkbox.setVisible(autostart.supported())
        layout.addWidget(self.start_at_login_checkbox)
        self.refresh_start_at_login_switch()
        self.system_integration_group.setVisible(
            command_shim.supported() or start_menu.supported() or autostart.supported())
        self.general_layout.insertWidget(self.general_layout.indexOf(self.misc_group), self.system_integration_group)

    def _system_integration_switches(self) -> List[Tuple[QCheckBox, str, bool]]:
        """(switch, config key, supported here) of the system integration group."""
        return [(self.command_on_path_checkbox, COMMAND_KEY, command_shim.supported()),
                (self.start_menu_entry_checkbox, START_MENU_KEY, start_menu.supported())]

    def _connect_general_page(self) -> None:
        """React to immediate-apply selectors and dependent checkboxes."""
        self.ui_language_selector.currentIndexChanged.connect(
            lambda _index: self.change_language(self.ui_language_selector.currentData()))
        self.color_theme_selector.currentIndexChanged.connect(self._on_color_theme_changed)
        for checkbox, key, _supported in self._system_integration_switches():
            checkbox.toggled.connect(lambda checked, config_key=key: self._on_system_integration_toggled(config_key, checked))
        self.start_at_login_checkbox.toggled.connect(self._on_start_at_login_toggled)
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

    def _on_system_integration_toggled(self, key: str, checked: bool) -> None:
        """Persist the switch and have the command or the Start menu entry added or removed now."""
        self.config[key] = checked
        self.ctx.save_config_keys(key)
        self.system_integration_changed.emit()

    def refresh_start_at_login_switch(self) -> None:
        """Show whether WhisperTyper starts at login, as the registry says right now (the tray may have changed it)."""
        if autostart.supported():
            with QSignalBlocker(self.start_at_login_checkbox):
                self.start_at_login_checkbox.setChecked(autostart.is_enabled())

    def _on_start_at_login_toggled(self, checked: bool) -> None:
        """Turn the start at login on or off now, as the tray switch does."""
        switch_start_at_login(self.ctx, checked)
        self.refresh_start_at_login_switch()  # a failure leaves the registry, and so the switch, unchanged

    def populate_input_devices(self) -> None:
        """Fill the input-device dropdown with the system default + available microphones."""
        with QSignalBlocker(self.input_device_selector):
            self.input_device_selector.clear()
            self.input_device_selector.addItem(self.translator.tr("input_device_default"), "")
            for device in self._recording.selectable_input_devices():
                self.input_device_selector.addItem(device.get("label", device["name"]), device["name"])
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
        show_about(self, self.translator.tr)
