"""The settings window: composes the pages, loads/saves the form and announces changes."""
from __future__ import annotations

import logging
import time
from typing import Callable

from PyQt6 import uic
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import QApplication, QMenuBar, QMessageBox, QSizePolicy, QStyle

from app.context import AppContext
from app.controllers.file_actions import FileActions
from app.controllers.hotkeys import HotkeyController
from app.controllers.recording import RecordingController
from app.core.constants import WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH
from app.core.env import is_MACOS
from app.core.hotkeys import is_clipboard_shortcut, normalize_hotkey_string, pretty_hotkey
from app.core.paths import resource_path
from app.core.replacements import ReplacementError, Replacements
from app.ui.api_keys import ApiKeysTab
from app.ui.connection_tester import ConnectionTester
from app.ui.replacements import ReplacementsTab
from app.ui.settings.api_page import ApiPage
from app.ui.settings.bindings import load_bindings, save_bindings
from app.ui.settings.general_page import GeneralPage
from app.ui.settings.recording_page import RecordingPage
from app.ui.settings.texts import apply_texts
from app.ui.settings.theme_page import ThemePage
from app.ui.settings.transcription_page import TranscriptionPage
from app.ui.settings.widgets import SliderWheelToScrollArea, fit_button_to_captions
from app.ui.transformations_tab import TransformationsEditor
from app.ui.tray_icons import app_icon

#: Default window width on first launch (the height default lives in the config).
_DEFAULT_WINDOW_WIDTH = 760


class SettingsWindow(ApiPage, TranscriptionPage, RecordingPage, GeneralPage, ThemePage):
    """Settings form; the runtime components react to its signals instead of being driven from here."""

    #: The form was validated and written to the config (and disk).
    saved = pyqtSignal()
    #: Saved hotkeys differ from the running listeners' (not emitted on macOS, which asks for a restart).
    hotkeys_changed = pyqtSignal()
    #: The UI language changed; other translated surfaces (tray menu) rebuild.
    language_changed = pyqtSignal()

    def __init__(self, ctx: AppContext, *, recording: RecordingController, hotkeys: HotkeyController,
                 files: FileActions, quit_app: Callable[[], None]) -> None:
        """Build the window from main_window.ui and load the saved configuration into it."""
        super().__init__()
        self.ctx = ctx
        self.config = ctx.config
        self.translator = ctx.translator
        self._recording = recording
        self._hotkeys = hotkeys
        self._files = files
        self._quit_app = quit_app

        uic.loadUi(resource_path("resources", "main_window.ui"), self)
        self._build_menu_bar()

        # Pages and controls the .ui file does not contain.
        self._api_keys_tab = ApiKeysTab(self.config["api_key_profiles"], self.translator, self)
        self._api_keys_tab.groq_rotation.setChecked(self.config["groq_key_rotation"])
        self.tabs.insertTab(self.tabs.indexOf(self.general_tab), self._api_keys_tab, "")
        self._replacements_tab = ReplacementsTab(self.config["replacements_rules"], self.config["replacements_enabled"],
                                                 self.translator, self)
        self.tabs.insertTab(self.tabs.indexOf(self.general_tab), self._replacements_tab, "")
        self._init_transcription_page()
        self._build_recording_controls()
        self._build_general_page()
        self._init_transformations_page()

        load_bindings(self, self.config)
        self.hotkey_display.setText(self.config["hotkey"])
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
        self.set_hotkey_button.clicked.connect(
            lambda: self._hotkeys.start_capture(self.hotkey_display, self.set_hotkey_button))
        self.set_pr_hotkey_button.clicked.connect(
            lambda: self._hotkeys.start_capture(self.pr_hotkey_display, self.set_pr_hotkey_button))
        self.play_g_button.clicked.connect(self._files.play_latest_recording)
        self.liveprompt_help_button.setFixedSize(22, 22)  # Rendered as a round "?" badge by the theme.
        self.liveprompt_help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.liveprompt_help_button.clicked.connect(self.show_liveprompt_help)
        if is_MACOS:
            # Trackpad scrolling over a slider should scroll the page, not change the temperature.
            SliderWheelToScrollArea(self.transcription_temp_slider, self.transcription_scroll_area)
            SliderWheelToScrollArea(self.rephrasing_temp_slider, self.rephrasing_scroll_area)
        self._build_save_row()
        self.aac_bitrates_ready.connect(self._apply_aac_bitrates)
        ctx.files_changed.connect(self.refresh_actions)

        icon = app_icon()
        style = self.style()
        if icon.isNull() and style is not None:
            icon = style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.setWindowIcon(icon)
        self.apply_theme()
        self.retranslate_ui()
        self.refresh_actions()

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
        file_menu = self.menu_bar.addMenu("")
        help_menu = self.menu_bar.addMenu("")
        assert file_menu is not None and help_menu is not None
        self.file_menu, self.help_menu = file_menu, help_menu
        for menu, attr, handler in (
            (file_menu, "open_config_action", self._files.open_config_file),
            (file_menu, "open_log_file_action", self._files.open_log_file),
            (file_menu, "play_last_recording_action", self._files.play_latest_recording),
            (file_menu, None, None),
            (file_menu, "exit_action", self._quit_app),
            (help_menu, "about_action", self.show_about_dialog),
            (help_menu, "github_action", self._files.open_github_link),
        ):
            if attr is None or handler is None:
                menu.addSeparator()
                continue
            action = QAction("", self)
            action.triggered.connect(lambda _checked=False, handler=handler: handler())
            menu.addAction(action)
            setattr(self, attr, action)

    def _build_save_row(self) -> None:
        """Footer row: the Prompts tab's add/remove buttons share the line with a compact Save button."""
        self.save_button.clicked.connect(self.save_settings)
        self.save_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.save_button.setMinimumWidth(150)
        self.save_button.setMaximumWidth(200)
        self.tabs.currentChanged.connect(self._update_footer_buttons)
        self._update_footer_buttons()

    def _update_footer_buttons(self, *_args: object) -> None:
        """The add/remove buttons act on the prompt list, so they show only on the Prompts tab."""
        on_prompts = self.tabs.currentWidget() is self.post_rephrasing_tab
        self.post_rp_add_btn.setVisible(on_prompts)
        self.post_rp_remove_btn.setVisible(on_prompts)

    def _init_transformations_page(self) -> None:
        """Prompt editor on the Prompts tab; edits are mirrored into the live config."""
        # Description texts take minimal height; the editor splitter takes the rest.
        for label in (self.transformations_tab_description_label, self.transformations_unavailable_label,
                      self.transformations_info_label):
            label.setSizePolicy(label.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Maximum)
        self.splitter.setSizePolicy(self.splitter.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        self._transformations = TransformationsEditor(
            self.config.get("post_rephrasing_entries", []), self.translator, splitter=self.splitter,
            list_placeholder=self.post_rp_list_placeholder, caption_edit=self.post_rp_caption_edit,
            text_edit=self.post_rp_text_edit, show_during_recording=self.post_rp_show_during_recording_checkbox,
            auto_apply=self.post_rp_auto_apply_checkbox, add_button=self.post_rp_add_btn, remove_button=self.post_rp_remove_btn,
        )
        self.post_rp_list = self._transformations.list
        self._transformations.changed.connect(self._sync_transformations_to_config)

    def _sync_transformations_to_config(self) -> None:
        """Mirror the edited templates into the live config, which the palettes read."""
        self.config["post_rephrasing_entries"] = self._transformations.entries()
        self._refresh_api_state()

    def refresh_actions(self) -> None:
        """Enable the file actions only when their files exist."""
        self.play_last_recording_action.setEnabled(self.ctx.recordings.exists())
        self.open_log_file_action.setEnabled(self._files.log_file_exists())

    # --- Save ----------------------------------------------------------------------------
    def save_settings(self) -> None:
        """Validate and store the form; ``saved`` lets the runtime apply it."""
        raw_rules = self._replacements_tab.editor.toPlainText()
        try:
            Replacements(raw_rules)
        except ReplacementError as error:
            QMessageBox.warning(self, self.translator.tr("tab_replacements"),
                                self.translator.tr("replacements_invalid", line=error.line))
            self.tabs.setCurrentWidget(self._replacements_tab)
            self._replacements_tab.editor.setFocus()
            return
        if not self._validate_hotkeys():
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
        save_bindings(self, self.config)
        self._save_recording_controls()
        self._sync_transformations_to_config()
        hotkeys_changed = self._apply_pending_hotkeys()

        self.ctx.save_config()
        self.update_brand_header()
        if hotkeys_changed:
            self.hotkeys_changed.emit()
        self.saved.emit()

    def _validate_hotkeys(self) -> bool:
        """Reject Select all/Copy/Paste as a global hotkey before anything is saved."""
        for field in (self.hotkey_display, self.pr_hotkey_display):
            if not is_clipboard_shortcut(field.text()):
                continue
            QMessageBox.warning(self, self.translator.tr("hotkey_clipboard_conflict_title"),
                                self.translator.tr("hotkey_clipboard_conflict_text",
                                                   hotkey=pretty_hotkey(normalize_hotkey_string(field.text()))))
            for index in range(self.tabs.count()):
                tab = self.tabs.widget(index)
                if tab is not None and tab.isAncestorOf(field):
                    self.tabs.setCurrentWidget(tab)
            field.setFocus()
            return False
        return True

    def _apply_pending_hotkeys(self) -> bool:
        """Adopt edited hotkeys; True if the running listeners must restart."""
        pending = normalize_hotkey_string(self.hotkey_display.text()) or self.hotkey_display.text().strip()
        pending_pr = normalize_hotkey_string(self.pr_hotkey_display.text()) or self.pr_hotkey_display.text().strip()
        self.hotkey_display.setText(pending)
        self.pr_hotkey_display.setText(pending_pr)
        if pending == self.config["hotkey"] and pending_pr == self.config["post_rephrase_hotkey"]:
            return False
        self.config["hotkey"] = pending
        self.config["post_rephrase_hotkey"] = pending_pr
        if is_MACOS:
            # Restarting the listener at runtime conflicts with the macOS accessibility permissions.
            QMessageBox.information(self, self.translator.tr("macos_hotkey_restart_title"),
                                    self.translator.tr("macos_hotkey_restart_text"))
            return False
        return True

    # --- Translation and window lifecycle -----------------------------------------------
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
        # Test buttons show "testing…" while running; a fixed width keeps the slider/proxy field still.
        for button in (self.test_internet_button, self.test_transcription_api_button, self.test_rephrasing_api_button):
            fit_button_to_captions(button, tr("api_test_testing_button"))
        self._update_recording_format_controls()
        self._refresh_ffmpeg_status()
        self._update_prompt_token_counter()
        self.update_brand_header()
        self.language_changed.emit()

    def show_window(self) -> None:
        """Bring the window reliably to the foreground and log its geometry."""
        t0 = time.perf_counter()
        try:
            if self.isMinimized():
                self.showNormal()
            else:
                self.show()
            self.raise_()
            self.activateWindow()
            t_shown = time.perf_counter()
            # show() returns before the first paint happens; flush pending events so the timing
            # below reflects the real time until the window is actually rendered.
            QApplication.processEvents()
            t_painted = time.perf_counter()
            logging.info(
                "show_settings_window timing: show/raise/activate=%.0fms, first_paint_flush=%.0fms, total=%.0fms",
                (t_shown - t0) * 1000, (t_painted - t_shown) * 1000, (t_painted - t0) * 1000,
            )

            geo = self.geometry()
            frame = self.frameGeometry()
            screen = QApplication.screenAt(frame.center()) or QApplication.primaryScreen()
            screen_info = ""
            if screen:
                sg = screen.availableGeometry()
                screen_info = (f", screen='{screen.name()}' "
                               f"available=({sg.x()},{sg.y()},{sg.width()}x{sg.height()})")
            logging.info(
                "Settings window shown: "
                f"visible={self.isVisible()}, active={self.isActiveWindow()}, minimized={self.isMinimized()}, "
                f"geometry=({geo.x()},{geo.y()},{geo.width()}x{geo.height()}), "
                f"frame=({frame.x()},{frame.y()},{frame.width()}x{frame.height()})"
                f"{screen_info}"
            )
        except Exception as e:
            logging.error(f"Failed to show settings window: {e}")

        # Refresh the mic list AFTER the window is visible, on the next event-loop tick, so a slow
        # PyAudio enumeration never blocks the window from appearing.
        QTimer.singleShot(0, self.populate_input_devices)

    def show_transformations(self) -> None:
        """Open the window directly on the transformations templates tab."""
        self.tabs.setCurrentWidget(self.post_rephrasing_tab)
        self.show_window()

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802 - Qt API
        """Hide instead of quitting; remember the size and prune old recordings."""
        # An unfinished hotkey capture must restore the global listeners, or every hotkey stays dead.
        self._hotkeys.cancel_capture()
        try:
            self.config["window_width"] = self.width()
            self.config["window_height"] = self.height()
            self.ctx.save_config()
        except Exception as e:
            logging.debug(f"Could not persist window size on close: {e}")
        if event is not None:
            event.ignore()
        self.hide()
        self.ctx.recordings.keep_only_latest()
