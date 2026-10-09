"""MacMixin — macOS permission prompts and their information dialogs."""
from __future__ import annotations

import logging
import sys
from typing import Callable

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox

from app.audio import macos_recorder
from app.core.env import is_MACOS
from app.services.macos_permissions import PERMISSIONS_GUIDE_URL, request_accessibility, request_input_monitoring

#: Translation keys (title, text) of the information dialog shown once per permission.
_PERMISSION_DIALOGS = {
    "input_monitoring": ("macos_input_monitoring_title", "macos_input_monitoring_text"),
    "accessibility": ("macos_accessibility_title", "macos_accessibility_text"),
    "microphone": ("macos_microphone_title", "macos_microphone_text"),
}


class MacMixin:
    """macOS permission prompts and their information dialogs."""

    def _check_and_warn_macos_permissions(self, permission_type: str) -> None:
        """Show the information dialog for a missing permission once (thread-safe, non-blocking).

        Args:
            permission_type (str): 'input_monitoring', 'accessibility' or 'microphone'.
        """
        if not is_MACOS or permission_type not in _PERMISSION_DIALOGS:
            return
        config_key = f"macos_{permission_type}_info_shown"
        if self.config.get(config_key, False):
            return
        # Mark as shown immediately to prevent repeated dialogs
        self.config[config_key] = True
        self.save_config()
        title_key, text_key = _PERMISSION_DIALOGS[permission_type]
        self.show_permission_dialog_signal.emit(
            self.translator.tr(title_key), self.translator.tr(text_key), PERMISSIONS_GUIDE_URL,
        )

    def _request_permission(self, permission_type: str, request: Callable[[], bool]) -> bool:
        """Run one permission request and point the user at the guide if it is missing."""
        granted = request()
        if not granted:
            logging.info("macOS %s permission is not granted yet.", permission_type)
            self._check_and_warn_macos_permissions(permission_type)
        return granted

    def _should_request_macos_startup_permissions(self) -> bool:
        """Return whether the proactive macOS permission flow should run on launch."""
        return is_MACOS and bool(getattr(sys, "frozen", False))

    def _request_macos_microphone_permission(self) -> bool:
        """Trigger the macOS microphone prompt once on app startup."""
        try:
            # Opening the microphone once triggers the system prompt.
            if macos_recorder.available():
                self._mac_recorder.start(self.recordings.new_path())
                self._mac_recorder.stop(discard=True)
            else:
                self._microphone.open()
                self._microphone.close()
        except Exception as e:
            logging.info(f"macOS microphone permission preflight did not complete: {e}")
            self._check_and_warn_macos_permissions('microphone')
            return False
        logging.info("macOS microphone permission preflight succeeded.")
        return True

    def _request_macos_startup_permissions(self) -> None:
        """Proactively trigger the important macOS permission prompts on app launch."""
        if not self._should_request_macos_startup_permissions() or self._macos_startup_permissions_requested:
            return
        self._macos_startup_permissions_requested = True
        self._request_permission('input_monitoring', request_input_monitoring)
        self._request_permission('accessibility', request_accessibility)
        self._request_macos_microphone_permission()

    def _ensure_macos_hotkey_permissions(self) -> None:
        """Prompt for macOS hotkey permissions once per session, even for source-based runs."""
        if not is_MACOS or self._macos_hotkey_permissions_checked:
            return
        self._macos_hotkey_permissions_checked = True
        self._request_permission('input_monitoring', request_input_monitoring)
        self._request_permission('accessibility', request_accessibility)

    def _show_permission_dialog_slot(self, title: str, text: str, settings_url: str) -> None:
        """Show the macOS permission information dialog (GUI thread).

        Args:
            title (str): The dialog title.
            text (str): The dialog message.
            settings_url (str): The permission guide opened by the instructions button.
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.addButton(self.translator.tr("ok_button"), QMessageBox.ButtonRole.AcceptRole)
        open_instructions_button = None
        if settings_url:
            open_instructions_button = msg_box.addButton(
                self.translator.tr("macos_github_instructions_button"), QMessageBox.ButtonRole.ActionRole)
        msg_box.exec()
        if msg_box.clickedButton() == open_instructions_button:
            QDesktopServices.openUrl(QUrl(settings_url))
