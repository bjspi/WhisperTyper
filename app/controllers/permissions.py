"""macOS privacy permissions: one-time information dialogs and the proactive startup prompts."""
from __future__ import annotations

import logging
import sys
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox, QWidget

from app.context import AppContext
from app.core.env import is_MACOS
from app.services.macos_permissions import PERMISSIONS_GUIDE_URL, request_accessibility, request_input_monitoring

#: Translation keys (title, text) of the information dialog shown once per permission.
_PERMISSION_DIALOGS = {
    "input_monitoring": ("macos_input_monitoring_title", "macos_input_monitoring_text"),
    "accessibility": ("macos_accessibility_title", "macos_accessibility_text"),
    "microphone": ("macos_microphone_title", "macos_microphone_text"),
}


class MacPermissions(QObject):
    """Explain missing macOS permissions once and trigger the system prompts. No-ops elsewhere."""

    # Warnings can originate on capture/listener threads; the dialog always opens on the GUI thread.
    _dialog_requested = pyqtSignal(str, str, str)

    def __init__(self, ctx: AppContext, dialog_parent: Callable[[], Optional[QWidget]]) -> None:
        """``dialog_parent`` returns the window the information dialogs belong to."""
        super().__init__()
        self._ctx = ctx
        self._dialog_parent = dialog_parent
        self._startup_requested = False
        self._hotkey_permissions_checked = False
        self._dialog_requested.connect(self._show_dialog)

    def warn(self, permission_type: str) -> None:
        """Show the information dialog for a missing permission once (thread-safe, non-blocking).

        Args:
            permission_type: 'input_monitoring', 'accessibility' or 'microphone'.
        """
        if not is_MACOS or permission_type not in _PERMISSION_DIALOGS:
            return
        config_key = f"macos_{permission_type}_info_shown"
        if self._ctx.config.get(config_key, False):
            return
        # Mark as shown immediately to prevent repeated dialogs
        self._ctx.config[config_key] = True
        self._ctx.save_config()
        title_key, text_key = _PERMISSION_DIALOGS[permission_type]
        self._dialog_requested.emit(self._ctx.tr(title_key), self._ctx.tr(text_key), PERMISSIONS_GUIDE_URL)

    def _request(self, permission_type: str, request: Callable[[], bool]) -> bool:
        """Run one permission request and point the user at the guide if it is missing."""
        granted = request()
        if not granted:
            logging.info("macOS %s permission is not granted yet.", permission_type)
            self.warn(permission_type)
        return granted

    @staticmethod
    def startup_prompts_wanted() -> bool:
        """The proactive permission flow runs on launch of the packaged macOS app only."""
        return is_MACOS and bool(getattr(sys, "frozen", False))

    def request_at_startup(self, probe_microphone: Callable[[], None]) -> None:
        """Proactively trigger the important macOS permission prompts on app launch."""
        if not self.startup_prompts_wanted() or self._startup_requested:
            return
        self._startup_requested = True
        self._request('input_monitoring', request_input_monitoring)
        self._request('accessibility', request_accessibility)
        try:
            # Opening the microphone once triggers the system prompt.
            probe_microphone()
        except Exception as e:
            logging.info(f"macOS microphone permission preflight did not complete: {e}")
            self.warn('microphone')
            return
        logging.info("macOS microphone permission preflight succeeded.")

    def ensure_hotkey_permissions(self) -> None:
        """Prompt for macOS hotkey permissions once per session, even for source-based runs."""
        if not is_MACOS or self._hotkey_permissions_checked:
            return
        self._hotkey_permissions_checked = True
        self._request('input_monitoring', request_input_monitoring)
        self._request('accessibility', request_accessibility)

    def _show_dialog(self, title: str, text: str, settings_url: str) -> None:
        """Show the permission information dialog (GUI thread)."""
        msg_box = QMessageBox(self._dialog_parent())
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.addButton(self._ctx.tr("ok_button"), QMessageBox.ButtonRole.AcceptRole)
        open_instructions_button = None
        if settings_url:
            open_instructions_button = msg_box.addButton(
                self._ctx.tr("macos_github_instructions_button"), QMessageBox.ButtonRole.ActionRole)
        msg_box.exec()
        if msg_box.clickedButton() == open_instructions_button:
            QDesktopServices.openUrl(QUrl(settings_url))
