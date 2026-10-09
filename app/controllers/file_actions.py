"""Open the files and links offered by both the tray menu and the settings window."""
from __future__ import annotations

import logging
import os

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

from app.context import AppContext
from app.core.constants import CONFIG_FILE, LOG_FILE_PATH
from app.platform.system import open_with_default_app
from app.ui.durations import BALLOON_ERROR_MS, BALLOON_SHORT_MS

GITHUB_URL = "https://github.com/bjspi/WhisperTyper"


class FileActions:
    """Config file, log file, latest recording and project page; failures become balloons."""

    def __init__(self, ctx: AppContext) -> None:
        """Use the shared notifier and announce availability changes via ``ctx.files_changed``."""
        self._ctx = ctx

    @staticmethod
    def log_file_exists() -> bool:
        """Whether the rotating log file has been written."""
        return os.path.isfile(LOG_FILE_PATH)

    def open_config_file(self) -> None:
        """Open config.json in the default editor."""
        if os.path.exists(CONFIG_FILE):
            QDesktopServices.openUrl(QUrl.fromLocalFile(CONFIG_FILE))
        else:
            self._ctx.notifier.show(self._ctx.tr("config_file_not_found"), BALLOON_ERROR_MS)

    def open_log_file(self) -> None:
        """Open the log file with the system's default application."""
        if not self.log_file_exists():
            self._ctx.notifier.show(self._ctx.tr("log_file_not_exist_message"), BALLOON_SHORT_MS)
            self._ctx.files_changed.emit()
            return
        try:
            open_with_default_app(LOG_FILE_PATH)
        except Exception as e:
            logging.error(f"Failed to open log file: {e}")
            self._ctx.notifier.show(self._ctx.tr("log_file_open_fail_message", error=e), BALLOON_ERROR_MS)

    def play_latest_recording(self) -> None:
        """Open the latest recording in the system's default media player."""
        latest = self._ctx.recordings.latest()
        if not latest:
            self._ctx.notifier.show(self._ctx.tr("no_recording_found_message"), BALLOON_SHORT_MS)
            return
        try:
            open_with_default_app(latest)
        except Exception as e:
            self._ctx.notifier.show(self._ctx.tr("could_not_play_file_message", error=e), BALLOON_SHORT_MS)

    @staticmethod
    def open_github_link() -> None:
        """Open the project's GitHub repository."""
        QDesktopServices.openUrl(QUrl(GITHUB_URL))
