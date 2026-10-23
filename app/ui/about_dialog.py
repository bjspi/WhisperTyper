"""The About dialog: logo, version, tagline and a card of facts about this installation."""
from __future__ import annotations

import html
import platform
from typing import Callable, Optional

from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QWidget

from app import __version__
from app.core.constants import APP_DATA_DIR
from app.services.gitutil import find_git_root, project_root
from app.services.releases import REPOSITORY_URL as GITHUB_URL
from app.ui import message_dialog
from app.ui.tray_icons import app_icon


def show_about(parent: Optional[QWidget], tr: Callable[..., str]) -> None:
    """Show the About dialog; "Open data folder" and "GitHub" keep it open."""
    checkout = find_git_root()
    installation = tr("about_installation_checkout" if checkout else "about_installation_package")
    facts = [
        (tr("about_fact_version"), html.escape(__version__)),
        (tr("about_fact_installation"), f"{installation}<br>{html.escape(checkout or project_root())}"),
        (tr("about_fact_data"), html.escape(APP_DATA_DIR)),
        (tr("about_fact_runtime"),
         f"Python {platform.python_version()} · PyQt {PYQT_VERSION_STR} · Qt {QT_VERSION_STR}"),
        (tr("about_fact_project"), f'<a href="{GITHUB_URL}">{GITHUB_URL.removeprefix("https://")}</a> · MIT'),
    ]
    message_dialog.show_dialog(
        parent, f"WhisperTyper {__version__}", tr("about_tagline"),
        [(tr("dialog_close_button"), message_dialog.PRIMARY)],
        window_title=tr("about_dialog_title"), icon=app_icon(), facts=facts,
        actions=[(tr("about_open_data_folder"), _open_data_folder)])


def _open_data_folder() -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(APP_DATA_DIR))
