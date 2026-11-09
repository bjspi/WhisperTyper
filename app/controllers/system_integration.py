"""How WhisperTyper sits in the system: the ``whispertyper`` terminal command and the Start menu entry.

Both follow their settings (on by default): applied at every start, which also installs them
after a fresh install or a moved environment, and again when a switch is toggled. An enabled
start at login (the tray switch; the registry is its only state) is pointed at this installation too. The work runs
on a background thread (the Start menu link is written through PowerShell); the icon is rendered
on the GUI thread first, because it needs Qt.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from PyQt6.QtCore import QBuffer, QIODevice, Qt
from PyQt6.QtGui import QImage

from app.context import AppContext
from app.core.constants import APP_DATA_DIR
from app.core.env import is_WINDOWS
from app.core.icon_file import IconImage, ico_file
from app.core.paths import resource_path
from app.platform import autostart, command_shim, start_menu
from app.services.gitutil import project_root
from app.ui.durations import BALLOON_WARNING_MS

#: Set for a run with a throwaway home folder (local_test/fresh_start.py) and inherited by its child
#: processes: it must neither add that folder to the real user PATH nor rewrite the Start menu entry
#: or the start at login.
SKIP_ENV = "WHISPERTYPER_SKIP_SYSTEM_INTEGRATION"
#: Config switches (see DEFAULT_CONFIG).
COMMAND_KEY = "command_on_path"
START_MENU_KEY = "start_menu_entry"
#: Pixel sizes of the icon in the launchers and the ``.ico`` (Windows picks the closest one).
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
_ICON_SOURCE = ("resources", "app_icon.png")
_ICO_PATH = Path(APP_DATA_DIR) / "whispertyper.ico"


@dataclass(frozen=True)
class _Wanted:
    """The two switches, snapshotted on the GUI thread."""

    command: bool
    start_menu: bool


class SystemIntegration:
    """Keeps the terminal command and the Start menu entry as the settings say."""

    def __init__(self, ctx: AppContext) -> None:
        """Changes are serialized, so quick toggles cannot interleave their file operations."""
        self._ctx = ctx
        self._lock = threading.Lock()

    def apply(self, report: bool = False) -> None:
        """Install or remove both as configured, in the background.

        ``report`` shows failures in a balloon (a toggle); at startup they are only logged.
        """
        if os.environ.get(SKIP_ENV):
            logging.info("System integration skipped (%s).", SKIP_ENV)
            return
        wanted = _wanted_by(self._ctx.config)
        icon = render_icon() if wanted.command or wanted.start_menu else []
        threading.Thread(target=self._apply, args=(wanted, icon, report), name="system-integration",
                         daemon=True).start()

    def _apply(self, wanted: _Wanted, icon: Sequence[IconImage], report: bool) -> None:
        """Background thread: bring both in line with ``wanted``; failures are logged (and reported)."""
        with self._lock:
            problems = self._apply_both(wanted, icon)
        if problems and report:
            self._ctx.notifier.show(self._ctx.tr("system_integration_failed_message", error="; ".join(problems)),
                                    BALLOON_WARNING_MS)

    @staticmethod
    def _apply_both(wanted: _Wanted, icon: Sequence[IconImage]) -> List[str]:
        """Bring both in line with ``wanted``; returns the problems."""
        results = (SystemIntegration._apply_command(wanted.command, icon),
                   SystemIntegration._apply_start_menu(wanted.start_menu, icon),
                   SystemIntegration._refresh_autostart())
        return [problem for problem in results if problem]

    @staticmethod
    def _apply_command(wanted: bool, icon: Sequence[IconImage]) -> str:
        """Install or remove the terminal command; returns the problem, or ""."""
        if not command_shim.supported():
            return ""
        try:
            if not wanted:
                command_shim.uninstall()
                return ""
            if command_shim.install(icon):
                logging.info("Terminal command installed in %s", command_shim.bin_dir())
        except OSError as error:
            logging.warning("Terminal command not updated: %s", error)
            return str(error)
        shadow = command_shim.shadowed_by()
        if shadow is not None:
            logging.warning("Another whispertyper comes first on PATH: %s", shadow)
        return ""

    @staticmethod
    def _apply_start_menu(wanted: bool, icon: Sequence[IconImage]) -> str:
        """Create (rewrite) or remove the Start menu entry; returns the problem, or ""."""
        if not start_menu.supported():
            return ""
        try:
            if not wanted:
                start_menu.remove()
                return ""
            start_menu.create(tray_command(), _entry_icon(icon), Path(project_root()))
        except OSError as error:
            logging.warning("Start menu entry not updated: %s", error)
            return str(error)
        return ""

    @staticmethod
    def _refresh_autostart() -> str:
        """Point an enabled start at login at this installation; returns the problem, or ""."""
        if not autostart.supported():
            return ""
        try:
            if autostart.refresh(tray_command()):
                logging.info("Start at login now runs %s", tray_command())
        except OSError as error:
            logging.warning("Start at login not updated: %s", error)
            return str(error)
        return ""


def _wanted_by(config: Mapping[str, Any]) -> _Wanted:
    """What the settings ask for, as far as this installation supports it."""
    return _Wanted(command=bool(config.get(COMMAND_KEY, True)) and command_shim.supported(),
                   start_menu=bool(config.get(START_MENU_KEY, True)) and start_menu.supported())


def apply_now(config: Mapping[str, Any]) -> List[str]:
    """Install or remove both as configured and wait for it (installer, no app running); returns the problems."""
    wanted = _wanted_by(config)
    return SystemIntegration._apply_both(wanted, render_icon() if wanted.command or wanted.start_menu else [])


def tray_command() -> List[str]:
    """The command line that starts this installation's tray app without a console window."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    folder = Path(sys.executable).parent
    windowless = folder / "pythonw.exe"
    interpreter = windowless if is_WINDOWS and windowless.is_file() else Path(sys.executable)
    return [str(interpreter), os.path.join(project_root(), "run.py")]


def render_icon() -> List[IconImage]:
    """The app icon as PNG images in :data:`ICON_SIZES` (GUI thread: uses Qt)."""
    source = QImage(resource_path(*_ICON_SOURCE))
    if source.isNull():
        return []
    images = []
    for size in ICON_SIZES:
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        source.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation).save(buffer, "PNG")
        images.append(IconImage(size, buffer.data().data()))
    return images


def _entry_icon(icon: Sequence[IconImage]) -> Path:
    """The icon file of the entry: an ``.ico`` in the data folder (Windows), the bundled PNG elsewhere."""
    if not is_WINDOWS:
        return Path(resource_path(*_ICON_SOURCE))
    content = ico_file(icon)
    if not _ICO_PATH.is_file() or _ICO_PATH.read_bytes() != content:
        _ICO_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ICO_PATH.write_bytes(content)
    return _ICO_PATH
