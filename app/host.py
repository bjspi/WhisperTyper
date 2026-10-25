"""Host the tray application: Qt application, single-instance lock and event loop.

Started by ``whispertyper`` without a command (or ``whispertyper run``). Qt is imported here
only, so the other commands of the command line start fast and work without a display. Started
in a terminal, the tray app moves into a detached process, so the terminal stays free.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from app.bootstrap import (
    WINDOWED_ENV,
    acquire_single_instance_lock,
    configure_base_logging,
    enable_native_crash_log,
    install_qt_message_handler,
    mark_clean_exit,
    maybe_reexec_with_project_venv,
    report_missing_dependency,
)
from app.core.env import is_WINDOWS
from app.services.gitutil import project_root


def run_tray(restart: bool = False, show_settings: Optional[int] = None) -> int:
    """Run WhisperTyper in the system tray until it quits; returns the exit code.

    ``restart`` hard-stops a running instance first; ``show_settings`` opens the settings
    window on that tab after startup (a restart passes the open tab).
    """
    configure_base_logging()
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        from app.application import WhisperTyperApp
        from app.ui import tooltip_wrap
        from app.ui.fonts import install_emoji_font
    except ModuleNotFoundError as exc:
        if exc.name == "PyQt6":
            maybe_reexec_with_project_venv()
        else:
            report_missing_dependency(exc.name or str(exc))
        raise

    app = QApplication([sys.argv[0]])
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("WhisperTyper")
    install_emoji_font()
    tooltip_wrap.install(app)

    instance_lock, message = acquire_single_instance_lock(restart)
    if instance_lock is None:
        print(message, file=sys.stderr)
        QMessageBox.information(None, "WhisperTyper", message)
        return 0
    # Only the instance that keeps running records crashes and Qt's own warnings.
    install_qt_message_handler()
    enable_native_crash_log()

    # The instance lock and the application stay referenced while the event loop runs.
    application = WhisperTyperApp(show_settings=show_settings)
    exit_code = app.exec()
    mark_clean_exit()
    del application, instance_lock
    return exit_code


def started_from_terminal() -> bool:
    """Whether the tray start would occupy a terminal: an interactive console, not a windowless launch."""
    if getattr(sys, "frozen", False) or os.environ.get(WINDOWED_ENV):
        return False
    return sys.stdin is not None and sys.stdin.isatty()


def relaunch_detached(arguments: List[str]) -> None:
    """Start the tray app again, detached from this terminal (on Windows without a console window).

    The terminal stays usable and closing it does not end WhisperTyper.
    """
    script = os.path.join(project_root(), "run.py")
    interpreter = Path(sys.executable)
    if is_WINDOWS:
        windowless = interpreter.with_name("pythonw.exe")
        interpreter = windowless if windowless.is_file() else interpreter
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen([str(interpreter), script, *arguments], creationflags=flags, close_fds=True)
        return
    subprocess.Popen([str(interpreter), script, *arguments], start_new_session=True, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
