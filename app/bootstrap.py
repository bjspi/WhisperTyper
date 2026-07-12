"""Process bootstrap for WhisperTyper.

Base logging setup, crash-log redirect (for windowed/``pythonw`` launches), project-venv
re-exec, and the single-instance lock. These are kept out of the app class *and* out of
``run.py`` so the entry point stays a thin shim. Only the lock helper needs PyQt6, and it
imports it lazily, so the stream redirect + venv re-exec still work when PyQt6 isn't
importable yet.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Optional, Tuple

APP_INSTANCE_LOCK_KEY = "WhisperTyper.instance.lock"
RESTART_WAIT_TIMEOUT_S = 5.0

# Project root = the directory above this package (…/app/bootstrap.py -> …/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def configure_base_logging() -> None:
    """Install the root console handler before any app module logs at import time.

    The level starts at DEBUG so early import-time diagnostics are captured; once the
    config is loaded, ``apply_logging_configuration`` re-applies the user's chosen level
    and adds the rotating file handler.
    """
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def redirect_std_streams_if_windowed() -> None:
    """
    When started via ``pythonw`` (no console) or as a frozen/windowed binary, ``sys.stdout``
    and ``sys.stderr`` are unavailable, so crashes and tracebacks vanish silently.

    This redirects both streams to a file inside the app data directory and installs an
    ``excepthook`` so uncaught exceptions are always recorded, even before the Qt app and
    its logging handlers are initialized.
    """
    exe_name = os.path.basename(sys.executable or "").lower()
    is_windowed = (
        sys.stderr is None
        or sys.stdout is None
        or exe_name.startswith("pythonw")
        or bool(getattr(sys, "frozen", False))
    )
    if not is_windowed:
        return

    # Mark windowed mode so the app can keep the stderr crash-log lean (warnings + crashes
    # only, not every DEBUG line — otherwise this file grows unbounded during a long session).
    os.environ["WHISPERTYPER_WINDOWED"] = "1"

    try:
        app_data_dir = os.path.join(os.path.expanduser("~"), ".WhisperTyper")
        os.makedirs(app_data_dir, exist_ok=True)
        stderr_path = os.path.join(app_data_dir, "WhisperTyper.stderr.log")
        # Truncate per launch (single-instance app) so the file stays bounded but keeps
        # the full current session including any crash traceback.
        stream = open(stderr_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = stream
        sys.stderr = stream

        def _excepthook(exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
            try:
                stream.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
                stream.flush()
            except Exception:
                pass
            # Also surface through logging so it lands in the rotating log file if configured.
            try:
                import logging
                logging.getLogger().critical(
                    "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
                )
            except Exception:
                pass

        sys.excepthook = _excepthook
    except Exception:
        # Never let logging setup prevent the app from starting.
        pass


def find_project_venv_python() -> Optional[str]:
    """Return the bundled virtualenv interpreter next to the project root, if one exists."""
    if sys.platform.startswith("win"):
        candidates = [
            os.path.join(_PROJECT_ROOT, "venv", "Scripts", "python.exe"),
            os.path.join(_PROJECT_ROOT, ".venv", "Scripts", "python.exe"),
        ]
    else:
        candidates = [
            os.path.join(_PROJECT_ROOT, "venv", "bin", "python"),
            os.path.join(_PROJECT_ROOT, ".venv", "bin", "python"),
        ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def maybe_reexec_with_project_venv() -> None:
    """Restart the entry script with the project venv if the current interpreter lacks deps."""
    venv_python = find_project_venv_python()
    current_python = os.path.abspath(sys.executable) if sys.executable else ""
    if not venv_python or os.path.abspath(venv_python) == current_python:
        return

    entry_script = os.path.abspath(sys.argv[0])
    os.execv(venv_python, [venv_python, entry_script, *sys.argv[1:]])


def _hard_stop_process(pid: int) -> bool:
    """Forcefully stop a process by PID."""
    try:
        if sys.platform.startswith("win"):
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0

        os.kill(pid, signal.SIGKILL)
        return True
    except Exception:
        return False


def _restart_existing_instance(instance_lock: Any) -> bool:
    """Stop the currently running instance and wait until the lock becomes available."""
    locked, pid, _hostname, _appname = instance_lock.getLockInfo()
    if not locked or not pid:
        return False

    if not _hard_stop_process(pid):
        return False

    deadline = time.time() + RESTART_WAIT_TIMEOUT_S
    while time.time() < deadline:
        if instance_lock.tryLock(100):
            return True
        time.sleep(0.1)

    instance_lock.removeStaleLockFile()
    return instance_lock.tryLock(100)


def acquire_single_instance_lock(restart: bool) -> Tuple[Optional[Any], Optional[str]]:
    """Acquire the single-instance lock (PyQt6 imported lazily).

    Returns ``(lock, None)`` when acquired — keep the returned lock referenced for the
    application's lifetime — or ``(None, message)`` when another instance already holds it
    and no successful restart happened.
    """
    from PyQt6.QtCore import QDir, QLockFile

    lock_path = QDir.temp().absoluteFilePath(APP_INSTANCE_LOCK_KEY)
    instance_lock = QLockFile(lock_path)
    instance_lock.setStaleLockTime(0)

    if instance_lock.tryLock(100):
        return instance_lock, None

    if restart and _restart_existing_instance(instance_lock):
        return instance_lock, None

    message = (
        "WhisperTyper is already running."
        if not restart
        else "Failed to restart the running WhisperTyper instance."
    )
    if not restart:
        message += " Use -r to restart it with the current code."
    return None, message
