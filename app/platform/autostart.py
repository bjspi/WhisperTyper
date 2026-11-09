"""Start WhisperTyper at login: the ``WhisperTyper`` value of the user's Run key (Windows only).

The registry value is the only state: the tray switch reads it whenever the menu opens, so an
entry removed elsewhere (Task Manager's startup page, a cleanup tool) is shown as off. An enabled
entry is rewritten at every start, so it follows a moved environment or a new interpreter. macOS
would need an app bundle (a LaunchAgent pointing into a checkout is fragile) and gets no switch.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Optional, Sequence

#: Name of the value in the Run key.
VALUE_NAME = "WhisperTyper"

if sys.platform == "win32":
    import winreg

    _RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"

    def _read_run_value() -> Optional[str]:
        """The registered command line, or None without an entry."""
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
                value, _kind = winreg.QueryValueEx(key, VALUE_NAME)
        except OSError:
            return None
        return str(value)

    def _write_run_value(command_line: Optional[str]) -> None:
        """Register ``command_line``, or remove the entry for None."""
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            if command_line is not None:
                winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command_line)
            elif _read_run_value() is not None:
                winreg.DeleteValue(key, VALUE_NAME)

else:

    def _read_run_value() -> Optional[str]:
        """There is no Run key outside Windows."""
        return None

    def _write_run_value(command_line: Optional[str]) -> None:
        """There is no Run key outside Windows."""
        del command_line
        raise OSError("starting at login is only supported on Windows")


def supported() -> bool:
    """Whether this platform gets the switch (Windows only)."""
    return sys.platform == "win32"


def is_enabled() -> bool:
    """Whether WhisperTyper starts at login."""
    return _read_run_value() is not None


def enable(command: Sequence[str]) -> None:
    """Start ``command`` at login.

    Raises:
        OSError: The entry could not be written.
    """
    _write_run_value(subprocess.list2cmdline(list(command)))


def disable() -> None:
    """No longer start at login.

    Raises:
        OSError: The entry could not be removed.
    """
    _write_run_value(None)


def refresh(command: Sequence[str]) -> bool:
    """Point an enabled entry at ``command``; returns whether it was rewritten.

    Raises:
        OSError: The entry could not be written.
    """
    current = _read_run_value()
    command_line = subprocess.list2cmdline(list(command))
    if current is None or current == command_line:
        return False
    _write_run_value(command_line)
    return True
