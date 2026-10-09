"""Windows-only system queries."""
from __future__ import annotations

import ctypes
import logging

from app.core.env import is_WINDOWS

_CONSOLE_WINDOW_CLASSES = frozenset({
    "ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS", "VirtualConsoleClass", "mintty",
})


def is_console_foreground_window() -> bool:
    """Whether the foreground window is a console/terminal host (where Ctrl+C would interrupt)."""
    if not is_WINDOWS:
        return False
    try:
        user32 = ctypes.WinDLL("user32")
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        class_name = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_name, len(class_name))
        return class_name.value in _CONSOLE_WINDOW_CLASSES
    except Exception as e:
        logging.debug(f"Failed to inspect foreground window class: {e}")
        return False
