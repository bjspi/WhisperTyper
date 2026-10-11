"""Windows-only system queries."""
from __future__ import annotations

import ctypes
import logging
from typing import Any, Optional

from app.core.env import is_WINDOWS
from app.core.win32 import MOD_NOREPEAT, WM_HOTKEY

if is_WINDOWS:
    from ctypes import wintypes

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


def register_window_hotkey(hwnd: int, hotkey_id: int, vk: int) -> bool:
    """Send WM_HOTKEY to ``hwnd`` for the unmodified key ``vk``; False if another program owns the key."""
    if not is_WINDOWS:
        return False
    try:
        return bool(ctypes.WinDLL("user32").RegisterHotKey(wintypes.HWND(hwnd), hotkey_id, MOD_NOREPEAT, vk))
    except Exception as e:
        logging.debug(f"RegisterHotKey failed for vk={vk:#x}: {e}")
        return False


def unregister_window_hotkey(hwnd: int, hotkey_id: int) -> None:
    """Release a hotkey registered with :func:`register_window_hotkey`."""
    if not is_WINDOWS:
        return
    try:
        ctypes.WinDLL("user32").UnregisterHotKey(wintypes.HWND(hwnd), hotkey_id)
    except Exception as e:
        logging.debug(f"UnregisterHotKey failed for id={hotkey_id}: {e}")


def hotkey_id_of(event_type: Any, message: Any) -> Optional[int]:
    """The hotkey id of a native WM_HOTKEY message delivered to a Qt window, else None."""
    if not is_WINDOWS or bytes(event_type) != b"windows_generic_MSG" or not message:
        return None
    msg = wintypes.MSG.from_address(int(message))
    return int(msg.wParam) if msg.message == WM_HOTKEY else None

