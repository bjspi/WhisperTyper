"""Optional Windows paste diagnostics: target window, held modifiers and clipboard access.

Every probe is a cheap Win32 query that never opens the clipboard, so enabling the
diagnostics does not delay or disturb the paste itself. Window titles and clipboard
contents are never logged.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Any, Optional

from app.core.env import is_WINDOWS

_MODIFIERS = (("shift", 0x10), ("ctrl", 0x11), ("alt", 0x12), ("lwin", 0x5B), ("rwin", 0x5C))
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class _GuiThreadInfo(ctypes.Structure):
    """GUITHREADINFO: yields the focused child control, e.g. Notepad++'s Scintilla editor."""

    _fields_ = [
        ("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


_api: Optional[tuple[Any, Any]] = None


def _win32() -> tuple[Any, Any]:
    """user32/kernel32 with typed signatures, bound once."""
    global _api
    if _api is None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, argtypes, restype in (
            ("GetForegroundWindow", [], wintypes.HWND),
            ("GetClassNameW", [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
            ("GetWindowThreadProcessId", [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
            ("GetGUIThreadInfo", [wintypes.DWORD, ctypes.POINTER(_GuiThreadInfo)], wintypes.BOOL),
            ("GetAsyncKeyState", [ctypes.c_int], ctypes.c_int16),
            ("GetClipboardSequenceNumber", [], wintypes.DWORD),
            ("GetOpenClipboardWindow", [], wintypes.HWND),
            ("GetClipboardOwner", [], wintypes.HWND),
        ):
            function = getattr(user32, name)
            function.argtypes, function.restype = argtypes, restype
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        _api = user32, kernel32
    return _api


def _class_name(hwnd: Any) -> str:
    """Window class, or "none" without a window."""
    if not hwnd:
        return "none"
    buffer = ctypes.create_unicode_buffer(256)
    return buffer.value if _win32()[0].GetClassNameW(hwnd, buffer, len(buffer)) else "?"


def _process_name(hwnd: Any) -> str:
    """Executable name of the window's process ("none" without a window, "?" if inaccessible)."""
    if not hwnd:
        return "none"
    user32, kernel32 = _win32()
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value == os.getpid():
        return "self"
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return f"pid{pid.value}"
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buffer))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return f"pid{pid.value}"
        return os.path.basename(buffer.value)
    finally:
        kernel32.CloseHandle(handle)


def clipboard_sequence() -> int:
    """Windows' clipboard change counter; 0 off Windows or on failure."""
    if not is_WINDOWS:
        return 0
    try:
        return int(_win32()[0].GetClipboardSequenceNumber())
    except Exception:
        return 0


def probe() -> str:
    """Key=value snapshot of focus, held modifiers and clipboard access for one log line."""
    if not is_WINDOWS:
        return "unsupported_platform"
    try:
        user32 = _win32()[0]
        foreground = user32.GetForegroundWindow()
        focus = None
        if foreground:
            thread_id = user32.GetWindowThreadProcessId(foreground, None)
            info = _GuiThreadInfo(cbSize=ctypes.sizeof(_GuiThreadInfo))
            if user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
                focus = info.hwndFocus
        held = [name for name, key in _MODIFIERS if user32.GetAsyncKeyState(key) & 0x8000]
        open_window = user32.GetOpenClipboardWindow()
        return (f"fg_process={_process_name(foreground)} fg_class={_class_name(foreground)} "
                f"focus_class={_class_name(focus)} held_modifiers={','.join(held) or 'none'} "
                f"clip_open_by={_process_name(open_window) if open_window else 'none'} "
                f"clip_owner={_process_name(user32.GetClipboardOwner())}")
    except Exception as error:
        return f"probe_error={type(error).__name__}"
