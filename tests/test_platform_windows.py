"""Windows user32 access in app/platform/windows.py (the library is mocked; runs on Windows only)."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.core.env import is_WINDOWS
from app.platform import windows

pytestmark = pytest.mark.skipif(not is_WINDOWS, reason="ctypes.wintypes signatures are Windows-only")


@pytest.fixture
def user32(monkeypatch):
    api = Mock()
    api.RegisterHotKey.return_value = 1
    monkeypatch.setattr(windows, "_user32", None)
    monkeypatch.setattr(windows.ctypes, "WinDLL", Mock(return_value=api))
    return api


def test_user32_is_loaded_once_with_declared_signatures(user32):
    assert windows.register_window_hotkey(0x1234, 1, 0x78) is True
    windows.unregister_window_hotkey(0x1234, 1)
    assert windows.ctypes.WinDLL.call_count == 1
    user32.RegisterHotKey.assert_called_once()
    user32.UnregisterHotKey.assert_called_once()
    for name in ("GetForegroundWindow", "GetClassNameW", "RegisterHotKey", "UnregisterHotKey"):
        function = getattr(user32, name)
        assert isinstance(function.argtypes, list) and function.restype is not None


def test_console_window_is_recognised_by_class_name(user32):
    user32.GetForegroundWindow.return_value = 42

    def class_name(_hwnd, buffer, _size):
        buffer.value = "ConsoleWindowClass"
        return len(buffer.value)

    user32.GetClassNameW.side_effect = class_name
    assert windows.is_console_foreground_window() is True
    user32.GetForegroundWindow.return_value = 0
    assert windows.is_console_foreground_window() is False
