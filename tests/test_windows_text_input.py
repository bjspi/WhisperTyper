"""Verify Windows Unicode input layout and events without injecting real input."""
from __future__ import annotations

import ctypes
import struct
from unittest.mock import Mock

import pytest

import app.services.windows_text_input as native


@pytest.fixture
def user32(monkeypatch):
    api = Mock()
    api.GetAsyncKeyState.return_value = 0
    api.SendInput.side_effect = lambda count, _events, _size: count
    monkeypatch.setattr(native, "is_WINDOWS", True)
    monkeypatch.setattr(native.ctypes, "WinDLL", Mock(return_value=api), raising=False)
    monkeypatch.setattr(native.ctypes, "set_last_error", Mock(), raising=False)
    monkeypatch.setattr(native.ctypes, "get_last_error", Mock(return_value=5), raising=False)
    return api


def test_native_layout_and_unicode_batch_preserve_emoji_and_line_breaks(user32):
    expected_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
    assert ctypes.sizeof(native._Input) == expected_size
    text = "Grüße 🦄\r\n中文\n\tfin"
    units = [value for (value,) in struct.iter_unpack("<H", "Grüße 🦄\r中文\r\tfin".encode("utf-16-le"))]
    assert native.send_unicode_text(text) == (2 * len(units), 2 * len(units))
    user32.SendInput.assert_called_once()
    count, events, size = user32.SendInput.call_args.args
    assert size == expected_size
    assert count == len(events)
    for index, unit in enumerate(units):
        down, up = events[2 * index], events[2 * index + 1]
        assert down.type == up.type == 1
        assert down.payload.ki.wVk == up.payload.ki.wVk == 0
        assert down.payload.ki.wScan == up.payload.ki.wScan == unit
        assert down.payload.ki.dwFlags == 4
        assert up.payload.ki.dwFlags == 6


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_zero_or_partial_acceptance_is_reported_without_retry(user32, accepted):
    user32.SendInput.side_effect = None
    user32.SendInput.return_value = accepted
    assert native.send_unicode_text("ab") == (accepted, 4)
    user32.SendInput.assert_called_once()


@pytest.mark.parametrize("key", [0x10, 0x11, 0x12, 0x5B, 0x5C])
def test_held_modifier_prevents_unicode_injection(user32, key):
    user32.GetAsyncKeyState.side_effect = lambda candidate: 0x8000 if candidate == key else 0
    with pytest.raises(OSError, match="Modifier"):
        native.send_unicode_text("text")
    user32.SendInput.assert_not_called()


def test_empty_or_invalid_text_never_injects(user32):
    assert native.send_unicode_text("") == (0, 0)
    with pytest.raises(ValueError):
        native.send_unicode_text("text\x00tail")
    user32.SendInput.assert_not_called()


def test_unavailable_off_windows(user32, monkeypatch):
    monkeypatch.setattr(native, "is_WINDOWS", False)
    with pytest.raises(OSError, match="only available on Windows"):
        native.send_unicode_text("text")
    user32.SendInput.assert_not_called()
