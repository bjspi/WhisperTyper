"""Optional Unicode text injection through Windows SendInput, without clipboard access."""
from __future__ import annotations

import ctypes
import logging
import struct

from app.core.env import is_WINDOWS


class _KeyboardInput(ctypes.Structure):
    """KEYBDINPUT with pointer-sized extra information."""

    _fields_ = [
        ("wVk", ctypes.c_uint16), ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32), ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MouseInput(ctypes.Structure):
    """MOUSEINPUT fixes the size and alignment of the native INPUT union."""

    _fields_ = [
        ("dx", ctypes.c_int32), ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32), ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32), ("dwExtraInfo", ctypes.c_size_t),
    ]


class _InputPayload(ctypes.Union):
    """Native INPUT payload; its largest member is MOUSEINPUT."""

    _fields_ = [("ki", _KeyboardInput), ("mi", _MouseInput)]


class _Input(ctypes.Structure):
    """Native INPUT, including the union's padding on 64-bit Windows."""

    _fields_ = [("type", ctypes.c_uint32), ("payload", _InputPayload)]


def send_unicode_text(text: str) -> tuple[int, int]:
    """Return accepted/total events; acceptance does not confirm target insertion.

    All events are sent in one batch. Partial acceptance must never trigger a full
    clipboard retry, which could duplicate the already accepted prefix.
    """
    if not is_WINDOWS:
        raise OSError("SendInput is only available on Windows")
    if any(ord(char) < 32 and char not in "\r\n\t" for char in text):
        logging.warning("text_input mode=sendinput rejected reason=unsupported_control_character")
        raise ValueError("Unsupported control character")
    # VK_PACKET delivers Unicode characters, including CR for line breaks, without
    # synthesizing Return/Tab shortcuts that could submit a message or change focus.
    encoded = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r").encode("utf-16-le")
    count = len(encoded)  # Two keyboard events per UTF-16 unit.
    if not count:
        return 0, 0

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    key_state = user32.GetAsyncKeyState
    key_state.argtypes = [ctypes.c_int]
    key_state.restype = ctypes.c_int16
    if any(key_state(key) & 0x8000 for key in (0x10, 0x11, 0x12, 0x5B, 0x5C)):
        logging.warning("text_input mode=sendinput rejected reason=modifier_keys_held")
        raise OSError("Modifier key is held")

    events = (_Input * count)()
    for index, (unit,) in enumerate(struct.iter_unpack("<H", encoded)):
        for offset, flags in ((0, 0x0004), (1, 0x0004 | 0x0002)):
            events[2 * index + offset].type = 1
            events[2 * index + offset].payload.ki = _KeyboardInput(0, unit, flags, 0, 0)
    send_input = user32.SendInput
    send_input.argtypes = [ctypes.c_uint, ctypes.POINTER(_Input), ctypes.c_int]
    send_input.restype = ctypes.c_uint
    ctypes.set_last_error(0)
    accepted = send_input(count, events, ctypes.sizeof(_Input))
    if accepted != count:
        logging.warning("text_input mode=sendinput native_error=%s accepted_events=%s total_events=%s",
                        ctypes.get_last_error(), accepted, count)
    return accepted, count
