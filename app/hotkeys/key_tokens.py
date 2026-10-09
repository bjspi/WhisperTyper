"""Convert raw key events (pynput listener, Qt capture) into canonical hotkey tokens.

The token grammar itself (normalization, matching, VK tables) lives in :mod:`app.core.hotkeys`.
"""
from __future__ import annotations

from typing import Any, Dict, Set

from PyQt6.QtCore import Qt

from app.core import hotkeys
from app.core.env import is_MACOS, is_WINDOWS

_PYNPUT_SPECIAL_KEYS = {
    "ctrl": "<ctrl>", "ctrl_l": "<ctrl>", "ctrl_r": "<ctrl>",
    "alt": "<alt>", "alt_l": "<alt>", "alt_r": "<alt>", "alt_gr": "<alt_gr>",
    "shift": "<shift>", "shift_l": "<shift>", "shift_r": "<shift>",
    "cmd": "<cmd>", "cmd_l": "<cmd>", "cmd_r": "<cmd>",
    "caps_lock": "<caps_lock>", "esc": "<esc>", "space": "<space>",
    "enter": "<enter>", "tab": "<tab>", "backspace": "<backspace>",
    "delete": "<delete>", "insert": "<insert>", "home": "<home>", "end": "<end>",
    "page_up": "<page_up>", "page_down": "<page_down>",
    "left": "<left>", "right": "<right>", "up": "<up>", "down": "<down>",
    # macOS surfaces F7-F12 as media keys when the Fn toggle is set to media mode.
    "media_previous": "<f7>", "media_play_pause": "<f8>", "media_next": "<f9>",
    "media_volume_mute": "<f10>", "media_volume_down": "<f11>", "media_volume_up": "<f12>",
}

_QT_MODIFIER_KEYS = {Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta}
_QT_SPECIAL_KEYS: Dict[int, str] = {
    Qt.Key.Key_Escape: "<esc>", Qt.Key.Key_Tab: "<tab>", Qt.Key.Key_Backtab: "<tab>",
    Qt.Key.Key_Backspace: "<backspace>", Qt.Key.Key_Return: "<enter>", Qt.Key.Key_Enter: "<enter>",
    Qt.Key.Key_Insert: "<insert>", Qt.Key.Key_Delete: "<delete>", Qt.Key.Key_Home: "<home>",
    Qt.Key.Key_End: "<end>", Qt.Key.Key_Left: "<left>", Qt.Key.Key_Up: "<up>",
    Qt.Key.Key_Right: "<right>", Qt.Key.Key_Down: "<down>", Qt.Key.Key_PageUp: "<page_up>",
    Qt.Key.Key_PageDown: "<page_down>", Qt.Key.Key_CapsLock: "<caps_lock>", Qt.Key.Key_Space: "<space>",
    Qt.Key.Key_Plus: "<plus>",
}
# macOS reports F7-F12 as media keys when the Fn toggle is in media mode.
_QT_MACOS_FUNCTION_ALIASES: Dict[int, str] = {
    getattr(Qt.Key, name): token for name, token in (
        ("Key_MediaPrevious", "<f7>"), ("Key_MediaTogglePlayPause", "<f8>"), ("Key_MediaPlay", "<f8>"),
        ("Key_MediaNext", "<f9>"), ("Key_VolumeMute", "<f10>"), ("Key_VolumeDown", "<f11>"),
        ("Key_VolumeUp", "<f12>"),
    ) if hasattr(Qt.Key, name)
}
_QT_PUNCTUATION: Dict[int, str] = {
    Qt.Key.Key_Minus: "-", Qt.Key.Key_Equal: "=", Qt.Key.Key_BracketLeft: "[",
    Qt.Key.Key_BracketRight: "]", Qt.Key.Key_Backslash: "\\", Qt.Key.Key_Semicolon: ";",
    Qt.Key.Key_Apostrophe: "'", Qt.Key.Key_Comma: ",", Qt.Key.Key_Period: ".",
    Qt.Key.Key_Slash: "/", Qt.Key.Key_QuoteLeft: "`",
}


def pynput_key_tokens(key: Any) -> Set[str]:
    """Canonical tokens of a pynput key (name, character and virtual-key code)."""
    tokens: Set[str] = set()
    key_name = getattr(key, "name", None)
    if key_name in _PYNPUT_SPECIAL_KEYS:
        tokens.add(_PYNPUT_SPECIAL_KEYS[key_name])
    function_key_token = hotkeys.normalize_hotkey_part(key_name or "")
    if function_key_token.startswith("<f") and function_key_token.endswith(">"):
        tokens.add(function_key_token)
    # On Windows, AltGr is often surfaced as the right Alt key plus an implicit Ctrl.
    if is_WINDOWS and key_name == "alt_r":
        tokens.update({"<alt_gr>", "<alt>", "<ctrl>"})
    char = getattr(key, "char", None)
    if char:
        tokens.add(char.lower())
    # Only Windows reports virtual-key codes here; macOS hardware keycodes would turn e.g. 't' into <ctrl>.
    vk_token = hotkeys.vk_to_key_token(getattr(key, "vk", None)) if is_WINDOWS else None
    if vk_token:
        tokens.add(vk_token)
    if "<alt_gr>" in tokens:
        tokens.update({"<ctrl>", "<alt>"})
    return {hotkeys.normalize_hotkey_part(token) for token in tokens}


def qt_key_tokens(key: int, modifiers: Qt.KeyboardModifier, text: str) -> Set[str]:
    """Canonical tokens of a Qt key press (used for capture on macOS, where pynput needs permissions)."""
    tokens: Set[str] = set()
    ctrl, meta = Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.MetaModifier
    if is_MACOS:  # Qt reports the Command key as ControlModifier and the Control key as MetaModifier.
        ctrl, meta = meta, ctrl
    for flag, token in ((ctrl, "<ctrl>"), (Qt.KeyboardModifier.ShiftModifier, "<shift>"),
                        (Qt.KeyboardModifier.AltModifier, "<alt>"),
                        (meta, "<cmd>" if is_MACOS else "<win>")):
        if modifiers & flag:
            tokens.add(token)
    if key in _QT_MODIFIER_KEYS:
        return tokens
    if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F35:
        return tokens | {f"<f{int(key) - int(Qt.Key.Key_F1) + 1}>"}
    named = (_QT_MACOS_FUNCTION_ALIASES.get(key) if is_MACOS else None) or _QT_SPECIAL_KEYS.get(key)
    if named:
        return tokens | {named}
    text = text.lower()
    if len(text) == 1 and text.isprintable() and not text.isspace():
        return tokens | {text}
    if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
        return tokens | {chr(ord("a") + int(key) - int(Qt.Key.Key_A))}
    if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
        return tokens | {chr(ord("0") + int(key) - int(Qt.Key.Key_0))}
    punctuation = _QT_PUNCTUATION.get(key)
    return tokens | {punctuation} if punctuation else tokens


def injected_event_counts(tokens: Set[str]) -> bool:
    """Whether an injected event may trigger hotkeys: macOS remappers inject special keys such as F-keys."""
    # <plus> is a printable key that only needs brackets because '+' separates tokens.
    return is_MACOS and any(hotkeys.is_non_modifier_special_token(token) and token != "<plus>" for token in tokens)
