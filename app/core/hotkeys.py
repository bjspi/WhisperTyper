"""Hotkey string parsing, normalization and binding models — pure logic, no Qt, no pynput.

The token grammar ("<ctrl>+<f9>", "Cmd+Shift+F", …) lives here so it is unit-testable
and has exactly one implementation. The hotkey mixin keeps only the parts that talk to
real input events (pynput/Qt key objects) and the listener lifecycle.

Canonical token format: special keys are wrapped in angle brackets ("<ctrl>", "<f9>",
"<caps_lock>"); printable characters are bare lowercase ("c", "9"). A hotkey string joins
tokens with "+".
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set

from app.core.env import is_WINDOWS
from app.core.frameworks import MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN

#: Tokens that behave like modifiers when matching combos.
MODIFIER_TOKENS = frozenset({
    "<ctrl>", "<alt>", "<shift>", "<cmd>", "<win>", "<alt_gr>", "<caps_lock>",
})

#: Aliases accepted in user input, mapped to their canonical token core.
_ALIAS_MAP = {
    "control": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl",
    "alt_l": "alt", "alt_r": "alt",
    "shift_l": "shift", "shift_r": "shift",
    "cmd_l": "cmd", "cmd_r": "cmd",
    "super": "win", "super_l": "win", "super_r": "win", "windows": "win", "win": "win",
    "option": "alt", "option_r": "alt_gr",
    "command": "cmd", "meta": "cmd",
    "escape": "esc", "return": "enter", "capslock": "caps_lock",
    "pageup": "page_up", "pgup": "page_up",
    "pagedown": "page_down", "pgdown": "page_down",
}

#: Non-character keys that get the angle-bracket form.
_SPECIAL_TOKENS = frozenset({
    "ctrl", "alt", "shift", "cmd", "win", "alt_gr", "caps_lock",
    "backspace", "tab", "enter", "esc", "space", "delete", "insert",
    "home", "end", "page_up", "page_down", "left", "right", "up", "down",
})

#: Windows virtual-key code -> canonical token. Single source of truth; the reverse
#: mapping below is derived from it so the two can never drift apart.
VK_TO_TOKEN: Dict[int, str] = {
    0x08: "<backspace>", 0x09: "<tab>", 0x0D: "<enter>", 0x14: "<caps_lock>",
    0x1B: "<esc>", 0x20: "<space>", 0x21: "<page_up>", 0x22: "<page_down>",
    0x23: "<end>", 0x24: "<home>", 0x25: "<left>", 0x26: "<up>",
    0x27: "<right>", 0x28: "<down>", 0x2D: "<insert>", 0x2E: "<delete>",
    0x70: "<f1>", 0x71: "<f2>", 0x72: "<f3>", 0x73: "<f4>", 0x74: "<f5>",
    0x75: "<f6>", 0x76: "<f7>", 0x77: "<f8>", 0x78: "<f9>", 0x79: "<f10>",
    0x7A: "<f11>", 0x7B: "<f12>",
    0xBA: ";", 0xBB: "+", 0xBC: ",", 0xBD: "-", 0xBE: ".", 0xBF: "/",
    # Modifier VKs (generic + left/right variants) all collapse to the generic token.
    0x11: "<ctrl>", 0xA2: "<ctrl>", 0xA3: "<ctrl>",
    0x12: "<alt>", 0xA4: "<alt>",
    0x10: "<shift>", 0xA0: "<shift>", 0xA1: "<shift>",
    0x5B: "<win>", 0x5C: "<win>",
}

#: Canonical token -> Windows virtual-key code (non-modifier trigger keys only).
#: Derived from VK_TO_TOKEN, keeping the lowest VK for tokens with several codes.
TOKEN_TO_VK: Dict[str, int] = {}
for _vk in sorted(VK_TO_TOKEN, reverse=True):
    _token = VK_TO_TOKEN[_vk]
    if _token not in MODIFIER_TOKENS:
        TOKEN_TO_VK[_token] = _vk

#: Display names for pretty-printing tokens (settings header badge etc.).
_PRETTY_NAMES = {
    "ctrl": "Ctrl", "ctrl_l": "Ctrl", "ctrl_r": "Ctrl", "control": "Ctrl",
    "alt": "Alt", "alt_l": "Alt", "alt_r": "Alt", "alt_gr": "AltGr",
    "shift": "Shift", "shift_l": "Shift", "shift_r": "Shift",
    "cmd": "Cmd", "win": "Win", "caps_lock": "Caps Lock", "space": "Space",
    "enter": "Enter", "esc": "Esc", "tab": "Tab",
}

#: Stable modifier ordering for display strings.
_TOKEN_ORDER = {
    "<ctrl>": 0, "<shift>": 1, "<alt>": 2, "<alt_gr>": 3,
    "<cmd>": 4, "<win>": 5, "<caps_lock>": 6,
}


def normalize_hotkey_part(part: str) -> str:
    """Normalize one user-entered hotkey token ('Ctrl', 'F9', 'x') to its canonical form."""
    normalized = part.strip().lower()
    if not normalized:
        return ""
    if normalized.startswith("<") and normalized.endswith(">"):
        normalized = normalized[1:-1]
    normalized = normalized.replace(" ", "").replace("-", "_")
    normalized = _ALIAS_MAP.get(normalized, normalized)
    if re.fullmatch(r"f\d{1,2}", normalized) or normalized in _SPECIAL_TOKENS:
        return f"<{normalized}>"
    return normalized


def normalize_hotkey_string(hotkey_str: str) -> str:
    """Normalize a user-entered hotkey like 'F9' or 'Cmd+Shift+K' into canonical tokens."""
    normalized_parts: List[str] = []
    for raw_part in hotkey_str.split("+"):
        token = normalize_hotkey_part(raw_part)
        if token and token not in normalized_parts:
            normalized_parts.append(token)
    return "+".join(normalized_parts)


def is_modifier_token(token: str) -> bool:
    """Return whether a token behaves like a modifier for hotkey matching."""
    return token in MODIFIER_TOKENS


def is_non_modifier_special_token(token: str) -> bool:
    """Return whether a token is a non-modifier special token such as '<f9>'."""
    return token.startswith("<") and token.endswith(">") and not is_modifier_token(token)


def format_hotkey_tokens(tokens: Iterable[str]) -> str:
    """Format canonical hotkey tokens into a stable display string (modifiers first)."""
    display_tokens = set(tokens)
    if "<alt_gr>" in display_tokens:
        # AltGr already implies Ctrl+Alt; showing them separately would be misleading.
        display_tokens.discard("<ctrl>")
        display_tokens.discard("<alt>")
    sorted_tokens = sorted(display_tokens, key=lambda token: (_TOKEN_ORDER.get(token, 50), token))
    return "+".join(sorted_tokens)


def pretty_hotkey(hotkey: str) -> str:
    """Turn a stored hotkey like '<caps_lock>+<ctrl_l>' into 'Caps Lock  +  Ctrl'."""
    parts = []
    for raw in (hotkey or "").split("+"):
        token = raw.strip().strip("<>")
        if not token:
            continue
        parts.append(_PRETTY_NAMES.get(
            token,
            token.replace("_", " ").upper() if len(token) <= 3 else token.replace("_", " ").title(),
        ))
    return "  +  ".join(parts) or "—"


def vk_to_key_token(vk: Optional[int]) -> Optional[str]:
    """Translate a Windows virtual-key code into a canonical token (None if unmapped)."""
    if vk is None:
        return None
    token = VK_TO_TOKEN.get(vk)
    if token:
        return token
    if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
        return chr(vk).lower()
    return None


def vk_to_hotkey_tokens(vk: Optional[int]) -> Set[str]:
    """Translate a Windows virtual-key code to one or more canonical hotkey tokens.

    VK 0xA5 (right Alt) is AltGr on Windows keyboards and implies Ctrl+Alt.
    """
    if vk is None:
        return set()
    if vk == 0xA5:
        return {"<alt_gr>", "<alt>", "<ctrl>"}
    token = vk_to_key_token(vk)
    return {token} if token else set()


def token_to_windows_vk(token: str) -> Optional[int]:
    """Map a canonical trigger token to a Windows virtual-key code (None if unmappable)."""
    if token in TOKEN_TO_VK:
        return TOKEN_TO_VK[token]
    if len(token) == 1 and (token.isalpha() or token.isdigit()):
        return ord(token.upper())
    return None


def parse_hotkey_binding(hotkey_str: str, action: str) -> Optional[Dict[str, Any]]:
    """Parse a hotkey string into a normalized binding model.

    The binding dict carries everything both listener backends need: the canonical token
    set, the modifier subset, the non-modifier trigger tokens, and — on Windows — the
    RegisterHotKey modifier bitmask/VK plus whether the combo is OS-bindable at all.
    """
    if not hotkey_str:
        return None
    raw_parts = [part for part in hotkey_str.split("+") if part.strip()]
    if not raw_parts:
        return None

    tokens: List[str] = []
    for part in raw_parts:
        token = normalize_hotkey_part(part)
        if token and token not in tokens:
            tokens.append(token)

    modifiers = {token for token in tokens if is_modifier_token(token)}
    trigger_tokens = [token for token in tokens if token not in modifiers]

    windows_modifiers = 0
    if is_WINDOWS:
        if "<ctrl>" in modifiers or "<alt_gr>" in modifiers:
            windows_modifiers |= MOD_CONTROL
        if "<alt>" in modifiers or "<alt_gr>" in modifiers:
            windows_modifiers |= MOD_ALT
        if "<shift>" in modifiers:
            windows_modifiers |= MOD_SHIFT
        if "<win>" in modifiers or "<cmd>" in modifiers:
            windows_modifiers |= MOD_WIN

    windows_vk = token_to_windows_vk(trigger_tokens[0]) if is_WINDOWS and len(trigger_tokens) == 1 else None
    bindable_on_windows = bool(
        is_WINDOWS and len(trigger_tokens) == 1 and windows_vk is not None and "<caps_lock>" not in modifiers
    )

    return {
        "action": action,
        "display": format_hotkey_tokens(set(tokens)),
        "tokens": set(tokens),
        "modifiers": modifiers,
        "trigger_tokens": set(trigger_tokens),
        "windows_modifiers": windows_modifiers,
        "windows_vk": windows_vk,
        "windows_bindable": bindable_on_windows,
    }


def binding_matches_pressed(binding: Dict[str, Any], pressed_tokens: Set[str]) -> bool:
    """Return whether the currently pressed token set still satisfies a binding."""
    if not binding["modifiers"].issubset(pressed_tokens):
        return False
    trigger_tokens = binding["trigger_tokens"]
    if trigger_tokens:
        return any(token in pressed_tokens for token in trigger_tokens)
    return bool(binding["modifiers"])


def binding_matches_current_press(binding: Dict[str, Any], pressed_tokens: Set[str],
                                  key_tokens: Set[str]) -> bool:
    """Return whether the key press carried in ``key_tokens`` completes a binding."""
    if not binding["modifiers"].issubset(pressed_tokens):
        return False
    trigger_tokens = binding["trigger_tokens"]
    if trigger_tokens:
        return bool(key_tokens.intersection(trigger_tokens))
    return bool(key_tokens.intersection(binding["tokens"]))


def binding_release_tokens(binding: Dict[str, Any]) -> Set[str]:
    """Return the token set whose release should stop an active push-to-talk binding."""
    release_tokens = set(binding["tokens"])
    if "<alt_gr>" in release_tokens:
        release_tokens.update({"<ctrl>", "<alt>"})
    return release_tokens


def binding_needs_manual_suppression(binding: Dict[str, Any]) -> bool:
    """Return whether a Windows binding must use the low-level listener for event suppression.

    RegisterHotKey cannot suppress combos that involve Caps Lock or plain character keys
    without also breaking their normal function, so those go through the pynput hook.
    """
    return "<caps_lock>" in binding["tokens"] or any(len(token) == 1 for token in binding["trigger_tokens"])
