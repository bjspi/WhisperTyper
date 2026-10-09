"""Synthesized Ctrl/Cmd shortcuts (copy, paste, select all) for the focused application."""
from __future__ import annotations

import logging
import subprocess
from typing import Optional

import pyautogui
from pynput import keyboard

from app.core.env import is_MACOS

_controller: Optional[keyboard.Controller] = None


def simulate_shortcut(char: str, *, alt_lib: bool, fast: bool = False) -> bool:
    """Press Ctrl+``char`` (Cmd on macOS); returns whether the keys were dispatched.

    macOS first uses System Events, which is more reliable than synthesized key events.
    ``alt_lib`` selects pyautogui instead of pynput; ``fast`` skips pyautogui's pause.
    """
    if is_MACOS and _system_events_shortcut(char):
        return True
    if alt_lib:
        logging.debug("Using pyautogui for key simulation.")
        try:
            pyautogui.hotkey('command' if is_MACOS else 'ctrl', char, **({"_pause": False} if fast else {}))
            return True
        except Exception as e:
            logging.error(f"pyautogui key simulation failed: {e}")
            return False
    logging.debug("Using pynput for key simulation.")
    global _controller
    try:
        _controller = _controller or keyboard.Controller()
        with _controller.pressed(keyboard.Key.cmd if is_MACOS else keyboard.Key.ctrl):
            _controller.press(char)
            _controller.release(char)
        return True
    except Exception as e:
        logging.error(f"pynput key simulation failed: {e}")
        return False


def _system_events_shortcut(char: str) -> bool:
    """Send Cmd+``char`` through AppleScript System Events."""
    script = f'tell application "System Events" to keystroke "{char}" using command down'
    try:
        subprocess.run(['osascript', '-e', script], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.warning("macOS System Events key simulation failed for Cmd+%s: %s",
                        char, (e.stderr or e.stdout or str(e)).strip())
    except Exception as e:
        logging.warning(f"macOS System Events key simulation failed for Cmd+{char}: {e}")
    return False
